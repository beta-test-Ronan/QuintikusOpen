"""
RLM-Lastrimy v6.2 — KV Cache + Percepção Estendida + Modo Resposta
==================================================================
- Identidade do token = hash(tok) (do v6.1)
- KV cache de similaridade (pair_score) — 10-20x mais rápido
- Atenção em janela estendida (janela_ctx), separada do n-gram (n_max)
- Estado semântico da consulta (query_state)
- Semente de resposta escolhida semanticamente (_response_seed)
- Modo "responder" (não autocompleta) vs "continuar" (comportamento v6.1)
- Cache invalidado em absorver / carregar_estado
"""

import hashlib
import math
import random
import time
import re
import os
import pickle
from collections import Counter, defaultdict
import numpy as np

_SENT = "\uE000"


# ═══════════════════════════════════════════════════════════════════════
# GNS CLASSIFICADOR (AUTOENCODER) — inalterado
# ═══════════════════════════════════════════════════════════════════════
class Camada:
    ESCALA_INTERNA = 1.0
    ESCALA_ENTRE   = 10.0

    def __init__(self, camada_idx, n_in, n_out, saida_linear=False):
        self.camada_idx = camada_idx
        self.n_in = n_in
        self.n_out = n_out
        self.saida_linear = saida_linear

        self.lastros = np.array([
            int(hashlib.sha256(f"G{camada_idx+1}:N{i}".encode()).hexdigest()[:12], 16)
            for i in range(n_out)
        ], dtype=np.int64)

        self.posicoes = (
            (camada_idx + 1) * self.ESCALA_ENTRE
            + np.arange(n_out) * self.ESCALA_INTERNA
        )

        rngs = [np.random.default_rng(int(l) % (2**32)) for l in self.lastros]
        self.pesos = np.stack([r.standard_normal(n_in) * 0.5 for r in rngs])
        self.bias  = np.array([r.standard_normal() * 0.5 for r in rngs])

    def forward_batch(self, X):
        Z = X @ self.pesos.T + self.bias
        if self.saida_linear:
            Z = Z - Z.max(axis=1, keepdims=True)
            E = np.exp(Z)
            return E / (E.sum(axis=1, keepdims=True) + 1e-12)
        return np.tanh(Z)


class RedeLinhagem:
    def __init__(self, tamanhos, alpha=1.0, softmax_saida=True):
        self.tamanhos = tamanhos
        self.alpha = alpha
        self.softmax_saida = softmax_saida

        self.camadas = [
            Camada(camada_idx=c,
                   n_in=tamanhos[c], n_out=tamanhos[c + 1],
                   saida_linear=(c == len(tamanhos) - 2 and softmax_saida))
            for c in range(len(tamanhos) - 1)
        ]

        self.matrizes_dist = []
        for c in range(len(self.camadas) - 1):
            pos_a = self.camadas[c].posicoes
            pos_b = self.camadas[c + 1].posicoes
            d = np.abs(pos_a[:, None] - pos_b[None, :])
            M = 1.0 / (1.0 + d) ** alpha
            M /= (M.sum(axis=1, keepdims=True) + 1e-9)
            self.matrizes_dist.append(M)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        ativacoes = [X]
        for camada in self.camadas:
            X = camada.forward_batch(X)
            ativacoes.append(X)
        return ativacoes[-1], ativacoes

    def treinar(self, X, Y, epocas=400, lr=0.05, tol_erro=0.01, verbose=True):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        n_ex, n_cam = len(X), len(self.camadas)
        hist = []

        for ep in range(epocas):
            saida, ativacoes = self.forward(X)
            erros = saida - Y
            erro_total = float(np.mean(np.sum(erros**2, axis=1)))

            delta = erros / n_ex
            for c in range(n_cam - 1, -1, -1):
                camada = self.camadas[c]
                grad_W = delta.T @ ativacoes[c]
                grad_b = delta.sum(axis=0)

                camada.pesos -= lr * grad_W
                camada.bias  -= lr * grad_b
                np.clip(camada.pesos, -5.0, 5.0, out=camada.pesos)
                np.clip(camada.bias,  -5.0, 5.0, out=camada.bias)

                if c > 0:
                    delta = delta @ self.matrizes_dist[c - 1].T

            hist.append(erro_total)
            if verbose and (ep % 50 == 0 or ep == epocas - 1):
                print(f"      epoca {ep:3d} | erro={erro_total:.6f}")

            if erro_total <= tol_erro:
                if verbose:
                    print(f"      early stop ep {ep} | erro={erro_total:.6f}")
                break
        return hist


# ═══════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════
def _entropia_chars(tok):
    if not tok:
        return 0.0
    c = Counter(tok)
    t = len(tok)
    return -sum((n / t) * math.log2(n / t) for n in c.values())


def _rada_token(tok, n_angulos=12):
    acc = [0.0] * n_angulos
    for i, ch in enumerate(tok):
        d = (ord(ch) + i * 7) % n_angulos
        acc[d] += 1.0 / (1 + i)
    s = sum(acc) or 1.0
    return [a / s for a in acc]


def lf_forma(tok, visitas=0, is_sent=False):
    h = _entropia_chars(tok) / 5.0
    len_rel = min(len(tok) / 12.0, 1.0)
    freq = math.log1p(visitas) / 5.0
    flag = 1.0 if is_sent else 0.0
    rada = _rada_token(tok)
    return np.array([h, len_rel, freq, flag] + rada, dtype=float)


def laminy(fa, fb):
    n = len(fa)
    col = lambda a, b: -math.log2(1.0 - min(1.0, abs(a - b)) + 1e-9)
    return math.exp(-sum(col(fa[i], fb[i]) for i in range(n)) / n)


def _softmax(x, T=1.0):
    x = np.asarray(x, dtype=float) / max(T, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


# ═══════════════════════════════════════════════════════════════════════
# LASTRO E RLM
# ═══════════════════════════════════════════════════════════════════════
class Lastro:
    __slots__ = ("id", "token", "pos", "lf_forma", "dados_ids",
                 "transicoes", "transicoes_aum", "vizinhos", "sucessores",
                 "perfil", "pesos_head")

    def __init__(self, rid, token, pos, lf_f):
        self.id = rid
        self.token = token
        self.pos = pos
        self.lf_forma = lf_f
        self.dados_ids = []
        self.transicoes = Counter()
        self.transicoes_aum = None
        self.vizinhos = []
        self.sucessores = set()
        self.perfil = None
        self.pesos_head = None


class RLM:
    HEADS = ("pos", "forma", "ctx", "trans", "freq")

    def __init__(self, n_max=4, D=0.75, tau_pos=2.0, k_vizinhos=5,
                 limiar_forma=0.70, limiar_ctx=0.15, peso_vizinho=0.5,
                 anti_loop_janela=3, anti_loop_forca=0.3,
                 anti_loop_sintatico_forca=0.2, tau_perfil=1.0,
                 auto_ajuste_limiar=True, usar_rede_neural=True,
                 gns_max_vocab=800, gns_hidden=24, gns_epocas=400,
                 gns_lr=0.05):
        self.n_max = n_max
        self.D = D
        self.tau_pos = tau_pos
        self.k_vizinhos = k_vizinhos
        self.limiar_forma = limiar_forma
        self.limiar_ctx = limiar_ctx
        self.peso_vizinho = peso_vizinho
        self.anti_loop_janela = anti_loop_janela
        self.anti_loop_forca = anti_loop_forca
        self.anti_loop_sintatico_forca = anti_loop_sintatico_forca
        self.tau_perfil = tau_perfil
        self.auto_ajuste_limiar = auto_ajuste_limiar
        self.usar_rede_neural = usar_rede_neural
        self.gns_max_vocab = gns_max_vocab
        self.gns_hidden = gns_hidden
        self.gns_epocas = gns_epocas
        self.gns_lr = gns_lr

        self.BOS = self._hash((_SENT, "BOS", -1))
        self.EOS = self._hash((_SENT, "EOS", -2))
        self.lastros = {}
        self.transicoes = {}
        self._unigram = {}
        self._stats_globais = {}

        # ─── KV cache (novo) ───
        self._kv_init()

        self._criar_lastro(self.BOS, "⟨BOS⟩", -1)
        self._criar_lastro(self.EOS, "⟨EOS⟩", -2)

    # ───────────────────────────────────────────────────────────────────
    # KV CACHE
    # ───────────────────────────────────────────────────────────────────
    def _kv_init(self):
        """Invalida caches de similaridade. Chamar após absorver / carregar."""
        self._pair_cache = {}

    def _hash(self, x):
        return int(hashlib.sha256(str(x).encode()).hexdigest()[:16], 16)

    def _criar_lastro(self, rid, tok, pos, frase_id=None):
        if rid in self.lastros:
            if frase_id is not None:
                self.lastros[rid].dados_ids.append(frase_id)
            return self.lastros[rid]
        is_sent = tok in ("⟨BOS⟩", "⟨EOS⟩")
        lf = lf_forma(tok, visitas=0, is_sent=is_sent)
        self.lastros[rid] = Lastro(rid, tok, pos, lf)
        if frase_id is not None:
            self.lastros[rid].dados_ids.append(frase_id)
        return self.lastros[rid]

    def _acende(self, texto, frase_id=None, criar=True):
        texto_limpo = re.sub(r"[^\w\s]", " ", texto.lower())
        tokens = [t for t in texto_limpo.split() if _SENT not in t and t]
        ramos = []
        for i, tok in enumerate(tokens):
            rid = self._hash(tok)
            if criar:
                if rid in self.lastros:
                    self.lastros[rid].pos = i
                else:
                    self._criar_lastro(rid, tok, i, frase_id)
                ramos.append(rid)
            elif rid in self.lastros:
                ramos.append(rid)
        return tuple(ramos)

    # ───────────────────────────────────────────────────────────────────
    # KN e pré-cálculos (mesmos do v6.1)
    # ───────────────────────────────────────────────────────────────────
    def _prep_continuacao(self):
        cont = {}
        for ctx, counter in self.transicoes.items():
            for w in counter:
                cont.setdefault(w, set()).add(ctx)
        total = sum(len(s) for s in cont.values())
        self._unigram = ({w: len(s) / total for w, s in cont.items()}
                         if total else {})

    def _kn_rec(self, c):
        if not c:
            return dict(self._unigram)
        if c not in self.transicoes:
            return self._kn_rec(c[1:])
        counter = self.transicoes[c]
        total = sum(counter.values())
        lam = self.D * len(counter) / total
        fundo = self._kn_rec(c[1:])
        dist = {w: lam * p for w, p in fundo.items()}
        for w, cnt in counter.items():
            dist[w] = dist.get(w, 0) + max(cnt - self.D, 0) / total
        return dist

    def _calcular_stats_globais(self):
        sucs  = [len(la.sucessores) for la in self.lastros.values()]
        freqs = [len(la.dados_ids) for la in self.lastros.values()]
        self._stats_globais = {
            "suc_med": float(np.mean(sucs)) if sucs else 1.0,
            "freq_med": float(np.mean(freqs)) if freqs else 1.0,
        }

    def _derivar_perfil(self, la):
        st = self._stats_globais
        p_pos   = min(abs(la.pos - 4) / 8.0, 1.0)
        p_forma = float(la.lf_forma[0])
        p_ctx   = min(len(la.sucessores) / max(st["suc_med"] * 2, 1.0), 1.0)
        p_trans = 1.0 if len(la.transicoes) > 0 else 0.2
        p_freq  = 1.0 / (1.0 + math.log1p(len(la.dados_ids)))
        return np.array([p_pos, p_forma, p_ctx, p_trans, p_freq])

    def _precalcular_perfis_e_pesos(self):
        self._calcular_stats_globais()
        for la in self.lastros.values():
            la.perfil = self._derivar_perfil(la)
            la.pesos_head = _softmax(la.perfil, T=self.tau_perfil)

    def _pretreinar_perfis_por_rede(self, texto):
        contador = Counter()
        linhas_tokens = []
        for linha in texto.strip().splitlines():
            linha = linha.strip()
            if not linha:
                continue
            toks = [t for t in re.sub(r"[^\w\s]", " ", linha.lower()).split()
                    if _SENT not in t and t]
            if not toks:
                continue
            contador.update(toks)
            linhas_tokens.append(toks)

        vocab = [w for w, _ in contador.most_common(self.gns_max_vocab)]
        if len(vocab) < 20:
            return False
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)

        N_doc = len(linhas_tokens)
        df = Counter()
        for toks in linhas_tokens:
            for w in set(toks):
                if w in vocab_idx:
                    df[w] += 1
        idf = np.array([np.log((1 + N_doc) / (1 + df[w])) + 1.0 for w in vocab])

        X = np.zeros((N_doc, V), dtype=float)
        for i, toks in enumerate(linhas_tokens):
            tf = Counter(t for t in toks if t in vocab_idx)
            for w, c in tf.items():
                X[i, vocab_idx[w]] = c * idf[vocab_idx[w]]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        nz = norms[:, 0] > 0
        X[nz] = X[nz] / norms[nz]

        print(f"  [GNS] V={V} | docs={N_doc} | arquitetura={V}-{self.gns_hidden}-{self.gns_hidden}-{V}")
        rede = RedeLinhagem([V, self.gns_hidden, self.gns_hidden, V],
                            alpha=1.0, softmax_saida=False)
        hist = rede.treinar(X, X.copy(), epocas=self.gns_epocas,
                            lr=self.gns_lr, tol_erro=0.01, verbose=True)

        W0 = rede.camadas[0].pesos
        W0 = (W0 - W0.min()) / (W0.max() - W0.min() + 1e-9)

        n_heads = len(self.HEADS)
        n_aplicados = 0
        for la in self.lastros.values():
            if la.token not in vocab_idx:
                continue
            v = vocab_idx[la.token]
            emb = W0[:, v]
            partes = np.array_split(emb, n_heads)
            p = np.array([float(pt.mean()) for pt in partes])
            p = np.clip(p, 0, 1)
            la.perfil = p
            la.pesos_head = _softmax(p, T=self.tau_perfil)
            n_aplicados += 1

        print(f"  [GNS] erro final={hist[-1]:.6f} | perfis={n_aplicados}/{len(self.lastros)}")
        return True

    def _precalcular_sucessores(self):
        for la in self.lastros.values():
            la.sucessores = set(la.transicoes.keys())

    def _precalcular_vizinhos(self):
        por_sucessor = defaultdict(set)
        for la in self.lastros.values():
            for prox in la.sucessores:
                por_sucessor[prox].add(la.id)

        cand_total, aceitos_total = 0, 0
        for ra in self.lastros:
            la = self.lastros[ra]
            if not la.sucessores:
                la.vizinhos = []
                continue
            cands = Counter()
            for prox in la.sucessores:
                for rb in por_sucessor[prox]:
                    if rb != ra:
                        cands[rb] += 1
            cand_total += len(cands)

            vizinhos = []
            for rb, comuns in cands.items():
                lb = self.lastros[rb]
                sf = laminy(la.lf_forma, lb.lf_forma)
                if sf < self.limiar_forma:
                    continue
                uniao = len(la.sucessores | lb.sucessores)
                jacc = comuns / uniao if uniao > 0 else 0.0
                if jacc < self.limiar_ctx:
                    continue
                vizinhos.append((rb, sf, jacc, sf * jacc))
            vizinhos.sort(key=lambda x: -x[3])
            la.vizinhos = vizinhos[:self.k_vizinhos]
            aceitos_total += len(la.vizinhos)
        return cand_total, aceitos_total

    def _ajustar_limiar_se_preciso(self, aceitos_total, n_lastros):
        if not self.auto_ajuste_limiar:
            return False
        media = aceitos_total / max(n_lastros, 1)
        mudou = False
        if media < 1.5 and self.limiar_ctx > 0.08:
            self.limiar_ctx = max(0.08, self.limiar_ctx - 0.03)
            mudou = True
        elif media > 4.5 and self.limiar_ctx < 0.30:
            self.limiar_ctx = min(0.30, self.limiar_ctx + 0.03)
            mudou = True
        return mudou

    def _precalcular_transicoes_aumentadas(self):
        for la in self.lastros.values():
            t = Counter(la.transicoes)
            viz_ids = {v[0] for v in la.vizinhos}
            for viz_id, sf, jacc, score in la.vizinhos:
                vizinho = self.lastros[viz_id]
                for prox, cnt in vizinho.transicoes.items():
                    if prox in viz_ids or prox == la.id:
                        continue
                    t[prox] += self.peso_vizinho * score * cnt
            la.transicoes_aum = t

    # ───────────────────────────────────────────────────────────────────
    # ABSORÇÃO
    # ───────────────────────────────────────────────────────────────────
    def absorver(self, texto):
        linhas = 0
        for i, linha in enumerate(texto.strip().splitlines()):
            linha = linha.strip()
            if not linha:
                continue
            vidro = (self.BOS,) + self._acende(linha, frase_id=i) + (self.EOS,)
            for j in range(1, len(vidro)):
                alvo = vidro[j]
                for k in range(1, min(self.n_max, j) + 1):
                    ctx = vidro[j - k:j]
                    self.transicoes.setdefault(ctx, Counter())[alvo] += 1
            for a, b in zip(vidro, vidro[1:]):
                self.lastros[a].transicoes[b] += 1
            linhas += 1

        for la in self.lastros.values():
            is_sent = la.token in ("⟨BOS⟩", "⟨EOS⟩")
            la.lf_forma = lf_forma(la.token, visitas=len(la.dados_ids),
                                   is_sent=is_sent)

        self._prep_continuacao()
        self._precalcular_sucessores()
        self._precalcular_perfis_e_pesos()

        if self.usar_rede_neural:
            self._pretreinar_perfis_por_rede(texto)

        for _ in range(3):
            cand, aceitos = self._precalcular_vizinhos()
            if not self._ajustar_limiar_se_preciso(aceitos, len(self.lastros)):
                break

        self._precalcular_transicoes_aumentadas()

        # ─── invalida KV cache após mudanças de pesos/transições ───
        self._kv_init()

        print(f"[ABSORVEU] {linhas} linhas | {len(self.transicoes)} ctx | {len(self.lastros)} lastros")

    def absorver_arquivo(self, caminho, encoding="utf-8"):
        with open(caminho, "r", encoding=encoding) as f:
            texto = f.read()
        self.absorver(texto)
        return self

    # ───────────────────────────────────────────────────────────────────
    # HEADS + KV cache de similaridade
    # ───────────────────────────────────────────────────────────────────
    def _head_pos(self, li, lj):
        return math.exp(-abs(li.pos - lj.pos) / self.tau_pos)

    def _head_forma(self, li, lj):
        return laminy(li.lf_forma, lj.lf_forma)

    def _head_ctx(self, li, lj):
        si, sj = li.sucessores, lj.sucessores
        if not si or not sj:
            return 0.0
        return len(si & sj) / len(si | sj)

    def _head_trans(self, li, lj):
        return 1.0 if (lj.id in li.transicoes or li.id in lj.transicoes) else 0.0

    def _head_freq(self, li, lj):
        fi = math.log1p(len(li.dados_ids))
        fj = math.log1p(len(lj.dados_ids))
        return 1.0 / (1.0 + (fi + fj) / 2)

    def _head(self, nome, li, lj):
        return {
            "pos":   self._head_pos,
            "forma": self._head_forma,
            "ctx":   self._head_ctx,
            "trans": self._head_trans,
            "freq":  self._head_freq,
        }[nome](li, lj)

    def _pair_score(self, li, lj):
        """Score bruto (não normalizado) entre dois lastros.
        Cacheado por chave canônica (heads são todas simétricas)."""
        a, b = li.id, lj.id
        if a > b:
            a, b = b, a
        key = (a, b)
        hit = self._pair_cache.get(key)
        if hit is not None:
            return hit

        soma = 0.0
        for h_idx, h_nome in enumerate(self.HEADS):
            w = math.sqrt(max(li.pesos_head[h_idx] * lj.pesos_head[h_idx], 1e-9))
            soma += w * self._head(h_nome, li, lj)
        self._pair_cache[key] = soma
        return soma

    def matriz_lastrimy(self, contexto):
        n = len(contexto)
        A = np.zeros((n, n))
        for i in range(n):
            li = self.lastros[contexto[i]]
            for j in range(n):
                lj = self.lastros[contexto[j]]
                A[i, j] = self._pair_score(li, lj)
        return A / (A.sum(axis=1, keepdims=True) + 1e-9)

    def _lastrimy_row(self, contexto):
        """Só a última linha (query = último token). Usa KV cache."""
        li = self.lastros[contexto[-1]]
        row = np.array([self._pair_score(li, self.lastros[rj])
                        for rj in contexto])
        return row / (row.sum() + 1e-9)

    # ───────────────────────────────────────────────────────────────────
    # Anti-loop sintático, contexto específico
    # ───────────────────────────────────────────────────────────────────
    def _anti_loop_sintatico(self, probs, contexto):
        recentes = contexto[-2:]
        for c in list(probs):
            s_c = self.lastros[c].sucessores
            if not s_c:
                continue
            for rid_r in recentes:
                s_r = self.lastros[rid_r].sucessores
                if not s_r:
                    continue
                jacc = len(s_c & s_r) / len(s_c | s_r)
                if jacc > 0.4:
                    probs[c] *= self.anti_loop_sintatico_forca
                    break
        s = sum(probs.values())
        return {k: v / s for k, v in probs.items()} if s > 1e-12 else probs

    def _contexto_especifico_sucessores(self, contexto):
        for k in range(len(contexto), 0, -1):
            sub = tuple(contexto[-k:])
            if sub in self.transicoes:
                return set(self.transicoes[sub].keys())
        return set()

    # ───────────────────────────────────────────────────────────────────
    # ESTADO SEMÂNTICO + SEMENTE DE RESPOSTA (novo)
    # ───────────────────────────────────────────────────────────────────
    def _query_state(self, consulta):
        """Vetor semântico da consulta inteira.
        Concatenção de: média dos pesos_head + média do lf_forma."""
        if not consulta:
            return None
        p = np.zeros(len(self.HEADS))
        lf = np.zeros_like(self.lastros[consulta[0]].lf_forma)
        for rid in consulta:
            la = self.lastros[rid]
            p += la.pesos_head
            lf += la.lf_forma
        p /= len(consulta)
        lf /= len(consulta)
        return np.concatenate([p, lf])

    def _response_seed(self, query_state, consulta, top_k=8, temperatura=0.5):
        """Escolhe o PRIMEIRO token da RESPOSTA.
        Prefere tokens com estado semântico próximo da consulta
        que NÃO aparecem na consulta e que têm continuação."""
        if query_state is None:
            return None
        qids = set(consulta)
        nq = np.linalg.norm(query_state)
        if nq < 1e-9:
            return None

        scores = []
        for rid, la in self.lastros.items():
            if rid in qids:
                continue
            if la.token.startswith("⟨"):
                continue
            if not la.sucessores:
                continue
            v = np.concatenate([la.pesos_head, la.lf_forma])
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            sim = float(np.dot(query_state, v) / (nq * nv))
            scores.append((rid, sim))

        if not scores:
            return None

        scores.sort(key=lambda x: -x[1])
        top = scores[:top_k]

        if temperatura <= 0:
            return top[0][0]

        sims = np.array([s for _, s in top])
        z = sims / max(temperatura, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        idx = int(np.random.choice(len(top), p=p))
        return top[idx][0]

    # ───────────────────────────────────────────────────────────────────
    # PERCEBER (v6.2 — modo responder ou continuar)
    # ───────────────────────────────────────────────────────────────────
    def perceber(self, texto, cache_memoria=None, passos=15, temperatura=0.7,
                 alpha=0.5, min_passos=3, mostrar=True,
                 modo="responder", janela_ctx=12):

        consulta = tuple(r for r in self._acende(texto, criar=False)
                         if r in self.lastros)
        if not consulta:
            if mostrar:
                print(f"\n[PERCEBEU] '{texto}' — nenhum lastro conhecido")
            return ""

        if mostrar:
            print(f"\n[PERCEBEU] '{texto}' | modo={modo} | "
                  f"T={temperatura} | α={alpha} | janela={janela_ctx}")

        # ─── FASE 1: PERCEBER (estado semântico da consulta) ───
        query_state = self._query_state(consulta)

        # ─── FASE 2: SEMEAR a resposta ───
        if modo == "responder":
            seed = self._response_seed(query_state, consulta,
                                       temperatura=max(temperatura, 0.3))
            if seed is not None:
                historico = [seed]
                saida     = [self.lastros[seed].token]
            else:
                historico = list(consulta)
                saida     = []
        else:  # "continuar" — comportamento v6.1
            historico = list(consulta)
            saida     = []

        # ─── FASE 3: GERAR com janela estendida + KV cache ───
        for passo in range(passos):
            if historico[-1] == self.EOS:
                break

            ctx_full = tuple(historico[-janela_ctx:])
            ctx_kn   = tuple(historico[-self.n_max:])

            # KN (usa contexto curto)
            kn = self._kn_rec(ctx_kn)
            if not kn:
                break

            # Atenção (usa contexto estendido) — só última linha
            pesos = self._lastrimy_row(ctx_full)

            # Votos agregados
            votos = Counter()
            for j, rid in enumerate(ctx_full):
                w = pesos[j]
                for prox, cnt in self.lastros[rid].transicoes_aum.items():
                    votos[prox] += w * cnt
            tot = sum(votos.values()) or 1.0
            lastrimy_dist = {k: v / tot for k, v in votos.items()}

            # Candidatos
            kn_esp = self._contexto_especifico_sucessores(ctx_kn)
            cands = kn_esp | set(self.lastros[ctx_full[-1]].transicoes_aum.keys())
            if self.EOS in kn_esp:
                cands.add(self.EOS)
            cands = [c for c in cands if c in self.lastros]

            # Anti-EOS precoce
            if passo < min_passos:
                filt = [c for c in cands if c != self.EOS]
                if filt:
                    cands = filt
                elif self.EOS in cands:
                    cands = [self.EOS]

            if not cands:
                break

            # Mistura KN + Lastrimy
            probs = {c: (1 - alpha) * kn.get(c, 1e-9)
                     + alpha * lastrimy_dist.get(c, 1e-9)
                     for c in cands}
            s = sum(probs.values()) or 1.0
            probs = {k: v / s for k, v in probs.items()}

            # Cache de curto prazo
            if cache_memoria:
                for c in cands:
                    if self.lastros[c].token in cache_memoria:
                        probs[c] *= 1.5
                s = sum(probs.values()) or 1.0
                probs = {k: v / s for k, v in probs.items()}

            # Anti-loop de token
            recentes = Counter(self.lastros[r].token
                               for r in ctx_kn[-self.anti_loop_janela:])
            for c in list(probs):
                n = recentes.get(self.lastros[c].token, 0)
                if n > 0:
                    probs[c] *= self.anti_loop_forca ** n

            s = sum(probs.values())
            if s > 1e-12:
                probs = {k: v / s for k, v in probs.items()}
            probs = self._anti_loop_sintatico(probs, ctx_kn)

            # Temperatura
            if temperatura > 0:
                z = {k: v ** (1.0 / temperatura) for k, v in probs.items()}
                s = sum(z.values()) or 1.0
                probs = {k: v / s for k, v in z.items()}

            if temperatura <= 0:
                w = max(probs, key=probs.get)
            else:
                itens = list(probs.items())
                ramos = [c for c, _ in itens]
                ps    = np.array([p for _, p in itens])
                w = ramos[int(np.random.choice(len(ramos), p=ps / ps.sum()))]

            if w == self.EOS:
                break
            saida.append(self.lastros[w].token)
            historico.append(w)

        resultado = " ".join(saida)
        if mostrar:
            print(f"  -> {resultado}")
        return resultado

    # ───────────────────────────────────────────────────────────────────
    # PERSISTÊNCIA
    # ───────────────────────────────────────────────────────────────────
    def salvar_estado(self, caminho="rlm_state.pkl"):
        with open(caminho, "wb") as f:
            pickle.dump({
                "lastros": self.lastros,
                "transicoes": self.transicoes,
                "unigram": self._unigram,
                "limiar_ctx": self.limiar_ctx,
            }, f)
        print(f"💾 Estado do RLM salvo em '{caminho}'")

    @classmethod
    def carregar_estado(cls, caminho="rlm_state.pkl", **kwargs):
        if not os.path.exists(caminho):
            return None
        with open(caminho, "rb") as f:
            d = pickle.load(f)
        g = cls(**kwargs)
        g.lastros = d["lastros"]
        g.transicoes = d["transicoes"]
        g._unigram = d["unigram"]
        g.limiar_ctx = d["limiar_ctx"]
        g._kv_init()  # <── invalida cache após carregar
        print(f"⚡ Estado carregado de '{caminho}' (instantâneo).")
        return g


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ARQUIVO_CORPUS = "corpus.txt"
    ARQUIVO_ESTADO = "rlm_state.pkl"

    print("=" * 70)
    print("RLM-Lastrimy v6.2 — KV Cache + Percepção Estendida + Resposta")
    print("=" * 70)

    kwargs_rlm = dict(
        n_max=4, D=0.75, tau_pos=2.0, k_vizinhos=5,
        limiar_forma=0.70, limiar_ctx=0.15, peso_vizinho=0.5,
        anti_loop_sintatico_forca=0.2, tau_perfil=1.0,
        auto_ajuste_limiar=True, usar_rede_neural=True,
        gns_max_vocab=800, gns_hidden=24, gns_epocas=400, gns_lr=0.05,
    )

    rlm = RLM.carregar_estado(ARQUIVO_ESTADO, **kwargs_rlm)

    if rlm is None:
        if not os.path.exists(ARQUIVO_CORPUS):
            print(f"[ERRO] '{ARQUIVO_CORPUS}' não encontrado.")
            raise SystemExit(1)

        print("\n⚙️ Iniciando treinamento pesado (400 épocas)...")
        rlm = RLM(**kwargs_rlm)
        rlm.absorver_arquivo(ARQUIVO_CORPUS)
        rlm.salvar_estado(ARQUIVO_ESTADO)

    print("\nDigite uma frase (ou 'sair' para encerrar).")
    print("Comandos: sair | T=0.3 | a=0.7 | limpar | m=responder | m=continuar | j=12")
    print("-" * 70)

    temperatura = 0.5
    alpha = 0.5
    modo = "responder"     # "responder" (novo) ou "continuar" (antigo)
    janela_ctx = 12
    cache_memoria = set()

    while True:
        try:
            entrada = input("\nvocê > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not entrada:
            continue
        low = entrada.lower()

        if low in ("sair", "exit", "quit"):
            break
        if low == "limpar":
            cache_memoria.clear()
            print("[Memória neural limpa]")
            continue
        if low.startswith("t="):
            temperatura = float(entrada.split("=", 1)[1])
            print(f"[T={temperatura}]")
            continue
        if low.startswith("a="):
            alpha = float(entrada.split("=", 1)[1])
            print(f"[α={alpha}]")
            continue
        if low.startswith("m="):
            modo = entrada.split("=", 1)[1].strip()
            print(f"[modo={modo}]")
            continue
        if low.startswith("j="):
            janela_ctx = int(entrada.split("=", 1)[1])
            print(f"[janela_ctx={janela_ctx}]")
            continue

        print("-" * 70)

        resposta = rlm.perceber(
            entrada,
            cache_memoria=cache_memoria,
            passos=12,
            temperatura=temperatura,
            alpha=alpha,
            modo=modo,
            janela_ctx=janela_ctx,
        )
        resposta = resposta.replace("⟨EOS⟩", "").replace("⟨BOS⟩", "").strip()

        if not resposta:
            print("\nRLM  > [Silêncio / Nenhum caminho]")
        else:
            print(f"\nRLM  > {resposta}")

            stopwords = {
                "o", "a", "de", "que", "e", "do", "da", "em", "um", "uma",
                "para", "com", "não", "sim", "você", "eu", "tá", "é", "se",
                "por", "os", "as", "no", "na",
            }
            palavras_novas = set(
                re.findall(r'\b\w+\b', (entrada + " " + resposta).lower())
            ) - stopwords
            cache_memoria = palavras_novas

        print("-" * 70)
