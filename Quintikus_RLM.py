"""
RLM-Lastrimy v5.7 — GNS neural + Planejamento por Viterbi
==========================================================
FUSÃO:
  - GNS classificador faz pré-treino neural dos tokens (TF-IDF → autoencoder)
  - RLM consome W0 como `perfil` do lastro (Lastrimy + KN)
  - NOVO: modo "plan" faz Viterbi sobre o grafo — escolhe o caminho
    inteiro antes de emitir, em vez de token a token.

Modos:
  auto → perceber() — geração autoregressiva (token a token)
  plan → planejar() — Viterbi sobre o grafo inteiro
"""

import hashlib
import math
import random
import time
import re
from collections import Counter, defaultdict
import numpy as np

_SENT = "\uE000"


# ═══════════════════════════════════════════════════════════════════════
# GNS CLASSIFICADOR — Camada + RedeLinhagem
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

    def treinar(self, X, Y, epocas=500, lr=0.3, tol_erro=0.01, verbose=True):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        n_ex, n_cam = len(X), len(self.camadas)
        hist = []

        for ep in range(epocas):
            saida, ativacoes = self.forward(X)
            erros = saida - Y
            erro_total = float(np.einsum('ij,ij->', erros, erros)) / n_ex

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
            if verbose and (ep % 25 == 0 or ep == epocas - 1):
                print(f"      epoca {ep:3d} | erro={erro_total:.6f}")

            if erro_total <= tol_erro:
                if verbose:
                    print(f"      early stop ep {ep} | erro={erro_total:.6f}")
                break
        return hist


# ═══════════════════════════════════════════════════════════════════════
# Funções auxiliares
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
# Lastro
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

    def __init__(self, n_max=4, D=0.75,
                 beta_lf=0.6, tau_pos=2.0,
                 k_vizinhos=5,
                 limiar_forma=0.70, limiar_ctx=0.15,
                 peso_vizinho=0.5,
                 anti_loop_janela=3, anti_loop_forca=0.3,
                 anti_loop_lf_limiar=0.93, anti_loop_lf_forca=0.5,
                 anti_loop_sintatico_forca=0.2,
                 tau_perfil=1.0,
                 auto_ajuste_limiar=True,
                 usar_rede_neural=True,
                 gns_max_vocab=800,
                 gns_hidden=24,
                 gns_epocas=120,
                 gns_lr=0.2):
        self.n_max = n_max
        self.D = D
        self.beta_lf = beta_lf
        self.tau_pos = tau_pos
        self.k_vizinhos = k_vizinhos
        self.limiar_forma = limiar_forma
        self.limiar_ctx = limiar_ctx
        self.peso_vizinho = peso_vizinho
        self.anti_loop_janela = anti_loop_janela
        self.anti_loop_forca = anti_loop_forca
        self.anti_loop_lf_limiar = anti_loop_lf_limiar
        self.anti_loop_lf_forca = anti_loop_lf_forca
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
        self._auditoria = []
        self._criar_lastro(self.BOS, "⟨BOS⟩", -1)
        self._criar_lastro(self.EOS, "⟨EOS⟩", -2)

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
            rid = self._hash((i, tok))
            if criar:
                self._criar_lastro(rid, tok, i, frase_id)
                ramos.append(rid)
            elif rid in self.lastros:
                ramos.append(rid)
        return tuple(ramos)

    def _prep_continuacao(self):
        cont = {}
        for ctx, counter in self.transicoes.items():
            for w in counter:
                cont.setdefault(w, set()).add(ctx)
        total = sum(len(s) for s in cont.values())
        self._unigram = {w: len(s) / total for w, s in cont.items()} if total else {}

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
        sucs = [len(la.sucessores) for la in self.lastros.values()]
        freqs = [len(la.dados_ids) for la in self.lastros.values()]
        ents = [float(la.lf_forma[0]) for la in self.lastros.values()]
        self._stats_globais = {
            "suc_med": float(np.mean(sucs)) if sucs else 1.0,
            "freq_med": float(np.mean(freqs)) if freqs else 1.0,
            "ent_med": float(np.mean(ents)) if ents else 0.5,
        }

    def _derivar_perfil(self, la):
        st = self._stats_globais
        p_pos = min(abs(la.pos - 4) / 8.0, 1.0)
        p_forma = float(la.lf_forma[0])
        p_ctx = min(len(la.sucessores) / max(st["suc_med"] * 2, 1.0), 1.0)
        p_trans = 1.0 if len(la.transicoes) > 0 else 0.2
        p_freq = 1.0 / (1.0 + math.log1p(len(la.dados_ids)))
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
            print(f"  [GNS] vocab insuficiente ({len(vocab)}) — perfil manual")
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

        print(f"  [GNS] V={V} | docs={N_doc} | "
              f"arquitetura={V}-{self.gns_hidden}-{self.gns_hidden}-{V}")
        t0 = time.time()
        rede = RedeLinhagem([V, self.gns_hidden, self.gns_hidden, V],
                            alpha=1.0, softmax_saida=False)
        hist = rede.treinar(X, X.copy(), epocas=self.gns_epocas,
                            lr=self.gns_lr, tol_erro=0.01, verbose=True)
        dt = time.time() - t0

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

        print(f"  [GNS] erro final={hist[-1]:.6f} | tempo={dt:.2f}s | "
              f"perfis aplicados={n_aplicados}/{len(self.lastros)}")
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
                    if prox in viz_ids:
                        continue
                    if prox == la.id:
                        continue
                    t[prox] += self.peso_vizinho * score * cnt
            la.transicoes_aum = t

    def _auditar_vizinhos_como_sucessores(self):
        problemas = []
        for la in self.lastros.values():
            viz_ids = {v[0] for v in la.vizinhos}
            suc_obs = set(la.transicoes.keys()) & viz_ids
            suc_aum = set(la.transicoes_aum.keys()) & viz_ids
            if suc_aum - suc_obs:
                for rid in (suc_aum - suc_obs):
                    problemas.append((la.token, la.pos, self.lastros[rid].token))
        return problemas

    def absorver(self, texto):
        t0 = time.time()
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

        t1 = time.time()
        self._prep_continuacao()
        self._precalcular_sucessores()
        self._precalcular_perfis_e_pesos()

        if self.usar_rede_neural:
            self._pretreinar_perfis_por_rede(texto)

        for tentativa in range(3):
            cand, aceitos = self._precalcular_vizinhos()
            media = aceitos / max(len(self.lastros), 1)
            if not self._ajustar_limiar_se_preciso(aceitos, len(self.lastros)):
                break

        self._precalcular_transicoes_aumentadas()
        t2 = time.time()

        self._auditoria = self._auditar_vizinhos_como_sucessores()

        print(f"[ABSORVEU] {linhas} linhas | {len(self.transicoes)} ctx | "
              f"{len(self.lastros)} lastros")
        print(f"  [pré-cálculo] KN={t1-t0:.3f}s | pesada={t2-t1:.3f}s | "
              f"cand={cand} aceitos={aceitos} | "
              f"vizinhos/lastro={media:.2f} | limiar_ctx={self.limiar_ctx:.3f}")
        if self._auditoria:
            print(f"  [AUDITORIA] ⚠ {len(self._auditoria)} vizinho(s) "
                  f"viraram sucessor — REGRA 1 falhou!")
        else:
            print(f"  [AUDITORIA] ✓ nenhum vizinho virou sucessor (REGRA 1 ok)")

    def absorver_arquivo(self, caminho, encoding="utf-8"):
        with open(caminho, "r", encoding=encoding) as f:
            texto = f.read()
        self.absorver(texto)
        return self

    # ==================== HEADS ====================
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

    def matriz_lastrimy(self, contexto):
        n = len(contexto)
        A = np.zeros((n, n))
        for i in range(n):
            li = self.lastros[contexto[i]]
            for j in range(n):
                lj = self.lastros[contexto[j]]
                soma = 0.0
                for h_idx, h_nome in enumerate(self.HEADS):
                    w = math.sqrt(max(li.pesos_head[h_idx] *
                                      lj.pesos_head[h_idx], 1e-9))
                    soma += w * self._head(h_nome, li, lj)
                A[i, j] = soma
        A = A / (A.sum(axis=1, keepdims=True) + 1e-9)
        return A

    def _anti_loop_sintatico(self, probs, contexto):
        recentes = contexto[-2:]
        forca = self.anti_loop_sintatico_forca
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
                    probs[c] *= forca
                    break
        s = sum(probs.values())
        return {k: v / s for k, v in probs.items()} if s > 1e-12 else probs

    def _contexto_especifico_sucessores(self, contexto):
        for k in range(len(contexto), 0, -1):
            sub = tuple(contexto[-k:])
            if sub in self.transicoes:
                return set(self.transicoes[sub].keys())
        return set()

    def prever_probs(self, contexto, temperatura=0.7, alpha=0.4,
                     min_passos=0, passo=0):
        kn = self._kn_rec(tuple(contexto))
        if not kn:
            return {}, None

        A = self.matriz_lastrimy(contexto)
        pesos = A[-1]

        votos = Counter()
        for j, rid in enumerate(contexto):
            w = pesos[j]
            for prox, cnt in self.lastros[rid].transicoes_aum.items():
                votos[prox] += w * cnt
        tot = sum(votos.values()) or 1.0
        lastrimy_dist = {k: v / tot for k, v in votos.items()}

        ultimo_rid = contexto[-1]
        kn_esp = self._contexto_especifico_sucessores(contexto)
        cands_aum = set(self.lastros[ultimo_rid].transicoes_aum.keys())

        cands = kn_esp | cands_aum
        if self.EOS in kn_esp:
            cands.add(self.EOS)
        cands = [c for c in cands if c in self.lastros]

        if passo < min_passos:
            cands = [c for c in cands if c != self.EOS]
        if not cands:
            return {}, A

        probs = {}
        for c in cands:
            probs[c] = (1 - alpha) * kn.get(c, 1e-9) + alpha * lastrimy_dist.get(c, 1e-9)
        s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}

        recentes = Counter(self.lastros[r].token
                           for r in contexto[-self.anti_loop_janela:])
        for c in list(probs):
            n = recentes.get(self.lastros[c].token, 0)
            if n > 0:
                probs[c] *= self.anti_loop_forca ** n
        s = sum(probs.values())
        if s < 1e-12:
            probs = {c: (1 - alpha) * kn.get(c, 1e-9) +
                        alpha * lastrimy_dist.get(c, 1e-9) for c in cands}
            s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}

        probs = self._anti_loop_sintatico(probs, contexto)

        if temperatura > 0:
            z = {k: v ** (1.0 / temperatura) for k, v in probs.items()}
            s = sum(z.values()) or 1.0
            probs = {k: v / s for k, v in z.items()}
        return probs, A

    def perceber(self, texto, passos=15, temperatura=0.7, alpha=0.4,
                 min_passos=3, mostrar_matriz=False, mostrar=True):
        consulta_bruta = self._acende(texto, criar=False)
        consulta = tuple(r for r in consulta_bruta if r in self.lastros)
        descartados = len(consulta_bruta) - len(consulta)

        print(f"\n{'='*70}")
        print(f"[PERCEBEU] '{texto}'" +
              (f" (descartados {descartados})" if descartados else ""))
        print(f"  T={temperatura} | α={alpha} | τ_perfil={self.tau_perfil}")
        if not consulta:
            print("  -> nenhum lastro conhecido")
            return texto

        historico = list(consulta)
        saida = texto.split()

        for passo in range(passos):
            if historico[-1] == self.EOS:
                break
            contexto = tuple(historico[-self.n_max:])
            probs, A = self.prever_probs(contexto, temperatura, alpha,
                                          min_passos, passo)
            if not probs:
                break
            if mostrar:
                top = sorted(probs.items(), key=lambda x: -x[1])[:5]
                amostra = "  ".join(
                    f"{self.lastros[c].token}@{self.lastros[c].pos}:{p:.2f}"
                    for c, p in top
                )
                print(f"  ctx≈{len(contexto)}  [{amostra}]")
            if mostrar_matriz and A is not None and len(contexto) >= 2:
                self._print_matriz(A, contexto)

            if temperatura <= 0:
                w = max(probs, key=probs.get)
            else:
                itens = list(probs.items())
                ramos = [c for c, _ in itens]
                ps = np.array([p for _, p in itens])
                ps = ps / ps.sum()
                w = ramos[int(np.random.choice(len(ramos), p=ps))]

            saida.append(self.lastros[w].token)
            historico.append(w)

        resultado = " ".join(saida)
        print(f"  -> {resultado}")
        return resultado

    def _print_matriz(self, A, contexto):
        toks = [f"{self.lastros[r].token}@{self.lastros[r].pos}" for r in contexto]
        print(f"  matriz Lastrimy (multi-head ponderada por perfil):")
        print("         " + "  ".join(f"{t:>10}" for t in toks))
        for i, tok_i in enumerate(toks):
            linha = "  ".join(f"{A[i,j]:>10.3f}" for j in range(len(toks)))
            print(f"  {tok_i:>8} {linha}")

    # ═══════════════════════════════════════════════════════════════
    # ═══ NOVO: PLANEJAMENTO POR VITERBI ═══
    # ═══════════════════════════════════════════════════════════════
    def _formar_grupo(self, lastro_raiz, profundidade=2, max_grupo=60):
        """BFS pelos vizinhos_lf a partir do lastro raiz."""
        grupo = {lastro_raiz}
        fronteira = [lastro_raiz]
        for _ in range(profundidade):
            nova = []
            for rid in fronteira:
                if rid not in self.lastros:
                    continue
                for viz_id, _, _, _ in self.lastros[rid].vizinhos:
                    if viz_id not in grupo:
                        grupo.add(viz_id)
                        nova.append(viz_id)
                        if len(grupo) >= max_grupo:
                            return grupo
            fronteira = nova
            if not fronteira:
                break
        return grupo

    def planejar(self, texto, passos=8,
                 alpha_kn=0.5, alpha_last=0.3, alpha_grupo=0.2,
                 penal_comprimento=0.05, anti_loop_janela=3,
                 min_passos=3, max_grupo=60, profundidade_grupo=2,
                 usar_grupo=True, mostrar=True):
        """
        Viterbi sobre o grafo de lastros. Escolhe o caminho INTEIRO de uma vez.
        Score = α_kn·log P_KN + α_last·Lastrimy + α_grupo·1[∈grupo]
                - penal_comprimento - anti_loop
        """
        consulta = tuple(r for r in self._acende(texto, criar=False)
                         if r in self.lastros)
        if not consulta:
            print("  -> nenhum lastro conhecido")
            return texto

        ultimo = consulta[-1]

        grupo = self._formar_grupo(ultimo, profundidade_grupo, max_grupo) \
                if usar_grupo else set()

        # DP de Viterbi: {lastro: (score, caminho)}
        dp = {ultimo: (0.0, [ultimo])}

        for t in range(passos):
            novo_dp = {}
            for atual, (score_atual, caminho_atual) in dp.items():
                if atual == self.EOS:
                    continue
                trans = self.lastros[atual].transicoes_aum
                if not trans:
                    continue
                for prox, cnt in trans.items():
                    if prox not in self.lastros:
                        continue
                    s_kn = math.log(cnt + 1e-9)
                    s_last = self._head_pos(self.lastros[atual],
                                             self.lastros[prox])
                    s_grupo = 1.0 if prox in grupo else 0.0
                    s_pen = -penal_comprimento
                    s_loop = 0.0
                    if prox in caminho_atual[-anti_loop_janela:]:
                        s_loop = -2.0

                    delta = (alpha_kn * s_kn + alpha_last * s_last +
                             alpha_grupo * s_grupo + s_pen + s_loop)
                    novo_score = score_atual + delta
                    novo_caminho = caminho_atual + [prox]

                    if prox not in novo_dp or novo_dp[prox][0] < novo_score:
                        novo_dp[prox] = (novo_score, novo_caminho)

            if t + 1 >= min_passos and self.EOS not in novo_dp:
                # adiciona EOS como candidato de parada (a partir do melhor atual)
                if novo_dp:
                    melhor_atual = max(novo_dp.items(),
                                        key=lambda kv: kv[1][0])
                    rid_melhor, (sc, cam) = melhor_atual
                    novo_dp[self.EOS] = (sc, cam + [self.EOS])

            dp = novo_dp
            if not dp:
                break

        if not dp:
            print("  -> plano vazio")
            return texto

        melhor_score, melhor_caminho = max(dp.values(), key=lambda x: x[0])

        saida = texto.split()
        for rid in melhor_caminho:
            if rid != self.EOS:
                saida.append(self.lastros[rid].token)
        resultado = " ".join(saida)

        if mostrar:
            toks = [self.lastros[r].token for r in melhor_caminho
                    if r != self.EOS]
            print(f"  [PLANO] score={melhor_score:.3f} | "
                  f"grupo={len(grupo)} | caminho={toks}")

        return resultado

    def stats_pesos(self):
        print("\n[PESOS POR HEAD — distribuição]")
        for h_idx, h_nome in enumerate(self.HEADS):
            vals = [la.pesos_head[h_idx] for la in self.lastros.values()]
            print(f"  H_{h_nome:>5}  mean={np.mean(vals):.3f}  "
                  f"std={np.std(vals):.3f}  "
                  f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    def stats(self):
        n = len(self.lastros)
        viz = [len(la.vizinhos) for la in self.lastros.values()]
        obs = [len(la.transicoes) for la in self.lastros.values()]
        aum = [len(la.transicoes_aum) for la in self.lastros.values()]
        print("\n[STATS]")
        print(f"  lastros            : {n}")
        print(f"  vizinhos/lastro    : média={np.mean(viz):.1f} max={max(viz)}")
        print(f"  transições obs/las : média={np.mean(obs):.1f} max={max(obs)}")
        print(f"  transições aum/las : média={np.mean(aum):.1f} max={max(aum)}")
        print(f"  ganho vizinhança   : +{sum(aum) - sum(obs)} arestas")


# ============================================================
# MAIN — loop de conversa com modo auto/plan
# ============================================================
if __name__ == "__main__":
    import os

    caminho = "corpus.txt"
    if not os.path.exists(caminho):
        print(f"[ERRO] '{caminho}' não encontrado.")
        raise SystemExit(1)

    print("=" * 70)
    print("RLM-Lastrimy v5.7 — Conversa + GNS neural + Planejamento")
    print("=" * 70)

    rlm = RLM(n_max=4, D=0.75, tau_pos=2.0, k_vizinhos=5,
              limiar_forma=0.70, limiar_ctx=0.15, peso_vizinho=0.5,
              anti_loop_sintatico_forca=0.2, tau_perfil=1.0,
              auto_ajuste_limiar=True,
              usar_rede_neural=True,
              gns_max_vocab=800,
              gns_hidden=25,
              gns_epocas=350,
              gns_lr=0.2)
    rlm.absorver_arquivo(caminho)
    rlm.stats()
    rlm.stats_pesos()

    print("\nDigite uma frase (ou 'sair' para encerrar).")
    print("Comandos: sair | stats | T=0.3 | a=0.7 | modo=auto | modo=plan")
    print("-" * 70)

    temperatura = 0.5
    alpha = 0.5
    modo = "auto"

    while True:
        try:
            entrada = input("\nvocê > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[encerrado]")
            break

        if not entrada:
            continue

        low = entrada.lower()

        if low in ("sair", "exit", "quit"):
            print("[encerrado]")
            break

        if low == "stats":
            rlm.stats()
            rlm.stats_pesos()
            continue

        if low.startswith("t="):
            try:
                temperatura = float(entrada.split("=", 1)[1])
                print(f"[T={temperatura}]")
            except ValueError:
                print("[formato: T=0.5]")
            continue

        if low.startswith("a="):
            try:
                alpha = float(entrada.split("=", 1)[1])
                print(f"[α={alpha}]")
            except ValueError:
                print("[formato: a=0.5]")
            continue

        if low.startswith("modo="):
            m = entrada.split("=", 1)[1].strip().lower()
            if m in ("auto", "plan"):
                modo = m
                print(f"[modo={m}]")
            else:
                print("[modo=auto ou modo=plan]")
            continue

        print("-" * 70)

        if modo == "plan":
            saida = rlm.planejar(entrada, passos=12,
                                 alpha_kn=0.5, alpha_last=0.3,
                                 alpha_grupo=0.2)
        else:
            saida = rlm.perceber(entrada, passos=12,
                                 temperatura=temperatura, alpha=alpha,
                                 mostrar=True)

        toks_in = entrada.lower().split()
        toks_out = saida.split()
        if toks_out[:len(toks_in)] == toks_in:
            resposta = " ".join(toks_out[len(toks_in):])
        else:
            resposta = saida
        print(f"\nRLM  > {resposta}")
        print("-" * 70)
