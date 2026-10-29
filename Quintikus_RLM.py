"""
RLM-Lastrimy v6.8 — Amostragem + Confiança + Multi-turno
                    + Motor Recursivo + PromptHandle
                    + Attention Residuals (AttnRes) na síntese recursiva
=========================================================================
v6.8 adiciona (inspirado em arxiv.org/pdf/2603.15031):
- AttentionResidual: agregação softmax sobre "profundidade" (respostas recursivas)
- Pseudo-query aprendida = embedding da pergunta original
- Substitui Jaccard por atenção content-aware na síntese do MotorLLMRecursivo
- Mantém fallback Jaccard quando AttnRes está desativado ou indisponível
"""

import hashlib
import math
import random
import re
import os
import pickle
from collections import Counter, defaultdict
import numpy as np

_SENT = "\uE000"


# ═══════════════════════════════════════════════════════════════════════
# GNS (inalterado)
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
        self.posicoes = ((camada_idx + 1) * self.ESCALA_ENTRE
                         + np.arange(n_out) * self.ESCALA_INTERNA)
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
            Camada(camada_idx=c, n_in=tamanhos[c], n_out=tamanhos[c + 1],
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
        ativ = [X]
        for camada in self.camadas:
            X = camada.forward_batch(X)
            ativ.append(X)
        return ativ[-1], ativ

    def treinar(self, X, Y, epocas=400, lr=0.05, tol_erro=0.01, verbose=True):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        n_ex, n_cam = len(X), len(self.camadas)
        hist = []
        for ep in range(epocas):
            saida, ativ = self.forward(X)
            erros = saida - Y
            erro = float(np.mean(np.sum(erros**2, axis=1)))
            delta = erros / n_ex
            for c in range(n_cam - 1, -1, -1):
                cam = self.camadas[c]
                grad_W = delta.T @ ativ[c]
                grad_b = delta.sum(axis=0)
                cam.pesos -= lr * grad_W
                cam.bias  -= lr * grad_b
                np.clip(cam.pesos, -5.0, 5.0, out=cam.pesos)
                np.clip(cam.bias,  -5.0, 5.0, out=cam.bias)
                if c > 0:
                    delta = delta @ self.matrizes_dist[c - 1].T
            hist.append(erro)
            if verbose and (ep % 50 == 0 or ep == epocas - 1):
                print(f"      epoca {ep:3d} | erro={erro:.6f}")
            if erro <= tol_erro:
                if verbose:
                    print(f"      early stop ep {ep} | erro={erro:.6f}")
                break
        return hist


# ═══════════════════════════════════════════════════════════════════════
# AUXILIARES
# ═══════════════════════════════════════════════════════════════════════
def _entropia_chars(tok):
    if not tok:
        return 0.0
    c = Counter(tok); t = len(tok)
    return -sum((n / t) * math.log2(n / t) for n in c.values())


def _rada_token(tok, n=12):
    acc = [0.0] * n
    for i, ch in enumerate(tok):
        d = (ord(ch) + i * 7) % n
        acc[d] += 1.0 / (1 + i)
    s = sum(acc) or 1.0
    return [a / s for a in acc]


def lf_forma(tok, visitas=0, is_sent=False):
    h = _entropia_chars(tok) / 5.0
    len_rel = min(len(tok) / 12.0, 1.0)
    freq = math.log1p(visitas) / 5.0
    flag = 1.0 if is_sent else 0.0
    return np.array([h, len_rel, freq, flag] + _rada_token(tok), dtype=float)


def laminy(fa, fb):
    n = len(fa)
    col = lambda a, b: -math.log2(1.0 - min(1.0, abs(a - b)) + 1e-9)
    return math.exp(-sum(col(fa[i], fb[i]) for i in range(n)) / n)


def _softmax(x, T=1.0):
    x = np.asarray(x, dtype=float) / max(T, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _tokenize(texto):
    limpo = re.sub(r"[^\w\s]", " ", texto.lower())
    return [t for t in limpo.split() if _SENT not in t and t]


# ═══════════════════════════════════════════════════════════════════════
# ATTENTION RESIDUAL (AttnRes) — agregação softmax sobre profundidade
# ═══════════════════════════════════════════════════════════════════════
class AttentionResidual:
    """
    Implementação do AttnRes do paper arxiv:2603.15031, adaptada para
    respostas textuais da recursão.

    Dado um conjunto de "valores" (respostas) e uma "pseudo-query"
    (pergunta original), calcula pesos softmax e retorna a agregação.

    No paper, a agregação é linear sobre tensores. Aqui, como os valores
    são strings, a "agregação" é a seleção da resposta com maior peso
    (hard attention), que é o análogo discreto da soma ponderada.
    """

    def __init__(self, rlm):
        self.rlm = rlm
        # pseudo-query aprendida: usamos o embedding da pergunta como proxy.
        # Em um modelo treinável, w_l seria um vetor aprendido por camada.
        # Aqui, a "camada" é a síntese final, então w = embedding(pergunta).

    def _embed(self, texto):
        """Embedding de um texto via perfil/pesos_head dos lastros."""
        partes = []
        for t in _tokenize(texto):
            rid = self.rlm._hash(t)
            la = self.rlm.lastros.get(rid)
            if la is not None:
                if la.pesos_head is not None:
                    partes.append(la.pesos_head)   # 5-dim, softmax do perfil
                elif la.perfil is not None:
                    partes.append(la.perfil)
        if not partes:
            return np.zeros(5, dtype=float)
        return np.mean(partes, axis=0)

    def forward(self, pergunta, valores):
        """
        valores: lista de strings (respostas das sub-consultas).
        Retorna: a string selecionada pela atenção softmax.
        """
        if not valores:
            return None
        if len(valores) == 1:
            return valores[0]

        # V: [N, d] — valores empilhados
        V = np.stack([self._embed(v) for v in valores])
        # q: [d] — pseudo-query (embedding da pergunta)
        q = self._embed(pergunta)

        # Normalização L2 (como no paper, as chaves são normalizadas)
        V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        q_norm = q / (np.linalg.norm(q) + 1e-9)

        # logits = q · k_i  (produto escalar)
        logits = V_norm @ q_norm

        # softmax sobre a profundidade
        alpha = np.exp(logits - logits.max())
        alpha /= alpha.sum()

        # hard attention: seleciona o valor de maior peso
        idx = int(np.argmax(alpha))
        return valores[idx]


# ═══════════════════════════════════════════════════════════════════════
# LASTRO E RLM
# ═══════════════════════════════════════════════════════════════════════
class Lastro:
    __slots__ = ("id", "token", "pos", "lf_forma", "dados_ids",
                 "transicoes", "transicoes_aum", "vizinhos", "sucessores",
                 "perfil", "pesos_head")
    def __init__(self, rid, tok, pos, lf_f):
        self.id = rid; self.token = tok; self.pos = pos; self.lf_forma = lf_f
        self.dados_ids = []; self.transicoes = Counter()
        self.transicoes_aum = None; self.vizinhos = []
        self.sucessores = set(); self.perfil = None; self.pesos_head = None


class RLM:
    HEADS = ("pos", "forma", "ctx", "trans", "freq")

    def __init__(self, n_max=4, D=0.75, tau_pos=2.0, k_vizinhos=5,
                 limiar_forma=0.70, limiar_ctx=0.15, peso_vizinho=0.5,
                 anti_loop_janela=3, anti_loop_forca=0.3,
                 anti_loop_sintatico_forca=0.2, tau_perfil=1.0,
                 auto_ajuste_limiar=True, usar_rede_neural=True,
                 gns_max_vocab=800, gns_hidden=24, gns_epocas=400,
                 gns_lr=0.05, rag_top_k=5, rag_penalidade_reuso=0.5,
                 rag_threshold_alta=0.60, rag_threshold_media=0.35,
                 rag_echo_limiar=0.6, rag_min_tokens_linha=2,
                 rag_bonus_curta=1.20, rag_bonus_media=1.10,
                 rag_penalidade_longa=0.85):
        self.n_max = n_max; self.D = D; self.tau_pos = tau_pos
        self.k_vizinhos = k_vizinhos
        self.limiar_forma = limiar_forma; self.limiar_ctx = limiar_ctx
        self.peso_vizinho = peso_vizinho
        self.anti_loop_janela = anti_loop_janela
        self.anti_loop_forca = anti_loop_forca
        self.anti_loop_sintatico_forca = anti_loop_sintatico_forca
        self.tau_perfil = tau_perfil
        self.auto_ajuste_limiar = auto_ajuste_limiar
        self.usar_rede_neural = usar_rede_neural
        self.gns_max_vocab = gns_max_vocab; self.gns_hidden = gns_hidden
        self.gns_epocas = gns_epocas; self.gns_lr = gns_lr

        self.rag_top_k = rag_top_k
        self.rag_penalidade_reuso = rag_penalidade_reuso
        self.rag_threshold_alta  = rag_threshold_alta
        self.rag_threshold_media = rag_threshold_media
        self.rag_echo_limiar = rag_echo_limiar
        self.rag_min_tokens_linha = rag_min_tokens_linha
        self.rag_bonus_curta = rag_bonus_curta
        self.rag_bonus_media = rag_bonus_media
        self.rag_penalidade_longa = rag_penalidade_longa

        self.BOS = self._hash((_SENT, "BOS", -1))
        self.EOS = self._hash((_SENT, "EOS", -2))
        self.lastros = {}; self.transicoes = {}
        self._unigram = {}; self._stats_globais = {}
        self._mkn_discounts = (0.5, 1.0, 1.5)
        self._rag_lines = []; self._rag_idf = {}
        self._last_rag_info = {}
        self._kv_init()
        self._criar_lastro(self.BOS, "⟨BOS⟩", -1)
        self._criar_lastro(self.EOS, "⟨EOS⟩", -2)

    def _kv_init(self):
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
        tokens = _tokenize(texto)
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

    def _compute_mkn_discounts(self):
        N1 = N2 = N3 = N4 = 0
        for counter in self.transicoes.values():
            for _, c in counter.items():
                if c == 1: N1 += 1
                elif c == 2: N2 += 1
                elif c == 3: N3 += 1
                elif c == 4: N4 += 1
        if N1 == 0 or N2 == 0 or N3 == 0:
            return (0.5, 1.0, 1.5)
        Y = N1 / (N1 + 2.0 * N2)
        D1 = 1.0 - 2.0 * Y * N2 / N1
        D2 = 2.0 - 3.0 * Y * N3 / N2
        D3 = 3.0 - 4.0 * Y * N4 / N3 if N3 > 0 and N4 > 0 else 2.5
        return (max(0.05, min(0.95, D1)),
                max(0.5,  min(1.95, D2)),
                max(1.0,  min(2.95, D3)))

    def _prep_continuacao(self):
        cont = {}
        for ctx, counter in self.transicoes.items():
            for w in counter:
                cont.setdefault(w, set()).add(ctx)
        total = sum(len(s) for s in cont.values())
        self._unigram = ({w: len(s) / total for w, s in cont.items()}
                         if total else {})
        self._mkn_discounts = self._compute_mkn_discounts()

    def _kn_rec(self, c):
        if not c: return dict(self._unigram)
        if c not in self.transicoes: return self._kn_rec(c[1:])
        counter = self.transicoes[c]
        total = sum(counter.values())
        D1, D2, D3 = self._mkn_discounts
        N1c = N2c = N3c = 0
        for _, cnt in counter.items():
            if cnt == 1: N1c += 1
            elif cnt == 2: N2c += 1
            else: N3c += 1
        gamma = (D1 * N1c + D2 * N2c + D3 * N3c) / total
        fundo = self._kn_rec(c[1:])
        dist = {w: gamma * p for w, p in fundo.items()}
        for w, cnt in counter.items():
            D = D1 if cnt == 1 else (D2 if cnt == 2 else D3)
            dist[w] = dist.get(w, 0.0) + max(cnt - D, 0.0) / total
        return dist

    def _build_rag_index(self, texto):
        linhas = []
        for linha in texto.strip().splitlines():
            linha = linha.strip()
            if not linha: continue
            toks = _tokenize(linha)
            if len(toks) < self.rag_min_tokens_linha: continue
            linhas.append((linha, toks))
        if not linhas:
            self._rag_lines = []; self._rag_idf = {}
            return
        N = len(linhas)
        df = Counter()
        for _, toks in linhas:
            for w in set(toks):
                df[w] += 1
        idf = {w: math.log((1.0 + N) / (1.0 + df[w])) + 1.0 for w in df}
        rag = []
        for linha, toks in linhas:
            tf = Counter(toks)
            weights = {w: (1.0 + math.log(c)) * idf[w] for w, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
            weights = {w: v / norm for w, v in weights.items()}
            rag.append((linha, weights))
        self._rag_lines = rag
        self._rag_idf = idf
        print(f"  [RAG] índice: {len(rag)} frases | vocab={len(idf)}")

    def _rag_retrieve(self, texto, top_k=5, linhas_recentes=None):
        if not self._rag_lines: return []
        toks = _tokenize(texto)
        if not toks: return []
        tf = Counter(toks)
        q_weights = {}
        for w, c in tf.items():
            if w in self._rag_idf:
                q_weights[w] = (1.0 + math.log(c)) * self._rag_idf[w]
        q_norm = math.sqrt(sum(v * v for v in q_weights.values()))
        if q_norm < 1e-12: return []
        q_weights = {w: v / q_norm for w, v in q_weights.items()}
        scores = []
        for i, (linha, weights) in enumerate(self._rag_lines):
            s = sum(q_weights.get(w, 0.0) * v for w, v in weights.items())
            if s <= 0: continue
            if linhas_recentes and i in linhas_recentes:
                s *= self.rag_penalidade_reuso
            n = len(linha.split())
            if n <= 8:  s *= self.rag_bonus_curta
            elif n <= 15: s *= self.rag_bonus_media
            elif n >= 30: s *= self.rag_penalidade_longa
            scores.append((s, i, linha))
        scores.sort(key=lambda x: -x[0])
        return [(linha, s, i) for s, i, linha in scores[:top_k]]

    def _is_question(self, linha):
        l = linha.strip()
        return l.endswith("?") or l.endswith("?!")

    def _is_echo(self, linha, texto):
        l_toks = set(_tokenize(linha))
        t_toks = set(_tokenize(texto))
        if not l_toks or not t_toks: return False
        return len(l_toks & t_toks) / len(l_toks | t_toks) > self.rag_echo_limiar

    def _confidence_band(self, score):
        if score >= self.rag_threshold_alta:  return "alta"
        if score >= self.rag_threshold_media: return "media"
        return "baixa"

    def _sample_from_topk(self, retrieved, temperatura):
        if not retrieved: return None
        if len(retrieved) == 1 or temperatura <= 0:
            return retrieved[0]
        scores = np.array([s for _, s, _ in retrieved])
        z = scores / max(temperatura, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        idx = int(np.random.choice(len(retrieved), p=p))
        return retrieved[idx]

    def _fallback_honesto(self, retrieved):
        if not retrieved:
            return "Não tenho nada no que aprendi que se aproxime disso."
        top = retrieved[:3]
        pistas = " / ".join(f"'{l.rstrip('.').rstrip('?')}'" for l, _, _ in top)
        return (f"Não tenho certeza do que responder a isso. "
                f"O mais próximo que aprendi foi: {pistas}")

    def _assemble_query_context(self, texto, ultimo_par):
        if not ultimo_par:
            return texto
        n = len(_tokenize(texto))
        if n <= 3:
            pergunta_ant, resposta_ant = ultimo_par
            return f"{pergunta_ant} {resposta_ant} {texto}"
        return texto

    def _calcular_stats_globais(self):
        sucs  = [len(la.sucessores) for la in self.lastros.values()]
        freqs = [len(la.dados_ids) for la in self.lastros.values()]
        self._stats_globais = {
            "suc_med": float(np.mean(sucs)) if sucs else 1.0,
            "freq_med": float(np.mean(freqs)) if freqs else 1.0,
        }

    def _derivar_perfil(self, la):
        st = self._stats_globais
        return np.array([
            min(abs(la.pos - 4) / 8.0, 1.0),
            float(la.lf_forma[0]),
            min(len(la.sucessores) / max(st["suc_med"] * 2, 1.0), 1.0),
            1.0 if len(la.transicoes) > 0 else 0.2,
            1.0 / (1.0 + math.log1p(len(la.dados_ids))),
        ])

    def _precalcular_perfis_e_pesos(self):
        self._calcular_stats_globais()
        for la in self.lastros.values():
            la.perfil = self._derivar_perfil(la)
            la.pesos_head = _softmax(la.perfil, T=self.tau_perfil)

    def _pretreinar_perfis_por_rede(self, texto):
        contador = Counter(); linhas_tokens = []
        for linha in texto.strip().splitlines():
            toks = _tokenize(linha)
            if not toks: continue
            contador.update(toks); linhas_tokens.append(toks)
        vocab = [w for w, _ in contador.most_common(self.gns_max_vocab)]
        if len(vocab) < 20: return False
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        N_doc = len(linhas_tokens)
        df = Counter()
        for toks in linhas_tokens:
            for w in set(toks):
                if w in vocab_idx: df[w] += 1
        idf = np.array([np.log((1 + N_doc) / (1 + df[w])) + 1.0 for w in vocab])
        X = np.zeros((N_doc, V), dtype=float)
        for i, toks in enumerate(linhas_tokens):
            tf = Counter(t for t in toks if t in vocab_idx)
            for w, c in tf.items(): X[i, vocab_idx[w]] = c * idf[vocab_idx[w]]
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
        n_heads = len(self.HEADS); n_aplicados = 0
        for la in self.lastros.values():
            if la.token not in vocab_idx: continue
            v = vocab_idx[la.token]
            emb = W0[:, v]
            partes = np.array_split(emb, n_heads)
            p = np.clip(np.array([float(pt.mean()) for pt in partes]), 0, 1)
            la.perfil = p
            la.pesos_head = _softmax(p, T=self.tau_perfil)
            n_aplicados += 1
        print(f"  [GNS] erro final={hist[-1]:.6f} | perfis={n_aplicados}/{len(self.lastros)}")
        return True

    def _precalcular_sucessores(self):
        for la in self.lastros.values():
            la.sucessores = set(la.transicoes.keys())

    def _precalcular_vizinhos(self):
        por_suc = defaultdict(set)
        for la in self.lastros.values():
            for prox in la.sucessores: por_suc[prox].add(la.id)
        cand_t, aceitos_t = 0, 0
        for ra in self.lastros:
            la = self.lastros[ra]
            if not la.sucessores:
                la.vizinhos = []; continue
            cands = Counter()
            for prox in la.sucessores:
                for rb in por_suc[prox]:
                    if rb != ra: cands[rb] += 1
            cand_t += len(cands)
            viz = []
            for rb, comuns in cands.items():
                lb = self.lastros[rb]
                sf = laminy(la.lf_forma, lb.lf_forma)
                if sf < self.limiar_forma: continue
                uniao = len(la.sucessores | lb.sucessores)
                jacc = comuns / uniao if uniao > 0 else 0.0
                if jacc < self.limiar_ctx: continue
                viz.append((rb, sf, jacc, sf * jacc))
            viz.sort(key=lambda x: -x[3])
            la.vizinhos = viz[:self.k_vizinhos]
            aceitos_t += len(la.vizinhos)
        return cand_t, aceitos_t

    def _ajustar_limiar_se_preciso(self, aceitos, n):
        if not self.auto_ajuste_limiar: return False
        media = aceitos / max(n, 1); mudou = False
        if media < 1.5 and self.limiar_ctx > 0.08:
            self.limiar_ctx = max(0.08, self.limiar_ctx - 0.03); mudou = True
        elif media > 4.5 and self.limiar_ctx < 0.30:
            self.limiar_ctx = min(0.30, self.limiar_ctx + 0.03); mudou = True
        return mudou

    def _precalcular_transicoes_aumentadas(self):
        for la in self.lastros.values():
            t = Counter(la.transicoes)
            viz_ids = {v[0] for v in la.vizinhos}
            for viz_id, sf, jacc, score in la.vizinhos:
                viz = self.lastros[viz_id]
                for prox, cnt in viz.transicoes.items():
                    if prox in viz_ids or prox == la.id: continue
                    t[prox] += self.peso_vizinho * score * cnt
            la.transicoes_aum = t

    def absorver(self, texto):
        linhas = 0
        for i, linha in enumerate(texto.strip().splitlines()):
            linha = linha.strip()
            if not linha: continue
            vidro = (self.BOS,) + self._acende(linha, frase_id=i) + (self.EOS,)
            for j in range(1, len(vidro)):
                alvo = vidro[j]
                for k in range(1, min(self.n_max, j) + 1):
                    self.transicoes.setdefault(vidro[j - k:j], Counter())[alvo] += 1
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
            _, aceitos = self._precalcular_vizinhos()
            if not self._ajustar_limiar_se_preciso(aceitos, len(self.lastros)):
                break
        self._precalcular_transicoes_aumentadas()
        self._build_rag_index(texto)
        self._kv_init()
        print(f"[ABSORVEU] {linhas} linhas | {len(self.transicoes)} ctx | "
              f"{len(self.lastros)} lastros | MKN D=({self._mkn_discounts[0]:.2f},"
              f"{self._mkn_discounts[1]:.2f},{self._mkn_discounts[2]:.2f})")

    def absorver_arquivo(self, caminho, encoding="utf-8"):
        with open(caminho, "r", encoding=encoding) as f:
            self.absorver(f.read())
        return self

    def _head_pos(self, li, lj):
        return math.exp(-abs(li.pos - lj.pos) / self.tau_pos)
    def _head_forma(self, li, lj):
        return laminy(li.lf_forma, lj.lf_forma)
    def _head_ctx(self, li, lj):
        si, sj = li.sucessores, lj.sucessores
        if not si or not sj: return 0.0
        return len(si & sj) / len(si | sj)
    def _head_trans(self, li, lj):
        return 1.0 if (lj.id in li.transicoes or li.id in lj.transicoes) else 0.0
    def _head_freq(self, li, lj):
        return 1.0 / (1.0 + (math.log1p(len(li.dados_ids))
                             + math.log1p(len(lj.dados_ids))) / 2)
    def _head(self, nome, li, lj):
        return {"pos": self._head_pos, "forma": self._head_forma,
                "ctx": self._head_ctx, "trans": self._head_trans,
                "freq": self._head_freq}[nome](li, lj)

    def _pair_score(self, li, lj):
        a, b = (li.id, lj.id) if li.id <= lj.id else (lj.id, li.id)
        key = (a, b)
        hit = self._pair_cache.get(key)
        if hit is not None: return hit
        soma = 0.0
        for h_idx, h_nome in enumerate(self.HEADS):
            w = math.sqrt(max(li.pesos_head[h_idx] * lj.pesos_head[h_idx], 1e-9))
            soma += w * self._head(h_nome, li, lj)
        self._pair_cache[key] = soma
        return soma

    def _lastrimy_row(self, contexto):
        li = self.lastros[contexto[-1]]
        row = np.array([self._pair_score(li, self.lastros[rj]) for rj in contexto])
        return row / (row.sum() + 1e-9)

    def _anti_loop_sintatico(self, probs, contexto):
        recentes = contexto[-2:]
        for c in list(probs):
            s_c = self.lastros[c].sucessores
            if not s_c: continue
            for rid_r in recentes:
                s_r = self.lastros[rid_r].sucessores
                if not s_r: continue
                if len(s_c & s_r) / len(s_c | s_r) > 0.4:
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

    def _query_state(self, consulta):
        if not consulta: return None
        p = np.zeros(len(self.HEADS))
        lf = np.zeros_like(self.lastros[consulta[0]].lf_forma)
        for rid in consulta:
            la = self.lastros[rid]
            p += la.pesos_head; lf += la.lf_forma
        p /= len(consulta); lf /= len(consulta)
        return np.concatenate([p, lf])

    def _response_seed(self, q_state, consulta, temperatura=0.5, seeds_usados=None):
        if q_state is None: return None
        qids = set(consulta); nq = np.linalg.norm(q_state)
        if nq < 1e-9: return None
        scores = []
        for rid, la in self.lastros.items():
            if rid in qids or la.token.startswith("⟨") or not la.sucessores:
                continue
            if seeds_usados and rid in seeds_usados: continue
            v = np.concatenate([la.pesos_head, la.lf_forma])
            nv = np.linalg.norm(v)
            if nv < 1e-9: continue
            scores.append((rid, float(np.dot(q_state, v) / (nq * nv))))
        if not scores:
            return (self._response_seed(q_state, consulta, temperatura, None)
                    if seeds_usados else None)
        scores.sort(key=lambda x: -x[1])
        top = scores[:8]
        if temperatura <= 0: return top[0][0]
        sims = np.array([s for _, s in top])
        z = sims / max(temperatura, 1e-6); z -= z.max()
        p = np.exp(z); p /= p.sum()
        return top[int(np.random.choice(len(top), p=p))][0]

    def perceber(self, texto, cache_memoria=None, passos=12, temperatura=0.5,
                 alpha=0.5, min_passos=3, mostrar=True,
                 modo="rag", janela_ctx=12,
                 seeds_usados=None, linhas_recentes=None, ultimo_par=None):
        consulta = tuple(r for r in self._acende(texto, criar=False)
                         if r in self.lastros)
        consulta_ctx = self._assemble_query_context(texto, ultimo_par)

        if mostrar:
            print(f"\n[PERCEBEU] '{texto}' | modo={modo} | T={temperatura} | "
                  f"α={alpha} | janela={janela_ctx}")
            if consulta_ctx != texto:
                print(f"  [contexto herdado] '{consulta_ctx}'")

        retrieved = []
        top_linha, top_score, top_idx = None, 0.0, None
        if modo in ("rag", "rag_puro", "rag_curto", "rag_longo"):
            retrieved = self._rag_retrieve(consulta_ctx, top_k=self.rag_top_k,
                                           linhas_recentes=linhas_recentes)
            if mostrar and retrieved:
                print(f"  [RAG] frases recuperadas (cosseno TF-IDF, com bônus curta):")
                for linha, s, idx in retrieved:
                    print(f"    ({s:.3f}) {linha[:78]}")
            if retrieved:
                top_linha, top_score, top_idx = retrieved[0]

        self._last_rag_info = {"top_linha": top_linha, "top_idx": top_idx,
                                "banda": None, "amostrada": False}

        if modo == "rag_puro" and top_linha:
            self._last_rag_info["banda"] = "puro"
            return top_linha

        if modo == "rag" and top_linha:
            banda = self._confidence_band(top_score)

            if banda == "alta" and not self._is_echo(top_linha, texto):
                if mostrar:
                    print(f"  [confiança ALTA ({top_score:.3f}) → resposta direta]")
                self._last_rag_info.update({"banda": "alta"})
                return top_linha

            if banda == "alta" and top_idx is not None and self._is_question(top_linha):
                if top_idx + 1 < len(self._rag_lines):
                    nxt = self._rag_lines[top_idx + 1][0]
                    if (not self._is_question(nxt)
                            and not self._is_echo(nxt, texto)
                            and len(nxt.split()) >= self.rag_min_tokens_linha):
                        if mostrar:
                            print(f"  [Q&A chaining: pergunta → linha {top_idx+1}]")
                        self._last_rag_info.update({"banda": "alta_qa",
                                                    "top_idx": top_idx + 1})
                        return nxt

            if banda == "media":
                amostra = self._sample_from_topk(retrieved[:3], temperatura)
                if amostra:
                    l, s, i = amostra
                    if mostrar:
                        print(f"  [confiança MÉDIA ({s:.3f}) → amostra do top-3]")
                    self._last_rag_info.update({"banda": "media",
                                                "amostrada": True,
                                                "top_idx": i})
                    return l

            if banda == "baixa":
                if mostrar:
                    print(f"  [confiança BAIXA ({top_score:.3f}) → fallback honesto]")
                self._last_rag_info.update({"banda": "baixa"})
                return self._fallback_honesto(retrieved)

        passos_ger = passos
        if modo == "rag_curto": passos_ger = min(5, passos)
        elif modo == "rag": passos_ger = min(5, passos)
        elif modo == "rag_longo": passos_ger = passos

        historico, saida = None, []
        if top_linha and modo in ("rag", "rag_curto", "rag_longo"):
            seeds = tuple(r for r in self._acende(top_linha, criar=False)
                          if r in self.lastros)
            if seeds:
                historico = list(seeds)
                saida = [self.lastros[r].token for r in seeds]

        if historico is None:
            q_state = self._query_state(consulta) if consulta else None
            if modo in ("rag", "rag_curto", "rag_longo", "responder") and q_state is not None:
                seed = self._response_seed(q_state, consulta,
                                           temperatura=max(temperatura, 0.3),
                                           seeds_usados=seeds_usados)
                if seed is not None:
                    historico = [seed]
                    saida = [self.lastros[seed].token]
                else:
                    historico = list(consulta) if consulta else []
            else:
                historico = list(consulta) if consulta else []

        if not historico:
            if mostrar: print("  -> [vazio]")
            return ""

        for passo in range(passos_ger):
            if historico[-1] == self.EOS: break
            ctx_full = tuple(historico[-janela_ctx:])
            ctx_kn   = tuple(historico[-self.n_max:])
            kn = self._kn_rec(ctx_kn)
            if not kn: break
            pesos = self._lastrimy_row(ctx_full)
            votos = Counter()
            for j, rid in enumerate(ctx_full):
                w = pesos[j]
                for prox, cnt in self.lastros[rid].transicoes_aum.items():
                    votos[prox] += w * cnt
            tot = sum(votos.values()) or 1.0
            lastrimy = {k: v / tot for k, v in votos.items()}
            kn_esp = self._contexto_especifico_sucessores(ctx_kn)
            cands = kn_esp | set(self.lastros[ctx_full[-1]].transicoes_aum.keys())
            if self.EOS in kn_esp: cands.add(self.EOS)
            cands = [c for c in cands if c in self.lastros]
            if passo < min_passos:
                filt = [c for c in cands if c != self.EOS]
                if filt: cands = filt
                elif self.EOS in cands: cands = [self.EOS]
            if not cands: break
            probs = {c: (1 - alpha) * kn.get(c, 1e-9)
                     + alpha * lastrimy.get(c, 1e-9) for c in cands}
            s = sum(probs.values()) or 1.0
            probs = {k: v / s for k, v in probs.items()}
            if cache_memoria:
                for c in cands:
                    if self.lastros[c].token in cache_memoria:
                        probs[c] *= 1.5
                s = sum(probs.values()) or 1.0
                probs = {k: v / s for k, v in probs.items()}
            recentes = Counter(self.lastros[r].token
                               for r in ctx_kn[-self.anti_loop_janela:])
            for c in list(probs):
                n = recentes.get(self.lastros[c].token, 0)
                if n > 0: probs[c] *= self.anti_loop_forca ** n
            s = sum(probs.values())
            if s > 1e-12: probs = {k: v / s for k, v in probs.items()}
            probs = self._anti_loop_sintatico(probs, ctx_kn)
            if temperatura > 0:
                z = {k: v ** (1.0 / temperatura) for k, v in probs.items()}
                s = sum(z.values()) or 1.0
                probs = {k: v / s for k, v in z.items()}
            if temperatura <= 0:
                w = max(probs, key=probs.get)
            else:
                itens = list(probs.items())
                ramos = [c for c, _ in itens]
                ps = np.array([p for _, p in itens])
                w = ramos[int(np.random.choice(len(ramos), p=ps / ps.sum()))]
            if w == self.EOS: break
            saida.append(self.lastros[w].token)
            historico.append(w)

        resultado = " ".join(saida)
        if mostrar: print(f"  -> {resultado}")
        return resultado

    def salvar_estado(self, caminho="rlm_state.pkl"):
        with open(caminho, "wb") as f:
            pickle.dump({
                "lastros": self.lastros,
                "transicoes": self.transicoes,
                "unigram": self._unigram,
                "limiar_ctx": self.limiar_ctx,
                "mkn_discounts": self._mkn_discounts,
                "rag_lines": self._rag_lines,
                "rag_idf": self._rag_idf,
                "rag_thresholds": (self.rag_threshold_alta,
                                   self.rag_threshold_media),
            }, f)
        print(f"💾 Estado do RLM salvo em '{caminho}'")

    @classmethod
    def carregar_estado(cls, caminho="rlm_state.pkl", **kwargs):
        if not os.path.exists(caminho): return None
        with open(caminho, "rb") as f: d = pickle.load(f)
        g = cls(**kwargs)
        g.lastros = d["lastros"]; g.transicoes = d["transicoes"]
        g._unigram = d["unigram"]; g.limiar_ctx = d["limiar_ctx"]
        g._mkn_discounts = d.get("mkn_discounts", (0.5, 1.0, 1.5))
        g._rag_lines = d.get("rag_lines", []); g._rag_idf = d.get("rag_idf", {})
        th = d.get("rag_thresholds", (g.rag_threshold_alta, g.rag_threshold_media))
        g.rag_threshold_alta, g.rag_threshold_media = th
        g._kv_init()
        print(f"⚡ Estado carregado de '{caminho}' (instantâneo).")
        if g._rag_lines:
            print(f"   [RAG] {len(g._rag_lines)} frases indexadas")
        return g


# ═══════════════════════════════════════════════════════════════════════
# PROMPT HANDLE — "prompt como ambiente" (paper RLM, MIT 2026)
# ═══════════════════════════════════════════════════════════════════════
class PromptHandle:
    def __init__(self, rlm):
        self.rlm = rlm
        self._linhas = [l for l, _ in rlm._rag_lines]
        self._emb = np.full((len(self._linhas), 5), np.nan, dtype=float)
        for i, linha in enumerate(self._linhas):
            embs = []
            for t in _tokenize(linha):
                rid = rlm._hash(t)
                la = rlm.lastros.get(rid)
                if la is not None and la.perfil is not None:
                    embs.append(la.perfil)
            if embs:
                self._emb[i] = np.mean(embs, axis=0)

    def __len__(self):
        return len(self._linhas)

    def peek(self, i):
        return self._linhas[i] if 0 <= i < len(self._linhas) else ""

    def chunks(self, k=8):
        for i in range(0, len(self._linhas), k):
            yield (i, "\n".join(self._linhas[i:i + k]))

    def semantic_top(self, pergunta, m=20):
        if not len(self._linhas):
            return []
        q_parts = []
        for t in _tokenize(pergunta):
            rid = self.rlm._hash(t)
            la = self.rlm.lastros.get(rid)
            if la is not None and la.perfil is not None:
                q_parts.append(la.perfil)
        if not q_parts:
            return list(range(min(m, len(self._linhas))))
        q_emb = np.mean(q_parts, axis=0)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        valid = ~np.isnan(self._emb).any(axis=1)
        sims = np.full(len(self._linhas), -1.0, dtype=float)
        if valid.any():
            E = self._emb[valid]
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            sims[valid] = E @ q_emb
        return np.argsort(-sims)[:m].tolist()


# ═══════════════════════════════════════════════════════════════════════
# MOTOR LLM RECURSIVO — agora com Attention Residuals
# ═══════════════════════════════════════════════════════════════════════
class MotorLLMRecursivo:
    """
    Orquestra o RLM em raciocínio multi-nível.

    Estratégias:
      "hibrido"        — decomposição + crítica + refino
      "decomposicao"   — só decompõe
      "refino"         — só refina
      "critica"        — só crítica + refino
      "denso"          — SEMPRE processamento denso (PromptHandle + varredura)
      "hibrido_denso"  — denso SÓ se _detectar_densa() identificar agregação

    AttnRes:
      use_attn_res=True → síntese usa AttentionResidual (softmax sobre respostas)
      use_attn_res=False → fallback Jaccard (comportamento v6.7)
    """

    _CONECTORES = re.compile(
        r'\s*(?:;|\.\.\.|,)\s*|'
        r'\s+(?:e|ou|mas|porém|porque|pois|também|além\s+disso|'
        r'depois|então|enquanto|quando|se|caso)\s+',
        re.IGNORECASE,
    )

    def __init__(self, rlm, max_depth=2, max_refinos=2,
                 limiar_convergencia=0.82, strategy="hibrido",
                 min_tokens_decompor=5, verbose=False,
                 max_linhas_denso=40, use_attn_res=True):
        self.rlm = rlm
        self.max_depth = max_depth
        self.max_refinos = max_refinos
        self.limiar_convergencia = limiar_convergencia
        self.strategy = strategy
        self.min_tokens_decompor = min_tokens_decompor
        self.verbose = verbose
        self.max_linhas_denso = max_linhas_denso
        self.use_attn_res = use_attn_res
        self._attn_res = AttentionResidual(rlm)
        self._trace = []
        self._calls = 0
        self._last_info = {}
        self._cached_handle = None
        self._cached_handle_len = -1

    # ─── utilidades ────────────────────────────────────────────────
    def _sim(self, a, b):
        ta, tb = set(_tokenize(a)), set(_tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _decompor(self, texto):
        toks = _tokenize(texto)
        if len(toks) < self.min_tokens_decompor:
            return [texto]
        partes = [p.strip() for p in self._CONECTORES.split(texto) if p.strip()]
        partes = [p for p in partes if len(_tokenize(p)) >= 3]
        if len(partes) <= 1:
            return [texto]
        if texto.count("?") >= 2:
            return partes
        lens = [len(_tokenize(p)) for p in partes]
        if min(lens) / max(lens) >= 0.55:
            return partes
        return [texto]

    def _get_handle(self):
        n_current = len(self.rlm._rag_lines)
        if (self._cached_handle is None
                or self._cached_handle_len != n_current):
            self._cached_handle = PromptHandle(self.rlm)
            self._cached_handle_len = n_current
        return self._cached_handle

    def _perceber(self, pergunta, kwargs):
        self._calls += 1
        return self.rlm.perceber(
            pergunta,
            cache_memoria=kwargs.get("cache_memoria"),
            passos=kwargs.get("passos", 12),
            temperatura=kwargs.get("temperatura", 0.5),
            alpha=kwargs.get("alpha", 0.5),
            modo=kwargs.get("sub_modo", "rag"),
            janela_ctx=kwargs.get("janela_ctx", 12),
            seeds_usados=kwargs.get("seeds_usados"),
            linhas_recentes=kwargs.get("linhas_recentes"),
            ultimo_par=kwargs.get("ultimo_par"),
            mostrar=False,
        )

    def batch_subcall(self, perguntas, **kwargs):
        return [self._perceber(q, kwargs) for q in perguntas]

    def _detectar_densa(self, pergunta):
        toks = _tokenize(pergunta)
        if len(toks) < 4:
            return False
        low = pergunta.lower()
        if re.search(
            r'\b(todos?|todas?|liste|listar|enumere|quantos?|quantas?|quais)\b',
            low):
            return True
        if pergunta.count("?") >= 2:
            return True
        if 'entre' in low and re.search(r'rela(ç|c)(ão|ao|ões|oes)', low):
            return True
        return False

    # ─── síntese com AttnRes ───────────────────────────────────────
    def _sintetizar(self, respostas, pergunta=None):
        respostas = [r for r in respostas
                     if r and not r.startswith("Não tenho")]
        if not respostas:
            return None
        if len(respostas) == 1:
            return respostas[0]

        # AttnRes: atenção softmax sobre a profundidade (respostas)
        if self.use_attn_res and pergunta is not None:
            return self._attn_res.forward(pergunta, respostas)

        # Fallback Jaccard (v6.7)
        scores = []
        for i, r in enumerate(respostas):
            sim = sum(self._sim(r, o)
                      for j, o in enumerate(respostas) if j != i)
            scores.append((sim / max(len(respostas) - 1, 1), i))
        scores.sort(reverse=True)
        return respostas[scores[0][1]]

    def _sintetizar_denso(self, respostas, pergunta):
        return self._sintetizar(respostas, pergunta=pergunta)

    # ─── núcleo do modo denso ──────────────────────────────────────
    def _raciocinar_denso(self, pergunta, kwargs):
        handle = kwargs.pop("_prompt_handle", None) or self._get_handle()
        n = len(handle)
        if n == 0:
            return self._perceber(pergunta, kwargs)

        if n > 60:
            retrieved = self.rlm._rag_retrieve(
                pergunta, top_k=self.max_linhas_denso)
            indices = [i for _, _, i in retrieved] or \
                      list(range(min(self.max_linhas_denso, n)))
        elif n > 20:
            indices = handle.semantic_top(pergunta, m=self.max_linhas_denso)
        else:
            indices = list(range(n))

        if self.verbose:
            print(f"  [denso] varrendo {len(indices)}/{n} linhas")

        relevantes = []
        for i in indices:
            linha = handle.peek(i)
            if not linha:
                continue
            consulta = f"{pergunta} {linha}"
            self._calls += 1
            r = self.rlm.perceber(
                consulta, modo="rag", mostrar=False,
                temperatura=0.0,
                cache_memoria=kwargs.get("cache_memoria"),
                alpha=kwargs.get("alpha", 0.5),
                janela_ctx=kwargs.get("janela_ctx", 12),
            )
            info = getattr(self.rlm, "_last_rag_info", {}) or {}
            banda = info.get("banda")
            if banda in ("alta", "media", "alta_qa") and r:
                relevantes.append((i, linha, r))

        if self.verbose:
            print(f"  [denso] {len(relevantes)} linhas com confiança ≥ média")

        if not relevantes:
            amostra = "\n".join(handle.peek(i) for i in indices[:8])
            return self._perceber(
                f"{pergunta}\n\nContexto próximo:\n{amostra}", kwargs)

        sintese = self._sintetizar_denso([r for _, _, r in relevantes], pergunta)
        if sintese:
            self._trace.append(("denso", pergunta, sintese))
            return sintese
        return self._perceber(pergunta, kwargs)

    # ─── refinamento iterativo ─────────────────────────────────────
    def _refinar(self, pergunta, resposta0, profundidade, kwargs):
        atual = resposta0
        historico = [atual]
        for i in range(self.max_refinos):
            consulta_refino = f"{pergunta} {atual}"
            nova = self._perceber(consulta_refino, kwargs)
            if not nova or nova == atual:
                break
            sim = self._sim(atual, nova)
            historico.append(nova)
            if self.verbose:
                print(f"    {'  ' * profundidade}[refino {i}] sim={sim:.2f} | {nova[:60]}")
            if sim >= self.limiar_convergencia:
                atual = nova if len(nova) > len(atual) else atual
                break
            atual = nova
        return atual, historico

    def _critica(self, pergunta, resposta):
        if not resposta or resposta.startswith("Não tenho"):
            return None
        toks = [t for t in _tokenize(resposta) if len(t) > 3]
        if not toks:
            return None
        amostra = random.sample(toks, min(2, len(toks)))
        return (f"o que é {amostra[0]}? "
                f"qual a relação entre {amostra[0]} e {pergunta}?")

    # ─── núcleo recursivo ──────────────────────────────────────────
    def raciocinar(self, pergunta, **kwargs):
        self._trace = []
        self._calls = 0
        self._last_info = {"depth": self.max_depth,
                           "refinos": self.max_refinos,
                           "strategy": self.strategy}

        modo_denso_explicito = kwargs.pop("modo_denso", False)
        usar_denso = (
            modo_denso_explicito
            or self.strategy == "denso"
            or (self.strategy == "hibrido_denso"
                and self._detectar_densa(pergunta))
        )

        if usar_denso:
            if self.verbose:
                print(f"  [motor] modo Denso ativado")
            return self._raciocinar_denso(pergunta, kwargs)

        if self.strategy == "hibrido_denso":
            saved = self.strategy
            self.strategy = "hibrido"
            try:
                return self._raciocinar(pergunta, 0, kwargs)
            finally:
                self.strategy = saved

        return self._raciocinar(pergunta, 0, kwargs)

    def _raciocinar(self, pergunta, profundidade, kwargs):
        prefixo = "  " * profundidade

        if profundidade > self.max_depth:
            return self._perceber(pergunta, kwargs)

        if self.strategy in ("hibrido", "decomposicao"):
            partes = self._decompor(pergunta)
            if len(partes) > 1:
                if self.verbose:
                    print(f"{prefixo}[decompõe d={profundidade}] {len(partes)} partes")
                respostas = []
                for sub in partes:
                    r = self._raciocinar(sub, profundidade + 1, kwargs)
                    if r:
                        respostas.append(r)
                sintese = self._sintetizar(respostas, pergunta=pergunta)
                if sintese:
                    self._trace.append(("decomposta", pergunta, sintese))
                    if self.strategy == "decomposicao":
                        return sintese
                    if self.max_refinos > 0:
                        refinada, _ = self._refinar(
                            pergunta, sintese, profundidade, kwargs)
                        self._trace.append(("refinada", pergunta, refinada))
                        return refinada
                    return sintese

        base = self._perceber(pergunta, kwargs)
        if not base:
            return base

        if (self.strategy in ("hibrido", "critica")
                and profundidade < self.max_depth):
            critica = self._critica(pergunta, base)
            if critica:
                if self.verbose:
                    print(f"{prefixo}[crítica] {critica[:60]}")
                info_extra = self._raciocinar(critica, profundidade + 1, kwargs)
                if info_extra and info_extra != base:
                    base = self._sintetizar([base, info_extra],
                                            pergunta=pergunta) or base
                    self._trace.append(("critica", pergunta, base))

        if self.strategy in ("hibrido", "refino", "critica"):
            refinada, _ = self._refinar(pergunta, base, profundidade, kwargs)
            self._trace.append(("refinada", pergunta, refinada))
            return refinada

        return base


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ARQUIVO_CORPUS = "corpus.txt"
    ARQUIVO_ESTADO = "rlm_state.pkl"

    print("=" * 70)
    print("RLM-Lastrimy v6.8 — AttnRes + Motor Recursivo + PromptHandle")
    print("=" * 70)

    kwargs = dict(
        n_max=4, D=0.75, tau_pos=2.0, k_vizinhos=5,
        limiar_forma=0.70, limiar_ctx=0.15, peso_vizinho=0.5,
        anti_loop_sintatico_forca=0.2, tau_perfil=1.0,
        auto_ajuste_limiar=True, usar_rede_neural=True,
        gns_max_vocab=800, gns_hidden=24, gns_epocas=400, gns_lr=0.05,
        rag_top_k=5, rag_threshold_alta=0.60, rag_threshold_media=0.35,
    )

    rlm = RLM.carregar_estado(ARQUIVO_ESTADO, **kwargs)
    if rlm is None:
        if not os.path.exists(ARQUIVO_CORPUS):
            print(f"[ERRO] '{ARQUIVO_CORPUS}' não encontrado.")
            raise SystemExit(1)
        print("\n⚙️ Iniciando treinamento pesado...")
        rlm = RLM(**kwargs)
        rlm.absorver_arquivo(ARQUIVO_CORPUS)
        rlm.salvar_estado(ARQUIVO_ESTADO)

    # use_attn_res=True ativa a síntese por AttentionResidual
    motor = MotorLLMRecursivo(rlm, max_depth=2, max_refinos=2,
                              strategy="hibrido", verbose=True,
                              max_linhas_denso=40, use_attn_res=True)

    MODOS = ("rag", "rag_puro", "rag_curto", "rag_longo",
             "responder", "continuar",
             "recursivo", "recursivo_profundo",
             "denso", "hibrido_denso")

    print("\nModos:")
    print("  rag | rag_puro | rag_curto | rag_longo | responder | continuar")
    print("  recursivo (leve) | recursivo_profundo (pesado)")
    print("  denso (varredura total) | hibrido_denso (detector automático)")
    print("Comandos: sair | T=0.3 | a=0.7 | limpar | thA=0.6 | thM=0.35 | m=rag")
    print("-" * 70)

    temperatura = 0.5
    alpha = 0.5
    modo = "rag"
    janela_ctx = 12
    cache_memoria = set()
    seeds_usados = set()
    linhas_usadas = []
    ultimo_par = None

    while True:
        try:
            entrada = input("\nvocê > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not entrada: continue
        low = entrada.lower()

        if low in ("sair", "exit", "quit"): break
        if low == "limpar":
            cache_memoria.clear(); seeds_usados.clear()
            linhas_usadas = []; ultimo_par = None
            print("[Memória limpa]"); continue
        if low.startswith("t="):
            temperatura = float(entrada.split("=", 1)[1]); print(f"[T={temperatura}]"); continue
        if low.startswith("a="):
            alpha = float(entrada.split("=", 1)[1]); print(f"[α={alpha}]"); continue
        if low.startswith("j="):
            janela_ctx = int(entrada.split("=", 1)[1]); print(f"[janela={janela_ctx}]"); continue
        if low.startswith("tha="):
            rlm.rag_threshold_alta = float(entrada.split("=", 1)[1])
            print(f"[thA={rlm.rag_threshold_alta}]"); continue
        if low.startswith("thm="):
            rlm.rag_threshold_media = float(entrada.split("=", 1)[1])
            print(f"[thM={rlm.rag_threshold_media}]"); continue

        mode_str = None
        if low in MODOS: mode_str = low
        elif low.startswith("m=") or low.startswith("m "): mode_str = low[2:].strip()
        elif low.startswith("modo=") or low.startswith("modo "): mode_str = low[5:].strip()
        if mode_str and "|" in mode_str:
            partes = [p.strip().replace("m=", "").replace("modo=", "")
                      for p in mode_str.split("|")]
            validas = [p for p in partes if p in MODOS]
            if validas: mode_str = validas[-1]
        if mode_str is not None:
            if mode_str in MODOS:
                modo = mode_str; print(f"[modo={modo}]")
            else:
                print(f"[modo inválido — use: {', '.join(MODOS)}]")
            continue

        print("-" * 70)

        if modo in ("recursivo", "recursivo_profundo"):
            motor.max_depth   = 2 if modo == "recursivo" else 3
            motor.max_refinos = 2 if modo == "recursivo" else 3
            motor.strategy    = "hibrido"
            resposta = motor.raciocinar(
                entrada,
                cache_memoria=cache_memoria,
                temperatura=temperatura,
                alpha=alpha,
                janela_ctx=janela_ctx,
                seeds_usados=seeds_usados,
                linhas_recentes=linhas_usadas,
                ultimo_par=ultimo_par,
                sub_modo="rag",
            )
            if motor._trace:
                print(f"  [motor] {motor._calls} subcalls | "
                      f"{len(motor._trace)} passos")
                for tipo, q, r in motor._trace[-3:]:
                    print(f"    · {tipo}: '{q[:42]}' → '{r[:42]}'")

        elif modo == "denso":
            motor.strategy = "denso"
            motor.verbose  = True
            resposta = motor.raciocinar(
                entrada,
                cache_memoria=cache_memoria,
                temperatura=temperatura,
                alpha=alpha,
                janela_ctx=janela_ctx,
                ultimo_par=ultimo_par,
                sub_modo="rag",
            )
            print(f"  [denso] {motor._calls} subcalls")

        elif modo == "hibrido_denso":
            motor.strategy = "hibrido_denso"
            motor.verbose  = True
            resposta = motor.raciocinar(
                entrada,
                cache_memoria=cache_memoria,
                temperatura=temperatura,
                alpha=alpha,
                janela_ctx=janela_ctx,
                seeds_usados=seeds_usados,
                linhas_recentes=linhas_usadas,
                ultimo_par=ultimo_par,
                sub_modo="rag",
            )
            print(f"  [hibrido_denso] {motor._calls} subcalls")

        else:
            resposta = rlm.perceber(
                entrada,
                cache_memoria=cache_memoria,
                passos=12,
                temperatura=temperatura,
                alpha=alpha,
                modo=modo,
                janela_ctx=janela_ctx,
                seeds_usados=seeds_usados if modo == "responder" else None,
                linhas_recentes=linhas_usadas if modo == "rag" else None,
                ultimo_par=ultimo_par,
            )

        resposta = resposta.replace("⟨EOS⟩", "").replace("⟨BOS⟩", "").strip()

        if not resposta:
            print("\nRLM  > [Silêncio]")
        else:
            print(f"\nRLM  > {resposta}")

            stopwords = {"o", "a", "de", "que", "e", "do", "da", "em", "um",
                         "uma", "para", "com", "não", "sim", "você", "eu",
                         "tá", "é", "se", "por", "os", "as", "no", "na"}
            cache_memoria = (set(re.findall(r'\b\w+\b',
                                           (entrada + " " + resposta).lower()))
                             - stopwords)

            if modo == "responder" and resposta:
                seeds_usados.add(rlm._hash(resposta.split()[0].lower()))
                if len(seeds_usados) > 8: seeds_usados.pop()

            if modo == "rag":
                info = getattr(rlm, "_last_rag_info", {})
                if info.get("top_idx") is not None:
                    linhas_usadas.append(info["top_idx"])
                    if len(linhas_usadas) > 6:
                        linhas_usadas.pop(0)

            ultimo_par = (entrada, resposta)

        print("-" * 70)
