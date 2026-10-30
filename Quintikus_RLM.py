"""
RLM-Lastrimy v8.2 — Motor Recursivo + Auto-Evolução Lifelong Adaptativa
=========================================================================
- AdaptadorLinear: ajusta hiperparâmetros linearmente com o tempo
- MotorAutoEvolucao: detecta lacunas e faz perguntas ao usuário
- MotorLLMRecursivo escolhe modo (recursivo/construir/denso) por performance
- Persiste histórico de evolução em rlm_evolucao.pkl
- Adaptação dinâmica: thresholds, temperatura, alpha, n_frags, modo padrão
"""

import hashlib
import math
import random
import re
import os
import pickle
from collections import Counter, defaultdict, deque
import numpy as np

_SENT = "\uE000"


# ═══════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════════
class PositionalEncoding:
    def __init__(self, dropout=0.0):
        self.dropout = dropout

    @staticmethod
    def sinusoidal(seq_len, d_model):
        pe = np.zeros((seq_len, d_model), dtype=float)
        if d_model == 0 or seq_len == 0:
            return pe
        pos = np.arange(seq_len, dtype=float)[:, None]
        div = np.exp(np.arange(0, d_model, 2, dtype=float) *
                     -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(pos * div)
        else:
            pe[:, 1::2] = np.cos(pos * div[:pe[:, 1::2].shape[1]])
        return pe

    def apply(self, X):
        if self.dropout > 0:
            mask = (np.random.rand(*X.shape) >= self.dropout).astype(float)
            X = X * mask / (1.0 - self.dropout)
        return X + self.sinusoidal(X.shape[0], X.shape[1])


# ═══════════════════════════════════════════════════════════════════════
# ADAMW
# ═══════════════════════════════════════════════════════════════════════
class AdamW:
    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.001):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def step(self, key, param, grad, apply_decay=True):
        if key not in self.state:
            self.state[key] = {
                "m": np.zeros_like(param, dtype=float),
                "v": np.zeros_like(param, dtype=float),
                "t": 0,
            }
        s = self.state[key]
        s["t"] += 1
        if apply_decay and self.weight_decay > 0:
            param *= (1.0 - self.lr * self.weight_decay)
        s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * grad
        s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * (grad * grad)
        m_hat = s["m"] / (1 - self.beta1 ** s["t"])
        v_hat = s["v"] / (1 - self.beta2 ** s["t"])
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param


# ═══════════════════════════════════════════════════════════════════════
# GNS
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
    def __init__(self, tamanhos, alpha=1.0, softmax_saida=True,
                 usar_pos_enc=False, pos_enc_dropout=0.0,
                 usar_adamw=True, adamw_lr=0.01,
                 adamw_betas=(0.9, 0.999),
                 adamw_eps=1e-8, adamw_wd=0.001):
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

        self.usar_pos_enc = usar_pos_enc
        self.pos_enc = PositionalEncoding(dropout=pos_enc_dropout) \
            if usar_pos_enc else None
        self.usar_adamw = usar_adamw
        self.opt = AdamW(lr=adamw_lr, betas=adamw_betas,
                         eps=adamw_eps, weight_decay=adamw_wd) \
            if usar_adamw else None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if self.usar_pos_enc and self.pos_enc is not None:
            X = self.pos_enc.apply(X)
        ativ = [X]
        for camada in self.camadas:
            X = camada.forward_batch(X)
            ativ.append(X)
        return ativ[-1], ativ

    def treinar(self, X, Y, epocas=400, lr=0.05, tol_erro=0.01, verbose=True):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        n_ex, n_cam = len(X), len(self.camadas)
        hist = []
        if verbose:
            modo = (f"AdamW(lr={self.opt.lr}, wd={self.opt.weight_decay})"
                    if self.usar_adamw else f"SGD(lr={lr})")
            print(f"      [otimizador] {modo} | "
                  f"pos_enc={'ON' if self.usar_pos_enc else 'OFF'}")
        for ep in range(epocas):
            saida, ativ = self.forward(X)
            erros = saida - Y
            erro = float(np.mean(np.sum(erros ** 2, axis=1)))
            delta = erros / n_ex
            for c in range(n_cam - 1, -1, -1):
                cam = self.camadas[c]
                grad_W = delta.T @ ativ[c]
                grad_b = delta.sum(axis=0)
                if self.usar_adamw:
                    cam.pesos = self.opt.step(
                        (c, "W"), cam.pesos, grad_W, apply_decay=True)
                    cam.bias = self.opt.step(
                        (c, "b"), cam.bias, grad_b, apply_decay=False)
                else:
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
# ADAPTADOR LINEAR — ajusta hiperparâmetros ao longo do tempo
# ═══════════════════════════════════════════════════════════════════════
class AdaptadorLinear:
    """
    Adapta hiperparâmetros linearmente com base em métricas observadas.

    Métricas rastreadas:
      - media_reward:       EMA do reward (0.9 antigo + 0.1 novo)
      - taxa_sucesso:       % de interações com reward > 0
      - taxa_fallback:      % de respostas em banda baixa
      - diversidade:        nº de padrões distintos / interações
      - confianca:          0→1, cresce com sucesso contínuo

    Ajustes lineares:
      - temperatura:    0.5 → 0.2 (menos exploração com mais confiança)
      - threshold_alta: 0.55 → 0.45 (mais exigente se errou muito)
      - threshold_media:0.30 → 0.40 (mais exigente se errou muito)
      - alpha:          0.5 → 0.7 (mais Lastrimy quando MKN falha)
      - n_frags:        3 → 6 (mais fragmentos com corpus maior)
      - modo_padrao:    performance histórica escolhe
    """

    MODOS = ("recursivo", "construir", "denso")

    def __init__(self):
        self.turnos = 0
        self.media_reward = 0.0
        self.rewards_recentes = deque(maxlen=20)
        self.taxa_sucesso = 0.5
        self.taxa_fallback = 0.0
        self.confianca = 0.3
        self.padroes_distintos = 0
        self.fallbacks_recentes = 0

        # performance por modo
        self.performance_modo = {
            "recursivo": {"usos": 0, "sucesso": 0, "soma_reward": 0.0},
            "construir": {"usos": 0, "sucesso": 0, "soma_reward": 0.0},
            "denso":     {"usos": 0, "sucesso": 0, "soma_reward": 0.0},
        }

        # hiperparâmetros (mutáveis)
        self.temperatura_sugerida = 0.5
        self.threshold_alta = 0.55
        self.threshold_media = 0.30
        self.alpha_sugerido = 0.5
        self.n_frags_sugerido = 3
        self.modo_padrao = "recursivo"

        # histórico de adaptação (para visualizar evolução linear)
        self.historico = []

    def registrar(self, reward, modo, foi_fallback):
        self.turnos += 1
        self.rewards_recentes.append(reward)
        self.media_reward = 0.9 * self.media_reward + 0.1 * reward

        if reward > 0:
            self.taxa_sucesso = 0.9 * self.taxa_sucesso + 0.1 * 1.0
        else:
            self.taxa_sucesso = 0.9 * self.taxa_sucesso + 0.1 * 0.0

        if foi_fallback:
            self.fallbacks_recentes += 1
        else:
            self.fallbacks_recentes = max(0, self.fallbacks_recentes - 1)
        self.taxa_fallback = self.fallbacks_recentes / max(self.turnos, 1)

        # performance por modo
        if modo in self.performance_modo:
            p = self.performance_modo[modo]
            p["usos"] += 1
            p["soma_reward"] += reward
            if reward > 0:
                p["sucesso"] += 1

        # confiança: cresce com sucesso, decresce com fallback
        delta_conf = (0.01 if reward > 0 else -0.015)
        self.confianca = max(0.05, min(0.95, self.confianca + delta_conf))

        # adapta hiperparâmetros linearmente
        self._adaptar()
        self._escolher_melhor_modo()

    def _adaptar(self):
        # temperatura: decresce com confiança (0.5 → 0.2)
        self.temperatura_sugerida = 0.5 - 0.3 * self.confianca

        # threshold_alta: sobe se fallback alto, desce se sucesso alto
        if self.taxa_fallback > 0.3:
            self.threshold_alta = min(0.70, self.threshold_alta + 0.01)
            self.threshold_media = min(0.50, self.threshold_media + 0.01)
        elif self.taxa_sucesso > 0.7:
            self.threshold_alta = max(0.40, self.threshold_alta - 0.005)
            self.threshold_media = max(0.20, self.threshold_media - 0.005)

        # alpha: mais Lastrimy se MKN falha, mais MKN se funciona
        self.alpha_sugerido = 0.5 + 0.2 * self.taxa_sucesso

        # n_frags: cresce com confiança (3 → 6)
        self.n_frags_sugerido = 3 + int(3 * self.confianca)

        # registra snapshot a cada 5 turnos
        if self.turnos % 5 == 0:
            self.historico.append({
                "turno": self.turnos,
                "confianca": round(self.confianca, 3),
                "taxa_sucesso": round(self.taxa_sucesso, 3),
                "taxa_fallback": round(self.taxa_fallback, 3),
                "temperatura": round(self.temperatura_sugerida, 3),
                "thr_alta": round(self.threshold_alta, 3),
                "thr_media": round(self.threshold_media, 3),
                "alpha": round(self.alpha_sugerido, 3),
                "n_frags": self.n_frags_sugerido,
                "modo": self.modo_padrao,
            })

    def _escolher_melhor_modo(self):
        """Escolhe modo com maior média de reward (mín. 3 usos)."""
        melhor_modo = self.modo_padrao
        melhor_media = -999.0
        for modo, p in self.performance_modo.items():
            if p["usos"] < 3:
                continue
            media = p["soma_reward"] / p["usos"]
            if media > melhor_media:
                melhor_media = media
                melhor_modo = modo
        self.modo_padrao = melhor_modo

    def sugerir_modo(self):
        return self.modo_padrao

    def estado_atual(self):
        return {
            "turnos": self.turnos,
            "media_reward": round(self.media_reward, 3),
            "confianca": round(self.confianca, 3),
            "taxa_sucesso": round(self.taxa_sucesso, 3),
            "taxa_fallback": round(self.taxa_fallback, 3),
            "temperatura": round(self.temperatura_sugerida, 3),
            "thr_alta": round(self.threshold_alta, 3),
            "thr_media": round(self.threshold_media, 3),
            "alpha": round(self.alpha_sugerido, 3),
            "n_frags": self.n_frags_sugerido,
            "modo_padrao": self.modo_padrao,
            "performance_modo": {
                m: {
                    "usos": p["usos"],
                    "media_reward": (round(p["soma_reward"] / p["usos"], 3)
                                     if p["usos"] > 0 else 0.0)
                }
                for m, p in self.performance_modo.items()
            },
            "historico": self.historico[-10:],
        }

    def salvar(self, caminho="rlm_adaptador.pkl"):
        with open(caminho, "wb") as f:
            pickle.dump({
                "turnos": self.turnos,
                "media_reward": self.media_reward,
                "taxa_sucesso": self.taxa_sucesso,
                "taxa_fallback": self.taxa_fallback,
                "confianca": self.confianca,
                "performance_modo": self.performance_modo,
                "temperatura_sugerida": self.temperatura_sugerida,
                "threshold_alta": self.threshold_alta,
                "threshold_media": self.threshold_media,
                "alpha_sugerido": self.alpha_sugerido,
                "n_frags_sugerido": self.n_frags_sugerido,
                "modo_padrao": self.modo_padrao,
                "historico": self.historico,
            }, f)

    def carregar(self, caminho="rlm_adaptador.pkl"):
        if not os.path.exists(caminho):
            return False
        with open(caminho, "rb") as f:
            d = pickle.load(f)
        self.turnos = d.get("turnos", 0)
        self.media_reward = d.get("media_reward", 0.0)
        self.taxa_sucesso = d.get("taxa_sucesso", 0.5)
        self.taxa_fallback = d.get("taxa_fallback", 0.0)
        self.confianca = d.get("confianca", 0.3)
        self.performance_modo = d.get("performance_modo", self.performance_modo)
        self.temperatura_sugerida = d.get("temperatura_sugerida", 0.5)
        self.threshold_alta = d.get("threshold_alta", 0.55)
        self.threshold_media = d.get("threshold_media", 0.30)
        self.alpha_sugerido = d.get("alpha_sugerido", 0.5)
        self.n_frags_sugerido = d.get("n_frags_sugerido", 3)
        self.modo_padrao = d.get("modo_padrao", "recursivo")
        self.historico = d.get("historico", [])
        return True


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
    EMB_DIM = 21

    def __init__(self, n_max=4, D=0.75, tau_pos=2.0, k_vizinhos=5,
                 limiar_forma=0.70, limiar_ctx=0.15, peso_vizinho=0.5,
                 anti_loop_janela=3, anti_loop_forca=0.3,
                 anti_loop_sintatico_forca=0.2, tau_perfil=1.0,
                 auto_ajuste_limiar=True, usar_rede_neural=True,
                 gns_max_vocab=800, gns_hidden=24, gns_epocas=400,
                 gns_lr=0.05, gns_pos_enc=False, gns_adamw=True,
                 gns_adamw_lr=0.01, gns_adamw_wd=0.001,
                 emb_threshold_alta=0.55, emb_threshold_media=0.30,
                 emb_penalidade_reuso=0.5, emb_echo_limiar=0.6,
                 emb_min_tokens=2):
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
        self.gns_pos_enc = gns_pos_enc
        self.gns_adamw = gns_adamw
        self.gns_adamw_lr = gns_adamw_lr
        self.gns_adamw_wd = gns_adamw_wd

        self.emb_threshold_alta  = emb_threshold_alta
        self.emb_threshold_media = emb_threshold_media
        self.emb_penalidade_reuso = emb_penalidade_reuso
        self.emb_echo_limiar = emb_echo_limiar
        self.emb_min_tokens = emb_min_tokens

        self.BOS = self._hash((_SENT, "BOS", -1))
        self.EOS = self._hash((_SENT, "EOS", -2))
        self.lastros = {}; self.transicoes = {}
        self._unigram = {}; self._stats_globais = {}
        self._mkn_discounts = (0.5, 1.0, 1.5)
        self._rag_lines = []
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

    def _embed_token(self, tok):
        rid = self._hash(tok)
        la = self.lastros.get(rid)
        if la is None:
            return np.zeros(self.EMB_DIM, dtype=float)
        perfil = la.perfil if la.perfil is not None else np.zeros(5)
        return np.concatenate([perfil, la.lf_forma])

    def _embed_texto(self, texto):
        toks = _tokenize(texto)
        embs = []
        for t in toks:
            rid = self._hash(t)
            if rid in self.lastros:
                embs.append(self._embed_token(t))
        if not embs:
            return np.zeros(self.EMB_DIM, dtype=float)
        return np.mean(embs, axis=0)

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

    def _build_embed_index(self, texto):
        linhas = []
        for linha in texto.strip().splitlines():
            linha = linha.strip()
            if not linha: continue
            toks = _tokenize(linha)
            if len(toks) < self.emb_min_tokens: continue
            emb = self._embed_texto(linha)
            if np.linalg.norm(emb) < 1e-9: continue
            linhas.append((linha, emb))
        self._rag_lines = linhas
        dim = linhas[0][1].shape[0] if linhas else 0
        print(f"  [EMBED] índice neural: {len(linhas)} frases | dim={dim}")

    def _retrieve_por_embedding(self, texto, top_k=5, linhas_recentes=None):
        if not self._rag_lines: return []
        q = self._embed_texto(texto)
        qn = np.linalg.norm(q)
        if qn < 1e-9: return []
        q = q / qn

        negadores = {"não", "nao", "nunca", "jamais", "sem", "nem"}
        q_negado = bool(set(_tokenize(texto)) & negadores)

        scores = []
        for i, (linha, emb) in enumerate(self._rag_lines):
            en = np.linalg.norm(emb)
            if en < 1e-9: continue
            s = float(np.dot(q, emb / en))
            if s <= 0: continue
            if linhas_recentes and i in linhas_recentes:
                s *= self.emb_penalidade_reuso
            if q_negado:
                l_toks = set(_tokenize(linha))
                if not (l_toks & negadores):
                    s *= 0.5
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
        return len(l_toks & t_toks) / len(l_toks | t_toks) > self.emb_echo_limiar

    def _confidence_band(self, score, thr_a=None, thr_m=None):
        if thr_a is None: thr_a = self.emb_threshold_alta
        if thr_m is None: thr_m = self.emb_threshold_media
        if score >= thr_a:  return "alta"
        if score >= thr_m:  return "media"
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
        print(f"  [GNS] V={V} | docs={N_doc} | "
              f"arquitetura={V}-{self.gns_hidden}-{self.gns_hidden}-{V}")
        print(f"  [GNS] pos_enc={self.gns_pos_enc} | adamw={self.gns_adamw}")
        rede = RedeLinhagem(
            [V, self.gns_hidden, self.gns_hidden, V],
            alpha=1.0, softmax_saida=False,
            usar_pos_enc=self.gns_pos_enc,
            usar_adamw=self.gns_adamw,
            adamw_lr=self.gns_adamw_lr,
            adamw_wd=self.gns_adamw_wd)
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
        print(f"  [GNS] erro final={hist[-1]:.6f} | "
              f"perfis={n_aplicados}/{len(self.lastros)}")
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
        self._build_embed_index(texto)
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
                 seeds_usados=None, linhas_recentes=None, ultimo_par=None,
                 thr_alta=None, thr_media=None):
        consulta = tuple(r for r in self._acende(texto, criar=False)
                         if r in self.lastros)
        consulta_ctx = self._assemble_query_context(texto, ultimo_par)
        n_input = len(_tokenize(texto))

        if mostrar:
            print(f"\n[PERCEBEU] '{texto}' | modo={modo} | T={temperatura}")
            if consulta_ctx != texto:
                print(f"  [contexto herdado] '{consulta_ctx}'")

        retrieved = []
        top_linha, top_score, top_idx = None, 0.0, None
        if modo in ("rag", "rag_puro", "rag_curto", "rag_longo"):
            retrieved = self._retrieve_por_embedding(
                consulta_ctx, top_k=5, linhas_recentes=linhas_recentes)
            if mostrar and retrieved:
                print(f"  [EMBED] frases recuperadas (cosseno neural):")
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
            ta = thr_alta if thr_alta is not None else self.emb_threshold_alta
            tm = thr_media if thr_media is not None else self.emb_threshold_media
            if n_input <= 2:
                ta += 0.20; tm += 0.20
            elif n_input <= 4:
                ta += 0.10; tm += 0.10

            banda = self._confidence_band(top_score, ta, tm)

            if banda == "alta" and not self._is_echo(top_linha, texto):
                if mostrar:
                    print(f"  [ALTA ({top_score:.3f}) → verbatim]")
                self._last_rag_info.update({"banda": "alta"})
                return top_linha

            if banda == "alta" and top_idx is not None and self._is_question(top_linha):
                if top_idx + 1 < len(self._rag_lines):
                    nxt = self._rag_lines[top_idx + 1][0]
                    if (not self._is_question(nxt)
                            and not self._is_echo(nxt, texto)
                            and len(nxt.split()) >= self.emb_min_tokens):
                        if mostrar:
                            print(f"  [Q&A chaining]")
                        self._last_rag_info.update({"banda": "alta_qa",
                                                    "top_idx": top_idx + 1})
                        return nxt

            if banda == "media":
                amostra = self._sample_from_topk(retrieved[:3], temperatura)
                if amostra:
                    l, s, i = amostra
                    if mostrar:
                        print(f"  [MÉDIA ({s:.3f}) → amostra top-3]")
                    self._last_rag_info.update({"banda": "media",
                                                "amostrada": True,
                                                "top_idx": i})
                    return l

            if mostrar:
                print(f"  [BAIXA ({top_score:.3f}) → fallback]")
            self._last_rag_info.update({"banda": "baixa"})
            return self._fallback_honesto(retrieved)

        # geração via MKN + Lastrimy
        passos_ger = passos
        if modo in ("rag_curto", "rag"):
            passos_ger = min(5, passos)
        elif modo == "rag_longo":
            passos_ger = passos

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
                "emb_thresholds": (self.emb_threshold_alta,
                                   self.emb_threshold_media),
            }, f)
        print(f"💾 Estado salvo em '{caminho}'")

    @classmethod
    def carregar_estado(cls, caminho="rlm_state.pkl", **kwargs):
        if not os.path.exists(caminho): return None
        with open(caminho, "rb") as f: d = pickle.load(f)
        g = cls(**kwargs)
        g.lastros = d["lastros"]; g.transicoes = d["transicoes"]
        g._unigram = d["unigram"]; g.limiar_ctx = d["limiar_ctx"]
        g._mkn_discounts = d.get("mkn_discounts", (0.5, 1.0, 1.5))
        g._rag_lines = d.get("rag_lines", [])

        if g._rag_lines and isinstance(g._rag_lines[0][1], dict):
            print("   [EMBED] índice antigo detectado, reconstruindo neural...")
            linhas_str = [l for l, _ in g._rag_lines]
            g._rag_lines = []
            for l in linhas_str:
                emb = g._embed_texto(l)
                if np.linalg.norm(emb) > 1e-9:
                    g._rag_lines.append((l, emb))

        th = d.get("emb_thresholds",
                   (g.emb_threshold_alta, g.emb_threshold_media))
        g.emb_threshold_alta, g.emb_threshold_media = th
        g._kv_init()
        print(f"⚡ Estado carregado de '{caminho}'.")
        if g._rag_lines:
            print(f"   [EMBED] {len(g._rag_lines)} frases neurais indexadas")
        return g


# ═══════════════════════════════════════════════════════════════════════
# PROMPT HANDLE
# ═══════════════════════════════════════════════════════════════════════
class PromptHandle:
    def __init__(self, rlm):
        self.rlm = rlm
        self._linhas = [l for l, _ in rlm._rag_lines]
        self._emb = np.full((len(self._linhas), rlm.EMB_DIM), np.nan, dtype=float)
        for i, (linha, emb) in enumerate(rlm._rag_lines):
            self._emb[i] = emb

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
        q = self.rlm._embed_texto(pergunta)
        qn = np.linalg.norm(q)
        if qn < 1e-9:
            return list(range(min(m, len(self._linhas))))
        q = q / qn
        valid = ~np.isnan(self._emb).any(axis=1)
        sims = np.full(len(self._linhas), -1.0, dtype=float)
        if valid.any():
            E = self._emb[valid]
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            sims[valid] = E @ q
        return np.argsort(-sims)[:m].tolist()


# ═══════════════════════════════════════════════════════════════════════
# MOTOR LLM RECURSIVO — v8.2 adaptativo
# ═══════════════════════════════════════════════════════════════════════
class MotorLLMRecursivo:
    """
    Motor adaptativo. Usa AdaptadorLinear para:
      - escolher modo (recursivo/construir/denso) por performance histórica
      - ajustar temperatura, thresholds, alpha, n_frags
      - registrar cada interação para evolução
    """

    _CONECTORES = [
        "mas", "e", "então", "por isso", "porque", "e também",
        "só que", "ainda assim", "ao mesmo tempo", "é que",
    ]

    def __init__(self, rlm, max_depth=2, verbose=False, max_linhas_denso=40,
                 adaptador=None):
        self.rlm = rlm
        self.max_depth = max_depth
        self.verbose = verbose
        self.max_linhas_denso = max_linhas_denso
        self.adaptador = adaptador
        self._trace = []
        self._calls = 0
        self._last_info = {}
        self._cached_handle = None
        self._cached_handle_len = -1
        self._ultimo_modo_usado = None
        self._ultima_resposta_foi_fallback = False

    def _perceber(self, pergunta, kwargs):
        self._calls += 1
        return self.rlm.perceber(pergunta, mostrar=False, **kwargs)

    def batch_subcall(self, perguntas, **kwargs):
        return [self._perceber(q, kwargs) for q in perguntas]

    def _get_handle(self):
        n_current = len(self.rlm._rag_lines)
        if self._cached_handle is None or self._cached_handle_len != n_current:
            self._cached_handle = PromptHandle(self.rlm)
            self._cached_handle_len = n_current
        return self._cached_handle

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

    def _extrair_fragmentos(self, retrieved, min_tok=4, max_tok=20):
        fragmentos = []
        for linha, score, idx in retrieved:
            partes = re.split(r'[.!?;]\s+', linha)
            for parte in partes:
                parte = parte.strip().strip(",;")
                if not parte:
                    continue
                n = len(_tokenize(parte))
                if min_tok <= n <= max_tok:
                    fragmentos.append((parte, float(score)))
        unicos = []
        for frag, score in fragmentos:
            tf = set(_tokenize(frag))
            if not tf:
                continue
            dup = False
            for existente, _ in unicos:
                te = set(_tokenize(existente))
                if not te:
                    continue
                jacc = len(tf & te) / len(tf | te)
                if jacc > 0.75:
                    dup = True
                    break
            if not dup:
                unicos.append((frag, score))
        return unicos

    def _ordenar_fragmentos(self, fragmentos):
        if len(fragmentos) <= 1:
            return fragmentos

        fragmentos = sorted(fragmentos, key=lambda x: -x[1])
        ordenados = [fragmentos[0]]
        resto = list(fragmentos[1:])

        while resto:
            ult = ordenados[-1][0]
            ult_toks = _tokenize(ult)
            if not ult_toks:
                break
            ctx = tuple(
                self.rlm._hash(t) for t in ult_toks[-2:]
                if self.rlm._hash(t) in self.rlm.lastros)
            if not ctx:
                ordenados.append(resto.pop(0))
                continue

            dist = self.rlm._kn_rec(ctx)
            melhor_i, melhor_score = -1, -1.0
            for i, (cand, sc) in enumerate(resto):
                toks_c = _tokenize(cand)
                if not toks_c:
                    continue
                alvo = self.rlm._hash(toks_c[0])
                p_trans = dist.get(alvo, 0.0)
                score = p_trans * 0.7 + sc * 0.3
                if score > melhor_score:
                    melhor_score = score
                    melhor_i = i

            if melhor_i < 0:
                break
            ordenados.append(resto.pop(melhor_i))

        return ordenados

    def _conectar_fragmentos(self, ordenados):
        if not ordenados:
            return ""
        if len(ordenados) == 1:
            texto = ordenados[0][0].strip()
            if texto and texto[0].islower():
                texto = texto[0].upper() + texto[1:]
            if texto and texto[-1] not in ".!?":
                texto += "."
            return texto

        texto = ordenados[0][0].strip()
        if texto and texto[0].islower():
            texto = texto[0].upper() + texto[1:]

        for i in range(1, len(ordenados)):
            frag = ordenados[i][0].strip()
            if not frag:
                continue

            ult_toks = _tokenize(texto)
            prim_toks = _tokenize(frag)
            if not ult_toks or not prim_toks:
                continue

            ctx = tuple(
                self.rlm._hash(t) for t in ult_toks[-2:]
                if self.rlm._hash(t) in self.rlm.lastros)
            p_trans = 0.0
            if ctx:
                dist = self.rlm._kn_rec(ctx)
                alvo = self.rlm._hash(prim_toks[0])
                p_trans = dist.get(alvo, 0.0)

            if frag and frag[0].isupper():
                frag = frag[0].lower() + frag[1:]

            if p_trans > 0.05:
                texto += ", " + frag
            else:
                conector = random.choice(self._CONECTORES)
                texto += f". {conector.capitalize()} {frag}"

        if texto and texto[-1] not in ".?!":
            texto += "."

        return texto

    def construir_paragrafo(self, pergunta, kwargs, n_frags=None):
        if n_frags is None:
            n_frags = (self.adaptador.n_frags_sugerido
                       if self.adaptador else 3)

        self._trace = []
        self._calls = 0
        self._last_info = {"modo": "construir", "n_frags_alvo": n_frags}

        temperatura = (self.adaptador.temperatura_sugerida
                       if self.adaptador else kwargs.get("temperatura", 0.5))
        alpha = (self.adaptador.alpha_sugerido
                 if self.adaptador else kwargs.get("alpha", 0.5))

        retrieved = self.rlm._retrieve_por_embedding(pergunta, top_k=8)
        if len(retrieved) < 2:
            if self.verbose:
                print("  [construtor] poucos resultados → fallback para base")
            return self._perceber(pergunta, kwargs)

        if self.verbose:
            print(f"  [construtor] {len(retrieved)} linhas recuperadas")

        frags = self._extrair_fragmentos(retrieved)
        if self.verbose:
            print(f"  [construtor] {len(frags)} fragmentos extraídos")

        if len(frags) < 2:
            if self.verbose:
                print("  [construtor] poucos fragmentos → fallback para base")
            return self._perceber(pergunta, kwargs)

        frags = frags[:n_frags + 2]
        ordenados = self._ordenar_fragmentos(frags)
        if self.verbose:
            print(f"  [construtor] ordem final:")
            for f, _ in ordenados:
                print(f"    · {f[:60]}")

        texto = self._conectar_fragmentos(ordenados)

        toks = _tokenize(texto)
        if len(toks) > 60:
            partes = re.split(r'(?<=[.!?])\s+', texto)
            acumulado = ""
            for p in partes:
                if len(_tokenize(acumulado + " " + p)) > 60:
                    break
                acumulado = (acumulado + " " + p).strip()
            texto = acumulado or texto

        self._trace.append(("construir", pergunta, texto))
        if self.verbose:
            print(f"  [construtor] parágrafo final ({len(_tokenize(texto))} tokens)")
        self._ultimo_modo_usado = "construir"
        self._ultima_resposta_foi_fallback = False
        return texto

    def _raciocinar_denso(self, pergunta, kwargs):
        handle = self._get_handle()
        n = len(handle)
        if n == 0:
            return self._perceber(pergunta, kwargs)

        if n > 60:
            retrieved = self.rlm._retrieve_por_embedding(
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
                consulta, modo="rag", mostrar=False, temperatura=0.0,
                cache_memoria=kwargs.get("cache_memoria"),
                alpha=kwargs.get("alpha", 0.5),
                janela_ctx=kwargs.get("janela_ctx", 12),
            )
            info = getattr(self.rlm, "_last_rag_info", {}) or {}
            if info.get("banda") in ("alta", "media", "alta_qa") and r:
                relevantes.append((i, linha, r))

        if self.verbose:
            print(f"  [denso] {len(relevantes)} linhas com confiança ≥ média")

        if not relevantes:
            amostra = "\n".join(handle.peek(i) for i in indices[:8])
            return self._perceber(f"{pergunta}\n\nContexto próximo:\n{amostra}", kwargs)

        self._trace.append(("denso", pergunta, relevantes[0][2]))
        self._ultimo_modo_usado = "denso"
        self._ultima_resposta_foi_fallback = False
        return relevantes[0][2]

    def raciocinar(self, pergunta, modo_forcado=None, **kwargs):
        """
        modo_forcado: None → usa sugestão do adaptador
                      "recursivo"/"construir"/"denso" → força
        """
        self._trace = []
        self._calls = 0
        self._last_info = {"depth": self.max_depth}

        # escolhe modo
        if modo_forcado is None and self.adaptador is not None:
            modo_escolhido = self.adaptador.sugerir_modo()
        else:
            modo_escolhido = modo_forcado or "recursivo"

        if self.verbose:
            print(f"  [motor] modo escolhido: {modo_escolhido}")

        # denso automático se detector disparar
        if self._detectar_densa(pergunta) and modo_escolhido != "construir":
            return self._raciocinar_denso(pergunta, kwargs)

        if modo_escolhido == "construir":
            return self.construir_paragrafo(pergunta, kwargs)

        # recursivo padrão
        self._ultimo_modo_usado = "recursivo"
        self._ultima_resposta_foi_fallback = False

        frase = self._perceber(pergunta, kwargs)
        if not frase:
            return ""
        info = getattr(self.rlm, "_last_rag_info", {}) or {}
        self._ultima_resposta_foi_fallback = (info.get("banda") == "baixa")

        frases = [frase]
        self._trace.append(("base", pergunta, frase))
        if self.verbose:
            print(f"  [motor] base: {frase[:80]}")

        for i in range(self.max_depth):
            stripped = frase.rstrip()
            if stripped and stripped[-1] in ".?!":
                if self.verbose:
                    print(f"  [motor] encadeia {i}: base completa, para")
                break
            if len(_tokenize(frase)) >= 15:
                if self.verbose:
                    print(f"  [motor] encadeia {i}: base longa, para")
                break

            nova = self._perceber(frase, kwargs)
            if not nova or nova == frase or nova in frases:
                if self.verbose:
                    print(f"  [motor] encadeia {i}: sem variação, para")
                break
            frases.append(nova)
            self._trace.append((f"encadeia {i}", frase, nova))
            if self.verbose:
                print(f"  [motor] encadeia {i}: {nova[:80]}")
            frase = nova

        return frase

    def registrar_reward(self, reward):
        """Chamado pelo MAIN após calcular reward."""
        if self.adaptador is not None:
            self.adaptador.registrar(
                reward,
                self._ultimo_modo_usado or "recursivo",
                self._ultima_resposta_foi_fallback,
            )


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE DIFUSÃO ESTOCÁSTICA
# ═══════════════════════════════════════════════════════════════════════
class MotorDifusaoEstocastica:
    def __init__(self, rlm, motor, buffer_size=50, taxa_aprendizado=0.05,
                 min_ocorrencias_sintese=3, limiar_reforco_sintese=0.3):
        self.rlm = rlm
        self.motor = motor
        self.buffer = []
        self.buffer_size = buffer_size
        self.taxa = taxa_aprendizado
        self.min_ocorrencias = min_ocorrencias_sintese
        self.limiar_reforco = limiar_reforco_sintese
        self.padroes = {}
        self.pesos_caminho = {}
        self.diffs_aprendidos = {}
        self.linhas_sintetizadas = set()
        self.total_interacoes = 0
        self.total_positivas = 0

    def _sim(self, a, b):
        ta, tb = set(_tokenize(a)), set(_tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _inferir_reward(self, input_user, output_robo, continuou,
                        feedback, proxima_entrada):
        if feedback is not None:
            return float(feedback)
        r = 0.0
        if continuou:
            r += 0.3
        r += self._sim(input_user, output_robo) * 0.3
        if proxima_entrada:
            low = proxima_entrada.lower()
            if any(p in low for p in
                   ["não", "errado", "nada a ver", "que isso",
                    "n entendi", "não entendi", "estranho", "???"]):
                r -= 0.6
            if any(p in low for p in
                   ["obrigado", "valeu", "legal", "isso", "isso mesmo",
                    "gostei", "boa", "show", "perfeito", "exato"]):
                r += 0.6
            sim_prox = self._sim(proxima_entrada, output_robo)
            if sim_prox > 0.3:
                r += sim_prox * 0.4
        return r

    def _extrair_padrao(self, texto):
        toks = _tokenize(texto)
        if len(toks) <= 2:
            return " ".join(toks)
        return " ".join(toks[:2]) + f"__{len(toks)}"

    def _amostrar_batch(self, n=8):
        if not self.buffer:
            return []
        exps = self.buffer
        pesos = []
        for i, e in enumerate(exps):
            recencia = (i + 1) / len(exps)
            pesos.append((abs(e["reward"]) + 0.1) * recencia)
        total = sum(pesos)
        if total <= 0:
            return random.sample(exps, min(n, len(exps)))
        pesos = [p / total for p in pesos]
        try:
            indices = np.random.choice(
                len(exps), size=min(n, len(exps)),
                replace=False, p=pesos)
        except ValueError:
            return random.sample(exps, min(n, len(exps)))
        return [exps[i] for i in indices]

    def observar(self, input_user, output_robo, continuou=True,
                 feedback=None, proxima_entrada=None):
        reward = self._inferir_reward(
            input_user, output_robo, continuou, feedback, proxima_entrada)
        self.buffer.append({
            "input": input_user, "output": output_robo, "reward": reward,
        })
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        self.total_interacoes += 1
        if reward > 0:
            self.total_positivas += 1
        padrao = self._extrair_padrao(input_user)
        if padrao not in self.padroes:
            self.padroes[padrao] = {"contagem": 0, "reforco": 0.0,
                                     "respostas": {}}
        p = self.padroes[padrao]
        p["contagem"] += 1
        p["reforco"] = p["reforco"] * 0.85 + reward * 0.15
        p["respostas"][output_robo] = p["respostas"].get(output_robo, 0) + 1
        return reward

    def _aplicar_delta(self, ctx, alvo, delta):
        if ctx not in self.rlm.lastros or alvo not in self.rlm.lastros:
            return False
        la = self.rlm.lastros[ctx]
        if la.transicoes_aum is None:
            la.transicoes_aum = Counter()
        atual = la.transicoes_aum.get(alvo, 0.0)
        la.transicoes_aum[alvo] = max(0.01, atual + delta)
        chave = (ctx, alvo)
        self.pesos_caminho[chave] = self.pesos_caminho.get(chave, 0.0) + delta
        self.diffs_aprendidos[chave] = \
            self.diffs_aprendidos.get(chave, 0.0) + delta
        return True

    def difundir(self):
        if not self.buffer:
            return 0
        exps = self._amostrar_batch(n=min(8, len(self.buffer)))
        n = 0
        for exp in exps:
            reward = exp["reward"]
            if abs(reward) < 1e-3:
                continue
            input_toks = list(self.rlm._acende(exp["input"], criar=False))
            output_toks = list(self.rlm._acende(exp["output"], criar=False))
            if not input_toks or not output_toks:
                continue
            ctx = input_toks[-1]
            alvo = output_toks[0]
            if ctx not in self.rlm.lastros or alvo not in self.rlm.lastros:
                continue
            delta = self.taxa * reward + np.random.normal(0, 0.01)
            if self._aplicar_delta(ctx, alvo, delta):
                n += 1
            la = self.rlm.lastros[ctx]
            if la.vizinhos:
                k = min(3, len(la.vizinhos))
                pesos_viz = np.array([v[3] for v in la.vizinhos[:k]])
                if pesos_viz.sum() > 0:
                    pesos_viz = pesos_viz / pesos_viz.sum()
                    idx = int(np.random.choice(k, p=pesos_viz))
                    viz_id, sf, jacc, score = la.vizinhos[idx]
                    delta_viz = (self.taxa * reward * score * 0.3
                                 + np.random.normal(0, 0.005))
                    if self._aplicar_delta(viz_id, alvo, delta_viz):
                        n += 1
        return n

    def _adicionar_ao_corpus(self, linha):
        if linha in self.linhas_sintetizadas:
            return False
        if any(l == linha for l, _ in self.rlm._rag_lines):
            return False
        toks = _tokenize(linha)
        if len(toks) < 2:
            return False

        self.rlm._acende(linha, frase_id=-1, criar=True)
        for t in toks:
            rid = self.rlm._hash(t)
            la = self.rlm.lastros.get(rid)
            if la is not None and la.perfil is None:
                la.perfil = self.rlm._derivar_perfil(la)
                la.pesos_head = _softmax(la.perfil, T=self.rlm.tau_perfil)

        emb = self.rlm._embed_texto(linha)
        if np.linalg.norm(emb) < 1e-9:
            return False

        self.rlm._rag_lines.append((linha, emb))
        self.linhas_sintetizadas.add(linha)
        if self.motor:
            self.motor._cached_handle = None
        return True

    def aprender(self):
        n_paths = self.difundir()
        novas = []
        for padrao, p in list(self.padroes.items()):
            if (p["contagem"] >= self.min_ocorrencias
                    and p["reforco"] >= self.limiar_reforco
                    and p["respostas"]):
                melhor = max(p["respostas"].items(), key=lambda x: x[1])[0]
                if melhor and self._adicionar_ao_corpus(melhor):
                    novas.append(melhor)
                    p["contagem"] = 0
                    p["reforco"] = 0.0
        if len(self.buffer) > 10:
            self.buffer = self.buffer[-10:]
        return n_paths, novas

    def reaplicar_apos_treino(self):
        n = 0
        for (ctx, alvo), delta in self.diffs_aprendidos.items():
            if self._aplicar_delta(ctx, alvo, delta * 0.5):
                n += 1
        return n

    def relatorio(self):
        print("\n" + "═" * 60)
        print("  RELATÓRIO DE APRENDIZADO (v8.2)")
        print("═" * 60)
        taxa = self.total_positivas / max(self.total_interacoes, 1)
        print(f"  Interações: {self.total_interacoes} "
              f"({self.total_positivas} positivas = {taxa*100:.0f}%)")
        print(f"  Padrões: {len(self.padroes)} | "
              f"Pesos: {len(self.pesos_caminho)} | "
              f"Linhas sintetizadas: {len(self.linhas_sintetizadas)}")
        print(f"  Buffer: {len(self.buffer)}/{self.buffer_size}")
        print(f"  Índice neural: {len(self.rlm._rag_lines)} frases")
        top = sorted(self.padroes.items(),
                     key=lambda x: x[1]["reforco"] * x[1]["contagem"],
                     reverse=True)[:5]
        if top:
            print("\n  Top 5 padrões aprendidos:")
            for padrao, p in top:
                melhor = max(p["respostas"].items(),
                             key=lambda x: x[1])[0] if p["respostas"] else "—"
                print(f"    '{padrao}' ({p['contagem']}x, "
                      f"ref={p['reforco']:+.2f})")
                print(f"       → {melhor[:55]}")
        print("═" * 60 + "\n")

    def salvar(self, caminho="rlm_difusao.pkl"):
        with open(caminho, "wb") as f:
            pickle.dump({
                "padroes": self.padroes,
                "pesos_caminho": self.pesos_caminho,
                "diffs_aprendidos": self.diffs_aprendidos,
                "linhas_sintetizadas": list(self.linhas_sintetizadas),
                "total_interacoes": self.total_interacoes,
                "total_positivas": self.total_positivas,
            }, f)
        print(f"💾 Difusão salva em '{caminho}'")

    def carregar(self, caminho="rlm_difusao.pkl"):
        if not os.path.exists(caminho):
            return False
        with open(caminho, "rb") as f:
            d = pickle.load(f)
        self.padroes = d.get("padroes", {})
        self.pesos_caminho = d.get("pesos_caminho", {})
        self.diffs_aprendidos = d.get("diffs_aprendidos", {})
        self.linhas_sintetizadas = set(d.get("linhas_sintetizadas", []))
        self.total_interacoes = d.get("total_interacoes", 0)
        self.total_positivas = d.get("total_positivas", 0)
        n = self.reaplicar_apos_treino()
        print(f"⚡ Difusão carregada: {len(self.padroes)} padrões, "
              f"{len(self.pesos_caminho)} pesos, {n} reaplicados")
        return True


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE AUTO-EVOLUÇÃO LIFELONG
# ═══════════════════════════════════════════════════════════════════════
class MotorAutoEvolucao:
    def __init__(self, rlm, motor, difusao):
        self.rlm = rlm
        self.motor = motor
        self.difusao = difusao
        self.lacunas_atuais = []
        self.pergunta_pendente = None
        self.historico_evolucao = []
        self.turno_atual = 0
        self.melhorias_historicas = []

    def _detectar_lacunas(self):
        lacunas = []

        for rid, la in self.rlm.lastros.items():
            if la.token.startswith("⟨"):
                continue
            n = len(la.dados_ids)
            if 1 <= n <= 2:
                lacunas.append(("token_raro", la.token, 1.0 - n / 3.0))

        for linha, emb in self.rlm._rag_lines:
            if not isinstance(emb, np.ndarray):
                continue
            norma = float(np.linalg.norm(emb))
            if 0.0 < norma < 0.3:
                lacunas.append(("embedding_fraco", linha, 1.0 - norma / 0.3))

        for ctx, counter in self.rlm.transicoes.items():
            total = sum(counter.values())
            if len(counter) == 1 and total == 1:
                ctx_tokens = [
                    self.rlm.lastros[r].token
                    for r in ctx if r in self.rlm.lastros
                ]
                if ctx_tokens and not any(t.startswith("⟨") for t in ctx_tokens):
                    lacunas.append(
                        ("transicao_fraca", " ".join(ctx_tokens), 0.8))

        for exp in self.difusao.buffer:
            if exp["reward"] < -0.2:
                lacunas.append(
                    ("resposta_ruim", exp["input"], min(1.0, -exp["reward"])))

        for padrao, dados in self.difusao.padroes.items():
            if (dados["reforco"] > 0.3 and dados["contagem"] >= 2
                    and dados["respostas"]):
                melhor = max(dados["respostas"].items(),
                             key=lambda x: x[1])[0]
                if melhor not in self.difusao.linhas_sintetizadas:
                    lacunas.append(("padrao_orfao", padrao, 0.6))

        return lacunas

    def _pergunta_para_lacuna(self, lacuna):
        tipo, alvo, _ = lacuna
        if tipo == "token_raro":
            return f"o que significa '{alvo}' pra você?"
        if tipo == "embedding_fraco":
            return f"me conta mais sobre: {alvo[:70]}"
        if tipo == "transicao_fraca":
            return f"como você conecta '{alvo}' com outras ideias?"
        if tipo == "resposta_ruim":
            return (f"quando eu disse '{alvo[:40]}', "
                    f"o que você realmente queria ouvir?")
        if tipo == "padrao_orfao":
            return (f"sobre '{alvo}', tem algo que você gostaria "
                    f"que eu soubesse dizer melhor?")
        return None

    def explorar(self, n_perguntas=3):
        self.turno_atual += 1
        self.lacunas_atuais = self._detectar_lacunas()
        if not self.lacunas_atuais:
            return []

        self.lacunas_atuais.sort(key=lambda x: -x[2])
        perguntas = []
        tipos_usados = set()

        for lacuna in self.lacunas_atuais:
            tipo = lacuna[0]
            if tipo in tipos_usados:
                continue
            texto = self._pergunta_para_lacuna(lacuna)
            if texto:
                perguntas.append({
                    "pergunta": texto,
                    "lacuna": lacuna,
                    "tipo": tipo,
                })
                tipos_usados.add(tipo)
            if len(perguntas) >= n_perguntas:
                break

        if perguntas:
            self.pergunta_pendente = perguntas[0]
        return perguntas

    def processar_resposta(self, resposta_usuario):
        if not self.pergunta_pendente:
            return None

        lacuna = self.pergunta_pendente["lacuna"]
        tipo, alvo, _ = lacuna
        pergunta = self.pergunta_pendente["pergunta"]

        antes = len(self.rlm._rag_lines)

        self.rlm._acende(resposta_usuario, frase_id=-1, criar=True)
        for t in _tokenize(resposta_usuario):
            rid = self.rlm._hash(t)
            la = self.rlm.lastros.get(rid)
            if la is not None and la.perfil is None:
                la.perfil = self.rlm._derivar_perfil(la)
                la.pesos_head = _softmax(la.perfil, T=self.rlm.tau_perfil)

        emb = self.rlm._embed_texto(resposta_usuario)
        if np.linalg.norm(emb) > 1e-9:
            combinado = f"{pergunta} {resposta_usuario}"
            emb_comb = self.rlm._embed_texto(combinado)
            if np.linalg.norm(emb_comb) < np.linalg.norm(emb):
                emb_comb = emb
            self.rlm._rag_lines.append((resposta_usuario, emb_comb))

        self.difusao.observar(
            input_user=pergunta,
            output_robo=resposta_usuario,
            continuou=True,
            proxima_entrada=None,
        )

        depois = len(self.rlm._rag_lines)
        ganho = depois - antes
        self.historico_evolucao.append({
            "turno": self.turno_atual,
            "tipo": tipo,
            "alvo": alvo[:40],
            "ganho_linhas": ganho,
        })

        self.melhorias_historicas.append(lacuna)
        self.pergunta_pendente = None

        return {
            "tipo": tipo,
            "alvo": alvo,
            "linhas_novas": ganho,
        }

    def refletir(self):
        por_tipo = Counter()
        for lac in self.lacunas_atuais:
            por_tipo[lac[0]] += 1

        melh_por_tipo = Counter()
        for mel in self.melhorias_historicas:
            melh_por_tipo[mel[0]] += 1

        return {
            "turno_atual": self.turno_atual,
            "interacoes": self.difusao.total_interacoes,
            "lacunas_abertas": len(self.lacunas_atuais),
            "lacunas_por_tipo": dict(por_tipo),
            "melhorias_resolvidas": len(self.melhorias_historicas),
            "melhorias_por_tipo": dict(melh_por_tipo),
            "linhas_aprendidas": len(self.difusao.linhas_sintetizadas),
            "indice_atual": len(self.rlm._rag_lines),
        }

    def auto_curriculo(self, n=5):
        if not self.lacunas_atuais:
            self.lacunas_atuais = self._detectar_lacunas()

        pesos_tipo = {
            "resposta_ruim": 1.5,
            "transicao_fraca": 1.2,
            "token_raro": 1.0,
            "embedding_fraco": 1.0,
            "padrao_orfao": 0.8,
        }

        ranqueadas = []
        for lac in self.lacunas_atuais:
            tipo, alvo, sev = lac
            peso = pesos_tipo.get(tipo, 1.0)
            ranqueadas.append((sev * peso, lac))

        ranqueadas.sort(reverse=True)
        return [lac for _, lac in ranqueadas[:n]]


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ARQUIVO_CORPUS = "corpus.txt"
    ARQUIVO_ESTADO = "rlm_state.pkl"
    ARQUIVO_DIFUSAO = "rlm_difusao.pkl"
    ARQUIVO_ADAPTADOR = "rlm_adaptador.pkl"

    print("=" * 70)
    print("RLM-Lastrimy v8.2 — Auto-Evolução Lifelong Adaptativa")
    print("=" * 70)

    kwargs = dict(
        n_max=4, D=0.75, tau_pos=2.0, k_vizinhos=5,
        limiar_forma=0.70, limiar_ctx=0.15, peso_vizinho=0.5,
        anti_loop_sintatico_forca=0.2, tau_perfil=1.0,
        auto_ajuste_limiar=True, usar_rede_neural=True,
        gns_max_vocab=800, gns_hidden=24, gns_epocas=400, gns_lr=0.05,
        gns_pos_enc=False, gns_adamw=True,
        gns_adamw_lr=0.01, gns_adamw_wd=0.001,
        emb_threshold_alta=0.55, emb_threshold_media=0.30,
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

    # Adaptador linear
    adaptador = AdaptadorLinear()
    adaptador.carregar(ARQUIVO_ADAPTADOR)

    motor = MotorLLMRecursivo(rlm, max_depth=2, verbose=True,
                              max_linhas_denso=40, adaptador=adaptador)
    difusao = MotorDifusaoEstocastica(rlm, motor, taxa_aprendizado=0.05)
    difusao.carregar(ARQUIVO_DIFUSAO)
    auto_evol = MotorAutoEvolucao(rlm, motor, difusao)
    rlm._auto_evol = auto_evol

    MODOS = ("auto", "recursivo", "construir", "denso",
             "rag", "rag_puro", "rag_curto", "rag_longo",
             "responder", "continuar")

    print("\nModos (v8.2):")
    print("  auto (PADRÃO — adapta) | recursivo | construir | denso | rag")
    print("Comandos: sair | T=0.3 | a=0.7 | limpar | m=rag")
    print("          relatorio | aprender | explorar | refletir | curriculo")
    print("          adaptacao | estado")
    print("-" * 70)

    temperatura = 0.5
    alpha = 0.5
    modo = "auto"          # ← PADRÃO MUDADO
    janela_ctx = 12
    cache_memoria = set()
    seeds_usados = set()
    ultimo_par = None

    while True:
        try:
            entrada = input("\nvocê > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not entrada: continue
        low = entrada.lower()

        # ─── comandos ───
        if low in ("sair", "exit", "quit"):
            difusao.salvar(ARQUIVO_DIFUSAO)
            adaptador.salvar(ARQUIVO_ADAPTADOR)
            rlm.salvar_estado(ARQUIVO_ESTADO)
            break
        if low == "limpar":
            cache_memoria.clear(); seeds_usados.clear()
            ultimo_par = None
            print("[Memória limpa]"); continue
        if low == "relatorio":
            difusao.relatorio(); continue
        if low == "aprender":
            n_paths, novas = difusao.aprender()
            print(f"[aprender] {n_paths} caminhos, {len(novas)} linhas novas")
            for l in novas:
                print(f"  → '{l[:60]}'")
            continue
        if low == "explorar":
            perguntas = auto_evol.explorar(n_perguntas=3)
            if not perguntas:
                print("\n[explorar] nenhuma lacuna — modelo saturado")
            else:
                print(f"\n[explorar] {len(perguntas)} pergunta(s):")
                for i, p in enumerate(perguntas, 1):
                    print(f"  {i}. [{p['tipo']}] {p['pergunta']}")
                print(f"\n(responda a 1ª na próxima linha)")
            continue
        if low == "refletir":
            rel = auto_evol.refletir()
            print("\n" + "═" * 60)
            print("  REFLEXÃO DO ROBÔ")
            print("═" * 60)
            print(f"  Turno: {rel['turno_atual']} | "
                  f"Interações: {rel['interacoes']}")
            print(f"  Lacunas abertas: {rel['lacunas_abertas']}")
            for tipo, n in rel["lacunas_por_tipo"].items():
                print(f"    · {tipo}: {n}")
            print(f"  Melhorias resolvidas: {rel['melhorias_resolvidas']}")
            for tipo, n in rel["melhorias_por_tipo"].items():
                print(f"    · {tipo}: {n}")
            print(f"  Linhas aprendidas: {rel['linhas_aprendidas']}")
            print(f"  Índice neural: {rel['indice_atual']} frases")
            print("═" * 60 + "\n")
            continue
        if low == "curriculo":
            plano = auto_evol.auto_curriculo(n=8)
            if not plano:
                print("[curriculo] nada a priorizar")
            else:
                print(f"\n[curriculo] top {len(plano)} lacunas:")
                for i, (tipo, alvo, sev) in enumerate(plano, 1):
                    print(f"  {i}. [{tipo}] sev={sev:.2f} · {alvo[:60]}")
            continue
        if low == "adaptacao":
            est = adaptador.estado_atual()
            print("\n" + "═" * 60)
            print("  ESTADO DO ADAPTADOR LINEAR")
            print("═" * 60)
            print(f"  Turnos: {est['turnos']} | "
                  f"Média reward: {est['media_reward']:+.3f}")
            print(f"  Confiança: {est['confianca']:.3f} | "
                  f"Sucesso: {est['taxa_sucesso']:.2f} | "
                  f"Fallback: {est['taxa_fallback']:.2f}")
            print(f"  Temperatura: {est['temperatura']:.3f} | "
                  f"Alpha: {est['alpha']:.3f}")
            print(f"  Threshold alta: {est['thr_alta']:.3f} | "
                  f"media: {est['thr_media']:.3f}")
            print(f"  n_frags (construir): {est['n_frags']}")
            print(f"  Modo padrão: {est['modo_padrao']}")
            print("\n  Performance por modo:")
            for m, p in est["performance_modo"].items():
                print(f"    {m}: {p['usos']} usos, "
                      f"reward médio {p['media_reward']:+.3f}")
            if est["historico"]:
                print("\n  Últimos snapshots (adaptação ao longo do tempo):")
                for h in est["historico"][-5:]:
                    print(f"    t={h['turno']:3d} | "
                          f"conf={h['confianca']:.2f} | "
                          f"T={h['temperatura']:.2f} | "
                          f"α={h['alpha']:.2f} | "
                          f"modo={h['modo']}")
            print("═" * 60 + "\n")
            continue
        if low == "estado":
            print(f"\n[estado] modo={modo} | T={temperatura} | α={alpha}")
            print(f"         confiança adaptador={adaptador.confianca:.2f}")
            print(f"         modo padrão={adaptador.modo_padrao}")
            print(f"         índice={len(rlm._rag_lines)} frases | "
                  f"perguntas pendentes={auto_evol.pergunta_pendente is not None}")
            continue

        if low.startswith("t="):
            temperatura = float(entrada.split("=", 1)[1])
            print(f"[T={temperatura}]"); continue
        if low.startswith("a="):
            alpha = float(entrada.split("=", 1)[1])
            print(f"[α={alpha}]"); continue
        if low.startswith("j="):
            janela_ctx = int(entrada.split("=", 1)[1])
            print(f"[janela={janela_ctx}]"); continue

        mode_str = None
        if low in MODOS: mode_str = low
        elif low.startswith("m=") or low.startswith("m "): mode_str = low[2:].strip()
        elif low.startswith("modo=") or low.startswith("modo "): mode_str = low[5:].strip()
        if mode_str is not None:
            if mode_str in MODOS:
                modo = mode_str; print(f"[modo={modo}]")
            else:
                print(f"[modo inválido — use: {', '.join(MODOS)}]")
            continue

        print("-" * 70)

        # ─── se pergunta exploratória pendente, absorve resposta ───
        if auto_evol.pergunta_pendente is not None:
            resultado = auto_evol.processar_resposta(entrada)
            if resultado:
                print(f"  [evolução] absorvi sobre '{resultado['tipo']}': "
                      f"+{resultado['linhas_novas']} linha(s)")
                print(f"\nRLM  > Entendi. Aprendi sobre "
                      f"'{resultado['alvo'][:40]}'.")
                ultimo_par = (auto_evol.pergunta_pendente["pergunta"]
                              if auto_evol.pergunta_pendente else entrada,
                              entrada)
                continue

        # ─── pontua interação anterior ───
        if ultimo_par is not None:
            pergunta_ant, resposta_ant = ultimo_par
            reward = difusao.observar(
                pergunta_ant, resposta_ant,
                continuou=True, proxima_entrada=entrada)
            motor.registrar_reward(reward)
            if abs(reward) > 0.3:
                sinal = "✓" if reward > 0 else "✗"
                print(f"  [aprendizado] {sinal} anterior reward={reward:+.2f}")

        # ─── gera resposta ───
        if modo == "auto":
            resposta = motor.raciocinar(
                entrada,
                cache_memoria=cache_memoria,
                temperatura=adaptador.temperatura_sugerida,
                alpha=adaptador.alpha_sugerido,
                janela_ctx=janela_ctx,
                ultimo_par=ultimo_par,
                thr_alta=adaptador.threshold_alta,
                thr_media=adaptador.threshold_media,
            )
            print(f"  [auto] modo={motor._ultimo_modo_usado} | "
                  f"T={adaptador.temperatura_sugerida:.2f} | "
                  f"conf={adaptador.confianca:.2f}")

        elif modo == "construir":
            resposta = motor.construir_paragrafo(
                entrada,
                {"cache_memoria": cache_memoria,
                 "temperatura": temperatura,
                 "alpha": alpha,
                 "janela_ctx": janela_ctx,
                 "ultimo_par": ultimo_par})
            print(f"  [construtor] {len(_tokenize(resposta))} tokens")

        elif modo == "denso":
            resposta = motor._raciocinar_denso(
                entrada,
                {"cache_memoria": cache_memoria, "temperatura": temperatura,
                 "alpha": alpha, "janela_ctx": janela_ctx,
                 "ultimo_par": ultimo_par})
            print(f"  [denso] {motor._calls} chamadas")

        elif modo in ("recursivo",):
            resposta = motor.raciocinar(
                entrada,
                modo_forcado="recursivo",
                cache_memoria=cache_memoria,
                temperatura=temperatura,
                alpha=alpha,
                janela_ctx=janela_ctx,
                ultimo_par=ultimo_par)
            if motor._trace:
                print(f"  [motor] {motor._calls} chamadas | "
                      f"{len(motor._trace)} passos")

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
                ultimo_par=ultimo_par,
                thr_alta=adaptador.threshold_alta,
                thr_media=adaptador.threshold_media,
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
            ultimo_par = (entrada, resposta)

        # ─── difusão periódica ───
        if (difusao.total_interacoes > 0
                and difusao.total_interacoes % 3 == 0):
            n_paths, novas = difusao.aprender()
            if n_paths > 0 or novas:
                print(f"  [difusão] {n_paths} caminhos, "
                      f"{len(novas)} linha(s) sintetizada(s)")
                for l in novas:
                    print(f"    ✨ aprendi: '{l[:60]}'")

        # ─── adaptação automática a cada 5 turnos ───
        if adaptador.turnos > 0 and adaptador.turnos % 5 == 0:
            print(f"  [adaptação] conf={adaptador.confianca:.2f} | "
                  f"T={adaptador.temperatura_sugerida:.2f} | "
                  f"modo padrão={adaptador.modo_padrao}")

        print("-" * 70)
