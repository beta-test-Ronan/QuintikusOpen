"""
GNS-Linhagem
============
Hash = endereço = neurônio (lastro_id).
Nível 1: Counter + KN absoluto interpolado.
Nível 2: embeddings por hash, atenção temporal² (peso 1/(1+d)⁴).
Integração: prob-mix | rerank, com neural_T separado, anti-loop, clip de E.

Correções vs versão anterior:
  - lr=0.02 (era 0.05)         → menos decoração
  - E_clip=1.5                 → mata hub embedding
  - neural_T separado          → achata neural sem mexer no sampling
  - anti-loop por token        → quebra "a a a" mesmo em T=0
  - temperature no fim         → achatamento da mistura, não do logit
"""

import hashlib
import pickle
import random
from collections import Counter, deque
import numpy as np

_SENT = "\uE000"


class GNSLinhagem:
    def __init__(self, dim=32, n_max=4, D=0.75, lr=0.02,
                 alpha_neural=0.3, modo="mix", E_clip=1.5):
        self.dim = dim
        self.n_max = n_max
        self.D = D
        self.lr = lr
        self.alpha_neural = alpha_neural
        self.modo = modo
        self.E_clip = E_clip

        self.BOS = self._hash((_SENT, "BOS", 0))
        self.EOS = self._hash((_SENT, "EOS", -1))
        self.ramos = {
            self.BOS: {"token": "⟨BOS⟩", "pos": 0, "visitas": 0},
            self.EOS: {"token": "⟨EOS⟩", "pos": -1, "visitas": 0},
        }
        self.transicoes = {}
        self._unigram = {}

        self.E = {}
        self._init_embedding(self.BOS)
        self._init_embedding(self.EOS)

        self.loss_medio = 0.0
        self.n_passos = 0

    def _hash(self, x):
        return int(hashlib.sha256(str(x).encode()).hexdigest()[:16], 16)

    def _init_embedding(self, endereco):
        if endereco not in self.E:
            rng = np.random.default_rng(endereco % (2**32))
            self.E[endereco] = rng.standard_normal(self.dim) * 0.1
        return self.E[endereco]

    def _acende(self, texto, criar=True):
        tokens = [t for t in texto.lower().split() if _SENT not in t]
        ramos = []
        for i, tok in enumerate(tokens):
            ramo = self._hash((i, tok))
            ramos.append(ramo)
            if criar:
                self.ramos.setdefault(ramo, {"token": tok, "pos": i, "visitas": 0})
                self.ramos[ramo]["visitas"] += 1
                self._init_embedding(ramo)
        return tuple(ramos)

    # ==================== KN ====================
    def _prep_continuacao(self):
        cont = {}
        for ctx, counter in self.transicoes.items():
            for w in counter:
                cont.setdefault(w, set()).add(ctx)
        total = sum(len(s) for s in cont.values())
        self._unigram = (
            {w: len(s) / total for w, s in cont.items()} if total else {}
        )

    def _kn_rec(self, c):
        if not c:
            return dict(self._unigram)
        if c not in self.transicoes:
            return self._kn_rec(c[1:])
        counter = self.transicoes[c]
        total = sum(counter.values())
        n1mais = len(counter)
        lam = self.D * n1mais / total
        fundo = self._kn_rec(c[1:])
        dist = {w: lam * p for w, p in fundo.items()}
        for w, cnt in counter.items():
            dist[w] = dist.get(w, 0) + max(cnt - self.D, 0) / total
        return dist

    # ==================== atenção temporal² ====================
    def _atencao_temporal(self, contexto):
        n = len(contexto)
        if n == 0:
            return None, None
        pesos = np.array([1.0 / (1.0 + (n - 1 - i)) ** 4 for i in range(n)])
        pesos = pesos / (pesos.sum() + 1e-12)
        vecs = np.stack([self.E[r] for r in contexto])
        ctx_vec = (vecs * pesos[:, None]).sum(axis=0)
        return ctx_vec, pesos

    def _forward_neural(self, contexto, candidatos):
        ctx_vec, pesos = self._atencao_temporal(contexto)
        if ctx_vec is None or not candidatos:
            return np.zeros(len(candidatos)), None, None
        E_cand = np.stack([self.E[c] for c in candidatos])
        logits = E_cand @ ctx_vec
        return logits, ctx_vec, pesos

    def _treinar_passo(self, contexto, alvo, candidatos):
        if not contexto or not candidatos:
            return 0.0
        if alvo not in candidatos:
            candidatos = list(candidatos) + [alvo]

        logits, ctx_vec, pesos = self._forward_neural(contexto, candidatos)
        z = logits - logits.max()
        p = np.exp(z) / (np.exp(z).sum() + 1e-12)

        idx_alvo = candidatos.index(alvo)
        loss = -np.log(p[idx_alvo] + 1e-12)

        grad_logits = p.copy()
        grad_logits[idx_alvo] -= 1.0

        E_cand = np.stack([self.E[c] for c in candidatos])
        grad_ctx = grad_logits @ E_cand
        grad_E_cand = grad_logits[:, None] * ctx_vec[None, :]
        grad_vecs = grad_ctx[None, :] * pesos[:, None]

        lr = self.lr
        for j, c in enumerate(candidatos):
            self.E[c] = self.E[c] - lr * grad_E_cand[j]
            np.clip(self.E[c], -self.E_clip, self.E_clip, out=self.E[c])
        for i, r in enumerate(contexto):
            self.E[r] = self.E[r] - lr * grad_vecs[i]
            np.clip(self.E[r], -self.E_clip, self.E_clip, out=self.E[r])

        self.n_passos += 1
        return float(loss)

    # ==================== absorver ====================
    def absorver(self, texto, treinar=True, epocas=5):
        linhas = 0
        for linha in texto.strip().splitlines():
            linha = linha.strip()
            if not linha:
                continue
            vidro = (self.BOS,) + self._acende(linha) + (self.EOS,)
            for i in range(1, len(vidro)):
                alvo = vidro[i]
                for k in range(1, min(self.n_max, i) + 1):
                    ctx = vidro[i - k:i]
                    self.transicoes.setdefault(ctx, Counter())[alvo] += 1
            linhas += 1

        self._prep_continuacao()

        if treinar:
            for _ in range(epocas):
                self._epoca_treino(texto)

        print(f"[ABSORVEU] {linhas} linhas | {len(self.transicoes)} contextos | "
              f"{len(self.ramos)} ramos | {len(self.E)} embeddings | "
              f"loss={self.loss_medio:.3f}")

    def _epoca_treino(self, texto):
        losses = []
        for linha in texto.strip().splitlines():
            linha = linha.strip()
            if not linha:
                continue
            vidro = (self.BOS,) + self._acende(linha, criar=False) + (self.EOS,)
            for i in range(1, len(vidro)):
                alvo = vidro[i]
                ctx = vidro[max(0, i - self.n_max):i]
                candidatos = set()
                if ctx in self.transicoes:
                    candidatos.update(self.transicoes[ctx].keys())
                candidatos.add(alvo)
                if self._unigram:
                    top = sorted(self._unigram, key=lambda x: -self._unigram[x])[:30]
                    candidatos.update(top)
                loss = self._treinar_passo(ctx, alvo, list(candidatos))
                losses.append(loss)
        if losses:
            media = sum(losses) / len(losses)
            self.loss_medio = 0.9 * self.loss_medio + 0.1 * media

    # ==================== integração ====================
    def _misturar_mix(self, candidatos, kn_dist, neu_logits, neural_T, alpha):
        kn_arr = np.array([kn_dist.get(c, 1e-9) for c in candidatos])
        kn_arr = kn_arr / (kn_arr.sum() + 1e-12)
        if alpha <= 0 or neu_logits is None:
            return kn_arr
        z = neu_logits / max(neural_T, 1e-3)
        z = z - z.max()
        p_neu = np.exp(z) / (np.exp(z).sum() + 1e-12)
        probs = (1 - alpha) * kn_arr + alpha * p_neu
        return probs / (probs.sum() + 1e-12)

    def _misturar_rerank(self, candidatos, kn_dist, neu_logits, neural_T, alpha,
                         top_rerank=20):
        top_kn = sorted(kn_dist, key=lambda x: -kn_dist[x])[:top_rerank]
        idx_map = {c: i for i, c in enumerate(candidatos)}
        top_validos = [c for c in top_kn if c in idx_map]
        if not top_validos or neu_logits is None:
            arr = np.array([kn_dist.get(c, 1e-9) for c in candidatos])
            return arr / (arr.sum() + 1e-12)

        neu_sub = np.array([neu_logits[idx_map[c]] for c in top_validos])
        z = neu_sub / max(neural_T, 1e-3)
        z = z - z.max()
        p_neu = np.exp(z) / (np.exp(z).sum() + 1e-12)

        kn_top = np.array([kn_dist.get(c, 1e-9) for c in top_validos])
        kn_top = kn_top / (kn_top.sum() + 1e-12)

        misto = (1 - alpha) * kn_top + alpha * p_neu

        probs = np.zeros(len(candidatos))
        for c, p in zip(top_validos, misto):
            probs[idx_map[c]] = p
        return probs / (probs.sum() + 1e-12)

    def _anti_loop_mix(self, probs, candidatos, historico, janela, forca):
        if not historico or forca >= 1.0:
            return probs
        recentes = Counter(self.ramos[h]["token"] for h in historico[-janela:])
        if not recentes:
            return probs
        novo = probs.copy()
        for i, c in enumerate(candidatos):
            tok = self.ramos[c]["token"]
            n = recentes.get(tok, 0)
            if n > 0:
                novo[i] *= forca ** n
        s = novo.sum()
        return novo / s if s > 1e-12 else probs

    # ==================== perceber ====================
    def perceber(self, texto, passos=15, temperatura=0.7,
                 alpha_neural=None, modo=None,
                 neural_T=1.5, anti_loop_janela=3, anti_loop_forca=0.3,
                 min_passos=3, mostrar=True):
        consulta = self._acende(texto, criar=False)
        modo = modo or self.modo
        alpha = self.alpha_neural if alpha_neural is None else alpha_neural
        print(f"\n[PERCEBEU] '{texto}'  "
              f"(modo={modo}, α={alpha}, T={temperatura}, Tn={neural_T})")
        if not consulta:
            print("   -> nenhum ramo conhecido")
            return texto

        historico = list(consulta)
        saida = texto.split()

        for passo in range(passos):
            atual = historico[-1]
            if atual == self.EOS:
                break

            contexto = tuple(historico[-self.n_max:])
            kn_dist = self._kn_rec(contexto)
            if not kn_dist:
                break

            candidatos = sorted(kn_dist, key=lambda x: -kn_dist[x])[:30]
            if self.EOS not in candidatos and passo >= min_passos:
                candidatos.append(self.EOS)
            if self.EOS in candidatos and passo < min_passos:
                candidatos = [c for c in candidatos if c != self.EOS]

            neu_logits = None
            if alpha > 0:
                neu_logits, _, _ = self._forward_neural(contexto, candidatos)

            if modo == "rerank":
                probs = self._misturar_rerank(candidatos, kn_dist, neu_logits,
                                              neural_T, alpha)
            else:
                probs = self._misturar_mix(candidatos, kn_dist, neu_logits,
                                           neural_T, alpha)

            probs = self._anti_loop_mix(probs, candidatos, historico,
                                        anti_loop_janela, anti_loop_forca)

            if mostrar:
                top_idx = np.argsort(-probs)[:4]
                amostra = "  ".join(
                    f"{self.ramos[candidatos[i]]['token']}"
                    f"@{self.ramos[candidatos[i]]['pos']}:{probs[i]:.2f}"
                    for i in top_idx
                )
                print(f"   ctx≈{len(contexto)}  [{amostra}]")

            if temperatura <= 0:
                w = candidatos[int(np.argmax(probs))]
            else:
                z = probs ** (1.0 / max(temperatura, 1e-3))
                z = z / (z.sum() + 1e-12)
                w = candidatos[int(np.random.choice(len(candidatos), p=z))]

            saida.append(self.ramos[w]["token"])
            historico.append(w)

        resultado = " ".join(saida)
        print(f"   -> {resultado}")
        return resultado

    # ==================== análise / io ====================
    def analise(self):
        print("\n[ANÁLISE]")
        print(f"   ramos      : {len(self.ramos)}")
        print(f"   contextos  : {len(self.transicoes)}")
        print(f"   embeddings : {len(self.E)} (dim={self.dim})")
        print(f"   passos SGD : {self.n_passos}")
        print(f"   loss médio : {self.loss_medio:.3f}")
        print(f"   modo       : {self.modo} | lr={self.lr} | E_clip={self.E_clip}")
        print("   amostra (hash → pos, token, ||embedding||):")
        for r, info in list(self.ramos.items())[:4]:
            norma = float(np.linalg.norm(self.E[r]))
            print(f"      {r % 10**8:>8} → pos={info['pos']:>2} "
                  f"tok={info['token']!r:<12} ||e||={norma:.3f}")

    def salvar(self, caminho="gns_linhagem.pkl"):
        with open(caminho, "wb") as f:
            pickle.dump({
                "dim": self.dim, "n_max": self.n_max, "D": self.D,
                "alpha_neural": self.alpha_neural, "modo": self.modo,
                "E_clip": self.E_clip,
                "transicoes": self.transicoes, "ramos": self.ramos,
                "E": self.E, "loss_medio": self.loss_medio,
            }, f)

    @classmethod
    def carregar(cls, caminho="gns_linhagem.pkl"):
        with open(caminho, "rb") as f:
            d = pickle.load(f)
        g = cls(dim=d["dim"], n_max=d["n_max"], D=d["D"],
                alpha_neural=d["alpha_neural"], modo=d.get("modo", "mix"),
                E_clip=d.get("E_clip", 1.5))
        g.transicoes = d["transicoes"]
        g.ramos = d["ramos"]
        g.E = d["E"]
        g.loss_medio = d.get("loss_medio", 0.0)
        g._prep_continuacao()
        return g


if __name__ == "__main__":
    CORPUS = """
    o rato roeu a roupa do rei de roma
    o rato roeu a corda do navio
    o rato correu para o buraco da parede
    o rato fugiu para o buraco da parede
    o rato olhou para a rainha com medo
    o rato subiu na mesa do banquete
    o rei de roma ficou furioso com o rato
    o rei mandou chamar o gato da rainha
    a rainha viu o rato e gritou bem alto
    a roupa do rei ficou toda roída
    o gato correu atrás do rato pelo palácio
    o gato ficou sem jantar naquela noite
    o gato subiu no telhado da torre
    o gato olhou para a lua cheia
    """

    gns = GNSLinhagem(dim=32, n_max=4, D=0.75, lr=0.02,
                      alpha_neural=0.3, modo="mix", E_clip=1.5)
    gns.absorver(CORPUS, treinar=True, epocas=10)
    gns.analise()

    print("\n=== α=0 (só KN) ===")
    gns.perceber("o rato", passos=10, temperatura=0.7, alpha_neural=0.0)

    print("\n=== MIX α=0.3 — 3x (deve divergir sem loop) ===")
    for _ in range(3):
        gns.perceber("o rato", passos=10, temperatura=0.7, alpha_neural=0.3)

    print("\n=== MIX α=0.5 — 2x ===")
    for _ in range(2):
        gns.perceber("o rato", passos=10, temperatura=0.7, alpha_neural=0.5)

    print("\n=== MIX α=0.8 (neural manda; E_clip deve segurar) ===")
    gns.perceber("o rato", passos=10, temperatura=0.7, alpha_neural=0.8)

    print("\n=== RERANK α=0.7 — 2x ===")
    for _ in range(2):
        gns.perceber("o rato", passos=10, temperatura=0.7,
                     alpha_neural=0.7, modo="rerank")

    print("\n=== FRIO T=0 α=0.5 (anti-loop deve quebrar 'a a a') ===")
    gns.perceber("o rato", passos=10, temperatura=0.0, alpha_neural=0.5)