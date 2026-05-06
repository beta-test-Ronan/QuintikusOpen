#!/usr/bin/env python3
import os
import numpy as np
import time, psutil, json

# --- ❄️ ESCUDO TÉRMICO (LIMITAÇÃO DE NÚCLEOS) ---
os.environ["OMP_NUM_THREADS"] = "2" 
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

class Quintikus:
    def __init__(self, d_model=256, seq_len=48, batch=16):
        self.d, self.seq, self.batch = d_model, seq_len, batch
        self.scale = np.float32(d_model**-0.5)
        self.mask = (np.tril(np.ones((seq_len, seq_len), dtype=np.float32)) - 1) * 1e9
        self.b_idx = np.arange(batch)[:, None]
        self.t_idx = np.arange(seq_len)
        self.weights, self.m, self.v = {}, {}, {}
        self.step = 0
        self.vocab = None
        self.proc = psutil.Process()
        self.sono = 0.1 

    def _rms(self, x):
        s = (np.mean(x**2, axis=-1, keepdims=True) + 1e-6)**-0.5
        return x * s, s

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)

    def save(self, nome="brain_final.npz"):
        if self.vocab is None: return
        np.savez_compressed(nome, w=self.weights, v=self.vocab, d=self.d, s=self.seq, 
                            step=self.step, m=self.m, vv=self.v)
        print(f"\n💾 Cérebro e Memória de Adam consolidados!")

    def load(self, nome="brain_final.npz"):
        if os.path.exists(nome):
            try:
                data = np.load(nome, allow_pickle=True)
                self.weights = data['w'].item(); self.vocab = data['v'].item()
                self.ivocab = {i: c for c, i in self.vocab.items()}
                self.d, self.seq, self.step = data['d'].item(), data['s'].item(), data['step'].item()
                self.m, self.v = data['m'].item(), data['vv'].item()
                print(f"📥 Cérebro Híbrido Carregado (Step {self.step})")
                return True
            except: return False
        return False

    def _think(self, x, treino=False):
        t = x.shape[1]
        h = self.weights['emb'][x] + self.weights['pos'][:t]
        if treino: h += np.random.normal(0, 0.01, h.shape)
        qkv = h @ self.weights['qkv']
        q, k, v = qkv[..., :self.d], qkv[..., self.d:self.d*2], qkv[..., self.d*2:]
        scores = np.einsum('btd,bsd->bts', q, k, optimize=True) * self.scale
        scores += self.mask[:t, :t]
        probs = self.softmax(scores)
        a_raw = probs @ v
        n1, s1 = self._rms(a_raw)
        ff1 = np.maximum(0, n1 @ self.weights['ff1'])
        ff2, s2 = self._rms(ff1 @ self.weights['ff2'])
        return ff2 @ self.weights['emb'].T, probs, q, k, v, h, ff1, ff2, s1, s2, a_raw

    def treinar(self, texto, épocas=2000, target_loss=0.08, cpu_alvo=65.0):
        if self.vocab is None:
            chars = sorted(list(set(texto)))
            self.vocab = {c: i for i, c in enumerate(chars)}; self.ivocab = {i: c for c, i in self.vocab.items()}
        vs = len(self.vocab)
        if not self.weights:
            f = lambda i, o: (np.random.randn(i, o).astype(np.float32) * np.sqrt(2/i))
            self.weights = {'emb':f(vs,self.d)*0.1, 'pos':f(self.seq,self.d)*0.1, 'qkv':f(self.d,self.d*3),
                            'ff1':f(self.d,self.d*4), 'ff2':f(self.d*4,self.d)}
        for k, w in self.weights.items():
            if k not in self.m: self.m[k], self.v[k] = np.zeros_like(w), np.zeros_like(w)

        data = np.array([self.vocab[c] for c in texto if c in self.vocab], dtype=np.int32)
        lr, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
        t_ini_total = time.perf_counter()

        print(f"🔥 FORJA FINAL ATIVADA | ALVO LOSS: {target_loss} | CPU: {cpu_alvo}%")

        try:
            for e in range(épocas + 1):
                t_step = time.perf_counter()
                ix = np.random.randint(0, len(data) - self.seq - 1, (self.batch,))
                xb, yb = np.stack([data[i:i+self.seq] for i in ix]), np.stack([data[i+1:i+self.seq+1] for i in ix])
                
                logits, probs, q, k, v, h_in, ff1, ff2, s1, s2, a_raw = self._think(xb, treino=True)
                p = self.softmax(logits)
                
                # Backprop Full Circuit
                g_logits = p.copy(); g_logits[self.b_idx, self.t_idx, yb] -= 1; g_logits /= (self.batch * self.seq)
                g_emb_out = g_logits.reshape(-1, vs).T @ ff2.reshape(-1, self.d)
                d_ff2 = (g_logits @ self.weights['emb'])
                d_ff2 = (d_ff2 - ff2 * np.mean(d_ff2 * ff2, axis=-1, keepdims=True)) * s2
                g_ff2 = ff1.reshape(-1, self.d*4).T @ d_ff2.reshape(-1, self.d)
                d_ff1 = (d_ff2 @ self.weights['ff2'].T) * (ff1 > 0)
                g_ff1 = a_raw.reshape(-1, self.d).T @ d_ff1.reshape(-1, self.d*4)
                d_a = (d_ff1 @ self.weights['ff1'].T)
                d_a = (d_a - a_raw * np.mean(d_a * a_raw, axis=-1, keepdims=True)) * s1
                dV = np.einsum('bts,btd->bsd', probs, d_a); dprobs = np.einsum('btd,bsd->bts', d_a, v)
                dscores = dprobs * probs
                dQ = np.einsum('bts,bsd->btd', dscores, k) * self.scale; dK = np.einsum('bst,btd->bsd', dscores, q) * self.scale
                dQKV = np.concatenate([dQ, dK, dV], axis=-1); g_qkv = np.einsum('btd,btf->df', h_in, dQKV)
                dh = dQKV @ self.weights['qkv'].T; g_emb_in = np.zeros_like(self.weights['emb']); np.add.at(g_emb_in, xb, dh); g_pos = np.sum(dh, axis=0)

                self.step += 1
                for key, grad in [('qkv', g_qkv), ('emb', g_emb_in + g_emb_out), ('pos', g_pos), ('ff1', g_ff1), ('ff2', g_ff2)]:
                    grad = np.clip(grad, -1.0, 1.0)
                    self.m[key] = b1 * self.m[key] + (1 - b1) * grad
                    self.v[key] = b2 * self.v[key] + (1 - b2) * (grad**2)
                    mh = self.m[key]/(1 - b1**self.step); vh = self.v[key]/(1 - b2**self.step)
                    self.weights[key] -= lr * mh / (np.sqrt(vh) + eps)

                if e % 5 == 0:
                    cpu = self.proc.cpu_percent()
                    if cpu > cpu_alvo: self.sono += 0.01
                    elif cpu < cpu_alvo - 5: self.sono = max(0.001, self.sono - 0.01)
                time.sleep(self.sono)

                if e % 50 == 0:
                    loss = -np.mean(np.log(p[self.b_idx, self.t_idx, yb] + 1e-9))
                    print(f"E{e:04d} | Perda: {loss:.4f} | CPU: {cpu}% | Freio: {self.sono:.3f}")
                    if loss < target_loss: break
        except KeyboardInterrupt: pass
        finally: self.save()

    def gerar(self, frase, tamanho=100, temp=0.5, top_p=0.9): 
        idx = [self.vocab.get(c, 0) for c in frase]
        res = ""
        for _ in range(tamanho):
            logits, *_ = self._think(np.array([idx[-self.seq:]]), treino=False)
            logits = logits[0, -1, :] / (temp + 1e-9)
            # Top-p com proteção contra NaN
            sorted_i = np.argsort(logits)[::-1]
            probs = self.softmax(logits[sorted_i])
            probs[np.cumsum(probs) > top_p] = 0; probs /= (np.sum(probs) + 1e-9)
            if np.all(probs == 0): probs[0] = 1.0
            nxt = np.random.choice(sorted_i, p=probs)
            char = self.ivocab[nxt]
            res += char; idx.append(nxt)
            print(char, end="", flush=True)
            if char == "\n" and len(res) > 20: break
        return res

# ============================================================
# 🕹️ EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    ggpt = Quintikus(d_model=256, seq_len=48, batch=16)
    
    # NOVO DATASET: ENSINANDO PADRÕES (Variações de Sensores e Frases)
    wad = """
SE fome > 0.1 ENTAO comer. SE fome > 0.5 ENTAO buscar_comida. SE fome > 0.9 ENTAO desespero.
SE bateria < 0.1 ENTAO desligar. SE bateria < 0.4 ENTAO procurar_tomada. SE bateria > 0.8 ENTAO carregar_completo.
SE perigo = 1 ENTAO correr. SE perigo = 0 ENTAO explorar. SE humano = 1 ENTAO saudar.
O poeta e um fingidor. O poeta e um louco. O poeta e um visionario.
No meio do caminho tinha uma pedra. No meio da vida tinha um sonho. No meio do erro tinha a razao.
A alma e grande. A alma e pequena. Tudo vale a pena.
""" * 50 # Menos repetições, mais frases diferentes!

    if not ggpt.load("brain_final.npz"):
        print("🆕 Iniciando Treino de Forja...")
        ggpt.treinar(wad, épocas=3000, target_loss=0.08)
    
    print("\n--- 🤖 MODO INTERATIVO (Digite 'sair' para fechar) ---")
    while True:
        prompt = input("\nVocê: ")
        if prompt.lower() == "sair": break
        print("IA: ", end="")
        ggpt.gerar(prompt, temp=0.3, top_p=0.85)
