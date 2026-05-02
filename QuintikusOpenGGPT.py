#!/usr/bin/env python3
import os

# --- ❄️ ESCUDO TÉRMICO (LIMITAÇÃO DE NÚCLEOS PARA NÃO TRAVAR) ---
os.environ["OMP_NUM_THREADS"] = "2" 
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import time, psutil, json

class Quintikus:
    def __init__(self, d_model=128, seq_len=48, batch=12):
        self.d, self.seq, self.batch = d_model, seq_len, batch
        self.scale = np.float32(d_model**-0.5)
        self.mask = (np.tril(np.ones((seq_len, seq_len), dtype=np.float32)) - 1) * 1e9
        self.b_idx = np.arange(batch)[:, None]
        self.t_idx = np.arange(seq_len)
        self.weights, self.m, self.v = {}, {}, {}
        self.step = 0
        self.vocab = None
        self.proc = psutil.Process()
        self.sono = 0.05 # Freio inicial

    def _rms(self, x):
        s = (np.mean(x**2, axis=-1, keepdims=True) + 1e-6)**-0.5
        return x * s, s

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)

    def save(self, nome="brain_hybrid.npz"):
        if self.vocab is None: return
        np.savez_compressed(nome, w=self.weights, v=self.vocab, d=self.d, s=self.seq, step=self.step, m=self.m, vv=self.v)
        print(f"\n💾 Cérebro consolidado!")

    def load(self, nome="brain_hybrid.npz"):
        if os.path.exists(nome):
            try:
                data = np.load(nome, allow_pickle=True)
                self.weights = data['w'].item(); self.vocab = data['v'].item()
                self.ivocab = {i: c for c, i in self.vocab.items()}
                self.d, self.seq, self.step = data['d'].item(), data['s'].item(), data['step'].item()
                self.m, self.v = data['m'].item(), data['vv'].item()
                print(f"📥 Cérebro Híbrido carregado!")
                return True
            except: return False
        return False

    def _think(self, x, treino=False):
        t = x.shape[1]
        h_raw = self.weights['emb'][x] + self.weights['pos'][:t]
        if treino: h_raw += np.random.normal(0, 0.01, h_raw.shape)
        qkv = h_raw @ self.weights['qkv']
        q, k, v = qkv[..., :self.d], qkv[..., self.d:self.d*2], qkv[..., self.d*2:]
        scores = np.einsum('btd,bsd->bts', q, k, optimize=True) * self.scale
        scores += self.mask[:t, :t]
        probs = self.softmax(scores)
        a_raw = probs @ v
        attn_out, s1 = self._rms(a_raw)
        ff1 = np.maximum(0, attn_out @ self.weights['ff1'])
        ff2, s2 = self._rms(ff1 @ self.weights['ff2'])
        logits = ff2 @ self.weights['emb'].T 
        return logits, probs, q, k, v, h_raw, ff1, ff2, s1, s2, a_raw

    def treinar(self, texto, épocas=2000, target_loss=0.35, cpu_alvo=65.0):
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
        lr, b1, b2, eps = 0.003, 0.9, 0.999, 1e-8
        t_total = time.perf_counter()

        print(f"🏁 TITAN-BRIDGE 33.0 | ALVO CPU: {cpu_alvo}%")

        try:
            for e in range(épocas + 1):
                t_ini = time.perf_counter()
                ix = np.random.randint(0, len(data) - self.seq - 1, (self.batch,))
                xb, yb = np.stack([data[i:i+self.seq] for i in ix]), np.stack([data[i+1:i+self.seq+1] for i in ix])
                
                logits, probs, q, k, v, h_in, ff1, ff2, s1, s2, a_raw = self._think(xb, treino=True)
                p = self.softmax(logits)
                
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
                for key, grad in [('head', g_head if 'head' in self.weights else g_emb_out), ('qkv', g_qkv), ('emb', g_emb_in + g_emb_out), ('pos', g_pos), ('ff1', g_ff1), ('ff2', g_ff2)]:
                    if key in self.weights:
                        grad = np.clip(grad, -1.0, 1.0)
                        self.m[key] = b1 * self.m[key] + (1 - b1) * grad
                        self.v[key] = b2 * self.v[key] + (1 - b2) * (grad**2)
                        mh = self.m[key]/(1 - b1**self.step); vh = self.v[key]/(1 - b2**self.step)
                        self.weights[key] -= lr * mh / (np.sqrt(vh) + eps)

                # --- 🧊 V-SYNC PID (CONTROLE TÉRMICO) ---
                if e % 5 == 0:
                    cpu_atual = self.proc.cpu_percent()
                    if cpu_atual > cpu_alvo: self.sono += 0.005
                    elif cpu_atual < cpu_alvo - 5: self.sono = max(0.001, self.sono - 0.005)
                time.sleep(self.sono)

                if e % 100 == 0:
                    loss = -np.mean(np.log(p[self.b_idx, self.t_idx, yb] + 1e-9))
                    print(f"E{e:04d} | Perda: {loss:.4f} | CPU: {cpu_atual}% | Freio: {self.sono:.3f}")
                    if loss < target_loss: break
        except KeyboardInterrupt: pass
        finally: self.save()

    def gerar(self, frase, tamanho=100, temp=0.5, top_p=0.9): 
        if not self.vocab: return
        idx = [self.vocab.get(c, 0) for c in frase]
        print(f"\n--- ✨ IA HÍBRIDA ---\n{frase}", end="")
        for _ in range(tamanho):
            logits, *_ = self._think(np.array([idx[-self.seq:]]), treino=False)
            logits = logits[0, -1, :] / temp
            
            # Top-p Otimizado com Proteção contra NaN
            sorted_i = np.argsort(logits)[::-1]
            probs = self.softmax(logits[sorted_i])
            cum_probs = np.cumsum(probs)
            
            # --- 🛡️ FIX: PROTEÇÃO CONTRA ZERAR TUDO ---
            # Remove apenas o que ultrapassa o top_p, mas mantém pelo menos 1 token
            indices_to_remove = cum_probs > top_p
            indices_to_remove[1:] = indices_to_remove[:-1].copy()
            indices_to_remove[0] = False
            probs[indices_to_remove] = 0
            
            if np.sum(probs) == 0: probs[0] = 1.0 # Fallback final
            probs /= np.sum(probs)
            
            nxt = np.random.choice(sorted_i, p=probs)
            print(self.ivocab[nxt], end="", flush=True); idx.append(nxt)
        print("\n")

if __name__ == "__main__":
    ggpt = Quintikus()
    wad = """
SE fome > 0.8 ENTAO comer. SE bateria < 0.2 ENTAO carregar.
SE perigo = 1 ENTAO fugir. SE humano = 1 ENTAO oi.
No meio do caminho tinha uma pedra. O poeta e um fingidor.
""" * 100 
    
    if not ggpt.load(): print("🆕 Criando Nova Mente...")
    ggpt.treinar(wad, target_loss=0.1) # Alvo baixo para perfeição
    
    ggpt.gerar("SE fome ", temp=0.2)
    ggpt.gerar("No meio ", temp=0.5)
