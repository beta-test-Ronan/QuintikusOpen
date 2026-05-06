#!/usr/bin/env python3
import os, time, psutil, json, re, hashlib
import numpy as real_np

# --- ❄️ ESCUDO TÉRMICO ---
os.environ["OMP_NUM_THREADS"] = "4" 
try:
    import cupy as np
    from cupyx import scatter_add
    from cupy import asnumpy
    GPU_ACCEL = True
except ImportError:
    import numpy as np
    GPU_ACCEL = False
    def asnumpy(x): return x
    def scatter_add(a, slices, value): np.add.at(a, slices, value)

def safe_sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

# --- 1. TOKENIZER BPE (QI 16k) ---
class BPE:
    def __init__(self, target_vocab=16000):
        self.target_vocab = target_vocab
        self.merges, self.vocab = {}, {i: bytes([i]) for i in range(256)}
        self.vocab[256] = b"<|EOS|>"
        
    def train(self, text, min_tokens=500):
        print(f"🛠️ Treinando BPE para Vocab 16k...")
        tokens = list(text.encode("utf-8"))
        for i in range(self.target_vocab - 257):
            if len(tokens) <= min_tokens: break
            stats = {}
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            if not stats: break
            pair = max(stats, key=stats.get)
            idx = 257 + i
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            new_t, j = [], 0
            while j < len(tokens):
                if j < len(tokens)-1 and (tokens[j], tokens[j+1]) == pair:
                    new_t.append(idx); j += 2
                else: new_t.append(tokens[j]); j += 1
            tokens = new_t
        return tokens

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, idx in self.merges.items():
            new_t, j = [], 0
            while j < len(tokens):
                if j < len(tokens)-1 and (tokens[j], tokens[j+1]) == pair:
                    new_t.append(idx); j += 2
                else: new_t.append(tokens[j]); j += 1
            tokens = new_t
        return tokens

    def decode(self, tokens):
        res = [self.vocab.get(t, b"") for t in tokens]
        return b"".join(res).decode("utf-8", errors="replace")

# --- 2. ENGINE QUINTIKUS V7 (CIRCUITO FECHADO ESTÁVEL) ---
class QuintikusV7:
    def __init__(self, d_model=256, n_heads=8, n_layers=6, seq_len=256):
        self.d, self.h, self.layers, self.seq = d_model, n_heads, n_layers, seq_len
        self.d_head = d_model // n_heads
        self.scale = np.float32(self.d_head**-0.5)
        self.weights, self.m, self.v = {}, {}, {}
        self.step, self.dropout = 0, 0.3
        self.tok = BPE(target_vocab=16000)

    def _init_weights(self, vs):
        f = lambda i, o: np.random.randn(i, o).astype(np.float32) * np.sqrt(2.0/i)
        self.weights = {'emb': f(vs, self.d) * 0.02, 'pos': f(self.seq, self.d) * 0.02}
        for l in range(self.layers):
            self.weights[f'qkv_{l}'] = f(self.d, self.d * 3)
            self.weights[f'proj_{l}'] = f(self.d, self.d)
            self.weights[f'w1_{l}'] = f(self.d, self.d * 3)
            self.weights[f'w2_{l}'] = f(self.d, self.d * 3)
            self.weights[f'w3_{l}'] = f(self.d * 3, self.d) * 0.02
            self.weights[f'g1_{l}'] = np.ones((self.d,), dtype=np.float32)
            self.weights[f'g2_{l}'] = np.ones((self.d,), dtype=np.float32)
        for k, w in self.weights.items(): self.m[k], self.v[k] = np.zeros_like(w), np.zeros_like(w)

    def _rmsnorm_fwd(self, x, g):
        rms = (np.mean(x**2, axis=-1, keepdims=True) + 1e-6)**0.5
        x_norm = x / rms
        return x_norm * g, x_norm, rms

    def _rmsnorm_bwd(self, dy, x_norm, rms, g):
        dg = np.sum(dy * x_norm, axis=(0, 1))
        dx_n = dy * g
        dx = (1.0 / rms) * (dx_n - x_norm * np.mean(dx_n * x_norm, axis=-1, keepdims=True))
        return dx, dg

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)

    def forward(self, idx, mask, treino=False):
        B, T = idx.shape
        h = self.weights['emb'][idx] + self.weights['pos'][:T]
        drop_mask = None
        if treino:
            drop_mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float32)
            h = h * drop_mask / (1.0 - self.dropout)
        
        st = {'h_0': h, 'drop_mask': drop_mask}
        for l in range(self.layers):
            # MHA
            n1, xn1, rms1 = self._rmsnorm_fwd(h, self.weights[f'g1_{l}'])
            qkv = n1 @ self.weights[f'qkv_{l}']
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, self.h, self.d_head).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, self.h, self.d_head).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.h, self.d_head).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask[:T, :T]
            probs = self._softmax(scores)
            ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
            h = h + ctx @ self.weights[f'proj_{l}']
            # SwiGLU
            n2, xn2, rms2 = self._rmsnorm_fwd(h, self.weights[f'g2_{l}'])
            w1a, w2a = n2 @ self.weights[f'w1_{l}'], n2 @ self.weights[f'w2_{l}']
            si = w1a * safe_sigmoid(w1a)
            h = h + (si * w2a) @ self.weights[f'w3_{l}']
            st.update({f'n1_{l}':n1, f'xn1_{l}':xn1, f'rms1_{l}':rms1, f'q_{l}':q, f'k_{l}':k, f'v_{l}':v, 
                       f'probs_{l}':probs, f'ctx_{l}':ctx, f'n2_{l}':n2, f'xn2_{l}':xn2, f'rms2_{l}':rms2, 
                       f'w1a_{l}':w1a, f'w2a_{l}':w2a, f'si_{l}':si, f'h_{l+1}':h})
        return h @ self.weights['emb'].T, st

    def treinar(self, texto, épocas=300, batch=16, acc_steps=8):
        # Tokenize first, then split (Evita vazamento de BPE)
        all_tokens = self.tok.train(texto)
        vs = len(self.tok.vocab); self._init_weights(vs)
        data = np.array(all_tokens[:int(len(all_tokens)*0.9)], dtype=np.int32)
        val_data = np.array(all_tokens[int(len(all_tokens)*0.9):], dtype=np.int32)
        
        lr_max, wd, warmup = 0.0003, 0.01, 50
        mask = (np.tril(np.ones((self.seq, self.seq), dtype=np.float32)) - 1) * 1e9
        best_val_loss, patience, trigger = 1e9, 5, 0

        print(f"🚀 V7 TURBO ANALYTICS | VOCAB: {vs} | TOTAL BATCH: {batch * acc_steps}")

        for e in range(épocas + 1):
            t0 = time.perf_counter()
            acc_grads = {k: np.zeros_like(w) for k, w in self.weights.items()}
            total_train_loss = 0

            for _ in range(acc_steps):
                ix = np.random.randint(0, len(data) - self.seq - 1, (batch,))
                xb, yb = np.stack([data[i:i+self.seq] for i in ix]), np.stack([data[i+1:i+self.seq+1] for i in ix])
                logits, st = self.forward(xb, mask, treino=True)
                p = self._softmax(logits)
                b_idx, t_idx = np.arange(batch)[:, None], np.arange(self.seq)
                d_logits = p.copy(); d_logits[b_idx, t_idx, yb] -= 0.9; d_logits /= (batch * self.seq * acc_steps)
                
                # --- BACKPROP ANALÍTICO ---
                acc_grads['emb'] += d_logits.reshape(-1, vs).T @ st[f'h_{self.layers}'].reshape(-1, self.d)
                dh = d_logits @ self.weights['emb']
                for l in reversed(range(self.layers)):
                    # FFN Backward
                    d_ff = dh
                    acc_grads[f'w3_{l}'] += (st[f'si_{l}'] * st[f'w2a_{l}']).reshape(-1, self.d*3).T @ d_ff.reshape(-1, self.d)
                    d_swi_h = d_ff @ self.weights[f'w3_{l}'].T
                    d_w2a = d_swi_h * st[f'si_{l}']
                    sig = safe_sigmoid(st[f'w1a_{l}'])
                    d_w1a = d_swi_h * st[f'w2a_{l}'] * sig * (1 + st[f'w1a_{l}'] * (1 - sig))
                    acc_grads[f'w1_{l}'] += st[f'n2_{l}'].reshape(-1, self.d).T @ d_w1a.reshape(-1, self.d*3)
                    acc_grads[f'w2_{l}'] += st[f'n2_{l}'].reshape(-1, self.d).T @ d_w2a.reshape(-1, self.d*3)
                    dn2 = (d_w2a @ self.weights[f'w2_{l}'].T + d_w1a @ self.weights[f'w1_{l}'].T)
                    dx_ff, dg2 = self._rmsnorm_bwd(dn2, st[f'xn2_{l}'], st[f'rms2_{l}'], self.weights[f'g2_{l}'])
                    acc_grads[f'g2_{l}'] += dg2
                    dh = dh + dx_ff
                    # Attention Backward
                    acc_grads[f'proj_{l}'] += st[f'ctx_{l}'].reshape(-1, self.d).T @ dh.reshape(-1, self.d)
                    d_ctx = (dh @ self.weights[f'proj_{l}'].T).reshape(batch, self.seq, self.h, self.d_head).transpose(0, 2, 1, 3)
                    d_probs = d_ctx @ st[f'v_{l}'].transpose(0, 1, 3, 2)
                    dv = st[f'probs_{l}'].transpose(0, 1, 3, 2) @ d_ctx
                    ds = st[f'probs_{l}'] * (d_probs - np.sum(d_probs * st[f'probs_{l}'], axis=-1, keepdims=True))
                    dq, dk = (ds @ st[f'k_{l}']) * self.scale, (ds.transpose(0, 1, 3, 2) @ st[f'q_{l}']) * self.scale
                    dqkv = np.concatenate([dq, dk, dv], axis=-1).transpose(0, 2, 1, 3).reshape(batch, self.seq, -1)
                    acc_grads[f'qkv_{l}'] += st[f'n1_{l}'].reshape(-1, self.d).T @ dqkv
                    dx_att, dg1 = self._rmsnorm_bwd(dqkv @ self.weights[f'qkv_{l}'].T, st[f'xn1_{l}'], st[f'rms1_{l}'], self.weights[f'g1_{l}'])
                    acc_grads[f'g1_{l}'] += dg1
                    dh = dh + dx_att

                if st['drop_mask'] is not None: dh = dh * st['drop_mask'] / (1.0 - self.dropout)
                acc_grads['pos'] += np.sum(dh, axis=0)
                scatter_add(acc_grads['emb'], xb.reshape(-1), dh.reshape(-1, self.d))
                total_train_loss += float(asnumpy(-np.mean(np.log(p[b_idx, t_idx, yb] + 1e-9))))

            # --- SCHEDULER & ADAMW ---
            self.step += 1
            if self.step < warmup: lr = lr_max * self.step / warmup
            else: lr = lr_max * 0.5 * (1 + np.cos(np.pi * (self.step - warmup) / max(1, épocas - warmup)))
            
            for k in self.weights:
                if k in acc_grads:
                    g = acc_grads[k]
                    self.weights[k] *= (1 - wd * float(lr)) # DECOUPLED AdamW
                    self.m[k] = 0.9 * self.m[k] + 0.1 * g
                    self.v[k] = 0.999 * self.v[k] + 0.001 * (g**2)
                    mh, vh = self.m[k]/(1-0.9**self.step), self.v[k]/(1-0.999**self.step)
                    self.weights[k] -= float(lr) * (mh / (np.sqrt(vh) + 1e-8))

            # --- VALIDATION (ESTATÍSTICA) ---
            if e % 20 == 0 and len(val_data) > self.seq + 10:
                v_loss_acc = 0
                for _ in range(5):
                    iv = np.random.randint(0, len(val_data)-self.seq-1, (batch,))
                    xv, yv = np.stack([val_data[i:i+self.seq] for i in iv]), np.stack([val_data[i+1:i+self.seq+1] for i in iv])
                    b_val = np.arange(batch)[:, None]
                    lv, _ = self.forward(xv, mask, treino=False)
                    v_loss_acc += float(asnumpy(-np.mean(np.log(self._softmax(lv)[b_val, t_idx, yv] + 1e-9))))
                val_loss = v_loss_acc / 5
                print(f"E{e:<4} | Train: {total_train_loss/acc_steps:.4f} | Val: {val_loss:.4f} | LR: {float(lr):.6f}")
                if val_loss < best_val_loss:
                    best_val_loss, trigger = val_loss, 0
                    # Salva o melhor cérebro comprimido
                    w_cpu = {k: asnumpy(v) for k, v in self.weights.items()}
                    np.savez_compressed('brain_v7_best.npz', weights=w_cpu, merges=self.tok.merges, vocab=self.tok.vocab)
                else:
                    trigger += 1
                    if trigger >= patience: break

    def gerar(self, prompt, tokens=100, temp=0.4):
        idx = self.tok.encode(prompt)
        res = list(idx); mask = (np.tril(np.ones((self.seq, self.seq))) - 1) * 1e9
        print(f"\n[QUINTIKUS V7]:", end=" ")
        for _ in range(tokens):
            input_data = np.array([res[-self.seq:]])
            logits, _ = self.forward(input_data, mask)
            p = asnumpy(self._softmax(logits[0, -1] / temp))
            for t in set(res[-20:]): p[t] *= 0.1 
            nxt = int(real_np.random.choice(len(p), p=p/np.sum(p)))
            res.append(nxt); print(self.tok.decode([nxt]), end="", flush=True)
            if nxt == 256: break
        print(f"\n[FONTE: cache_tried_{hashlib.sha256(prompt.encode()).hexdigest()[:8]}]")

if __name__ == "__main__":
    dataset = "A proatividade e a integral do progresso acumulado no tempo. A objetividade aniquila o ruido. Eficacia e o resultado real. " * 300
    ggpt = QuintikusV7(d_model=256, n_layers=6, seq_len=256)
    ggpt.treinar(dataset, épocas=300, batch=16, acc_steps=8)
    ggpt.gerar("O que e proatividade?")
