#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSpark 2.0 + GNS TEXT CLASSIFIER (TURBO EXTREMO)
- Geração O(1) para tokens aceitos
- Batch Learning (Gradient Accumulation)
- Cache de TF-IDF no Draft
"""
import numpy as np, math, random, re, sys, json, time
from pathlib import Path
from collections import defaultdict, Counter

# ═══════════ 1. TARGET LM (Otimizado com Cache) ═══════════
class TargetLM:
    def __init__(self, corpus, max_vocab=3000):
        self.corpus = corpus
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        self.tri = defaultdict(lambda: defaultdict(Counter))
        
        for fr in corpus:
            tk = ['<s>'] + fr.split() + ['</s>']
            for i in range(len(tk)):
                self.uni[tk[i]] += 1
                if i >= 1: self.bi[tk[i-1]][tk[i]] += 1
                if i >= 2: self.tri[tk[i-2]][tk[i-1]][tk[i]] += 1
                
        top_tokens = [t for t, _ in self.uni.most_common(max_vocab)]
        self.vocab = sorted(top_tokens)
        self.v2i = {t: i for i, t in enumerate(self.vocab)}
        self.unk_idx = self.v2i.get('<UNK>', 0)
        
        # ⚡ CACHE O(1)
        self.uni_sum = sum(self.uni.values())
        self.bi_sums = {k: sum(v.values()) for k, v in self.bi.items()}
        self.tri_sums = {k1: {k2: sum(v.values()) for k2, v in v2.items()} for k1, v2 in self.tri.items()}
    
    def prob(self, ctx, tok):
        """Lookup rápido O(1) para o token específico."""
        c = ctx[-2:] if len(ctx) >= 2 else ctx
        if len(c) == 2 and c[0] in self.tri and c[1] in self.tri[c[0]]:
            p = self.tri[c[0]][c[1]].get(tok, 0)
            if p > 0: return p / self.tri_sums[c[0]][c[1]]
        if c and c[-1] in self.bi:
            p = self.bi[c[-1]].get(tok, 0)
            if p > 0: return p / self.bi_sums[c[-1]]
        p = self.uni.get(tok, 1)
        return p / self.uni_sum

    def dist(self, ctx):
        """Distribuição completa (O(N)). Usada SÓ em caso de rejeição."""
        p = np.array([self.prob(ctx, t) for t in self.vocab], float)
        return p / p.sum()

    def sample(self, ctx, top_k=8):
        probs = np.array([self.prob(ctx, t) for t in self.vocab])
        idxs = np.argsort(probs)[-top_k:]
        p = probs[idxs] / probs[idxs].sum()
        return self.vocab[np.random.choice(idxs, p=p)]

# ═══════════ 2. TF-IDF VECTORIZER ═══════════
class TFIDFVectorizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.v2i = {t: i for i, t in enumerate(vocab)}
        self.idf = np.ones(len(vocab))
    
    def fit(self, corpus):
        N = len(corpus)
        doc_counts = Counter()
        for doc in corpus:
            for w in set(doc.split()):
                if w in self.v2i: doc_counts[w] += 1
        for w, idx in self.v2i.items():
            self.idf[idx] = math.log((1 + N) / (1 + doc_counts[w])) + 1
            
    def transform(self, context_window):
        vec = np.zeros(len(self.vocab))
        if not context_window: return vec
        tf = Counter(context_window)
        for w, count in tf.items():
            if w in self.v2i:
                vec[self.v2i[w]] = (count / len(context_window)) * self.idf[self.v2i[w]]
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        return vec

# ═══════════ 3. GNS NETWORK ═══════════
class GNSNetwork:
    def __init__(self, n_vocab, n_hidden1=64, n_hidden2=32, lr=0.05):
        self.lr = lr
        self.n_vocab = n_vocab
        self.W1 = np.random.randn(n_vocab, n_hidden1) * np.sqrt(1.0 / n_vocab)
        self.b1 = np.zeros((1, n_hidden1))
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(1.0 / n_hidden1)
        self.b2 = np.zeros((1, n_hidden2))
        self.W3 = np.random.randn(n_hidden2, n_vocab) * np.sqrt(1.0 / n_hidden2)
        self.b3 = np.zeros((1, n_vocab))
        self.error_history = []

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = np.tanh(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        expZ = np.exp(self.Z3 - np.max(self.Z3, axis=1, keepdims=True))
        self.A3 = expZ / np.sum(expZ, axis=1, keepdims=True)
        return self.A3

    def backward(self, X, Y_idx):
        m = X.shape[0]
        dZ3 = self.A3.copy()
        dZ3[np.arange(m), Y_idx] -= 1.0
        dZ3 /= m
        
        dW3 = self.A2.T @ dZ3; db3 = np.sum(dZ3, axis=0, keepdims=True)
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (1 - self.A2**2)
        dW2 = self.A1.T @ dZ2; db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - self.A1**2)
        dW1 = X.T @ dZ1; db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        
        loss = -np.mean(np.log(self.A3[np.arange(m), Y_idx] + 1e-12))
        self.error_history.append(loss)
        return loss

    def save(self, filename="modelo_gns_text"):
        np.savez_compressed(f"{filename}.npz", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)
    
    def load(self, filename="modelo_gns_text"):
        try:
            data = np.load(f"{filename}.npz")
            self.W1, self.b1 = data['W1'], data['b1']
            self.W2, self.b2 = data['W2'], data['b2']
            self.W3, self.b3 = data['W3'], data['b3']
            return True
        except: return False

# ═══════════ 4. GNS DRAFTER ═══════════
class GNSDrafter:
    def __init__(self, target_lm, block_size=6, context_window=3):
        self.target = target_lm
        self.block_size = block_size
        self.ctx_win = context_window
        self.vectorizer = TFIDFVectorizer(self.target.vocab)
        self.vectorizer.fit(self.target.corpus)
        self.network = GNSNetwork(n_vocab=len(self.target.vocab))
        self.model_file = "modelo_gns_text"
        
        # Buffer para Gradient Accumulation
        self.batch_X = []
        self.batch_Y = []
        
        if self.network.load(self.model_file):
            print("💾 Pesos GNS carregados (modelo_gns_text.npz)")
        else:
            print("🌱 Inicializando nova GNS Network...")
            self._pretrain()

    def _pretrain(self):
        print("   ⚡ Pré-treinando com Batch Processing...")
        start = time.time()
        X_batch, Y_batch = [], []
        BATCH_SIZE = 64
        
        for _ in range(3): 
            for fr in self.target.corpus:
                tokens = fr.split()
                for i in range(len(tokens) - 1):
                    ctx = tokens[max(0, i - self.ctx_win):i+1]
                    X_batch.append(self.vectorizer.transform(ctx))
                    tok_idx = self.target.v2i.get(tokens[i+1], self.target.unk_idx)
                    Y_batch.append(tok_idx)
                    
                    if len(X_batch) == BATCH_SIZE:
                        X_np = np.array(X_batch); Y_np = np.array(Y_batch)
                        self.network.forward(X_np)
                        self.network.backward(X_np, Y_np)
                        X_batch, Y_batch = [], []
                        
        self.network.save(self.model_file)
        with open(f"{self.model_file}.json", 'w') as f: 
            json.dump({"vocab": self.target.vocab, "idf": self.vectorizer.idf.tolist()}, f)
        print(f"   ✅ Pré-treino concluído em {time.time()-start:.2f}s.")

    def draft(self, anchor, context):
        ctx = (list(context) + [anchor]) if context else [anchor]
        toks, dists, X_cache = [], [], []
        for _ in range(self.block_size):
            recent = ctx[-self.ctx_win:] if len(ctx) >= self.ctx_win else ctx
            X = self.vectorizer.transform(recent).reshape(1, -1)
            X_cache.append(X) # ⚡ Guarda para o aprendizado
            
            probs = self.network.forward(X)[0]
            # Amostragem rápida O(1)
            idx = int(np.searchsorted(np.cumsum(probs), np.random.rand()))
            idx = min(idx, len(self.target.vocab) - 1)
            
            tok = self.target.vocab[idx]
            toks.append(tok)
            dists.append(probs)
            ctx.append(tok)
        return toks, dists, X_cache

    def accumulate_learn(self, X, target_idx):
        """Apenas guarda na memória, não calcula nada ainda."""
        self.batch_X.append(X.flatten())
        self.batch_Y.append(target_idx)

    def flush_learn(self):
        """Atualiza a rede UMA VEZ só no final do bloco."""
        if not self.batch_X: return
        X_np = np.array(self.batch_X)
        Y_np = np.array(self.batch_Y)
        self.network.forward(X_np)
        self.network.backward(X_np, Y_np)
        self.batch_X, self.batch_Y = [], []

# ═══════════ 5. LOOP INTERATIVO ═══════════
def main():
    caminho = Path(__file__).parent / "corpus.txt"
    if not caminho.exists(): sys.exit("❌ ERRO: corpus.txt não encontrado.")

    texto = caminho.read_text(encoding='utf-8', errors='ignore')
    frases = [s.strip() for s in re.split(r'[.!?\n]+', texto) if 3 < len(s.split()) < 40]
    if not frases: sys.exit("❌ ERRO: 0 frases válidas.")
    print(f"✅ corpus.txt: {len(frases)} frases.")

    print("🧠 Treinando TargetLM (Verificador)...")
    target = TargetLM(frases, max_vocab=3000) 
    
    print("🕸️ Construindo GNS Network (Drafter Neural)...")
    drafter = GNSDrafter(target, block_size=6)

    print("═" * 60)
    print("  GNS TEXT CLASSIFIER + DSPARK (TURBO EXTREMO)")
    print("  (Digite 'sair' para encerrar)")
    print("═" * 60)

    while True:
        try:
            pergunta = input("\n❓ prompt> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not pergunta or pergunta.lower() == 'sair': break

        tokens = pergunta.split()
        accepted = []
        acc_pre, attempted = 0, 0

        while len(accepted) < 30:
            anchor = tokens[-1]
            ctx = tokens[:-1] if len(tokens) > 1 else []
            full = ctx + [anchor]
            
            # Draft retorna o Cache do TF-IDF
            draft, dists, X_cache = drafter.draft(anchor, ctx)
            cur = list(full)
            prefix = []
            
            for i, tok in enumerate(draft):
                idx = target.v2i.get(tok, target.unk_idx)
                
                # ⚡ MÁGICA 1: Usa lookup O(1) para aceitar
                pt = target.prob(cur, tok) 
                pd = float(dists[i][idx])
                pacc = min(1.0, pt / (pd + 1e-12))
                attempted += 1
                
                if random.random() < pacc:
                    prefix.append(tok)
                    cur.append(tok)
                    acc_pre += 1
                    # ⚡ MÁGICA 2: Acumula o gradiente no buffer
                    drafter.accumulate_learn(X_cache[i], idx)
                else:
                    # ⚡ MÁGICA 3: Só calcula O(N) se for REJEITADO
                    pt_vec = target.dist(cur) 
                    corr = np.maximum(0.0, pt_vec - dists[i])
                    s = corr.sum()
                    if s > 1e-12:
                        new_idx = int(np.searchsorted(np.cumsum(corr), np.random.rand() * s))
                        new_idx = min(new_idx, len(target.vocab) - 1)
                        new_tok = target.vocab[new_idx]
                    else:
                        new_tok = target.sample(cur)
                        new_idx = target.v2i.get(new_tok, target.unk_idx)
                        
                    prefix.append(new_tok)
                    cur.append(new_tok)
                    drafter.accumulate_learn(X_cache[i], new_idx)
                    break
            
            # ⚡ FLUSH: Atualiza a rede em lote (super rápido)
            drafter.flush_learn()
            
            accepted.extend(prefix)
            tokens.extend(prefix)
            if '</s>' in accepted or len(accepted) >= 30: break

        acc = acc_pre / max(1, attempted)
        ultimo_erro = drafter.network.error_history[-1] if drafter.network.error_history else 0.0
        
        print(f"💬 {' '.join(accepted)}")
        print(f"   [Aceitação: {acc:.1%} | Loss: {ultimo_erro:.4f}]")

    drafter.network.save(drafter.model_file)
    print("\n💾 Pesos GNS salvos.")

if __name__ == "__main__":
    main()
