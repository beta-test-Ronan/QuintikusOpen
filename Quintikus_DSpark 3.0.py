#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSpark 2.0 + GNS TEXT CLASSIFIER ARCHITECTURE
- TF-IDF Vectorization
- GNS Neural Network (2 Hidden Layers, tanh, Softmax)
- Backward Flow (Online Learning)
- Model Persistence (.npz + .json)
"""
import numpy as np, math, random, re, sys, json
from pathlib import Path
from collections import defaultdict, Counter

# ═══════════ 1. TARGET LM (N-grama para verificação) ═══════════
class TargetLM:
    def __init__(self, corpus):
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
        self.vocab = sorted(self.uni.keys())
        self.v2i = {t: i for i, t in enumerate(self.vocab)}
    
    def prob(self, ctx, tok):
        c = ctx[-2:] if len(ctx) >= 2 else ctx
        if len(c) == 2 and c[0] in self.tri and c[1] in self.tri[c[0]]:
            p = self.tri[c[0]][c[1]].get(tok, 0)
            if p > 0: return p / sum(self.tri[c[0]][c[1]].values())
        if c and c[-1] in self.bi:
            p = self.bi[c[-1]].get(tok, 0)
            if p > 0: return p / sum(self.bi[c[-1]].values())
        p = self.uni.get(tok, 1)
        return p / sum(self.uni.values())

    def dist(self, ctx):
        p = np.array([self.prob(ctx, t) for t in self.vocab], float)
        return p / p.sum()

# ═══════════ 2. TF-IDF VECTORIZER ═══════════
class TFIDFVectorizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.v2i = {t: i for i, t in enumerate(vocab)}
        self.idf = {}
    
    def fit(self, corpus):
        N = len(corpus)
        doc_counts = Counter()
        for doc in corpus:
            for w in set(doc.split()):
                doc_counts[w] += 1
        for w in self.vocab:
            self.idf[w] = math.log((1 + N) / (1 + doc_counts[w])) + 1  # Smoothing
            
    def transform(self, context_window):
        """Transforma o contexto recente em um vetor TF-IDF."""
        vec = np.zeros(len(self.vocab))
        if not context_window: return vec
        tf = Counter(context_window)
        for w, count in tf.items():
            if w in self.v2i:
                vec[self.v2i[w]] = (count / len(context_window)) * self.idf.get(w, 1.0)
        # Normalização L2
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        return vec

# ═══════════ 3. GNS NETWORK (Linhagem Architecture) ═══════════
class GNSNetwork:
    def __init__(self, n_vocab, n_hidden1=64, n_hidden2=32, lr=0.01):
        self.lr = lr
        self.n_vocab = n_vocab
        self.n_h1 = n_hidden1
        self.n_h2 = n_hidden2
        
        # Weights & Biases (Inicialização Xavier)
        self.W1 = np.random.randn(n_vocab, n_hidden1) * np.sqrt(1.0 / n_vocab)
        self.b1 = np.zeros((1, n_hidden1))
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(1.0 / n_hidden1)
        self.b2 = np.zeros((1, n_hidden2))
        self.W3 = np.random.randn(n_hidden2, n_vocab) * np.sqrt(1.0 / n_hidden2)
        self.b3 = np.zeros((1, n_vocab))
        
        self.error_history = []

    def forward(self, X):
        """Forward Pass com ativação tanh e softmax."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.tanh(self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        # Softmax
        expZ3 = np.exp(self.Z3 - np.max(self.Z3, axis=1, keepdims=True))
        self.A3 = expZ3 / np.sum(expZ3, axis=1, keepdims=True)
        return self.A3

    def backward(self, X, Y_target):
        """Backward Flow (Gradient Descent)"""
        m = X.shape[0]
        # Cross-Entropy derivative + Softmax
        dZ3 = self.A3 - Y_target
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (1 - np.power(self.A2, 2)) # tanh derivative
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (1 - np.power(self.A1, 2))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update Weights
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        
        # Error Monitor
        loss = -np.sum(Y_target * np.log(self.A3 + 1e-12)) / m
        self.error_history.append(loss)
        return loss

    def save(self, filename="modelo_gns_text"):
        np.savez(f"{filename}.npz", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)
    
    def load(self, filename="modelo_gns_text"):
        try:
            data = np.load(f"{filename}.npz")
            self.W1, self.b1 = data['W1'], data['b1']
            self.W2, self.b2 = data['W2'], data['b2']
            self.W3, self.b3 = data['W3'], data['b3']
            return True
        except:
            return False

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
        
        # Tenta carregar persistência
        if self.network.load(self.model_file):
            print("💾 Pesos GNS carregados (modelo_gns_text.npz)")
        else:
            print("🌱 Inicializando nova GNS Network...")
            self._pretrain()

    def _pretrain(self):
        """Pré-treina a rede com o corpus para não começar aleatória."""
        print("   Treinando GNS no corpus...")
        epochs = 10
        for _ in range(epochs):
            for fr in self.target.corpus:
                tokens = fr.split()
                for i in range(len(tokens) - 1):
                    ctx = tokens[max(0, i - self.ctx_win):i+1]
                    X = self.vectorizer.transform(ctx).reshape(1, -1)
                    Y = np.zeros((1, len(self.target.vocab)))
                    Y[0, self.target.v2i[tokens[i+1]]] = 1.0
                    self.network.forward(X)
                    self.network.backward(X, Y)
        self.network.save(self.model_file)
        # Metadados
        meta = {"vocab": self.target.vocab, "idf": self.vectorizer.idf}
        with open(f"{self.model_file}.json", 'w') as f: json.dump(meta, f)
        print("   ✅ Pré-treino concluído.")

    def draft(self, anchor, context):
        """Gera bloco usando a GNS Network."""
        ctx = (list(context) + [anchor]) if context else [anchor]
        toks, dists, prev = [], [], anchor
        for _ in range(self.block_size):
            # Pega os últimos `ctx_win` tokens
            recent = ctx[-self.ctx_win:] if len(ctx) >= self.ctx_win else ctx
            X = self.vectorizer.transform(recent).reshape(1, -1)
            
            # Forward pass da GNS
            probs = self.network.forward(X)[0]
            
            # Sample
            idx = np.random.choice(len(self.target.vocab), p=probs)
            tok = self.target.vocab[idx]
            toks.append(tok)
            dists.append(probs)
            ctx.append(tok)
            prev = tok
            
        return toks, dists

    def learn_online(self, context, target_token, accepted):
        """Backward Flow para aprender com o draft."""
        recent = context[-self.ctx_win:] if len(context) >= self.ctx_win else context
        X = self.vectorizer.transform(recent).reshape(1, -1)
        
        # O target é o token que o TargetLM (verificador) escolheu/aceitou
        Y = np.zeros((1, len(self.target.vocab)))
        Y[0, self.target.v2i[target_token]] = 1.0
        
        self.network.forward(X)
        loss = self.network.backward(X, Y)
        return loss

# ═══════════ 5. LOOP INTERATIVO ═══════════
def main():
    caminho = Path(__file__).parent / "corpus.txt"
    if not caminho.exists():
        sys.exit("❌ ERRO: corpus.txt não encontrado na pasta do script.")

    texto = caminho.read_text(encoding='utf-8', errors='ignore')
    frases = [s.strip() for s in re.split(r'[.!?\n]+', texto) if 3 < len(s.split()) < 40]
    if not frases: sys.exit("❌ ERRO: 0 frases válidas.")
    print(f"✅ corpus.txt: {len(frases)} frases.")

    print("🧠 Treinando TargetLM (Verificador)...")
    target = TargetLM(frases)
    
    print("🕸️ Construindo GNS Network (Drafter Neural)...")
    drafter = GNSDrafter(target, block_size=6)

    print("═" * 60)
    print("  GNS TEXT CLASSIFIER + DSPARK - LOOP INTERATIVO")
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
            
            # 1. GNS Network faz o Draft
            draft, dists = drafter.draft(anchor, ctx)
            
            cur = list(full)
            prefix = []
            
            # 2. TargetLM verifica + Aprendizado Online
            for i, tok in enumerate(draft):
                idx = target.v2i.get(tok, 0)
                pt_vec = target.dist(cur)
                pt = pt_vec[idx]
                pd = float(dists[i][idx])
                pacc = min(1.0, pt / (pd + 1e-12))
                attempted += 1
                
                if random.random() < pacc:
                    prefix.append(tok)
                    cur.append(tok)
                    acc_pre += 1
                    # 🔥 AQUI: A rede GNS aprende o padrão do token aceito
                    drafter.learn_online(cur[:-1], tok, True)
                else:
                    # Rejeitou: Resample do target
                    corr = np.maximum(0.0, pt_vec - dists[i])
                    s = corr.sum()
                    new_tok = (target.vocab[np.random.choice(len(target.vocab), p=corr/s)]
                               if s > 1e-12 else target.sample(cur))
                    prefix.append(new_tok)
                    cur.append(new_tok)
                    # 🔥 AQUI: A rede GNS aprende com a correção do target
                    drafter.learn_online(cur[:-1], new_tok, False)
                    break
            
            accepted.extend(prefix)
            tokens.extend(prefix)
            if '</s>' in accepted or len(accepted) >= 30: break

        acc = acc_pre / max(1, attempted)
        ultimo_erro = drafter.network.error_history[-1] if drafter.network.error_history else 0.0
        
        print(f"💬 {' '.join(accepted)}")
        print(f"   [Aceitação: {acc:.1%} | Error Monitor (Loss): {ultimo_erro:.4f}]")

    # Save Persistence ao sair
    drafter.network.save(drafter.model_file)
    print("\n💾 Pesos da GNS Network salvos em modelo_gns_text.npz")

if __name__ == "__main__":
    main()
