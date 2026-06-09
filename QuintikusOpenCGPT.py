#!/usr/bin/env python3
import os
import numpy as np
import time
import threading

# --- ESCUDO TÉRMICO ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class CortexGeometricoGPT:
    def __init__(self, d_model=64, seq_len=32):
        self.d = d_model
        self.seq = seq_len
        
        # Coeficiente geométrico (escala do Kernel)
        self.gamma = np.float32(1.0 / (2.0 * (d_model ** 0.5)))
        self.mask = (np.tril(np.ones((seq_len, seq_len), dtype=np.float32)) - 1) * 1e9
        
        # Vocabulário estruturado
        letras = " abcdefghijklmnopqrstuvwxyzáéíóúçãõâêô-,.!?\n"
        self.vocab = {char: i for i, char in enumerate(letras)}
        self.ivocab = {i: char for i, char in enumerate(letras)}
        self.vs = len(self.vocab)
        
        # Inicialização dos Tensores Geométricos
        f = lambda i, o: (np.random.randn(i, o).astype(np.float32) * np.sqrt(2.0 / i))
        self.weights = {
            'emb': f(self.vs, self.d) * 0.1,
            'pos': f(self.seq, self.d) * 0.1,
            'qkv': f(self.d, self.d * 3),
            'ff1': f(self.d, self.d * 4),
            'ff2': f(self.d * 4, self.d)
        }
        
        self.m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.step = 0
        
        self.dopamine = 1.0     
        self.adenosine = 0.1    
        self.short_term_memory = [] 
        
        self.running = True
        self.lock = threading.Lock()

    def _rms(self, x):
        s = (np.mean(x**2, axis=-1, keepdims=True) + 1e-6)**-0.5
        return x * s, s

    def _rms_backward(self, dy, x, s):
        y = x * s
        return s * (dy - y * np.mean(dy * y, axis=-1, keepdims=True))

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)

    def _think(self, x):
        t = x.shape[1]
        h = self.weights['emb'][x] + self.weights['pos'][:t]
        
        # Normalização geométrica básica
        n1, s1 = self._rms(h)
        
        qkv = n1 @ self.weights['qkv']
        q, k, v = qkv[..., :self.d], qkv[..., self.d:self.d*2], qkv[..., self.d*2:]
        
        # --- ATENÇÃO POR DISTÂNCIA MÉTRICA (GEOMÉTRICA) ---
        # Cálculo de: -gamma * ||q - k||^2 = -gamma * (q^2 - 2qk + k^2)
        q_sq = np.sum(q**2, axis=-1, keepdims=True)
        k_sq = np.sum(k**2, axis=-1, keepdims=True).transpose(0, 2, 1)
        qk = np.einsum('btd,bsd->bts', q, k, optimize=True)
        
        dist_geometrica = q_sq - 2.0 * qk + k_sq
        scores = -self.gamma * dist_geometrica
        
        scores += self.mask[:t, :t]
        probs = self.softmax(scores)
        attn_out = probs @ v
        
        h1 = h + attn_out
        n2, s2 = self._rms(h1)
        ff1 = np.maximum(0, n2 @ self.weights['ff1'])
        ffn_out = ff1 @ self.weights['ff2']
        
        h2 = h1 + ffn_out
        logits = h2 @ self.weights['emb'].T
        
        return logits, probs, q, k, v, h, n1, s1, attn_out, h1, n2, s2, ff1, h2

    def atualizar_sinapses(self, seq_indices, lr_custom=None):
        if len(seq_indices) < self.seq + 1:
            return
            
        max_start = len(seq_indices) - self.seq - 1
        start = np.random.randint(0, max_start + 1)
        
        xb = np.array([seq_indices[start : start + self.seq]])
        yb = np.array([seq_indices[start + 1 : start + self.seq + 1]])
        
        with self.lock:
            logits, probs, q, k, v, h, n1, s1, attn_out, h1, n2, s2, ff1, h2 = self._think(xb)
            p = self.softmax(logits)
            
            # Backward dos logits de saída
            g_logits = p.copy()
            g_logits[0, np.arange(self.seq), yb[0]] -= 1
            g_logits /= self.seq
            
            g_emb_out = g_logits.reshape(-1, self.vs).T @ h2.reshape(-1, self.d)
            g_h2 = g_logits @ self.weights['emb']
            
            # FFN Backward
            g_ffn_out = g_h2
            g_ff2 = ff1.reshape(-1, self.d*4).T @ g_ffn_out.reshape(-1, self.d)
            g_ff1_post = g_ffn_out @ self.weights['ff2'].T
            g_ff1_pre = g_ff1_post * (ff1 > 0)
            g_ff1 = n2.reshape(-1, self.d).T @ g_ff1_pre.reshape(-1, self.d*4)
            g_n2 = g_ff1_pre @ self.weights['ff1'].T
            
            g_h1 = g_h2 + self._rms_backward(g_n2, h1, s2)
            
            # Backward do Bloco de Atenção Geométrica
            g_v = np.einsum('bts,btd->bsd', probs, g_h1)
            g_probs = np.einsum('btd,bsd->bts', g_h1, v)
            g_scores = probs * (g_probs - np.sum(g_probs * probs, axis=-1, keepdims=True))
            
            # Derivada exata em relação à distância euclidiana
            # d(scores)/dq = -2 * gamma * (q - k)
            # d(scores)/dk = 2 * gamma * (q - k)
            sum_g_scores_t = np.sum(g_scores, axis=-1, keepdims=True)
            sum_g_scores_s = np.sum(g_scores, axis=1, keepdims=True).transpose(0, 2, 1)
            
            g_q = -2.0 * self.gamma * (sum_g_scores_t * q - g_scores @ k)
            g_k = 2.0 * self.gamma * (g_scores.transpose(0, 2, 1) @ q - sum_g_scores_s * k)
            
            g_qkv = np.einsum('btd,btf->df', n1, np.concatenate([g_q, g_k, g_v], axis=-1))
            g_n1 = np.concatenate([g_q, g_k, g_v], axis=-1) @ self.weights['qkv'].T
            g_h = g_h1 + self._rms_backward(g_n1, h, s1)
            
            g_emb_in = np.zeros_like(self.weights['emb'])
            np.add.at(g_emb_in, xb, g_h)
            g_pos = np.sum(g_h, axis=0)
            
            # Atualização do Otimizador Adam
            self.step += 1
            lr = lr_custom if lr_custom is not None else (0.002 * self.dopamine)
            b1, b2, eps = 0.9, 0.999, 1e-8
            
            grads = {'emb': g_emb_in + g_emb_out, 'pos': g_pos, 'qkv': g_qkv, 'ff1': g_ff1, 'ff2': g_ff2}
            
            for key, grad in grads.items():
                grad = np.clip(grad, -1.0, 1.0)
                self.m[key] = b1 * self.m[key] + (1 - b1) * grad
                self.v[key] = b2 * self.v[key] + (1 - b2) * (grad**2)
                mh = self.m[key] / (1 - b1**self.step)
                vh = self.v[key] / (1 - b2**self.step)
                self.weights[key] -= lr * mh / (np.sqrt(vh) + eps)

    def pre_treinar_base(self, corpus, passos=600):
        print("⏳ Geometrizando o espaço latente: estruturando o manifold semântico...")
        indices = [self.vocab[c] for c in corpus.lower() if c in self.vocab]
        for p in range(passos):
            self.atualizar_sinapses(indices, lr_custom=0.004)
            if p % 150 == 0:
                print(f"   Mapeamento geométrico: {p}/{passos} ciclos...")
        print("✅ Manifold Semântico Estruturado!")

    def responder_em_tempo_real(self, texto_usuario):
        novos_indices = []
        for char in texto_usuario.lower():
            if char in self.vocab:
                val = self.vocab[char]
                novos_indices.append(val)
                self.short_term_memory.append(val)
                
        while len(self.short_term_memory) > 128:
            self.short_term_memory.pop(0)

        # Atualização rápida das vizinhanças semânticas
        if len(self.short_term_memory) >= self.seq + 1:
            for _ in range(6):  
                self.atualizar_sinapses(self.short_term_memory)

        contexto = novos_indices[-self.seq:]
        if len(contexto) < self.seq:
            contexto = [0] * (self.seq - len(contexto)) + contexto
            
        resposta = []
        temp_dinamica = 0.35
        
        print("CÉREBRO GEOMÉTRICO: ", end="", flush=True)
        for _ in range(40):
            input_arr = np.array([contexto[-self.seq:]])
            with self.lock:
                logits, *_ = self._think(input_arr)
            
            logits_finais = logits[0, -1, :].copy()
            
            # Impedimento de repetição na vizinhança local (8 caracteres)
            for char_passado in resposta[-8:]:
                logits_finais[char_passado] -= 3.0  
                
            sub_logits = logits_finais / (temp_dinamica + 1e-9)
            probs = self.softmax(sub_logits)
            
            nxt = np.random.choice(self.vs, p=probs)
            char = self.ivocab[nxt]
            
            print(char, end="", flush=True)
            resposta.append(nxt)
            contexto.append(nxt)
            
            if char == '\n' or len(resposta) > 35:
                break
        print()


# --- EXECUÇÃO DO TESTE ---
if __name__ == "__main__":
    brain = CortexGeometricoGPT(d_model=64, seq_len=16)
    
    corpus_treino = """
    oi tudo bem como vai voce
    ola eu sou o gpt realtime e estou aprendendo
    tudo bem por aqui e com voce meu amigo
    como voce esta hoje eu estou muito bem
    voce gosta de conversar comigo eu gosto muito de falar
    bom dia como vai a sua vida por ai
    tudo certo por aqui vamos conversar mais um pouco
    eu sou um cortex geometrico aprendendo portugues
    """
    
    brain.pre_treinar_base(corpus_treino, passos=600)
    
    print("\n📐 CGPT (CORTEX GEOMÉTRICO PRE-TREINADO) v2.2 ATIVO!")
    print("Atenção estruturada por distância euclidiana no espaço latente.")
    print("Digite 'sair' para fechar.\n")
    
    try:
        while True:
            prompt = input("VOCÊ: ")
            if prompt.lower() == 'sair':
                brain.running = False
                break
            if not prompt.strip():
                continue
            
            brain.responder_em_tempo_real(prompt)
            
    except KeyboardInterrupt:
        pass
