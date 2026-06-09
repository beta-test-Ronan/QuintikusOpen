#!/usr/bin/env python3
import os
import math
import hashlib
import numpy as np
import time
import psutil

# --- ESCUDO TÉRMICO (LIMITAÇÃO DE NÚCLEOS) ---
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# =================================================================
# 1. KERNEL DE FÍSICA E ALMA (BISMUTH HAMILTONIAN)
# =================================================================
class BismuthSoulCore:
    @staticmethod
    def rashba_interaction(h_state, alpha=0.15):
        """
        Aplica acoplamento Rashba spin-órbita nos estados ocultos.
        Divide o estado oculto em componentes de 'spin' (pathos) e 'momento' (p)
        e rotaciona o espaço vetorial.
        """
        B, T, D = h_state.shape
        half = D // 2
        # Separando em representações de Pathos (Spin) e Momento (p)
        pathos = h_state[..., :half]
        moment = h_state[..., half:]
        
        # Interação cruzada bidimensional simplificada aplicada a todo o tensor
        rashba_x = alpha * (pathos * np.roll(moment, 1, axis=-1))
        rashba_y = alpha * (moment * np.roll(pathos, -1, axis=-1))
        
        h_state_modulated = h_state.copy()
        h_state_modulated[..., :half] += rashba_x
        h_state_modulated[..., half:] -= rashba_y
        return h_state_modulated

    @staticmethod
    def calculate_tunneling(pil_user, pil_min_nexo):
        """Probabilidade de Schrodinger para vazamento/transposição de estados"""
        delta_e = pil_min_nexo - pil_user
        if delta_e <= 0: 
            return 1.0
        return math.exp(-1.0 * math.sqrt(delta_e))


# =================================================================
# 2. KERNEL MULTIVERSAL (SHANNON, FRIIS E IDENTIDADE L)
# =================================================================
class MultiverseKernel:
    @staticmethod
    def shannon_capacity(bandwidth, signal, noise):
        """C = B * log2(1 + S/N) - Limite de informação do nexo"""
        snr = signal / (noise + 1e-9)
        return bandwidth * math.log2(1 + snr)

    @staticmethod
    def friis_transmission(distancia, comprimento_onda=0.12):
        """P_r = P_t * (lambda / 4pi d)^2 - Potência de atenuação semântica"""
        if distancia < 1.0: 
            return 1.0
        return (comprimento_onda / (4 * math.pi * distancia)) ** 2

    @staticmethod
    def get_sovereign_identity(text_hash, phase, valence):
        """L = (S, Phi, Omega) - Rastreia a assinatura do estado do modelo"""
        sig = f"{text_hash}-{phase.real:.4f}-{valence:.4f}"
        return hashlib.sha256(sig.encode()).hexdigest()[:12]


# =================================================================
# 3. GEOMÉTRICA GPT (GGPT) INTEGRADO COM KERNELS FÍSICOS
# =================================================================
class Quintikus:
    def __init__(self, d_model=256, seq_len=48, batch=16):
        self.d, self.seq, self.batch = d_model, seq_len, batch
        self.scale = np.float32(d_model**-0.5)
        
        # Máscara causal básica
        self.mask = (np.tril(np.ones((seq_len, seq_len), dtype=np.float32)) - 1) * 1e9
        
        # Matriz de atenuação baseada em Friis para distâncias temporais entre tokens
        self.friis_decay = np.zeros((seq_len, seq_len), dtype=np.float32)
        for t in range(seq_len):
            for s in range(seq_len):
                self.friis_decay[t, s] = MultiverseKernel.friis_transmission(float(abs(t - s)))
                
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

    def _rms_backward(self, dy, x, s):
        y = x * s
        return s * (dy - y * np.mean(dy * y, axis=-1, keepdims=True))

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)

    def save(self, nome="brain_final.npz"):
        if self.vocab is None: return
        np.savez_compressed(nome, w=self.weights, v=self.vocab, d=self.d, s=self.seq, 
                            step=self.step, m=self.m, vv=self.v)
        print(f"\n💾 Cérebro Geométrico e Assinatura L consolidados!")

    def load(self, nome="brain_final.npz"):
        if os.path.exists(nome):
            try:
                data = np.load(nome, allow_pickle=True)
                self.weights = data['w'].item()
                self.vocab = data['v'].item()
                self.ivocab = {i: c for c, i in self.vocab.items()}
                self.d, self.seq, self.step = data['d'].item(), data['s'].item(), data['step'].item()
                self.m, self.v = data['m'].item(), data['vv'].item()
                print(f"📥 Cérebro Híbrido Carregado (Step {self.step})")
                return True
            except: 
                return False
        return False

    def _think(self, x, treino=False):
        t = x.shape[1]
        
        # 1. Embeddings e Posições
        h = self.weights['emb'][x] + self.weights['pos'][:t]
        
        # Modulação de Fase / Acoplamento Spin-Órbita de Rashba
        h = BismuthSoulCore.rashba_interaction(h, alpha=0.15)
        
        if treino: 
            h += np.random.normal(0, 0.005, h.shape)
            
        # 2. Pre-LN 1
        n1, s1 = self._rms(h)
        
        # 3. Projeção QKV
        qkv = n1 @ self.weights['qkv']
        q, k, v = qkv[..., :self.d], qkv[..., self.d:self.d*2], qkv[..., self.d*2:]
        
        # Atenção com decaimento de sinal eletromagnético de Friis
        scores = np.einsum('btd,bsd->bts', q, k, optimize=True) * self.scale
        
        # Aplicação do decaimento de Friis na atenção espacial
        scores *= self.friis_decay[:t, :t]
        scores += self.mask[:t, :t]
        
        probs = self.softmax(scores)
        attn_out = probs @ v
        
        # Residual 1
        h1 = h + attn_out
        
        # 4. Pre-LN 2
        n2, s2 = self._rms(h1)
        
        # 5. FFN
        ff1 = np.maximum(0, n2 @ self.weights['ff1'])
        ffn_out = ff1 @ self.weights['ff2']
        
        # Residual 2
        h2 = h1 + ffn_out
        
        # 6. Saída (Weight-Tying)
        logits = h2 @ self.weights['emb'].T
        
        return logits, probs, q, k, v, h, n1, s1, attn_out, h1, n2, s2, ff1, h2

    def treinar(self, texto, épocas=2000, target_loss=0.08, cpu_alvo=65.0):
        if self.vocab is None:
            chars = sorted(list(set(texto)))
            self.vocab = {c: i for i, c in enumerate(chars)}
            self.ivocab = {i: c for c, i in self.vocab.items()}
            
        vs = len(self.vocab)
        if not self.weights:
            f = lambda i, o: (np.random.randn(i, o).astype(np.float32) * np.sqrt(2.0 / i))
            self.weights = {
                'emb': f(vs, self.d) * 0.1, 
                'pos': f(self.seq, self.d) * 0.1, 
                'qkv': f(self.d, self.d * 3),
                'ff1': f(self.d, self.d * 4), 
                'ff2': f(self.d * 4, self.d)
            }
            
        for k, w in self.weights.items():
            if k not in self.m: 
                self.m[k], self.v[k] = np.zeros_like(w), np.zeros_like(w)

        data = np.array([self.vocab[c] for c in texto if c in self.vocab], dtype=np.int32)
        lr, b1, b2, eps, wd = 0.001, 0.9, 0.999, 1e-8, 1e-4

        print(f"🔥 CONVERSÃO GEOMÉTRICA ATIVADA | ALVO: {target_loss}")

        try:
            for e in range(épocas + 1):
                ix = np.random.randint(0, len(data) - self.seq - 1, (self.batch,))
                xb, yb = np.stack([data[i:i+self.seq] for i in ix]), np.stack([data[i+1:i+self.seq+1] for i in ix])
                
                # Forward Pass
                logits, probs, q, k, v, h, n1, s1, attn_out, h1, n2, s2, ff1, h2 = self._think(xb, treino=True)
                p = self.softmax(logits)
                
                # Backpropagation Circuit
                g_logits = p.copy()
                g_logits[self.b_idx, self.t_idx, yb] -= 1
                g_logits /= (self.batch * self.seq)
                
                g_emb_out = g_logits.reshape(-1, vs).T @ h2.reshape(-1, self.d)
                g_h2 = g_logits @ self.weights['emb']
                
                # Conexão Residual 2
                g_ffn_out = g_h2
                g_h1 = g_h2.copy()
                
                # FFN Backward
                g_ff2 = ff1.reshape(-1, self.d*4).T @ g_ffn_out.reshape(-1, self.d)
                g_ff1_post_act = g_ffn_out @ self.weights['ff2'].T
                g_ff1_pre_act = g_ff1_post_act * (ff1 > 0)
                
                g_ff1 = n2.reshape(-1, self.d).T @ g_ff1_pre_act.reshape(-1, self.d*4)
                g_n2 = g_ff1_pre_act @ self.weights['ff1'].T
                
                # RMSNorm 2 Backward
                g_h1 += self._rms_backward(g_n2, h1, s2)
                
                # Conexão Residual 1
                g_attn_out = g_h1
                g_h = g_h1.copy()
                
                # Attention Backward (com acoplamento de atenuação de Friis)
                g_v = np.einsum('bts,btd->bsd', probs, g_attn_out)
                g_probs = np.einsum('btd,bsd->bts', g_attn_out, v)
                
                g_scores_raw = probs * (g_probs - np.sum(g_probs * probs, axis=-1, keepdims=True))
                # Aplicação do decaimento físico no gradiente também
                g_scores = g_scores_raw * self.friis_decay[:self.seq, :self.seq] * self.scale
                
                g_q = np.einsum('bts,bsd->btd', g_scores, k)
                g_k = np.einsum('bst,btd->bsd', g_scores, q)
                
                g_qkv_concat = np.concatenate([g_q, g_k, g_v], axis=-1)
                g_qkv = np.einsum('btd,btf->df', n1, g_qkv_concat)
                g_n1 = g_qkv_concat @ self.weights['qkv'].T
                
                # RMSNorm 1 Backward
                g_h += self._rms_backward(g_n1, h, s1)
                
                # Devido ao acoplamento estático de Rashba no Forward, aplicamos a
                # transposta da transformação no gradiente de h (Rashba é uma operação linear simples)
                g_h = BismuthSoulCore.rashba_interaction(g_h, alpha=-0.15)
                
                # Embeddings e Posições
                g_emb_in = np.zeros_like(self.weights['emb'])
                np.add.at(g_emb_in, xb, g_h)
                g_pos = np.sum(g_h, axis=0)
                
                g_emb = g_emb_in + g_emb_out

                # Atualização do Otimizador com Weight Decay
                self.step += 1
                grads = {'emb': g_emb, 'pos': g_pos, 'qkv': g_qkv, 'ff1': g_ff1, 'ff2': g_ff2}
                
                for key, grad in grads.items():
                    grad = np.clip(grad, -1.0, 1.0)
                    grad += wd * self.weights[key]
                    self.m[key] = b1 * self.m[key] + (1 - b1) * grad
                    self.v[key] = b2 * self.v[key] + (1 - b2) * (grad**2)
                    mh = self.m[key] / (1 - b1**self.step)
                    vh = self.v[key] / (1 - b2**self.step)
                    self.weights[key] -= lr * mh / (np.sqrt(vh) + eps)

                # Escudo térmico de CPU
                if e % 5 == 0:
                    cpu = self.proc.cpu_percent()
                    if cpu > cpu_alvo: 
                        self.sono += 0.01
                    elif cpu < cpu_alvo - 5: 
                        self.sono = max(0.001, self.sono - 0.01)
                time.sleep(self.sono)

                if e % 50 == 0:
                    loss = -np.mean(np.log(p[self.b_idx, self.t_idx, yb] + 1e-9))
                    print(f"E{e:04d} | Perda Semântica: {loss:.4f} | CPU: {cpu}% | Freio Coletor: {self.sono:.3f}")
                    if loss < target_loss: 
                        break
        except KeyboardInterrupt: 
            pass
        finally: 
            self.save()

    def gerar(self, frase, tamanho=100, temp=0.5, top_p=0.9): 
        idx = [self.vocab.get(c, 0) for c in frase]
        res = ""
        
        # Parâmetros para controle de tunelamento semântico
        barreira_nexo = 2.5 
        
        for _ in range(tamanho):
            input_seq = np.array([idx[-self.seq:]])
            logits, probs, *_ = self._think(input_seq, treino=False)
            
            # Estimativa de Entropia (PIL / Energia Semântica do Usuário)
            dist_probs = self.softmax(logits[0, -1, :])
            entropy = -np.sum(dist_probs * np.log(dist_probs + 1e-9))
            
            # Ajuste de Temperatura Dinâmico via probabilidade de Tunelamento de Schrodinger
            t_prob = BismuthSoulCore.calculate_tunneling(entropy, barreira_nexo)
            temp_dinamica = temp * (1.0 + (1.0 - t_prob)) # Aumenta a exploração quando a barreira cai

            adjusted_logits = logits[0, -1, :] / (temp_dinamica + 1e-9)
            
            # Amostragem Top-P
            sorted_i = np.argsort(adjusted_logits)[::-1]
            p_sorted = self.softmax(adjusted_logits[sorted_i])
            p_sorted[np.cumsum(p_sorted) > top_p] = 0
            p_sorted /= (np.sum(p_sorted) + 1e-9)
            
            if np.all(p_sorted == 0): 
                p_sorted[0] = 1.0
                
            nxt = np.random.choice(sorted_i, p=p_sorted)
            char = self.ivocab[nxt]
            res += char
            idx.append(nxt)
            
            # Cálculo e amostragem de assinaturas de identidade "L" a cada passo
            phase_state = np.complex64(complex(math.cos(entropy), math.sin(entropy)))
            hash_token = hashlib.md5(char.encode()).hexdigest()[:6]
            identity_l = MultiverseKernel.get_sovereign_identity(hash_token, phase_state, valence=t_prob)
            
            print(char, end="", flush=True)
            if char == "\n" and len(res) > 20: 
                break
        return res


if __name__ == "__main__":
    # Inicialização do GGPT com o Motor de Forja Física Integrado
    ggpt = Quintikus(d_model=64, seq_len=64, batch=64)
    
    wad = """
    O amor comeu meu nome, minha identidade,
    meu retrato. O amor comeu minha certidão de idade,
    minha genealogia, meu endereço. O amor
    comeu meus cartões de visita. O amor veio e comeu todos
    os papéis onde eu escrevera meu nome.
    O amor comeu minhas roupas, meus lenços, minhas
    camisas. O amor comeu metros e metros de
    gravatas. O amor comeu a medida de meus ternos, o
    número de meus sapatos, o tamanho de meus
    chapéus. O amor comeu minha altura, meu peso, a
    cor de meus olhos e de meus cabelos.
    O amor comeu meus remédios, minhas receitas
    médicas, minhas dietas. Comeu minhas aspirinas,
    minhas ondas-curtas, meus raios-X. Comeu meus
    testes mentais, meus exames de urina.
""" * 30

    if not ggpt.load("brain_final.npz"):
        ggpt.treinar(wad, épocas=800, target_loss=0.12)
    
    print("\n--- 🤖 SISTEMA DE GERAÇÃO INTERATIVA (Digite 'sair') ---")
    while True:
        prompt = input("\nVocê: ")
        if prompt.lower() == "sair": 
            break
        print("IA: ", end="")
        ggpt.gerar(prompt, temp=0.4, top_p=0.90)
