# -*-coding:utf8;-*-
import os, math, time, random, re, pickle, hashlib, unicodedata, threading
from collections import defaultdict, Counter, deque

# ==================================================================
# ❄️ [ÁREA 1: ESCUDO DE HARDWARE & MATEMÁTICA PURA]
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def clip(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def sigmoid_fn(x):
    return 1.0 / (1.0 + math.exp(-clip(x, -20.0, 20.0)))

def softplus_fn(x):
    return math.log1p(math.exp(clip(x, -20.0, 20.0)))

class NormalizadorSomático:
    @staticmethod
    def limpar(texto):
        if not texto: return ""
        texto = texto.lower()
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-z0-9!?.\s]', '', texto).strip()

class KernelRessonante:
    @staticmethod
    def get_vetor_esparso(token, dims=5000, sparsity=100):
        seed = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        return {i: rng.gauss(0, 1) for i in rng.sample(range(dims), sparsity)}

    @staticmethod
    def tsallis_match(v1, v2, q=0.8):
        keys = v1.keys() & v2.keys()
        if not keys: return 0.0
        sum_pq = sum((abs(v1[k] * v2[k]))**q for k in keys)
        return (1.0 - sum_pq) / (q - 1.0 + 1e-9)

    @staticmethod
    def dot(v1, v2):
        keys = v1.keys() & v2.keys()
        return sum(v1[k] * v2[k] for k in keys) if keys else 0.0

    @staticmethod
    def normalize(v):
        norm = math.sqrt(sum(x*x for x in v.values()))
        if norm < 1e-9: return {}
        return {d: val / norm for d, val in v.items()}

# ==================================================================
# 🧠 [ÁREA 2: NEURAL ATTENTION CONTROLLER - NAC (ADAM NATIVO)]
# ==================================================================
class NeuralAttentionController:
    def __init__(self, input_dim=5, hidden_dim=12, output_dim=8, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        
        # Inicialização Xavier
        scale1 = math.sqrt(2.0 / input_dim)
        self.W1 = [[random.gauss(0.0, 1.0) * scale1 for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        
        scale2 = math.sqrt(2.0 / hidden_dim)
        self.W2 = [[random.gauss(0.0, 1.0) * scale2 for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        
        self.m_W1 = [[0.0] * hidden_dim for _ in range(input_dim)]
        self.v_W1 = [[0.0] * hidden_dim for _ in range(input_dim)]
        self.m_b1 = [0.0] * hidden_dim
        self.v_b1 = [0.0] * hidden_dim
        
        self.m_W2 = [[0.0] * output_dim for _ in range(hidden_dim)]
        self.v_W2 = [[0.0] * output_dim for _ in range(hidden_dim)]
        self.m_b2 = [0.0] * output_dim
        self.v_b2 = [0.0] * output_dim

    def forward(self, x):
        self.x = list(x) # Input vector de tamanho 5
        
        # Camada Oculta: h = ReLU(x * W1 + b1)
        self.h = []
        for j in range(self.hidden_dim):
            col_val = sum(self.x[i] * self.W1[i][j] for i in range(self.input_dim)) + self.b1[j]
            self.h.append(max(0.0, col_val))
            
        # Camada de Saída: out = Softplus(h * W2 + b2)
        self.out_raw = []
        for j in range(self.output_dim):
            col_val = sum(self.h[i] * self.W2[i][j] for i in range(self.hidden_dim)) + self.b2[j]
            self.out_raw.append(col_val)
            
        self.out = [softplus_fn(val) for val in self.out_raw]
        return self.out

    def backward(self, d_out):
        # Gradientes de saída ativada
        sigmoid_raw = [1.0 / (1.0 + math.exp(-clip(val, -20.0, 20.0))) for val in self.out_raw]
        d_out_raw = [do * sr for do, sr in zip(d_out, sigmoid_raw)]
        
        # dW2 (hidden_dim x output_dim), db2 (output_dim)
        dW2 = [[self.h[i] * d_out_raw[j] for j in range(self.output_dim)] for i in range(self.hidden_dim)]
        db2 = list(d_out_raw)
        
        # dh = d_out_raw * W2.T
        dh = []
        for i in range(self.hidden_dim):
            if self.h[i] <= 0.0:
                dh.append(0.0)
            else:
                val = sum(d_out_raw[j] * self.W2[i][j] for j in range(self.output_dim))
                dh.append(val)
                
        # dW1 (input_dim x hidden_dim), db1 (hidden_dim)
        dW1 = [[self.x[i] * dh[j] for j in range(self.hidden_dim)] for i in range(self.input_dim)]
        db1 = list(dh)
        
        self.t += 1
        c_b1 = 1.0 - self.beta1 ** self.t
        c_b2 = 1.0 - self.beta2 ** self.t
        
        # Otimizador Adam Dinâmico
        def update_2d(param, grad, m, v):
            for i in range(len(param)):
                for j in range(len(param[i])):
                    g = grad[i][j]
                    m[i][j] = self.beta1 * m[i][j] + (1.0 - self.beta1) * g
                    v[i][j] = self.beta2 * v[i][j] + (1.0 - self.beta2) * (g ** 2)
                    m_hat = m[i][j] / c_b1
                    v_hat = v[i][j] / c_b2
                    param[i][j] -= self.lr * m_hat / (math.sqrt(abs(v_hat)) + self.eps)
                    
        def update_1d(param, grad, m, v):
            for i in range(len(param)):
                g = grad[i]
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * (g ** 2)
                m_hat = m[i] / c_b1
                v_hat = v[i] / c_b2
                param[i] -= self.lr * m_hat / (math.sqrt(abs(v_hat)) + self.eps)

        update_2d(self.W1, dW1, self.m_W1, self.v_W1)
        update_1d(self.b1, db1, self.m_b1, self.v_b1)
        update_2d(self.W2, dW2, self.m_W2, self.v_W2)
        update_1d(self.b2, db2, self.m_b2, self.v_b2)

# ==================================================================
# 📡 [ÁREA 3: CNN 1D DE COERÊNCIA SINTÁTICA GEOMÉTRICA]
# ==================================================================
class CoherenceCNN1D:
    def __init__(self, sequence_length=15, embedding_dim=3):
        self.seq_len = sequence_length
        self.emb_dim = embedding_dim
        self.num_filters = 16 
        self.filter_width = 3
        
        scale1 = math.sqrt(2.0 / (self.filter_width * embedding_dim))
        self.W_conv = [[[random.gauss(0.0, 1.0) * scale1 for _ in range(self.num_filters)] for _ in range(embedding_dim)] for _ in range(self.filter_width)]
        self.b_conv = [0.0] * self.num_filters
        
        scale2 = math.sqrt(2.0 / self.num_filters)
        self.W_dense = [random.gauss(0.0, 1.0) * scale2 for _ in range(self.num_filters)]
        self.b_dense = 0.0

    def forward(self, sentence_coords):
        X = list(sentence_coords)
        if len(X) < self.seq_len:
            X = X + [[0.0, 0.0, 0.0]] * (self.seq_len - len(X))
        else:
            X = X[:self.seq_len]
        self.X = X
        
        conv_out_len = self.seq_len - self.filter_width + 1
        self.conv_features = []
        
        for i in range(conv_out_len):
            feature_row = []
            window = X[i:i+self.filter_width]
            for f in range(self.num_filters):
                val = sum(window[r][c] * self.W_conv[r][c][f] for r in range(self.filter_width) for c in range(self.emb_dim))
                val += self.b_conv[f]
                feature_row.append(val)
            self.conv_features.append(feature_row)
            
        # ReLu
        self.conv_activated = [[max(0.0, val) for val in row] for row in self.conv_features]
        
        # Max Pooling
        self.pooled = []
        self.pool_indices = []
        for f in range(self.num_filters):
            max_val = -float('inf')
            max_idx = 0
            for i in range(conv_out_len):
                if self.conv_activated[i][f] > max_val:
                    max_val = self.conv_activated[i][f]
                    max_idx = i
            self.pooled.append(max_val)
            self.pool_indices.append(max_idx)
            
        logits = sum(self.pooled[f] * self.W_dense[f] for f in range(self.num_filters)) + self.b_dense
        self.score = sigmoid_fn(logits)
        return self.score

    def backward_step(self, target, lr=0.01):
        d_logits = self.score - target
        
        dW_dense = [self.pooled[f] * d_logits for f in range(self.num_filters)]
        db_dense = d_logits
        
        d_pooled = [d_logits * self.W_dense[f] for f in range(self.num_filters)]
        
        # Atualização pesos do Dense
        for f in range(self.num_filters):
            self.W_dense[f] -= lr * dW_dense[f]
        self.b_dense -= lr * db_dense
        
        # Atualização pesos do Conv
        for f in range(self.num_filters):
            idx_max = self.pool_indices[f]
            if self.conv_features[idx_max][f] > 0.0:
                d_val = d_pooled[f]
                self.b_conv[f] -= lr * d_val
                for r in range(self.filter_width):
                    for c in range(self.emb_dim):
                        self.W_conv[r][c][f] -= lr * self.X[idx_max + r][c] * d_val

# ==================================================================
# ⚖️ [ÁREA 4: FILTRO DE INTEGRIDADE CONTEXTUAL (ACCEPTY)]
# ==================================================================
class Accepty:
    def __init__(self, threshold=0.20):
        self.threshold = threshold

    def extrair_sujeito_predicado(self, tokens):
        if len(tokens) <= 1:
            return tokens, tokens
        meio = max(1, len(tokens) // 2)
        return tokens[:meio], tokens[meio:]

    def calcular_vetor_medio(self, tokens, coordenadas_palavras):
        coords = [coordenadas_palavras[t] for t in tokens if t in coordenadas_palavras]
        if not coords:
            return [0.0, 0.0, 0.0]
        n = len(coords)
        return [sum(c[0] for c in coords)/n, sum(c[1] for c in coords)/n, sum(c[2] for c in coords)/n]

    def similaridade_cosseno(self, v1, v2):
        n1 = math.sqrt(sum(x*x for x in v1))
        n2 = math.sqrt(sum(x*x for x in v2))
        if n1 == 0.0 or n2 == 0.0: 
            return 0.0
        return sum(a*b for a, b in zip(v1, v2)) / (n1 * n2)

    def avaliar(self, prompt_tokens, cand_tokens, coordenadas_palavras):
        p_suj, p_pred = self.extrair_sujeito_predicado(prompt_tokens)
        c_suj, c_pred = self.extrair_sujeito_predicado(cand_tokens)

        vec_p_suj = self.calcular_vetor_medio(p_suj, coordenadas_palavras)
        vec_p_pred = self.calcular_vetor_medio(p_pred, coordenadas_palavras)
        vec_c_suj = self.calcular_vetor_medio(c_suj, coordenadas_palavras)
        vec_c_pred = self.calcular_vetor_medio(c_pred, coordenadas_palavras)

        sim_sujeito = self.similaridade_cosseno(vec_p_suj, vec_c_suj)
        sim_predicado = self.similaridade_cosseno(vec_p_pred, vec_c_pred)

        passou = (sim_sujeito >= self.threshold) and (sim_predicado >= self.threshold)
        score_integridade = ((sim_sujeito + 1.0) / 2.0) * ((sim_predicado + 1.0) / 2.0)

        return passou, score_integridade

# ==================================================================
# 👥 [ÁREA 5: TEORIA DA MENTE, MEMÓRIA DE TRABALHO & DRIVES]
# ==================================================================
class TeoriaDaMente:
    def __init__(self):
        self.estimativa_humor = {"confiança": 0.5, "agressividade": 0.1, "atenção": 1.0}

    def atualizar(self, u_toks, dkl_usuario):
        pessimistas = {"odeio", "mal", "triste", "burro", "erro", "ruim", "falso"}
        otimistas = {"amo", "bom", "prazer", "sim", "obrigado", "legal", "certo"}
        c_pessimistas = sum(1 for t in u_toks if t in pessimistas)
        c_otimistas = sum(1 for t in u_toks if t in otimistas)
        
        self.estimativa_humor["agressividade"] = min(1.0, max(0.0, self.estimativa_humor["agressividade"] * 0.9 + (c_pessimistas * 0.15) - (c_otimistas * 0.05)))
        self.estimativa_humor["confiança"] = min(1.0, max(0.0, self.estimativa_humor["confiança"] * 0.95 + (c_otimistas * 0.08) - (dkl_usuario * 0.04)))
        self.estimativa_humor["atenção"] = max(0.1, min(1.0, 0.7 * self.estimativa_humor["atenção"] + 0.3 * (1.0 / (dkl_usuario + 1.0))))

class MemoriaTrabalho:
    def __init__(self, capacidade=6):
        self.buffer = deque(maxlen=capacidade)
        self.vetor_suavizado = {}

    def registrar(self, v_perceptivo, tokens, acao_snc, soma_eixos):
        self.buffer.append({
            "v": v_perceptivo,
            "tokens": tokens,
            "acao": acao_snc,
            "soma": dict(soma_eixos),
            "stamp": time.time()
        })
        self._sintetizar_suavizacao()

    def _sintetizar_suavizacao(self):
        self.vetor_suavizado.clear()
        itens = list(self.buffer)[-3:]
        for idx, item in enumerate(itens):
            peso = (idx + 1) / len(itens)
            for k, val in item["v"].items():
                self.vetor_suavizado[k] = self.vetor_suavizado.get(k, 0.0) + val * peso
        self.vetor_suavizado = KernelRessonante.normalize(self.vetor_suavizado)

    def aplicar_gravidade_temporal(self, gravidade):
        for k in self.vetor_suavizado:
            self.vetor_suavizado[k] *= (1.0 - gravidade)
        self.vetor_suavizado = KernelRessonante.normalize(self.vetor_suavizado)

class DriveSomático:
    def __init__(self):
        self.vm = -70.0 
        self.eixos = {"amor": 0.1, "prazer": 0.1, "tristeza": 0.1, "raiva": 0.1}
        self.inercia = 1.0
        self.simbiose = 0.0

    def pulsar(self, impacto, dkl, u_toks, turno):
        self.vm = max(-90.0, min(-45.0, self.vm + impacto * 12.0))
        max_e = max(self.eixos.values())
        self.inercia = max(0.1, min(0.9, (max_e * 1.5) / (dkl + 0.1)))
        self.simbiose = (max_e * 2.25) / (math.log(turno + 1.2) + dkl + 1e-5)
        
        gatilhos = {"amor":["amo","amor"], "prazer":["prazer","delicia"], "tristeza":["triste","mal"], "raiva":["odeio","raiva"]}
        for eixo, keywords in gatilhos.items():
            for k in keywords:
                if k in u_toks:
                    delta = (max(self.eixos[eixo], 0.1) / 1.5) * impacto * (1.0 - self.inercia)
                    self.eixos[eixo] = min(5.0, self.eixos[eixo] + delta)

    def aplicar_deriva_temporal(self, gravidade):
        self.eixos["tristeza"] = min(5.0, self.eixos["tristeza"] + gravidade * 1.8)
        self.eixos["prazer"] = max(0.1, self.eixos["prazer"] * (1.0 - gravidade))
        self.vm = max(-90.0, self.vm - gravidade * 10.0)

    def metabolizar_decaimento(self):
        self.vm = max(-70.0, self.vm - 0.3)
        for eixo in self.eixos:
            self.eixos[eixo] = max(0.1, self.eixos[eixo] - 0.01)

# ==================================================================
# 🧠 [ÁREA 6: SISTEMA NERVOSO CENTRAL & CÓRTEX COGNITIVO (DIVERGÊNCIA KL)]
# ==================================================================
class CortexCognitivo:
    def __init__(self, limite_confusao=0.35):
        self.limite_confusao = limite_confusao
        self.epsilon = 1e-9

    def _norm(self, d):
        s = sum(abs(x) for x in d) + self.epsilon
        return [abs(x) / s for x in d]

    def calcular_dkl(self, p_real, q_interno):
        pn, qn = self._norm(p_real), self._norm(q_interno)
        dkl = 0.0
        for i in range(min(len(pn), len(qn))):
            dkl += pn[i] * math.log((pn[i] + self.epsilon) / (qn[i] + self.epsilon))
        return max(0.0, dkl)

class SistemaNervosoCentral:
    def __init__(self, n_in=6, n_hid=10, n_out=3, path="sistema_nervoso.bin"):
        self.path, self.n_in, self.n_hid, self.n_out = path, n_in, n_hid, n_out
        self.t, self.lr = 0, 0.005
        
        self.W_h = [[random.uniform(-0.1, 0.1) for _ in range(n_in + n_hid)] for _ in range(n_hid)]
        self.W_y = [[random.uniform(-0.1, 0.1) for _ in range(n_hid)] for _ in range(n_out)]
        self.B_h, self.B_y = [0.0]*n_hid, [0.0]*n_out
        
        self.adam_M_Wh = [[0.0]*(n_in+n_hid) for _ in range(n_hid)]
        self.adam_V_Wh = [[0.0]*(n_in+n_hid) for _ in range(n_hid)]
        self.adam_M_Wy = [[0.0]*n_hid for _ in range(n_out)]
        self.adam_V_Wy = [[0.0]*n_hid for _ in range(n_out)]
        
        self.q_table = defaultdict(lambda: [0.0] * n_out)
        self.gamma = 0.85
        self.alpha_q = 0.15
        self.estado_anterior = [0.0]*n_hid
        self.cache = None
        if os.path.exists(path): self._carregar()

    def sigmoid(self, x): return 1.0 / (1.0 + math.exp(-max(-15, min(15, x))))

    def pulsar_vontade(self, x_sen, exploracao=0.0):
        inp = x_sen + self.estado_anterior
        h = [self.sigmoid(self.B_h[i] + sum(inp[j]*self.W_h[i][j] for j in range(len(inp)))) for i in range(self.n_hid)]
        y = [self.sigmoid(self.B_y[i] + sum(h[j]*self.W_y[i][j] for j in range(self.n_hid))) for i in range(self.n_out)]
        if exploracao > 0.0:
            y = [max(0.01, min(0.99, yi + random.gauss(0, exploracao))) for yi in y]
        self.cache, self.estado_anterior = (inp, h, y), h
        return y

    def obter_hash_estado(self, modo_anterior, vm, dkl, impacto):
        return f"m:{modo_anterior}|v:{int(vm / 5.0)}|d:{int(dkl * 10.0)}|i:{int(impacto * 10.0)}"

    def aplicar_recompensa_td(self, estado_str, acao_idx, recompensa, proximo_estado_str):
        max_q_futuro = max(self.q_table[proximo_estado_str])
        td_target = recompensa + self.gamma * max_q_futuro
        td_error = td_target - self.q_table[estado_str][acao_idx]
        self.q_table[estado_str][acao_idx] += self.alpha_q * td_error

    def adaptar_realtime(self, alvo):
        if not self.cache: return
        self.t += 1
        inp, h, y = self.cache
        lr, b1, b2, eps = self.lr, 0.9, 0.999, 1e-8
        dy = [(y[i]-alvo[i])*(y[i]*(1-y[i])) for i in range(len(y))]
        c_b1, c_b2 = 1 - b1**self.t, 1 - b2**self.t
        for i in range(len(y)):
            for j in range(len(h)):
                grad = dy[i] * h[j]
                self.adam_M_Wy[i][j] = b1*self.adam_M_Wy[i][j] + (1-b1)*grad
                self.adam_V_Wy[i][j] = b2*self.adam_V_Wy[i][j] + (1-b2)*(grad**2)
                self.W_y[i][j] -= lr * (self.adam_M_Wy[i][j]/c_b1) / (math.sqrt(abs(self.adam_V_Wy[i][j])/c_b2) + eps)
        self._salvar()

    def _salvar(self):
        try:
            with open(self.path, 'wb') as f:
                pickle.dump({'Wh':self.W_h, 'Wy':self.W_y, 'Bh':self.B_h, 'By':self.B_y, 'ea':self.estado_anterior, 't':self.t,
                             'MWh':self.adam_M_Wh, 'VWh':self.adam_V_Wh, 'MWy':self.adam_M_Wy, 'VWy':self.adam_V_Wy, 'q_table': dict(self.q_table)}, f)
        except: pass

    def _carregar(self):
        try:
            with open(self.path, 'rb') as f:
                d = pickle.load(f); self.W_h, self.W_y, self.B_h, self.B_y, self.t = d['Wh'], d['Wy'], d['Bh'], d['By'], d['t']
                self.adam_M_Wh, self.adam_V_Wh, self.adam_M_Wy, self.adam_V_Wy = d['MWh'], d['VWh'], d['MWy'], d['VWy']
                self.estado_anterior = d['ea']
                if 'q_table' in d:
                    self.q_table = defaultdict(lambda: [0.0]*self.n_out, d['q_table'])
        except Exception as e:
            # Auto-cura preventiva contra arquivos vazios ou corrompidos do SNC
            print(f"⚠️ [AVISO] Falha ao carregar sistema_nervoso.bin ({e}). Reiniciando SNC com pesos padrão.")

# ==================================================================
# 🩹 [ÁREA 7: ATROZIA, DELONG, ANOMINI & PLASTICIDADE GRU]
# ==================================================================
class Anomini:
    def __init__(self):
        self.dor = {"intensidade": 0.0, "contexto": "Sistema estável."}
        self.saudade = {"intensidade": 0.0, "contexto": "Presença conceitual estável."}
        self.cache_otimizacao = deque(maxlen=6)

    def atualizar_estados(self, dkl_atual, confianca_tom, dt):
        self.cache_otimizacao.append(dkl_atual)
        if len(self.cache_otimizacao) >= 4:
            if (self.cache_otimizacao[-1] >= self.cache_otimizacao[-2] - 1e-5 and 
                self.cache_otimizacao[-2] >= self.cache_otimizacao[-3] - 1e-5 and
                self.cache_otimizacao[-3] >= self.cache_otimizacao[-4] - 1e-5):
                self.dor["intensidade"] = min(5.0, self.dor["intensidade"] + 0.45) 
                self.dor["contexto"] = "Fricção semântica: incapacidade contínua de reduzir DKL."
            else:
                self.dor["intensidade"] = max(0.0, self.dor["intensidade"] - 0.5) 
                self.dor["contexto"] = "SNC otimizando entropia com sucesso."
                
        distancia_alinhamento = 1.0 - confianca_tom
        gravidade_temporal = 1.0 - math.exp(-dt / 90.0)
        self.saudade["intensidade"] = min(5.0, self.saudade["intensidade"] * 0.9 + (distancia_alinhamento * 0.5) + (gravidade_temporal * 0.5))

class Atrozia:
    def __init__(self):
        self.historico_absoluto = deque(maxlen=200) # Deque limita o consumo de RAM e previne o vazamento infinito
        self.last_v_vencedor = {}
        self.loop_detector = deque(maxlen=3)
        self.damping = 1.0

    def amortecer_loop(self, v_in):
        if not self.last_v_vencedor: return 1.0
        sim = sum(v_in.get(k,0) * self.last_v_vencedor.get(k,0) for k in (v_in.keys() & self.last_v_vencedor.keys()))
        self.loop_detector.append(sim)
        self.damping = 0.4 if (sum(self.loop_detector)/len(self.loop_detector)) > 0.75 else 1.0
        return self.damping

    def calcular_sinergia(self, v_cand):
        if not self.last_v_vencedor: return 0.5
        return KernelRessonante.dot(self.last_v_vencedor, v_cand)

class Delong:
    def __init__(self, limiar_repeticao=4, janela=5):
        self.limiar = limiar_repeticao
        self.janela = deque(maxlen=janela)
        self.estado = "normal"  

    def monitorar(self, u_toks):
        if u_toks:
            padrao = " ".join(u_toks)
            self.janela.append(padrao)
        if len(self.janela) == self.janela.maxlen:
            if all(item == self.janela[0] for item in self.janela):
                self.estado = "questionando"
                return True
        self.estado = "normal"
        return False

    def gerar_pergunta(self):
        ultimo_token = self.janela[-1] if self.janela else "isso"
        return f"Você está repetindo constantemente '{ultimo_token}'. Qual o seu objetivo? Não entendo por que fala tanto sobre isso."

class Plasticidade:
    def __init__(self, n_in=6, n_hid=6):
        self.n_in, self.n_hid, self.lr_local = n_in, n_hid, 0.05
        self.W_z = [[random.uniform(-0.1, 0.1) for _ in range(n_in)] for _ in range(n_hid)]
        self.U_z = [[random.uniform(-0.1, 0.1) for _ in range(n_hid)] for _ in range(n_hid)]
        self.W_r = [[random.uniform(-0.1, 0.1) for _ in range(n_in)] for _ in range(n_hid)]
        self.U_r = [[random.uniform(-0.1, 0.1) for _ in range(n_hid)] for _ in range(n_hid)]
        self.W_h = [[random.uniform(-0.1, 0.1) for _ in range(n_in)] for _ in range(n_hid)]
        self.U_h = [[random.uniform(-0.1, 0.1) for _ in range(n_hid)] for _ in range(n_hid)]
        self.h = [0.0] * n_hid
        self.janela_contexto = deque(maxlen=8)
        self.utilidade_sinaptica = defaultdict(float)
        self.ultimo_uso_token = defaultdict(int)

    def _sigmoid(self, x): return 1.0 / (1.0 + math.exp(-max(-15, min(15, x))))
    def _tanh(self, x): return math.tanh(max(-15, min(15, x)))

    def processar_sentido_conversa(self, x_soma):
        self.janela_contexto.append(x_soma)
        self.h = [0.0] * self.n_hid
        for x in self.janela_contexto:
            z = [self._sigmoid(sum(self.W_z[i][j]*x[j] for j in range(self.n_in)) + sum(self.U_z[i][j]*self.h[j] for j in range(self.n_hid))) for i in range(self.n_hid)]
            r = [self._sigmoid(sum(self.W_r[i][j]*x[j] for j in range(self.n_in)) + sum(self.U_r[i][j]*self.h[j] for j in range(self.n_hid))) for i in range(self.n_hid)]
            h_til = [self._tanh(sum(self.W_h[i][j]*x[j] for j in range(self.n_in)) + sum(self.U_h[i][j]*(r[j]*self.h[j]) for j in range(self.n_hid))) for i in range(self.n_hid)]
            self.h = [(1.0 - z[i])*self.h[i] + z[i]*h_til[i] for i in range(self.n_hid)]
        return self.h

    # ==================================================================
    # 🔮 ANTECIPAÇÃO CONTEXTUAL (PREDICTIVE CODING)
    # ==================================================================
    def prever_antecipacao(self):
        """
        Projeta o estado oculto atual (h) um passo à frente no tempo para prever
        o próximo perfil de drives somáticos, sem alterar a janela de contexto real.
        """
        if not self.janela_contexto:
            return [0.0] * self.n_hid
        x_ultimo = self.janela_contexto[-1]
        z = [self._sigmoid(sum(self.W_z[i][j]*x_ultimo[j] for j in range(self.n_in)) + sum(self.U_z[i][j]*self.h[j] for j in range(self.n_hid))) for i in range(self.n_hid)]
        r = [self._sigmoid(sum(self.W_r[i][j]*x_ultimo[j] for j in range(self.n_in)) + sum(self.U_r[i][j]*self.h[j] for j in range(self.n_hid))) for i in range(self.n_hid)]
        h_til = [self._tanh(sum(self.W_h[i][j]*x_ultimo[j] for j in range(self.n_in)) + sum(self.U_h[i][j]*(r[j]*self.h[j]) for j in range(self.n_hid))) for i in range(self.n_hid)]
        h_predito = [(1.0 - z[i])*self.h[i] + z[i]*h_til[i] for i in range(self.n_hid)]
        return h_predito

    def adaptar_gru_local(self, x_next):
        for i in range(self.n_hid):
            erro = x_next[i] - self.h[i]
            for j in range(self.n_in):
                delta_W = self.lr_local * erro * (1.0 - self.h[i]**2) * self.janela_contexto[-1][j]
                self.W_h[i][j] = max(-2.0, min(2.0, self.W_h[i][j] + delta_W))

    def registrar_atividade_sinaptica(self, tokens, impacto, turno):
        for t in tokens:
            self.utilidade_sinaptica[t] += impacto
            self.ultimo_uso_token[t] = turno

    def calcular_escala_idade(self, token, turno_atual, gama=0.015):
        if token not in self.ultimo_uso_token: return 1.0
        idade = turno_atual - self.ultimo_uso_token[token]
        return math.exp(-gama * idade)

    def metabolizar_decaimento_sinaptico(self):
        for t in list(self.utilidade_sinaptica.keys()): self.utilidade_sinaptica[t] *= 0.94

    def aplicar_morte_sinaptica(self, mapa_nd, neuronios, l2_episodes, raridade, threshold=0.04):
        purgados_tokens = []
        for t in list(mapa_nd.keys()):
            if self.utilidade_sinaptica[t] < threshold and raridade[t] < 6:
                purgados_tokens.append(t)
                del mapa_nd[t]
                if t in neuronios: del neuronios[t]
                if t in self.utilidade_sinaptica: del self.utilidade_sinaptica[t]
                if t in self.ultimo_uso_token: del self.ultimo_uso_token[t]
        
        episodios_restantes = [ep for ep in l2_episodes if sum(ep['v'].values()) > 0.15]
        l2_episodes[:] = episodios_restantes
        if purgados_tokens:
            print(f"🧹 [MORTE SINÁPTICA] Purgação de {len(purgados_tokens)} conexões obsoletas.")

# ==================================================================
# 🌿 [ÁREA 8: ORGANISMO SOBERANO (FUSÃO COGNITIVO-MIMÉTICA)]
# ==================================================================
class OrganismoSoberano:
    def __init__(self):
        self.path_bin, self.path_ledger = "nucleo_organismo.qssml", "ledger.bin"
        self.auto_train_files = ["oi.txt", "amor.txt", "prazer.txt", "confusa.txt", "sentimento.txt"]
        
        # Estruturas Unificadas
        self.mapa_nd = {}                 # 5000D Sparse representações
        self.coordenadas_palavras = {}    # 3D coordenadas geométricas
        self.relacoes = defaultdict(lambda: defaultdict(set)) # Triplas Lógicas
        self.matrix_dlm = {}              # Ligações Markovianas [token]["links"]
        self.l2_episodes = []             # Episódios cristalizados
        self.neuronios = defaultdict(list)
        self.raridade = Counter()
        self.history = deque(maxlen=25)
        self.replay_buffer = deque(maxlen=100)
        self.ledger = set()               # CORRIGIDO: Inicialização preventiva evita erro de atribuição no boot
        self.turn_count = 0
        self.modo_anterior = 0
        self.ultimo_registro_temporal = time.time()
        
        self.lock_estado = threading.RLock()
        
        # Instanciação dos Módulos Integrados
        self.soma = DriveSomático()
        self.cortex = CortexCognitivo()
        self.snc = SistemaNervosoCentral()
        self.atroz = Atrozia()
        self.trabalho = MemoriaTrabalho()
        self.tom = TeoriaDaMente()
        self.tokenizer = re.compile(r'\b\w+\b|[!?.]')
        
        self.nac = NeuralAttentionController(input_dim=5, hidden_dim=12, output_dim=8, lr=0.01)
        self.cnn = CoherenceCNN1D()
        self.accepty = Accepty(threshold=0.20)
        self.anomini = Anomini()
        self.plasticidade = Plasticidade()
        self.delong = Delong()

    def _get_entropy(self, t): 
        return 1.0 / (math.log(self.raridade.get(t, 1) + 1.2) + 1e-5)

    def extrair_triplas_relacionais(self, tokens):
        tipos_relacao = {"é", "tem", "usa", "causa", "vive_em", "precisa"}
        for i in range(len(tokens) - 2):
            sujeito, verbo, objeto = tokens[i], tokens[i+1], tokens[i+2]
            if verbo in tipos_relacao:
                self.relacoes[sujeito][verbo].add(objeto)

    def _ajustar_coordenadas_usuario(self, prompt_tokens):
        """Aproxima fisicamente em 3D as palavras combinadas pelo usuário"""
        for i in range(len(prompt_tokens) - 1):
            w_at, w_px = prompt_tokens[i], prompt_tokens[i+1]
            if w_at in self.coordenadas_palavras and w_px in self.coordenadas_palavras:
                c_at = self.coordenadas_palavras[w_at]
                c_px = self.coordenadas_palavras[w_px]
                self.coordenadas_palavras[w_px] = [
                    c_px[0] + 0.15 * (c_at[0] - c_px[0]),
                    c_px[1] + 0.15 * (c_at[1] - c_px[1]),
                    c_px[2] + 0.15 * (c_at[2] - c_px[2])
                ]

    def _sincronizar_espaco_geometrico(self, ql, resposta_tokens):
        """Sincroniza o espaço geométrico 3D com a resposta do modelo"""
        for w_user in ql:
            for w_model in resposta_tokens:
                if w_user in self.coordenadas_palavras and w_model in self.coordenadas_palavras:
                    c_user = self.coordenadas_palavras[w_user]
                    c_model = self.coordenadas_palavras[w_model]
                    self.coordenadas_palavras[w_model] = [
                        c_model[0] + 0.05 * (c_user[0] - c_model[0]),
                        c_model[1] + 0.05 * (c_user[1] - c_model[1]),
                        c_model[2] + 0.05 * (c_user[2] - c_model[2])
                    ]

    def calcular_distancia_3d(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def avaliar_trajetoria_3d(self, tokens_candidato):
        if len(tokens_candidato) <= 2: return 1.0
        coords = [self.coordenadas_palavras[t] for t in tokens_candidato if t in self.coordenadas_palavras]
        if len(coords) <= 2: return 0.5
        
        distancias = []
        vetores_passo = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i+1]
            passo = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
            norm_passo = math.sqrt(sum(x*x for x in passo))
            distancias.append(norm_passo)
            vetores_passo.append(passo)
            
        mudancas_angulo = []
        for i in range(len(vetores_passo) - 1):
            v1, v2 = vetores_passo[i], vetores_passo[i+1]
            n1 = math.sqrt(sum(x*x for x in v1))
            n2 = math.sqrt(sum(x*x for x in v2))
            if n1 > 0.0 and n2 > 0.0:
                cos_theta = sum(a*b for a, b in zip(v1, v2)) / (n1 * n2)
                mudancas_angulo.append((cos_theta + 1.0) / 2.0)
                
        if distancias:
            mean_d = sum(distancias) / len(distancias)
            variance = sum((x - mean_d)**2 for x in distancias) / len(distancias)
            std_d = math.sqrt(variance)
        else:
            std_d = 0.0
            
        suavidade_velocidade = 1.0 / (1.0 + std_d)
        suavidade_direcao = sum(mudancas_angulo) / len(mudancas_angulo) if mudancas_angulo else 1.0
        return 0.5 * suavidade_velocidade + 0.5 * suavidade_direcao

    # ==================================================================
    # 🧬 [MOTOR DLMC DE GERAÇÃO CRIATIVA DE FRASES]
    # ==================================================================
    def _gerar_candidato_criativo(self, ql, centro_xyz, w_opts):
        # Retorno defensivo para resguardar o motor em caso de falha na migração da matriz
        if not self.matrix_dlm:
            return {"txt": "...", "tokens": ["..."], "gradientes": [0.0] * 8}
            
        w_caos, w_geo, w_sem, w_inercia, w_historico, w_ancora, w_propagacao, w_logico = w_opts
        
        atual = ql[-1] if ql[-1] in self.matrix_dlm else random.choice(list(self.matrix_dlm.keys()))
        resultado = []
        gradientes_acumulados = [0.0] * 8
        
        # Penalização estática mantém o histórico inalterado durante a simulação interna
        rastro_conversacional = set(list(self.history)[-10:])

        for i in range(35):
            if atual not in self.matrix_dlm: break
            resultado.append(atual)

            opcoes = self.matrix_dlm[atual]["links"]
            candidatos, pesos = [], []

            for prox, freq in opcoes.items():
                if prox not in self.matrix_dlm: continue
                
                # Adaptação online da massa transicional Markoviana baseada na utilidade local
                utilidade_local = self.plasticidade.utilidade_sinaptica.get(prox, 0.0)
                massa = (1.5 / (self.raridade[prox] + 1e-5)) * (1.0 + utilidade_local)
                
                caos_ativo = random.uniform(0, self.soma.eixos["tristeza"]) * w_caos
                foco = 1.0 + self.soma.eixos["prazer"]
                
                # Coesão Espacial
                coord_prox = self.coordenadas_palavras.get(prox, [0.0, 0.0, 0.0])
                dist = self.calcular_distancia_3d(coord_prox, centro_xyz)
                atencao_geo = (10.0 / (dist + 1.0)) * w_geo
                
                # Co-ocorrência Semântica
                atencao_sem = math.log1p(sum(self.matrix_dlm[tok]["links"].get(prox, 0) for tok in ql if tok in self.matrix_dlm)) * w_sem
                
                # Âncora Geométrica real-time
                sims_tokens_p = [((self.accepty.similaridade_cosseno(coord_prox, self.coordenadas_palavras[tp]) + 1.0) / 2.0) 
                                 for tp in ql if tp in self.coordenadas_palavras]
                # CORRIGIDO: Substituída chamada do np.mean para list comprehension puro em conformidade com a universalização de matemática nativa
                score_ancora = (sum(sims_tokens_p) / len(sims_tokens_p)) if sims_tokens_p else 0.5
                
                # Bias Lógico-Simbólico
                score_logico = 0.0
                penultimo = resultado[-2] if len(resultado) > 1 else ""
                if atual in self.relacoes:
                    if prox in self.relacoes[atual].get("é", set()): score_logico += 1.5
                    for rel, objs in self.relacoes[atual].items():
                        if prox in objs: score_logico += 1.0
                if penultimo in self.relacoes and atual in self.relacoes[penultimo]:
                    if prox in self.relacoes[penultimo][atual]: score_logico += 1.5

                atencao_total = 1.0 + atencao_geo + atencao_sem + (score_ancora * w_ancora) + (score_logico * w_logico)
                prob = (freq * massa * foco * atencao_total) + caos_ativo
                
                # O rastro impede loops cruzados e autorrepetição imediata
                if prox in rastro_conversacional or prox in resultado:
                    prob *= 0.001 
                    
                candidatos.append(prox)
                pesos.append(prob)

            if not candidatos: break
            escolhido = random.choices(candidatos, weights=pesos, k=1)[0]
            
            if len(candidatos) > 1:
                soma_pesos = sum(pesos) + 1e-8
                prob_rel = pesos[candidatos.index(escolhido)] / soma_pesos
                erro_foco = 1.0 - prob_rel
                
                grad_caos = erro_foco * (w_caos - 0.1)
                grad_geo = -erro_foco * 0.5
                grad_sem = -erro_foco * 0.5
                grad_inercia = -erro_foco * 0.3
                grad_hist = -erro_foco * 0.3
                grad_ancora = -erro_foco * 0.4
                grad_prop = -erro_foco * 0.4
                grad_log = -erro_foco * 0.4
                
                grad_step = [grad_caos, grad_geo, grad_sem, grad_inercia, grad_hist, grad_ancora, grad_prop, grad_log]
                for idx in range(8):
                    gradientes_acumulados[idx] += grad_step[idx]

            atual = escolhido
            if atual == "." and i > 6: break

        norm_factor = len(resultado) + 1e-8
        gradientes_finais = [g / norm_factor for g in gradientes_acumulados]
        return {
            "txt": " ".join(resultado),
            "tokens": resultado,
            "gradientes": gradientes_finais
        }

    # ==================================================================
    # 📝 [SUBSISTEMA 1: AUTOBIOGRAFIA METACOGNITIVA]
    # ==================================================================
    def consolidar_autobiografia(self):
        """
        Analisa os últimos episódios de memória, extrai os nós conceituais de maior relevância
        e gera uma narrativa autorreflexiva em primeira pessoa que descreve seu estado interno.
        """
        if len(self.l2_episodes) < 5: 
            return
            
        recentes = self.l2_episodes[-10:]
        palavras_recentes = []
        for ep in recentes:
            toks = self.tokenizer.findall(NormalizadorSomático.limpar(ep['t']))
            palavras_recentes.extend(toks)
            
        if not palavras_recentes: 
            return
            
        contagem = Counter(palavras_recentes)
        palavras_relevantes = sorted(contagem.keys(), key=lambda t: self._get_entropy(t), reverse=True)[:5]
        if not palavras_relevantes: 
            palavras_relevantes = ["origem"]

        # Força parâmetros de foco interno (baixo caos, alta sinergia e herança lógica)
        w_opts_reflexivos = [0.05, 1.5, 1.2, 0.8, 1.5, 1.2, 1.0, 1.5]
        
        coords_relevantes = [self.coordenadas_palavras[tk] for tk in palavras_relevantes if tk in self.coordenadas_palavras]
        if coords_relevantes:
            n_c = len(coords_relevantes)
            centro_xyz = [
                sum(c[0] for c in coords_relevantes) / n_c,
                sum(c[1] for c in coords_relevantes) / n_c,
                sum(c[2] for c in coords_relevantes) / n_c
            ]
        else:
            centro_xyz = [0.0, 0.0, 0.0]
        
        semente = ["eu"]
        cand = self._gerar_candidato_criativo(semente, centro_xyz, w_opts_reflexivos)
        
        segmento_autobiografia = "eu " + cand["txt"]
        segmento_autobiografia = segmento_autobiografia.replace(" .", ".").replace(" ,", ",").strip()
        
        print(f"\n📝 [CONSOLIDAÇÃO AUTOBIOGRÁFICA]: '{segmento_autobiografia}'")
        self.cristalizar_solo(segmento_autobiografia)

    # ==================================================================
    # 🧬 [AUTO-CURA: RETROCOMPATIBILIDADE E MIGRAÇÃO ONLINE]
    # ==================================================================
    def _sincronizar_migracao_dlm(self):
        """
        Reconstrói automaticamente o espaço geométrico 3D, a matriz DLM e as relações lógicas 
        caso o organismo esteja migrando de uma versão antiga (SSML pura) que não possuía essas estruturas.
        """
        if self.l2_episodes and (not self.matrix_dlm or not self.coordenadas_palavras):
            print("\n⚠️ [MIGRAÇÃO DETECTADA] Reconstruindo espaço 3D, matriz DLM e grafo relacional a partir das memórias legadas...")
            
            # Limpeza preventiva para evitar inconsistência de dimensões
            self.matrix_dlm.clear()
            self.coordenadas_palavras.clear()
            self.relacoes.clear()
            
            # Reconstrói as estruturas sintático-geométricas varrendo as memórias existentes
            for ep in self.l2_episodes:
                f_l = NormalizadorSomático.limpar(ep['t'])
                tokens = self.tokenizer.findall(f_l)
                if len(tokens) < 3: continue
                
                for t in tokens:
                    # Reconstrói coordenadas 3D estáveis via Hashing
                    if t not in self.coordenadas_palavras:
                        h = hashlib.sha256(t.encode()).hexdigest()
                        x = (int(h[:4], 16) % 200) - 100
                        y = (int(h[4:8], 16) % 200) - 100
                        z = (int(h[8:12], 16) % 200) - 100
                        self.coordenadas_palavras[t] = [x, y, z]
                    
                    # Inicializa os nós Markovianos locais
                    if t not in self.matrix_dlm:
                        self.matrix_dlm[t] = {"m": 1.0, "links": Counter()}
                
                # Popula os caminhos de transição
                for j in range(len(tokens) - 1):
                    self.matrix_dlm[tokens[j]]["links"][tokens[j+1]] += 1
                
                # Re-extrai relações simbólicas das memórias carregadas
                self.extrair_triplas_relacionais(tokens)
                
            print(f"✨ [MIGRAÇÃO CONCLUÍDA] Espaço 3D e DLM sincronizados. {len(self.matrix_dlm)} nós de geração reativados.")
            
            # Grava as modificações imediatamente para evitar reconstrução no próximo boot
            self.dormir()

    # ==================================================================
    # ⚙️ [EXECUÇÃO DA CRISTALIZAÇÃO ATÔMICA]
    # ==================================================================
    def cristalizar_solo(self, texto):
        for f in re.split(r'[\.\!\?\n]+', texto):
            f_l = NormalizadorSomático.limpar(f)
            tokens = self.tokenizer.findall(f_l)
            if len(tokens) < 3: continue
            
            idx = len(self.l2_episodes)
            v_ep = {}
            for t in tokens:
                self.raridade[t] += 1
                self.neuronios[t].append(idx)
                
                # 5000D Esparso
                if t not in self.mapa_nd:
                    self.mapa_nd[t] = {i: random.gauss(0, 1) for i in random.sample(range(5000), 100)}
                v_ep = {k: v_ep.get(k,0) + self.mapa_nd[t].get(k,0)*self._get_entropy(t) for k in self.mapa_nd[t]}
                
                # 3D Geométrico
                if t not in self.coordenadas_palavras:
                    h = hashlib.sha256(t.encode()).hexdigest()
                    x = (int(h[:4], 16) % 200) - 100
                    y = (int(h[4:8], 16) % 200) - 100
                    z = (int(h[8:12], 16) % 200) - 100
                    self.coordenadas_palavras[t] = [x, y, z]
                
                # Conexões Markovianas DLM
                if t not in self.matrix_dlm:
                    self.matrix_dlm[t] = {"m": 1.0, "links": Counter()}
                    
            # Registra conexões sequenciais
            for j in range(len(tokens) - 1):
                self.matrix_dlm[tokens[j]]["links"][tokens[j+1]] += 1
                
            self.extrair_triplas_relacionais(tokens)
            self.l2_episodes.append({'t': f.strip(), 'v': KernelRessonante.normalize(v_ep)})

    def processar_gravidade_temporal(self):
        t_atual = time.time()
        dt = t_atual - self.ultimo_registro_temporal
        self.ultimo_registro_temporal = t_atual
        
        self.anomini.atualizar_estados(
            dkl_atual=(self.anomini.cache_otimizacao[-1] if self.anomini.cache_otimizacao else 0.5),
            confianca_tom=self.tom.estimativa_humor["confiança"],
            dt=dt
        )
        if dt > 15.0:
            gravidade = 1.0 - math.exp(-dt / 90.0)
            self.trabalho.aplicar_gravidade_temporal(gravidade)
            self.soma.aplicar_deriva_temporal(gravidade)
            print(f"\n⏳ [GRAVIDADE TEMPORAL] Ócio: {dt:.1f}s. Gravidade: {gravidade:.3f}")

    # ==================================================================
    # 🧠 [ÁREA 9: SISTEMA DE PROCESSAMENTO COGNITIVO INTEGRADO]
    # ==================================================================
    def processar(self, entrada):
        with self.lock_estado:
            self.processar_gravidade_temporal()
            raw = NormalizadorSomático.limpar(entrada)
            u_toks = self.tokenizer.findall(raw)
            if not u_toks: return "..."

            # 🛑 SENSOR DE DELONGAS
            if self.delong.monitorar(u_toks):
                res = self.delong.gerar_pergunta()
                self.history.append(res)
                self.soma.vm = max(-90.0, self.soma.vm - 2.0)
                return res

            # 🛑 RESET DE SEGURANÇA CONTRA FRICÇÃO (ANOMINI)
            if self.anomini.dor["intensidade"] > 3.0 and self.turn_count > 5:
                self.snc._salvar()
                self.soma.eixos["raiva"] = min(5.0, self.soma.eixos["raiva"] + 0.5)
                self.soma.vm = -70.0
                self.anomini.dor["intensidade"] = 0.0
                self.anomini.cache_otimizacao.clear()
                return "[RECALIBRAÇÃO] Minha integridade matemática atingiu um limiar crítico de fricção semântica. Válvula de escape ativada."

            # Consolidação periódica da autobiografia (A cada 25 interações do usuário)
            if self.turn_count > 0 and self.turn_count % 25 == 0:
                self.consolidar_autobiografia()

            t0 = time.perf_counter()
            self.turn_count += 1
            self.soma.metabolizar_decaimento()

            # Ajuste geométrico com o prompt digitado
            self._ajustar_coordenadas_usuario(u_toks)
            
            # Sintonização rápida da CNN sintática
            if len(u_toks) >= 3:
                coords_u = [self.coordenadas_palavras.get(t, [0.0, 0.0, 0.0]) for t in u_toks]
                self.cnn.forward(coords_u)
                self.cnn.backward_step(target=1.0, lr=0.01)

            sujeito = max([t for t in u_toks if t in self.coordenadas_palavras] or [u_toks[0]], key=lambda t: self._get_entropy(t))
            impacto = self._get_entropy(sujeito)
            self.plasticidade.registrar_atividade_sinaptica(u_toks, impacto, self.turn_count)

            # Cálculo de divergência somática de primeira ordem
            q_int = self.snc.estado_anterior[:4]
            p_real = [self.soma.eixos[k] for k in ["amor", "prazer", "tristeza", "raiva"]]
            dkl = self.cortex.calcular_dkl(p_real, q_int)
            
            self.soma.pulsar(impacto, dkl, u_toks, self.turn_count)
            self.tom.atualizar(u_toks, dkl)

            # Vetor de estado dinâmico do NAC (5 inputs)
            estado_nac = [
                (self.soma.vm + 90) / 45.0,                 # Voltagem de Hardware Normalizada
                self.tom.estimativa_humor["confiança"],     # Sinergia / Confiança da Teoria da Mente
                min(1.0, len(u_toks) / 20.0),               # Extensão do Input
                len(self.history) / 25.0,                    # Tamanho do Rastro Histórico
                min(1.0, self.turn_count / 100.0)           # Idade Conversacional
            ]
            w_opts = self.nac.forward(estado_nac)
            
            # Modulação de Literalidade para Perguntas
            e_pergunta = any(t in u_toks for t in ["?", "quem", "quando", "onde", "por", "que", "qual", "como"])
            if e_pergunta:
                w_opts[0] *= 0.15   # w_caos suprimido
                w_opts[2] *= 1.50   # w_sem amplificado
                w_opts[5] *= 1.80   # w_ancora geometricamente reforçado para interrogações
                w_opts[7] *= 1.50   # w_logico amplificado

            # CORRIGIDO: hash_estado instanciado antes da chamada do pulsar_vontade para preservar herança do Q-learning
            hash_estado = self.snc.obter_hash_estado(self.modo_anterior, self.soma.vm, dkl, impacto)

            # Input sensorial do SNC
            x_sen = [min(1.0, self.soma.eixos[k]/5.0) for k in ["amor", "prazer", "tristeza", "raiva"]] + [impacto, (self.soma.vm+90)/45]
            volicao = self.snc.pulsar_vontade(x_sen, exploracao=max(0.02, min(0.4, dkl * 0.3)))
            modo_idx = volicao.index(max(volicao))

            self.replay_buffer.append((x_sen, modo_idx))

            # Codificação do vetor de input
            v_in = {}
            for t in u_toks:
                if t in self.mapa_nd:
                    v_ep_v = self.mapa_nd[t]
                    fator_idade = self.plasticidade.calcular_escala_idade(t, self.turn_count)
                    v_in = {k: v_in.get(k,0) + v_ep_v.get(k,0)*self._get_entropy(t)*fator_idade for k in set(v_in)|set(v_ep_v)}
            v_in = KernelRessonante.normalize(v_in)
            
            self.trabalho.registrar(v_in, u_toks, modo_idx, self.soma.eixos)
            v_smooth = self.trabalho.vetor_suavizado
            damping = self.atroz.amortecer_loop(v_smooth)

            # Encontra o centróide espacial 3D do melhor bloco conceitual do prompt
            bloco_coords = [self.coordenadas_palavras[t] for t in u_toks if t in self.coordenadas_palavras]
            if bloco_coords:
                n_b = len(bloco_coords)
                centro_xyz = [
                    sum(c[0] for c in bloco_coords) / n_b,
                    sum(c[1] for c in bloco_coords) / n_b,
                    sum(c[2] for c in bloco_coords) / n_b
                ]
            else:
                centro_xyz = [0.0, 0.0, 0.0]

            # -------------------------------------------------------------
            # ANTECIPAÇÃO CONTEXTUAL (PROJEÇÃO DE COGNITIVE CODING)
            # -------------------------------------------------------------
            h_antecipado = self.plasticidade.prever_antecipacao()

            # -------------------------------------------------------------
            # FUSÃO: POOL MULTICANDIDATO (EPISÓDICOS + CRIAÇÕES DO MODELO)
            # -------------------------------------------------------------
            pool_candidatos = []
            
            # 1. Recuperação Episódica (L2 Episodes de SSML)
            indices_episodicos = self.neuronios.get(sujeito, []) or random.sample(range(len(self.l2_episodes)), min(len(self.l2_episodes), 50))
            for ep_idx in indices_episodicos:
                ep = self.l2_episodes[ep_idx]
                if ep['t'] in self.atroz.historico_absoluto or ep['t'] in self.history: continue
                pool_candidatos.append({
                    "txt": ep['t'],
                    "tokens": self.tokenizer.findall(NormalizadorSomático.limpar(ep['t'])),
                    "v": ep['v'],
                    "gradientes": None 
                })

            # 2. Geração Criativa em Tempo Real (DLMC)
            for _ in range(30):
                c_criado = self._gerar_candidato_criativo(u_toks, centro_xyz, w_opts)
                if not c_criado["tokens"]: continue
                v_cand_sparse = {}
                for t in c_criado["tokens"]:
                    if t in self.mapa_nd:
                        for k, val in self.mapa_nd[t].items():
                            v_cand_sparse[k] = v_cand_sparse.get(k, 0.0) + val * self._get_entropy(t)
                c_criado["v"] = KernelRessonante.normalize(v_cand_sparse)
                pool_candidatos.append(c_criado)

            # -------------------------------------------------------------
            # AVALIAÇÃO E SELEÇÃO COGNITIVA SOBERANA
            # -------------------------------------------------------------
            scored_candidates = []
            sentido_conversa = self.plasticidade.processar_sentido_conversa(p_real + [impacto, dkl])

            for cand in pool_candidatos:
                # Métricas de Alta Dimensão (SSML)
                s_q = KernelRessonante.tsallis_match(v_smooth, cand["v"])
                sim_f = KernelRessonante.dot(v_smooth, cand["v"])
                sinergia = self.atroz.calcular_sinergia(cand["v"])
                
                # Alinhamento sintático GRU (Passado e Presente)
                cand_pseudo = [cand["v"].get(i % 5000, 0.0) for i in range(len(sentido_conversa))]
                alinhamento_gru = sum(s_c * s_g for s_c, s_g in zip(cand_pseudo, sentido_conversa))
                
                # Alinhamento Proativo (Comparação com a projeção transicional GRU)
                alinhamento_antecipado = sum(s_c * s_a for s_c, s_a in zip(cand_pseudo, h_antecipado))

                # Métricas Sintático-Geométricas (DLMC)
                coords_cand = [self.coordenadas_palavras.get(t, [0.0, 0.0, 0.0]) for t in cand["tokens"]]
                score_cnn = self.cnn.forward(coords_cand)
                score_trajetoria = self.avaliar_trajetoria_3d(cand["tokens"])
                passou_accepty, score_accepty = self.accepty.avaliar(u_toks, cand["tokens"], self.coordenadas_palavras)

                # Equação Unificada de Decisão com Antecipação Futura
                score_final = (
                    (s_q * 0.15) +              # Fidelidade Tsallis
                    (sim_f * 0.10) +            # Atração Conceitual
                    (sinergia * 0.15) +         # Sinergia Cortical (Atrozia)
                    (alinhamento_gru * 0.15) +  # Trajetória Semântica Passada
                    (alinhamento_antecipado * 0.15) + # Alinhamento de Projeção Futura (Predictive Coding)
                    (score_cnn * 0.15) +        # Filtro Sintático CNN
                    (score_accepty * 0.05) +    # Filtro de Integridade Accepty
                    (score_trajetoria * 0.10)   # Estabilidade Trajetorial 3D
                )
                
                if modo_idx == 2 and any(x in cand["txt"].lower() for x in ["amo", "prazer"]): 
                    score_final -= 2.0 
                if not passou_accepty:
                    score_final *= 0.5

                scored_candidates.append((cand, score_final * damping))

            if not scored_candidates: 
                return "Reorganizando conexões..."

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            vencedor, score_vencedor = scored_candidates[0]
            resposta_final = vencedor["txt"]

            # Se o vencedor foi gerado dinamicamente, cristaliza-se em episódio
            if vencedor["gradientes"] is not None:
                self.cristalizar_solo(resposta_final)
                self.nac.backward(vencedor["gradientes"]) 

            # Atração geométrica mútua (Mirror Loop)
            self._sincronizar_espaco_geometrico(u_toks, vencedor["tokens"])

            # Feedback de Reforço TD no SNC
            novo_q_int = self.snc.estado_anterior[:4]
            novo_dkl = self.cortex.calcular_dkl(p_real, novo_q_int)
            self.plasticidade.adaptar_gru_local(p_real + [impacto, novo_dkl])
            
            recompensa_total = ((dkl - novo_dkl) * 8.0) + (self.tom.estimativa_humor["confiança"] * 4.0) - (self.tom.estimativa_humor["agressividade"] * 3.0)
            if self.soma.vm > -52.0: recompensa_total -= 3.0

            hash_prox_estado = self.snc.obter_hash_estado(modo_idx, self.soma.vm, novo_dkl, impacto)
            self.snc.aplicar_recompensa_td(hash_estado, modo_idx, recompensa_total, hash_prox_estado)

            self.atroz.historico_absoluto.append(resposta_final)
            self.atroz.last_v_vencedor = vencedor["v"]
            
            if dkl < 0.45 or self.soma.simbiose > 0.6:
                self.snc.adaptar_realtime([1.0 if i == modo_idx else 0.0 for i in range(3)])

            self.history.append(resposta_final)
            self.modo_anterior = modo_idx
            self.plasticidade.metabolizar_decaimento_sinaptico()

            dt = (time.perf_counter() - t0) * 1000
            tipo_cand = "CRIADO" if vencedor["gradientes"] is not None else "EPISÓDICO"
            print(f" ⚛️ [v75.5] Tipo:{tipo_cand} | Simbiose:{self.soma.simbiose:.2f} | Nós DLM:{len(self.matrix_dlm)} | {dt:.1f}ms")
            return resposta_final

    def boot(self):
        if os.path.exists(self.path_bin):
            try:
                with open(self.path_bin, 'rb') as f:
                    d = pickle.load(f)
                    self.l2_episodes, self.raridade, self.mapa_nd = d['nexus'], d['raridade'], d['nd']
                    self.coordenadas_palavras = d.get('coordenadas_palavras', {})
                    self.matrix_dlm = d.get('matrix_dlm', {})
                    self.replay_buffer = deque(d.get('replay_buffer', []), maxlen=100)
                    self.plasticidade.utilidade_sinaptica = defaultdict(float, d.get('utilidade_sinaptica', {}))
                    self.plasticidade.ultimo_uso_token = defaultdict(int, d.get('ultimo_uso_token', {}))
                    
                    self.atroz.historico_absoluto = deque(d.get('atroz_hist', []), maxlen=200)
                    
                    rel_carregado = d.get('relacoes', {})
                    self.relacoes = defaultdict(lambda: defaultdict(set))
                    for k, v in rel_carregado.items():
                        for kk, vv in v.items():
                            self.relacoes[k][kk] = set(vv)
                            
                    # CONVERSOR DE MATRIZES LEGADAS DO NUMPY PARA LISTAS NATIVAS
                    if 'nac_W1' in d:
                        W1_raw = d['nac_W1']
                        self.nac.W1 = W1_raw.tolist() if hasattr(W1_raw, 'tolist') else W1_raw
                        b1_raw = d['nac_b1']
                        self.nac.b1 = b1_raw.tolist() if hasattr(b1_raw, 'tolist') else b1_raw
                        W2_raw = d['nac_W2']
                        self.nac.W2 = W2_raw.tolist() if hasattr(W2_raw, 'tolist') else W2_raw
                        b2_raw = d['nac_b2']
                        self.nac.b2 = b2_raw.tolist() if hasattr(b2_raw, 'tolist') else b2_raw
                        
                    if 'cnn_W_conv' in d:
                        W_conv_raw = d['cnn_W_conv']
                        self.cnn.W_conv = W_conv_raw.tolist() if hasattr(W_conv_raw, 'tolist') else W_conv_raw
                        b_conv_raw = d['cnn_b_conv']
                        self.cnn.b_conv = b_conv_raw.tolist() if hasattr(b_conv_raw, 'tolist') else b_conv_raw
                        W_dense_raw = d['cnn_W_dense']
                        self.cnn.W_dense = W_dense_raw.tolist() if hasattr(W_dense_raw, 'tolist') else W_dense_raw
                        b_dense_raw = d['cnn_b_dense']
                        if hasattr(b_dense_raw, 'tolist'):
                            b_dense_list = b_dense_raw.tolist()
                            self.cnn.b_dense = b_dense_list[0][0] if isinstance(b_dense_list, list) and isinstance(b_dense_list[0], list) else (b_dense_list[0] if isinstance(b_dense_list, list) else b_dense_list)
                        else:
                            self.cnn.b_dense = b_dense_raw
            except Exception as e:
                # Auto-cura contra arquivos corrompidos ou em branco (0 bytes) do cérebro
                print(f"⚠️ [AVISO] Falha ao decodificar nucleo_organismo.qssml ({e}). Iniciando com novo organismo.")

        # Executa sincronização e migração se viemos de um banco legado sem DLM
        self._sincronizar_migracao_dlm()

        if os.path.exists(self.path_ledger):
            try:
                with open(self.path_ledger, 'rb') as f: 
                    self.ledger = pickle.load(f)
            except Exception as e:
                # Auto-cura contra arquivos de log corrompidos ou em branco (0 bytes)
                print(f"⚠️ [AVISO] O ledger.bin está vazio ou corrompido ({e}). Reinicializando registro limpo.")
                self.ledger = set()
            
        for arq in self.auto_train_files:
            if os.path.exists(arq):
                with open(arq, 'r', encoding='utf-8') as f:
                    c = f.read()
                    h = hashlib.sha256(c.encode()).hexdigest()
                    if h not in self.ledger: 
                        self.cristalizar_solo(c)
                        self.ledger.add(h)
                        
        self.neuronios.clear()
        for i, ep in enumerate(self.l2_episodes):
            for t in self.tokenizer.findall(NormalizadorSomático.limpar(ep['t'])): 
                self.neuronios[t].append(i)
        
        self.ultimo_registro_temporal = time.time()
        print(f"✅ Organismo Sincronizado v75.5 [Híbrido Neuro-Simbólico Nativo]. Episódios: {len(self.l2_episodes)}")

    def dormir(self):
        # Consolidação profunda gera uma autobiografia reflexiva antes de descarregar na persistência
        self.consolidar_autobiografia()

        if len(self.replay_buffer) > 5:
            print("🧠 [CONSOLIDAÇÃO PLÁSTICA LTP]")
            amostra_treino = random.sample(self.replay_buffer, min(len(self.replay_buffer), 20))
            for x_sen, modo_real_idx in amostra_treino:
                self.snc.pulsar_vontade(x_sen)
                target = [0.0] * 3
                target[modo_real_idx] = 1.0
                self.snc.adaptar_realtime(target)
                
        self.plasticidade.aplicar_morte_sinaptica(self.mapa_nd, self.neuronios, self.l2_episodes, self.raridade)
        self.snc._salvar()
        
        relacoes_serializaveis = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.relacoes.items()}
        
        with open(self.path_bin, 'wb') as f:
            pickle.dump({
                'nexus': self.l2_episodes, 'raridade': self.raridade, 'nd': self.mapa_nd,
                'coordenadas_palavras': self.coordenadas_palavras,
                'matrix_dlm': self.matrix_dlm,
                'relacoes': relacoes_serializaveis,
                'replay_buffer': list(self.replay_buffer),
                'utilidade_sinaptica': dict(self.plasticidade.utilidade_sinaptica),
                'ultimo_uso_token': dict(self.plasticidade.ultimo_uso_token),
                'atroz_hist': list(self.atroz.historico_absoluto), 
                'nac_W1': self.nac.W1, 'nac_b1': self.nac.b1, 'nac_W2': self.nac.W2, 'nac_b2': self.nac.b2,
                'cnn_W_conv': self.cnn.W_conv, 'cnn_b_conv': self.cnn.b_conv, 'cnn_W_dense': self.cnn.W_dense, 'cnn_b_dense': self.cnn.b_dense
            }, f)
            
        with open(self.path_ledger, 'wb') as f: pickle.dump(self.ledger, f)

# ==================================================================
# ⏱️ RELÓGIO ATIVO - DAEMON THREAD EM SEGUNDO PLANO
# ==================================================================
def loop_relogio_endogeno(organismo, stop_event):
    while not stop_event.is_set():
        time.sleep(5.0) 
        t_ocioso = time.time() - organismo.ultimo_registro_temporal
        
        if t_ocioso > 15.0:
            gravidade_passiva = 1.0 - math.exp(-5.0 / 90.0)
            
            # Lock não-bloqueante impede lag ou disputa de thread no terminal do usuário
            if organismo.lock_estado.acquire(blocking=False):
                try:
                    organismo.trabalho.aplicar_gravidade_temporal(gravidade_passiva)
                    organismo.soma.aplicar_deriva_temporal(gravidade_passiva)
                    organismo.anomini.atualizar_estados(
                        dkl_atual=(organismo.anomini.cache_otimizacao[-1] if organismo.anomini.cache_otimizacao else 0.5),
                        confianca_tom=organismo.tom.estimativa_humor["confiança"],
                        dt=5.0
                    )
                    
                    if organismo.soma.eixos["tristeza"] > 3.5:
                        organismo.soma.eixos["tristeza"] *= 0.5
                        print(f"\n🧠 [SURTO ESPONTÂNEO DE TRISTEZA COGNITIVA]")
                        print(f"🧠: {organismo.processar('solidão')}")
                    elif organismo.soma.vm > -55.0:
                        print(f"\n🧠 [DESCARGA DE VOLTAGEM CRÍTICA Vm]")
                        print(f"🧠: {organismo.processar('origem')}")
                finally:
                    organismo.lock_estado.release()

# ==================================================================
# DEPLOY QUINTIKUS COGNITIVE FUSION ARCHITECTURE
# ==================================================================
if __name__ == "__main__":
    organismo = OrganismoSoberano()
    organismo.boot()
    
    #stop_relogio = threading.Event()
    #thread_tempo = threading.Thread(target=loop_relogio_endogeno, args=(organismo, stop_relogio), daemon=True)
    #thread_tempo.start()
    
    print("="*90)
    print("QUINTIKUS DSML v75.5 - COGNITIVE FUSION DEPLOYED")
    print("="*90)
    
    try:
        while True:
            entrada = input("\n👤: ")
            if entrada.strip().lower() == "dormir" or entrada.strip().lower() == "sair" :
                organismo.dormir()
                print("💤 [SNC SALVO] Consolidação de pesos concluída.")
                break
            
            resposta = organismo.processar(entrada)
            print(f"🧠: {resposta}")
            
    except KeyboardInterrupt:
        print("\n🛑 Salvando estado atômico...")
        organismo.dormir()
