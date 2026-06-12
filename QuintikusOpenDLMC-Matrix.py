# -*-coding:utf8;-*-
import hashlib, math, random, time, sys, os, pickle
from collections import defaultdict, deque, Counter
import numpy as np
#class 2 dlm 
# =====================================================================
# 1. COMPONENTE NEURAL: CONTROLADOR DE ATENÇÃO (ADAM - 8 OUTS)
# =====================================================================
class NeuralAttentionController:
    """MLP Metacognitiva que calibra 8 parâmetros incluindo o peso de Lógica Simbólica"""
    def __init__(self, input_dim=5, hidden_dim=12, output_dim=8, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        
        # Inicialização Xavier
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Otimizador Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

    def forward(self, x):
        self.x = np.array(x).reshape(1, -1)
        self.h = np.maximum(0, np.dot(self.x, self.W1) + self.b1) # ReLU
        self.out_raw = np.dot(self.h, self.W2) + self.b2
        self.out = np.log1p(np.exp(np.clip(self.out_raw, -20, 20))) # Softplus
        return self.out[0]

    def backward(self, d_out):
        d_out = d_out.reshape(1, -1)
        sigmoid_raw = 1.0 / (1.0 + np.exp(-np.clip(self.out_raw, -20, 20)))
        d_out_raw = d_out * sigmoid_raw
        
        dW2 = np.dot(self.h.T, d_out_raw)
        db2 = d_out_raw
        
        dh = np.dot(d_out_raw, self.W2.T)
        dh[self.h <= 0] = 0
        
        dW1 = np.dot(self.x.T, dh)
        db1 = dh
        
        self.t += 1
        
        for param, grad, m, v in [
            (self.W1, dW1, self.m_W1, self.v_W1),
            (self.b1, db1, self.m_b1, self.v_b1),
            (self.W2, dW2, self.m_W2, self.v_W2),
            (self.b2, db2, self.m_b2, self.v_b2)
        ]:
            m[...] = self.beta1 * m + (1.0 - self.beta1) * grad
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# =====================================================================
# 2. COMPONENTE NEURAL: CNN 1D DE COERÊNCIA SINTÁTICA (16 FILTROS)
# =====================================================================
class CoherenceCNN1D:
    def __init__(self, sequence_length=15, embedding_dim=3):
        self.seq_len = sequence_length
        self.emb_dim = embedding_dim
        self.num_filters = 16 
        self.filter_width = 3
        
        self.W_conv = np.random.randn(self.filter_width, self.emb_dim, self.num_filters) * np.sqrt(2.0 / (self.filter_width * self.emb_dim))
        self.b_conv = np.zeros((1, self.num_filters))
        self.W_dense = np.random.randn(self.num_filters, 1) * np.sqrt(2.0 / self.num_filters)
        self.b_dense = np.zeros((1, 1))

    def forward(self, sentence_coords):
        if len(sentence_coords) < self.seq_len:
            pad_size = self.seq_len - len(sentence_coords)
            sentence_coords = sentence_coords + [[0.0, 0.0, 0.0]] * pad_size
        else:
            sentence_coords = sentence_coords[:self.seq_len]
            
        self.X = np.array(sentence_coords) 
        conv_out_len = self.seq_len - self.filter_width + 1
        self.conv_features = np.zeros((conv_out_len, self.num_filters))
        
        for i in range(conv_out_len):
            window = self.X[i:i+self.filter_width, :]
            for f in range(self.num_filters):
                self.conv_features[i, f] = np.sum(window * self.W_conv[:, :, f]) + self.b_conv[0, f]
        
        self.conv_activated = np.maximum(0, self.conv_features)
        
        self.pooled = np.zeros((1, self.num_filters))
        self.pool_indices = []
        for f in range(self.num_filters):
            idx = np.argmax(self.conv_activated[:, f])
            self.pooled[0, f] = self.conv_activated[idx, f]
            self.pool_indices.append(idx)
            
        logits = np.dot(self.pooled, self.W_dense) + self.b_dense
        self.score = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -20, 20)))
        return self.score

    def backward_step(self, target, lr=0.01):
        d_logits = self.score - target
        dW_dense = self.pooled.T * d_logits
        db_dense = np.array([[d_logits]])
        
        d_pooled = d_logits * self.W_dense.T 
        
        dW_conv = np.zeros_like(self.W_conv)
        db_conv = np.zeros_like(self.b_conv)
        
        for f in range(self.num_filters):
            idx_max = self.pool_indices[f]
            if self.conv_features[idx_max, f] > 0:
                d_val = d_pooled[0, f]
                window = self.X[idx_max:idx_max+self.filter_width, :]
                dW_conv[:, :, f] += window * d_val
                db_conv[0, f] += d_val
                
        self.W_conv -= lr * dW_conv
        self.b_conv -= lr * db_conv
        self.W_dense -= lr * dW_dense
        self.b_dense -= lr * db_dense


# =====================================================================
# 3. FILTRO DE VALIDAÇÃO DE CONTEXTO E INTENÇÃO (ACCEPTY)
# =====================================================================
class Accepty:
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def extrair_sujeito_predicado(self, tokens):
        if len(tokens) <= 1:
            return tokens, tokens
        meio = max(1, len(tokens) // 2)
        return tokens[:meio], tokens[meio:]

    def calcular_vetor_medio(self, tokens, coordenadas_palavras):
        coords = [coordenadas_palavras[t] for t in tokens if t in coordenadas_palavras]
        if not coords:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(coords, axis=0)

    def similaridade_cosseno(self, v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0: 
            return 0.0
        return np.dot(v1, v2) / (n1 * n2)

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

        return passou, score_integridade, sim_sujeito, sim_predicado


# =====================================================================
# 4. MOTOR CENTRAL MATRIX DLMC V70 ULTRA NEURO-SIMBÓLICA
# =====================================================================
class Quintikus_DLMC_V70:
    def __init__(self, raw_text, brain_file="matrix_brain.bin"):
        self.brain_file = brain_file
        self.matrix = {} 
        self.blocos = [] 
        self.estados = [0.5, 0.5] 
        self.rastro = deque(maxlen=100)
        self.termometro = {'erro': -0.3, 'falha': -0.2, 'ruído': -0.1, 'bom': 0.2, 'sinergia': 0.3, 'paz': 0.2}
        self.coordenadas_palavras = {} 
        
        self.memoria_historica_prompts = deque(maxlen=5) 
        self.memoria_conhecimento_sequencias = [] 
        self.ativacao_atual = {} 
        
        # --- ATRIBUTOS DE INFRAESTRUTURA LÓGICA DO NDLM ---
        self.relacoes = defaultdict(lambda: defaultdict(set))
        self.tipos_relacao = {"é", "tem", "usa", "causa", "vive_em", "precisa"}

        # 1. PRÉ-PROCESSAMENTO DLM
        self.tokens = raw_text.lower().replace(".", " . ").replace(",", " , ").split()
        self.build_dlm_matrix()
        
        # Extrai relações explícitas e induz classes por covariância (Nível 4 do NDLM)
        self.extrair_triplas_relacionais(self.tokens)
        self._consolidar_abstracoes_automaticas()

        # 2. SISTEMAS NEURAIS INTEGRADOS (NAC configurada com 8 saídas para calibrar Lógica)
        self.nac = NeuralAttentionController(input_dim=5, hidden_dim=12, output_dim=8, lr=0.01)
        self.cnn = CoherenceCNN1D()
        self.accepty = Accepty(threshold=0.20) 
        
        cerebro_carregado = self.load_brain()
        if not cerebro_carregado:
            self.treinar_cnn_com_corpus()

    def get_id_geometrico(self, tokens_bloco):
        raras = sorted(tokens_bloco, key=lambda x: self.matrix.get(x, {"m":0})["m"], reverse=True)[:5]
        h = hashlib.sha256(" ".join(raras).encode()).hexdigest()
        x = (int(h[:4], 16) % 200) - 100
        y = (int(h[4:8], 16) % 200) - 100
        z = (int(h[8:12], 16) % 200) - 100
        return h[:4], [x, y, z]

    def build_dlm_matrix(self):
        freq = Counter(self.tokens)
        for t in freq:
            self.matrix[t] = {"m": 1.5 / (freq[t] + 1e-5), "links": Counter()}

        for i in range(len(self.tokens) - 1):
            self.matrix[self.tokens[i]]["links"][self.tokens[i+1]] += 1

        tamanho_bloco = 256
        for i in range(0, len(self.tokens), tamanho_bloco):
            bloco_tokens = self.tokens[i:i+tamanho_bloco]
            if not bloco_tokens: continue
            id_b, xyz = self.get_id_geometrico(bloco_tokens)
            self.blocos.append({"id": id_b, "xyz": xyz, "txt": bloco_tokens})

        temp_coords = defaultdict(list)
        for bloco in self.blocos:
            for token in bloco["txt"]:
                temp_coords[token].append(bloco["xyz"])
        
        for token, lista_xyz in temp_coords.items():
            xs = [p[0] for p in lista_xyz]
            ys = [p[1] for p in lista_xyz]
            zs = [p[2] for p in lista_xyz]
            self.coordenadas_palavras[token] = [sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)]

        for i in range(len(self.tokens) - 15):
            seq = self.tokens[i:i+15]
            vec_medio = self.calcular_vetor_medio(seq)
            if not np.all(vec_medio == 0.0):
                self.memoria_conhecimento_sequencias.append(vec_medio)

        print(f"🧬 Matrix DLMC Ativa: {len(self.matrix)} nós | {len(self.blocos)} blocos de massa.")
        print(f"💾 Dimensões de Sequência Carregadas: {len(self.memoria_conhecimento_sequencias)} vetores estruturais.")

    # =====================================================================
    # EXTRATORES DE LÓGICA E ABSTRAÇÃO (PROPRIEDADES DO NDLM)
    # =====================================================================
    def extrair_triplas_relacionais(self, tokens):
        """Mapeia triplas lógicas explícitas da estrutura: Entidade -> Relação -> Atributo"""
        for i in range(len(tokens) - 2):
            sujeito = tokens[i]
            verbo = tokens[i+1]
            objeto = tokens[i+2]
            if verbo in self.tipos_relacao:
                self.relacoes[sujeito][verbo].add(objeto)

    def _consolidar_abstracoes_automaticas(self):
        """Induz classes taxonômicas por covariância lógica de atributos"""
        entidades = list(self.relacoes.keys())
        for i in range(len(entidades)):
            ent_a = entidades[i]
            targets_a = set()
            for rel, objs in self.relacoes[ent_a].items():
                targets_a.update(objs)
            if not targets_a: continue
            
            for j in range(len(entidades)):
                if i == j: continue
                ent_b = entidades[j]
                targets_b = set()
                for rel, objs in self.relacoes[ent_b].items():
                    targets_b.update(objs)
                if not targets_b: continue
                
                comum = targets_a.intersection(targets_b)
                if len(comum) >= 2 and (len(comum) / len(targets_a)) >= 0.5:
                    massa_a = self.matrix.get(ent_a, {}).get("m", 999)
                    massa_b = self.matrix.get(ent_b, {}).get("m", 999)
                    if massa_b < massa_a:
                        if ent_b not in self.relacoes[ent_a]["é"]:
                            self.relacoes[ent_a]["é"].add(ent_b)
                            print(f"✨ [Abstração Automática] Induzido por covariância: {ent_a} é {ent_b}")

    def treinar_cnn_com_corpus(self, passos_treino=1000):
        print("⚙️ Iniciando treinamento de sequenciamento lógico da CNN...")
        tempo_inicio = time.time()
        for _ in range(passos_treino):
            idx = random.randint(0, len(self.tokens) - 16)
            tokens_reais = self.tokens[idx:idx+15]
            coords_reais = [self.coordenadas_palavras.get(t, [0.0, 0.0, 0.0]) for t in tokens_reais]
            self.cnn.forward(coords_reais)
            self.cnn.backward_step(target=1.0, lr=0.005)

            tokens_falsos = list(tokens_reais)
            random.shuffle(tokens_falsos)
            coords_falsas = [self.coordenadas_palavras.get(t, [0.0, 0.0, 0.0]) for t in tokens_falsos]
            self.cnn.forward(coords_falsas)
            self.cnn.backward_step(target=0.0, lr=0.005)
            
        print(f"✅ Treinamento concluído em {time.time() - tempo_inicio:.2f}s.")
        self.save_brain()

    def load_brain(self):
        if os.path.exists(self.brain_file):
            try:
                with open(self.brain_file, 'rb') as f:
                    data = pickle.load(f)
                    
                    nac_compativel = (
                        'nac_W1' in data and data['nac_W1'].shape == self.nac.W1.shape and
                        'nac_W2' in data and data['nac_W2'].shape == self.nac.W2.shape
                    )
                    cnn_compativel = (
                        'cnn_W_conv' in data and data['cnn_W_conv'].shape == self.cnn.W_conv.shape and
                        'cnn_W_dense' in data and data['cnn_W_dense'].shape == self.cnn.W_dense.shape
                    )

                    if nac_compativel and cnn_compativel:
                        self.nac.W1 = data['nac_W1']
                        self.nac.b1 = data['nac_b1']
                        self.nac.W2 = data['nac_W2']
                        self.nac.b2 = data['nac_b2']
                        self.nac.t = data.get('nac_t', 0)
                        
                        self.cnn.W_conv = data['cnn_W_conv']
                        self.cnn.b_conv = data['cnn_b_conv']
                        self.cnn.W_dense = data['cnn_W_dense']
                        self.cnn.b_dense = data['cnn_b_dense']
                        print("🧠 Cérebro Unificado e padrões CNN 16D carregados com sucesso.")
                        return True
                    else:
                        print("⚠️ Modificação na arquitetura física detectada. Reiniciando novo cérebro.")
                        return False
            except Exception as e:
                print(f"⚠️ Erro ao decodificar binário: {e}. Inicializando pesos padrão.")
                return False
        return False

    def save_brain(self):
        try:
            temp_file = self.brain_file + ".tmp"
            data = {
                'nac_W1': self.nac.W1, 'nac_b1': self.nac.b1,
                'nac_W2': self.nac.W2, 'nac_b2': self.nac.b2,
                'nac_t': self.nac.t,
                'cnn_W_conv': self.cnn.W_conv, 'cnn_b_conv': self.cnn.b_conv,
                'cnn_W_dense': self.cnn.W_dense, 'cnn_b_dense': self.cnn.b_dense
            }
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)
            os.replace(temp_file, self.brain_file)
        except Exception as e:
            print(f"⚠️ Falha ao salvar persistência do Cérebro: {e}")

    def atualizar_termica(self, tokens_in):
        for t in tokens_in:
            if t in self.termometro:
                val = self.termometro[t]
                if val < 0: self.estados[0] = min(1.0, self.estados[0] + abs(val))
                else: self.estados[1] = min(1.0, self.estados[1] + val)
        self.estados = [s * 0.95 for s in self.estados]

    def calcular_distancia_3d(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def calcular_vetor_medio(self, tokens):
        coords = [self.coordenadas_palavras[t] for t in tokens if t in self.coordenadas_palavras]
        if not coords:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(coords, axis=0)

    def similaridade_cosseno(self, v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0: return 0.0
        return np.dot(v1, v2) / (n1 * n2)

    def avaliar_trajetoria_3d(self, tokens_candidato):
        if len(tokens_candidato) <= 2:
            return 1.0 
            
        coords = [self.coordenadas_palavras[t] for t in tokens_candidato if t in self.coordenadas_palavras]
        if len(coords) <= 2:
            return 0.5

        distâncias = []
        vetores_passo = []
        for i in range(len(coords) - 1):
            p1, p2 = np.array(coords[i]), np.array(coords[i+1])
            passo = p2 - p1
            distâncias.append(np.linalg.norm(passo))
            vetores_passo.append(passo)

        mudanças_angulo = []
        for i in range(len(vetores_passo) - 1):
            v1, v2 = vetores_passo[i], vetores_passo[i+1]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_theta = np.dot(v1, v2) / (n1 * n2)
                mudanças_angulo.append((cos_theta + 1.0) / 2.0)

        suavidade_velocidade = 1.0 / (1.0 + np.std(distâncias))
        suavidade_direcao = np.mean(mudanças_angulo) if mudanças_angulo else 1.0
        
        return 0.5 * suavidade_velocidade + 0.5 * suavidade_direcao

    def propagar_ativacao_relacional(self, ql, passos=3, decaimento=0.5):
        ativacao = {t: 0.0 for t in self.matrix}
        for t in ql:
            if t in ativacao:
                ativacao[t] = 1.0
                
        for _ in range(passos):
            nova_ativacao = {t: 0.0 for t in self.matrix}
            for node, energia in ativacao.items():
                if energia > 0.0:
                    links = self.matrix[node]["links"]
                    soma_links = sum(links.values())
                    if soma_links > 0:
                        for vizinho, freq in links.items():
                            if vizinho in nova_ativacao:
                                nova_ativacao[vizinho] += energia * (freq / soma_links) * decaimento
            
            for node in ativacao:
                ativacao[node] = min(1.0, ativacao[node] + nova_ativacao[node])
                
        self.ativacao_atual = ativacao

    # =====================================================================
    # GERAÇÃO DE CANDIDATOS COM BIAS DE LÓGICA SUAVE (NEURO-SIMBÓLICA)
    # =====================================================================
    def _gerar_candidato_solitario(self, prompt, ql, centro_xyz, w_caos, w_geo, w_sem, w_ancora, w_propagacao, w_logico):
        atual = ql[-1] if ql[-1] in self.matrix else random.choice(
            max(self.blocos, key=lambda b: len(set(ql).intersection(b["txt"])), default=self.blocos[0])["txt"]
        )
        
        resultado = []
        gradientes_acumulados = np.zeros((1, 8)) # Gradientes expandidos para 8 dimensões
        rastro_local = list(self.rastro)

        for i in range(40):
            if atual not in self.matrix: break
            resultado.append(atual)
            rastro_local.append(atual)
            if len(rastro_local) > 10: rastro_local.pop(0)

            opcoes = self.matrix[atual]["links"]
            candidatos, pesos = [], []

            for prox, freq in opcoes.items():
                if prox not in self.matrix: continue
                massa = self.matrix[prox]["m"]
                
                caos_ativo = random.uniform(0, self.estados[0]) * w_caos
                foco = 1.0 + self.estados[1]
                
                # Atenção Geométrica
                coord_prox = self.coordenadas_palavras.get(prox, [0, 0, 0])
                dist = self.calcular_distancia_3d(coord_prox, centro_xyz)
                atencao_geo = (10.0 / (dist + 1.0)) * w_geo
                
                # Atenção Semântica tradicional
                atencao_semantica = 0.0
                for token_prompt in ql:
                    if token_prompt in self.matrix:
                        atencao_semantica += self.matrix[token_prompt]["links"].get(prox, 0)
                atencao_semantica = math.log1p(atencao_semantica) * w_sem

                # Âncora Real-Time Token a Token
                sims_tokens_prompt = []
                for token_p in ql:
                    if token_p in self.coordenadas_palavras:
                        sim = self.similaridade_cosseno(coord_prox, self.coordenadas_palavras[token_p])
                        sims_tokens_prompt.append((sim + 1.0) / 2.0)
                score_ancora_realtime = np.mean(sims_tokens_prompt) if sims_tokens_prompt else 0.5

                # Ativação Relacional (Spreading Activation)
                score_propagacao = self.ativacao_atual.get(prox, 0.0)

                # -------------------------------------------------------------
                # BIAS DE REGRAS LÓGICAS E HERANÇA TRANSITIVA SUAVE (NDLM)
                # -------------------------------------------------------------
                score_logico = 0.0
                penultimo = resultado[-2] if len(resultado) > 1 else ""
                
                if atual in self.relacoes:
                    # Direta
                    for rel, objs in self.relacoes[atual].items():
                        if prox in objs:
                            score_logico += 1.5
                    # Transitiva
                    if "é" in self.relacoes[atual]:
                        for classe_b in self.relacoes[atual]["é"]:
                            if classe_b in self.relacoes:
                                for rel_b, objs_b in self.relacoes[classe_b].items():
                                    if prox in objs_b:
                                        score_logico += 1.0
                                        
                # Objeto Direto / Consequência causal
                if penultimo in self.relacoes and atual in self.relacoes[penultimo]:
                    if prox in self.relacoes[penultimo][atual]:
                        score_logico += 1.5

                # Terminação Dinâmica de Sentença (Nível 3)
                if prox == "." and len(resultado) >= 3:
                    sub_at = resultado[-3]
                    ver_at = resultado[-2]
                    obj_at = resultado[-1]
                    tem_vinculo = False
                    if sub_at in self.relacoes and ver_at in self.relacoes[sub_at] and obj_at in self.relacoes[sub_at][ver_at]:
                        tem_vinculo = True
                    elif sub_at in self.relacoes and "é" in self.relacoes[sub_at]:
                        for cl in self.relacoes[sub_at]["é"]:
                            if cl in self.relacoes and ver_at in self.relacoes[cl] and obj_at in self.relacoes[cl][ver_at]:
                                tem_vinculo = True
                                break
                    if tem_vinculo:
                        score_logico += 2.0 # Empurrão probabilístico suave para pontuar e fechar a relação

                # Equação Final unificada
                atencao_total = (
                    1.0 + atencao_geo + atencao_semantica + 
                    (score_ancora_realtime * w_ancora) + 
                    (score_propagacao * w_propagacao) +
                    (score_logico * w_logico)
                )
                prob = (freq * massa * foco * atencao_total) + caos_ativo
                
                if prox in rastro_local: prob *= 0.001 

                candidatos.append(prox)
                pesos.append(prob)

            if not candidatos: break
            escolhido = random.choices(candidatos, weights=pesos, k=1)[0]
            
            # Ajuste de gradiente para as 8 dimensões
            if len(candidatos) > 1:
                soma_pesos = sum(pesos) + 1e-8
                prob_relativa = pesos[candidatos.index(escolhido)] / soma_pesos
                erro_foco = 1.0 - prob_relativa
                
                d_caos = erro_foco * (w_caos - 0.1)
                d_geo = -erro_foco * 0.5
                d_sem = -erro_foco * 0.5
                d_inercia = -erro_foco * 0.3
                d_hist = -erro_foco * 0.3
                d_ancora = -erro_foco * 0.4 
                d_propagacao = -erro_foco * 0.4
                d_logico = -erro_foco * 0.4 # Gradiente para calibrar o uso de regras simbólicas
                gradientes_acumulados += np.array([d_caos, d_geo, d_sem, d_inercia, d_hist, d_ancora, d_propagacao, d_logico])

            atual = escolhido
            if atual == "." and i > 10: break

        return {
            "txt": resultado,
            "gradientes": gradientes_acumulados / (len(resultado) + 1e-8)
        }

    def pensar(self, prompt, num_candidatos=50):
        ql = prompt.lower().split()
        qs = set(ql)
        self.atualizar_termica(ql)

        melhor_bloco = max(self.blocos, key=lambda b: len(qs.intersection(b["txt"])), default=self.blocos[0])
        centro_xyz = melhor_bloco["xyz"]
        
        self.propagar_ativacao_relacional(ql, passos=3, decaimento=0.50)
        
        vetor_prompt_atual = self.calcular_vetor_medio(ql)
        self.memoria_historica_prompts.append(vetor_prompt_atual)
        
        fluxo_conversacional_vec = np.mean(list(self.memoria_historica_prompts), axis=0)

        estado_vetor = [
            self.estados[0],                        
            self.estados[1],                        
            min(1.0, len(ql) / 20.0),               
            len(self.rastro) / 100.0,               
            min(1.0, len(self.memoria_historica_prompts) / 5.0) 
        ]
        
        # Extrai os 8 parâmetros gerados dinamicamente (incluindo w_logico)
        w_caos, w_geo, w_sem, w_inercia, w_historico, w_ancora, w_propagacao, w_logico = self.nac.forward(estado_vetor)
        
        pool_candidatos = []
        candidatos_aprovados = []

        for _ in range(num_candidatos):
            cand = self._gerar_candidato_solitario(prompt, ql, centro_xyz, w_caos, w_geo, w_sem, w_ancora, w_propagacao, w_logico)
            if not cand["txt"]: continue
            
            passou, score_accepty, sim_suj, sim_pred = self.accepty.avaliar(ql, cand["txt"], self.coordenadas_palavras)
            
            coords_frase = [self.coordenadas_palavras.get(t, [0.0, 0.0, 0.0]) for t in cand["txt"]]
            
            score_cnn = self.cnn.forward(coords_frase)
            score_trajetoria = self.avaliar_trajetoria_3d(cand["txt"])
            
            vetor_candidato = self.calcular_vetor_medio(cand["txt"])
            
            if self.memoria_conhecimento_sequencias:
                similaridades = [self.similaridade_cosseno(vetor_candidato, v_mem) for v_mem in self.memoria_conhecimento_sequencias]
                score_memoria_txt = max(similaridades) if similaridades else 0.0
            else:
                score_memoria_txt = 0.5
                
            score_historico = (self.similaridade_cosseno(fluxo_conversacional_vec, vetor_candidato) + 1.0) / 2.0

            score_final = (
                (0.20 * score_accepty) + 
                (0.20 * score_cnn) + 
                (0.20 * score_trajetoria * w_inercia) + 
                (0.20 * score_memoria_txt) + 
                (0.20 * score_historico * w_historico)
            )
            
            cand["score"] = score_final
            cand["accepty_pass"] = passou
            cand["debug_info"] = (sim_suj, sim_pred, score_cnn, score_memoria_txt)
            
            pool_candidatos.append(cand)
            if passou:
                candidatos_aprovados.append(cand)

        if candidatos_aprovados:
            melhor_candidato = max(candidatos_aprovados, key=lambda c: c["score"])
            status_validacao = "Aprovado"
        else:
            melhor_candidato = max(pool_candidatos, key=lambda c: c["score"])
            status_validacao = "Forçado"
        
        texto_final = melhor_candidato["txt"]

        if melhor_candidato["accepty_pass"]:
            self.nac.backward(melhor_candidato["gradientes"][0])
            self.save_brain()

        for t in texto_final:
            self.rastro.append(t)

        sim_suj, sim_pred, scn, smem = melhor_candidato["debug_info"]
        prefixo = f"[VAL:{status_validacao} S:{sim_suj:.2f}|P:{sim_pred:.2f} | CNN:{scn:.2f}|Mem_Txt:{smem:.2f}] "
        
        return f"{prefixo}{' '.join(texto_final).capitalize()}."


# =====================================================================
# 5. CARREGAMENTO E EXECUÇÃO
# =====================================================================
if __name__ == "__main__":

    with open('amor.txt', 'r', encoding='utf-8') as arquivo:
        conteudo = arquivo.read()

    motor = Quintikus_DLMC_V70(conteudo)

    print("="*85)
    print("QUINTIKUS DLMC V70 ULTRA NEURO-SIMBÓLICA")
    print("="*85)
    
    while True:
        try:
            p = input("\nINPUT > ")
            if p.lower() in ['sair', 'exit']: break
            
            res = motor.pensar(p, num_candidatos=5)
            
            sys.stdout.write("Processando Matrix: ")
            for char in res:
                sys.stdout.write(char); sys.stdout.flush()
                time.sleep(0.005)
            print(f"\n[Status T:{motor.estados[0]:.2f} | S:{motor.estados[1]:.2f}]")
            
        except KeyboardInterrupt: 
            break
