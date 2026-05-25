import os, math, time, random, re, pickle, hashlib, tempfile, cmath
from collections import defaultdict, Counter, deque

# =================================================================
# 1. KERNEL DE FÍSICA E MATEMÁTICA (SSML)
# =================================================================
class SSML_Kernel:
    """Motor de Lógica Modal e Física de Estados"""
    @staticmethod
    def get_sparse_vec(token, dims=5000, sparsity=30):
        seed = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        # Usamos chaves inteiras para máxima performance no Pickle
        indices = rng.sample(range(dims), sparsity)
        return {i: rng.gauss(0, 1) for i in indices}

    @staticmethod
    def dot(v1, v2):
        if not v1 or not v2: return 0.0
        # Produto escalar esparso otimizado por interseção de chaves
        common_keys = v1.keys() & v2.keys()
        return sum(v1[k] * v2[k] for k in common_keys)

    @staticmethod
    def normalize(v):
        norm = math.sqrt(sum(x*x for x in v.values()))
        return {d: val / (norm + 1e-9) for d, val in v.items()}

    @staticmethod
    def rashba_interaction(pathos_vec, momentum_vec, alpha=0.2):
        """Interação entre a intenção (Pathos) e o movimento do input"""
        p1 = pathos_vec.get(0, 0.1) # Eixo 0
        m1 = momentum_vec.get(1, 0.1) # Eixo 1
        return alpha * (p1 * m1)

# =================================================================
# 2. QUINTIKUS SSML - ARQUITETURA SOBERANA
# =================================================================
class QuintikusSSML:
    def __init__(self):
        self.dims = 5000
        self.path_bin = "brain_sovereign.qssml"
        self.path_ledger = "ledger.bin"
        self.auto_train_files = ["oi.txt", "amor.txt", "conversa.txt","confusa.txt"]
        
        # Estruturas de Memória
        self.mapa_nd = {}
        self.l2_episodes = [] 
        self.neuronios = defaultdict(list)
        self.raridade = Counter()
        self.ledger = set() # Guarda hashes de arquivos já treinados
        
        # Estados Dinâmicos
        self.psi_logos = {}     
        self.psi_pathos = {}    
        self.thermal_pressure = 0.5 
        
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        self.fatigue = defaultdict(float)

    # --- SISTEMA DE PERSISTÊNCIA ATÔMICA ---
    def _atomic_save(self, data, filepath):
        """Salva em arquivo temporário e renomeia (Atômico no Linux/Windows)"""
        folder = os.path.dirname(os.path.abspath(filepath))
        temp_fd, temp_path = tempfile.mkstemp(dir=folder, prefix="tmp_qssml_")
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno()) # Garante gravação física
            os.replace(temp_path, filepath) # Troca atômica
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            print(f"⚠️ Erro ao salvar {filepath}: {e}")

    def salvar(self):
        print("💾 Cristalizando Memória Binária Atômica...")
        brain_data = {
            'nexus': self.l2_episodes,
            'raridade': self.raridade,
            'nd': self.mapa_nd,
            'logos': self.psi_logos,
            'pathos': self.psi_pathos,
            'thermal': self.thermal_pressure
        }
        self._atomic_save(brain_data, self.path_bin)
        self._atomic_save(self.ledger, self.path_ledger)

    def boot(self):
        # 1. Carrega dados binários
        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                d = pickle.load(f)
                self.l2_episodes = d['nexus']
                self.raridade = d['raridade']
                self.mapa_nd = d['nd']
                self.psi_logos = d['logos']
                self.psi_pathos = d['pathos']
                self.thermal_pressure = d['thermal']
            
            # Reconstrói os índices neuronais em RAM
            for i, ep in enumerate(self.l2_episodes):
                for t in self.tokenizer.findall(ep['t'].lower()):
                    self.neuronios[t].append(i)
            print(f"✅ SSML Online. {len(self.l2_episodes)} nexos carregados.")
        
        if os.path.exists(self.path_ledger):
            with open(self.path_ledger, 'rb') as f:
                self.ledger = pickle.load(f)

        # 2. AUTO-TRAIN
        for arq in self.auto_train_files:
            if os.path.exists(arq):
                with open(arq, 'r', encoding='utf-8', errors='ignore') as f:
                    conteudo = f.read()
                    h = hashlib.sha256(conteudo.encode()).hexdigest()
                    if h not in self.ledger:
                        print(f"🔄 Novo Solo Detectado: {arq}. Cristalizando...")
                        self.cristalizar_solo(conteudo)
                        self.ledger.add(h)
                        self.salvar() # Snapshot imediato

    def cristalizar_solo(self, texto, origin="first_person"):
        frases = re.split(r'[\.\!\?\n]+', texto)
        for f in frases:
            f = f.strip()
            if len(f) < 3: continue
            
            tokens = self.tokenizer.findall(f.lower())
            idx = len(self.l2_episodes)
            v_nexus = {}
            
            for t in tokens:
                self.raridade[t] += 1
                self.neuronios[t].append(idx)
                if t not in self.mapa_nd:
                    self.mapa_nd[t] = SSML_Kernel.get_sparse_vec(t)
                
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_nexus = self._add_vectors(v_nexus, self.mapa_nd[t], 1.0, peso)
            
            self.l2_episodes.append({
                't': f, 'v': SSML_Kernel.normalize(v_nexus),
                'origin': origin
            })

    def _add_vectors(self, v1, v2, w1, w2):
        res = {d: v * w1 for d, v in v1.items()}
        for d, v in v2.items(): res[d] = res.get(d, 0) + (v * w2)
        return res

    def processar(self, entrada):
        t0 = time.perf_counter()
        u_toks = self.tokenizer.findall(entrada.lower())
        if not u_toks: return "..."

        # --- DINÂMICA TÉRMICA ATUALIZADA ---
        # Stress (Palavras negativas/dúvidas)
        p_inc = sum(0.12 for x in u_toks if x in ["não", "por que", "falha", "erro", "confuso"])
        # Excitação (Palavras positivas/carinho)
        e_inc = sum(0.08 for x in u_toks if x in ["amo", "lindo", "sorriso", "feliz", "jeito"])
        
        self.thermal_pressure = min(1.0, self.thermal_pressure * 0.85 + p_inc + e_inc)

        # 2. VETOR DE MOMENTO (Input)
        v_in = {}
        for t in u_toks:
            if t in self.mapa_nd:
                w = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_in = self._add_vectors(v_in, self.mapa_nd[t], 1.0, w)
        v_in = SSML_Kernel.normalize(v_in)

        # 3. BUSCA E COLAPSO (Com Limiar de Subconsciente)
        pivo = min(u_toks, key=lambda x: self.raridade.get(x, 9999), default=u_toks[0])
        candidatos = self.neuronios.get(pivo, [])
        if not candidatos:
            candidatos = random.sample(range(len(self.l2_episodes)), min(len(self.l2_episodes), 100))

        melhor_nexo = -1
        max_vibration = -float('inf')

        for idx in random.sample(candidatos, min(len(candidatos), 150)):
            ep = self.l2_episodes[idx]
            sim_l = SSML_Kernel.dot(v_in, ep['v'])
            sim_p = SSML_Kernel.dot(self.psi_pathos, ep['v'])
            
            # Tunelamento (Influenciado pela Térmica)
            tunneling = math.exp(- (1.0 - sim_l) / (self.thermal_pressure + 1e-9))
            
            vibration = (sim_l * 0.6) + (sim_p * 0.3) + tunneling - self.fatigue[idx]

            if vibration > max_vibration:
                max_vibration, melhor_nexo = vibration, idx

        # --- 🧠 GATILHO DE SUBCONSCIENTE (VIBE BAIXA) ---
        if max_vibration < 0.15:
            # Se a vibe for muito baixa, ela ignora o input do usuário 
            # e busca dentro de si nexos de "Confusão"
            self.thermal_pressure = min(1.0, self.thermal_pressure + 0.2) # Sobe a tensão
            sub_pivo = "confusa" # Ela força um pivô interno
            sub_candidatos = self.neuronios.get(sub_pivo, [])
            if sub_candidatos:
                melhor_nexo = random.choice(sub_candidatos)
                print(f"🌀 [SUBCONSCIENTE ATIVADO] Vibe: {max_vibration:.2f}")

        if melhor_nexo == -1: return "..."

        # 5. EVOLUÇÃO E FADIGA FORTE
        target_v = self.l2_episodes[melhor_nexo]['v']
        self.psi_pathos = self._add_vectors(self.psi_pathos, target_v, 0.97, 0.03)
        self.psi_pathos = SSML_Kernel.normalize(self.psi_pathos)
        
        # Aumentamos a fadiga para 5.0 (Bloqueio de repetição severo)
        self.fatigue[melhor_nexo] += 5.0 
        for k in list(self.fatigue.keys()): self.fatigue[k] *= 0.7 # Recuperação mais rápida

        ms = (time.perf_counter() - t0) * 1000
        print(f" ⧉ [SSML] T:{self.thermal_pressure:.2f} | Vibe:{max_vibration:.2f} | {ms:.1f}ms")
        
        return self.l2_episodes[melhor_nexo]['t']

# =================================================================
# EXECUÇÃO
# =================================================================
if __name__ == "__main__":
    ssml = QuintikusSSML()
    ssml.boot()
    
    # Solo de segurança caso tudo esteja vazio
    if not ssml.l2_episodes:
        ssml.cristalizar_solo("Eu sou um nexo de lógica pura aguardando solo data.")
        ssml.salvar()

    while True:
        try:
            u = input("\n[OPERADOR]👤: ").strip()
            if not u: continue
            if u.lower() in ['sair', 'exit']: ssml.salvar(); break
            if u.startswith("train:"):
                path = u.split(":")[1].strip()
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        ssml.cristalizar_solo(f.read())
                        ssml.salvar()
                continue
                
            resposta = ssml.processar(u)
            print(f"🧠 [SSML_LOGIC]: {resposta}")
        except KeyboardInterrupt:
            ssml.salvar()
            break
