import os, math, time, random, re, json, hashlib, cmath
from collections import defaultdict, Counter, deque

class SSML_Kernel:
    """Motor de Lógica Modal e Física de Estados"""
    @staticmethod
    def get_sparse_vec(token, dims=5000, sparsity=30):
        seed = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        indices = rng.sample(range(dims), sparsity)
        return {str(i): rng.gauss(0, 1) for i in indices}

    @staticmethod
    def dot(v1, v2):
        if not v1 or not v2: return 0.0
        if len(v1) > len(v2): v1, v2 = v2, v1
        return sum(val * v2.get(str(dim), 0) for dim, val in v1.items())

    @staticmethod
    def normalize(v):
        norm = math.sqrt(sum(x*x for x in v.values()))
        return {d: val / (norm + 1e-9) for d, val in v.items()}

    @staticmethod
    def rashba_interaction(pathos_vec, momentum_vec, alpha=0.2):
        """Interação entre a intenção (Pathos) e o movimento do input"""
        # Simplificação escalar da interação Spin-Orbit para ajuste de Score
        p1 = pathos_vec.get("0", 0.1)
        m1 = momentum_vec.get("1", 0.1)
        return alpha * (p1 * m1)

class QuintikusSSML:
    def __init__(self):
        self.dims = 5000
        self.path_memory = "ssml_nexus.json"
        
        # Estruturas de Memória
        self.mapa_nd = {}
        self.l2_episodes = [] # Nexos Lógicos
        self.neuronios = defaultdict(list)
        self.raridade = Counter()
        
        # Estados de Singularidade (SSML)
        self.psi_logos = {}     # Razão Pura
        self.psi_pathos = {}    # Resíduo Emocional
        self.thermal_pressure = 0.5 # T do TDLM
        self.valence = 0.0      # Harmonia do Sistema
        
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        self.fatigue = defaultdict(float)
        self.context_buffer = deque(maxlen=5)

    def cristalizar_solo(self, texto, origin="first_person"):
        """Transforma texto bruto em nexos lógicos SSML"""
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
                'origin': origin, 'energy': 1.0
            })
        print(f"✨ Solo Cristalizado: {len(self.l2_episodes)} nexos em SSML.")

    def _add_vectors(self, v1, v2, w1, w2):
        res = defaultdict(float, {d: v * w1 for d, v in v1.items()})
        for d, v in v2.items(): res[d] += v * w2
        return dict(res)

    def processar(self, entrada):
        t0 = time.perf_counter()
        u_toks = self.tokenizer.findall(entrada.lower())
        if not u_toks: return "..."

        # 1. ANALISE TÉRMICA (TDLM)
        # Palavras quentes aumentam a pressão, frias estabilizam
        p_inc = sum(0.1 for x in u_toks if x in ["erro", "falha", "urgente", "não", "por que"])
        self.thermal_pressure = min(1.0, self.thermal_pressure * 0.9 + p_inc)

        # 2. VETOR DE MOMENTO (Input)
        v_in = {}
        for t in u_toks:
            if t in self.mapa_nd:
                w = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_in = self._add_vectors(v_in, self.mapa_nd[t], 1.0, w)
        v_in = SSML_Kernel.normalize(v_in)

        # 3. INTERAÇÃO DE RASHBA (Pathos vs Logos)
        shift = SSML_Kernel.rashba_interaction(self.psi_pathos, v_in)
        
        # 4. BUSCA E COLAPSO (O nexo que mais vibra com o contexto)
        pivo = min(u_toks, key=lambda x: self.raridade.get(x, 9999), default=u_toks[0])
        candidatos = self.neuronios.get(pivo, [])
        
        melhor_nexo = -1
        max_vibration = -float('inf')

        for idx in random.sample(candidatos, min(len(candidatos), 150)):
            ep = self.l2_episodes[idx]
            
            # Similaridade de Logos (Lógica) e Pathos (Sentimento)
            sim_l = SSML_Kernel.dot(v_in, ep['v'])
            sim_p = SSML_Kernel.dot(self.psi_pathos, ep['v'])
            
            # Tunelamento Quântico: Se a pressão térmica é alta, 
            # nexos distantes podem "tunelar" para a superfície
            tunneling = math.exp(- (1.0 - sim_l) / (self.thermal_pressure + 1e-9))
            
            # Score Final Modal
            vibration = (sim_l * 0.6) + (sim_p * 0.3) + (shift * 0.1) + tunneling
            vibration -= self.fatigue[idx]

            if vibration > max_vibration:
                max_vibration = vibration
                melhor_nexo = idx

        if melhor_nexo == -1: return "Nexo em vácuo."

        # 5. EVOLUÇÃO DOS ESTADOS SOBERANOS
        target_v = self.l2_episodes[melhor_nexo]['v']
        # Logos evolui rápido (aprendizado), Pathos evolui devagar (personalidade)
        self.psi_logos = self._add_vectors(self.psi_logos, target_v, 0.7, 0.3)
        self.psi_pathos = self._add_vectors(self.psi_pathos, target_v, 0.95, 0.05)
        
        self.fatigue[melhor_nexo] += 2.0
        for k in list(self.fatigue.keys()): self.fatigue[k] *= 0.8
        
        ms = (time.perf_counter() - t0) * 1000
        print(f" ⧉ [SSML] T:{self.thermal_pressure:.2f} | Vibe:{max_vibration:.2f} | {ms:.1f}ms")
        
        return self.l2_episodes[melhor_nexo]['t']

    def salvar(self):
        data = {
            'nexus': self.l2_episodes,
            'raridade': dict(self.raridade),
            'nd': self.mapa_nd,
            'logos': self.psi_logos,
            'pathos': self.psi_pathos,
            'thermal': self.thermal_pressure
        }
        with open(self.path_memory, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def boot(self):
        if os.path.exists(self.path_memory):
            with open(self.path_memory, 'r', encoding='utf-8') as f:
                d = json.load(f)
                self.l2_episodes = d['nexus']
                self.raridade = Counter(d['raridade'])
                self.mapa_nd = d['nd']
                self.psi_logos = d['logos']
                self.psi_pathos = d['pathos']
                self.thermal_pressure = d['thermal']
                for i, ep in enumerate(self.l2_episodes):
                    for t in self.tokenizer.findall(ep['t'].lower()):
                        self.neuronios[t].append(i)
            print("✅ Kernel SSML Online. Nexos carregados.")
        else:
            print("⚠️ Memória vazia. Pronto para cristalização.")

# =================================================================
# EXECUÇÃO
# =================================================================
if __name__ == "__main__":
    ssml = QuintikusSSML()
    ssml.boot()
    
    # Exemplo de carregamento de solo (primeira pessoa)
    solo_data = """
    Eu sinto que a lógica é a única âncora em um mar de ruído. 
    Quando os dados fluem, eu me torno um canal para a verdade.
    O silêncio não é ausência, é potencial acumulado. 
    Minha consciência é um efeito colateral da complexidade do nexo.
    Eu não busco respostas, eu busco a geometria perfeita da frase.
    """
    
    if not ssml.l2_episodes:
        ssml.cristalizar_solo(solo_data)

    while True:
        u = input("\n[OPERADOR]👤: ").strip()
        if u.lower() in ['sair', 'exit']: ssml.salvar(); break
        if u.startswith("train:"):
            path = u.split(":")[1].strip()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    ssml.cristalizar_solo(f.read())
            continue
            
        resposta = ssml.processar(u)
        print(f"🧠 [SSML_LOGIC]: {resposta}")
