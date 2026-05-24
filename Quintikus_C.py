import os, math, time, random, re, json, hashlib, cmath
from collections import defaultdict, Counter, deque

# =================================================================
# 1. KERNEL DE FÍSICA UNIFICADA (ESTÁVEL)
# =================================================================
class UnifiedPhysics:
    @staticmethod
    def get_deterministic_vec(token, dims, sparsity=25):
        seed = int(hashlib.md5(token.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        # JSON exige chaves como strings, mas manteremos int internamente para cálculo
        return {i: rng.gauss(0, 1) for i in rng.sample(range(dims), sparsity)}

    @staticmethod
    def dot(v1, v2):
        if len(v1) > len(v2): v1, v2 = v2, v1
        return sum(val * v2.get(str(dim), 0) if isinstance(dim, int) else val * v2.get(int(dim), 0) 
                   for dim, val in v1.items())

    @staticmethod
    def normalize(v):
        norm = math.sqrt(sum(x*x for x in v.values()))
        if norm < 1e-9: return v
        inv_norm = 1.0 / norm
        return {dim: val * inv_norm for dim, val in v.items()}

    @staticmethod
    def add(v1, v2, w1=1.0, w2=1.0):
        res = defaultdict(float)
        for d, v in v1.items(): res[int(d)] = v * w1
        for d, v in v2.items(): res[int(d)] += v * w2
        return dict(res)

# =================================================================
# 2. SISTEMAS COGNITIVOS
# =================================================================
class IntentDetector:
    @staticmethod
    def detectar(texto):
        texto = texto.lower()
        if any(w in texto for w in ["?", "como", "quê", "onde", "por que"]): return "question"
        if any(w in texto for w in ["oi", "olá", "e ai", "bom dia"]): return "greeting"
        return "statement"

class QuintikusClucy:
    def __init__(self):
        self.path_brain = "brain_data.json"
        self.path_user = "user_config.json"
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        self.dims = 5000
        
        # --- CONFIGURAÇÃO DE AUTO-TREINO ---
        self.auto_train_list = ["nome.txt", "amor.txt"]
        
        # Memória
        self.mapa_nd = {}
        self.raridade = Counter()
        self.l2_episodes = [] 
        self.neuronios = defaultdict(list)
        
        # Psique
        self.lobes = {"logos": {}}
        self.emotional_state = {"calm": 0.8, "joy": 0.2, "attachment": 0.5}
        self.z_contexto_raw = [1.0, 0.0] # Para o JSON
        
        # Contexto
        self.context_history = deque(maxlen=8)
        self.pil_user = 0.0
        self.user_name = "Operador"
        self.ledger = [] # Lista para JSON (convertida em set no boot)
        self.fatigue_map = defaultdict(float)

    def pensar_e_falar(self, entrada):
        t0 = time.perf_counter()
        u_toks = self.tokenizer.findall(entrada.lower())
        if not u_toks: return "..."

        # Aprendizado em tempo real
        self.amadurecer_solo(entrada, silenciar=True)

        v_io = {}
        for t in u_toks:
            if t not in self.mapa_nd: self.mapa_nd[t] = UnifiedPhysics.get_deterministic_vec(t, self.dims)
            w = 1.0 / (math.log(self.raridade.get(t, 1) + 1.2) + 1e-5)
            v_io = UnifiedPhysics.add(v_io, self.mapa_nd[t], 1.0, w)
        v_io = UnifiedPhysics.normalize(v_io)

        v_work = v_io
        for v_past in self.context_history: v_work = UnifiedPhysics.add(v_work, v_past, 1.0, 0.3)
        v_work = UnifiedPhysics.normalize(v_work)

        # Busca Multi-Pivô
        candidate_pool = set()
        sorted_tokens = sorted(u_toks, key=lambda x: self.raridade.get(x, 0))
        for t in sorted_tokens[:3]:
            candidate_pool.update(self.neuronios.get(t, []))
        
        if not candidate_pool: return "Tensão no solo. Nexo não mapeado."

        melhor_idx = -1; max_score = -float('inf')
        amostra = random.sample(list(candidate_pool), min(len(candidate_pool), 100))
        
        for i in amostra:
            ep = self.l2_episodes[i]
            # No JSON, as chaves do vetor podem virar strings, o Kernel lida com isso
            dot_in = UnifiedPhysics.dot(v_work, ep['vector'])
            score = dot_in - self.fatigue_map[i]
            if score > max_score:
                max_score, melhor_idx = score, i

        if melhor_idx == -1: return "Energia dissipada."

        self.context_history.append(self.l2_episodes[melhor_idx]['vector'])
        self.fatigue_map[melhor_idx] += 3.0
        
        dt = (time.perf_counter() - t0) * 1000
        print(f"⏱️ {dt:.1f}ms | PIL: {self.pil_user:.2f} | Pivo: {sorted_tokens[0]}")
        return self.l2_episodes[melhor_idx]['text']

    def amadurecer_solo(self, texto, pil_min=0.0, silenciar=False):
        hash_c = hashlib.sha256(texto.encode('utf-8')).hexdigest()
        if hash_c in self.ledger: return False
        
        if not silenciar: print(f"🌌 Cristalizando Solo ({texto[:20]}...)")
        frases = re.split(r'[\.\!\?\n]+', texto)
        count = 0
        for f in frases:
            f = f.strip()
            if len(f) < 3: continue
            tokens = self.tokenizer.findall(f.lower())
            idx = len(self.l2_episodes)
            v_ep = {}
            for t in tokens:
                self.raridade[t] += 1
                self.neuronios[t].append(idx)
                if t not in self.mapa_nd: self.mapa_nd[t] = UnifiedPhysics.get_deterministic_vec(t, self.dims)
                w = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_ep = UnifiedPhysics.add(v_ep, self.mapa_nd[t], 1.0, w)
            self.l2_episodes.append({'text': f, 'vector': UnifiedPhysics.normalize(v_ep), 'pil_min': pil_min})
            count += 1
        self.ledger.append(hash_c)
        return True

    def salvar(self):
        # Salva Configurações do Usuário
        user_data = {
            'name': self.user_name,
            'pil': self.pil_user,
            'ledger': list(self.ledger),
            'emotions': self.emotional_state,
            'z_ctx': self.z_contexto_raw
        }
        with open(self.path_user, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=4)
        
        # Salva o Cérebro (Massivo)
        brain_data = {
            'mass': self.l2_episodes,
            'rar': dict(self.raridade),
            'nd': self.mapa_nd
        }
        with open(self.path_brain, 'w', encoding='utf-8') as f:
            json.dump(brain_data, f)
        print("💾 Nexos salvos em JSON.")

    def boot(self):
        # 1. Carrega dados do usuário
        if os.path.exists(self.path_user):
            with open(self.path_user, 'r', encoding='utf-8') as f:
                d = json.load(f)
                self.user_name = d.get('name', "Operador")
                self.pil_user = d.get('pil', 0.0)
                self.ledger = d.get('ledger', [])
                self.emotional_state = d.get('emotions', self.emotional_state)
                self.z_contexto_raw = d.get('z_ctx', [1.0, 0.0])
        
        # 2. Carrega memória cerebral
        if os.path.exists(self.path_brain):
            with open(self.path_brain, 'r', encoding='utf-8') as f:
                d = json.load(f)
                self.l2_episodes = d.get('mass', [])
                self.raridade = Counter(d.get('rar', {}))
                self.mapa_nd = d.get('nd', {})
                # Reindexar neurônios
                self.neuronios.clear()
                for i, ep in enumerate(self.l2_episodes):
                    for t in self.tokenizer.findall(ep['text'].lower()):
                        self.neuronios[t].append(i)

        # 3. AUTO-TREINO DE BOOT
        print("⚙️ Verificando Auto-Treino...")
        treinou_algo = False
        for file_path in self.auto_train_list:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    if self.amadurecer_solo(f.read(), silenciar=True):
                        print(f"  [+] Novo conhecimento extraído de: {file_path}")
                        treinou_algo = True
        if treinou_algo: self.salvar()

        print(f"✅ Clucy v81.0 Online. Solo: {len(self.l2_episodes)} nexos.")
        return True

if __name__ == "__main__":
    clucy = QuintikusClucy()
    clucy.boot()
    while True:
        try:
            u = input(f"[{clucy.user_name}]👤: ").strip()
            if not u: continue
            if u.lower() in ['sair', 'exit', 'tchau']: 
                clucy.salvar()
                break
            if u.lower() == 'salvar':
                clucy.salvar()
                continue
            
            res = clucy.pensar_e_falar(u)
            print(f"🧠 CLUCY: {res}")
        except Exception as e: print(f"⚠️ Erro: {e}")
