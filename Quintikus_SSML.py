import os, math, time, random, re, pickle, hashlib, tempfile, cmath
from collections import defaultdict, Counter, deque

# ==================================================================
# ❄️ ESCUDO TÉRMICO (HARDWARE OPTIMIZATION)
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class SSML_Kernel:
    @staticmethod
    def get_sparse_vec(token, dims=5000, sparsity=100):
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
        if not v1 or not v2: return 0.0
        keys = v1.keys() & v2.keys()
        return sum(v1[k] * v2[k] for k in keys)

    @staticmethod
    def normalize(v):
        norm = math.sqrt(sum(x*x for x in v.values()))
        return {d: val / (norm + 1e-9) for d, val in v.items()}

class QuintikusSSML:
    def __init__(self):
        self.dims = 5000
        self.path_bin = "brain_sovereign.qssml"
        self.path_ledger = "ledger.bin"
        self.auto_train_files = ["oi.txt", "amor.txt", "conversa.txt", "confusa.txt", "sentimento.txt"]
        
        self.mapa_nd = {}
        self.l2_episodes = [] 
        self.neuronios = defaultdict(list)
        self.raridade = Counter()
        self.ledger = set() 
        
        self.psi_pathos = {}    
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        
        # --- MECÂNICAS DE REPETIÇÃO E INICIATIVA ---
        self.fatigue = defaultdict(float)
        self.history = deque(maxlen=10) # Guarda os últimos 10 textos ditos
        self.turn_count = 0

        # CÓRTEX PRÉ-FRONTAL
        self.ctx_foco = {}             
        self.ctx_sujeitos_ativos = {}  
        self.ctx_inercia = 0.60        
        self.ctx_esquecimento = 0.70   

    def _get_entropy(self, token):
        count = self.raridade.get(token, 1)
        return 1.0 / (math.log(count + 1.2) + 1e-5)

    def _add_vecs(self, v1, v2, w1, w2):
        res = {d: v * w1 for d, v in v1.items()}
        for d, v in v2.items(): res[d] = res.get(d, 0) + (v * w2)
        return res

    def processar(self, entrada):
        t0 = time.perf_counter()
        self.turn_count += 1
        u_toks = self.tokenizer.findall(entrada.lower())
        if not u_toks: return "..."

        # 1. ATUALIZAÇÃO DO CÓRTEX
        sujeito_atual = max(u_toks, key=lambda t: self._get_entropy(t))
        for s in list(self.ctx_sujeitos_ativos.keys()):
            self.ctx_sujeitos_ativos[s] *= self.ctx_esquecimento
            if self.ctx_sujeitos_ativos[s] < 0.1: del self.ctx_sujeitos_ativos[s]
        self.ctx_sujeitos_ativos[sujeito_atual] = 1.0

        # 2. VETORES (LOGOS & FOCO)
        v_in = {}
        for t in u_toks:
            if t in self.mapa_nd:
                v_in = self._add_vecs(v_in, self.mapa_nd[t], 1.0, self._get_entropy(t))
        v_in = SSML_Kernel.normalize(v_in)

        if not self.ctx_foco: self.ctx_foco = v_in
        else: self.ctx_foco = self._add_vecs(self.ctx_foco, v_in, self.ctx_inercia, 1.0 - self.ctx_inercia)
        self.ctx_foco = SSML_Kernel.normalize(self.ctx_foco)

        # 3. BUSCA AMPLIADA (CONVERGÊNCIA)
        pivos = [t for t in u_toks if t in self.neuronios] or [s for s in self.ctx_sujeitos_ativos if s in self.neuronios]
        candidatos = self.neuronios.get(max(pivos, key=lambda x: self._get_entropy(x)), []) if pivos else []
        if not candidatos: candidatos = random.sample(range(len(self.l2_episodes)), min(len(self.l2_episodes), 200))

        # 4. SCORING COM FILTRO DE REPETIÇÃO ATIVO
        scored_candidates = []
        for idx in random.sample(list(candidatos), min(len(candidatos), 300)):
            ep = self.l2_episodes[idx]
            
            # Se o texto já está no histórico, aplicamos uma penalidade insuperável
            history_penalty = 100.0 if ep['t'] in self.history else 0.0
            
            s_q = SSML_Kernel.tsallis_match(v_in, ep['v'])
            sim_foco = SSML_Kernel.dot(self.ctx_foco, ep['v'])
            sim_pathos = SSML_Kernel.dot(self.psi_pathos, ep['v'])
            
            score = (s_q * 0.45) + (sim_foco * 0.25) + (sim_pathos * 0.3) - self.fatigue[idx] - history_penalty
            scored_candidates.append((idx, score))

        # Ordena e pega o melhor (que não seja repetido)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        melhor_idx = scored_candidates[0][0]
        max_score = scored_candidates[0][1]

        # 5. LÓGICA DE INICIATIVA (JOIN ACADA 3 TURNOS)
        res_principal = self.l2_episodes[melhor_idx]['t']
        
        if self.turn_count % 3 == 0:
            # Busca um Sujeito de Iniciativa (Raro, mas diferente do atual)
            potential_hooks = [w for w, c in self.raridade.items() if c < 15 and w != sujeito_atual]
            if potential_hooks:
                hook_word = random.choice(potential_hooks)
                hooks = self.neuronios.get(hook_word, [])
                if hooks:
                    hook_idx = random.choice(hooks)
                    hook_text = self.l2_episodes[hook_idx]['t']
                    if hook_text not in self.history and len(hook_text) < 60:
                        res_principal = f"{res_principal}. Alias, {hook_text}"
                        print(f"[INICIATIVA] Puxando assunto novo: '{hook_word}'")

        # 6. EVOLUÇÃO E FADIGA
        v_vencedor = self.l2_episodes[melhor_idx]['v']
        self.psi_pathos = SSML_Kernel.normalize(self._add_vecs(self.psi_pathos, v_vencedor, 0.94, 0.06))
        self.history.append(self.l2_episodes[melhor_idx]['t'])
        self.fatigue[melhor_idx] += 15.0 # Bloqueio de longo prazo
        for k in list(self.fatigue.keys()): self.fatigue[k] *= 0.5 # Recuperação mais rápida (limpa o campo)

        dt = (time.perf_counter() - t0) * 1000
        print(f" [SSML v10] Subj: {sujeito_atual} | Singularity: {max_score:.2f} | {dt:.1f}ms")
        
        return res_principal

    def boot(self):
        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                d = pickle.load(f)
                self.l2_episodes, self.raridade, self.mapa_nd = d['nexus'], d['raridade'], d['nd']
                self.psi_pathos = d['pathos']
            for i, ep in enumerate(self.l2_episodes):
                for t in self.tokenizer.findall(ep['t'].lower()): self.neuronios[t].append(i)
            print(f"✅ SSML v10.0 Online ({len(self.l2_episodes)} nexos)")
        
        if os.path.exists(self.path_ledger):
            with open(self.path_ledger, 'rb') as f: self.ledger = pickle.load(f)

        for arq in self.auto_train_files:
            if os.path.exists(arq):
                with open(arq, 'r', encoding='utf-8', errors='ignore') as f:
                    conteudo = f.read(); h = hashlib.sha256(conteudo.encode()).hexdigest()
                    if h not in self.ledger: self.cristalizar_solo(conteudo); self.ledger.add(h); self.salvar()

    def cristalizar_solo(self, texto):
        frases = re.split(r'[\.\!\?\n]+', texto)
        for f in frases:
            f = f.strip()
            if len(f) < 3: continue
            tokens = self.tokenizer.findall(f.lower())
            idx = len(self.l2_episodes); v_ep = {}
            for t in tokens:
                self.raridade[t] += 1; self.neuronios[t].append(idx)
                if t not in self.mapa_nd: self.mapa_nd[t] = SSML_Kernel.get_sparse_vec(t)
                v_ep = self._add_vecs(v_ep, self.mapa_nd[t], 1.0, self._get_entropy(t))
            self.l2_episodes.append({'t': f, 'v': SSML_Kernel.normalize(v_ep)})

    def salvar(self):
        self._atomic_save({'nexus': self.l2_episodes, 'raridade': self.raridade, 'nd': self.mapa_nd, 'pathos': self.psi_pathos}, self.path_bin)
        self._atomic_save(self.ledger, self.path_ledger)

    def _atomic_save(self, data, filepath):
        folder = os.path.dirname(os.path.abspath(filepath))
        temp_fd, temp_path = tempfile.mkstemp(dir=folder, prefix="tmp_qssml_")
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush(); os.fsync(f.fileno()) 
            os.replace(temp_path, filepath) 
        except Exception: 
            if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    ssml = QuintikusSSML()
    ssml.boot()
    while True:
        try:
            u = input("\n👤: ").strip()
            if not u: continue
            if u.lower() in ['sair', 'exit']: ssml.salvar(); break
            print(f"🧠 LUCY: {ssml.processar(u)}")
        except KeyboardInterrupt: ssml.salvar(); break
