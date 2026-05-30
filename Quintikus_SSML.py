import os, math, time, random, re, pickle, hashlib, tempfile, cmath
from collections import defaultdict, Counter, deque
from array import array

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
    def normalize(v):
        norm = math.sqrt(sum(x*x for x in v.values()))
        return {d: val / (norm + 1e-9) for d, val in v.items()}

# =================================================================
# 2. MOTOR BIO-LOGIC (Vm + WAVEFUNCTION)
# =================================================================
class BioLogicDrive:
    def __init__(self):
        self.vm = -70.0  # Potencial de Repouso
        self.vm_max, self.vm_min = -45.0, -90.0
        self.wavefunction = [0.5, 0.5, 0.5] # [Físico, Cognitivo, Abstrato]

    def pulsar(self, impacto):
        # Despolarização baseada na raridade do input
        self.vm = max(self.vm_min, min(self.vm_max, self.vm + impacto * 12))
        # Ajuste da Onda de Estado
        for i in range(3):
            self.wavefunction[i] = self.wavefunction[i] * 0.75 + (abs(impacto) * 0.25)

    def get_tau(self):
        # Converte milivolts em Temperatura Softmax
        return ((self.vm - self.vm_min) / (self.vm_max - self.vm_min)) + 0.15

# =================================================================
# 3. QUINTIKUS SSML v15.0 - UNIFIED TRINITY
# =================================================================
class QuintikusSSML:
    def __init__(self):
        self.dims = 5000
        self.path_bin = "brain_sovereign.qssml"
        self.path_ledger = "ledger.bin"
        self.auto_train_files = ["oi.txt", "amor.txt", "conversa.txt", "confusa.txt", "sentimento.txt"]
        
        self.mapa_nd, self.l2_episodes = {}, [] 
        self.neuronios = defaultdict(list)
        self.raridade = Counter()
        self.ledger = set() 
        
        self.psi_pathos = {}    
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        self.fatigue = defaultdict(float)
        self.history = deque(maxlen=15)
        self.turn_count = 0

        # --- DRIVE 1: CÓRTEX FRONTAL ---
        self.ctx_foco = {}             
        self.ctx_sujeitos_ativos = {}  
        self.ctx_inercia, self.ctx_esquecimento = 0.65, 0.75   

        # --- DRIVE 2: BIO-LOGIC ---
        self.bio = BioLogicDrive()

    def _get_entropy(self, token):
        count = self.raridade.get(token, 1)
        return 1.0 / (math.log(count + 1.2) + 1e-5)

    def _softmax_quantum(self, candidatos, scores):
        tau = self.bio.get_tau()
        try:
            max_s = max(scores)
            exp_vals = [math.exp(max(-10, min(10, (s - max_s) / tau))) for s in scores]
            total = sum(exp_vals)
            probs = [ev / total for ev in exp_vals]
            return random.choices(candidatos, weights=probs, k=1)[0]
        except: return candidatos[0]

    def _gerar_proatividade(self, sujeito, u_toks):
        """DRIVE 3: PROATIVIDADE DE 3 CAMADAS (CAOS, SUJEITO, PREDICADO)"""
        # A escolha do modo é influenciada pelo Vm (Bio)
        # Se Vm > -60mV (excitada), ela prefere CAOS. Se baixo, prefere SUJEITO.
        tau = self.bio.get_tau()
        if tau > 0.8: modo = random.choice(["CAOS", "PREDICADO"])
        else: modo = random.choice(["SUJEITO", "PREDICADO"])

        if modo == "CAOS":
            raros = [w for w, c in self.raridade.items() if 1 < c < 12]
            if raros:
                idx = random.choice(self.neuronios.get(random.choice(raros), [0]))
                return f"Alias, você já parou pra pensar que {self.l2_episodes[idx]['t']}?"
        elif modo == "SUJEITO":
            return random.choice(["O que você acha de {s}?", "Você acredita em {s}?"]).format(s=sujeito)
        elif modo == "PREDICADO":
            acao = u_toks[1] if len(u_toks) > 1 else "isso"
            return f"E se {acao} fosse o sentido de tudo?"
        return ""

    def processar(self, entrada):
        t0 = time.perf_counter()
        self.turn_count += 1
        u_toks = self.tokenizer.findall(entrada.lower())
        if not u_toks: return "..."

        # 1. UPDATE BIO & CORTEX
        sujeito_atual = max(u_toks, key=lambda t: self._get_entropy(t))
        self.bio.pulsar(self._get_entropy(sujeito_atual))
        for s in list(self.ctx_sujeitos_ativos.keys()): self.ctx_sujeitos_ativos[s] *= self.ctx_esquecimento
        self.ctx_sujeitos_ativos[sujeito_atual] = 1.0

        # 2. VETORES
        v_in = {}
        for t in u_toks:
            if t in self.mapa_nd: v_in = self._add_vecs(v_in, self.mapa_nd[t], 1.0, self._get_entropy(t))
        v_in = SSML_Kernel.normalize(v_in)
        
        if not self.ctx_foco: self.ctx_foco = v_in
        else: self.ctx_foco = self._add_vecs(self.ctx_foco, v_in, self.ctx_inercia, 1.0 - self.ctx_inercia)
        self.ctx_foco = SSML_Kernel.normalize(self.ctx_foco)

        # 3. BUSCA & SCORING
        pivos = [t for t in u_toks if t in self.neuronios] or [s for s in self.ctx_sujeitos_ativos if s in self.neuronios]
        candidatos_idx = self.neuronios.get(max(pivos, key=lambda x: self._get_entropy(x)), []) if pivos else []
        if not candidatos_idx: candidatos_idx = random.sample(range(len(self.l2_episodes)), min(len(self.l2_episodes), 150))

        scored_data = []
        for idx in random.sample(list(candidatos_idx), min(len(candidatos_idx), 250)):
            ep = self.l2_episodes[idx]
            if ep['t'] in self.history: continue
            
            s_q = SSML_Kernel.tsallis_match(v_in, ep['v'])
            # Similaridade com a Onda de Estado (Wavefunction)
            sim_wave = sum(v * self.bio.wavefunction[i % 3] for i, v in enumerate(ep['v'].values()))
            sim_pathos = sum(v * self.psi_pathos.get(k, 0) for k, v in ep['v'].items())
            
            score = (s_q * 0.5) + (sim_wave * 0.2) + (sim_pathos * 0.3) - self.fatigue[idx]
            scored_data.append((idx, score))

        # 4. COLAPSO QUANTUM (SOFTMAX)
        scored_data.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_data[:10]
        melhor_idx = self._softmax_quantum([x[0] for x in top_k], [x[1] for x in top_k])

        # 5. RESPOSTA + PROATIVIDADE (DRIVE 3)
        res_final = self.l2_episodes[melhor_idx]['t']
        if self.turn_count % 3 == 0:
            hook = self._gerar_proatividade(sujeito_atual, u_toks)
            res_final = f"{res_final}. {hook}"

        # 6. EVOLUÇÃO
        v_vencedor = self.l2_episodes[melhor_idx]['v']
        self.psi_pathos = SSML_Kernel.normalize(self._add_vecs(self.psi_pathos, v_vencedor, 0.94, 0.06))
        self.history.append(self.l2_episodes[melhor_idx]['t'])
        self.fatigue[melhor_idx] += 12.0
        for k in list(self.fatigue.keys()): self.fatigue[k] *= 0.6

        dt = (time.perf_counter() - t0) * 1000
        print(f" ⚛️ [Vm: {self.bio.vm:.1f}mV | Tau: {self.bio.get_tau():.2f}] Subj: {sujeito_atual} | {dt:.1f}ms")
        return res_final

    # --- BOOT / SALVAR / AUX ---
    def _add_vecs(self, v1, v2, w1, w2):
        res = {d: v * w1 for d, v in v1.items()}
        for d, v in v2.items(): res[d] = res.get(d, 0) + (v * w2)
        return res

    def boot(self):
        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                d = pickle.load(f)
                self.l2_episodes, self.raridade, self.mapa_nd, self.psi_pathos = d['nexus'], d['raridade'], d['nd'], d['pathos']
            for i, ep in enumerate(self.l2_episodes):
                for t in self.tokenizer.findall(ep['t'].lower()): self.neuronios[t].append(i)
            print(f"✅ SSML v15.0 Trinity Online ({len(self.l2_episodes)} nexos)")
        if os.path.exists(self.path_ledger):
            with open(self.path_ledger, 'rb') as f: self.ledger = pickle.load(f)
        for arq in self.auto_train_files:
            if os.path.exists(arq):
                with open(arq, 'r', encoding='utf-8', errors='ignore') as f:
                    c = f.read(); h = hashlib.sha256(c.encode()).hexdigest()
                    if h not in self.ledger: self.cristalizar_solo(c); self.ledger.add(h); self.salvar()

    def cristalizar_solo(self, texto):
        for f in re.split(r'[\.\!\?\n]+', texto):
            f = f.strip()
            if len(f) < 3: continue
            idx = len(self.l2_episodes); v_ep = {}
            for t in self.tokenizer.findall(f.lower()):
                self.raridade[t] += 1; self.neuronios[t].append(idx)
                if t not in self.mapa_nd: self.mapa_nd[t] = SSML_Kernel.get_sparse_vec(t)
                v_ep = self._add_vecs(v_ep, self.mapa_nd[t], 1.0, self._get_entropy(t))
            self.l2_episodes.append({'t': f, 'v': SSML_Kernel.normalize(v_ep)})

    def salvar(self):
        self._atomic_save({'nexus': self.l2_episodes, 'raridade': self.raridade, 'nd': self.mapa_nd, 'pathos': self.psi_pathos}, self.path_bin)
        self._atomic_save(self.ledger, self.path_ledger)

    def _atomic_save(self, data, filepath):
        folder = os.path.dirname(os.path.abspath(filepath))
        t_fd, t_path = tempfile.mkstemp(dir=folder, prefix="tmp_qssml_")
        try:
            with os.fdopen(t_fd, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush(); os.fsync(f.fileno()) 
            os.replace(t_path, filepath) 
        except: 
            if os.path.exists(t_path): os.remove(t_path)

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
