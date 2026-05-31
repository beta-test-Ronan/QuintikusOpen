import os, math, time, random, re, pickle, hashlib, tempfile, cmath, unicodedata
from collections import defaultdict, Counter, deque
from array import array

# ==================================================================
# ❄️ HARDWARE SHIELD
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class TextNormalizer:
    @staticmethod
    def limpar(texto):
        if not texto: return ""
        texto = texto.lower()
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                        if unicodedata.category(c) != 'Mn')
        texto = re.sub(r'[^a-z0-9!?.\s]', '', texto)
        return texto.strip()

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

# =================================================================
# 🧬 DRIVE BIOSSOMÁTICO (Vm + INÉRCIA + EXTASE)
# =================================================================
class SomaticDrive:
    def __init__(self):
        self.vm = -70.0 
        self.vm_max, self.vm_min = -45.0, -90.0
        self.wavefunction = [0.5, 0.5, 0.5] 
        
        # --- MOTOR DE ESTABILIDADE (COSMIC) ---
        self.inercia_eixos = {"amor": 0.0, "prazer": 0.0, "tristeza": 0.0, "raiva": 0.0}
        self.valvula_ativa = {"amor": False, "prazer": False, "tristeza": False, "raiva": False}
        self.limiar_extase = 4.5
        self.taxa_decaimento = 0.15

    def pulsar(self, impacto, tokens_input):
        # A. Atualização Bioelétrica (Vm)
        self.vm = max(self.vm_min, min(self.vm_max, self.vm + impacto * 12))
        
        # B. Análise Somática (Inércia de Eixos)
        # Mapeamento rápido de gatilhos somáticos
        gatilhos = {
            "amor": ["amo", "amor", "lindo", "querida", "beijo"],
            "prazer": ["prazer", "gosto", "delicia", "quente", "quero"],
            "tristeza": ["triste", "sozinho", "dor", "saudade", "mal"],
            "raiva": ["odeio", "raiva", "burra", "parar", "chega"]
        }

        for eixo, keywords in gatilhos.items():
            for k in keywords:
                if k in tokens_input:
                    # Se a válvula está estourada, o estímulo é ignorado (Apatia Reativa)
                    if self.valvula_ativa[eixo]:
                        self.inercia_eixos[eixo] *= 0.5 # Esfria a inércia
                        if self.inercia_eixos[eixo] < 1.0: self.valvula_ativa[eixo] = False
                        continue
                    
                    # Aplica atenuação exponencial (Regulador de Fluxo)
                    reacao = impacto * math.exp(-self.taxa_decaimento * self.inercia_eixos[eixo])
                    self.inercia_eixos[eixo] += reacao
                    
                    # Verifica Válvula de Escape (Êxtase)
                    if self.inercia_eixos[eixo] > self.limiar_extase:
                        self.valvula_ativa[eixo] = True
                        print(f"🔥 [VÁLVULA] Êxtase de '{eixo}' atingido! Inversão de fase somática.")

    def get_tau(self):
        tau = ((self.vm - self.vm_min) / (self.vm_max - self.vm_min)) + 0.15
        # Se houver qualquer válvula ativa, a Dany fica mais "fria" (Tau baixo)
        if any(self.valvula_ativa.values()): tau *= 0.5
        return tau

class QuintikusSSML:
    def __init__(self):
        self.dims = 5000
        self.path_bin = "brain_sovereign.qssml"
        self.path_ledger = "ledger.bin"
        self.auto_train_files = ["oi.txt", "amor.txt", "conversa.txt", "confusa.txt", "sentimento.txt", "prazer.txt"]
        
        self.mapa_nd, self.l2_episodes = {}, [] 
        self.neuronios = defaultdict(list)
        self.raridade = Counter()
        self.ledger = set() 
        self.psi_pathos = {}    
        self.tokenizer = re.compile(r'\b\w+\b|[!?.]')
        self.fatigue = defaultdict(float)
        self.history = deque(maxlen=20)
        self.turn_count = 0
        self.ctx_foco = {}             
        self.ctx_sujeitos_ativos = {}  
        self.ctx_inercia, self.ctx_esquecimento = 0.65, 0.75   
        
        # --- DRIVE SOMÁTICO UNIFICADO ---
        self.soma = SomaticDrive()

    def _get_entropy(self, token):
        count = self.raridade.get(token, 1)
        return 1.0 / (math.log(count + 1.2) + 1e-5)

    def _resgatar_sujeito_oculto(self, u_toks):
        conhecidos = [t for t in u_toks if t in self.neuronios]
        if not conhecidos or len(u_toks) <= 2:
            sujeitos_vivos = sorted(self.ctx_sujeitos_ativos.items(), key=lambda x: x[1], reverse=True)
            if sujeitos_vivos: return sujeitos_vivos[0][0]
        if conhecidos: return max(conhecidos, key=lambda t: self._get_entropy(t))
        return u_toks[0] if u_toks else "vazio"

    def processar(self, entrada):
        t0 = time.perf_counter()
        self.turn_count += 1
        entrada_limpa = TextNormalizer.limpar(entrada)
        u_toks = self.tokenizer.findall(entrada_limpa)
        if not u_toks: return "..."

        # 1. UPDATE SOMÁTICO (Vm + Inércia de Gatilhos)
        sujeito_atual = self._resgatar_sujeito_oculto(u_toks)
        impacto_total = self._get_entropy(sujeito_atual)
        self.soma.pulsar(impacto_total, u_toks)
        
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

        # 3. BUSCA E SCORING COM REJEIÇÃO SOMÁTICA
        candidatos_idx = self.neuronios.get(sujeito_atual, [])
        if not candidatos_idx: candidatos_idx = random.sample(range(len(self.l2_episodes)), min(len(self.l2_episodes), 150))

        scored_data = []
        for idx in candidatos_idx:
            ep = self.l2_episodes[idx]
            if ep['t'] in self.history: continue
            
            # --- PENALIDADE DE VÁLVULA ESTOURADA ---
            penalidade_valvula = 0.0
            texto_ep = ep['t'].lower()
            if self.soma.valvula_ativa["amor"] and any(x in texto_ep for x in ["amo", "amor", "lindo"]): penalidade_valvula = 0.8
            if self.soma.valvula_ativa["prazer"] and any(x in texto_ep for x in ["prazer", "delicia", "quente"]): penalidade_valvula = 0.8
            if self.soma.valvula_ativa["tristeza"] and any(x in texto_ep for x in ["triste", "saudade"]): penalidade_valvula = 0.8

            s_q = SSML_Kernel.tsallis_match(v_in, ep['v'])
            sim_f, sim_p = SSML_Kernel.dot(self.ctx_foco, ep['v']), SSML_Kernel.dot(self.psi_pathos, ep['v'])
            
            score = (s_q * 0.4) + (sim_f * 0.3) + (sim_p * 0.3) - self.fatigue[idx] - penalidade_valvula
            scored_data.append((idx, score))

        if not scored_data: return "..."

        # 4. COLAPSO QUANTUM
        scored_data.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_data[:10]
        tau = self.soma.get_tau()
        try:
            max_v = max(x[1] for x in top_k)
            exp_vals = [math.exp(max(-10, min(10, (x[1] - max_v) / tau))) for x in top_k]
            melhor_idx = random.choices([x[0] for x in top_k], weights=exp_vals, k=1)[0]
        except: melhor_idx = top_k[0][0]

        # 5. EVOLUÇÃO
        v_vencedor = self.l2_episodes[melhor_idx]['v']
        self.psi_pathos = SSML_Kernel.normalize(self._add_vecs(self.psi_pathos, v_vencedor, 0.94, 0.06))
        self.history.append(self.l2_episodes[melhor_idx]['t'])
        self.fatigue[melhor_idx] += 12.0
        for k in list(self.fatigue.keys()): self.fatigue[k] *= 0.6

        dt = (time.perf_counter() - t0) * 1000
        # Monitor Somático
        sat_status = "|".join([f"{k[0].upper()}:{v:.1f}" for k, v in self.soma.inercia_eixos.items()])
        print(f" ⚛️ [Vm:{self.soma.vm:.1f}mV] Soma:[{sat_status}] | {dt:.1f}ms")
        
        return self.l2_episodes[melhor_idx]['t']

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
                for t in self.tokenizer.findall(TextNormalizer.limpar(ep['t'])): self.neuronios[t].append(i)
            print(f"✅ SSML v19.0 Somatic Online ({len(self.l2_episodes)} nexos)")
        # Auto-train ledger mantido...
        if os.path.exists(self.path_ledger):
            with open(self.path_ledger, 'rb') as f: self.ledger = pickle.load(f)
        for arq in self.auto_train_files:
            if os.path.exists(arq):
                with open(arq, 'r', encoding='utf-8', errors='ignore') as f:
                    c = f.read(); h = hashlib.sha256(c.encode()).hexdigest()
                    if h not in self.ledger: self.cristalizar_solo(c); self.ledger.add(h); self.salvar()

    def cristalizar_solo(self, texto):
        for f in re.split(r'[\.\!\?\n]+', texto):
            f_limpa = TextNormalizer.limpar(f)
            if len(f_limpa) < 3: continue
            idx = len(self.l2_episodes); v_ep = {}
            for t in self.tokenizer.findall(f_limpa):
                self.raridade[t] += 1; self.neuronios[t].append(idx)
                if t not in self.mapa_nd: self.mapa_nd[t] = SSML_Kernel.get_sparse_vec(t)
                v_ep = self._add_vecs(v_ep, self.mapa_nd[t], 1.0, self._get_entropy(t))
            self.l2_episodes.append({'t': f.strip(), 'v': SSML_Kernel.normalize(v_ep)})

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
            print(f"🧠 Dany: {ssml.processar(u)}")
        except KeyboardInterrupt: ssml.salvar(); break
