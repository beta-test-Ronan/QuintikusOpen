import os, math, time, random, re, struct, pickle, hashlib, cmath
import unicodedata
from collections import defaultdict, Counter

# =================================================================
# 1. KERNEL ESPARSO (SPARSE VECTOR MATH)
# =================================================================
class SparseMath:
    @staticmethod
    def get_deterministic_vec(token, dims, sparsity=25):
        """Gera um vetor esparso fixo baseado no hash do token"""
        seed = int(hashlib.md5(token.encode()).hexdigest(), 16)
        random.seed(seed)
        # Ativa apenas 'sparsity' dimensões das 5000 disponíveis
        indices = random.sample(range(dims), sparsity)
        return {i: random.gauss(0, 1) for i in indices}

    @staticmethod
    def dot(v1, v2):
        """Produto escalar esparso: O(min(len(v1), len(v2)))"""
        if len(v1) > len(v2): v1, v2 = v2, v1
        return sum(val * v2.get(dim, 0) for dim, val in v1.items())

    @staticmethod
    def normalize(v):
        norm = math.sqrt(sum(x*x for x in v.values()))
        if norm < 1e-9: return v
        return {dim: val/norm for dim, val in v.items()}

    @staticmethod
    def add(v1, v2, w1=1.0, w2=1.0):
        res = defaultdict(float, {d: v * w1 for d, v in v1.items()})
        for d, v in v2.items(): res[d] += v * w2
        return dict(res)

# =================================================================
# 2. QUINTIKUS LUCY v54.0 - SPARSE SOVEREIGN
# =================================================================
class QuintikusLucy:
    def __init__(self):
        self.path_bin = "brain_v54_sparse.qbin"
        self.path_user = "user.bin"
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        
        self.dims = 5000 
        self.mapa_nd = {} 
        self.raridade = Counter()
        self.l2_episodes = [] # [{text, vector, pil_min, ts}]
        self.neuronios = defaultdict(list)
        
        # --- ESTADOS DINÂMICOS ---
        self.psi = {} # Vetor esparso de subjetividade
        self.context_history = [] # [(vector, timestamp)]
        self.pil_user = 0.0
        self.user_name = None
        self.fatigue_map = defaultdict(float)
        self.ledger = set()

    def amadurecer_solo(self, texto, pil_min=0.0):
        hash_c = hashlib.sha256(texto.encode('utf-8', 'ignore')).hexdigest()
        if hash_c in self.ledger: return False

        print(f"🌌 Cristalizando Solo Esparso (5000D)...")
        frases = re.split(r'([\.\!\?])', texto)
        for i in range(0, len(frases)-1, 2):
            f = (frases[i] + frases[i+1]).strip()
            if len(f) < 3: continue
            
            tokens = self.tokenizer.findall(f.lower())
            idx = len(self.l2_episodes)
            
            v_ep = {}
            for t in tokens:
                self.raridade[t] += 1; self.neuronios[t].append(idx)
                if t not in self.mapa_nd:
                    self.mapa_nd[t] = SparseMath.get_deterministic_vec(t, self.dims)
                
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_ep = SparseMath.add(v_ep, self.mapa_nd[t], 1.0, peso)
            
            self.l2_episodes.append({
                'text': f, 'vector': SparseMath.normalize(v_ep),
                'pil_min': pil_min, 'ts': time.time()
            })
        
        self.ledger.add(hash_c); self.salvar()
        return True

    def consolidar_memoria(self):
        """Modo Sono: Funde episódios redundantes para limpar o multiverso"""
        if len(self.l2_episodes) < 2: return
        print("🌙 Iniciando Consolidação Hipocampal...")
        novos_episodios = []
        skip = set()

        for i in range(len(self.l2_episodes)):
            if i in skip: continue
            v1 = self.l2_episodes[i]['vector']
            
            for j in range(i + 1, len(self.l2_episodes)):
                if j in skip: continue
                v2 = self.l2_episodes[j]['vector']
                
                # Se a similaridade for > 0.95, funde os nexos
                if SparseMath.dot(v1, v2) > 0.95:
                    skip.add(j)
            
            novos_episodios.append(self.l2_episodes[i])
        
        print(f"✨ Poda concluída: {len(self.l2_episodes)} -> {len(novos_episodios)} nexos.")
        self.l2_episodes = novos_episodios
        self.salvar()

    def pensar_e_falar(self, entrada):
        t0 = time.perf_counter()
        u_toks = self.tokenizer.findall(entrada.lower())
        if not u_toks: return "..."

        # 1. GERAÇÃO DE VETOR DE INPUT (ESPARSO)
        v_in = {}
        for t in u_toks:
            if t in self.mapa_nd:
                w = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_in = SparseMath.add(v_in, self.mapa_nd[t], 1.0, w)
        v_in = SparseMath.normalize(v_in)
        
        # 2. DYNAMICS: Context Window com Decay Temporal
        t_now = time.time()
        self.context_history.append((v_in, t_now))
        if len(self.context_history) > 5: self.context_history.pop(0)
        
        v_ctx = {}
        for v, ts in self.context_history:
            # Peso cai exponencialmente com o tempo
            peso_temporal = math.exp(-0.01 * (t_now - ts))
            v_ctx = SparseMath.add(v_ctx, v, 1.0, peso_temporal)
        v_ctx = SparseMath.normalize(v_ctx)

        # 3. COMPETIÇÃO NEURAL (Top-K)
        pivo = min(u_toks, key=lambda t: self.raridade.get(t, 999), default=u_toks[0])
        candidatos_idx = self.neuronios.get(pivo, [])
        if not candidatos_idx: return f"(Tensão em {pivo})"

        # Avalia candidatos usando similaridade esparsa
        scores = []
        for i in random.sample(candidatos_idx, min(len(candidatos_idx), 100)):
            dot = SparseMath.dot(v_ctx, self.l2_episodes[i]['vector'])
            # Aplica subjetividade Psi
            dot_psi = SparseMath.dot(self.psi, self.l2_episodes[i]['vector'])
            
            final_score = (dot * 0.7) + (dot_psi * 0.3) - self.fatigue_map[i]
            scores.append((i, final_score))

        # Pega o Top 3 para criar nuance
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:3]
        
        if not top_k: return "..."
        
        # O vencedor é o nexo com melhor balanço
        vencedor_idx = top_k[0][0]

        # 4. EVOLUÇÃO DO PSI
        self.psi = SparseMath.normalize(SparseMath.add(self.psi, self.l2_episodes[vencedor_idx]['vector'], 0.8, 0.2))
        self.fatigue_map[vencedor_idx] += 1.5
        for k in list(self.fatigue_map.keys()): self.fatigue_map[k] *= 0.8
        
        dt = (time.perf_counter() - t0) * 1000
        print(f"⏱️ Sparse Flow: {dt:.2f}ms | Active Dims: {len(v_in)}")
        
        return self.l2_episodes[vencedor_idx]['text']

    def salvar(self):
        with open(self.path_user, 'wb') as f:
            pickle.dump({'name': self.user_name, 'pil': self.pil_user, 'ld': self.ledger, 'psi': self.psi}, f)
        with open(self.path_bin, 'wb') as f:
            # Salva apenas o Solo (vetores esparsos são dicts, salvos via pickle por simplicidade aqui)
            pickle.dump({'mass': self.l2_episodes, 'rar': self.raridade, 'nd': self.mapa_nd}, f)

    def boot(self):
        if os.path.exists(self.path_user):
            with open(self.path_user, 'rb') as f:
                d = pickle.load(f)
                self.user_name, self.pil_user, self.ledger, self.psi = d['name'], d['pil'], d['ld'], d.get('psi', {})
        
        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                d = pickle.load(f)
                self.l2_episodes, self.raridade, self.mapa_nd = d['mass'], d['rar'], d['nd']
                for i, ep in enumerate(self.l2_episodes):
                    for t in self.tokenizer.findall(ep['text'].lower()):
                        self.neuronios[t].append(i)
            print(f"✅ Lucy v54.0 Sparse Online. {len(self.l2_episodes)} nexos estáveis.")
            return True
        return False

if __name__ == "__main__":
    lucy = QuintikusLucy()
    lucy.boot()
    while True:
        u = input(f"[{lucy.user_name}]👤: ").strip().lower()
        if u in ['sair', 'exit']: break
        if u == 'sonhar': lucy.consolidar_memoria(); continue
        if u.startswith("train:"):
            path = u.split(":")[1].strip()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lucy.amadurecer_solo(f.read())
            continue
        print(f"🧠 LUCY: {lucy.pensar_e_falar(u)}")
