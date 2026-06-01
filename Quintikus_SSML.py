import os, math, time, random, re, pickle, hashlib, tempfile, cmath, unicodedata
from collections import defaultdict, Counter, deque

# ==================================================================
# ❄️ HARDWARE SHIELD
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
        return {d: val / (norm + 1e-9) for d, val in v.items()}

# ==================================================================
# 🧠 CÓRTEX COGNITIVO
# ==================================================================
class CortexCognitivo:
    def __init__(self, limite_confusao=0.30):
        self.limite_confusao = limite_confusao
        self.epsilon = 1e-9
        self.taxa_pensamento = 0.12

    def _norm(self, d):
        s = sum(d) + self.epsilon
        return [x / s for x in d]

    def divergencia_kl(self, p, q):
        return sum(p[i] * math.log((p[i] + self.epsilon) / (q[i] + self.epsilon)) for i in range(len(p)))

    def processar_reflexao(self, estado_real, estado_interno):
        p, q = self._norm(estado_real), self._norm(estado_interno)
        ciclos, confusao = 0, self.divergencia_kl(p, q)
        while confusao > self.limite_confusao and ciclos < 45:
            ciclos += 1
            for i in range(len(q)): q[i] = q[i] + self.taxa_pensamento * (p[i] - q[i])
            q = self._norm(q)
            confusao = self.divergencia_kl(p, q)
        return q, ciclos, confusao

# ==================================================================
# 🧠 SISTEMA NERVOSO CENTRAL (RNN + ADAM)
# ==================================================================
class SistemaNervosoCentral:
    def __init__(self, n_in=6, n_hid=10, n_out=3, path="sistema_nervoso.bin"):
        self.n_in, self.n_hid, self.n_out, self.path = n_in, n_hid, n_out, path
        self.t, self.lr = 0, 0.005
        self.W_h = [[random.uniform(-0.1, 0.1) for _ in range(n_in + n_hid)] for _ in range(n_hid)]
        self.W_y = [[random.uniform(-0.1, 0.1) for _ in range(n_hid)] for _ in range(n_out)]
        self.B_h, self.B_y = [0.0]*n_hid, [0.0]*n_out
        
        self.adam_M_Wh = [[0.0]*(n_in+n_hid) for _ in range(n_hid)]
        self.adam_V_Wh = [[0.0]*(n_in+n_hid) for _ in range(n_hid)]
        self.adam_M_Wy = [[0.0]*n_hid for _ in range(n_out)]
        self.adam_V_Wy = [[0.0]*n_hid for _ in range(n_out)]
        
        self.estado_anterior = [0.0]*n_hid
        self.cache = None
        if os.path.exists(path): self._carregar()

    def sigmoid(self, x): return 1.0 / (1.0 + math.exp(-max(-15, min(15, x))))

    def pulsar_vontade(self, x_atual):
        inp = x_atual + self.estado_anterior
        h = [self.sigmoid(self.B_h[i] + sum(inp[j]*self.W_h[i][j] for j in range(len(inp)))) for i in range(self.n_hid)]
        y = [self.sigmoid(self.B_y[i] + sum(h[j]*self.W_y[i][j] for j in range(self.n_hid))) for i in range(self.n_out)]
        self.cache = (inp, h, y)
        self.estado_anterior = h
        return y

    def adaptar_realtime(self, alvo_ideal):
        if not self.cache: return
        self.t += 1
        lr, b1, b2, eps = self.lr, 0.9, 0.999, 1e-8
        inp, h, y = self.cache
        delta_y = [(y[i] - alvo_ideal[i]) * (y[i] * (1.0 - y[i])) for i in range(self.n_out)]
        delta_h = [0.0] * self.n_hid
        for j in range(self.n_hid):
            soma_err = sum(delta_y[i] * self.W_y[i][j] for i in range(self.n_out))
            delta_h[j] = soma_err * (h[j] * (1.0 - h[j]))
        
        c_b1, c_b2 = 1 - b1**self.t, 1 - b2**self.t
        for i in range(self.n_out):
            for j in range(self.n_hid):
                grad = delta_y[i] * h[j]
                self.adam_M_Wy[i][j] = b1*self.adam_M_Wy[i][j] + (1-b1)*grad
                self.adam_V_Wy[i][j] = b2*self.adam_V_Wy[i][j] + (1-b2)*(grad**2)
                self.W_y[i][j] -= lr * (self.adam_M_Wy[i][j]/c_b1) / (math.sqrt(self.adam_V_Wy[i][j]/c_b2) + eps)
            self.B_y[i] -= lr * delta_y[i]
        for i in range(self.n_hid):
            for j in range(len(inp)):
                grad = delta_h[i] * inp[j]
                self.adam_M_Wh[i][j] = b1*self.adam_M_Wh[i][j] + (1-b1)*grad
                self.adam_V_Wh[i][j] = b2*self.adam_V_Wh[i][j] + (1-b2)*(grad**2)
                self.W_h[i][j] -= lr * (self.adam_M_Wh[i][j]/c_b1) / (math.sqrt(self.adam_V_Wh[i][j]/c_b2) + eps)
            self.B_h[i] -= lr * delta_h[i]

    def _salvar(self):
        estado = {'Wh':self.W_h, 'Wy':self.W_y, 'Bh':self.B_h, 'By':self.B_y, 'ea':self.estado_anterior, 't':self.t,
                  'MWh':self.adam_M_Wh, 'VWh':self.adam_V_Wh, 'MWy':self.adam_M_Wy, 'VWy':self.adam_V_Wy}
        with open(self.path, 'wb') as f: pickle.dump(estado, f)

    def _carregar(self):
        with open(self.path, 'rb') as f:
            d = pickle.load(f)
            self.W_h, self.W_y, self.B_h, self.B_y, self.t = d['Wh'], d['Wy'], d['Bh'], d['By'], d.get('t', 0)
            self.adam_M_Wh, self.adam_V_Wh = d.get('MWh', self.adam_M_Wh), d.get('VWh', self.adam_V_Wh)
            self.adam_M_Wy, self.adam_V_Wy = d.get('MWy', self.adam_M_Wy), d.get('VWy', self.adam_V_Wy)
            self.estado_anterior = d.get('ea', self.estado_anterior)

# ==================================================================
# 🧬 DRIVE SOMÁTICO & SISTEMA DEEPY
# ==================================================================
class DriveSomático:
    def __init__(self):
        self.vm = -70.0 
        self.eixos = {"amor": 0.1, "prazer": 0.1, "tristeza": 0.1, "raiva": 0.1}
        self.valvulas = {k: False for k in self.eixos}

    def pulsar(self, impacto, u_toks):
        self.vm = max(-90.0, min(-45.0, self.vm + impacto * 12))
        gatilhos = {"amor":["amo","amor"], "prazer":["prazer","delicia"], "tristeza":["triste","mal"], "raiva":["odeio","raiva"]}
        for eixo, keywords in gatilhos.items():
            for k in keywords:
                if k in u_toks:
                    if self.valvulas[eixo]: self.eixos[eixo] *= 0.6
                    else: self.eixos[eixo] += impacto
                    self.valvulas[eixo] = self.eixos[eixo] > 4.5

class SistemaDeepy:
    def __init__(self, raridade):
        self.raridade = raridade
        self.turnos_think = 0
        self.frequencia_pulso = Counter()
        self.expansores = ('fale', 'sobre', 'tudo', 'detalhes', 'mais', 'explique')

    def crivo_meritocratico(self, tokens, impacto):
        if not tokens: return False, 0
        Q = len(tokens)
        P = sum(1.5 / (math.log(self.raridade.get(t, 1) + 1.2) + 1e-5) for t in tokens)
        x_apr = sum(self.frequencia_pulso.get(t, 0) for t in tokens) / (Q + 1e-5)
        x_nec = Q / (P + 1e-5)
        return (x_apr >= x_nec * 0.08), x_apr

    def filtrar_expansao(self, sujeito, u_toks, entrada_bruta, neuronios, episodes):
        if not any(word in entrada_bruta.lower() for word in self.expansores) or len(u_toks) < 2: return None
        contexto = [t for t in u_toks if t != sujeito]
        alvo = contexto[0]
        if sujeito in neuronios and alvo in neuronios:
            f_suj, f_ctx = set(neuronios[sujeito]), set(neuronios[alvo])
            comuns = list(f_suj.intersection(f_ctx))
            if comuns: return episodes[random.choice(comuns)]['t']
        return None

# ==================================================================
# 🌿 ORGANISMO SOBERANO (v31.1)
# ==================================================================
class OrganismoSoberano:
    def __init__(self):
        self.path_bin = "nucleo_organismo.qssml"
        self.path_ledger = "ledger.bin"
        self.auto_train_files = ["oi.txt", "amor.txt", "prazer.txt", "confusa.txt", "sentimento.txt"]
        
        self.mapa_nd, self.l2_episodes, self.neuronios = {}, [], defaultdict(list)
        self.raridade = Counter()
        self.history = deque(maxlen=20)
        self.fatigue = defaultdict(float)
        self.ctx_foco = {}             
        self.ledger = set() # FIX: Atributo ledger inicializado
        
        self.soma = DriveSomático()
        self.cortex = CortexCognitivo()
        self.snc = SistemaNervosoCentral()
        self.deepy = SistemaDeepy(self.raridade)
        self.tokenizer = re.compile(r'\b\w+\b|[!?.]')

    def _get_entropy(self, t): return 1.0 / (math.log(self.raridade.get(t, 1) + 1.2) + 1e-5)

    def processar(self, entrada):
        t0 = time.perf_counter()
        self.deepy.turnos_think += 1
        if self.deepy.turnos_think >= 7:
            print("\n🧠 [DEEPY] Reorganização REM ativada...")
            for k in list(self.fatigue.keys()): self.fatigue[k] *= 0.2
            self.deepy.turnos_think = 0

        raw = NormalizadorSomático.limpar(entrada)
        u_toks = self.tokenizer.findall(raw)
        if not u_toks: return "..."

        # 1. Percepção Somática e Mérito
        sujeito = max([t for t in u_toks if t in self.neuronios] or [u_toks[0]], key=lambda t: self._get_entropy(t))
        impacto = self._get_entropy(sujeito)
        self.soma.pulsar(impacto, u_toks)
        for t in u_toks: self.deepy.frequencia_pulso[t] += 1
        tem_merito, nivel = self.deepy.crivo_meritocratico(u_toks, impacto)

        # 2. Reflexão do Córtex e Volição do SNC
        p_real = [self.soma.eixos[k] for k in ["amor", "prazer", "tristeza", "raiva"]]
        q_int = self.snc.estado_anterior[:4]
        estado_em, ciclos, dkl = self.cortex.processar_reflexao(p_real, q_int)
        volicao = self.snc.pulsar_vontade(estado_em + [impacto, (self.soma.vm+90)/45])
        modo_idx = volicao.index(max(volicao))

        # 3. Busca por Ressonância
        v_in = {}
        for t in u_toks:
            if t in self.mapa_nd: 
                v_ep_vec = self.mapa_nd[t]
                v_in = {k: v_in.get(k,0) + v_ep_vec.get(k,0)*self._get_entropy(t) for k in set(v_in)|set(v_ep_vec)}
        v_in = KernelRessonante.normalize(v_in)
        
        if not self.ctx_foco: self.ctx_foco = v_in
        else: self.ctx_foco = KernelRessonante.normalize({k: self.ctx_foco.get(k,0)*0.6 + v_in.get(k,0)*0.4 for k in set(self.ctx_foco)|set(v_in)})

        # Busca com proteção
        cand = self.neuronios.get(sujeito, []) or random.sample(range(len(self.l2_episodes)), min(len(self.l2_episodes), 150))

        scored = []
        for idx in cand:
            ep = self.l2_episodes[idx]
            if ep['t'] in self.history: continue
            score = KernelRessonante.tsallis_match(v_in, ep['v']) + KernelRessonante.dot(self.ctx_foco, ep['v'])*0.3 - self.fatigue[ep['t']]
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        melhor_idx = scored[0][0] if scored else random.choice(range(len(self.l2_episodes)))
        res = self.l2_episodes[melhor_idx]['t']

        # 5. Evolução e Aprendizado Sináptico
        alvo = [0.0]*3; alvo[modo_idx] = 1.0
        if dkl < 0.45: self.snc.adaptar_realtime(alvo) # FIX: Variável impacto corrigida internamente
        self.history.append(res); self.fatigue[res] += 10.0
        for k in list(self.fatigue.keys()): self.fatigue[k] *= 0.65

        dt = (time.perf_counter() - t0) * 1000
        print(f" ⚛️ [SNC t:{self.snc.t}] Pensou {ciclos} Ciclos (DKL:{dkl:.2f}) | {dt:.1f}ms")
        return res

    def boot(self):
        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                d = pickle.load(f); self.l2_episodes, self.raridade, self.mapa_nd, self.ctx_foco = d['nexus'], d['raridade'], d['nd'], d.get('ctx_foco', {})
        if os.path.exists(self.path_ledger):
            with open(self.path_ledger, 'rb') as f: self.ledger = pickle.load(f)
        for arq in self.auto_train_files:
            if os.path.exists(arq):
                with open(arq, 'r') as f:
                    c = f.read(); h = hashlib.sha256(c.encode()).hexdigest()
                    if h not in self.ledger: self.cristalizar_solo(c); self.ledger.add(h)
        self.neuronios.clear()
        for i, ep in enumerate(self.l2_episodes):
            for t in self.tokenizer.findall(NormalizadorSomático.limpar(ep['t'])): self.neuronios[t].append(i)
        print(f"✅ Organismo Online. t:{self.snc.t} | Nexos: {len(self.l2_episodes)}")

    def cristalizar_solo(self, texto):
        for f in re.split(r'[\.\!\?\n]+', texto):
            f_l = NormalizadorSomático.limpar(f)
            if len(f_l) < 3: continue
            idx = len(self.l2_episodes); v_ep = {}
            for t in self.tokenizer.findall(f_l):
                self.raridade[t] += 1; self.neuronios[t].append(idx)
                if t not in self.mapa_nd: self.mapa_nd[t] = KernelRessonante.get_vetor_esparso(t)
                v_ep = {k: v_ep.get(k,0) + self.mapa_nd[t].get(k,0)*self._get_entropy(t) for k in self.mapa_nd[t]}
            self.l2_episodes.append({'t': f.strip(), 'v': KernelRessonante.normalize(v_ep)})

    def dormir(self):
        self.snc._salvar()
        with open(self.path_bin, 'wb') as f:
            pickle.dump({'nexus': self.l2_episodes, 'raridade': self.raridade, 'nd': self.mapa_nd, 'ctx_foco': self.ctx_foco}, f)
        with open(self.path_ledger, 'wb') as f: pickle.dump(self.ledger, f)

    def despertar(self):
        if not self.ctx_foco: return "Olá."
        cand = [ep['t'] for ep in self.l2_episodes if KernelRessonante.dot(self.ctx_foco, ep['v']) > 0.6]
        return f"'{random.choice(cand)}'... estive pensando nisso enquanto dormia." if cand else "Oi."

if __name__ == "__main__":
    org = OrganismoSoberano()
    org.boot(); print(f"🧠: {org.despertar()}")
    try:
        while True:
            u = input("\n👤: ").strip()
            if u.lower() == 'sair': break
            print(f"🧠: {org.processar(u)}")
    except: pass
    org.dormir()
