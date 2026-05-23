import os, math, time, random, re, struct, pickle, hashlib, threading, sys
import unicodedata
from collections import defaultdict, Counter

# =================================================================
# 1. KERNEL DE ÁLGEBRA LINEAR PURA (DNA QUINTIKUS)
# =================================================================
def pure_norm(v): return math.sqrt(sum(x * x for x in v))
def normalize(v):
    n = pure_norm(v)
    return [x / n for x in v] if n > 1e-9 else v
def pure_dot(v1, v2): return sum(a * b for a, b in zip(v1, v2))
def vec_add(v1, v2): return [a + b for a, b in zip(v1, v2)]
def vec_mul(v, scalar): return [x * scalar for x in v]

# =================================================================
# 2. QUINTIKUSC LUCY v41.0 - SOVEREIGN CORTICAL LOBE
# =================================================================
class QuintikusLucy:
    def __init__(self):
        self.path_bin = "brain_v41_sovereign.qbin"
        self.path_user = "user.bin"
        self.tokenizer = re.compile(r'\?\?+|\!\!+|\.\.\.+|[:;]-?[)DPpoO]|s2|<3|[\w]+|[\?\!\.]')
        
        self.dims = 1024
        self.mapa_nd = {}
        self.raridade = Counter()
        
        # --- ESTRUTURA DE SOLO (AAIGB + CORTEX) ---
        self.l2_mass = []
        self.l2_vectors = []
        self.l2_pil_min = []
        self.l2_neuron_tag = []
        self.neuronios = defaultdict(list)
        self.shard_signatures = {} # Mapa Mental das Regiões Neurais
        
        # --- ESTADOS DINÂMICOS (A PSIQUE) ---
        self.psi_pathos = [0.0] * self.dims   # Lobo Afetivo
        self.sombra_entropica = [0.0] * self.dims # Eco do Contexto
        self.valence = 0.0                    # Humor
        self.pil_user = 0.0
        
        self.ledger = set() 
        self.user_name = None
        self.exaustao = []
        self.fatigue_map = defaultdict(float)

    def amadurecer_solo(self, texto, pil_min=0.0, neuronio="conversa"):
        """Ingestão com Assinatura Cortical (Roteamento Neural)"""
        hash_c = hashlib.sha256((texto + neuronio).encode('utf-8', 'ignore')).hexdigest()
        if hash_c in self.ledger: return False

        print(f"🧪 Mapeando Região Cortical: {neuronio.upper()}...")
        frases = re.split(r'([\.\!\?])', texto)
        shard_acc = [0.0] * self.dims
        count = 0

        for i in range(0, len(frases)-1, 2):
            f = (frases[i] + frases[i+1]).strip()
            if len(f) < 3: continue
            
            tokens = self.tokenizer.findall(f.lower())
            idx = len(self.l2_mass)
            self.l2_mass.append(f)
            self.l2_pil_min.append(pil_min)
            self.l2_neuron_tag.append(neuronio)

            v_frase = [0.0] * self.dims
            for t in tokens:
                self.raridade[t] += 1
                self.neuronios[t].append(idx)
                if t not in self.mapa_nd:
                    self.mapa_nd[t] = normalize([random.gauss(0, 1) for _ in range(self.dims)])
                
                # BUG 2 FIX: Adicionado 1e-5 para estabilidade logarítmica
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_frase = vec_add(v_frase, vec_mul(self.mapa_nd[t], peso))
            
            v_norm = normalize(v_frase)
            self.l2_vectors.append(v_norm)
            shard_acc = vec_add(shard_acc, v_norm)
            count += 1
        
        if count > 0:
            self.shard_signatures[neuronio] = normalize(vec_mul(shard_acc, 1.0/count))
        
        self.ledger.add(hash_c)
        self.salvar()
        return True

    def pensar_e_falar(self, entrada):
        tokens = self.tokenizer.findall(entrada.lower())
        if not tokens: return "..."

        # 1. Análise de Valência e Humor
        if any(x in entrada.lower() for x in ["bom", "feliz", "amo", "lindo"]): self.valence = min(1.0, self.valence + 0.1)
        if any(x in entrada.lower() for x in ["ruim", "triste", "chato", "burra"]): self.valence = max(-1.0, self.valence - 0.1)

        # 2. Vetor de Entrada e Sombra Entrópica (Eco Cognitivo)
        v_in = [0.0] * self.dims
        for t in tokens:
            if t in self.mapa_nd:
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_in = vec_add(v_in, vec_mul(self.mapa_nd[t], peso))
        v_in = normalize(v_in)
        
        # Atualiza Sombra Entrópica (60% input atual, 40% eco anterior)
        self.sombra_entropica = normalize(vec_add(vec_mul(v_in, 0.6), vec_mul(self.sombra_entropica, 0.4)))

        # 3. Roteamento Cortical (Aura v25 logic)
        target_neuron = "conversa"
        forca_nexo = 0.0
        for n, sig in self.shard_signatures.items():
            att = pure_dot(v_in, sig)
            if att > forca_nexo: forca_nexo, target_neuron = att, n

        # 4. Peso Dinâmico do Psi (Sovereign Logic)
        # Se o nexo é forte, a Lucy se permite sentir mais (Psi influi mais)
        psi_weight = 0.2 + (forca_nexo * 0.4)
        v_interpretado = normalize(vec_add(vec_mul(self.sombra_entropica, 1-psi_weight), vec_mul(self.psi_pathos, psi_weight)))

        # 5. Busca e Colapso (Modern Hopfield Moderna)
        pivo = max(tokens, key=lambda t: self.raridade[t], default=tokens[0])
        candidatos = [i for i in self.neuronios.get(pivo, []) if self.l2_pil_min[i] <= self.pil_user]

        if not candidatos: return f"Nexo insuficiente na região {target_neuron.upper()}."

        def pontuar(idx):
            dot = pure_dot(v_interpretado, self.l2_vectors[idx])
            # Contraste Exponencial (Hopfield Moderna)
            try: score = math.exp(25.0 * dot)
            except: score = float('inf')
            
            # Bônus de Região Cortical
            if self.l2_neuron_tag[idx] == target_neuron: score *= 1.5
            # Penalidade de Fadiga (Inibição Lateral)
            score -= self.fatigue_map[idx]
            return score

        # Amostragem cirúrgica (80 candidatos para precisão/velocidade)
        amostra = random.sample(candidatos, min(len(candidatos), 80))
        idx_final = max(amostra, key=pontuar)

        # 6. Evolução e Metabolismo
        target_vec = self.l2_vectors[idx_final]
        self.psi_pathos = normalize(vec_add(vec_mul(self.psi_pathos, 0.8), vec_mul(target_vec, 0.2)))
        self.fatigue_map[idx_final] += 1.0
        for k in list(self.fatigue_map.keys()): 
            self.fatigue_map[k] *= 0.7 # Esfriamento de fadiga
        
        if pure_dot(v_in, target_vec) > 0.8: self.pil_user = min(100.0, self.pil_user + 0.1)

        return self.l2_mass[idx_final]

    def salvar(self):
        with open(self.path_user, 'wb') as f:
            pickle.dump({'name': self.user_name, 'pil': self.pil_user, 'ld': self.ledger, 'pathos': self.psi_pathos, 'val': self.valence}, f)
        with open(self.path_bin, 'wb') as f:
            f.write(b'V41B') # Magic: Version 41 Binary
            f.write(struct.pack('<I', len(self.l2_mass)))
            for i in range(len(self.l2_mass)):
                txt, tag = self.l2_mass[i].encode('utf-8'), self.l2_neuron_tag[i].encode('utf-8')
                f.write(struct.pack('<H', len(txt)))
                f.write(txt)
                f.write(struct.pack('<B', len(tag)))
                f.write(tag)
                f.write(struct.pack('<f', self.l2_pil_min[i]))
                f.write(struct.pack(f'<{self.dims}f', *self.l2_vectors[i]))
            
            f.write(struct.pack('<I', len(self.shard_signatures)))
            for k, v in self.shard_signatures.items():
                kb = k.encode('utf-8')
                f.write(struct.pack('<B', len(kb)))
                f.write(kb)
                f.write(struct.pack(f'<{self.dims}f', *v))

    def boot(self):
        if os.path.exists(self.path_user):
            with open(self.path_user, 'rb') as f:
                d = pickle.load(f)
                self.user_name, self.pil_user, self.ledger = d['name'], d['pil'], d['ld']
                self.psi_pathos, self.valence = d.get('pathos', [0.0]*self.dims), d.get('val', 0.0)
            print(f"✅ Lucy v41.0 Online. Córtex ativo ({self.pil_user:.2f} PIL).")
        else:
            self.user_name = input("👤 Seu nome: ")
            self.pil_user = 0.0
            self.salvar()

        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                if f.read(4) == b'V41B':
                    num = struct.unpack('<I', f.read(4))[0]
                    for _ in range(num):
                        size = struct.unpack('<H', f.read(2))[0]
                        txt = f.read(size).decode('utf-8')
                        tsize = struct.unpack('<B', f.read(1))[0]
                        tag = f.read(tsize).decode('utf-8')
                        p = struct.unpack('<f', f.read(4))[0]
                        vec = list(struct.unpack(f'<{self.dims}f', f.read(self.dims*4)))
                        self.l2_mass.append(txt); self.l2_neuron_tag.append(tag)
                        self.l2_pil_min.append(p); self.l2_vectors.append(vec)
                    
                    num_sig = struct.unpack('<I', f.read(4))[0]
                    for _ in range(num_sig):
                        klen = struct.unpack('<B', f.read(1))[0]
                        k = f.read(klen).decode('utf-8')
                        v = list(struct.unpack(f'<{self.dims}f', f.read(self.dims*4)))
                        self.shard_signatures[k] = v
            # BUG 1 FIX: Unificado para 'raridade'
            for i, txt in enumerate(self.l2_mass):
                for t in self.tokenizer.findall(txt.lower()):
                    self.neuronios[t].append(i)
                    self.raridade[t] += 1
        return True

# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    lucy = QuintikusLucy()
    lucy.boot()
    while True:
        u = input(f"[{lucy.user_name}]👤: ").strip()
        if u.lower() in ['sair', 'exit']: break
        
        if u.startswith("train:"):
            path = u.split(":")[1].split(" ")[0].strip()
            n_target = "conversa"
            if "neuron[" in u: n_target = u.split("neuron[")[1].split("]")[0]
            p_lock = 0.0
            if "pil[" in u: p_lock = float(u.split("pil[")[1].split("]")[0])
            
            if os.path.exists(path):
                print(f"📖 Lendo shard {path}...")
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lucy.amadurecer_solo(f.read(), pil_min=p_lock, neuronio=n_target)
                print("🧠 Córtex atualizado.")
            continue

        print(f"🧠 LUCY: {lucy.pensar_e_falar(u)}")
