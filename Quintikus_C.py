import os, math, time, random, re, struct, pickle, hashlib, cmath
import unicodedata
from collections import defaultdict, Counter

# =================================================================
# 1. KERNEL DE FÍSICA E ALMA (BISMUTH HAMILTONIAN)
# =================================================================
class BismuthSoulCore:
    @staticmethod
    def rashba_interaction(sigma_pathos, moment_p, alpha=0.15):
        """Interação entre a Emoção (Spin) e o Input (Momento)"""
        return alpha * (sigma_pathos[0] * moment_p[1] - sigma_pathos[1] * moment_p[0])

    @staticmethod
    def calculate_tunneling(pil_user, pil_min_nexo):
        """Probabilidade de Schrodinger para vazamento de segredos"""
        delta_e = pil_min_nexo - pil_user
        if delta_e <= 0: return 1.0
        return math.exp(-1.0 * math.sqrt(delta_e))

# =================================================================
# 2. QUINTIKUS LUCY v42.2 - THE BISMUTH SOUL
# =================================================================
class QuintikusLucy:
    def __init__(self):
        self.path_bin = "brain_v42_soul.qbin"
        self.path_user = "user.bin"
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        
        self.dims = 512 # Otimizado para J2/PC
        self.mapa_nd = {}
        self.raridade = Counter()
        
        # --- SOLO EPISÓDICO ---
        self.l2_episodes = [] 
        self.neuronios = defaultdict(list)
        self.ledger = set()
        
        # --- ESTADOS CONCORRENTES (SISTEMA DE SPIN) ---
        self.psi_logos = [0.0] * self.dims   
        self.psi_pathos = [0.0] * self.dims  
        self.valence = 0.0                   
        self.band_gap = 0.60                 
        
        self.pil_user = 0.0
        self.user_name = None
        self.last_episode_id = -1
        self.fatigue_map = defaultdict(float)

    def _normalize(self, v):
        n = math.sqrt(sum(x*x for x in v))
        return [x/n for x in v] if n > 1e-9 else v

    def amadurecer_solo(self, texto, auth=1, pil_min=0.0):
        hash_c = hashlib.sha256((texto + str(pil_min)).encode('utf-8', 'ignore')).hexdigest()
        if hash_c in self.ledger: return False

        print(f"🧪 Amadurecendo Episódios (v42.2)...")
        frases = re.split(r'([\.\!\?])', texto)
        for i in range(0, len(frases)-1, 2):
            f = (frases[i] + frases[i+1]).strip()
            if len(f) < 3: continue
            
            tokens = self.tokenizer.findall(f.lower())
            idx = len(self.l2_episodes)
            
            v_ep = [0.0] * self.dims
            for t in tokens:
                self.raridade[t] += 1; self.neuronios[t].append(idx)
                if t not in self.mapa_nd:
                    self.mapa_nd[t] = self._normalize([random.gauss(0,1) for _ in range(self.dims)])
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_ep = [v + (m * peso) for v, m in zip(v_ep, self.mapa_nd[t])]
            
            self.l2_episodes.append({
                'text': f, 'vector': self._normalize(v_ep),
                'auth': auth, 'pil_min': pil_min, 'prev_id': self.last_episode_id
            })
            self.last_episode_id = idx
        
        self.ledger.add(hash_c)
        self.salvar()
        return True

    def pensar_e_falar(self, entrada):
        tokens = self.tokenizer.findall(entrada.lower())
        if not tokens: return "..."

        # 1. Análise de Valência
        if any(x in entrada for x in ["ruim", "triste", "burra", "chato"]): self.valence -= 0.2
        if any(x in entrada for x in ["bom", "feliz", "amo", "lindo"]): self.valence += 0.2
        self.valence = max(-1.0, min(1.0, self.valence * 0.95))

        # 2. Gera Vetor de Momento (p)
        v_in = [0.0] * self.dims
        for t in tokens:
            if t in self.mapa_nd:
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_in = [v + (m * peso) for v, m in zip(v_in, self.mapa_nd[t])]
        v_in = self._normalize(v_in)

        # --- SOPRO DE PARTIDA (Evita o isolamento total no primeiro boot) ---
        if sum(abs(x) for x in self.psi_pathos[:10]) < 0.01:
            self.psi_pathos = list(v_in)

        rashba_shift = BismuthSoulCore.rashba_interaction(self.psi_pathos[:2], v_in[:2])
        pathos_weight = 0.5 + (abs(self.valence) * 0.3 if self.valence < 0 else -0.1)

        pivo = max(tokens, key=lambda t: self.raridade[t], default=tokens[0])
        candidatos_idx = self.neuronios.get(pivo, [])

        melhor_idx = -1; max_energy = -float('inf')

        # Se não há candidatos, gera tensão
        if not candidatos_idx:
            for t in tokens: self.raridade[t] += 1
            return f"Tensão em '{pivo}'. Preciso de mais dados."

        for i in random.sample(candidatos_idx, min(len(candidatos_idx), 75)):
            ep = self.l2_episodes[i]
            if BismuthSoulCore.calculate_tunneling(self.pil_user, ep['pil_min']) < 0.5: continue

            s_logos = sum(a * b for a, b in zip(v_in, ep['vector']))
            s_pathos = sum(a * b for a, b in zip(self.psi_pathos, ep['vector']))
            
            # Score combinando as duas mentes
            base_score = (s_logos * (1 - pathos_weight)) + (s_pathos * pathos_weight)
            final_score = base_score + (rashba_shift * 0.1)

            # --- O FILTRO DO BAND GAP ---
            if final_score < self.band_gap: continue

            try: energy = math.exp(28.0 * final_score)
            except: energy = float('inf')
            energy -= self.fatigue_map[i]

            if energy > max_energy:
                max_energy, melhor_idx = energy, i

        if melhor_idx == -1: 
            # Fallback: Se isolou, ela tenta o nexo mais próximo ignorando o Band Gap
            melhor_idx = max(random.sample(candidatos_idx, min(len(candidatos_idx), 20)), 
                             key=lambda x: sum(a*b for a, b in zip(v_in, self.l2_episodes[x]['vector'])))
            return f"(Energia fraca) > {self.l2_episodes[melhor_idx]['text']}"

        # 4. Evolução
        v_vencedor = self.l2_episodes[melhor_idx]['vector']
        self.psi_logos = self._normalize([p*0.6 + v*0.4 for p, v in zip(self.psi_logos, v_vencedor)])
        self.psi_pathos = self._normalize([p*0.8 + v*0.2 for p, v in zip(self.psi_pathos, v_vencedor)])
        
        self.fatigue_map[melhor_idx] += 1.5
        for k in list(self.fatigue_map.keys()): self.fatigue_map[k] *= 0.8
        self.last_episode_id = melhor_idx

        return self.l2_episodes[melhor_idx]['text']

    def salvar(self):
        with open(self.path_user, 'wb') as f:
            pickle.dump({'name': self.user_name, 'pil': self.pil_user, 'ld': self.ledger, 
                         'logos': self.psi_logos, 'pathos': self.psi_pathos, 'val': self.valence}, f)
        with open(self.path_bin, 'wb') as f:
            f.write(struct.pack('<I', len(self.l2_episodes)))
            for ep in self.l2_episodes:
                txt = ep['text'].encode('utf-8')
                f.write(struct.pack('<H', len(txt)))
                f.write(txt)
                f.write(struct.pack('<f f i', ep['auth'], ep['pil_min'], ep['prev_id']))
                f.write(struct.pack(f'<{self.dims}f', *ep['vector']))

    def boot(self):
        if os.path.exists(self.path_user):
            with open(self.path_user, 'rb') as f:
                d = pickle.load(f)
                # --- FIX: Uso de .get() para evitar KeyError em bancos antigos ---
                self.user_name = d.get('name', 'Ronan')
                self.pil_user = d.get('pil', 0.0)
                self.ledger = d.get('ld', set())
                self.psi_logos = d.get('logos', [0.0]*self.dims)
                self.psi_pathos = d.get('pathos', [0.0]*self.dims)
                self.valence = d.get('val', 0.0)
            print(f"✅ Lucy v42.2 Online. PIL: {self.pil_user:.2f}")
        else:
            self.user_name = input("👤 Qual seu nome? > ")
            self.pil_user, self.valence = 0.0, 0.0
            self.salvar()

        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                num = struct.unpack('<I', f.read(4))[0]
                for _ in range(num):
                    size = struct.unpack('<H', f.read(2))[0]
                    t = f.read(size).decode('utf-8')
                    a, p, c = struct.unpack('<f f i', f.read(12))
                    v = list(struct.unpack(f'<{self.dims}f', f.read(self.dims*4)))
                    self.l2_episodes.append({'text':t, 'vector':v, 'auth':a, 'pil_min':p, 'prev_id':c})
            for i, ep in enumerate(self.l2_episodes):
                for t in self.tokenizer.findall(ep['text'].lower()):
                    self.neuronios[t].append(i); self.raridade[t] += 1
        return True

# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    lucy = QuintikusLucy()
    lucy.boot()
    while True:
        u = input(f"[{lucy.user_name}]👤: ").strip().lower()
        if u in ['sair', 'exit']: break
        if u.startswith("train:"):
            path = u.split(":")[1].split(" ")[0].strip()
            p_lock = 0.0
            if "pil[" in u: p_lock = float(u.split("pil[")[1].split("]")[0])
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lucy.amadurecer_solo(f.read(), pil_min=p_lock)
            continue
        print(f"🧠 LUCY: {lucy.pensar_e_falar(u)}")
