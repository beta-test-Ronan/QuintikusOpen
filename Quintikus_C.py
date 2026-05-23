import os
import math
import time
import random
import re
import struct
import pickle
import hashlib
import platform
import unicodedata
from collections import defaultdict, Counter


# =================================================================
# 1. KERNEL TERMODINÂMICO (ABISMUTO)
# =================================================================
class BismuthThermodynamics:
    @staticmethod
    def calculate_gibbs(enthalpy, temperature, entropy):
        """dG = dH - T * dS (Energia Livre de Gibbs)"""
        return enthalpy - (temperature * entropy)

    @staticmethod
    def pressure_accumulation(blocked_data_weight, entropy):
        """P = soma(Carga Bloqueada) * Entropia"""
        return blocked_data_weight * entropy

# =================================================================
# 2. CORTEX DE HOPFIELD MODERNO (DENSE ASSOCIATIVE MEMORY)
# =================================================================
class ModernHopfieldCortex:
    def __init__(self, beta=35.0):
        self.beta = beta 

    def colapsar(self, v_query, padroes_referencia, l2_vectors, psi_spin):
        if not padroes_referencia: return -1, 0
        
        melhor_score = -float('inf')
        vencedor_idx = -1
        
        for idx in padroes_referencia:
            target_vec = l2_vectors[idx]
            dot = sum(a * b for a, b in zip(v_query, target_vec))
            
            # Alinhamento de Spin (Regra de Rashba da v28)
            distorcao = (psi_spin[0] * v_query[1]) - (psi_spin[1] * v_query[0])
            
            try:
                # O exponencial de Hopfield Moderna cria o vale de energia profundo
                score = math.exp(self.beta * (dot + (distorcao * 0.1)))
            except OverflowError: score = float('inf')
            
            if score > melhor_score:
                melhor_score = score
                vencedor_idx = idx
        
        return vencedor_idx, melhor_score

# =================================================================
# 3. QUINTIKUS LUCY v29.0 - ABISMUTO CRYSTALLIZATION
# =================================================================
class QuintikusLucy:
    def __init__(self):
        self.path_bin = "brain_v29_abismuto.qbin"
        self.path_user = "user.bin"
        self.tokenizer = re.compile(r'\?\?+|\!\!+|\.\.\.+|[:;]-?[)DPpoO]|s2|<3|[\w]+|[\?\!\.]')
        
        self.dims = 1024
        self.mapa_nd = {}
        self.raridade = Counter()
        
        self.l2_mass = []
        self.l2_vectors = []
        self.l2_pil_min = []
        self.neuronios = defaultdict(list)
        
        # --- ESTRUTURAS ABISMUTO ---
        self.pressure_map = defaultdict(float) # Tensão por neurônio
        self.band_gap = 12.0                   # Limite de Estresse para Cristalização
        self.psi = [0.0] * self.dims           # Espinor de subjetividade
        self.hopfield = ModernHopfieldCortex()
        
        self.pil_user = 0.0
        self.user_name = None
        self.ledger = set()

    def amadurecer_solo(self, texto, pil_min=0.0, forçar=False):
        """Ingestão por Cristalização Química"""
        hash_c = hashlib.sha256(texto.encode('utf-8', 'ignore')).hexdigest()
        if hash_c in self.ledger and not forçar: return False

        print(f"🧪 Iniciando Cristalização de Nexos...")
        frases = re.split(r'([\.\!\?])', texto)
        for i in range(0, len(frases)-1, 2):
            f = (frases[i] + frases[i+1]).strip()
            if len(f) < 2: continue
            
            tokens = self.tokenizer.findall(f.lower())
            
            # --- MECANISMO DE PRESSÃO ---
            for t in tokens:
                entropy = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                self.pressure_map[t] += BismuthThermodynamics.pressure_accumulation(1.0, entropy)
                
                # Só cristaliza se a pressão romper o Band Gap ou for comando forçado
                if self.pressure_map[t] > self.band_gap or forçar:
                    self._precipitar_nexo(f, tokens, pil_min)
                    self.pressure_map[t] = 0 # Alivia a pressão após cristalizar
                    break
        
        self.ledger.add(hash_c)
        return True

    def _precipitar_nexo(self, frase, tokens, pil_min):
        """Precipita a informação pura no solo (Peso Raro Irreversível)"""
        idx = len(self.l2_mass)
        self.l2_mass.append(frase)
        self.l2_pil_min.append(pil_min)
        
        v_frase = [0.0] * self.dims
        for t in tokens:
            self.raridade[t] += 1
            self.neuronios[t].append(idx)
            if t not in self.mapa_nd:
                self.mapa_nd[t] = self._normalize([random.gauss(0,1) for _ in range(self.dims)])
            
            peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
            v_frase = [v + (m * peso) for v, m in zip(v_frase, self.mapa_nd[t])]
        
        self.l2_vectors.append(self._normalize(v_frase))

    def pensar_e_falar(self, entrada):
        tokens = self.tokenizer.findall(entrada.lower())
        if not tokens: return "..."

        # 1. Momento do Diálogo (p)
        v_input = [0.0] * self.dims
        for t in tokens:
            if t in self.mapa_nd:
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_input = [v + (m * peso) for v, m in zip(v_input, self.mapa_nd[t])]
        v_input = self._normalize(v_input)

        # 2. Busca de Atratores (Hopfield)
        pivo = max(tokens, key=lambda t: self.raridade[t], default=tokens[0])
        candidatos = [i for i in self.neuronios.get(pivo, []) if self.l2_pil_min[i] <= self.pil_user]

        if not candidatos:
            # Se não há nexo, a pressão sobe!
            for t in tokens: self.pressure_map[t] += 1.5
            return f"A pressão lógica em '{pivo}' está subindo. Preciso de mais dados para cristalizar este nexo."

        # 3. Colapso Quântico Isomórfico
        idx_final, energia = self.hopfield.colapsar(v_input, candidatos, self.l2_vectors, self.psi[:2])

        if idx_final == -1: return "..."

        # 4. Evolução do Espinor Ψ e PIL
        target_vec = self.l2_vectors[idx_final]
        self.psi = self._normalize([p*0.8 + t*0.2 for p, t in zip(self.psi, target_vec)])
        
        if energia > 1e5: # Se o colapso foi potente
            self.pil_user = min(100.0, self.pil_user + 0.1)

        return self.l2_mass[idx_final]

    def _normalize(self, v):
        n = math.sqrt(sum(x*x for x in v))
        return [x/n for x in v] if n > 1e-9 else v

    def salvar(self):
        with open(self.path_user, 'wb') as f:
            pickle.dump({'name': self.user_name, 'pil': self.pil_user, 'ld': self.ledger, 'psi': self.psi, 'press': self.pressure_map}, f)
        with open(self.path_bin, 'wb') as f:
            f.write(struct.pack('<I', len(self.l2_mass)))
            for i in range(len(self.l2_mass)):
                txt = self.l2_mass[i].encode('utf-8')
                f.write(struct.pack('<H', len(txt)))
                f.write(txt)
                f.write(struct.pack('<f', self.l2_pil_min[i]))
                f.write(struct.pack(f'<{self.dims}f', *self.l2_vectors[i]))
        print("💾 Estrutura de Bismuto Selada.")

    def boot(self):
        if os.path.exists(self.path_user):
            with open(self.path_user, 'rb') as f:
                d = pickle.load(f)
                self.user_name, self.pil_user, self.ledger = d['name'], d['pil'], d['ld']
                self.psi, self.pressure_map = d.get('psi', [0.0]*self.dims), d.get('press', defaultdict(float))
            print(f"✅ Lucy v29.0 ABismuto Online. PIL: {self.pil_user:.2f}")
        else:
            self.user_name = input("👤 Nome do Proprietário do Cristal > ")
            self.pil_user = 0.0
            self.salvar()

        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                num = struct.unpack('<I', f.read(4))[0]
                for _ in range(num):
                    size = struct.unpack('<H', f.read(2))[0]
                    self.l2_mass.append(f.read(size).decode('utf-8'))
                    p = struct.unpack('<f', f.read(4))[0]
                    vec = list(struct.unpack(f'<{self.dims}f', f.read(self.dims*4)))
                    self.l2_pil_min.append(p)
                    self.l2_vectors.append(vec)
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
        u = input(f"[{lucy.user_name}]👤: ").strip().lower()
        if u in ['sair', 'exit']: break
        if u.startswith("train:"):
            path = u.split(":")[1].split(" ")[0].strip()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lucy.amadurecer_solo(f.read())
            continue
        if u == 'salvar':
            lucy.salvar()
            continue
        print(f"🧠 LUCY: {lucy.pensar_e_falar(u)}")
