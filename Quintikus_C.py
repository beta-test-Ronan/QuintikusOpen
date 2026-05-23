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
# 1. KERNEL E VOZ (MANTIDO)
# =================================================================
try:
    import androidhelper
    droid = androidhelper.Android()
    TEM_VOZ = True
except:
    droid = None
    TEM_VOZ = False

def falar(texto, imprimir=True):
    if imprimir: print(f"🧠 LUCY: {texto}")
    if TEM_VOZ:
        try: droid.ttsSpeak(texto)
        except: pass

def ouvir():
    if TEM_VOZ:
        try:
            r = droid.recognizeSpeech("Sintonizando Mapa Mental...", None, None).result
            if r: return r.strip().lower()
        except: pass
    return input("👤: ").strip().lower()

# =================================================================
# 2. ÁLGEBRA LINEAR PURA (COMPATIBILIDADE TOTAL)
# =================================================================
def pure_norm(v): return math.sqrt(sum(x * x for x in v))
def normalize_vector(v):
    n = pure_norm(v)
    return [x / n for x in v] if n > 1e-9 else v
def pure_randn(dims): return [random.gauss(0, 1) for _ in range(dims)]
def pure_dot(v1, v2): return sum(a * b for a, b in zip(v1, v2))
def vec_add(v1, v2): return [a + b for a, b in zip(v1, v2)]
def vec_mul(v, scalar): return [x * scalar for x in v]

# =================================================================
# 3. QUINTIKUS LUCY v25.0 - COGNITIVE ENTROPY MAP
# =================================================================
class QuintikusLucy:
    def __init__(self):
        self.path_bin = "brain_v25_entropy.qbin"
        self.path_user = "user.bin"
        self.tokenizer = re.compile(r'\?\?+|\!\!+|\.\.\.+|[:;]-?[)DPpoO]|s2|<3|[\w]+|[\?\!\.]')
        
        self.dims = 1024
        self.mapa_nd = {}
        self.raridade = Counter()
        
        # AAIGB Matrix + Entropy Maps
        self.l2_mass = []
        self.l2_vectors = []
        self.l2_pil_min = []
        self.l2_neuron_tag = []
        
        self.neuronios = defaultdict(list)
        self.shard_signatures = {} # Mapa Mental: Neuronio -> Vetor Médio (Assinatura)
        self.ledger = set() 
        
        self.pil_user = 0.0
        self.user_name = None
        self.sombra_entropica = [0.0] * self.dims
        self.exaustao = []

    def amadurecer_solo(self, texto, pil_min=0.0, neuronio="conversa"):
        """Geração de Gatilho Dinâmico via Entropia Cognitiva"""
        hash_c = hashlib.sha256((texto + neuronio).encode('utf-8', 'ignore')).hexdigest()
        if hash_c in self.ledger: return False

        print(f"🧠 Mapeando Entropia: {neuronio.upper()}...")
        frases = re.split(r'([\.\!\?])', texto)
        
        shard_vector_accumulator = [0.0] * self.dims
        count_frases = 0

        for i in range(0, len(frases)-1, 2):
            f = (frases[i] + frases[i+1]).strip()
            if len(f) < 2: continue
            
            tokens = self.tokenizer.findall(f.lower())
            if not tokens: continue

            idx = len(self.l2_mass)
            self.l2_mass.append(f)
            self.l2_pil_min.append(pil_min)
            self.l2_neuron_tag.append(neuronio)

            v_frase = [0.0] * self.dims
            for t in tokens:
                self.raridade[t] += 1
                self.neuronios[t].append(idx)
                if t not in self.mapa_nd:
                    self.mapa_nd[t] = normalize_vector(pure_randn(self.dims))
                
                # Peso de Raridade (Entropia Cognitiva)
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_frase = vec_add(v_frase, vec_mul(self.mapa_nd[t], peso))
            
            v_frase_norm = normalize_vector(v_frase)
            self.l2_vectors.append(v_frase_norm)
            
            # Acumula para a assinatura do neurônio
            shard_vector_accumulator = vec_add(shard_vector_accumulator, v_frase_norm)
            count_frases += 1
        
        # Define o "Nexo Médio" deste shard no Mapa Mental
        if count_frases > 0:
            self.shard_signatures[neuronio] = normalize_vector(vec_mul(shard_vector_accumulator, 1.0/count_frases))
        
        self.ledger.add(hash_c)
        return True

    def rotear_pelo_mapa(self, v_entrada):
        """Identifica qual neurônio está 'vibrando' com o input atual"""
        melhor_neuronio = "conversa"
        maior_atencao = -1.0
        
        for neuronio, assinatura in self.shard_signatures.items():
            atencao = pure_dot(v_entrada, assinatura)
            if atencao > maior_atencao:
                maior_atencao = atencao
                melhor_neuronio = neuronio
        
        return melhor_neuronio, maior_atencao

    def pensar_e_falar(self, entrada):
        tokens = self.tokenizer.findall(entrada.lower())
        if not tokens: return "..."

        # 1. Gera Vetor de Entrada
        v_entrada = [0.0] * self.dims
        for t in tokens:
            if t in self.mapa_nd:
                peso = 1.0 / (math.log(self.raridade[t] + 1.2) + 1e-5)
                v_entrada = vec_add(v_entrada, vec_mul(self.mapa_nd[t], peso))
        v_entrada = normalize_vector(v_entrada)

        # 2. Roteamento Dinâmico (Sem gatilhos expostos)
        target_neuron, forca_nexo = self.rotear_pelo_mapa(v_entrada)
        
        # 3. Sombra Entrópica (Memória de Curto Prazo)
        self.sombra_entropica = normalize_vector(
            vec_add(vec_mul(self.sombra_entropica, 0.4), vec_mul(v_entrada, 0.6))
        )

        pivo = max(tokens, key=lambda t: self.raridade[t], default=tokens[0])
        candidatos_base = self.neuronios.get(pivo, [])
        
        # Filtra por PIL e Prioriza Neurônio do Mapa Mental
        candidatos_final = [i for i in candidatos_base if self.l2_pil_min[i] <= self.pil_user]

        if not candidatos_final:
            return f"Nexo insuficiente no mapa mental '{target_neuron.upper()}'."

        # 4. Colapso Quântico Ponderado
        amostra = random.sample(candidatos_final, min(len(candidatos_final), 100))
        def pontuar(idx):
            score = pure_dot(self.sombra_entropica, self.l2_vectors[idx])
            # Bônus se o nexo pertencer ao neurônio que o mapa mental indicou
            if self.l2_neuron_tag[idx] == target_neuron: 
                score += (forca_nexo * 0.4) 
            if idx in self.exaustao: score -= 2.0
            return score

        idx_final = max(amostra, key=pontuar)
        self.exaustao.append(idx_final)
        if len(self.exaustao) > 15: self.exaustao.pop(0)

        # Evolução do PIL
        if forca_nexo > 0.8:
            self.pil_user = min(100.0, self.pil_user + 0.05)

        return self.l2_mass[idx_final]

    def salvar(self):
        with open(self.path_user, 'wb') as f:
            pickle.dump({'name': self.user_name, 'pil': self.pil_user, 'ld': self.ledger}, f)
        with open(self.path_bin, 'wb') as f:
            f.write(b'ENTR') # Magic: Entropy Map
            f.write(struct.pack('<I', len(self.l2_mass)))
            for i in range(len(self.l2_mass)):
                txt = self.l2_mass[i].encode('utf-8')
                tag = self.l2_neuron_tag[i].encode('utf-8')
                f.write(struct.pack('<H', len(txt)))
                f.write(txt)
                f.write(struct.pack('<B', len(tag)))
                f.write(tag)
                f.write(struct.pack('<f', self.l2_pil_min[i]))
                f.write(struct.pack(f'<{self.dims}f', *self.l2_vectors[i]))
            
            # Salva as Assinaturas do Mapa Mental
            f.write(struct.pack('<I', len(self.shard_signatures)))
            for k, v in self.shard_signatures.items():
                kb = k.encode('utf-8')
                f.write(struct.pack('<B', len(kb)))
                f.write(kb)
                f.write(struct.pack(f'<{self.dims}f', *v))
        print("💾 Mapa Mental e Solo selados.")

    def boot(self):
        if os.path.exists(self.path_user):
            with open(self.path_user, 'rb') as f:
                d = pickle.load(f)
                self.user_name, self.pil_user, self.ledger = d['name'], d['pil'], d.get('ld', set())
            falar(f"Olá, {self.user_name}. Mapa Mental carregado ({self.pil_user:.2f} PIL).")
        else:
            falar("Iniciando Protocolo de Entropia. Qual seu nome?")
            self.user_name = ouvir()
            self.pil_user = 0.0
            self.salvar()

        if os.path.exists(self.path_bin):
            with open(self.path_bin, 'rb') as f:
                if f.read(4) == b'ENTR':
                    num = struct.unpack('<I', f.read(4))[0]
                    for _ in range(num):
                        size = struct.unpack('<H', f.read(2))[0]
                        txt = f.read(size).decode('utf-8')
                        tsize = struct.unpack('<B', f.read(1))[0]
                        tag = f.read(tsize).decode('utf-8')
                        p = struct.unpack('<f', f.read(4))[0]
                        vec = list(struct.unpack(f'<{self.dims}f', f.read(self.dims*4)))
                        self.l2_mass.append(txt)
                        self.l2_neuron_tag.append(tag)
                        self.l2_pil_min.append(p)
                        self.l2_vectors.append(vec)
                    
                    # Carrega as assinaturas
                    try:
                        num_sig = struct.unpack('<I', f.read(4))[0]
                        for _ in range(num_sig):
                            klen = struct.unpack('<B', f.read(1))[0]
                            k = f.read(klen).decode('utf-8')
                            v = list(struct.unpack(f'<{self.dims}f', f.read(self.dims*4)))
                            self.shard_signatures[k] = v
                    except: pass
            # Reindexa
            for i, txt in enumerate(self.l2_mass):
                for t in self.tokenizer.findall(txt.lower()):
                    self.neuronios[t].append(i)
                    self.rarity[t] += 1
        return True

# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    lucy = QuintikusLucy()
    lucy.boot()
    while True:
        u = ouvir()
        if not u or u in ['sair', 'exit', 'tchau']: break
        
        if u.startswith("train:"):
            path = u.split(":")[1].split(" ")[0].strip()
            n_target = "conversa"
            if "neuron[" in u: n_target = u.split("neuron[")[1].split("]")[0]
            p_lock = 0.0
            if "pil[" in u: p_lock = float(u.split("pil[")[1].split("]")[0])
            
            if os.path.exists(path):
                falar(f"Extraindo entropia de {path}...")
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lucy.amadurecer_solo(f.read(), pil_min=p_lock, neuronio=n_target)
                falar("Mapa Mental atualizado.")
            continue

        if u == 'salvar':
            lucy.salvar()
            falar("Nexo persistido.")
            continue

        res = lucy.pensar_e_falar(u)
        falar(res)
