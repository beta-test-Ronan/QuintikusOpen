import os
import sys
import math
import time
import struct
import random
import json
import unicodedata
from array import array
from collections import defaultdict, Counter

# =================================================================
# 1. DATA CLEANER & NORMALIZER
# =================================================================
class DataCleaner:
    @staticmethod
    def normalizar(txt):
        if not txt: return ""
        txt = "".join(c for c in unicodedata.normalize('NFD', txt.lower()) 
                     if unicodedata.category(c) != 'Mn')
        return txt.split() # Retorna lista de tokens limpos

# =================================================================
# 2. AURIA FS (PERSISTÊNCIA BINÁRIA)
# =================================================================
class AuriaFS:
    @staticmethod
    def salvar(filepath, st, l2_mass, rarity, neuronios):
        with open(filepath, 'wb') as f:
            f.write(b'QOA3') # Versão 3
            f.write(struct.pack('3f', *st))
            f.write(struct.pack('I', len(l2_mass)))
            for frase in l2_mass:
                b_frase = frase.encode('utf-8')
                f.write(struct.pack('I', len(b_frase))) 
                f.write(b_frase)
            f.write(struct.pack('I', len(rarity)))
            for word, val in rarity.items():
                b_word = word.encode('utf-8')
                f.write(struct.pack('H', len(b_word))) 
                f.write(b_word)
                f.write(struct.pack('f', val))
            f.write(struct.pack('I', len(neuronios)))
            for word, indices in neuronios.items():
                b_word = word.encode('utf-8')
                f.write(struct.pack('H', len(b_word))) 
                f.write(b_word)
                arr = array('I', indices)
                f.write(struct.pack('I', len(arr)))
                arr.tofile(f)

    @staticmethod
    def carregar(filepath):
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, 'rb') as f:
                if f.read(4) != b'QOA3': return None
                st = list(struct.unpack('3f', f.read(12)))
                l2_count = struct.unpack('I', f.read(4))[0]
                l2_mass = [f.read(struct.unpack('I', f.read(4))[0]).decode('utf-8') for _ in range(l2_count)]
                r_count = struct.unpack('I', f.read(4))[0]
                rarity = {f.read(struct.unpack('H', f.read(2))[0]).decode('utf-8'): struct.unpack('f', f.read(4))[0] for _ in range(r_count)}
                n_count = struct.unpack('I', f.read(4))[0]
                neuronios = {}
                for _ in range(n_count):
                    w = f.read(struct.unpack('H', f.read(2))[0]).decode('utf-8')
                    arr = array('I')
                    arr.fromfile(f, struct.unpack('I', f.read(4))[0])
                    neuronios[w] = arr.tolist()
                return st, l2_mass, rarity, neuronios
        except: return None

# =================================================================
# 3. QUINTIKUS OPEN AURIA - SURGICAL ENGINE
# =================================================================
class QuintikusOpenAuria:
    def __init__(self):
        self.path = "brain_auria.qoa"
        self.st = [0.5, 0.5, 0.5]
        self.l2_mass = []
        self.neuronios = {}
        self.rarity = {}
        self.l2_tokens = [] # Cache de tokens para busca rápida
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com", "no", "na", "e"}

    def inicializar(self, conteudo):
        print("🧠 Amadurecendo Solo (Surgical Mode)...")
        # Extração simples se for JSON
        try:
            dados = json.loads(conteudo)
            linhas = [f"{i.get('instruction','')} {i.get('input','')} {i.get('output','')}" for i in dados]
            texto = " . ".join(linhas)
        except: texto = conteudo

        frases = [f.strip() for f in texto.split('.') if len(f.strip().split()) > 3]
        contagem = Counter()
        
        for i, f in enumerate(frases):
            self.l2_mass.append(f)
            tokens = DataCleaner.normalizar(f)
            self.l2_tokens.append(set(tokens)) # Guardamos como SET para busca O(1)
            for t in tokens:
                if t not in self.stop_words:
                    if t not in self.neuronios: self.neuronios[t] = []
                    if len(self.neuronios[t]) < 5000: self.neuronios[t].append(i)
                    contagem[t] += 1
            if i % 50000 == 0: print(f" > {i} nexos mapeados...")

        # IDF Lite
        N = len(frases)
        for t, q in contagem.items():
            self.rarity[t] = math.log(N / (q + 1))

        AuriaFS.salvar(self.path, self.st, self.l2_mass, self.rarity, self.neuronios)

    def perguntar(self, entrada):
        t0 = time.perf_counter()
        q_tokens = DataCleaner.normalizar(entrada)
        pivos = [t for t in q_tokens if t not in self.stop_words]
        
        if not pivos: return "Nexo carece de solo."
        
        # 1. Pega candidatos do pivo mais raro
        pivos_sorted = sorted(pivos, key=lambda t: self.rarity.get(t, 0), reverse=True)
        primeiro_pivo = pivos_sorted[0]
        
        candidatos_idx = self.neuronios.get(primeiro_pivo, [])
        if not candidatos_idx: return "Nexo não encontrado."

        best_idx = None
        max_score = -1
        
        # 2. Busca Cirúrgica
        amostra = random.sample(candidatos_idx, min(len(candidatos_idx), 1000))
        
        for idx in amostra:
            frase_tokens = self.l2_tokens[idx]
            
            # Interseção: Quantos pivos da pergunta estão na frase?
            matches = sum(1 for p in pivos if p in frase_tokens)
            
            # Cálculo de Score Galvânico (TF-IDF + Interseção)
            # Cada pivo encontrado multiplica o score de raridade
            score_base = sum(self.rarity.get(p, 0) for p in pivos if p in frase_tokens)
            
            # Bônus de Interseção: Se tem TODOS os pivos, explode o score
            if matches == len(pivos): score_base *= 10
            elif matches > 1: score_base *= 2
            
            if score_base > max_score:
                max_score = score_base
                best_idx = idx
        
        dt = (time.perf_counter() - t0) * 1000000
        
        if max_score <= 0: return "Nexo sem potência suficiente."
        
        return f"\n[{dt:.2f}μs | POT:{max_score*10:.0f}mV]\n> {self.l2_mass[best_idx]}"

    def boot(self):
        dados = AuriaFS.carregar(self.path)
        if dados:
            self.st, self.l2_mass, self.rarity, self.neuronios = dados
            # Reconstroi o cache de tokens (isso consome RAM, mas ganha μs)
            print("🔋 Carregando tokens na RAM...")
            self.l2_tokens = [set(DataCleaner.normalizar(f)) for f in self.l2_mass]
            return True
        return False

# =================================================================
# START
# =================================================================
if __name__ == "__main__":
    auria = QuintikusOpenAuria()
    if not auria.boot():
        with open('cabrita-dataset-52k.json', 'r', encoding='utf-8') as f:
            auria.inicializar(f.read())

    while True:
        u = input("\n👤: ").strip()
        if u.lower() in ['sair', 'exit']: break
        print(auria.perguntar(u))
