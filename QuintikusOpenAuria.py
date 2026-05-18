import os
import sys
import math
import time
import struct
import random
import json
import unicodedata
import hashlib
from array import array
from collections import defaultdict, Counter

# =================================================================
# 1. KERNEL DE DADOS E LIMPEZA (SURGICAL MODE)
# =================================================================
class DataCleaner:
    @staticmethod
    def normalizar(txt):
        if not txt: return []
        txt = "".join(c for c in unicodedata.normalize('NFD', txt.lower()) 
                     if unicodedata.category(c) != 'Mn')
        return txt.split()

# =================================================================
# 2. AURIA FS - PERSISTÊNCIA BINÁRIA V4 (ESTADOS TÉRMICOS)
# =================================================================
class AuriaFS:
    @staticmethod
    def salvar(filepath, st, l2_mass, rarity, neuronios):
        with open(filepath, 'wb') as f:
            f.write(b'QOA4') # Versão 4: Full AGI Integration
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
                if f.read(4) != b'QOA4': return None
                st = list(struct.unpack('3f', f.read(12)))
                l2_count = struct.unpack('I', f.read(4))[0]
                l2_mass = [f.read(struct.unpack('I', f.read(4))[0]).decode('utf-8') for _ in range(l2_count)]
                r_count = struct.unpack('I', f.read(4))[0]
                rarity = {f.read(struct.unpack('H', f.read(2))[0]).decode('utf-8'): struct.unpack('f', f.read(4))[0] for _ in range(r_count)}
                n_count = struct.unpack('I', f.read(4))[0]
                neuronios = {f.read(struct.unpack('H', f.read(2))[0]).decode('utf-8'): array('I', []).fromfile(f, struct.unpack('I', f.read(4))[0]) or [] for _ in range(n_count)}
                # Correção para leitura de array do neuronios
                f.seek(16) # Volta pro início do nexo de neurônios
                f.read(4) # Pula header
                f.read(12) # Pula st
                f.read(struct.unpack('I', f.read(4))[0]) # Pula l2... (Simplificado para o exemplo)
                # Recarregar neuronios de forma estável:
                return st, l2_mass, rarity, None # Recarregaremos os neurônios via boot
        except: return None

# =================================================================
# 3. QUINTIKUS OPEN AURIA AGI (FRANKENSTEIN FINAL)
# =================================================================
class QuintikusAuriaAGI:
    def __init__(self):
        self.path = "brain_auria.qoa"
        # Estados: 0:Temp(Pressão), 1:Sinergia(Harmonia), 2:Foco(Atenção)
        self.st = [0.5, 0.5, 0.5]
        self.l2_mass = []
        self.l2_tokens = []
        self.neuronios = defaultdict(list)
        self.rarity = {}
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com", "no", "na", "e", "como", "fazer", "qual"}
        self.th_map = {'bom':0.1, 'ótimo':0.2, 'sinergia':0.3, 'paz':0.2, 'erro':-0.2, 'urgente':-0.3, 'falha':-0.2}

    def inicializar(self, conteudo):
        print("🧠 Amadurecendo Nexo Galvânico (Surgical DLM Mode)...")
        try:
            dados = json.loads(conteudo)
            texto = " . ".join([f"{i.get('instruction','')} {i.get('input','')} {i.get('output','')}" for i in dados])
        except: texto = conteudo

        frases = [f.strip() for f in texto.split('.') if len(f.strip().split()) > 3]
        contagem = Counter()
        
        for i, f in enumerate(frases):
            self.l2_mass.append(f)
            tokens = DataCleaner.normalizar(f)
            self.l2_tokens.append(set(tokens))
            for t in tokens:
                if t not in self.stop_words:
                    if len(self.neuronios[t]) < 10000: self.neuronios[t].append(i)
                    contagem[t] += 1
            if i % 50000 == 0: print(f" > {i} nexos mapeados...")

        N = len(frases)
        for t, q in contagem.items():
            self.rarity[t] = math.log(N / (q + 1))

        AuriaFS.salvar(self.path, self.st, self.l2_mass, self.rarity, self.neuronios)

    def _upd_thermal(self, tokens):
        """TSPLS: Atualização Procedural do Estado Emocional"""
        p, n = 0, 0
        for t in tokens:
            val = self.th_map.get(t, 0)
            if val > 0: p += val
            else: n += abs(val)
        self.st[0] = self.st[0] * 0.95 + (n * 0.1)
        self.st[1] = self.st[1] * 0.95 + (p * 0.1)
        for i in range(3): self.st[i] = max(0, min(1, self.st[i]))

    def falar(self, entrada):
        t0 = time.perf_counter()
        q_tokens = DataCleaner.normalizar(entrada)
        self._upd_thermal(q_tokens) # Sente o input
        
        pivos = [t for t in q_tokens if t not in self.stop_words]
        if not pivos: return "[SISTEMA EM SILÊNCIO]"
        
        # 1. BUSCA CIRÚRGICA (Auria Engine)
        pivos_sorted = sorted(pivos, key=lambda t: self.rarity.get(t, 0), reverse=True)
        candidatos_idx = self.neuronios.get(pivos_sorted[0], [])
        
        best_idx = None
        max_score = -1
        
        if candidatos_idx:
            amostra = random.sample(candidatos_idx, min(len(candidatos_idx), 1000))
            for idx in amostra:
                f_tokens = self.l2_tokens[idx]
                matches = sum(1 for p in pivos if p in f_tokens)
                score = sum(self.rarity.get(p, 0) for p in pivos if p in f_tokens)
                if matches == len(pivos): score *= 10
                if score > max_score:
                    max_score = score
                    best_idx = idx

        # 2. SÍNTESE DLM-FLOW
        if best_idx is not None:
            f1 = self.l2_mass[best_idx]
            # Tenta pegar o próximo nexo (Simula a DLM Matrix)
            f2 = self.l2_mass[best_idx + 1] if best_idx + 1 < len(self.l2_mass) else ""
            sn = "DLM-ACTIVE"
        else:
            f1, f2, sn = (f"o nexo '{pivos[:1]}' carece de solo", "", "VOID")

        # 3. MODULAÇÃO TÉRMICA DA VOZ
        if self.st[0] > 0.7: _i, _c = ["Sob pressão, ", "Rancor ativo, "], ["Fim do estresse.", "Normalizando."]
        elif self.st[1] > 0.7: _i, _c = ["Em harmonia, ", "Sinergia plena, "], ["A luz brilha.", "Fluxo perfeito."]
        else: _i, _c = ["Pela razão, ", "No vácuo, "], ["Aguardando nexo.", "Selado."]
        
        _res = f"{random.choice(_i)} {f1}. {random.choice(_c)}"
        if f2 and len(f2) < 200: _res = _res.replace(".", f". Além disso, {f2}.", 1)

        dt = (time.perf_counter() - t0) * 1000000
        st_info = f"T:{self.st[0]:.1f}|S:{self.st[1]:.1f}|POT:{max_score*10:.0f}mV"
        
        return f"\n[DLM-FLOW | {dt:.2f}μs | {st_info} | {sn}]\n> {_res.capitalize()}"

    def boot(self):
        if os.path.exists(self.path):
            print("🔋 Carregando solo Auria (High-Speed Access)...")
            # Para manter a velocidade de microsegundos, o boot re-indexa em RAM
            # mas o L2_MASS e RARITY vêm do binário
            try:
                with open(self.path, 'rb') as f:
                    f.read(4) # header
                    self.st = list(struct.unpack('3f', f.read(12)))
                    l2_count = struct.unpack('I', f.read(4))[0]
                    for _ in range(l2_count):
                        frase = f.read(struct.unpack('I', f.read(4))[0]).decode('utf-8')
                        self.l2_mass.append(frase)
                        tokens = DataCleaner.normalizar(frase)
                        self.l2_tokens.append(set(tokens))
                        idx = len(self.l2_mass) - 1
                        for t in tokens:
                            if t not in self.stop_words and len(self.neuronios[t]) < 10000:
                                self.neuronios[t].append(idx)
                    
                    r_count = struct.unpack('I', f.read(4))[0]
                    for _ in range(r_count):
                        w = f.read(struct.unpack('H', f.read(2))[0]).decode('utf-8')
                        self.rarity[w] = struct.unpack('f', f.read(4))[0]
                print(f"✅ Auria AGI Online: {len(self.l2_mass)} nexos.")
                return True
            except: return False
        return False

# =================================================================
# START
# =================================================================
if __name__ == "__main__":
    auria = QuintikusAuriaAGI()
    if not auria.boot():
        with open('dataset-52k.json', 'r', encoding='utf-8') as f:
            auria.inicializar(f.read())

    while True:
        u = input("\n👤: ").strip()
        if u.lower() in ['sair', 'exit']: break
        if u: print(auria.falar(u))
