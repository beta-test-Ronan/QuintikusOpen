import os
import sys
import math
import time
import struct
import random
import unicodedata
from array import array
from collections import defaultdict

# =================================================================
# 1. KERNEL QMATH - ALGEBRA LINEAR DE BAIXO NÍVEL
# =================================================================
class QKernel:
    @staticmethod
    def dot_product_int8(vec_x, weights_int8, offset, size, scale=0.01):
        soma = 0.0
        for i in range(size):
            soma += vec_x[i] * (weights_int8[offset + i] * scale)
        return soma

# =================================================================
# 2. AURIA FS - PERSISTÊNCIA BINÁRIA CRUA
# =================================================================
class AuriaFS:
    """
    Motor de Persistência Quintikus Open Auria (.qoa)
    Layout: [HEAD:4b][ST:3f][L2_LEN:I][L2_DATA...][RAR_LEN:I][RAR_DATA...][NEU_LEN:I][NEU_DATA...]
    """
    @staticmethod
    def salvar(filepath, st, l2_mass, rarity, neuronios):
        with open(filepath, 'wb') as f:
            # 1. Header & Thermal State (3 floats)
            f.write(b'QOA1')
            f.write(struct.pack('3f', *st))
            
            # 2. L2 Mass (Sentenças do Corpus)
            f.write(struct.pack('I', len(l2_mass)))
            for frase in l2_mass:
                b_frase = frase.encode('utf-8')
                f.write(struct.pack('H', len(b_frase))) # Max 65k chars/frase
                f.write(b_frase)
            
            # 3. Rarity Map (Word:String -> Val:Float)
            f.write(struct.pack('I', len(rarity)))
            for word, val in rarity.items():
                b_word = word.encode('utf-8')
                f.write(struct.pack('B', len(b_word))) # Max 255 chars/palavra
                f.write(b_word)
                f.write(struct.pack('f', val))
            
            # 4. Neurônios (Word:String -> Indices:Array[I])
            f.write(struct.pack('I', len(neuronios)))
            for word, indices in neuronios.items():
                b_word = word.encode('utf-8')
                f.write(struct.pack('B', len(b_word)))
                f.write(b_word)
                
                arr = array('I', indices) # Array de unsigned int (4 bytes cada)
                f.write(struct.pack('I', len(arr)))
                arr.tofile(f)
        
        print(f"💾 QOA: Memória selada binariamente ({os.path.getsize(filepath)/1024:.1f} KB)")

    @staticmethod
    def carregar(filepath):
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, 'rb') as f:
                if f.read(4) != b'QOA1': return None
                st = list(struct.unpack('3f', f.read(12)))
                
                # L2 Mass
                l2_count = struct.unpack('I', f.read(4))[0]
                l2_mass = []
                for _ in range(l2_count):
                    flen = struct.unpack('H', f.read(2))[0]
                    l2_mass.append(f.read(flen).decode('utf-8'))
                
                # Rarity
                r_count = struct.unpack('I', f.read(4))[0]
                rarity = {}
                for _ in range(r_count):
                    wlen = struct.unpack('B', f.read(1))[0]
                    word = f.read(wlen).decode('utf-8')
                    val = struct.unpack('f', f.read(4))[0]
                    rarity[word] = val
                
                # Neurônios
                n_count = struct.unpack('I', f.read(4))[0]
                neuronios = defaultdict(list)
                for _ in range(n_count):
                    wlen = struct.unpack('B', f.read(1))[0]
                    word = f.read(wlen).decode('utf-8')
                    arr_len = struct.unpack('I', f.read(4))[0]
                    arr = array('I')
                    arr.fromfile(f, arr_len)
                    neuronios[word] = arr.tolist()
                
                return st, l2_mass, rarity, neuronios
        except Exception as e:
            print(f"⚠️ Erro no Boot Auria: {e}")
            return None

# =================================================================
# 3. QUINTIKUS OPEN AURIA (AGI CORE)
# =================================================================
class QuintikusThermal:
    def __init__(self):
        self.st = [0.5, 0.5, 0.5]
        self.mapa = {'bom':0.2, 'feliz':0.3, 'paz':0.2, 'erro':-0.3, 'urgente':-0.4, 'falha':-0.3}

    def processar(self, tokens):
        p, n = 0, 0
        for t in tokens:
            val = self.mapa.get(t, 0)
            if val > 0: p += val
            else: n += abs(val)
        self.st[0] = self.st[0] * 0.9 + (n * 0.1)
        self.st[1] = self.st[1] * 0.9 + (p * 0.1)

    def get_prefixo(self):
        if self.st[0] > 0.6: return "Sob pressão galvânica, "
        if self.st[1] > 0.6: return "Em fluxo de harmonia, "
        return "Pela lógica de solo, "

class QuintikusOpenAuria:
    def __init__(self):
        self.blockchain_file = "brain_auria.qoa"
        self.thermal = QuintikusThermal()
        self.l2_mass = []
        self.neuronios = defaultdict(list)
        self.rarity = {}
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com"}

    def _normalizar(self, txt):
        return "".join(c for c in unicodedata.normalize('NFD', txt.lower()) 
                       if unicodedata.category(c) != 'Mn').strip()

    def inicializar(self, raw_text):
        print("🧠 Amadurecendo nexo galvânico (Auria Mode)...")
        frases = [f.strip() for f in raw_text.split('.') if len(f.strip().split()) > 3]
        palavras_todas = []
        
        for i, f in enumerate(frases):
            clean_f = self._normalizar(f)
            tokens = [t for t in clean_f.split() if t not in self.stop_words]
            self.l2_mass.append(f)
            for t in tokens:
                self.neuronios[t].append(i)
                palavras_todas.append(t)
        
        for t in set(palavras_todas):
            q = palavras_todas.count(t)
            self.rarity[t] = 2.0 / (math.log(q + 1.1) + 1e-5)
            
        # Salva imediatamente após inicializar
        AuriaFS.salvar(self.blockchain_file, self.thermal.st, self.l2_mass, self.rarity, self.neuronios)

    def perguntar(self, entrada):
        t0 = time.perf_counter()
        clean = self._normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words]
        
        self.thermal.processar(tokens)
        pivos = sorted(tokens, key=lambda t: self.rarity.get(t, 0), reverse=True)
        
        if not pivos: return "Nexo carece de solo."

        candidatos = []
        for p in pivos[:2]:
            candidatos.extend(self.neuronios.get(p, []))
        
        # Lei 3: Ação > Descrição
        best_idx = self._aplicar_lei_3(list(set(candidatos)))
        
        if best_idx is not None:
            dt = (time.perf_counter() - t0) * 1000000
            res = self.l2_mass[best_idx]
            return f"\n[AURIA | {dt:.2f}μs | POT:{self.thermal.st[1]*1000:.0f}mV]\n> {self.thermal.get_prefixo()}{res}."
        
        return "Fato não encontrado na matriz."

    def _aplicar_lei_3(self, candidates_idx):
        if not candidates_idx: return None
        scored = []
        estaticos = ['era', 'estava', 'foi', 'tem']
        for idx in candidates_idx:
            frase = self.l2_mass[idx].lower()
            score_acao = sum(2.0 for w in frase.split() if len(w) > 5 and w not in estaticos)
            score_raridade = sum(self.rarity.get(w, 0) for w in frase.split())
            scored.append((idx, score_acao + score_raridade))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def boot(self):
        dados = AuriaFS.carregar(self.blockchain_file)
        if dados:
            self.thermal.st, self.l2_mass, self.rarity, self.neuronios = dados
            print(f"✅ Auria Online: {len(self.l2_mass)} nexos carregados.")
            return True
        return False

# =================================================================
# EXECUÇÃO
# =================================================================
if __name__ == "__main__":
    agi = QuintikusOpenAuria()
    
    if not agi.boot():
        texto_treino = """
        A inteligência artificial QuintikusOpenAuria opera em microsegundos. 
        O motor Auria utiliza persistência binária para economizar memória RAM. 
        A lei número três diz que a ação é sempre superior à descrição estática. 
        O silêncio é a presença de todas as palavras potenciais na matriz. 
        Quando ocorrem erros urgentes, a pressão do sistema aumenta. 
        Sinergia absoluta é o objetivo do processamento galvânico.
        """
        agi.inicializar(texto_treino)

    print("\n=== QUINTIKUS OPEN AURIA ATIVO ===")
    while True:
        u = input("\n👤: ").strip()
        if u.lower() in ['sair', 'exit']: break
        print(agi.perguntar(u))
