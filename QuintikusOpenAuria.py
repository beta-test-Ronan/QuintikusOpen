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
# 1. KERNEL QMATH & DATA CLEANER
# =================================================================
class DataCleaner:
    @staticmethod
    def normalizar(txt):
        """Remove acentos, converte para minusculo e limpa lixo"""
        if not txt: return ""
        # Normalização Unicode para remover acentos (NFD separa o caractere do acento)
        txt = "".join(c for c in unicodedata.normalize('NFD', txt.lower()) 
                     if unicodedata.category(c) != 'Mn')
        # Remove caracteres de controle e excesso de espaços
        return " ".join(txt.split())

    @staticmethod
    def extrair_texto_json(conteudo_bruto):
        """Detecta se é JSON e extrai os campos relevantes do dataset Cabrita/Alpaca"""
        try:
            dados = json.loads(conteudo_bruto)
            print(f"📦 JSON detectado. Processando {len(dados)} entradas...")
            textos_limpos = []
            for item in dados:
                if isinstance(item, dict):
                    # Junta instrução, entrada e saída para formar o nexo completo
                    instrucao = item.get("instruction", "")
                    entrada = item.get("input", "")
                    saida = item.get("output", "")
                    textos_limpos.append(f"{instrucao} {entrada} {saida}")
            return " . ".join(textos_limpos)
        except json.JSONDecodeError:
            print("📄 Formato de texto plano detectado.")
            return conteudo_bruto

# =================================================================
# 2. AURIA FS - PERSISTÊNCIA BINÁRIA CRUA (V2 - LONG TOKENS)
# =================================================================
class AuriaFS:
    @staticmethod
    def salvar(filepath, st, l2_mass, rarity, neuronios):
        t_ini = time.perf_counter()
        with open(filepath, 'wb') as f:
            f.write(b'QOA2') 
            f.write(struct.pack('3f', *st))
            
            # L2 Mass
            f.write(struct.pack('I', len(l2_mass)))
            for frase in l2_mass:
                b_frase = frase.encode('utf-8')
                f.write(struct.pack('I', len(b_frase))) 
                f.write(b_frase)
            
            # Rarity Map (H para suportar tokens longos de datasets de código)
            f.write(struct.pack('I', len(rarity)))
            for word, val in rarity.items():
                b_word = word.encode('utf-8')
                f.write(struct.pack('H', len(b_word))) 
                f.write(b_word)
                f.write(struct.pack('f', val))
            
            # Neurônios
            f.write(struct.pack('I', len(neuronios)))
            for word, indices in neuronios.items():
                b_word = word.encode('utf-8')
                f.write(struct.pack('H', len(b_word))) 
                f.write(b_word)
                
                arr = array('I', indices)
                f.write(struct.pack('I', len(arr)))
                arr.tofile(f)
        
        t_fim = time.perf_counter()
        print(f"💾 QOA: Salvo em {t_fim - t_ini:.2f}s | Tamanho: {os.path.getsize(filepath)/1024/1024:.2f} MB")

    @staticmethod
    def carregar(filepath):
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, 'rb') as f:
                if f.read(4) != b'QOA2': return None
                st = list(struct.unpack('3f', f.read(12)))
                l2_count = struct.unpack('I', f.read(4))[0]
                l2_mass = [None] * l2_count
                for i in range(l2_count):
                    flen = struct.unpack('I', f.read(4))[0]
                    l2_mass[i] = f.read(flen).decode('utf-8')
                
                r_count = struct.unpack('I', f.read(4))[0]
                rarity = {}
                for _ in range(r_count):
                    wlen = struct.unpack('H', f.read(2))[0]
                    word = f.read(wlen).decode('utf-8')
                    rarity[word] = struct.unpack('f', f.read(4))[0]
                
                n_count = struct.unpack('I', f.read(4))[0]
                neuronios = defaultdict(list)
                for _ in range(n_count):
                    wlen = struct.unpack('H', f.read(2))[0]
                    word = f.read(wlen).decode('utf-8')
                    arr_len = struct.unpack('I', f.read(4))[0]
                    arr = array('I')
                    arr.fromfile(f, arr_len)
                    neuronios[word] = arr.tolist()
                
                return st, l2_mass, rarity, neuronios
        except: return None

# =================================================================
# 3. QUINTIKUS OPEN AURIA - ENGINE
# =================================================================
class QuintikusOpenAuria:
    def __init__(self):
        self.blockchain_file = "brain_auria.qoa"
        self.st = [0.5, 0.5, 0.5]
        self.l2_mass = []
        self.neuronios = defaultdict(list)
        self.rarity = {}
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com", "no", "na"}

    def inicializar(self, conteudo_bruto):
        t_start = time.perf_counter()
        
        # 1. Data Cleaning & Extração JSON
        texto_limpo = DataCleaner.extrair_texto_json(conteudo_bruto)
        
        print("🧠 Amadurecendo Nexo (Auria Turbo)...")
        # Split por nexo (ponto final)
        frases = [f.strip() for f in texto_limpo.split('.') if len(f.strip().split()) > 3]
        total_f = len(frases)
        print(f"📊 Corpus: {total_f} nexos.")

        contagem_global = Counter()
        
        for i, f in enumerate(frases):
            clean_f = DataCleaner.normalizar(f)
            # Filtro de sanidade para tokens
            tokens = [t[:250] for t in clean_f.split() if t not in self.stop_words and 2 < len(t) < 500]
            
            self.l2_mass.append(f)
            
            for t in tokens:
                # O(1) de inserção, Saturação de nexo em 10k para evitar RAM leak
                if len(self.neuronios[t]) < 10000:
                    self.neuronios[t].append(i)
                contagem_global[t] += 1
            
            if i % 20000 == 0 and i > 0:
                print(f"  > Processado: {i}/{total_f} ({i/total_f*100:.1f}%)")

        print("⚖️ Calculando Pesos de Raridade...")
        for t, q in contagem_global.items():
            self.rarity[t] = 2.0 / (math.log(q + 1.1) + 1e-5)
            
        t_end = time.perf_counter()
        print(f"✅ Treino concluído em {t_end - t_start:.2f}s")
        AuriaFS.salvar(self.blockchain_file, self.st, self.l2_mass, self.rarity, self.neuronios)

    def perguntar(self, entrada):
        t0 = time.perf_counter()
        clean = DataCleaner.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words]
        
        # Busca pivos por raridade
        pivos = sorted(tokens, key=lambda t: self.rarity.get(t, 0), reverse=True)
        if not pivos: return "Nexo carece de solo."

        candidatos = []
        for p in pivos[:2]:
            candidatos.extend(self.neuronios.get(p, []))
        
        if not candidatos: return "Nexo não encontrado."

        # Amostragem para manter a resposta em microsegundos
        busca = list(set(candidatos))
        if len(busca) > 1000: busca = random.sample(busca, 1000)

        best_idx = None
        max_score = -1
        for idx in busca:
            # Score de relevância
            score = sum(self.rarity.get(w, 0.05) for w in DataCleaner.normalizar(self.l2_mass[idx]).split())
            if score > max_score:
                max_score = score
                best_idx = idx
        
        dt = (time.perf_counter() - t0) * 1000000
        return f"\n[{dt:.2f}μs | POT:{max_score*100:.0f}mV]\n> {self.l2_mass[best_idx]}"

    def boot(self):
        dados = AuriaFS.carregar(self.blockchain_file)
        if dados:
            self.st, self.l2_mass, self.rarity, self.neuronios = dados
            print(f"✅ Auria Online: {len(self.l2_mass)} nexos carregados.")
            return True
        return False

# =================================================================
# EXECUÇÃO PRINCIPAL
# =================================================================
if __name__ == "__main__":
    agi = QuintikusOpenAuria()
    
    # Tenta carregar o cérebro binário
    if not agi.boot():
        try:
            print("📂 Carregando dataset...")
            # Detecta automaticamente se existe o cabrita ou usa o teste
            nome_arquivo = 'dataset-52k.json' if os.path.exists('cabrita-dataset-52k.json') else 'texto.txt'
            
            with open(nome_arquivo, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            agi.inicializar(conteudo)
        except Exception as e:
            print(f"⚠️ Falha ao ler dataset: {e}")
            agi.inicializar("Solo de emergência. A Quintikus está online.")

    print("\n=== QUINTIKUS OPEN AURIA ATIVO ===")
    while True:
        u = input("\n👤: ").strip()
        if u.lower() in ['sair', 'exit', 'quit']: break
        if u: print(agi.perguntar(u))
