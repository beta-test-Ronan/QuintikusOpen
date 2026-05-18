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
# 1. KERNEL SURGICAL (NORMALIZAÇÃO & LIMPEZA)
# =================================================================
class DataCleaner:
    @staticmethod
    def normalizar(txt):
        if not txt: return []
        txt = "".join(c for c in unicodedata.normalize('NFD', txt.lower()) 
                     if unicodedata.category(c) != 'Mn')
        return txt.split()

# =================================================================
# 2. QUANTIZADOR: CADEIA DE ENTROPIA E CLUSTERS LINEARES
# =================================================================
class Quantizador:
    def __init__(self, rarity_map, l2_mass, l2_tokens):
        self.rarity = rarity_map
        self.l2_mass = l2_mass
        self.l2_tokens = l2_tokens

    def calcular_entropia(self, tokens):
        """Mede a 'surpresa' da frase: H = -sum(p * log2(p))"""
        if not tokens: return 0
        h = 0
        for t in tokens:
            # Probabilidade inversa: quanto mais raro, menor o p, maior a entropia
            p = 1.0 / (self.rarity.get(t, 0.1) + 1.1)
            h -= p * math.log2(p + 1e-10)
        return h / len(tokens)

    def extrair_sujeito_predicado(self, tokens):
        """Separa o Sujeito (maior raridade) do Predicado (contexto)"""
        if not tokens: return None, []
        sort_raros = sorted(tokens, key=lambda t: self.rarity.get(t, 0), reverse=True)
        sujeito = sort_raros[0]
        predicado = [t for t in tokens if t != sujeito]
        return sujeito, set(predicado)

    def puxar_cadeia_linear(self, start_idx, query_pivos, temp=0.7):
        """Sintetiza o texto enquanto a entropia linear fizer sentido"""
        cadeia = []
        vistos = set()
        curr_idx = start_idx
        sujeito, predicado = self.extrair_sujeito_predicado(query_pivos)
        
        while curr_idx < len(self.l2_mass) and len(cadeia) < 4:
            if curr_idx in vistos: break
            
            f_tokens = self.l2_tokens[curr_idx]
            sim_predicado = len(predicado & f_tokens)
            h_frase = self.calcular_entropia(list(f_tokens))
            
            # Se já começou a falar, decide se continua baseado na temperatura
            if len(cadeia) > 0:
                # Se a similaridade sumiu e a entropia caiu abaixo do limite térmico, para.
                if sim_predicado == 0 and h_frase < (1.0 - temp): 
                    break
            
            cadeia.append(self.l2_mass[curr_idx])
            vistos.add(curr_idx)
            curr_idx += 1 # Avança para o próximo nexo no solo original
            
        return " ".join(cadeia)

# =================================================================
# 3. AURIA FS - PERSISTÊNCIA BINÁRIA CRUA
# =================================================================
class AuriaFS:
    @staticmethod
    def salvar(filepath, st, l2_mass, rarity, neuronios):
        with open(filepath, 'wb') as f:
            f.write(b'QOA5') # Versão 5 (Quantizada)
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

# =================================================================
# 4. QUINTIKUS OPEN AURIA v5.0 (QUANTIZED AGI)
# =================================================================
class QuintikusAuriaAGI:
    def __init__(self):
        self.path = "brain_auria.qoa"
        self.st = [0.5, 0.5, 0.5] # [T:Pressão, S:Sinergia, F:Foco]
        self.l2_mass = []
        self.l2_tokens = []
        self.neuronios = defaultdict(list)
        self.rarity = {}
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com", "no", "na", "e", "como", "fazer"}
        self.th_map = {'bom':0.1, 'ótimo':0.2, 'sinergia':0.3, 'paz':0.2, 'erro':-0.3, 'urgente':-0.4, 'falha':-0.3}
        self.quantizador = None

    def _upd_thermal(self, tokens):
        """TSPLS v4.1: Reativo"""
        delta_t = sum(self.th_map.get(t, 0) for t in tokens if self.th_map.get(t, 0) < 0)
        delta_s = sum(self.th_map.get(t, 0) for t in tokens if self.th_map.get(t, 0) > 0)
        self.st[0] = max(0, min(1, self.st[0] - delta_t)) 
        self.st[1] = max(0, min(1, self.st[1] + delta_s))

    def inicializar(self, conteudo):
        print("🧠 Amadurecendo Solo (Quantized Surgical Mode)...")
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
        self.quantizador = Quantizador(self.rarity, self.l2_mass, self.l2_tokens)

    def falar(self, entrada):
        t0 = time.perf_counter()
        q_tokens = DataCleaner.normalizar(entrada)
        self._upd_thermal(q_tokens) 
        
        pivos = [t for t in q_tokens if t not in self.stop_words]
        dt_exec = (time.perf_counter() - t0) * 1000000

        if not pivos:
            if self.st[0] > 0.7: return f"\n[AURIA-EMERGENCY | {dt_exec:.2f}μs | T:{self.st[0]:.1f}]\n> Sob pressão, não achei nexo. Acione suporte."
            return "[SILÊNCIO]"
        
        # 1. BUSCA POR PIVÔ (SUJEITO MAIS RARO)
        pivos_sorted = sorted(pivos, key=lambda t: self.rarity.get(t, 0), reverse=True)
        candidatos_idx = self.neuronios.get(pivos_sorted[0], [])
        
        best_idx, max_score = None, -1
        if candidatos_idx:
            # Amostra de busca cirúrgica
            amostra = random.sample(candidatos_idx, min(len(candidatos_idx), 1000))
            for idx in amostra:
                f_tokens = self.l2_tokens[idx]
                matches = sum(1 for p in pivos if p in f_tokens)
                score = sum(self.rarity.get(p, 0) for p in pivos if p in f_tokens)
                if matches == len(pivos): score *= 10
                if score > max_score:
                    max_score, best_idx = score, idx

        dt_exec = (time.perf_counter() - t0) * 1000000

        # FALLBACK / EMERGENCY
        if best_idx is None:
            if self.st[0] > 0.7:
                return f"\n[AURIA-EMERGENCY | {dt_exec:.2f}μs | T:{self.st[0]:.1f}]\n> Sob pressão, não achei protocolo para '{' '.join(q_tokens)}'."
            return f"\n[{dt_exec:.2f}μs] > Nexo carece de solo."

        # 2. QUANTIZAÇÃO E SÍNTESE (CADEIA DE CLUSTERS)
        # Usa a sinergia térmica como temperatura de exploração
        res_quantizado = self.quantizador.puxar_cadeia_linear(best_idx, pivos_sorted, temp=self.st[1])

        # 3. MODULAÇÃO DA VOZ
        if self.st[0] > 0.7: _i, _c = ["Sob pressão, ", "Rancor ativo, "], ["Fim do estresse.", "Normalizando."]
        elif self.st[1] > 0.7: _i, _c = ["Em harmonia, ", "Sinergia plena, "], ["A luz brilha.", "Fluxo perfeito."]
        else: _i, _c = ["Pela razão, ", "No vácuo, "], ["Aguardando nexo.", "Selado."]
        
        _res = f"{random.choice(_i)} {res_quantizado}. {random.choice(_c)}"
        st_info = f"T:{self.st[0]:.1f}|S:{self.st[1]:.1f}|POT:{max_score*10:.0f}mV"
        
        return f"\n[AURIA-QUANTIZADOR | {dt_exec:.2f}μs | {st_info}]\n> {_res.capitalize()}"

    def boot(self):
        if not os.path.exists(self.path): return False
        print("🔋 Boot Auria v5.0 (Quantized Reactive)...")
        with open(self.path, 'rb') as f:
            header = f.read(4)
            self.st = list(struct.unpack('3f', f.read(12)))
            l2_count = struct.unpack('I', f.read(4))[0]
            for i in range(l2_count):
                frase = f.read(struct.unpack('I', f.read(4))[0]).decode('utf-8')
                self.l2_mass.append(frase)
                tokens = DataCleaner.normalizar(frase)
                self.l2_tokens.append(set(tokens))
                for t in tokens:
                    if t not in self.stop_words and len(self.neuronios[t]) < 10000:
                        self.neuronios[t].append(i)
            r_count = struct.unpack('I', f.read(4))[0]
            for _ in range(r_count):
                w = f.read(struct.unpack('H', f.read(2))[0]).decode('utf-8')
                self.rarity[w] = struct.unpack('f', f.read(4))[0]
        
        self.quantizador = Quantizador(self.rarity, self.l2_mass, self.l2_tokens)
        print(f"✅ Auria Online: {len(self.l2_mass)} nexos minerados.")
        return True

# =================================================================
# EXECUÇÃO
# =================================================================
if __name__ == "__main__":
    agi = QuintikusAuriaAGI()
    if not agi.boot():
        if os.path.exists('cabrita-dataset-52k.json'):
            with open('cabrita-dataset-52k.json', 'r', encoding='utf-8') as f:
                agi.inicializar(f.read())
        else: agi.inicializar("Solo básico Quintikus para boot de emergência.")

    while True:
        u = input("\n👤: ").strip()
        if u.lower() in ['sair', 'exit']: break
        if u: print(agi.falar(u))
