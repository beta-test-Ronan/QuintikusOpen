import os
import sys
import math
import time
import struct
import random
import json
import unicodedata
import numpy as np
from array import array
from collections import defaultdict, Counter

# =================================================================
# 1. KERNEL SURGICAL & DATA CLEANER
# =================================================================
class DataCleaner:
    @staticmethod
    def normalizar(txt):
        if not txt: return []
        txt = "".join(c for c in unicodedata.normalize('NFD', txt.lower()) 
                     if unicodedata.category(c) != 'Mn')
        return txt.split()

# =================================================================
# 2. SOVEREIGN THINKER: ATENÇÃO LINEAR PROBABILÍSTICA (Q, K, V)
# =================================================================
class SovereignThinker:
    def __init__(self, vocab_size, d_model=32):
        self.d_model = d_model
        # Inicialização de Xavier/Glorot para estabilidade galvânica
        limit = np.sqrt(6 / (vocab_size + d_model))
        self.embeddings = np.random.uniform(-limit, limit, (vocab_size, d_model))
        
        # Matrizes de Projeção (Sujeito, Predicado, Valor)
        self.Wq = np.random.randn(d_model, d_model) * 0.1 # Query
        self.Wk = np.random.randn(d_model, d_model) * 0.1 # Key
        self.Wv = np.random.randn(d_model, d_model) * 0.1 # Value

    def calcular_atencao(self, q_idx, doc_idx):
        """Calcula o foco vetorial: Sujeito x Predicado"""
        if not q_idx or not doc_idx: return 0
        
        # Projeta a Query (Sujeito da pergunta)
        q_vec = np.mean(self.embeddings[q_idx], axis=0) @ self.Wq
        
        # Projeta as Keys (Tokens do documento/nexo)
        k_vecs = self.embeddings[doc_idx] @ self.Wk
        
        # Dot product de Atenção (Similaridade de Cosseno Linear)
        scores = np.dot(k_vecs, q_vec) / np.sqrt(self.d_model)
        # Softmax simplificado (Entropia de Ativação)
        exp_scores = np.exp(scores - np.max(scores))
        atencao = exp_scores / exp_scores.sum()
        
        # Retorna a força do nexo (Soma ponderada da similaridade)
        return np.max(atencao) * np.sum(scores)

# =================================================================
# 3. QUANTIZADOR: CADEIA DE ENTROPIA
# =================================================================
class Quantizador:
    def __init__(self, rarity_map, l2_mass, l2_tokens_idx):
        self.rarity = rarity_map
        self.l2_mass = l2_mass
        self.l2_tokens_idx = l2_tokens_idx

    def calcular_entropia(self, tokens_idx):
        if not tokens_idx: return 0
        h = 0
        for idx in tokens_idx:
            # Probabilidade baseada na raridade indexada
            p = 1.0 / (idx + 1.1) 
            h -= p * math.log2(p + 1e-10)
        return h / len(tokens_idx)

    def puxar_cadeia_linear(self, start_idx, thinker, q_idx, temp=0.7):
        cadeia = []
        curr_idx = start_idx
        vistos = set()
        
        while curr_idx < len(self.l2_mass) and len(cadeia) < 3:
            if curr_idx in vistos: break
            
            f_idx = self.l2_tokens_idx[curr_idx]
            # Usa o Thinker para validar se a continuidade faz sentido
            att = thinker.calcular_atencao(q_idx, f_idx)
            
            if len(cadeia) > 0 and att < (1.0 - temp): break
            
            cadeia.append(self.l2_mass[curr_idx])
            vistos.add(curr_idx)
            curr_idx += 1
            
        return " ".join(cadeia)

# =================================================================
# 4. QUINTIKUS OPEN AURIA v6.0 (SOVEREIGN THINKER)
# =================================================================
class QuintikusAuriaAGI:
    def __init__(self):
        self.path = "brain_auria.qoa"
        self.st = [0.5, 0.5, 0.5]
        self.l2_mass = []
        self.l2_tokens_idx = [] 
        self.neuronios = defaultdict(list)
        self.rarity = {}
        self.word2idx = {}
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com", "no", "na", "e"}
        self.thinker = None
        self.quantizador = None

    def inicializar(self, conteudo):
        print("🧠 Amadurecendo Solo Soberano (Attention Linear Mode)...")
        try:
            dados = json.loads(conteudo)
            texto = " . ".join([f"{i.get('instruction','')} {i.get('input','')} {i.get('output','')}" for i in dados])
        except: texto = conteudo

        sentences = [s.strip() for s in texto.lower().replace('\n', ' ').split('.') if len(s.strip()) > 5]
        words_total = (" . ".join(sentences) + " .").split()
        vocab = sorted(list(set(words_total)))
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        
        contagem = Counter(words_total)
        for i, f in enumerate(sentences):
            self.l2_mass.append(f)
            tokens = DataCleaner.normalizar(f)
            # Converte tokens em IDs para o Thinker
            self.l2_tokens_idx.append([self.word2idx[t] for t in tokens if t in self.word2idx])
            
            for t in tokens:
                if t not in self.stop_words:
                    if len(self.neuronios[t]) < 10000: self.neuronios[t].append(i)
                    self.rarity[t] = 1.0 / (contagem[t] + 1e-5)

        # Inicializa o Thinker com os Embeddings de Solo
        self.thinker = SovereignThinker(len(vocab))
        self.quantizador = Quantizador(self.rarity, self.l2_mass, self.l2_tokens_idx)
        print(f"✅ Auria Soberana Online: {len(vocab)} neurônios vetoriais.")

    def falar(self, entrada):
        t0 = time.perf_counter()
        q_tokens = DataCleaner.normalizar(entrada)
        q_idx = [self.word2idx[t] for t in q_tokens if t in self.word2idx]
        
        pivos = sorted(q_tokens, key=lambda t: self.rarity.get(t, 0), reverse=True)
        if not pivos: return "[SILÊNCIO]"

        # 1. PEGA CANDIDATOS (PILHA DE LIXO)
        candidatos_idx = self.neuronios.get(pivos[0], [])
        if not candidatos_idx: return "[SEM NEXO]"

        # 2. THINKER: ATENÇÃO LINEAR SOBRE A PILHA
        best_idx = None
        max_att = -float('inf')
        
        amostra = random.sample(candidatos_idx, min(len(candidatos_idx), 500))
        for idx in amostra:
            doc_idx = self.l2_tokens_idx[idx]
            # Atenção: Pergunta (Query) x Documento (Key)
            att_score = self.thinker.calcular_atencao(q_idx, doc_idx)
            
            # Adiciona viés de raridade (IDF)
            rarity_bias = sum(self.rarity.get(t, 0) for t in q_tokens if self.word2idx.get(t) in doc_idx)
            final_score = att_score + (rarity_bias * 20)

            if final_score > max_att:
                max_att = final_score
                best_idx = idx

        dt = (time.perf_counter() - t0) * 1000000

        # 3. QUANTIZAÇÃO E SÍNTESE
        if best_idx is not None:
            res_quantizado = self.quantizador.puxar_cadeia_linear(best_idx, self.thinker, q_idx, temp=self.st[1])
            return f"\n[THINK-FLOW | {dt:.2f}μs | ATT:{max_att:.2f}]\n> {res_quantizado.capitalize()}."
        
        return f"[{dt:.2f}μs] > Sem protocolo."

if __name__ == "__main__":
    agi = QuintikusAuriaAGI()
    # Carregue seu JSON aqui
    if os.path.exists('dataset-52k.json'):
        with open('dataset-52k.json', 'r', encoding='utf-8') as f:
            agi.inicializar(f.read())
    
    while True:
        u = input("\n👤: ").strip()
        if u.lower() in ['sair', 'exit']: break
        print(agi.falar(u))
