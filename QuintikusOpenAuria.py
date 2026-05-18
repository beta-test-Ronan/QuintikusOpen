import os
import sys
import math
import time
import struct
import random
import json
import unicodedata
import hashlib
import numpy as np
import pickle
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
        limit = np.sqrt(6 / (vocab_size + d_model))
        self.embeddings = np.random.uniform(-limit, limit, (vocab_size, d_model)).astype(np.float32)
        self.Wq = (np.random.randn(d_model, d_model) * 0.1).astype(np.float32)
        self.Wk = (np.random.randn(d_model, d_model) * 0.1).astype(np.float32)

    def calcular_atencao(self, q_idx, doc_idx):
        if not q_idx or not doc_idx: return 0
        try:
            q_vec = np.mean(self.embeddings[q_idx], axis=0) @ self.Wq
            k_vecs = self.embeddings[doc_idx] @ self.Wk
            scores = np.dot(k_vecs, q_vec) / np.sqrt(self.d_model)
            exp_scores = np.exp(scores - np.max(scores))
            atencao = exp_scores / (exp_scores.sum() + 1e-10)
            return np.max(atencao) * np.sum(scores)
        except: return 0

# =================================================================
# 3. QUANTIZADOR: CADEIA DE ENTROPIA
# =================================================================
class Quantizador:
    def __init__(self, rarity_map, l2_mass, l2_tokens_idx):
        self.rarity = rarity_map
        self.l2_mass = l2_mass
        self.l2_tokens_idx = l2_tokens_idx

    def puxar_cadeia_linear(self, start_idx, thinker, q_idx, temp=0.7):
        cadeia = []
        curr_idx = start_idx
        vistos = set()
        while curr_idx < len(self.l2_mass) and len(cadeia) < 3:
            if curr_idx in vistos: break
            f_idx = self.l2_tokens_idx[curr_idx]
            att = thinker.calcular_atencao(q_idx, f_idx)
            if len(cadeia) > 0 and att < (1.0 - temp): break
            cadeia.append(self.l2_mass[curr_idx])
            vistos.add(curr_idx)
            curr_idx += 1
        return " ".join(cadeia)

# =================================================================
# 4. SOVEREIGN BLOCKCHAIN (GERENCIADOR BINÁRIO COM LEDGER)
# =================================================================
class SovereignBlockchain:
    def __init__(self, filename="brain_auria.qoa"):
        self.filename = filename

    def selar_e_salvar(self, bundle):
        """Salva o estado completo do cérebro e o histórico de treino"""
        print(f"💾 Selando cérebro binário em '{self.filename}'...")
        with open(self.filename, 'wb') as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Estado binário persistido com sucesso.")

    def carregar_estado(self):
        """Puxa os dados binários se o arquivo existir"""
        if os.path.exists(self.filename):
            print(f"📂 Puxando dados binários...")
            with open(self.filename, 'rb') as f:
                return pickle.load(f)
        return None

# =================================================================
# 5. QUINTIKUS OPEN AURIA v8.0 (SOVEREIGN AGGREGATOR)
# =================================================================
class QuintikusAuriaAGI:
    def __init__(self):
        self.blockchain = SovereignBlockchain()
        # Estados do Motor
        self.st = [0.5, 0.5, 0.5]
        self.l2_mass = []
        self.l2_tokens_idx = [] 
        self.neuronios = defaultdict(list)
        self.rarity = {}
        self.word2idx = {}
        self.ledger_treino = set() # Lista de hashes de arquivos já treinados
        
        self.thinker = None
        self.quantizador = None
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com"}

    def _gerar_hash_conteudo(self, texto):
        return hashlib.sha256(texto.encode('utf-8')).hexdigest()

    def inicializar_novo_solo(self, conteudo_bruto):
        """Amadurece solo novo e sela no binário"""
        hash_arquivo = self._gerar_hash_conteudo(conteudo_bruto)
        
        if hash_arquivo in self.ledger_treino:
            print("🚫 BLOQUEIO: Este modelo/arquivo já foi processado e selado anteriormente.")
            return False

        print(f"🧠 Amadurecendo novo nexo (Hash: {hash_arquivo[:12]})...")
        try:
            dados = json.loads(conteudo_bruto)
            texto = " . ".join([f"{i.get('instruction','')} {i.get('input','')} {i.get('output','')}" for i in dados])
        except: texto = conteudo_bruto

        sentences = [s.strip() for s in texto.lower().replace('\n', ' ').split('.') if len(s.strip()) > 5]
        words_all = (" . ".join(sentences) + " .").split()
        
        # Atualiza vocabulário global
        vocab = sorted(list(set(words_all + list(self.word2idx.keys()))))
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        
        contagem = Counter(words_all)
        offset = len(self.l2_mass)
        for i, f in enumerate(sentences):
            self.l2_mass.append(f)
            tokens = DataCleaner.normalizar(f)
            self.l2_tokens_idx.append([self.word2idx[t] for t in tokens if t in self.word2idx])
            
            for t in tokens:
                if t not in self.stop_words:
                    if len(self.neuronios[t]) < 10000: self.neuronios[t].append(offset + i)
                    self.rarity[t] = 1.0 / (contagem[t] + 1e-5)

        # Re-inicializa Thinker e Quantizador com novo tamanho de vocabulário
        self.thinker = SovereignThinker(len(vocab))
        self.quantizador = Quantizador(self.rarity, self.l2_mass, self.l2_tokens_idx)
        
        # Adiciona ao Ledger e Sela
        self.ledger_treino.add(hash_arquivo)
        
        bundle = {
            'l2': self.l2_mass, 'rarity': self.rarity, 'w2idx': self.word2idx,
            'neu': dict(self.neuronios), 'thinker': self.thinker, 
            'tokens_idx': self.l2_tokens_idx, 'ledger': self.ledger_treino
        }
        self.blockchain.selar_e_salvar(bundle)
        return True

    def falar(self, entrada):
        t0 = time.perf_counter()
        q_tokens = DataCleaner.normalizar(entrada)
        q_idx = [self.word2idx[t] for t in q_tokens if t in self.word2idx]
        pivos = sorted(q_tokens, key=lambda t: self.rarity.get(t, 0), reverse=True)
        
        if not pivos or not self.thinker: return "[SISTEMA EM ESPERA]"

        candidatos_idx = self.neuronios.get(pivos[0], [])
        if not candidatos_idx: return "[SOLO DESCONHECIDO]"

        best_idx, max_att = None, -float('inf')
        amostra = random.sample(candidatos_idx, min(len(candidatos_idx), 500))
        for idx in amostra:
            doc_idx = self.l2_tokens_idx[idx]
            att_score = self.thinker.calcular_atencao(q_idx, doc_idx)
            # Bias de solo
            rarity_bias = sum(self.rarity.get(t, 0) for t in q_tokens if t in self.l2_mass[idx].lower())
            final_score = att_score + (rarity_bias * 20)
            if final_score > max_att:
                max_att, best_idx = final_score, idx

        dt = (time.perf_counter() - t0) * 1000000
        if best_idx is not None:
            res_quantizado = self.quantizador.puxar_cadeia_linear(best_idx, self.thinker, q_idx)
            return f"\n[AURIA-V8 | {dt:.2f}μs | ATT:{max_att:.2f}]\n> {res_quantizado.capitalize()}."
        
        return "Nexo rompido."

    def boot(self):
        """Carrega a Blockchain binária"""
        bundle = self.blockchain.carregar_estado()
        if bundle:
            self.l2_mass = bundle['l2']
            self.rarity = bundle['rarity']
            self.word2idx = bundle['w2idx']
            self.neuronios = defaultdict(list, bundle['neu'])
            self.thinker = bundle['thinker']
            self.l2_tokens_idx = bundle['tokens_idx']
            self.ledger_treino = bundle['ledger']
            self.quantizador = Quantizador(self.rarity, self.l2_mass, self.l2_tokens_idx)
            print(f"✅ Auria Online via Binário. {len(self.l2_mass)} nexos carregados.")
            print(f"📜 Ledger: {len(self.ledger_treino)} modelos já integrados.")
            return True
        return False

# =================================================================
# EXECUÇÃO
# =================================================================
if __name__ == "__main__":
    agi = QuintikusAuriaAGI()
    agi.boot()
    
    while True:
        u = input("\n👤 (Comando ou Texto): ").strip()
        
        # Comando para treinar múltiplos modelos/arquivos
        # Uso: train:arquivo1.json
        if u.startswith("train:"):
            file_path = u.split(":")[1]
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    if agi.inicializar_novo_solo(f.read()):
                        print(f"✨ Conhecimento de '{file_path}' selado com sucesso.")
            else:
                print(f"❌ Arquivo '{file_path}' não encontrado.")
            continue

        if u.lower() in ['sair', 'exit']: break
        if u: print(agi.falar(u))
