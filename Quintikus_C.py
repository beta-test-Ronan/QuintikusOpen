import os
import math
import time
import struct
import random
import json
import unicodedata
import hashlib
import re
import numpy as np
import pickle
from collections import defaultdict, Counter

# =================================================================
# 1. CLASSE CALCULER (CHAIN-OF-THOUGHT & RECURSIVE CACHE)
# =================================================================
class Calculer:
    cache = {}

    @staticmethod
    def eh_matematica(texto):
        # Filtro de detecção: números e operadores, sem letras.
        padrao = re.compile(r'^[0-9\+\-\*\/\(\)\.\s\^]+$')
        return bool(padrao.match(texto)) and any(c in texto for c in "+-*/^")

    @staticmethod
    def resolver(expressao, auria):
        try:
            # Normalização galvânica da expressão
            exp_limpa = expressao.replace(" ", "")
            
            # --- BUSCA POR DEDUÇÃO (Cadeia de Memória) ---
            # Ordenamos o cache pelo tamanho da string para pegar a maior sub-expressão
            sub_conhecida = None
            valor_sub = None
            for chave in sorted(Calculer.cache.keys(), key=len, reverse=True):
                if chave in exp_limpa and len(chave) < len(exp_limpa):
                    sub_conhecida = chave
                    valor_sub = Calculer.cache[chave]
                    break
            
            if sub_conhecida:
                # Substitui o que ela já sabe pelo resultado salvo
                exp_decomposta = exp_limpa.replace(sub_conhecida, str(valor_sub), 1)
                final_res = eval(exp_decomposta.replace('^', '**'), {"__builtins__": None}, {})
                msg = f"LEMBRO que {sub_conhecida} = {valor_sub}, então {expressao} resulta em {final_res} <!>"
                status = "[CALCULER-DEDUCTION]"
            else:
                final_res = eval(exp_limpa.replace('^', '**'), {"__builtins__": None}, {})
                msg = f"O resultado de {expressao} é {final_res}."
                status = "[CALCULER-NEW-FACT]"

            # --- PERSISTÊNCIA DUPLA ---
            Calculer.cache[exp_limpa] = final_res
            # Injeta o cálculo no Solo Quântico para a IA "saber" conversar sobre isso
            auria.amadurecer_solo(f"A conta {expressao} é igual a {final_res}.", silenciar=True)
            
            return f"\n{status}\n> {msg}"
        except Exception as e:
            return f"\n[CALCULER-ERROR]\n> Vácuo lógico na expressão: {e}"

# =================================================================
# 2. SOVEREIGN TOKENIZER + LPS (V14)
# =================================================================
class SovereignTokenizer:
    def __init__(self):
        self.pattern = re.compile(r'\?\?+|\!\!+|\.\.\.+|[:;]-?[)DPpoO]|s2|<3|[\w]+|[\?\!\.]')
        self.special = ["<BOS>", "<EOS>", "<PAD>", "<?>", "<!>", "<.>"]
        
    def tokenize(self, text):
        # Limpeza agressiva de ruídos de arquivo
        text = text.replace('\x00', '').replace('\ufeff', '')
        norm = "".join(c for c in unicodedata.normalize("NFKD", text.lower().strip()) if not unicodedata.combining(c))
        raw_tokens = self.pattern.findall(norm)
        processed = []
        for t in raw_tokens:
            if t == "?": processed.append("<?>")
            elif t == "!": processed.append("<!>")
            elif t == ".": processed.append("<.>")
            else: processed.append(t)
        return processed

# =================================================================
# 3. QUANTUM LPS CORE (V14)
# =================================================================
class QuantumLPSCore:
    def __init__(self, vocab_size, d_model=32):
        self.d_model = d_model
        limit = np.sqrt(6 / (vocab_size + d_model))
        self.embeddings = np.random.uniform(-limit, limit, (vocab_size, d_model)).astype(np.float32)
        self.Wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.Wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_future = np.random.randn(d_model, d_model).astype(np.float32) * 0.2

    def colapsar_nexo(self, q_idx, lps_idx, candidatos_idx_list, rarity_map, word2idx, temp=0.6):
        if not q_idx or not candidatos_idx_list: return None
        q_weights = np.array([rarity_map.get(list(word2idx.keys())[idx], 0.1) for idx in q_idx])
        q_weights /= (q_weights.sum() + 1e-9)
        sujeito_vec = np.average(self.embeddings[q_idx], axis=0, weights=q_weights)
        lps_vec = self.embeddings[lps_idx]
        foco_vec = (sujeito_vec @ self.Wq + lps_vec) @ self.W_future
        
        scores = []
        for c_idx_list in candidatos_idx_list:
            if not c_idx_list: scores.append(-1.0); continue
            c_vec = np.mean(self.embeddings[c_idx_list], axis=0) @ self.Wk
            dot = np.dot(foco_vec, c_vec)
            norm = (np.linalg.norm(foco_vec) * np.linalg.norm(c_vec) + 1e-9)
            scores.append(dot / norm)
        exp_s = np.exp(np.array(scores) / temp)
        return candidatos_idx_list[np.argmax(exp_s / (exp_s.sum() + 1e-10))]

# =================================================================
# 4. QUINTIKUS OPEN AURIA v14.0 (COMPLETE ENGINE)
# =================================================================
class QuintikusOpenAuria:
    def __init__(self):
        self.path = "brain_v14_master.qoa"
        self.tokenizer = SovereignTokenizer()
        self.l2_mass, self.l2_tokens_idx = [], []
        self.neuronios = defaultdict(list)
        self.rarity, self.word2idx, self.ledger = {}, {}, set()
        self.core = None
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com", "na", "no"}

    def amadurecer_solo(self, raw_content, silenciar=False):
        hash_c = hashlib.sha256(raw_content.encode('utf-8', 'ignore')).hexdigest()
        if hash_c in self.ledger: return False
        
        if not silenciar: print(f"🧠 Amadurecendo Solo (v14)...")
        chunks = re.split(r'([\?\!\.])', raw_content)
        all_sentences = [chunks[i] + chunks[i+1] for i in range(0, len(chunks)-1, 2) if len(chunks[i].strip()) > 1]
        
        words_total = (" ".join(all_sentences)).lower().split()
        vocab = sorted(list(set(words_total + self.tokenizer.special + list(self.word2idx.keys()))))
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        contagem = Counter(words_total)
        
        offset = len(self.l2_mass)
        for i, s in enumerate(all_sentences):
            self.l2_mass.append(s)
            tokens = self.tokenizer.tokenize(s)
            self.l2_tokens_idx.append([self.word2idx[t] for t in tokens if t in self.word2idx])
            for t in tokens:
                if t not in self.stop_words and t in self.word2idx:
                    if len(self.neuronios[t]) < 10000: self.neuronios[t].append(offset + i)
                    self.rarity[t] = 2.0 / (math.log(contagem.get(t, 1) + 1.1) + 1e-5)

        self.core = QuantumLPSCore(len(vocab))
        self.ledger.add(hash_c)
        self.selar(silenciar)
        return True

    def selar(self, silenciar=False):
        bundle = {
            'l2': self.l2_mass, 'rar': self.rarity, 'w2i': self.word2idx, 
            'neu': dict(self.neuronios), 'core': self.core, 
            't_idx': self.l2_tokens_idx, 'ledger': self.ledger,
            'm_cache': Calculer.cache
        }
        with open(self.path, 'wb') as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not silenciar: print(f"💾 Estado Selado na Blockchain.")

    def boot(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                b = pickle.load(f)
                self.l2_mass, self.rarity, self.word2idx = b['l2'], b['rar'], b['w2i']
                self.neuronios, self.core, self.l2_tokens_idx = b['neu'], b['core'], b['t_idx']
                self.ledger = b['ledger']
                Calculer.cache = b.get('m_cache', {})
                print(f"✅ Conexão estável: {len(self.l2_mass)} nexos carregados.")
                return True
        return False

    def pensar_e_falar(self, entrada):
        # 1. LAYER MATEMÁTICA RECURSIVA
        if Calculer.eh_matematica(entrada):
            return Calculer.resolver(entrada, self)

        # 2. LAYER QUÂNTICA LPS
        t0 = time.perf_counter()
        u_toks = self.tokenizer.tokenize(entrada)
        if not u_toks: return "[SILÊNCIO]"

        lps_symbol = u_toks[-1] if u_toks[-1] in ["<?>", "<!>", "<.>"] else "<.>"
        lps_idx = self.word2idx.get(lps_symbol, self.word2idx.get("<.>"))
        q_idx = [self.word2idx[t] for t in u_toks if t in self.word2idx]
        pivos = sorted([t for t in u_toks if t in self.rarity], key=lambda x: self.rarity[x], reverse=True)
        
        if not pivos: return "[VÁCUO]"
        candidatos = self.neuronios.get(pivos[0], [])
        if not candidatos: return "[SEM PROTOCOLO NO SOLO]"

        amostra = random.sample(candidatos, min(len(candidatos), 600))
        amostra_idx_list = [self.l2_tokens_idx[i] for i in amostra]
        best_future = self.core.colapsar_nexo(q_idx, lps_idx, amostra_idx_list, self.rarity, self.word2idx)
        final_idx = amostra[amostra_idx_list.index(best_future)]

        raw_res = self.l2_mass[final_idx]
        palavras = raw_res.split()
        res_tokens_idx = self.l2_tokens_idx[final_idx]
        q_vec = np.mean(self.core.embeddings[q_idx], axis=0)
        reacoes = [np.dot(self.core.embeddings[t_id], q_vec) for t_id in res_tokens_idx]
        limiar = np.mean(reacoes) if reacoes else 0
        
        resultado = []
        for i, word in enumerate(palavras):
            t_id = res_tokens_idx[i] if i < len(res_tokens_idx) else None
            if t_id is not None and np.dot(self.core.embeddings[t_id], q_vec) > limiar:
                resultado.append(word.upper())
            else: resultado.append(word.lower())

        dt = (time.perf_counter() - t0) * 1000000
        return f"\n[V14-QUANTUM | {dt:.2f}μs | LPS:{lps_symbol}]\n> {' '.join(resultado)}"

# =================================================================
# EXECUÇÃO PRINCIPAL (AUTO-ENCODER INTEGRADO)
# =================================================================
if __name__ == "__main__":
    auria = QuintikusOpenAuria()
    auria.boot()
    print("init bd...")
    print("help: use train:,trein:,treino: 🧠 Amadurecendo Solo!")
    while True:
        u = input("\n[user]👤: ").strip()
        
        if any(u.startswith(p) for p in ["train:", "trein:", "treino:"]):
            path = u.split(":")[1]
            if os.path.exists(path):
                # TENTA ENCODERS DIFERENTES PARA RECUPERAR BANCO
                conteudo = None
                for encoder in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(path, 'r', encoding=encoder) as f:
                            conteudo = f.read()
                        print(f"✅ Sucesso com encoder: {encoder}")
                        break
                    except UnicodeDecodeError: continue
                
                if conteudo:
                    auria.amadurecer_solo(conteudo)
                else: print("❌ Falha crítica: arquivo em formato desconhecido.")
            continue

        if u.lower() in ['sair', 'exit']: break
        print(auria.pensar_e_falar(u))
