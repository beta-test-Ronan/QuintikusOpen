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
import platform
from collections import defaultdict, Counter

# =================================================================
# 1. SOVEREIGN CRYPT (XOR CIPHER + HARDWARE ANCHOR)
# =================================================================
class SovereignCrypt:
    @staticmethod
    def get_key(name):
        # platform.node() pega o nome do seu PC/Celular no sistema
        salt = platform.node() or "ROOT_NEXUS"
        return hashlib.sha256((name + salt).encode()).digest()

    @staticmethod
    def xor_cipher(text, key):
        res = []
        for i, char in enumerate(text):
            res.append(chr(ord(char) ^ key[i % len(key)]))
        return "".join(res)

# =================================================================
# 2. USER SOVEREIGN CHAIN (BLOCKCHAIN DE INTIMIDADE)
# =================================================================
class UserSovereignChain:
    def __init__(self, filename="user.bin"):
        self.filename = filename
        self.history = [] 
        self.current_pil = 0.0
        self.user_name = None

    def salvar(self):
        data = {'history': self.history, 'pil': self.current_pil, 'name': self.user_name}
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f)

    def carregar(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'rb') as f:
                    data = pickle.load(f)
                    self.history = data.get('history', [])
                    self.current_pil = data.get('pil', 0.0)
                    self.user_name = data.get('name', None)
                    return self.current_pil, self.user_name
            except: pass
        return 0.0, None

# =================================================================
# 3. CLASSES DE SUPORTE (TOKENIZER, CORE, CALCULER)
# =================================================================
class SovereignTokenizer:
    def __init__(self):
        self.pattern = re.compile(r'\?\?+|\!\!+|\.\.\.+|[:;]-?[)DPpoO]|s2|<3|\^\^|[\w]+|[\?\!\.]')
        self.special = ["<BOS>", "<EOS>", "<PAD>", "<?>", "<!>", "<.>"]
    def tokenize(self, text):
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

class QuantumLPSCore:
    def __init__(self, vocab_size, d_model=32):
        self.d_model = d_model
        limit = np.sqrt(6 / (vocab_size + d_model))
        self.embeddings = np.random.uniform(-limit, limit, (vocab_size, d_model)).astype(np.float32)
        self.Wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.Wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_future = np.random.randn(d_model, d_model).astype(np.float32) * 0.2

    def colapsar_nexo(self, q_idx, lps_idx, candidatos_idx_list, rarity_map, word2idx):
        if not q_idx or not candidatos_idx_list: return None, 0
        q_weights = np.array([rarity_map.get(list(word2idx.keys())[idx], 0.1) for idx in q_idx])
        q_weights /= (q_weights.sum() + 1e-9)
        sujeito_vec = np.average(self.embeddings[q_idx], axis=0, weights=q_weights)
        lps_vec = self.embeddings[lps_idx]
        foco_vec = (sujeito_vec @ self.Wq + lps_vec) @ self.W_future
        scores = [np.dot(foco_vec, np.mean(self.embeddings[c], axis=0) @ self.Wk) for c in candidatos_idx_list]
        max_i = np.argmax(scores)
        return candidatos_idx_list[max_i], scores[max_i]

class Calculer:
    cache = {}
    @staticmethod
    def eh_matematica(texto):
        padrao = re.compile(r'^[0-9\+\-\*\/\(\)\.\s\^]+$')
        return bool(padrao.match(texto)) and any(c in texto for c in "+-*/^")
    @staticmethod
    def resolver(expressao, auria):
        try:
            exp_limpa = expressao.replace(" ", "")
            sub_conhecida = None
            for chave in sorted(Calculer.cache.keys(), key=len, reverse=True):
                if chave in exp_limpa and len(chave) < len(exp_limpa):
                    sub_conhecida = chave
                    break
            if sub_conhecida:
                exp_para_calc = exp_limpa.replace(sub_conhecida, str(Calculer.cache[sub_conhecida]), 1)
                final_res = eval(exp_para_calc.replace('^', '**'), {"__builtins__": None}, {})
                msg = f"LEMBRO que {sub_conhecida}={Calculer.cache[sub_conhecida]}, então {expressao}={final_res} <!>"
            else:
                final_res = eval(exp_limpa.replace('^', '**'), {"__builtins__": None}, {})
                msg = f"O resultado de {expressao} é {final_res}."
            Calculer.cache[exp_limpa] = final_res
            auria.amadurecer_solo(f"{expressao} = {final_res}.", auth=2, silenciar=True)
            return f"\n[CALCULER]\n> {msg}"
        except Exception as e: return f"\n[MATH-ERROR]: {e}"

# =================================================================
# 4. QUINTIKUS OPEN AURIA v18.0 (ENGINE PRINCIPAL)
# =================================================================
class QuintikusOpenAuria:
    def __init__(self):
        self.path_brain = "brain_v18_sovereign.qoa"
        self.tokenizer = SovereignTokenizer()
        self.user_chain = UserSovereignChain("user.bin")
        
        self.l2_mass, self.l2_auth, self.l2_pil_min, self.l2_tokens_idx = [], [], [], []
        self.neuronios = defaultdict(list)
        self.rarity, self.word2idx, self.ledger = {}, {}, set()
        self.core = None
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "um", "para", "com", "na", "no"}
        
        self.user_name = None
        self.pil_user = 0.0
        self.crypto_key = None

    def amadurecer_solo(self, raw_content, auth=1, pil_min=0.0, silenciar=False):
        hash_c = hashlib.sha256((raw_content + str(pil_min)).encode('utf-8', 'ignore')).hexdigest()
        if hash_c in self.ledger: return False
        
        if not silenciar: print(f"🧠 Amadurecendo Solo (v18 - Trava PIL: {pil_min})...")
        chunks = re.split(r'([\?\!\.])', raw_content)
        all_sentences = [chunks[i] + chunks[i+1] for i in range(0, len(chunks)-1, 2) if len(chunks[i].strip()) > 1]
        
        vocab_input = (" ".join(all_sentences)).lower().split()
        vocab = sorted(list(set(vocab_input + self.tokenizer.special + list(self.word2idx.keys()))))
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        contagem = Counter(vocab_input)
        
        offset = len(self.l2_mass)
        for i, s in enumerate(all_sentences):
            # Criptografa se o nível de intimidade exigido for alto
            if pil_min >= 9.0 and self.crypto_key:
                stored_sentence = SovereignCrypt.xor_cipher(s, self.crypto_key)
            else:
                stored_sentence = s

            self.l2_mass.append(stored_sentence)
            self.l2_auth.append(auth)
            self.l2_pil_min.append(pil_min)
            
            toks = self.tokenizer.tokenize(s)
            self.l2_tokens_idx.append([self.word2idx[t] for t in toks if t in self.word2idx])
            for t in toks:
                if t not in self.stop_words and t in self.word2idx:
                    if len(self.neuronios[t]) < 10000: self.neuronios[t].append(offset + i)
                    self.rarity[t] = 2.0 / (math.log(contagem.get(t, 1) + 1.1) + 1e-5)
        
        self.core = QuantumLPSCore(len(vocab))
        self.ledger.add(hash_c)
        self.selar(silenciar)
        return True

    def selar(self, silenciar=False):
        bundle = {
            'mass': self.l2_mass, 'auth': self.l2_auth, 'pil_min': self.l2_pil_min,
            'rar': self.rarity, 'w2i': self.word2idx, 'neu': dict(self.neuronios),
            'core': self.core, 't_idx': self.l2_tokens_idx, 'ledger': self.ledger,
            'm_cache': Calculer.cache
        }
        with open(self.path_brain, 'wb') as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not silenciar: print(f"💾 Cérebro Selado.")

    def boot(self):
        # 1. Ritual do Nome
        self.pil_user, self.user_name = self.user_chain.carregar()
        if not self.user_name:
            print("\n" + "🌑"*20)
            print("  PRIMEIRA ATIVAÇÃO DETECTADA")
            print("  O nome é a única âncora que não colapsa.")
            self.user_name = input("  👤 Qual seu nome? > ").strip()
            self.pil_user = 0.0
            self.user_chain.user_name = self.user_name
            self.user_chain.salvar()
            print(f"  ✅ Lucy v18 Online. Olá, {self.user_name}. PIL: 0.00")
            print("🌑"*20 + "\n")
        else:
            print(f"✅ Lucy v18 Online. Olá, {self.user_name}. PIL: {self.pil_user:.2f}")

        self.crypto_key = SovereignCrypt.get_key(self.user_name)

        # 2. Carrega Cérebro
        if os.path.exists(self.path_brain):
            with open(self.path_brain, 'rb') as f:
                b = pickle.load(f)
                self.l2_mass, self.l2_auth, self.l2_pil_min = b['mass'], b['auth'], b['pil_min']
                self.rarity, self.word2idx, self.neuronios = b['rar'], b['w2i'], b['neu']
                self.core, self.l2_tokens_idx, self.ledger = b['core'], b['t_idx'], b['ledger']
                Calculer.cache = b.get('m_cache', {})
        return True

    def pensar_e_falar(self, entrada):
        if Calculer.eh_matematica(entrada): return Calculer.resolver(entrada, self)
        
        t0 = time.perf_counter()
        u_toks = self.tokenizer.tokenize(entrada)
        q_idx = [self.word2idx[t] for t in u_toks if t in self.word2idx]
        pivos = sorted([t for t in u_toks if t in self.rarity], key=lambda x: self.rarity[x], reverse=True)
        
        if not pivos:
            return f"oi, {self.user_name}... AINDA não te CONHEÇO <!>"

        # Filtragem AAIGB (Intimidade + Autoridade)
        candidatos = [idx for idx in self.neuronios.get(pivos[0], []) if self.l2_pil_min[idx] <= self.pil_user]

        if not candidatos:
            return f"\n[PIL-GATE | {self.pil_user:.1f}] > {self.user_name.upper()}, ainda não confio o bastante <!>"

        amostra = random.sample(candidatos, min(len(candidatos), 500))
        amostra_idx_list = [self.l2_tokens_idx[i] for i in amostra]
        lps_symbol = u_toks[-1] if u_toks[-1] in ["<?>", "<!>", "<.>"] else "<.>"
        lps_idx = self.word2idx.get(lps_symbol, self.word2idx.get("<.>"))
        
        best_future, gravidade = self.core.colapsar_nexo(q_idx, lps_idx, amostra_idx_list, self.rarity, self.word2idx)
        final_idx = amostra[amostra_idx_list.index(best_future)]

        # Decriptografia em tempo real
        frase_final = self.l2_mass[final_idx]
        if self.l2_pil_min[final_idx] >= 9.0:
            frase_final = SovereignCrypt.xor_cipher(frase_final, self.crypto_key)

        # Update PIL Linear
        if gravidade > 0.8:
            self.pil_user = min(25.0, self.pil_user + (gravidade * 0.05))
            self.user_chain.current_pil = self.pil_user
            self.user_chain.salvar()

        # Destaque de Atenção
        palavras = frase_final.split()
        res_toks_idx = self.l2_tokens_idx[final_idx]
        q_vec = np.mean(self.core.embeddings[q_idx], axis=0)
        reacoes = [np.dot(self.core.embeddings[t_id], q_vec) for t_id in res_toks_idx if t_id < len(self.core.embeddings)]
        limiar = np.mean(reacoes) if reacoes else 0
        
        resultado = []
        for i, word in enumerate(palavras):
            t_id = res_toks_idx[i] if i < len(res_toks_idx) else None
            if t_id is not None and t_id < len(self.core.embeddings) and np.dot(self.core.embeddings[t_id], q_vec) > limiar:
                resultado.append(word.upper())
            else: resultado.append(word.lower())

        dt = (time.perf_counter() - t0) * 1000000
        return f"\n[V18-SOVEREIGN | PIL:{self.pil_user:.2f} | {dt:.2f}μs]\n> {' '.join(resultado)}"

# =================================================================
# LOOP PRINCIPAL
# =================================================================
if __name__ == "__main__":
    auria = QuintikusOpenAuria()
    auria.boot()
    while True:
        u = input(f"\n[{auria.user_name}]👤: ").strip()
        
        if any(u.startswith(p) for p in ["train:", "trein:", "treino:"]):
            params = u.split(":", 1)[1]
            pil_lock = 0.0
            if "pil[" in params:
                try:
                    pil_lock = float(params.split("pil[")[1].split("]")[0])
                    path = params.split("pil[")[0].strip()
                except: path = params.strip()
            else: path = params.strip()

            if os.path.exists(path):
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(path, 'r', encoding=enc) as f:
                            auria.amadurecer_solo(f.read(), auth=1, pil_min=pil_lock)
                        print(f"✨ Conhecimento '{path}' integrado (PIL-MIN: {pil_lock})")
                        break
                    except: continue
            continue
            
        if u.lower() in ['sair', 'exit']: break
        print(auria.pensar_e_falar(u))
