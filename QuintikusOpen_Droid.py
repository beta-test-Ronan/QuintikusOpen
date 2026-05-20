
import os
import math
import time
import random
import unicodedata
import hashlib
import re
import pickle
import platform
from collections import defaultdict, Counter
from statistics import mean as py_mean

# ==========================================================
# ANDROID HELPER
# ==========================================================
try:
    import androidhelper
    droid = androidhelper.Android()
    TEM_VOZ = True
except Exception:
    droid = None
    TEM_VOZ = False

# ==========================================================
# VOZ
# ==========================================================
def falar(texto, imprimir=True):
    if imprimir:
        print(f"\n🧠 GATI: {texto}")

    if TEM_VOZ:
        try:
            droid.ttsSpeak(str(texto))
        except Exception:
            pass


def ouvir():
    if TEM_VOZ:
        try:
            print("\n🎤 Ouvindo...")
            resultado = droid.recognizeSpeech("Fale agora", None, None)

            if resultado and resultado.result:
                texto = resultado.result.strip().lower()

                if texto:
                    print(f"👤 Você disse: {texto}")
                    return texto

        except Exception as e:
            print(f"⚠️ Voz indisponível: {e}")

    try:
        return input("👤 Digite: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "sair"


# ==========================================================
# FUNÇÕES MATEMÁTICAS
# ==========================================================
def py_random_uniform(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]


def py_random_randn(rows, cols):
    return [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]


def py_dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def py_vec_add(x, y):
    return [a + b for a, b in zip(x, y)]


def py_mat_vec_mul(mat, vec):
    return [py_dot(row, vec) for row in mat]


def py_weighted_average(vectors, weights):
    total = sum(weights)

    if total == 0:
        return [0.0] * len(vectors[0])

    resultado = [0.0] * len(vectors[0])

    for vec, peso in zip(vectors, weights):
        for i, valor in enumerate(vec):
            resultado[i] += valor * peso

    return [v / total for v in resultado]


def py_argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])


# ==========================================================
# CRYPTO
# ==========================================================
class SovereignCrypt:

    @staticmethod
    def get_key(name):
        salt = platform.node() or "ROOT"
        return hashlib.sha256(f"{name}{salt}".encode()).digest()

    @staticmethod
    def xor_cipher(text, key):
        return "".join(
            chr(ord(c) ^ key[i % len(key)])
            for i, c in enumerate(text)
        )


# ==========================================================
# USER MEMORY
# ==========================================================
class UserSovereignChain:

    def __init__(self, filename="user.bin"):
        self.filename = filename
        self.history = []
        self.current_pil = 0.0
        self.user_name = None

    def salvar(self):
        data = {
            "history": self.history,
            "pil": self.current_pil,
            "name": self.user_name
        }

        with open(self.filename, "wb") as f:
            pickle.dump(data, f)

    def carregar(self):
        if not os.path.exists(self.filename):
            return 0.0, None

        try:
            with open(self.filename, "rb") as f:
                data = pickle.load(f)

            self.history = data.get("history", [])
            self.current_pil = data.get("pil", 0.0)
            self.user_name = data.get("name")

            return self.current_pil, self.user_name

        except Exception as e:
            print(f"Erro ao carregar usuário: {e}")
            return 0.0, None


# ==========================================================
# TOKENIZER
# ==========================================================
class SovereignTokenizer:

    def __init__(self):
        self.pattern = re.compile(
            r'\?\?+|\!\!+|\.\.\.+|[:;]-?[)DPpoO]|s2|<3|\^\^|[\w]+|[\?\!\.]'
        )

        self.special = [
            "<BOS>",
            "<EOS>",
            "<PAD>",
            "<?>",
            "<!>",
            "<.>"
        ]

    def normalize(self, text):
        text = text.replace('\x00', '').replace('\ufeff', '')

        return "".join(
            c for c in unicodedata.normalize("NFKD", text.lower().strip())
            if not unicodedata.combining(c)
        )

    def tokenize(self, text):
        norm = self.normalize(text)

        raw_tokens = self.pattern.findall(norm)

        processed = []

        for t in raw_tokens:
            if t == "?":
                processed.append("<?>")
            elif t == "!":
                processed.append("<!>")
            elif t == ".":
                processed.append("<.>")
            else:
                processed.append(t)

        return processed


# ==========================================================
# CORE
# ==========================================================
class QuantumLPSCore:

    def __init__(self, vocab_size, d_model=32):
        self.d_model = d_model

        limit = math.sqrt(6 / (vocab_size + d_model))

        self.embeddings = [
            py_random_uniform(-limit, limit, d_model)
            for _ in range(vocab_size)
        ]

        self.Wq = py_random_randn(d_model, d_model)
        self.Wk = py_random_randn(d_model, d_model)
        self.W_future = py_random_randn(d_model, d_model)

    def expand_vocab(self, new_size):
        atual = len(self.embeddings)

        if new_size <= atual:
            return

        limit = math.sqrt(6 / (new_size + self.d_model))

        for _ in range(new_size - atual):
            self.embeddings.append(
                py_random_uniform(-limit, limit, self.d_model)
            )

    def colapsar_nexo(
        self,
        q_idx,
        lps_idx,
        candidatos_idx_list,
        rarity_map,
        idx2word
    ):
        if not q_idx or not candidatos_idx_list:
            return None, 0

        pesos = []

        for idx in q_idx:
            palavra = idx2word.get(idx, "")
            pesos.append(rarity_map.get(palavra, 0.1))

        total = sum(pesos)

        if total > 0:
            pesos = [w / total for w in pesos]

        sujeito_vec = py_weighted_average(
            [self.embeddings[idx] for idx in q_idx],
            pesos
        )

        lps_vec = self.embeddings[lps_idx]

        foco = py_mat_vec_mul(
            self.W_future,
            py_vec_add(
                py_mat_vec_mul(self.Wq, sujeito_vec),
                lps_vec
            )
        )

        scores = []

        for candidato in candidatos_idx_list:
            media = [
                py_mean([self.embeddings[idx][k] for idx in candidato])
                for k in range(self.d_model)
            ]

            trans = py_mat_vec_mul(self.Wk, media)

            scores.append(py_dot(foco, trans))

        melhor = py_argmax(scores)

        return candidatos_idx_list[melhor], scores[melhor]


# ==========================================================
# CALCULER
# ==========================================================
class Calculer:

    cache = {}

    SAFE_PATTERN = re.compile(r'^[0-9\+\-\*\/\(\)\.\s\^]+$')

    @staticmethod
    def eh_matematica(texto):
        return (
            bool(Calculer.SAFE_PATTERN.match(texto))
            and any(c in texto for c in "+-*/^")
        )

    @staticmethod
    def resolver(expressao, auria):
        try:
            exp = expressao.replace(" ", "").replace("^", "**")

            resultado = eval(
                exp,
                {"__builtins__": None},
                {}
            )

            Calculer.cache[expressao] = resultado

            auria.amadurecer_solo(
                f"{expressao} = {resultado}.",
                auth=2,
                silenciar=True
            )

            return f"\n[CALCULER]\n> {expressao} = {resultado}"

        except Exception as e:
            return f"\n[MATH-ERROR]: {e}"


# ==========================================================
# AURIA
# ==========================================================
class QuintikusOpenAuria:

    def __init__(self):
        self.path_brain = "brain_v19.qoa"

        self.tokenizer = SovereignTokenizer()
        self.user_chain = UserSovereignChain()

        self.l2_mass = []
        self.l2_auth = []
        self.l2_pil_min = []
        self.l2_tokens_idx = []

        self.neuronios = defaultdict(list)

        self.rarity = {}

        self.word2idx = {}
        self.idx2word = {}

        self.ledger = set()

        self.core = QuantumLPSCore(10)

        self.stop_words = {
            "o", "a", "de", "que",
            "do", "da", "é", "em",
            "um", "para", "com",
            "na", "no"
        }

        self.user_name = None
        self.pil_user = 0.0
        self.crypto_key = None

        self._init_special_tokens()

    def _init_special_tokens(self):
        for token in self.tokenizer.special:
            self._add_word(token)

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)

            self.word2idx[word] = idx
            self.idx2word[idx] = word

            self.core.expand_vocab(len(self.word2idx))

    def amadurecer_solo(
        self,
        raw_content,
        auth=1,
        pil_min=0.0,
        silenciar=False
    ):

        hash_c = hashlib.sha256(
            (raw_content + str(pil_min)).encode("utf-8")
        ).hexdigest()

        if hash_c in self.ledger:
            return False

        if not silenciar:
            falar("🧠 Amadurecendo Solo...")

        sentencas = []

        if any(c in raw_content for c in ".!?"):
            partes = re.split(r'([\?\!\.])', raw_content)

            for i in range(0, len(partes) - 1, 2):
                s = (partes[i] + partes[i + 1]).strip()

                if len(s) > 1:
                    sentencas.append(s)
        else:
            sentencas = [
                s.strip()
                for s in raw_content.split("\n")
                if len(s.strip()) > 1
            ]

        if not sentencas:
            return False

        all_words = []

        for s in sentencas:
            toks = self.tokenizer.tokenize(s)

            all_words.extend(toks)

            for tok in toks:
                self._add_word(tok)

        contagem = Counter(all_words)

        offset = len(self.l2_mass)

        for i, sentenca in enumerate(sentencas):

            texto = sentenca

            if pil_min >= 9.0 and self.crypto_key:
                texto = SovereignCrypt.xor_cipher(
                    sentenca,
                    self.crypto_key
                )

            self.l2_mass.append(texto)
            self.l2_auth.append(auth)
            self.l2_pil_min.append(pil_min)

            toks = self.tokenizer.tokenize(sentenca)

            idxs = [
                self.word2idx[t]
                for t in toks
                if t in self.word2idx
            ]

            self.l2_tokens_idx.append(idxs)

            for t in toks:
                if t not in self.stop_words:

                    self.neuronios[t].append(offset + i)

                    self.rarity[t] = 2.0 / (
                        math.log(contagem.get(t, 1) + 1.2) + 1e-5
                    )

        self.ledger.add(hash_c)

        self.selar(silenciar)

        return True

    def selar(self, silenciar=False):

        data = {
            "mass": self.l2_mass,
            "auth": self.l2_auth,
            "pil_min": self.l2_pil_min,
            "tokens": self.l2_tokens_idx,
            "neuronios": dict(self.neuronios),
            "rarity": self.rarity,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "ledger": self.ledger,
            "core": self.core,
            "cache": Calculer.cache
        }

        with open(self.path_brain, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if not silenciar:
            falar(f"💾 Solo Selado ({len(self.l2_mass)} nexos).")

    def boot(self):

        self.pil_user, self.user_name = self.user_chain.carregar()

        if not self.user_name:
            falar("Primeira ativação.")

            self.user_name = input("👤 Nome: ").strip()

            self.user_chain.user_name = self.user_name
            self.user_chain.salvar()

        else:
            falar(
                f"✅ Online. "
                f"Olá {self.user_name}. "
                f"PIL: {self.pil_user:.2f}"
            )

        self.crypto_key = SovereignCrypt.get_key(self.user_name)

        if os.path.exists(self.path_brain):

            try:
                with open(self.path_brain, "rb") as f:
                    data = pickle.load(f)

                self.l2_mass = data["mass"]
                self.l2_auth = data["auth"]
                self.l2_pil_min = data["pil_min"]

                self.l2_tokens_idx = data["tokens"]

                self.neuronios = defaultdict(
                    list,
                    data["neuronios"]
                )

                self.rarity = data["rarity"]

                self.word2idx = data["word2idx"]
                self.idx2word = data["idx2word"]

                self.ledger = data["ledger"]

                self.core = data["core"]

                Calculer.cache = data.get("cache", {})

            except Exception as e:
                print(f"Erro ao carregar cérebro: {e}")

        return True

    def pensar_e_falar(self, entrada):

        if Calculer.eh_matematica(entrada):
            return Calculer.resolver(entrada, self)

        if not self.l2_mass:
            return "Ainda não aprendi nada."

        toks = self.tokenizer.tokenize(entrada)

        q_idx = [
            self.word2idx[t]
            for t in toks
            if t in self.word2idx
        ]

        if not q_idx:
            return "Ainda não conheço essas palavras."

        pivos = sorted(
            [t for t in toks if t in self.rarity],
            key=lambda x: self.rarity[x],
            reverse=True
        )

        candidatos = []

        for p in pivos:

            temp = [
                idx
                for idx in self.neuronios.get(p, [])
                if self.l2_pil_min[idx] <= self.pil_user
            ]

            if temp:
                candidatos = temp
                break

        if not candidatos:
            return "Ainda não confio o bastante."

        amostra = random.sample(
            candidatos,
            min(len(candidatos), 500)
        )

        amostra_tokens = [
            self.l2_tokens_idx[i]
            for i in amostra
        ]

        lps = toks[-1] if toks[-1] in ["<?>", "<!>", "<.>"] else "<.>"

        lps_idx = self.word2idx.get(lps, 0)

        melhor, score = self.core.colapsar_nexo(
            q_idx,
            lps_idx,
            amostra_tokens,
            self.rarity,
            self.idx2word
        )

        if not melhor:
            return "Não consegui conectar os nexos."

        final_idx = amostra[amostra_tokens.index(melhor)]

        frase = self.l2_mass[final_idx]

        if self.l2_pil_min[final_idx] >= 9.0:
            frase = SovereignCrypt.xor_cipher(
                frase,
                self.crypto_key
            )

        if score > 0.8:
            self.pil_user = min(
                35.0,
                self.pil_user + (score * 0.05)
            )

            self.user_chain.current_pil = self.pil_user
            self.user_chain.salvar()

        return frase


# ==========================================================
# COMANDOS
# ==========================================================
def processar_comandos(comando, auria):

    if any(p in comando for p in ["hora", "horas"]):
        return f"São {time.strftime('%H:%M')}"

    if "bateria" in comando and TEM_VOZ:
        try:
            droid.batteryStartMonitoring()

            time.sleep(0.5)

            nivel = droid.batteryGetLevel().result

            droid.batteryStopMonitoring()

            return f"Bateria: {nivel}%"

        except Exception:
            return "Não consegui acessar a bateria."

    if "piada" in comando:

        piadas = [
            "Python foi ao psicólogo porque tinha muitos loops internos.",
            "Java disse ao Python: você não tem classe.",
            "Problema de hardware não é bug."
        ]

        return random.choice(piadas)

    return auria.pensar_e_falar(comando)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    auria = QuintikusOpenAuria()

    auria.boot()

    falar("Assistente Quintikus pronto.")

    while True:

        comando = ouvir()

        if not comando:
            continue

        if comando.startswith(("train:", "treino:", "trein:")):

            path = comando.split(":", 1)[1].strip()

            if os.path.exists(path):

                falar(f"📂 Lendo {path}")

                for enc in ("utf-8", "latin-1", "cp1252"):

                    try:
                        with open(path, "r", encoding=enc) as f:
                            conteudo = f.read()

                        ok = auria.amadurecer_solo(conteudo)

                        if ok:
                            falar("✨ Conhecimento integrado.")
                        else:
                            falar("⚠️ Conteúdo já existente.")

                        break

                    except Exception:
                        continue

            else:
                falar("Arquivo não encontrado.")

            continue

        if any(p in comando for p in ["sair", "desligar", "exit"]):

            falar("Até logo.")

            break

        resposta = processar_comandos(comando, auria)

        falar(resposta)

        if TEM_VOZ:
            try:
                droid.eventWait(3000)
            except Exception:
                pass
