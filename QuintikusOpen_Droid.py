import os
import math
import time
import random
import re
import struct
import pickle  # compatibilidade com cérebros antigos
import cmath
from collections import defaultdict, Counter

# =================================================================
# ANDROID HELPER (QPYTHON) - OPCIONAL
# =================================================================
try:
    import androidhelper
    droid = androidhelper.Android()
    TEM_VOZ = True
except:
    droid = None
    TEM_VOZ = False

# =================================================================
# FUNÇÕES DE VOZ E TEXTO (COM EVENTWAIT)
# =================================================================
def falar(texto, imprimir=True):
    if imprimir:
        print(f"🧠 LUCY: {texto}")
    if TEM_VOZ:
        try:
            droid.ttsSpeak(texto)
            droid.eventWait(3000)
        except:
            pass
    else:
        time.sleep(0.5)

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
        return input("👤 Digite seu comando: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "sair"

# =================================================================
# ÁLGEBRA LINEAR PURA
# =================================================================
def pure_norm(v):
    return math.sqrt(sum(x * x for x in v))

def normalize_vector(v):
    n = pure_norm(v)
    return [x / n for x in v] if n > 1e-9 else v

def pure_randn(dims):
    return [random.gauss(0, 1) for _ in range(dims)]

def pure_dot(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def pure_zeros(n):
    return [0.0] * n

def vec_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def vec_sub(v1, v2):
    return [a - b for a, b in zip(v1, v2)]

def vec_mul(v, scalar):
    return [x * scalar for x in v]

# =================================================================
# ANÁLISE DE SENTIMENTO E ESTADO EMOCIONAL
# =================================================================
class SentimentAnalyzer:
    POS = {"amo","amor","lindo","maravilhoso","obrigado","obrigada","querido","querida",
           "gato","gata","fofo","fofa","bom","boa","ótimo","excelente","feliz","alegre",
           "carinho","carinhoso","carinhosa","beijo","abraço","saudade","parabéns","incrível",
           "espetacular","divertido","legal","grato","grata","obrigadão","valeu","top"}
    NEG = {"triste","chato","chata","ruim","horrível","ódio","raiva","nojento","nojenta",
           "feio","feia","burro","burra","idiota","imbecil","droga","merda","desculpa",
           "desculpe","cansado","cansada","estressado","estressada","puto","puta","lixo",
           "decepcionado","decepcionada","sozinho","sozinha","deprimido","deprimida"}
    
    @staticmethod
    def analisar(texto):
        tokens = set(texto.lower().split())
        p = len(tokens.intersection(SentimentAnalyzer.POS))
        n = len(tokens.intersection(SentimentAnalyzer.NEG))
        if p > n: return "positivo"
        elif n > p: return "negativo"
        return "neutro"

class EmotionState:
    def __init__(self):
        self.valencia = 0.0   # -1 (triste) a +1 (feliz)
        self.excitacao = 0.5  # 0 (calmo) a 1 (agitado)
        self.decay = 0.9      # taxa de retorno ao neutro
    
    def atualizar(self, sentimento):
        rv, re = random.gauss(0,0.1), random.gauss(0,0.05)
        if sentimento == "positivo":
            self.valencia += 0.2 + rv
            self.excitacao += 0.1 + re
        elif sentimento == "negativo":
            self.valencia -= 0.2 + rv
            self.excitacao += 0.1 + re
        else:
            self.valencia += rv
            self.excitacao += re
        self.valencia = max(-1.0, min(1.0, self.valencia))
        self.excitacao = max(0.0, min(1.0, self.excitacao))
        self.valencia *= self.decay
        self.excitacao = 0.5 + (self.excitacao-0.5)*self.decay
    
    @property
    def tom(self):
        if self.valencia > 0.3:
            return "caloroso"
        elif self.valencia < -0.3:
            return "frio"
        return "neutro"

# =================================================================
# NÚCLEO ARQUINET ND
# =================================================================
class ArquinetCore:
    def __init__(self, dims=1024):
        self.dims = dims
        self.mapa_nd = {}
        self.grafo = {}
        self.pulso = defaultdict(int)
        self.taxa_aprendizado = 0.1

    def treinar(self, tokens, auth=1.0):
        mapa, grafo = self.mapa_nd, self.grafo
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if t1 not in mapa:
                mapa[t1] = normalize_vector(pure_randn(self.dims))
            if t2 not in mapa:
                mapa[t2] = normalize_vector(pure_randn(self.dims))

            move = self.taxa_aprendizado * auth
            diff = vec_sub(mapa[t2], mapa[t1])
            novo_v = vec_add(mapa[t1], vec_mul(diff, move))
            mapa[t1] = normalize_vector(novo_v)

            if t1 not in grafo:
                grafo[t1] = {}
            if t2 not in grafo[t1]:
                grafo[t1][t2] = 0.5 + 0j

            grafo[t1][t2] *= cmath.exp(1j * 0.1)
            grafo[t1][t2] += 0.05 * auth
            self.pulso[t1] += 1

# =================================================================
# QUINTIKUS v22.0 + SERIALIZAÇÃO BINÁRIA + SENTIMENTO
# =================================================================
class QuintikusLucy:
    def __init__(self):
        self.path_brain = "brain_v22_sovereign.qoa"
        self.path_bin   = "brain_v22_sovereign.qbin"
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        self.cognition = ArquinetCore(dims=1024)

        self.l2_mass, self.l2_vectors, self.l2_auth, self.l2_tokens_len = [], [], [], []
        self.neuronios = defaultdict(list)
        self.triplas = defaultdict(list)
        self.raridade = Counter()

        self.pil_user = 0.0
        self.cache_reflexo = []
        self.drives = {"afetivo": 0.5, "curioso": 0.3, "analitico": 0.2, "criativo": 0.3,
                       "empatia": 0.3, "acidez": 0.5}  # novos drives emocionais
        self.sombra_entropica = pure_zeros(1024)
        self.exaustao = []

        # Componentes de sentimento
        self.sentiment = SentimentAnalyzer()
        self.emotion = EmotionState()

    # ---------- TREINAMENTO (sem salvamento automático) ----------
    def amadurecer_solo(self, texto, auth=1.0):
        frases = re.split(r'[\.\!\?]', texto)
        for f in frases:
            f = f.strip()
            if len(f) < 2:
                continue
            tokens = self.tokenizer.findall(f.lower())
            if len(tokens) < 2:
                continue

            idx = len(self.l2_mass)
            self.l2_mass.append(f)
            self.l2_auth.append(auth)
            self.l2_tokens_len.append(len(tokens))

            if len(tokens) >= 3:
                self.triplas[tokens[0]].append((tokens[1], " ".join(tokens[2:])))

            for t in tokens:
                self.raridade[t] += 1
                self.neuronios[t].append(idx)

            self.cognition.treinar(tokens, auth)

            v_frase = pure_zeros(1024)
            for t in tokens:
                if t in self.cognition.mapa_nd:
                    peso = 1.0 / (math.log(self.raridade[t] + 2))
                    v_frase = vec_add(v_frase, vec_mul(self.cognition.mapa_nd[t], peso))
            self.l2_vectors.append(normalize_vector(v_frase))

    # ---------- SERIALIZAÇÃO BINÁRIA (.qbin) ----------
    def salvar_binario(self, filename=None):
        if filename is None:
            filename = self.path_bin
        with open(filename, 'wb') as f:
            f.write(b'QKBR')                     # magic
            f.write(struct.pack('<H', 1))        # versão
            f.write(struct.pack('<I', len(self.l2_mass)))

            # 1) Frases
            for frase in self.l2_mass:
                data = frase.encode('utf-8')
                f.write(struct.pack('<H', len(data)))
                f.write(data)

            # 2) Vetores
            for vec in self.l2_vectors:
                f.write(struct.pack(f'<{1024}f', *vec))

            # 3) Neurônios
            neuro = {k: v for k, v in self.neuronios.items() if isinstance(k, str)}
            f.write(struct.pack('<I', len(neuro)))
            for palavra, indices in neuro.items():
                b = palavra.encode('utf-8')
                f.write(struct.pack('<H', len(b)))
                f.write(b)
                f.write(struct.pack('<I', len(indices)))
                f.write(struct.pack(f'<{len(indices)}I', *indices))

            # 4) Mapa ND
            mapa_items = [(k, v) for k, v in self.cognition.mapa_nd.items() if isinstance(k, str)]
            f.write(struct.pack('<I', len(mapa_items)))
            for palavra, vec in mapa_items:
                b = palavra.encode('utf-8')
                f.write(struct.pack('<H', len(b)))
                f.write(b)
                f.write(struct.pack(f'<{1024}f', *vec))

            # 5) Grafo
            grafo = {}
            for k, v in self.cognition.grafo.items():
                if isinstance(k, str):
                    grafo[k] = {k2: (v2.real, v2.imag) for k2, v2 in v.items() if isinstance(k2, str)}
            f.write(struct.pack('<I', len(grafo)))
            for origem, destinos in grafo.items():
                b = origem.encode('utf-8')
                f.write(struct.pack('<H', len(b)))
                f.write(b)
                f.write(struct.pack('<I', len(destinos)))
                for destino, (real, imag) in destinos.items():
                    d = destino.encode('utf-8')
                    f.write(struct.pack('<H', len(d)))
                    f.write(d)
                    f.write(struct.pack('<ff', real, imag))

            # 6) Raridade
            rar = {k: v for k, v in self.raridade.items() if isinstance(k, str)}
            f.write(struct.pack('<I', len(rar)))
            for palavra, contagem in rar.items():
                b = palavra.encode('utf-8')
                f.write(struct.pack('<H', len(b)))
                f.write(b)
                f.write(struct.pack('<I', contagem))

            # 7) Auth e Tokens Len
            f.write(struct.pack(f'<{len(self.l2_auth)}f', *self.l2_auth))
            f.write(struct.pack(f'<{len(self.l2_tokens_len)}I', *self.l2_tokens_len))

    def carregar_binario(self, filename=None):
        if filename is None:
            filename = self.path_bin
        if not os.path.exists(filename):
            return False
        with open(filename, 'rb') as f:
            if f.read(4) != b'QKBR':
                return False
            version = struct.unpack('<H', f.read(2))[0]
            num_frases = struct.unpack('<I', f.read(4))[0]

            self.l2_mass = []
            for _ in range(num_frases):
                size = struct.unpack('<H', f.read(2))[0]
                self.l2_mass.append(f.read(size).decode('utf-8'))

            self.l2_vectors = []
            for _ in range(num_frases):
                self.l2_vectors.append(list(struct.unpack(f'<{1024}f', f.read(1024*4))))

            self.neuronios = defaultdict(list)
            num_entradas = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_entradas):
                size = struct.unpack('<H', f.read(2))[0]
                palavra = f.read(size).decode('utf-8')
                num_idx = struct.unpack('<I', f.read(4))[0]
                self.neuronios[palavra] = list(struct.unpack(f'<{num_idx}I', f.read(num_idx*4)))

            self.cognition.mapa_nd = {}
            num_mapa = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_mapa):
                size = struct.unpack('<H', f.read(2))[0]
                palavra = f.read(size).decode('utf-8')
                self.cognition.mapa_nd[palavra] = list(struct.unpack(f'<{1024}f', f.read(1024*4)))

            self.cognition.grafo = {}
            num_origens = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_origens):
                size = struct.unpack('<H', f.read(2))[0]
                origem = f.read(size).decode('utf-8')
                num_dest = struct.unpack('<I', f.read(4))[0]
                destinos = {}
                for _ in range(num_dest):
                    dsize = struct.unpack('<H', f.read(2))[0]
                    destino = f.read(dsize).decode('utf-8')
                    real, imag = struct.unpack('<ff', f.read(8))
                    destinos[destino] = complex(real, imag)
                self.cognition.grafo[origem] = destinos

            self.raridade = Counter()
            num_rar = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_rar):
                size = struct.unpack('<H', f.read(2))[0]
                palavra = f.read(size).decode('utf-8')
                contagem = struct.unpack('<I', f.read(4))[0]
                self.raridade[palavra] = contagem

            self.l2_auth = list(struct.unpack(f'<{num_frases}f', f.read(num_frases*4)))
            self.l2_tokens_len = list(struct.unpack(f'<{num_frases}I', f.read(num_frases*4)))
        return True

    # ---------- MÉTODOS ANTIGOS (compatibilidade) ----------
    def salvar_pickle(self):
        with open(self.path_brain, 'wb') as f:
            pickle.dump({
                'm': self.l2_mass, 'v': self.l2_vectors, 'n': self.neuronios,
                'c': self.cognition, 't': self.triplas, 'r': self.raridade,
                'a': self.l2_auth, 'tl': self.l2_tokens_len
            }, f)

    def carregar_pickle(self):
        if os.path.exists(self.path_brain):
            with open(self.path_brain, 'rb') as f:
                b = pickle.load(f)
                self.l2_mass = b['m']
                self.l2_vectors = b['v']
                self.neuronios = b['n']
                self.cognition = b['c']
                self.triplas = b['t']
                self.raridade = b['r']
                self.l2_auth = b.get('a', [1]*len(self.l2_mass))
                self.l2_tokens_len = b.get('tl', [5]*len(self.l2_mass))
            return True
        return False

    # ---------- MOTOR DE RESPOSTA (COM MODULAÇÃO EMOCIONAL) ----------
    def pensar_e_falar(self, entrada):
        # --- NOVO: análise de sentimento e atualização emocional ---
        sentimento = self.sentiment.analisar(entrada)
        self.emotion.atualizar(sentimento)
        # Ajusta drives com base no estado emocional
        if self.emotion.valencia < -0.3:   # Lucy "solidária"
            self.drives["acidez"] *= 0.5   # reduz deboche
            self.drives["empatia"] += 0.2
        else:
            self.drives["acidez"] = min(1.0, self.drives["acidez"]*1.1)
            self.drives["empatia"] *= 0.9

        t0 = time.perf_counter()
        tokens = self.tokenizer.findall(entrada.lower())
        if not tokens:
            return "..."

        v_entrada = pure_zeros(1024)
        for t in tokens:
            if t in self.cognition.mapa_nd:
                peso = 1.0 / (math.log(self.raridade[t] + 2))
                v_entrada = vec_add(v_entrada, vec_mul(self.cognition.mapa_nd[t], peso))
        v_entrada = normalize_vector(v_entrada)

        for v_antigo, res_idx in self.cache_reflexo:
            if pure_dot(v_entrada, v_antigo) > 0.98:
                return f"[REFLEXO] > {self.l2_mass[res_idx]}"

        eh_pergunta = "?" in entrada
        sujeito = next((t for t in tokens if t in self.triplas), None)
        if eh_pergunta and sujeito:
            rel, obj = random.choice(self.triplas[sujeito])
            return f"[LÓGICA] > {sujeito.capitalize()} {rel} {obj}."

        if eh_pergunta:
            self.drives["curioso"] += 0.2
        if len(tokens) > 6:
            self.drives["afetivo"] += 0.1

        self.sombra_entropica = normalize_vector(
            vec_add(vec_mul(self.sombra_entropica, 0.4), vec_mul(v_entrada, 0.6))
        )

        pivo = max(tokens, key=lambda t: self.raridade[t], default=tokens[0])
        candidatos = self.neuronios.get(pivo, [])
        if not candidatos:
            return "Vácuo semântico detectado."

        def pontuar(idx):
            score = pure_dot(self.sombra_entropica, self.l2_vectors[idx])
            if eh_pergunta and self.l2_auth[idx] >= 2:
                score += self.drives["analitico"]
            if "?" in self.l2_mass[idx]:
                score += self.drives["curioso"] * 0.4
            assimetria = abs(len(tokens) - self.l2_tokens_len[idx])
            score -= assimetria * 0.1
            if idx in self.exaustao:
                score -= 2.0

            # --- NOVO: modulação por alinhamento emocional ---
            frase = self.l2_mass[idx]
            palavras_frase = set(frase.lower().split())
            # Consolo para usuário negativo
            if sentimento == "negativo" and palavras_frase.intersection(SentimentAnalyzer.POS):
                score += self.drives["empatia"] * 0.3
            # Mantém tom divertido para usuário positivo
            elif sentimento == "positivo":
                score += self.drives["criativo"] * 0.1
            # Penalidade para frases muito ácidas em momentos de tristeza
            if sentimento == "negativo" and palavras_frase.intersection(SentimentAnalyzer.NEG):
                score -= self.drives["acidez"] * 0.2

            return score

        amostra = random.sample(candidatos, min(len(candidatos), 100))
        idx_final = max(amostra, key=pontuar)

        self.cache_reflexo.append((v_entrada, idx_final))
        if len(self.cache_reflexo) > 5:
            self.cache_reflexo.pop(0)
        self.exaustao.append(idx_final)
        if len(self.exaustao) > 15:
            self.exaustao.pop(0)

        frase_final = self.l2_mass[idx_final]
        # Pós-edição suave quando a Lucy está empática e o usuário negativo
        if sentimento == "negativo" and self.drives["empatia"] > 0.5:
            frase_final = frase_final.replace("burra", "confusa")
            frase_final = frase_final.replace("idiota", "distraída")

        return f"\n> {frase_final}"

    def monologo_interno(self):
        chaves = list(self.cognition.mapa_nd.keys())
        if len(chaves) < 2:
            return
        for _ in range(100):
            t1, t2 = random.sample(chaves, 2)
            if pure_dot(self.cognition.mapa_nd[t1], self.cognition.mapa_nd[t2]) > 0.6:
                if t1 not in self.cognition.grafo:
                    self.cognition.grafo[t1] = {}
                self.cognition.grafo[t1][t2] = 0.5 + 0.1j
                self.cognition.pulso[t1] += 1

# =================================================================
# COMANDOS ESPECIAIS (ATUALIZADOS)
# =================================================================
def processar_comandos(cmd, auria):
    if any(p in cmd for p in ["horas", "hora", "que horas"]):
        return f"São {time.strftime('%H:%M')}"
    if "bateria" in cmd and TEM_VOZ:
        droid.batteryStartMonitoring()
        time.sleep(0.5)
        nivel = droid.batteryGetLevel().result
        droid.batteryStopMonitoring()
        return f"A bateria está em {nivel}%"
    if any(p in cmd for p in ["piada", "conte uma piada"]):
        return random.choice([
            "Por que o Python foi ao psicólogo? Porque tinha muitos loops internos!",
            "O que o Java disse pro Python? Você não tem classe!",
            "Quantos programadores para trocar uma lâmpada? Nenhum, é problema de hardware."
        ])
    if cmd == "modo neural":
        return "Modo neural não disponível nesta versão."
    # Novo comando para consultar estado emocional
    if any(p in cmd for p in ["como você está", "como voce esta", "como está", "como esta"]):
        return f"Estou me sentindo {auria.emotion.tom}. E você?"
    return auria.pensar_e_falar(cmd)

# =================================================================
# LOOP PRINCIPAL (COM NOVOS COMANDOS)
# =================================================================
if __name__ == "__main__":
    lucy = QuintikusLucy()

    # Tenta carregar binário primeiro; se não existir, fallback para pickle
    if not lucy.carregar_binario():
        if lucy.carregar_pickle():
            falar(f"✅ Lucy v22.0 Online (pickle convertido). Solo: {len(lucy.l2_mass)} nexos.")
        else:
            falar("✅ Lucy v22.0 Online. Nenhum conhecimento prévio.")
    else:
        falar(f"✅ Lucy v22.0 Online (binário). Solo: {len(lucy.l2_mass)} nexos.")

    falar("Comandos: 'horas', 'bateria', 'piada', 'sonhar', 'train:arquivo.txt', 'salvar', 'sair', 'como você está'")

    while True:
        comando = ouvir()
        if not comando:
            continue

        if any(p in comando for p in ["desligar", "tchau"]):
            falar("Até logo! Lucy se despede.")
            break

        if comando == 'sonhar':
            falar("🌙 Monólogo interno...")
            lucy.monologo_interno()
            falar("💾 Use 'salvar' para persistir o sonho.")
            continue

        if comando.startswith("train:"):
            path = comando.split(":")[1].strip()
            if os.path.exists(path):
                falar(f"📂 Treinando com {path}...")
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lucy.amadurecer_solo(f.read())
                falar(f"✨ Treinado! Total: {len(lucy.l2_mass)} nexos. Use 'salvar' para gravar.")
            else:
                falar(f"❌ Arquivo '{path}' não encontrado.")
            continue

        if comando == 'salvar':
            lucy.salvar_binario()
            falar("💾 Cérebro salvo em formato binário (.qbin).")
            continue

        if comando == 'salvar_pickle':
            lucy.salvar_pickle()
            falar("💾 Cérebro salvo em formato pickle (.qoa).")
            continue

        resposta = processar_comandos(comando, lucy)
        falar(resposta)
