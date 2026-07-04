import os
import numpy as np
import unicodedata
import time
import math
import json
import threading
import random
from collections import defaultdict

# ==================================================================
#    ❄️ ESCUDO TÉRMICO (CONFIGURAÇÃO DE HARDWARE)
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 

# ==================================================================
# 1. MOTOR DE MATRIZES (AutoLeiEngine 128D - Homeostático)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=128, state_file="brain_state.json"):
        self.dim = dim
        self.state_file = state_file
        self.words = {}  
        self.W = {}      
        self.lr = 0.05   
        self.temperatura = 0.0 
        self.frequencia_pulso = defaultdict(int)
        self.pending_updates = 0
        self.load_state()

    def _get_v(self, w):
        if w not in self.words:
            v = np.random.randn(self.dim) * 0.01
            self.words[w] = v.tolist()
        return np.array(self.words[w])

    def _get_W(self, v):
        if v not in self.W:
            self.W[v] = np.eye(self.dim).tolist()
        return np.array(self.W[v])

    def pulsar(self, s, v, o, autoridade):
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        delta = (vs @ Wv) - vo
        diss = np.linalg.norm(delta)
        
        # Ajuste de Pesos (Lógica de Solo)
        W_new = Wv - self.lr * (autoridade/100) * np.outer(vs, delta)
        self.W[v] = W_new.tolist()
        v_o_new = vo + self.lr * delta
        self.words[o] = v_o_new.tolist()
        
        self.frequencia_pulso[s] += 1
        self.frequencia_pulso[o] += 1
        self.pending_updates += 1
        return diss

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump({"words": self.words, "W": self.W, "freq": dict(self.frequencia_pulso)}, f)

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.words, self.W = state.get("words", {}), state.get("W", {})
                    self.frequencia_pulso = defaultdict(int, state.get("freq", {}))
            except: pass

# ==================================================================
# 2. ÁREA DE BROCA (Modulação Térmica e Emocional)
# ==================================================================
class AreaDeBroca:
    def __init__(self):
        self.st = [0.5, 0.5] # [Pressão, Sinergia]
        self.th = {'bom': 0.1, 'otimo': 0.2, 'paz': 0.2, 'erro': -0.2, 'urgente': -0.3, 'falha': -0.2}
        self.tons = {
            "quente": ["Sob alerta, ", "Detectando urgência, "],
            "harmonia": ["Com clareza, ", "Em sinergia, "],
            "neutro": ["Pela lógica, ", "Observando os nexos, "]
        }

    def sentir(self, tokens):
        pos = sum(self.th[t] for t in tokens if t in self.th and self.th[t] > 0)
        neg = sum(abs(self.th[t]) for t in tokens if t in self.th and self.th[t] < 0)
        self.st[0] = self.st[0] * 0.95 + (neg * 0.1)
        self.st[1] = self.st[1] * 0.95 + (pos * 0.1)

    def modular_voz(self):
        if self.st[0] > 0.6: return random.choice(self.tons["quente"])
        if self.st[1] > 0.6: return random.choice(self.tons["harmonia"])
        return random.choice(self.tons["neutro"])

# ==================================================================
# 3. LOGIC CORTEX (Filtro Contextual e Probabilidade Linear)
# ==================================================================

class MathAGI:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-max(-500, min(500, x))))

    @staticmethod
    def d_sigmoid(x):
        return x * (1 - x)

    @staticmethod
    def dot(A, B):
        # Multiplicação de Matriz por Vetor (A: Matriz, B: Vetor)
        return [sum(row[i] * B[i] for i in range(len(B))) for row in A]


class TRM_Brain:
    def __init__(self):
        # Dimensões: x(4) + y(4) + z(8) = 16 entradas para o nível latente
        self.dim_x, self.dim_y, self.dim_z = 4, 4, 8
        
        # Pesos da Rede Neural Atômica (Iniciados aleatoriamente)
        self.w_z = [[random.uniform(-0.2, 0.2) for _ in range(16)] for _ in range(8)]
        self.w_y = [[random.uniform(-0.2, 0.2) for _ in range(12)] for _ in range(4)]
        
        self.z = [0.1] * self.dim_z # Estado Latente (Raciocínio)
        self.y = [0.0] * self.dim_y # Predição (Ação)
        self.lr = 0.15 # Taxa de Aprendizado

    def processar_recursivo(self, x_input, ciclos=6):
        """Implementação do Fast-Slow Loop (Ponto 1 do HRM)"""
        self.y = [0.0] * self.dim_y
        self.z = [0.1] * self.dim_z
        
        for _ in range(ciclos):
            # 1. Update Latent z: f(x, y, z) - Raciocínio Hierárquico
            concat_z = x_input + self.y + self.z
            self.z = [MathAGI.sigmoid(s) for s in MathAGI.dot(self.w_z, concat_z)]
            
            # 2. Update Answer y: f(y, z) - Refinamento da Resposta
            concat_y = self.y + self.z
            self.y = [MathAGI.sigmoid(s) for s in MathAGI.dot(self.w_y, concat_y)]
            
            # Halting Head (Ponto 2 do HRM): Para se estiver confiante
            if (sum(self.y)/4) > 0.85: break
        return self.y

    def aprender_backprop(self, x_input, alvo_y):
        """Aprendizado por Experiência (Ponto 10)"""
        # Ajusta Pesos de Y
        concat_y = self.y + self.z
        for i in range(self.dim_y):
            erro = alvo_y[i] - self.y[i]
            delta = erro * MathAGI.d_sigmoid(self.y[i])
            for j in range(len(concat_y)):
                self.w_y[i][j] += self.lr * delta * concat_y[j]
                
class MotorCausalAGI:
    def __init__(self, raw_dataset):
        # O "Solo" agora é mais rico em informações de causa e efeito
        self.solo = self._normalizar(raw_dataset)
        self.memoria_trabalho = {}

    def _normalizar(self, texto):
        """Remove acentos e pontuação para o robô não se perder."""
        nfd = "".join(c for c in unicodedata.normalize('NFD', texto.lower()) if unicodedata.category(c) != 'Mn')
        for p in "?!,.;":
            nfd = nfd.replace(p, "")
        return nfd

    def processar(self, entrada_usuario):
        entrada_limpa = self._normalizar(entrada_usuario)
        # Extrai a palavra principal (ex: copo)
        alvo = self._extrair_alvo(entrada_limpa)
        
        print(f"\n[SINAL]: '{entrada_usuario}'")
        print("-" * 60)

        # --- PASSO 1: EXPLICAÇÃO (O que o texto diz que É) ---
        print(f"  [PASSO 1]: Buscando definições de '{alvo}' no solo...")
        definicao = self._buscar_sentenca(alvo, "e") # Busca "Copo é..."
        time.sleep(0.5)

        # --- PASSO 2: PENSAR (O que PODE ACONTECER - Causalidade) ---
        print(f"  [PASSO 2]: Simulando causalidade (Ação -> Efeito) para '{alvo}'...")
        # Busca no texto palavras como "se", "quebra", "cai", "entao"
        causa_efeito = self._buscar_causalidade(alvo)
        time.sleep(0.5)

        # --- PASSO 3: AJUSTAR (Construir o entendimento final) ---
        print(f"  [PASSO 3]: Ajustando resposta baseada na simulação física...")
        
        return self._sintetizar(alvo, definicao, causa_efeito)

    def _extrair_alvo(self, texto):
        # Pega a palavra mais importante da pergunta
        palavras = [p for p in texto.split() if len(p) > 3]
        for p in palavras:
            if p in self.solo: return p
        return palavras[-1] if palavras else "objeto"

    def _buscar_sentenca(self, alvo, conectivo):
        frases = self.solo.split("\n")
        for f in frases:
            if alvo in f:
                return f.strip()
        return f"Não há dados sobre a natureza de {alvo}"

    def _buscar_causalidade(self, alvo):
        # Varre o solo procurando a lógica de causa (se derrubar -> quebra)
        palavras_causais = ["quebra", "cai", "estraga", "muda", "acontece", "se"]
        frases = self.solo.split("\n")
        relacoes = []
        for f in frases:
            if alvo in f and any(pc in f for pc in palavras_causais):
                relacoes.append(f.strip())
        return relacoes if relacoes else ["Nenhuma lei causal encontrada"]

    def _sintetizar(self, alvo, definicao, causa_efeito):
        # Monta a estrutura de entendimento
        res =  f"1. EXPLICAÇÃO: {definicao}.\n"
        res += f"2. PENSAMENTO: Analisei que para o {alvo}, as leis do solo dizem: '{causa_efeito[0]}'.\n"
        res += f"3. AJUSTE: Portanto, entendo que ações sobre o {alvo} geram resultados físicos reais."
        return res
        
class LogicCortex:
    def __init__(self, sub):
        self.sub = sub
        self.expansores = ('fale', 'sobre', 'tudo', 'detalhes', 'mais', 'explique')

    def filtrar(self, sujeito, tokens_pergunta, entrada_bruta):
        eh_expansiva = any(word in entrada_bruta.lower() for word in self.expansores)
        contexto = [t for t in tokens_pergunta if t != sujeito]
        
        if not contexto: return None, None, eh_expansiva

        alvo = contexto[0] # Ex: 'cabelo'
        f_suj = set(self.sub.neuronios.get(sujeito, []))
        f_ctx = set(self.sub.neuronios.get(alvo, []))
        frases_comuns = list(f_suj.intersection(f_ctx))

        if not frases_comuns: return None, None, eh_expansiva

        # Busca Linear (Corta e Cola do nexo vizinho)
        for f in frases_comuns:
            palavras = self.sub.fatos_originais[f]
            if alvo in palavras:
                idx = palavras.index(alvo)
                proximos = [p for p in palavras[idx+1:] if p not in self.sub.stop_words]
                if proximos:
                    return proximos[0], f, eh_expansiva
        
        return None, frases_comuns[0], eh_expansiva

# ==================================================================
# 4. SUBCONSCIENTE (Cosmus TDLM - Mapeamento de Solo)
# ==================================================================
class SubconscienteCosmus:
    def __init__(self, stop_words):
        self.neuronios = defaultdict(list)
        self.autoridade_frase = {} 
        self.pesos_raros = {} 
        self.fatos_originais = {} 
        self.fatos_tokens = {} 
        self.stop_words = stop_words

    def inicializar(self, txt, autoridade=1.0):
        f = txt.lower().replace(",", "").replace(".", "").strip()
        palavras = f.split()
        if f not in self.autoridade_frase:
            self.fatos_originais[f] = palavras
            self.autoridade_frase[f] = autoridade
        else: self.autoridade_frase[f] += autoridade
        
        tokens = [t for t in palavras if t not in self.stop_words and len(t) > 2]
        self.fatos_tokens[f] = tokens 
        for t in tokens:
            if f not in self.neuronios[t]: self.neuronios[t].append(f)
            q = sum(1 for fb in self.fatos_originais if t in fb)
            self.pesos_raros[t] = 2.0 / (math.log(q + 1.2) + 1e-5)

# ==================================================================
# 5. VOID MERITOCRÁTICO (Crivo de Sanidade)
# ==================================================================
class VoidMeritocratico:
    def __init__(self, engine, sub):
        self.engine, self.sub = engine, sub

    def meditar(self, tokens):
        if not tokens: return False, 0
        Q = len(tokens)
        P = sum(self.sub.pesos_raros.get(t, 0.05) for t in tokens)
        x_apr = sum(self.engine.frequencia_pulso.get(t, 0) for t in tokens) / Q
        x_nec = Q / (P + 1e-5)
        # Se aprendeu menos que o necessário para a raridade do tema, silencia
        return (x_apr >= x_nec * 0.1), x_apr

# ==================================================================
# 6. QUINTIKUS AGI (ORQUESTRADOR FINAL)
# ==================================================================
class QuintikusAGI:
    FONTES = {"livro_escola": 100.0, "professor": 50.0, "povo_falou": 5.0}

    def __init__(self, debug=False):
        self.stop_words = {"o", "a", "de", "que", "do", "da", "é", "em", "no", "na", "com", "tem", "um", "uma", "sobre", "qual", "seu", "sua", "fale", "saber"}
        self.engine = AutoLeiEngine()
        self.sub = SubconscienteCosmus(self.stop_words)
        self.broca = AreaDeBroca()
        self.logic = LogicCortex(self.sub)
        self.void = VoidMeritocratico(self.engine, self.sub)
        self.debug = debug

    def normalizar(self, t):
        return "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn').replace("?", "").strip()

    def escutar(self, entrada, fonte="povo_falou"):
        t0 = time.perf_counter()
        clean = self.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words and len(t) > 2]
        
        self.broca.sentir(tokens)
        eh_pergunta = entrada.strip().endswith('?') or any(clean.startswith(c) for c in ('quem', 'qual', 'como', 'sobre', 'fale'))

        if eh_pergunta:
            liberado, x_apr = self.void.meditar(tokens)
            if not liberado: return f"O nexo carece de solo. (Massa: {x_apr:.2f})"

            sujeito = max(tokens, key=lambda t: self.sub.pesos_raros.get(t, 0) + (0.1 * len(self.sub.neuronios.get(t, []))), default=None)
            nexo_dir, frase_mae, expansiva = self.logic.filtrar(sujeito, tokens, entrada)
            
            prefixo = self.broca.modular_voz()
            
            if nexo_dir and not expansiva:
                # Resposta Direta (Corta e Cola)
                alvo = [t for t in tokens if t != sujeito][0]
                res = f"{prefixo}{sujeito.capitalize()} tem {alvo} {nexo_dir}."
            else:
                # Resposta Geral ou Expansiva
                f_alvo = frase_mae if frase_mae else max(self.sub.neuronios.get(sujeito, []), key=lambda f: self.sub.autoridade_frase.get(f, 0))
                res = f"{prefixo}{f_alvo.capitalize()}."
                if expansiva:
                    detalhes = [t for t in self.sub.fatos_tokens[f_alvo] if t not in tokens]
                    if detalhes: res += f" Notei detalhes como {', '.join(detalhes)}."

            if self.debug:
                return f"[FLOW: {(time.perf_counter()-t0)*1e6:.2f}μs]\n{res}"
            return res

        # Ingestão
        if len(tokens) >= 1:
            auth = self.FONTES.get(fonte, 5.0)
            self.sub.inicializar(entrada, autoridade=auth)
            for t in tokens: self.engine.pulsar(t, "vinculo", "fato", auth)
            if self.engine.pending_updates > 20: self.engine.save_state()
            return None

# ==================================================================
# TESTE DA FUSÃO
# ==================================================================
if __name__ == "__main__":
    q = QuintikusAGI(debug=True)
    
    # Ensinando
    q.escutar("Maria tem cabelo preto", fonte="povo_falou")
    q.escutar("Maria tem cabelo ondulado e cheiroso", fonte="povo_falou")
    q.escutar("Maria é uma excelente programadora", fonte="livro_escola")
    q.escutar("O motor Quintikus funciona com lógica de solo", fonte="professor")
    q.escutar("O martelo de ferro e aço são bom para prega na parede", fonte="professor")
    q.escutar("O Martelos de borracha são ideais para aplicar impacto sem danificar ou riscar superfícies. ", fonte="professor")
    
    print("--- QUINTIKUS C: FUSÃO TOTAL ---")
    
    print("\n[Input Direto]: qual a cor do cabelo de maria?")
    print("[Output]:", q.escutar("qual a cor do cabelo de maria?"))

    print("\n[Input Expansivo]: fale sobre o cabelo da maria?")
    print("[Output]:", q.escutar("fale sobre o cabelo da maria?"))

    print("\n[Input Geral]: quem é maria?")
    print("[Output]:", q.escutar("quem é maria?"))
    
    print("\n[Input Térmico/Erro]: houve um erro urgente com o motor?")
    print("[Output]:", q.escutar("houve um erro urgente com o motor?"))


    print("\n[Input Geral]: quero coloca um prego na minha parede?")
    print("[Output]:", q.escutar("quero coloca um prego na minha parede?"))
