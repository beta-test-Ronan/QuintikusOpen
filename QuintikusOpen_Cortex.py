import os
import numpy as np
import unicodedata
import time
import math
import json
import random
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ==================================================================
# 1. MOTOR DE MATRIZES (AutoLeiEngine 128D) – original
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=128, state_file="brain_state.json"):
        self.dim = dim
        self.state_file = state_file
        self.words = {}
        self.W = {}
        self.lr = 0.05
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
        W_new = Wv - self.lr * (autoridade/100) * np.outer(vs, delta)
        self.W[v] = W_new.tolist()
        v_o_new = vo + self.lr * delta
        self.words[o] = v_o_new.tolist()
        self.frequencia_pulso[s] += 1
        self.frequencia_pulso[o] += 1
        self.pending_updates += 1
        return np.linalg.norm(delta)

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
# 2. ÁREA DE BROCA – original
# ==================================================================
class AreaDeBroca:
    def __init__(self):
        self.st = [0.5, 0.5]
        self.th = {'bom':0.1, 'otimo':0.2, 'paz':0.2, 'erro':-0.2, 'urgente':-0.3, 'falha':-0.2}
        self.tons = {
            "quente": ["Sob alerta, ", "Detectando urgência, "],
            "harmonia": ["Com clareza, ", "Em sinergia, "],
            "neutro": ["Pela lógica, ", "Observando os nexos, "]
        }
    def sentir(self, tokens):
        pos = sum(self.th[t] for t in tokens if t in self.th and self.th[t] > 0)
        neg = sum(abs(self.th[t]) for t in tokens if t in self.th and self.th[t] < 0)
        self.st[0] = self.st[0]*0.95 + neg*0.1
        self.st[1] = self.st[1]*0.95 + pos*0.1
    def modular_voz(self):
        if self.st[0] > 0.6: return random.choice(self.tons["quente"])
        if self.st[1] > 0.6: return random.choice(self.tons["harmonia"])
        return random.choice(self.tons["neutro"])

# ==================================================================
# 3. SUBCONSCIENTE – original
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
        f = txt.lower().replace(",","").replace(".","").strip()
        palavras = f.split()
        if f not in self.autoridade_frase:
            self.fatos_originais[f] = palavras
            self.autoridade_frase[f] = autoridade
        else: self.autoridade_frase[f] += autoridade
        tokens = [t for t in palavras if t not in self.stop_words and len(t)>2]
        self.fatos_tokens[f] = tokens
        for t in tokens:
            if f not in self.neuronios[t]: self.neuronios[t].append(f)
            q = sum(1 for fb in self.fatos_originais if t in fb)
            self.pesos_raros[t] = 2.0 / (math.log(q+1.2)+1e-5)

    def frase_mais_autoridade(self, palavra):
        frases = self.neuronios.get(palavra, [])
        return max(frases, key=lambda f: self.autoridade_frase.get(f,0)) if frases else None

# ==================================================================
# 4. LOGIC CORTEX – ajuste mínimo: tentar segundo alvo
# ==================================================================
class LogicCortex:
    def __init__(self, sub):
        self.sub = sub
        self.expansores = ('fale','sobre','tudo','detalhes','mais','explique')

    def filtrar(self, sujeito, tokens_pergunta, entrada_bruta):
        eh_expansiva = any(word in entrada_bruta.lower() for word in self.expansores)
        contexto = [t for t in tokens_pergunta if t != sujeito]
        if not contexto:
            return None, None, eh_expansiva

        alvo = contexto[0]
        f_suj = set(self.sub.neuronios.get(sujeito, []))
        f_ctx = set(self.sub.neuronios.get(alvo, []))
        frases_comuns = list(f_suj.intersection(f_ctx))

        # Se não achou interseção, tenta com o segundo alvo (se houver)
        if not frases_comuns and len(contexto) > 1:
            alvo2 = contexto[1]
            f_ctx2 = set(self.sub.neuronios.get(alvo2, []))
            frases_comuns = list(f_suj.intersection(f_ctx2))
            if frases_comuns:
                alvo = alvo2

        if not frases_comuns:
            return None, None, eh_expansiva

        # Procura a palavra 'alvo' na frase comum e retorna a seguinte
        for f in frases_comuns:
            palavras = self.sub.fatos_originais[f]
            if alvo in palavras:
                idx = palavras.index(alvo)
                proximos = [p for p in palavras[idx+1:] if p not in self.sub.stop_words]
                if proximos:
                    return proximos[0], f, eh_expansiva
        return None, frases_comuns[0], eh_expansiva

# ==================================================================
# 5. MOTOR CAUSAL AGI – original
# ==================================================================
class MotorCausalAGI:
    def __init__(self, raw_dataset):
        self.solo = self._normalizar(raw_dataset)
    def _normalizar(self, texto):
        nfd = "".join(c for c in unicodedata.normalize('NFD', texto.lower()) if unicodedata.category(c) != 'Mn')
        for p in "?!,.;": nfd = nfd.replace(p, "")
        return nfd
    def _buscar_causalidade(self, alvo):
        palavras_causais = ["quebra","cai","estraga","muda","acontece","se"]
        frases = self.solo.split("\n")
        relacoes = []
        for f in frases:
            if alvo in f and any(pc in f for pc in palavras_causais):
                relacoes.append(f.strip())
        return relacoes[0] if relacoes else None
    def processar(self, entrada):
        entrada = self._normalizar(entrada)
        # extrai o alvo (última palavra >3 letras)
        palavras = [p for p in entrada.split() if len(p)>3]
        alvo = palavras[-1] if palavras else ""
        causal = self._buscar_causalidade(alvo)
        if causal:
            return f"Simulação causal: {causal}"
        return f"Não tenho leis causais sobre '{alvo}'."

# ==================================================================
# 6. VOID MERITOCRÁTICO – original
# ==================================================================
class VoidMeritocratico:
    def __init__(self, engine, sub):
        self.engine, self.sub = engine, sub
    def meditar(self, tokens):
        if not tokens: return False, 0
        Q = len(tokens)
        P = sum(self.sub.pesos_raros.get(t,0.05) for t in tokens)
        x_apr = sum(self.engine.frequencia_pulso.get(t,0) for t in tokens) / Q
        x_nec = Q / (P+1e-5)
        return (x_apr >= x_nec*0.1), x_apr

# ==================================================================
# 7. QUINTIKUS AGI – CONECTANDO AS PEÇAS
# ==================================================================
class QuintikusAGI:
    FONTES = {"livro_escola":100.0, "professor":50.0, "povo_falou":5.0}
    VERBOS = {"tem","é","gosta","sabe","vai","fica","funciona","serve","quebra","esquenta",
              "cai","desliga","acabar","colocar","cair","derrubar","aplicar","usar"}

    def __init__(self, debug=False):
        self.stop_words = {"o","a","de","que","do","da","é","em","no","na","com","tem","um","uma",
                           "sobre","qual","seu","sua","fale","saber","e","ou","para","como","porque"}
        self.engine = AutoLeiEngine()
        self.sub = SubconscienteCosmus(self.stop_words)
        self.broca = AreaDeBroca()
        self.logic = LogicCortex(self.sub)
        self.void = VoidMeritocratico(self.engine, self.sub)
        # dataset causal (pode ser expandido depois)
        self.dataset_causal = ""
        self.causal_engine = None
        self.debug = debug
        if os.path.exists("brain_state.json"):
            os.remove("brain_state.json")

    def normalizar(self, t):
        return "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn').replace("?","").strip()

    def _extrair_tripla(self, frase):
        """Retorna (sujeito, predicado, objeto) usando heurística simples."""
        frase = self.normalizar(frase)
        tokens = [t for t in frase.split() if t not in self.stop_words]
        if not tokens: return None,None,None
        sujeito = tokens[0]
        idx_pred = -1
        for i, t in enumerate(tokens[1:], start=1):
            if t in self.VERBOS:
                idx_pred = i
                break
        if idx_pred == -1:
            if len(tokens) > 1: idx_pred = 1
            else: return None,None,None
        predicado = tokens[idx_pred]
        objeto = "_".join(tokens[idx_pred+1:]) if idx_pred+1 < len(tokens) else "fato"
        return sujeito, predicado, objeto

    def escutar(self, entrada, fonte="povo_falou"):
        t0 = time.perf_counter()
        clean = self.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words and len(t)>2]
        self.broca.sentir(tokens)

        eh_pergunta = entrada.strip().endswith('?') or any(clean.startswith(c) for c in ('quem','qual','como','sobre','fale','por que','o que'))

        # --- MODO PERGUNTA ---
        if eh_pergunta:
            # Se for pergunta causal, usa o MotorCausalAGI (se tiver dataset)
            if "acontece" in clean and "se" in clean:
                if self.causal_engine:
                    return self.causal_engine.processar(entrada)
                else:
                    return "Nenhum solo causal foi carregado."

            liberado, x_apr = self.void.meditar(tokens)
            if not liberado:
                return f"O nexo carece de solo. (Massa: {x_apr:.2f})"

            # Escolhe sujeito: se tem "de", o dono é a última palavra
            if " de " in clean or clean.endswith(" de") or " de" in clean or "de " in clean:
                partes = clean.split()
                sujeito = partes[-1]
                if sujeito in self.stop_words and len(partes)>1:
                    sujeito = partes[-2]
            else:
                sujeito = max(tokens, key=lambda t: self.sub.pesos_raros.get(t,0) + (0.1*len(self.sub.neuronios.get(t,[]))), default=None)

            if not sujeito:
                return "Ainda não conheço esse assunto."

            nexo_dir, frase_mae, expansiva = self.logic.filtrar(sujeito, tokens, entrada)
            prefixo = self.broca.modular_voz()

            if nexo_dir and not expansiva:
                # Resposta direta: "sujeito tem alvo nexo_dir"
                alvo = [t for t in tokens if t != sujeito][0] if len(tokens)>1 else ""
                res = f"{prefixo}{sujeito.capitalize()} tem {alvo} {nexo_dir}."
            else:
                # Resposta geral / expansiva
                if frase_mae:
                    melhor_frase = frase_mae
                else:
                    melhor_frase = self.sub.frase_mais_autoridade(sujeito)
                    if not melhor_frase:
                        melhor_frase = f"Não localizei dados sobre {sujeito}."
                res = f"{prefixo}{melhor_frase.capitalize()}."
                if expansiva and melhor_frase in self.sub.fatos_tokens:
                    detalhes = [t for t in self.sub.fatos_tokens[melhor_frase] if t not in tokens]
                    if detalhes:
                        res += f" Notei detalhes como {', '.join(detalhes[:3])}."

            if self.debug:
                return f"[{(time.perf_counter()-t0)*1e6:.0f}μs]\n{res}"
            return res

        # --- MODO INGESTÃO ---
        autoridade = self.FONTES.get(fonte, 5.0)
        self.sub.inicializar(entrada, autoridade=autoridade)

        # 1. Manter frequência para o Void
        for t in tokens:
            self.engine.pulsar(t, "vinculo", "fato", autoridade)

        # 2. Ensinar tripla ao motor vetorial
        suj, pred, obj = self._extrair_tripla(entrada)
        if suj and pred and obj:
            self.engine.pulsar(suj, pred, obj, autoridade)

        # 3. Se a frase for causal, adicionar ao dataset do MotorCausalAGI
        if entrada.lower().startswith("se ") or " se " in entrada.lower():
            self.dataset_causal += entrada + "\n"
            self.causal_engine = MotorCausalAGI(self.dataset_causal)

        if self.engine.pending_updates > 20:
            self.engine.save_state()
        return None

# ==================================================================
# TESTE FINAL
# ==================================================================
if __name__ == "__main__":
    q = QuintikusAGI(debug=True)

    # Ensinando fatos
    q.escutar("Maria tem cabelo preto", fonte="povo_falou")
    q.escutar("Maria tem cabelo ondulado e cheiroso", fonte="povo_falou")
    q.escutar("Maria é uma excelente programadora", fonte="livro_escola")
    q.escutar("O motor Quintikus funciona com lógica de solo", fonte="professor")
    q.escutar("O martelo de ferro e aço são bom para prega na parede", fonte="professor")
    q.escutar("O martelo de borracha são ideais para aplicar impacto sem danificar", fonte="professor")
    # Ensinando causalidade
    q.escutar("Se a bateria acabar o sistema desliga imediatamente", fonte="professor")
    q.escutar("Se mari gosta do ronan ronan vai fica muito feliz", fonte="povo_falou")
    q.escutar("No final ronan fica triste", fonte="povo_falou")

    print("--- RESPOSTAS CORRIGIDAS ---")
    print("1. Cor do cabelo:", q.escutar("qual a cor do cabelo de maria?"))
    print("2. Sobre cabelo:", q.escutar("fale sobre o cabelo da maria?"))
    print("3. Quem é Maria:", q.escutar("quem é maria?"))
    print("4. Erro motor:", q.escutar("houve um erro urgente com o motor?"))
    print("5. Prego parede:", q.escutar("quero coloca um prego na minha parede?"))
    print("6. Causal bateria:", q.escutar("O que acontece se a bateria acabar?"))
    print("7. Causal ronan:", q.escutar("o que acontece se mari gosta do ronan?"))
