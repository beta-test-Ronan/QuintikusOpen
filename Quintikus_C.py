import numpy as np
import unicodedata
import time
import os
import math
import random
import json
from collections import defaultdict

# ==================================================================
# 1. HARDWARE PERSISTENTE (QuintikusC - Dimensão 128)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=128, state_file="brain_state.json"):
        self.dim = dim
        self.state_file = state_file
        self.words = {}  
        self.W = {}      
        self.lr = 0.05   
        self.frequencia_pulso = defaultdict(int) 
        self.load_state()

    def _get_v(self, w):
        if w not in self.words:
            # Inicialização em alta dimensão (128D)
            self.words[w] = (np.random.randn(self.dim) * 0.01).tolist()
        return np.array(self.words[w])

    def _get_W(self, v_name):
        if v_name not in self.W:
            self.W[v_name] = np.eye(self.dim).tolist()
        return np.array(self.W[v_name])

    def pulsar(self, s, v, o, autoridade):
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        
        delta = (vs @ Wv) - vo
        dissonancia = np.linalg.norm(delta)
        
        # Ajuste Geométrico em 128D
        W_new = Wv - self.lr * (autoridade/100) * np.outer(vs, delta)
        self.W[v] = W_new.tolist()
        v_o_new = vo + self.lr * delta
        self.words[o] = v_o_new.tolist()
        
        self.frequencia_pulso[s] += 1
        self.frequencia_pulso[o] += 1
        return dissonancia

    def save_state(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump({"words": self.words, "W": self.W, "freq": dict(self.frequencia_pulso)}, f)
        except Exception as e:
            print(f"Erro de Persistência: {e}")

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.words = state.get("words", {})
                    self.W = state.get("W", {})
                    self.frequencia_pulso = defaultdict(int, state.get("freq", {}))
            except: pass

# ==================================================================
# 2. QUESTYONIK (Identificação de Ambiguidade)
# ==================================================================
class Questyonik:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def avaliar_necessidade(self, scores_list):
        if not scores_list or len(scores_list) < 2:
            return 0.0
        
        sorted_scores = sorted(scores_list, reverse=True)
        top1, top2 = sorted_scores[0], sorted_scores[1]
        
        # Se a diferença entre os dois fatos mais fortes é mínima, há dúvida.
        igualdade = 1 - (abs(top1 - top2) / (top1 + self.eps))
        return igualdade

    def gerar_pergunta(self, tokens, necessidade):
        entidade = tokens[0] if tokens else "este solo"
        return f"QUESTYONIK: Ambiguidade detectada ({necessidade:.2f}). Me dê mais solo sobre [{entidade}]."

# ==================================================================
# 3. SUBCONSCIENTE (Cosmus TDLM)
# ==================================================================
class SubconscienteCosmus:
    def __init__(self, neuron_file="neuron_state.json"):
        self.neuron_file = neuron_file
        self.neuronios = defaultdict(list)
        self.pesos_raros = {} 
        self.fatos_brutos = []
        self.load_neurons()

    def inicializar(self, txt):
        clean = txt.lower().replace(",", "").split('.')
        for frase in clean:
            frase = frase.strip()
            if not frase or frase in self.fatos_brutos: continue
            
            self.fatos_brutos.append(frase)
            tokens = [t for t in frase.split() if len(t) > 2]
            
            for t in tokens:
                self.neuronios[t].append(frase)
                q = sum(1 for f in self.fatos_brutos if t in f)
                # G(M): Constante Gravitacional
                self.pesos_raros[t] = 1.5 / (math.log(q + 1.1) + 1e-5)
        self.save_neurons()

    def save_neurons(self):
        with open(self.neuron_file, "w") as f:
            json.dump({"fatos": self.fatos_brutos, "pesos": self.pesos_raros}, f)

    def load_neurons(self):
        if os.path.exists(self.neuron_file):
            try:
                with open(self.neuron_file, "r") as f:
                    state = json.load(f)
                    self.fatos_brutos = state.get("fatos", [])
                    self.pesos_raros = state.get("pesos", {})
                    for frase in self.fatos_brutos:
                        for t in frase.split():
                            if len(t) > 2: self.neuronios[t].append(frase)
            except: pass

# ==================================================================
# 4. VOID E GENERALIZADOR (Mérito e Síntese)
# ==================================================================
class VoidMeritocratico:
    def __init__(self, engine, sub):
        self.engine, self.sub = engine, sub

    def meditar(self, tokens):
        if not tokens: return "VOID", 0, 1
        Q = len(tokens)
        P = sum(self.sub.pesos_raros.get(t, 0.05) for t in tokens)
        x_apr = sum(self.engine.frequencia_pulso.get(t, 0) for t in tokens) / Q
        x_nec = 1.0 / (P + 1e-5)
        
        if x_apr < 0.2: # Limiar de silêncio
            return "ZONA_SILENCIO", x_nec, x_apr
        return "LIBERADO", x_nec, x_apr

class Generalizado:
    def __init__(self, sub):
        self.sub = sub

    def sintetizar(self, tokens_input, candidatos_frases):
        if not candidatos_frases: return None
        
        # Seleção por Massa Gravitacional G(M)
        pontuacao_frases = {}
        for frase in candidatos_frases:
            score = sum(self.sub.pesos_raros.get(w, 0) for w in frase.split())
            pontuacao_frases[frase] = score
        
        melhor_frase = max(pontuacao_frases, key=pontuacao_frases.get)
        
        # Ressonância de palavras raras
        pool = set(melhor_frase.split()) - set(tokens_input)
        keywords = sorted(pool, key=lambda x: self.sub.pesos_raros.get(x, 0), reverse=True)[:2]
        
        ressonancia = f"[{' & '.join(keywords)}]" if keywords else ""
        return f"{melhor_frase.capitalize()}. {ressonancia}"

# ==================================================================
# 5. QUINTIKUS AGI (QuintikusC)
# ==================================================================
class QuintikusAGI:
    FONTES = {"livro_escola": 100.0, "professor": 50.0, "povo_falou": 5.0}

    def __init__(self, debug=True):
        self.engine = AutoLeiEngine(dim=128) # Evolução para 128D
        self.sub = SubconscienteCosmus()
        self.quest = Questyonik()
        self.gen = Generalizado(self.sub)
        self.void = VoidMeritocratico(self.engine, self.sub)
        self.debug = debug
        self.stop_words = {"o", "a", "de", "que", "do", "da", "um", "uma", "é", "em", "no", "na"}
        self.comandos = ('quem', 'o que', 'qual', 'como', 'explique', 'fale')

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return t.replace("?", "").replace("!", "").strip()

    def escutar(self, entrada, fonte="povo_falou"):
        t0 = time.perf_counter()
        clean = self.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words]
        
        if len(tokens) < 2: return "VOID: Sinal baixo."

        eh_pergunta = entrada.strip().endswith('?') or clean.startswith(self.comandos)

        if eh_pergunta:
            # 1. Mérito
            status, x_nec, x_apr = self.void.meditar(tokens)
            if status == "ZONA_SILENCIO" and not self.debug: 
                return "Solo insuficiente para resposta meritocrática."

            # 2. Busca
            candidatos = defaultdict(float)
            for t in tokens:
                if t in self.sub.neuronios:
                    g_m = self.sub.pesos_raros.get(t, 0.1)
                    for f in self.sub.neuronios[t]:
                        candidatos[f] += g_m
            
            if not candidatos: return "Nenhum solo ressoa com essa pergunta."

            # 3. Ambiguidade
            necessidade = self.quest.avaliar_necessidade(list(candidatos.values()))
            if necessidade > 0.85:
                return self.quest.gerar_pergunta(tokens, necessidade)

            # 4. Síntese
            res = self.gen.sintetizar(tokens, list(candidatos.keys()))
            et = (time.perf_counter() - t0) * 1000000
            
            if self.debug:
                return f"\n[FLOW: {et:.2f}μs | DIM: 128D | MÉRITO: {x_apr:.2f}/{x_nec:.2f}]\n{res}"
            return res

        # Aprendizado
        if len(tokens) >= 3:
            self.sub.inicializar(entrada)
            s, v, o = tokens[0], tokens[1], " ".join(tokens[2:])
            diss = self.engine.pulsar(s, v, o, self.FONTES.get(fonte, 5.0))
            self.engine.save_state()
            return f"Solo integrado. (Dissonância: {diss:.4f})"

        return "Ruído processado."

if __name__ == "__main__":
    q = QuintikusAGI(debug=True)
    # O QuintikusC agora opera em 128 dimensões.
    print(q.escutar("O motor Quintikus funciona com lógica de solo", fonte="livro_escola"))
    print(q.escutar("Como funciona o motor Quintikus?"))
