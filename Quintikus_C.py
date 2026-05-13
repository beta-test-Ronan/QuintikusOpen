# ==================================================================
#    ❄️ ESCUDO TÉRMICO (LIMITAÇÃO DE NÚCLEOS)   
# ==================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import unicodedata
import time
import math
import json
import heapq
import threading
from collections import defaultdict

# ==================================================================
# 0. OBLONGA (Bulbo) & GRATRY (Estômago)
# ==================================================================
class Oblonga:
    def __init__(self, engine):
        self.engine = engine
        self.acidez = 0.0
        self.apneia_ativa = False
        self.vivo = True
        self.thread = threading.Thread(target=self._respirar, daemon=True)
        self.thread.start()

    def _respirar(self):
        while self.vivo:
            self.acidez = self.engine.pending_updates / 20.0
            if self.acidez > 0.8: 
                self.apneia_ativa = True
                self.engine.lr = np.float32(0.01)
                if self.engine.pending_updates > 15:
                    self.engine.save_state()
                    self.engine.pending_updates = 0
            else:
                self.apneia_ativa = False
                self.engine.lr = np.float32(0.05)
            time.sleep(0.3)

class Gratry:
    """ Metabolismo: Filtra se o dado tem valor nutricional """
    def __init__(self, sub):
        self.sub = sub

    def digerir(self, tokens, autoridade):
        if not tokens: return 0.0
        massa = sum(self.sub.pesos_raros.get(t, 0.05) for t in tokens) / len(tokens)
        valor = massa * autoridade
        return -1.0 if valor < 0.05 else valor

# ==================================================================
# 1. HARDWARE PERSISTENTE (GEMM 128D - Auto-Cura)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=128, state_file="brain_state.json"):
        self.dim = dim
        self.state_file = state_file
        self.words, self.W = {}, {}
        self.lr = np.float32(0.05)
        self.frequencia_pulso = defaultdict(int)
        self.pending_updates = 0 
        self.last_pulse_time = time.time()
        self.load_state()

    def _get_v(self, w):
        if w not in self.words:
            self.words[w] = np.random.randn(self.dim).astype(np.float32) * 0.01
        return self.words[w]

    def _get_W(self, v_name):
        if v_name not in self.W:
            self.W[v_name] = np.identity(self.dim, dtype=np.float32)
        return self.W[v_name]

    def pulsar(self, s, v, o, autoridade):
        if self.dim == 0: return 0.0
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        vs /= (np.linalg.norm(vs) + 1e-8)
        vo /= (np.linalg.norm(vo) + 1e-8)
        delta = (vs @ Wv) - vo
        diss = np.linalg.norm(delta)
        self.W[v] -= self.lr * (autoridade/100) * np.outer(vs, delta)
        self.words[o] += self.lr * delta
        self.frequencia_pulso[s] += int(autoridade)
        self.last_pulse_time = time.time()
        self.pending_updates += 1
        return diss

    def save_state(self):
        try:
            w_ser = {k: v.tolist() for k, v in self.words.items()}
            W_ser = {k: v.tolist() for k, v in self.W.items()}
            with open(self.state_file, "w") as f:
                json.dump({"words": w_ser, "W": W_ser, "freq": dict(self.frequencia_pulso), "t": self.last_pulse_time}, f)
        except: pass

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    ld_w = state.get("words", {})
                    if ld_w and len(next(iter(ld_w.values()))) != self.dim:
                        print(f"[⚠({len(next(iter(ld_w.values())))})!=({self.dim}) Reset]")
                        return 
                    self.words = {k: np.array(v, dtype=np.float32) for k, v in ld_w.items()}
                    self.W = {k: np.array(v, dtype=np.float32) for k, v in state.get("W", {}).items()}
                    self.frequencia_pulso = defaultdict(int, state.get("freq", {}))
                    self.last_pulse_time = state.get("t", time.time())
            except: pass

# ==================================================================
# 3. SUBCONSCIENTE (Cosmus TDLM - Com Medula Epistemológica)
# ==================================================================
class SubconscienteCosmus:
    def __init__(self, neuron_file="neuron_state.json"):
        self.neuron_file = neuron_file
        self.neuronios = defaultdict(list)
        self.autoridade_frase = {} 
        self.pesos_raros = {} 
        self.fatos_brutos = []
        self.fatos_tokens = {} 
        self.load_neurons()

    def inicializar(self, txt, autoridade=1.0):
        f = txt.lower().replace(",", "").strip()
        if not f: return True
        
        # --- MEDULA (Sistema Imune) ---
        tokens = [t for t in f.split() if len(t) > 2]
        sujeito = tokens[0] if tokens else ""
        for f_antiga, auth_antiga in self.autoridade_frase.items():
            if sujeito in f_antiga and auth_antiga > autoridade * 10:
                return False # Rejeita fofoca se o solo for 10x mais forte
        
        if f not in self.autoridade_frase:
            self.fatos_brutos.append(f)
            self.autoridade_frase[f] = autoridade
        else:
            self.autoridade_frase[f] += autoridade
            
        self.fatos_tokens[f] = tokens 
        for t in tokens:
            if f not in self.neuronios[t]: self.neuronios[t].append(f)
            q = sum(1 for fb in self.fatos_brutos if t in fb)
            self.pesos_raros[t] = 1.5 / (math.log(q + 1.1) + 1e-5)
        self.save_neurons()
        return True

    def save_neurons(self):
        with open(self.neuron_file, "w") as f:
            json.dump({"fatos": self.fatos_brutos, "pesos": self.pesos_raros, "auth": self.autoridade_frase}, f)

    def load_neurons(self):
        if os.path.exists(self.neuron_file):
            try:
                with open(self.neuron_file, "r") as f:
                    state = json.load(f)
                    self.fatos_brutos = state.get("fatos", [])
                    self.pesos_raros = state.get("pesos", {})
                    self.autoridade_frase = state.get("auth", {f: 1.0 for f in self.fatos_brutos})
                    for f in self.fatos_brutos:
                        tkns = [t for t in f.split() if len(t) > 2]
                        self.fatos_tokens[f] = tkns
                        for t in tkns: self.neuronios[t].append(f)
            except: pass

# ==================================================================
# 2. QUESTYONIK (Entropia e Injoi)
# ==================================================================
class Questyonik:
    def __init__(self, eps=1e-9):
        self.eps = eps

    def avaliar_necessidade(self, scores_list, last_pulse):
        n = len(scores_list)
        if n < 2: return 0.0, 0.0
        total = sum(scores_list) + self.eps
        ent = 0.0
        for s in scores_list:
            p = s / total
            if p > 0: ent -= p * math.log(p)
        amb = ent / math.log(n)
        injoi = min(((time.time() - last_pulse) / 1800), 1.0)
        return amb, injoi

# ==================================================================
# 3. SUBCONSCIENTE (Cosmus TDLM)
# ==================================================================
class SubconscienteCosmus:
    def __init__(self, neuron_file="neuron_state.json"):
        self.neuron_file = neuron_file
        self.neuronios = defaultdict(list)
        self.autoridade_frase = {} 
        self.pesos_raros = {} 
        self.fatos_brutos = []
        self.fatos_tokens = {} 
        self.load_neurons()

    def inicializar(self, txt, autoridade=1.0):
        frases = txt.lower().replace(",", "").split('.')
        for f_bruta in frases:
            f = f_bruta.strip()
            if not f: continue
            if f not in self.autoridade_frase:
                self.fatos_brutos.append(f)
                self.autoridade_frase[f] = autoridade
            else:
                self.autoridade_frase[f] += autoridade
            tokens = [t for t in f.split() if len(t) > 2]
            self.fatos_tokens[f] = tokens 
            for t in tokens:
                if f not in self.neuronios[t]: self.neuronios[t].append(f)
                q = sum(1 for fb in self.fatos_brutos if t in fb)
                self.pesos_raros[t] = 1.5 / (math.log(q + 1.1) + 1e-5)
        self.save_neurons()

    def save_neurons(self):
        with open(self.neuron_file, "w") as f:
            json.dump({"fatos": self.fatos_brutos, "pesos": self.pesos_raros, "auth": self.autoridade_frase}, f)

    def load_neurons(self):
        if os.path.exists(self.neuron_file):
            try:
                with open(self.neuron_file, "r") as f:
                    state = json.load(f)
                    self.fatos_brutos = state.get("fatos", [])
                    self.pesos_raros = state.get("pesos", {})
                    self.autoridade_frase = state.get("auth", {f: 1.0 for f in self.fatos_brutos})
                    for f in self.fatos_brutos:
                        tkns = [t for t in f.split() if len(t) > 2]
                        self.fatos_tokens[f] = tkns
                        for t in tkns: self.neuronios[t].append(f)
            except: pass

# ==================================================================
# 4. VOID E GENERALIZADOR (Síntese)
# ==================================================================
class VoidMeritocratico:
    def __init__(self, engine, sub):
        self.engine, self.sub = engine, sub

    def meditar(self, tokens):
        if not tokens: return "VOID", 0, 1
        x_apr = sum(self.engine.frequencia_pulso.get(t, 0) for t in tokens) / len(tokens)
        if x_apr < 0.1: return "ZONA_SILENCIO", 0, x_apr
        x_nec = 1.0 / (sum(self.sub.pesos_raros.get(t, 0.05) for t in tokens) + 1e-5)
        return "LIBERADO", x_nec, x_apr

class Generalizado:
    def __init__(self, sub):
        self.sub = sub

    def sintetizar(self, tokens_set, candidatos_scores):
        if not candidatos_scores: return None
        melhor_frase = max(candidatos_scores, key=candidatos_scores.get)
        tokens_frase = self.sub.fatos_tokens.get(melhor_frase, [])
        k1, k2 = "", ""
        v1, v2 = -1.0, -1.0
        for w in tokens_frase:
            if w not in tokens_set:
                val = self.sub.pesos_raros.get(w, 0.0)
                if val > v1: v2, k2 = v1, k1; v1, k1 = val, w
                elif val > v2: v2, k2 = val, w
        res = f"{melhor_frase.capitalize()}."
        if k1: res += f" [{k1}{' & ' + k2 if k2 else ''}]"
        return res

# ==================================================================
# 5. QUINTIKUS AGI (Orquestrador)
# ==================================================================
class QuintikusAGI:
    FONTES = {"livro_escola": 100.0, "professor": 50.0, "povo_falou": 5.0}

    def __init__(self, dim=128, debug=True):
        self.engine = AutoLeiEngine(dim=dim)
        self.oblonga = Oblonga(self.engine)
        self.sub = SubconscienteCosmus()
        self.gratry = Gratry(self.sub) # Estômago
        self.quest = Questyonik()
        self.gen = Generalizado(self.sub)
        self.void = VoidMeritocratico(self.engine, self.sub)
        self.debug = debug
        self.stop_words = {"o", "a", "de", "que", "do", "da", "um", "uma", "é", "em", "no", "na", "com", "não", "tem"}
        self.comandos = ('quem', 'o que', 'qual', 'como', 'explique', 'fale', 'sobre', 'quero', 'saber')

    def normalizar(self, t):
        if all(ord(c) < 128 for c in t): return t.lower().replace("?", "").strip()
        return "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn').replace("?", "").strip()

    def escutar(self, entrada, fonte="povo_falou"):
        t0 = time.perf_counter()
        clean = self.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words]
        tokens_set = set(tokens)
        
        if len(tokens) < 1: return "Sinal baixo."
        eh_pergunta = entrada.strip().endswith('?') or clean.startswith(self.comandos)

        if eh_pergunta:
            status, x_nec, x_apr = self.void.meditar(tokens)
            candidatos = defaultdict(float)
            for t in tokens:
                if t in self.sub.neuronios:
                    g_m = self.sub.pesos_raros.get(t, 0.1)
                    for f in self.sub.neuronios[t]:
                        candidatos[f] += g_m * self.sub.autoridade_frase.get(f, 1.0)
            if not candidatos: return "Sem solo."
            amb, injoi = self.quest.avaliar_necessidade(list(candidatos.values()), self.engine.last_pulse_time)
            res = self.gen.sintetizar(tokens_set, candidatos)
            bulbo = "APNEIA" if self.oblonga.apneia_ativa else "NORMAL"
            et = (time.perf_counter() - t0) * 1e6
            return f"[FLOW: {et:.2f}μs | AMB: {amb:.2f} | BULBO: {bulbo}]\n{res}"

        # APRENDIZADO (Com Gratry e Medula)
        if len(tokens) >= 2:
            auth = self.FONTES.get(fonte, 5.0)
            nutricao = self.gratry.digerir(tokens, auth)
            if nutricao < 0: return "[GRATRY]: Refluxo. Dado sem valor."
            
            aceito = self.sub.inicializar(entrada, autoridade=auth)
            if not aceito: return "[MEDULA]: Fofoca rejeitada. Solo protegido."
            
            s, v, o = tokens[0], tokens[1], " ".join(tokens[2:])
            diss = self.engine.pulsar(s, v, o, auth)
            return f"[PULSO: {nutricao:.2f} | Dissonância: {diss:.4f}]"
        return "Ruído."

if __name__ == "__main__":
    q = QuintikusAGI(dim=128, debug=True) # Corda Afinada
    q.escutar("O motor Quintikus funciona com lógica de solo avançada", fonte="livro_escola")
    q.escutar("Maria é bonita", fonte="livro_escola")
    print("fofoca sobre maria feia:\n", q.escutar("maria é feia", fonte="povo_falou"))
    print("\nsobre maria:\n", q.escutar("sobre a maria quero saber?"))
