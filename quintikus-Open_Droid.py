import numpy as np
import unicodedata
import time
import os
import math
import random
import androidhelper
from collections import defaultdict

# ==================================================================
# 1. HARDWARE QUINTIKUS (Engine 16D)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=16):
        self.dim = dim
        self.words = {}  
        self.W = {}      
        self.lr = 0.05   
        self.frequencia_pulso = defaultdict(int)

    def _get_v(self, w):
        if w not in self.words:
            v = np.random.randn(self.dim) * 0.1
            self.words[w] = v / (np.linalg.norm(v) + 1e-9)
        return self.words[w]

    def _get_W(self, v):
        if v not in self.W:
            self.W[v] = np.eye(self.dim) + np.random.randn(self.dim, self.dim) * 0.01
        return self.W[v]

    def pulsar(self, s, v, o, autoridade):
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        delta = (vs @ Wv) - vo
        self.W[v] -= self.lr * (autoridade/100) * np.outer(vs, delta)
        self.words[o] += self.lr * delta
        self.frequencia_pulso[s] += 1
        self.frequencia_pulso[o] += 1
        return np.linalg.norm(delta)

# ==================================================================
# 2. VOID MERITOCRÁTICO
# ==================================================================
class VoidMeritocratico:
    def __init__(self, engine, sub):
        self.engine = engine
        self.sub = sub

    def meditar(self, tokens):
        Q = len(tokens)
        P = sum(self.sub.pesos_raros.get(t, 0.01) for t in tokens)
        D = 1.0 
        x_necessario = Q / (P + 1e-5)
        frequencias = [self.engine.frequencia_pulso.get(t, 0) for t in tokens]
        x_aprendido = sum(frequencias) / (Q + 1e-5)
        erro = D - (D * x_aprendido * P / (Q + 1e-5))
        
        if erro > 0.1 * D: return "DESEQUILIBRIO", x_necessario, erro
        if x_aprendido < x_necessario * 0.9: return "ZONA_SILENCIO", x_necessario, erro
        return "LIBERADO", x_necessario, erro

    def perguntar(self, tokens, x, erro):
        desconhecidos = [t for t in tokens if self.engine.frequencia_pulso.get(t, 0) < 2]
        entidades = ", ".join(desconhecidos) if desconhecidos else "nexo complexo"
        return f"VOID: Falta entendimento para [{entidades}]. Me dê o solo disso."

# ==================================================================
# 3. SUBCONSCIENTE E GENERALIZADOR
# ==================================================================
class SubconscienteCosmus:
    def __init__(self):
        self.neuronios = defaultdict(list)
        self.pesos_raros = {} 
        self.fatos_brutos = []

    def inicializar(self, txt):
        clean = txt.lower().split('.')
        for frase in clean:
            frase = frase.strip()
            if not frase: continue
            if frase not in self.fatos_brutos:
                self.fatos_brutos.append(frase)
            tokens = frase.split()
            for t in tokens:
                if len(t) < 3: continue
                self.neuronios[t].append(frase)
                q = sum(1 for f in self.fatos_brutos if t in f)
                self.pesos_raros[t] = 2.0 / (math.log(q + 1.1) + 1e-5)

class Generalizado:
    def __init__(self, sub, engine):
        self.sub = sub
        self.engine = engine

    def sintetizar(self, tokens_input):
        contexto = []
        for t in tokens_input:
            if t in self.sub.neuronios: contexto.extend(self.sub.neuronios[t])
        contexto = list(set(contexto))
        if not contexto: return None
        pool = list(set(" ".join(contexto).split()))
        vencedores = [t for t in pool if len(t) > 3 and t not in tokens_input]
        resumo = ", ".join(vencedores[:3])
        return f"Ressoa com {resumo}. Solo: {random.choice(contexto)}"

# ==================================================================
# 4. QUINTIKUS AGI (Com Persistência em TXT)
# ==================================================================
class QuintikusAGI:
    def __init__(self, debug=False):
        self.engine = AutoLeiEngine()
        self.sub = SubconscienteCosmus()
        self.gen = Generalizado(self.sub, self.engine)
        self.void = VoidMeritocratico(self.engine, self.sub)
        self.debug = debug
        self.stop_words = ["o", "a", "de", "que", "do", "da", "um", "uma", "é", "com"]
        self.banco_path = "quintikus_banco.txt"
        self.carregar_memoria()

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return t.replace("?", "").replace("!", "").strip()

    def carregar_memoria(self):
        if os.path.exists(self.banco_path):
            with open(self.banco_path, "r", encoding="utf-8") as f:
                for linha in f:
                    self.escutar(linha.strip(), autoridade=100, salvar=False)

    def escutar(self, entrada, autoridade=1.0, salvar=True):
        clean = self.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words]
        if len(tokens) < 2: return "Sinal baixo."

        eh_pergunta = entrada.strip().endswith('?') or clean.startswith(('como', 'quem', 'qual', 'o que'))

        if eh_pergunta:
            status, x, erro = self.void.meditar(tokens)
            if status != "LIBERADO":
                return self.void.perguntar(tokens, x, erro)
            return self.gen.sintetizar(tokens)

        # Aprendizado
        if len(tokens) >= 3:
            self.sub.inicializar(entrada)
            s, v, o = tokens[0], tokens[1], " ".join(tokens[2:])
            self.engine.pulsar(s, v, o, autoridade)
            if salvar:
                with open(self.banco_path, "a", encoding="utf-8") as f:
                    f.write(entrada + "\n")
            return "Entendido e gravado."
        return "Informação incompleta."

# ==================================================================
# 5. INTERFACE GATI (Android)
# ==================================================================
class GatiAssistant:
    def __init__(self):
        self.droid = androidhelper.Android()
        self.brain = QuintikusAGI(debug=False)
        self.nome_chave = "gati"

    def falar(self, texto):
        print(f"Gati: {texto}")
        self.droid.makeToast(texto)
        self.droid.ttsSpeak(texto)
        while self.droid.ttsIsSpeaking().result:
            time.sleep(0.2)

    def ouvir(self):
        try:
            res = self.droid.recognizeSpeech("Gati ouvindo...", None, None)
            return res.result if res.result else ""
        except:
            return ""

    def processar_hardware(self, comando):
        """Comandos diretos de sistema"""
        if "bateria" in comando:
            self.droid.batteryStartMonitoring()
            nivel = self.droid.batteryGetLevel().result
            return f"A bateria está em {nivel} por cento."
        if "vibrar" in comando:
            self.droid.vibrate(500)
            return "Vibração de teste concluída."
        if "wifi" in comando:
            estado = "ligar" in comando
            self.droid.toggleWifiState(estado)
            return "Comando de Wi-Fi enviado."
        return None

    def executar(self):
        self.falar("Gati iniciada. Mente Quintikus pronta.")
        
        while True:
            voz = self.ouvir()
            
            if self.nome_chave in voz.lower():
                frase = voz.lower().replace(self.nome_chave, "").strip()
                
                if not frase:
                    self.falar("Oi, estou ouvindo.")
                    continue

                # 1. Tenta Hardware
                res_hw = self.processar_hardware(frase)
                
                if res_hw:
                    self.falar(res_hw)
                else:
                    # 2. Tenta Quintikus
                    autoridade = 80.0 if len(frase) > 15 else 50.0
                    resposta = self.brain.escutar(frase, autoridade=autoridade)
                    self.falar(resposta)
            
            time.sleep(0.5)

if __name__ == "__main__":
    app = GatiAssistant()
    app.executar()
