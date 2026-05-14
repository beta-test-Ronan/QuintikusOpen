import numpy as np
import unicodedata
import time
import os
import math
import random
import requests
import androidhelper
from bs4 import BeautifulSoup
from collections import defaultdict

# ==================================================================
# 1. MOTOR CASSINUS AI (Otimizador de Performance)
# ==================================================================
class Cassinus:
    def __init__(self, n_tarefas):
        self.pesos = np.random.uniform(-0.1, 0.1, (n_tarefas, 1))
        self.vies = 0.0
    def prever(self, i): return max(0.01, self.pesos[i][0] + self.vies)
    def treinar(self, i, real):
        pred = self.prever(i)
        erro = pred - real
        self.pesos[i] -= 0.05 * erro
        self.vies -= 0.05 * erro

# ==================================================================
# 2. MOTOR QUINTIKUS AGI v12.4.1 (Mérito e Vácuo)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=16):
        self.dim, self.words, self.W, self.lr = dim, {}, {}, 0.05
        self.frequencia_pulso = defaultdict(int)
    def _get_v(self, w):
        if w not in self.words:
            v = np.random.randn(self.dim) * 0.1
            self.words[w] = v / (np.linalg.norm(v) + 1e-9)
        return self.words[w]
    def _get_W(self, v):
        if v not in self.W: self.W[v] = np.eye(self.dim) + np.random.randn(self.dim, self.dim) * 0.01
        return self.W[v]
    def pulsar(self, s, v, o, autoridade):
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        delta = (vs @ Wv) - vo
        self.W[v] -= self.lr * (autoridade/100) * np.outer(vs, delta)
        self.words[o] += self.lr * delta
        self.frequencia_pulso[s] += 1; self.frequencia_pulso[o] += 1

class VoidMeritocratico:
    def __init__(self, engine, sub): self.engine, self.sub = engine, sub
    def meditar(self, tokens):
        Q = len(tokens)
        P = sum(self.sub.pesos_raros.get(t, 0.01) for t in tokens)
        x_necessario = Q / (P + 1e-5)
        x_aprendido = sum(self.engine.frequencia_pulso.get(t, 0) for t in tokens) / (Q + 1e-5)
        erro = 1.0 - (x_aprendido * P / (Q + 1e-5))
        if erro > 0.1 or x_aprendido < x_necessario * 0.9: return "VOID", x_necessario, erro
        return "LIBERADO", x_necessario, erro

class SubconscienteCosmus:
    def __init__(self): self.neuronios, self.pesos_raros, self.fatos_brutos = defaultdict(list), {}, []
    def inicializar(self, txt):
        for frase in txt.lower().split('.'):
            frase = frase.strip()
            if not frase or frase in self.fatos_brutos: continue
            self.fatos_brutos.append(frase)
            for t in frase.split():
                if len(t) < 3: continue
                self.neuronios[t].append(frase)
                q = sum(1 for f in self.fatos_brutos if t in f)
                self.pesos_raros[t] = 2.0 / (math.log(q + 1.1) + 1e-5)

class QuintikusAGI:
    def __init__(self):
        self.engine = AutoLeiEngine()
        self.sub = SubconscienteCosmus()
        self.void = VoidMeritocratico(self.engine, self.sub)
        self.stop_words = ["o", "a", "de", "que", "do", "da", "um", "uma", "é"]
    def escutar(self, entrada, autoridade=1.0):
        clean = "".join(c for c in unicodedata.normalize('NFD', entrada.lower()) if unicodedata.category(c) != 'Mn').replace("?","")
        tokens = [t for t in clean.split() if t not in self.stop_words]
        if len(tokens) < 2: return "Ruído."
        eh_pergunta = entrada.strip().endswith('?') or clean.startswith(('como', 'quem', 'qual', 'o que'))
        if eh_pergunta:
            status, x, erro = self.void.meditar(tokens)
            if status == "VOID": return "VOID"
            contexto = []
            for t in tokens: contexto.extend(self.sub.neuronios.get(t, []))
            return random.choice(list(set(contexto))) if contexto else "VOID"
        if len(tokens) >= 3:
            self.sub.inicializar(entrada)
            self.engine.pulsar(tokens[0], tokens[1], " ".join(tokens[2:]), autoridade)
            return "Entendido."
        return "Incompleto."

# ==================================================================
# 3. GATI ASSISTANT (Orquestrador)
# ==================================================================
class GatiAssistant:
    def __init__(self):
        self.droid = androidhelper.Android()
        self.q = QuintikusAGI()
        self.nome_chave = "gati"
        self.modo_conversa = False
        self.banco_txt = "quintikus_banco.txt"
        self.variacoes_ok = ["Ok", "Tudo bem", "Entendido", "Certo", "Combinado"]
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Tarefas Cassinus
        self.tarefas = [
            {"name": "clima", "func": self.get_clima},
            {"name": "versiculo", "func": self.get_versiculo},
            {"name": "bateria", "func": self.get_bateria}
        ]
        self.cassinus = Cassinus(len(self.tarefas))
        self.carregar_banco()

    def carregar_banco(self):
        if os.path.exists(self.banco_txt):
            with open(self.banco_txt, "r") as f:
                for linha in f: self.q.escutar(linha.strip(), autoridade=100)

    def get_clima(self): return f"Clima: {requests.get('https://wttr.in/?format=3').text.strip()}"
    def get_versiculo(self):
        soup = BeautifulSoup(requests.get("https://www.bibliaon.com/versiculo_do_dia/").text, 'html.parser')
        return f"Versículo: {soup.find(id='versiculo_hoje').get_text(separator=' ')}"
    def get_bateria(self):
        self.droid.batteryStartMonitoring()
        return f"Bateria em {self.droid.batteryGetLevel().result}%"

    def pesquisar_google(self, termo):
        try:
            res = requests.get(f"https://www.google.com/search?q={termo}", headers=self.headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            resumo = soup.find('h3').get_text() if soup.find('h3') else "um nexo complexo"
            fato = f"{termo} é {resumo}"
            # Ensina o Quintikus repetidamente para gerar Mérito
            for _ in range(5): self.q.escutar(fato, autoridade=100)
            with open(self.banco_txt, "a") as f: f.write(fato + "\n")
            return fato
        except: return None

    def falar(self, texto):
        print(f"Gati: {texto}"); self.droid.ttsSpeak(texto)

    def ouvir(self):
        try:
            res = self.droid.recognizeSpeech("Gati ouvindo...", None, None)
            return res.result.lower() if res.result else ""
        except: return ""

    def processar(self, voz):
        if any(x in voz for x in ["sair", "goodbye", "desligar"]):
            self.falar("Ok, goodbye!"); return "EXIT"
            
        if "modo conversa" in voz:
            if "desativa" in voz: self.modo_conversa = False; self.falar("Modo conversa desativado.")
            else: self.modo_conversa = True; self.falar(f"{random.choice(self.variacoes_ok)}, vamos conversar.")
            return "KEEP"

        # 1. Verifica Cassinus
        for i, t in enumerate(self.tarefas):
            if t['name'] in voz:
                start = time.perf_counter()
                resp = t['func']()
                self.cassinus.treinar(i, time.perf_counter() - start)
                self.falar(resp); return "KEEP"

        # 2. Quintikus NLU
        resp_q = self.q.escutar(voz)
        if resp_q == "VOID":
            self.falar("Não tenho mérito sobre isso. Buscando solo na internet...")
            fato = self.pesquisar_google(voz)
            if fato: self.falar(f"Aprendi que {fato}")
            else: self.falar("O vazio persiste, não encontrei nada.")
        else:
            self.falar(resp_q)
        return "KEEP"

    def executar(self):
        self.falar("Gati Online.")
        while True:
            voz = self.ouvir()
            if self.modo_conversa and voz:
                if self.processar(voz) == "EXIT": break
            elif self.nome_chave in voz:
                cmd = voz.replace(self.nome_chave, "").strip()
                if not cmd: self.falar(random.choice(self.variacoes_ok))
                else:
                    if self.processar(cmd) == "EXIT": break
            time.sleep(0.5)

if __name__ == "__main__":
    app = GatiAssistant()
    app.executar()
