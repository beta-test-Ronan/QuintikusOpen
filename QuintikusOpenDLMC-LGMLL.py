# -*-coding:utf8;-*-
import json, hashlib, math, os, random, time, sys
from collections import defaultdict, deque, Counter

# HARDWARE SHIELD
os.environ["OMP_NUM_THREADS"] = "1" 

class AnalystCore:
    def __init__(self):
        self.volatilidade_ma = deque(maxlen=5) 
        self.v_suavizada = 0.0
    def calcular_entropia(self, pesos):
        if not pesos: return 0.0
        soma = sum(pesos)
        probs = [p/soma for p in pesos]
        v = -sum(p * math.log2(p) for p in probs if p > 0)
        self.volatilidade_ma.append(v)
        self.v_suavizada = sum(self.volatilidade_ma) / len(self.volatilidade_ma)
        return v
    def calcular_risco(self, dist, massa, freq):
        return (dist * 0.45) / (massa + math.log(freq + 1.2))

class NeuralPL:
    def __init__(self, mapa_global):
        self.mapa_global = mapa_global
        self.nucleo_ativo = set()
        self.campo_geodesico = []

    def sintonizar(self, tokens_in):
        self.nucleo_ativo = set(tokens_in)
        self.campo_geodesico = [self.mapa_global[t]["pos"] for t in tokens_in if t in self.mapa_global]
        for t in tokens_in:
            if t in self.mapa_global:
                self.nucleo_ativo.update(self.mapa_global[t]["links"].keys())

    def validar_disparo(self, token, prob_base, risco, frase):
        if token not in self.mapa_global: return 1e-25
        
        # --- MATADOR DE LOOPS DE PADRAO (A-B-A-B) ---
        # Mantemos apenas a inibicao de tokens exatos, sem repulsao geografica
        inibicao_padrao = 1.0
        if len(frase) >= 3:
            if token == frase[-2] and frase[-1] == frase[-3]:
                inibicao_padrao = 0.0001
        
        fator_integracao = 1.0 / (risco + 0.05)
        
        if token in self.nucleo_ativo:
            prob = prob_base * 2.5 * fator_integracao
        else:
            coords = self.mapa_global[token]["pos"]
            dist_min = min([math.sqrt(sum((a-b)**2 for a, b in zip(coords, cg))) for cg in self.campo_geodesico]) if self.campo_geodesico else 100
            prob = prob_base * fator_integracao if dist_min < 15 else prob_base * 0.00001
        
        return prob * inibicao_padrao

class LGMLL_Engine:
    def __init__(self, dataset="conhecimento.txt"):
        self.mapa = {}
        self.gps_pos = [0.0, 0.0, 0.0]
        self.rastro_longo = deque(maxlen=60)
        self.modo = "frio" 
        self.analyst = AnalystCore()
        self.neural_pl = None
        self.cooler_delay = 0.006
        self.construir_geometria(dataset)

    def hash_geodesico(self, token):
        h = hashlib.sha256(token.encode()).hexdigest()
        lat, lon, alt = (int(h[:4],16)%181)-90, (int(h[4:8],16)%361)-180, int(h[8:12],16)%101
        return [float(lat), float(lon), float(alt)]

    def construir_geometria(self, path):
        if not os.path.exists(path): return
        t0 = time.time()
        with open(path, "r", encoding="utf-8") as f:
            tokens = f.read().lower().replace(".", " . ").replace(",", " , ").split()
        freq = Counter(tokens); total = len(tokens)
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if t1 not in self.mapa:
                self.mapa[t1] = {"pos": self.hash_geodesico(t1), "links": Counter(), "massa": math.log(total / (freq[t1] + 1))}
            self.mapa[t1]["links"][t2] += 1
        self.neural_pl = NeuralPL(self.mapa)
        print(f"LGMLL v38.0: Gravity Core Ready.")

    def pensar(self, entrada):
        tokens_in = [t for t in entrada.lower().split() if t in self.mapa]
        if not tokens_in: return "LGMLL: Fora de orbita."
        
        self.neural_pl.sintonizar(tokens_in)
        atual = max(tokens_in, key=lambda t: self.mapa[t]["massa"])
        
        # --- ANCORA DE INTENCAO ---
        self.gps_pos = self.mapa[atual]["pos"]
        ancora_gps = list(self.gps_pos) # Fixa o centro de gravidade do prompt
        
        frase = [atual]
        self.rastro_longo.clear()
        self.rastro_longo.append(atual)

        mapa_tau = {"frio": 0.22, "neutro": 0.7, "poeta": 1.2}
        tau = mapa_tau.get(self.modo, 0.7)

        for i in range(60):
            if atual not in self.mapa: break
            time.sleep(self.cooler_delay)
            
            opcoes = self.mapa[atual]["links"]
            candidatos, pesos_final = [], []

            for prox, freq in opcoes.items():
                if prox not in self.mapa: continue
                d_p = self.mapa[prox]
                
                # --- CALCULO DE GRAVIDADE RELATIVA ---
                dist_atual = math.sqrt(sum((a-b)**2 for a, b in zip(self.gps_pos, d_p["pos"])))
                dist_ancora = math.sqrt(sum((a-b)**2 for a, b in zip(ancora_gps, d_p["pos"])))
                
                # A distancia final e ponderada: a ancora original puxa 60%
                dist_final = (dist_atual * 0.4) + (dist_ancora * 0.6)
                
                risco = self.analyst.calcular_risco(dist_final, d_p["massa"], freq)
                fadiga = 0.0001 if prox in self.rastro_longo else 1.0
                
                prob_base = (freq * (d_p["massa"]**1.1) * fadiga) / (dist_final + 0.5)
                prob_neural = self.neural_pl.validar_disparo(prox, prob_base, risco, frase)

                candidatos.append(prox); pesos_final.append(prob_neural ** (1.0 / tau))

            if not candidatos or sum(pesos_final) == 0: break
            atual = random.choices(candidatos, weights=pesos_final, k=1)[0]
            
            self.analyst.calcular_entropia(pesos_final)
            for d in range(3): self.gps_pos[d] = self.gps_pos[d] * 0.75 + self.mapa[atual]["pos"][d] * 0.25
            
            frase.append(atual)
            self.rastro_longo.append(atual)
            if atual == "." and i > 12: break
            
        return " ".join(frase).replace(" .", ".").capitalize()

def typewriter(text, v, mode):
    sys.stdout.write(f"LGMLL [{mode.upper()}][V-MA: {v:.2f}]: ")
    for word in text.split():
        for char in word: sys.stdout.write(char); sys.stdout.flush(); time.sleep(0.01)
        sys.stdout.write(" "); sys.stdout.flush(); time.sleep(0.02)
    print("\n")

if __name__ == "__main__":
    motor = LGMLL_Engine()
    while True:
        try:
            prompt = input("INPUT > ").strip()
            if not prompt: continue
            if prompt.lower() in ['sair', 'exit']: break
            if prompt.lower() == "modo frio": motor.modo = "frio"; continue
            if prompt.lower() == "modo neutro": motor.modo = "neutro"; continue
            if prompt.lower() == "modo poeta": motor.modo = "poeta"; continue
            res = motor.pensar(prompt)
            typewriter(res, motor.analyst.v_suavizada, motor.modo)
        except KeyboardInterrupt: break
