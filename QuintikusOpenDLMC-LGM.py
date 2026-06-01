# -*-coding:utf8;-*-
import json, hashlib, math, os, random, time, sys
from collections import defaultdict, deque, Counter

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ==================================================================
# 🌐 LGM-SPS v25.0 - GEODESIC CORE (Refração Atmosférica)
# ==================================================================

class LGM_SPS_Geodesic:
    def __init__(self, dataset="conhecimento.txt"):
        self.mapa = {}
        self.posicao_gps = [0.0, 0.0, 0.0] # [Lat, Lon, Alt]
        self.inercia_vetorial = [0.0, 0.0, 0.0]
        self.memoria_rastro = deque(maxlen=20)
        self.cooler_delay = 0.005
        self.triangular_universo(dataset)

    def hash_sps(self, token):
        """Mapeia o token para uma coordenada geodésica fixa"""
        h = hashlib.sha256(token.encode()).hexdigest()
        lat = (int(h[:4], 16) % 181) - 90
        lon = (int(h[4:8], 16) % 361) - 180
        alt = int(h[8:12], 16) % 101
        return [float(lat), float(lon), float(alt)]

    def triangular_universo(self, path):
        if not os.path.exists(path): return
        t0 = time.time()
        with open(path, "r", encoding="utf-8") as f:
            tokens = f.read().lower().replace(".", " . ").replace(",", " , ").split()

        self.freq_global = Counter(tokens)
        total = len(tokens)

        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if t1 not in self.mapa:
                coords = self.hash_sps(t1)
                # Massa Geodésica: Raridade do token
                massa = math.log(total / (self.freq_global[t1] + 1))
                self.mapa[t1] = {"pos": coords, "links": Counter(), "massa": massa}
            
            self.mapa[t1]["links"][t2] += 1
            if i % 10000 == 0: time.sleep(0.001)

        print(f"🌐 LGM-SPS: Geodésica Sincronizada em {(time.time()-t0)*1000:.2f}ms")

    def navegar_geodesica(self, atual, entrada):
        """Navega pela superfície curva da linguagem usando gravidade e refração"""
        if atual not in self.mapa: return "."
        time.sleep(self.cooler_delay)
        
        opcoes = self.mapa[atual]["links"]
        candidatos, pesos = [], []

        for prox, freq in opcoes.items():
            if prox not in self.mapa: continue
            dados_p = self.mapa[prox]
            
            # 1. Distância Euclidiana 3D (Distância Geográfica)
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(self.posicao_gps, dados_p["pos"])))
            
            # 2. Refração Atmosférica (Atenuação pelo Ar Semântico)
            # Se a altitude for alta, a refração é menor (o sinal é mais claro)
            refracao = dados_p["pos"][2] / 100.0
            
            # 3. Bússola de Atenção
            foco = 40.0 if prox in entrada.lower() else 1.0
            fadiga = 0.0001 if prox in self.memoria_rastro else 1.0
            
            # EQUAÇÃO GEODÉSICA v25:
            # (Frequência * Foco * Gravidade) / (Distância + Inércia)
            # Aplicamos a refração para dar brilho a palavras raras
            prob = (freq * foco * fadiga * (dados_p["massa"] ** 1.5)) / (dist + 0.5)
            prob *= refracao

            candidatos.append(prox)
            pesos.append(prob)

        if not candidatos: return "."
        escolhido = random.choices(candidatos, weights=pesos, k=1)[0]
        
        # Atualização do GPS com Inércia Vetorial
        coords_esc = self.mapa[escolhido]["pos"]
        for i in range(3):
            # O sistema 'persegue' o sinal semântico (Inércia Geográfica)
            self.posicao_gps[i] = self.posicao_gps[i] * 0.85 + coords_esc[i] * 0.15
            
        self.memoria_rastro.append(escolhido)
        return escolhido

    def pensar(self, entrada):
        palavras_in = entrada.lower().split()
        conhecidas = [p for p in palavras_in if p in self.mapa]
        
        if conhecidas:
            # Fixa o GPS na palavra de maior 'Altitude' (Montanha de significado)
            alvo = max(conhecidas, key=lambda p: self.mapa[p]["pos"][2])
            self.posicao_gps = self.mapa[alvo]["pos"]
            atual = alvo
        else:
            atual = random.choice(list(self.mapa.keys()))

        frase = [atual]
        for i in range(45):
            atual = self.navegar_geodesica(atual, entrada)
            if atual == "." and i > 8: break
            frase.append(atual)
            
        return " ".join(frase).capitalize()

def typewriter(text):
    sys.stdout.write("🤖 LGM-V25: ")
    for word in text.split():
        for char in word:
            sys.stdout.write(char); sys.stdout.flush()
            time.sleep(0.01)
        sys.stdout.write(" "); sys.stdout.flush()
        time.sleep(0.02)
    print("\n")

if __name__ == "__main__":
    sps = LGM_SPS_Geodesic()
    
    print("\n" + "_"*70)
    print("  LGM-SPS v25.0 | GEODESIC LANGUAGE MODEL | UNIVERSAL NAVIGATION")
    print("_"*70)
    
    while True:
        try:
            prompt = input("👤 > ").strip()
            if not prompt: continue
            if prompt.lower() in ['sair', 'exit']: break
            
            res = sps.pensar(prompt)
            typewriter(res)
            
            # Log Geodésico
            print(f" 🛰️ [SPS FIX: {list(map(int, sps.posicao_gps))} | Rastro: {len(sps.memoria_rastro)}]")
            print("-" * 70)
            
        except KeyboardInterrupt: break
