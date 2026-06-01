# -*-coding:utf8;-*-
import json, hashlib, math, os, random, time, sys
from collections import defaultdict, deque, Counter

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ==================================================================
# 🛰️ SPS CORE - Semantic Positioning System (LGM v24.0)
# ==================================================================

class LGM_SPS:
    """O primeiro GPS da Linguagem Universal"""
    def __init__(self, dataset="conhecimento.txt"):
        self.num_setores = 360  # Estilo Graus de uma Bússola
        self.grid_res = 100     # Resolução por setor
        self.mapa_global = defaultdict(lambda: defaultdict(list))
        self.conhecimento = {}
        
        # Coordenadas GPS Semânticas
        self.lat_long_alt = [0.0, 0.0, 0.0] # Latitude, Longitude, Altitude (Info)
        self.setor_ativo = 0
        self.memoria_epocal = deque(maxlen=30)
        self.cooler_delay = 0.005
        
        self.triangular_universo(dataset)

    def get_sps_coordinates(self, token):
        """Triangulação Determinística: Token -> Latitude, Longitude, Altitude"""
        h = hashlib.sha256(token.encode()).hexdigest()
        # Latitude semântica (-90 a 90)
        lat = (int(h[:4], 16) % 181) - 90
        # Longitude semântica (-180 a 180)
        lon = (int(h[4:8], 16) % 361) - 180
        # Altitude de Informação (0 a 100)
        alt = int(h[8:12], 16) % 101
        # Setor (0 a 359)
        setor = int(h[12:16], 16) % self.num_setores
        return setor, lat, lon, alt

    def triangular_universo(self, path):
        """Mapeia o Model para o Sistema de Posicionamento Global Semântico"""
        if not os.path.exists(path): return
        t0 = time.time()
        with open(path, "r", encoding="utf-8") as f:
            tokens = f.read().lower().replace(".", " . ").split()

        self.freq = Counter(tokens)
        total = len(tokens)

        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            setor, lat, lon, alt = self.get_sps_coordinates(t1)
            
            if t1 not in self.conhecimento:
                # Massa Informativa (Altitude Geográfica)
                massa = math.log(total / (self.freq[t1] + 1))
                self.conhecimento[t1] = {
                    "coords": [lat, lon, alt], "setor": setor, 
                    "links": Counter(), "massa": massa
                }
                self.mapa_global[setor][(lat, lon)].append(t1)
            
            self.conhecimento[t1]["links"][t2] += 1
            if i % 10000 == 0: time.sleep(0.001)

        print(f"🛰️ SPS: Satélites Semânticos Sincronizados em {(time.time()-t0)*1000:.2f}ms")

    def navegar_sps(self, atual, entrada):
        """Navegação por Triangulação de GPS Semântico"""
        if atual not in self.conhecimento: return "."
        time.sleep(self.cooler_delay)
        
        # Pega dados da posição atual
        dados_atuais = self.conhecimento[atual]
        opcoes = dados_atuais["links"]
        
        candidatos, pesos = [], []
        
        for prox, freq in opcoes.items():
            if prox not in self.conhecimento: continue
            dados_p = self.conhecimento[prox]
            
            # --- CÁLCULO DE DISTÂNCIA HAVERSINE (Simplificada para SPS) ---
            # O quão longe o próximo token está do meu alvo mental?
            d_lat = abs(self.lat_long_alt[0] - dados_p["coords"][0])
            d_lon = abs(self.lat_long_alt[1] - dados_p["coords"][1])
            dist_geografica = math.sqrt(d_lat**2 + d_lon**2)
            
            # --- ATRAÇÃO POR ALTITUDE (MASSA) ---
            atracao_altitude = dados_p["coords"][2] / 100.0
            
            # --- RELEVÂNCIA DETERMINÍSTICA ---
            foco = 30.0 if prox in entrada.lower() else 1.0
            fadiga = 0.0001 if prox in self.memoria_epocal else 1.0
            
            # Equação SPS: (Frequência * Foco * Fadiga) / (Distância + Inércia)
            prob = (freq * foco * fadiga * atracao_altitude) / (dist_geografica + 0.5)
            
            candidatos.append(prox)
            pesos.append(prob)

        if not candidatos: return "."
        escolhido = random.choices(candidatos, weights=pesos, k=1)[0]
        
        # Atualiza o Posicionamento Global (O pensamento 'se move' no mapa)
        self.memoria_epocal.append(escolhido)
        coords_esc = self.conhecimento[escolhido]["coords"]
        for i in range(3):
            self.lat_long_alt[i] = self.lat_long_alt[i] * 0.8 + coords_esc[i] * 0.2
            
        return escolhido

    def localizar_intencao(self, entrada):
        """Identifica onde a pergunta do usuário se localiza no mapa SPS"""
        palavras_in = entrada.lower().split()
        conhecidas = [p for p in palavras_in if p in self.conhecimento]
        
        if conhecidas:
            # Trava o GPS na coordenada da palavra de maior 'Altitude' (Massa)
            alvo = max(conhecidas, key=lambda p: self.conhecimento[p]["coords"][2])
            self.lat_long_alt = self.conhecimento[alvo]["coords"]
            self.setor_ativo = self.conhecimento[alvo]["setor"]
            return alvo
        return random.choice(list(self.conhecimento.keys()))

    def pensar(self, entrada):
        atual = self.localizar_intencao(entrada)
        frase = [atual]
        
        for i in range(50): # Profundidade de exploração
            atual = self.navegar_sps(atual, entrada)
            if atual == "." and i > 10: break
            frase.append(atual)
            
        return " ".join(frase).capitalize()

def typewriter(text):
    sys.stdout.write("🤖 SPS-V24: ")
    for word in text.split():
        for char in word:
            sys.stdout.write(char); sys.stdout.flush()
            time.sleep(0.01)
        sys.stdout.write(" "); sys.stdout.flush()
        time.sleep(0.02)
    print("\n")

if __name__ == "__main__":
    sps = LGM_SPS()
    
    print("\n" + "═"*70)
    print("  SPS v24.0 | SEMANTIC POSITIONING SYSTEM | LANGUAGE GEOMETRIC MODEL")
    print("═"*70)
    
    while True:
        try:
            prompt = input("👤 > ").strip()
            if not prompt: continue
            if prompt.lower() in ['sair', 'exit']: break
            
            res = sps.pensar(prompt)
            typewriter(res)
            
            # Status do GPS Semântico
            print(f" 🛰️ [Fix: Lat {sps.lat_long_alt[0]:.1f} | Lon {sps.lat_long_alt[1]:.1f} | Alt {sps.lat_long_alt[2]:.1f}]")
            print("-" * 70)
            
        except KeyboardInterrupt: break
