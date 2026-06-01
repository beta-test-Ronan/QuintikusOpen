# -*-coding:utf8;-*-
import json, hashlib, math, os, random, time, sys
from array import array

# ==================================================================
# ❄️ HARDWARE SHIELD
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class DLMC_QTZ:
    def __init__(self, dataset="conhecimento.txt"):
        self.dims = 100000 # Hiperespaço de 100k dimensões
        self.grid_size = 100 # Projeção 100x100 (Zonalização)
        self.teia_zonal = defaultdict(list)
        self.versor_mental = [0.0, 0.0, 0.0] # Eixos X, Y, Z da Consciência
        self.potencial_qtz = -60.0 
        
        self.carregar_e_transmutar(dataset)

    def hash_qtz(self, token):
        """Quantização e Versorização: Transforma token em coordenada XYZ e ID de 100k"""
        h = hashlib.sha256(token.encode()).hexdigest()
        # ID Dimensional (1 de 100.000)
        dim_id = int(h[:8], 16) % self.dims
        # Coordenadas Zonalizadas (0 a 99)
        x = int(h[8:10], 16) % self.grid_size
        y = int(h[10:12], 16) % self.grid_size
        z = int(h[12:14], 16) % self.grid_size
        return dim_id, x, y, z

    def carregar_e_transmutar(self, path):
        """Mineração profunda bidimensional linear"""
        if not os.path.exists(path):
            with open(path, "w") as f: f.write("o sistema transmuta o código em realidade.")
            
        print(f"🌀 Zonalizando Universo Semântico (100x100)...")
        with open(path, "r", encoding="utf-8") as f:
            tokens = f.read().lower().replace(".", " .").split()

        self.conhecimento = {}
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            d_id, x, y, z = self.hash_qtz(t1)
            
            if t1 not in self.conhecimento:
                self.conhecimento[t1] = {"xyz": [x, y, z], "id": d_id, "links": []}
            
            self.conhecimento[t1]["links"].append(t2)
            # Zonalização: Mapeia o token na grade 100x100 baseada no X e Y
            self.teia_zonal[(x, y)].append(t1)

    def calcular_proporcao(self, p1, p2):
        """Cálculo de Sinergia por Proporção Multiversal"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    def renderizar_tokens(self, atual, entrada):
        """Geração Linear Dinâmica: Raycasting Semântico (Estilo DOOM)"""
        if atual not in self.conhecimento: return "."
        
        opcoes = self.conhecimento[atual]["links"]
        tau = (self.potencial_qtz + 120) / 60.0
        
        buffer_decisao = []
        pesos = []

        # O Versor Mental atua como a câmera do DOOM, focando em uma zona do espaço
        for prox in opcoes:
            if prox not in self.conhecimento: continue
            
            xyz_prox = self.conhecimento[prox]["xyz"]
            # Distância Proporcional entre o desejo (Versor Mental) e o objeto (XYZ)
            dist = self.calcular_proporcao(self.versor_mental, xyz_prox)
            
            # Transmutação de Peso: Proximidade na Teia 100x100
            atencao = 50.0 if prox in entrada else 1.0
            ressonancia = 1.0 / (dist + 0.001)
            
            prob = math.exp(ressonancia / tau) * atencao
            buffer_decisao.append(prox)
            pesos.append(prob)

        if not buffer_decisao: return "."
        escolhido = random.choices(buffer_decisao, weights=pesos, k=1)[0]
        
        # Ajuste do Versor: A consciência 'viaja' para a zona do token escolhido
        for i in range(3):
            self.versor_mental[i] = self.versor_mental[i] * 0.5 + self.conhecimento[escolhido]["xyz"][i] * 0.5
            
        return escolhido

    def pensar(self, entrada):
        palavras = entrada.lower().split()
        
        # Injeção de Sujeito/Objeto: O input altera o Versor Mental (Eixos XYZ)
        for p in palavras:
            if p in self.conhecimento:
                coords = self.conhecimento[p]["xyz"]
                for i in range(3): 
                    # Cruzamento de dados por proporção (Média Móvel)
                    self.versor_mental[i] = (self.versor_mental[i] + coords[i]) / 2.0

        # Ponto de Partida (Trigger do Sujeito)
        atual = next((p for p in palavras if p in self.conhecimento), random.choice(list(self.conhecimento.keys())))
        
        resultado = [atual]
        for _ in range(30): # Gera fluxo profundo
            atual = self.renderizar_tokens(atual, entrada)
            if atual == ".": break
            resultado.append(atual)
            
        return " ".join(resultado).capitalize()

def typewriter(text):
    sys.stdout.write("🤖 DLMC-QTZ: ")
    for word in text.split():
        for char in word:
            sys.stdout.write(char); sys.stdout.flush()
            time.sleep(0.01)
        sys.stdout.write(" "); sys.stdout.flush()
        time.sleep(0.03)
    print("\n")

from collections import defaultdict

if __name__ == "__main__":
    # Inicializa a Versão QTZ
    engine = DLMC_QTZ()
    print("--- DLMC-QTZ v13.0: MULTIVERSAL VERSORIZATION ---")
    print("Grade 100x100 | Hiperespaço 100k Dims | Raycasting Semântico\n")
    
    while True:
        try:
            prompt = input("👤 PROMPT > ").strip()
            if prompt.lower() in ['sair', 'exit']: break
            
            res = engine.pensar(prompt)
            typewriter(res)
            
            # Status do Multiverso
            print(f" 📊 [Versor XYZ: {[round(c,1) for c in engine.versor_mental]} | Energia: {engine.potencial_qtz}mV]")
            print("-" * 70)
            
        except KeyboardInterrupt: break
