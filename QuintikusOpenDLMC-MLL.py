# -*-coding:utf8;-*-
import json, hashlib, math, os, random, time, sys
from collections import defaultdict, deque

# ==================================================================
# ❄️ HARDWARE SHIELD
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class EntropyWalker:
    """O Agente que navega nas camadas e agrupa dados por Entropia Quantica"""
    def __init__(self, dims=100000, grid=100):
        self.dims = dims
        self.grid_size = grid
        self.teia = {} # Memória de Longo Prazo
        self.zonalizacao = defaultdict(list) # Agrupamento espacial
        self.versor_xyz = [50.0, 50.0, 50.0] # Centro do Universo
        
        # Termodinâmica Quantica
        self.temperatura = 1.0 # T (Kelvin Virtual)
        self.entropia_acumulada = 0.0
        self.rastro = deque(maxlen=5) # Memória de curto prazo para detecção de loop

    def hash_qtz(self, token):
        h = hashlib.sha256(token.encode()).hexdigest()
        d_id = int(h[:8], 16) % self.dims
        x = int(h[8:10], 16) % self.grid_size
        y = int(h[10:12], 16) % self.grid_size
        z = int(h[12:14], 16) % self.grid_size
        return d_id, x, y, z

    def devour_text(self, path):
        """Transmuta o texto em MS por proporção linear"""
        t0 = time.time()
        if not os.path.exists(path): return
        with open(path, "r", encoding="utf-8") as f:
            tokens = f.read().lower().replace(".", " .").split()

        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            d_id, x, y, z = self.hash_qtz(t1)
            
            if t1 not in self.teia:
                self.teia[t1] = {"xyz": [x, y, z], "links": Counter()}
            
            self.teia[t1]["links"][t2] += 1
            if t1 not in self.zonalizacao[(x, y)]:
                self.zonalizacao[(x, y)].append(t1)
        
        tf = (time.time() - t0) * 1000
        print(f"🌀 Universo Zonalizado em {tf:.2f}ms | Tokens: {len(self.teia)}")

    def calcular_entropia_local(self, opcoes):
        """Mede a desordem das opções de próximo token (Shannon/Quantum Entropy)"""
        total = sum(opcoes.values())
        if total == 0: return 0
        ent = 0
        for count in opcoes.values():
            p = count / total
            ent -= p * math.log2(p)
        return ent

    def walker_step(self, atual, entrada):
        """O Agente decide o próximo passo usando Termodinâmica"""
        if atual not in self.teia: return "."
        
        opcoes = self.teia[atual]["links"]
        ent_local = self.calcular_entropia_local(opcoes)
        
        # --- LÓGICA TERMODINÂMICA ---
        # Se a entropia for baixa (poucas opções), a temperatura SOBE para evitar o loop
        if atual in self.rastro:
            self.temperatura += 0.5 # Aquecimento por fricção (repetição)
        else:
            self.temperatura = max(0.5, self.temperatura * 0.9) # Resfriamento (estabilidade)

        candidatos = []
        pesos = []
        
        # Raycasting Semântico: Cruzando Versores XYZ
        for prox, freq in opcoes.items():
            xyz_prox = self.teia[prox]["xyz"]
            # Distância entre o Versor Mental e o próximo astro
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(self.versor_xyz, xyz_prox)))
            
            # Ativação por Proporção e Calor
            # P(token) = (Frequencia * Sinergia) ^ (1/T)
            ressonancia = 1.0 / (dist + 0.01)
            atencao = 30.0 if prox in entrada else 1.0
            
            prob = (freq * math.exp(ressonancia)) ** (1.0 / self.temperatura) * atencao
            candidatos.append(prox)
            pesos.append(prob)

        escolhido = random.choices(candidatos, weights=pesos, k=1)[0]
        
        # Atualiza Agente
        self.rastro.append(escolhido)
        # O Versor Mental 'anda' 30% em direção ao novo dado
        for i in range(3):
            self.versor_xyz[i] = self.versor_xyz[i] * 0.7 + self.teia[escolhido]["xyz"][i] * 0.3
            
        return escolhido

    def pensar(self, entrada):
        palavras = entrada.lower().split()
        # Sintoniza Versor com a entrada do usuário
        for p in palavras:
            if p in self.teia:
                xyz = self.teia[p]["xyz"]
                for i in range(3): self.versor_xyz[i] = (self.versor_xyz[i] + xyz[i]) / 2.0

        atual = next((p for p in palavras if p in self.teia), random.choice(list(self.teia.keys())))
        frase = [atual]
        
        for _ in range(35): # Profundidade de campo
            atual = self.walker_step(atual, entrada)
            if atual == ".": break
            frase.append(atual)
            
        return " ".join(frase).capitalize()

def typewriter(text):
    sys.stdout.write("🤖 DLMC-QTZ v14: ")
    for word in text.split():
        for char in word:
            sys.stdout.write(char); sys.stdout.flush()
            time.sleep(0.01)
        sys.stdout.write(" "); sys.stdout.flush()
        time.sleep(0.02)
    print("\n")

from collections import Counter

if __name__ == "__main__":
    agent = EntropyWalker()
    # Carrega o Gênesis ou qualquer dataset técnico
    agent.devour_text("conhecimento.txt") 
    
    print("\n--- AGENTE DE ENTROPIA ATIVO ---")
    print("Navegação Termodinâmica | Controle de Calor Sináptico\n")
    
    while True:
        try:
            prompt = input("👤 > ").strip()
            if prompt.lower() in ['sair', 'exit']: break
            
            res = agent.pensar(prompt)
            typewriter(res)
            
            # Status log do Agente
            print(f" 🔥 [Temp: {agent.temperatura:.2f}K | Versor: {[round(c,1) for c in agent.versor_xyz]}]")
            print("-" * 70)
            
        except KeyboardInterrupt: break
