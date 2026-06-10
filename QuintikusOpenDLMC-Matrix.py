# -*-coding:utf8;-*-
import hashlib, math, random, time, sys
from collections import defaultdict, deque, Counter
#class 02 dlm
class Quintikus_DLMC_V70:
    def __init__(self, raw_text):
        self.matrix = {} # Deep Logic Matrix
        self.blocos = [] # Blocos de Conhecimento (Snapshots)
        self.estados = [0.5, 0.5] # [Temperatura/Pressão, Sinergia/Harmonia]
        self.rastro = deque(maxlen=10)
        self.termometro = {'erro': -0.3, 'falha': -0.2, 'ruído': -0.1, 'bom': 0.2, 'sinergia': 0.3, 'paz': 0.2}
        
        # 1. PRÉ-PROCESSAMENTO DLM
        self.tokens = raw_text.lower().replace(".", " . ").replace(",", " , ").split()
        self.build_matrix()

    def get_id_geometrico(self, tokens_bloco):
        """Gera ID de Bloco baseado nas 5 palavras mais raras (Massa Crítica)"""
        raras = sorted(tokens_bloco, key=lambda x: self.matrix.get(x, {"m":0})["m"], reverse=True)[:5]
        h = hashlib.sha256(" ".join(raras).encode()).hexdigest()
        # Coordenadas 3D derivadas do ID do Bloco
        x = (int(h[:4], 16) % 200) - 100
        y = (int(h[4:8], 16) % 200) - 100
        z = (int(h[8:12], 16) % 200) - 100
        return h[:4], [x, y, z]

    def build_matrix(self):
        freq = Counter(self.tokens)
        # MASSA INVERSA: Palavras raras pesam mais (Conceito Tesla/Hyper-Nexus)
        for t in freq:
            self.matrix[t] = {"m": 1.5 / (freq[t] + 1e-5), "links": Counter()}

        # Mapeamento de Adjacência
        for i in range(len(self.tokens) - 1):
            self.matrix[self.tokens[i]]["links"][self.tokens[i+1]] += 1

        # CRIAÇÃO DE BLOCOS (DLM-SNAPSHOTS)
        tamanho_bloco = 12
        for i in range(0, len(self.tokens), tamanho_bloco):
            bloco_tokens = self.tokens[i:i+tamanho_bloco]
            if not bloco_tokens: continue
            id_b, xyz = self.get_id_geometrico(bloco_tokens)
            self.blocos.append({"id": id_b, "xyz": xyz, "txt": bloco_tokens})

        print(f"🧬 Matrix DLMC Ativa: {len(self.matrix)} nós | {len(self.blocos)} blocos de massa.")

    def atualizar_termica(self, tokens_in):
        """TSPLS: Ajusta o estado emocional (Pressão/Sinergia)"""
        for t in tokens_in:
            if t in self.termometro:
                val = self.termometro[t]
                if val < 0: self.estados[0] = min(1.0, self.estados[0] + abs(val))
                else: self.estados[1] = min(1.0, self.estados[1] + val)
        # Resfriamento natural
        self.estados = [s * 0.95 for s in self.estados]

    def pensar(self, prompt):
        ql = prompt.lower().split()
        qs = set(ql)
        self.atualizar_termica(ql)

        # LOCALIZAÇÃO POR INTERSEÇÃO (Acha o Bloco mais próximo do Input)
        melhor_bloco = max(self.blocos, key=lambda b: len(qs.intersection(b["txt"])), default=self.blocos[0])
        
        # PONTO DE IGNIÇÃO
        atual = ql[-1] if ql[-1] in self.matrix else random.choice(melhor_bloco["txt"])
        centro_xyz = melhor_bloco["xyz"]
        
        resultado = []
        for i in range(40):
            if atual not in self.matrix: break
            resultado.append(atual)
            self.rastro.append(atual)

            opcoes = self.matrix[atual]["links"]
            candidatos, pesos = [], []

            for prox, freq in opcoes.items():
                if prox not in self.matrix: continue
                massa = self.matrix[prox]["m"]
                
                # Influência Térmica: Pressão (estados[0]) gera caos, Sinergia (estados[1]) gera foco
                caos = random.uniform(0, self.estados[0])
                foco = 1.0 + self.estados[1]
                
                # Geometria: Atração pelo Bloco de Massa Crítica
                # Se o token estiver no bloco alvo, ganha bônus
                bonus_bloco = 2.0 if prox in melhor_bloco["txt"] else 0.5
                
                # Equação DLM-FLOW
                prob = (freq * massa * foco * bonus_bloco) + caos
                if prox in self.rastro: prob *= 0.001 # Anti-loop

                candidatos.append(prox)
                pesos.append(prob)

            if not candidatos: break
            atual = random.choices(candidatos, weights=pesos, k=1)[0]
            if atual == "." and i > 10: break

        # MODULAÇÃO DE VOZ (Baseada na Temperatura)
        prefixo = "DLM-ACTIVE | "
        if self.estados[0] > 0.6: prefixo = "[SOB PRESSÃO] "
        elif self.estados[1] > 0.6: prefixo = "[SINERGIA] "

        return f"{prefixo}{' '.join(resultado).capitalize()}."

# --- DATASET EMBUTIDO (ALMA DA MATRIX) ---


banco_dlm = """

Gato é animal.
Animal precisa comer,quando gato fico com fome ele precisa de comer.
Gato tem comer comida de animal.
Pedra é mineral.
Mineral é sólido.
Pedra tem massa.


"""
with open('amor.txt', 'r', encoding='utf-8') as arquivo:
    conteudo = arquivo.read()
   


if __name__ == "__main__":
    motor = Quintikus_DLMC_V70(banco_dlm)

    print("="*60)
    print("QUINTIKUS DLMC V70: DEEP LOGIC MATRIX FLOW")
    print("="*60)
    
    while True:
        try:
            p = input("\nINPUT > ")
            if p.lower() in ['sair', 'exit']: break
            
            res = motor.pensar(p)
            
            # Efeito de Saída Térmica
            sys.stdout.write("Processando Matrix: ")
            for char in res:
                sys.stdout.write(char); sys.stdout.flush()
                time.sleep(0.01)
            print(f"\n[Status T:{motor.estados[0]:.2f} | S:{motor.estados[1]:.2f}]")
            
        except KeyboardInterrupt: break
