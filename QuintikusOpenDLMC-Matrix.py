# -*-coding:utf8;-*-
import hashlib, math, random, time, sys
from collections import defaultdict, deque, Counter

class Quintikus_DLMC_V70:
    def __init__(self, raw_text):
        self.matrix = {} # Deep Logic Matrix
        self.blocos = [] # Blocos de Conhecimento (Snapshots)
        self.estados = [0.5, 0.5] # [Temperatura/Pressão, Sinergia/Harmonia]
        self.rastro = deque(maxlen=10)
        self.termometro = {'erro': -0.3, 'falha': -0.2, 'ruído': -0.1, 'bom': 0.2, 'sinergia': 0.3, 'paz': 0.2}
        self.coordenadas_palavras = {} # Coordenadas 3D de cada palavra individual
        
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

        # CÁLCULO DE COORDENADAS DE PALAVRAS (Mapeamento espacial para Atenção Geométrica)
        temp_coords = defaultdict(list)
        for bloco in self.blocos:
            for token in bloco["txt"]:
                temp_coords[token].append(bloco["xyz"])
        
        # Define a posição média de cada palavra no espaço 3D dos blocos
        for token, lista_xyz in temp_coords.items():
            xs = [p[0] for p in lista_xyz]
            ys = [p[1] for p in lista_xyz]
            zs = [p[2] for p in lista_xyz]
            self.coordenadas_palavras[token] = [sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)]

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

    def calcular_distancia_3d(self, p1, p2):
        """Calcula a distância euclidiana simples entre dois pontos 3D"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def pensar(self, prompt):
        ql = prompt.lower().split()
        qs = set(ql)
        self.atualizar_termica(ql)

        # LOCALIZAÇÃO POR INTERSEÇÃO (Acha o Bloco mais próximo do Input)
        melhor_bloco = max(self.blocos, key=lambda b: len(qs.intersection(b["txt"])), default=self.blocos[0])
        centro_xyz = melhor_bloco["xyz"]
        
        # PONTO DE IGNIÇÃO
        atual = ql[-1] if ql[-1] in self.matrix else random.choice(melhor_bloco["txt"])
        
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
                
                # Influência Térmica
                caos = random.uniform(0, self.estados[0])
                foco = 1.0 + self.estados[1]
                
                # 1. ATENÇÃO GEOMÉTRICA (Proximidade Espacial)
                # Mede a distância entre a palavra candidata e o bloco central de contexto
                coord_prox = self.coordenadas_palavras.get(prox, [0, 0, 0])
                dist = self.calcular_distancia_3d(coord_prox, centro_xyz)
                # Normaliza para que distâncias menores deem maior pontuação (mínimo de 0.1 para evitar divisão por zero)
                atencao_geo = 10.0 / (dist + 1.0) 
                
                # 2. ATENÇÃO SEMÂNTICA (Conexão direta com o Prompt)
                # Verifica a relevância de 'prox' em relação a todos os termos do prompt digitado
                atencao_semantica = 0.0
                for token_prompt in ql:
                    if token_prompt in self.matrix:
                        # Se o token do prompt costuma apontar para 'prox', aumenta o peso
                        atencao_semantica += self.matrix[token_prompt]["links"].get(prox, 0)
                
                # Suavização da atenção semântica
                atencao_semantica = math.log1p(atencao_semantica)

                # Equação DLM-FLOW com os novos fatores de Atenção
                atencao_total = 1.0 + atencao_geo + atencao_semantica
                
                prob = (freq * massa * foco * atencao_total) + caos
                if prox in self.rastro: prob *= 0.001 # Anti-loop

                candidatos.append(prox)
                pesos.append(prob)

            if not candidatos: break
            atual = random.choices(candidatos, weights=pesos, k=1)[0]
            if atual == "." and i > 10: break

        # MODULAÇÃO DE VOZ
        prefixo = "DLM-ACTIVE | "
        if self.estados[0] > 0.6: prefixo = "[SOB PRESSÃO] "
        elif self.estados[1] > 0.6: prefixo = "[SINERGIA] "

        return f"{prefixo}{' '.join(resultado).capitalize()}."

# --- DATASET EMBUTIDO (ALMA DA MATRIX) ---


banco_dlm = """

    João era cego, mas tinha o profundo desejo de conhecer as cores. Para isso, ele precisava de alguém que o ajudasse a enxergar o mundo através das palavras. 
    Ele nunca tinha visto a cor verde, mas, ainda assim, amava a camisa verde que é favorita. 
    João ficava muito triste quando alguém o ofendia, chamando-o de "cego burro". 
    No entanto, existia uma esperança, ele poderia voltar a enxergar se fizesse uma cirurgia para implantar um olho robótico.

    Maria era vesga, mas tinha o profundo desejo de conhecer ver melhor. Para isso, ele precisava de alguém que o ajudasse a enxergar o mundo através das palavras. 
    Maria tinha camisa de cor rosa a rua da casa dela, mas, ainda assim, amava a sua camisa rosa. 
    Maria ficava muito triste quando alguém o ofendia, chamando-o de "vesga burro". 
    No entanto, existia uma esperança, ele poderia voltar a enxergar se fizesse uma cirurgia para implantar um olho robótico.


"""
with open('love.txt', 'r', encoding='utf-8') as arquivo:
    conteudo = arquivo.read()
   


if __name__ == "__main__":
    motor = Quintikus_DLMC_V70(conteudo)

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
