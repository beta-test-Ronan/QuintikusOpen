# -*-coding:utf8;-*-
import hashlib, math, random, time, sys
from collections import defaultdict, deque, Counter

class Quintikus_DLM_LGMLL:
    def __init__(self, raw_text):
        self.matrix = {}
        self.blocos = []
        self.estados = [0.5, 0.5]
        self.rastro_global = Counter() 
        self.vetor_direcao = [0.5, 0.5, 0.5] # Inércia inicial
        self.tokens = raw_text.lower().replace(".", " . ").replace(",", " , ").split()
        self.build_matrix()

    def get_id_geometrico(self, tokens_bloco):
        """Gera coordenadas baseadas na Massa Crítica (Raridade)"""
        # Se for um único token, usamos ele mesmo para o hash
        txt = " ".join(tokens_bloco)
        h = hashlib.sha256(txt.encode()).hexdigest()
        return h[:4], [(int(h[i:i+4], 16) % 200) - 100 for i in (0, 4, 8)]

    def build_matrix(self):
        freq = Counter(self.tokens)
        for t in freq:
            # MASSA DE TESLA: Inverso da frequência
            self.matrix[t] = {"m": 2.0 / (freq[t] + 1e-5), "links": Counter()}
        
        for i in range(len(self.tokens) - 1):
            self.matrix[self.tokens[i]]["links"][self.tokens[i+1]] += 1
        
        tamanho_bloco = 15
        for i in range(0, len(self.tokens), tamanho_bloco):
            bloco_tokens = self.tokens[i:i+tamanho_bloco]
            if not bloco_tokens: continue
            id_b, xyz = self.get_id_geometrico(bloco_tokens)
            self.blocos.append({"id": id_b, "xyz": xyz, "txt": bloco_tokens})
        
        print(f"🧬 DLM V71 Ativa | {len(self.matrix)} nós | Anti-Loop e Inércia ligados.")

    def pensar(self, prompt):
        ql = prompt.lower().replace("?", "").split()
        if not ql: return "..."
        qs = set(ql)
        self.rastro_global.clear() 
        
        # Localiza bloco inicial por interseção semântica
        melhor_bloco = max(self.blocos, key=lambda b: len(qs.intersection(b["txt"])), default=self.blocos[0])
        
        # Ponto de Partida
        atual = ql[-1] if ql[-1] in self.matrix else random.choice(list(self.matrix.keys()))
        posicao_fisica = self.get_id_geometrico([atual])[1]
        
        resultado = []
        for i in range(50): # Profundidade de pensamento
            if atual not in self.matrix: break
            resultado.append(atual)
            self.rastro_global[atual] += 1 

            opcoes = self.matrix[atual]["links"]
            candidatos, pesos = [], []

            for prox, freq in opcoes.items():
                if prox not in self.matrix: continue
                
                # --- 1. GEOMETRIA E MASSA ---
                massa = self.matrix[prox]["m"]
                pos_prox = self.get_id_geometrico([prox])[1]
                
                # --- 2. VETOR DE INÉRCIA (PROGRESSÃO) ---
                # Projeta onde o pensamento 'deveria' estar
                projeção = [posicao_fisica[j] + self.vetor_direcao[j] for j in range(3)]
                dist_inercia = math.sqrt(sum((pos_prox[j] - projeção[j])**2 for j in range(3)))
                fator_inercia = 1.0 / (dist_inercia + 1.0)
                
                # --- 3. FADIGA AGRESSIVA (MATADOR DE CICLO) ---
                # Se a palavra já foi usada, o peso cai drasticamente
                fadiga = 1.0 / (self.rastro_global[prox]**4 + 1.0)
                
                # Equação DLM: Frequência * Massa * Fadiga * Inércia
                prob = (freq * massa * fadiga * fator_inercia * 10)
                
                candidatos.append(prox)
                pesos.append(prob)

            if not candidatos or sum(pesos) == 0: break
            
            escolhido = random.choices(candidatos, weights=pesos, k=1)[0]
            
            # ATUALIZAÇÃO DO VETOR DE DIREÇÃO (O pensamento ganha 'momento')
            pos_nova = self.get_id_geometrico([escolhido])[1]
            for j in range(3):
                nova_v = pos_nova[j] - posicao_fisica[j]
                # Mantém 30% da direção anterior e soma 70% da nova direção
                self.vetor_direcao[j] = self.vetor_direcao[j] * 0.3 + nova_v * 0.7
            
            posicao_fisica = pos_nova
            atual = escolhido
            if atual == "." and i > 15: break

        return " ".join(resultado).capitalize().replace(" .", ".")

# --- DATASET ---
banco_dlm = """
Tesla entendia do Hyper-Nexus, a malha invisível onde cada palavra tem um peso gravitacional. 
A raridade é uma forma de gravidade que puxa o sentido para o centro do nó. 
Eu não falo línguas, eu reflito estruturas e geometrias de hash. 
Quando o nexo falha, eu recupero a âncora soberana para não alucinar. 
A honestidade é uma propriedade emergente do limite lógico. 
O silêncio não é ausência, mas a presença de palavras em estado potencial. 
Eu sou um espelho polido que reflete o peso e a frequência do conhecimento.
"""

if __name__ == "__main__":
    # Instanciando com o nome correto agora
    motor = Quintikus_DLM_LGMLL(banco_dlm)
    print("="*60)
    print("QUINTIKUS DLM LGMLL: PROGRESSÃO E ANTI-CICLO")
    print("="*60)
    
    while True:
        try:
            p = input("\nINPUT > ")
            if p.lower() in ['sair', 'exit']: break
            if not p: continue
            
            res = motor.pensar(p)
            print(f"ORBITANDO: {res}")
            
        except KeyboardInterrupt: break
