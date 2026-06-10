# -*-coding:utf8;-*-
import os, hashlib, math, random, time, sys, unicodedata
from collections import defaultdict, deque, Counter

# HARDWARE SHIELD
os.environ["OMP_NUM_THREADS"] = "1" 

class Quintikus_LGMLL_V120_Ultra:
    def __init__(self, raw_text):
        self.matrix = defaultdict(Counter)
        self.metadata = {}
        self.mapa_sentencas = defaultdict(list)
        self.tokens_por_sentenca = []
        self.cabecalhos = {}
        self.st = [0.5, 0.5]
        self.termometro = {'erro':-0.3, 'falha':-0.2, 'ruido':-0.1, 'bom':0.2, 'sinergia':0.3, 'paz':0.2}
        self.raw_tokens = self.normalize(raw_text).split()
        self._aprender()

    def normalize(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', str(t).lower().strip()) if unicodedata.category(c) != 'Mn')
        return t.replace(".", " . ").replace(",", " , ").replace("?", " ? ")

    def _get_multimodal_xyz(self, token):
        h = hashlib.sha256(token.encode()).hexdigest()
        return {'L': (int(h[0:8],16)%100000, int(h[8:16],16)%100000, int(h[16:24],16)%100000),
                'S': (int(h[24:32],16)%100000, int(h[32:40],16)%100000, int(h[40:48],16)%100000),
                'E': (int(h[48:56],16)%100000, int(h[56:64],16)%100000, 50000)}

    def _aprender(self):
        sentencas = [s.strip() for s in " ".join(self.raw_tokens).split(".") if s.strip()]
        for idx, s in enumerate(sentencas):
            tokens_s = s.split()
            self.tokens_por_sentenca.append(set(tokens_s))
            if len(tokens_s) >= 2: self.cabecalhos[idx] = (tokens_s[0], tokens_s[1])
            for t in tokens_s: self.mapa_sentencas[t].append(idx)
        for i in range(len(self.raw_tokens) - 2):
            self.matrix[(self.raw_tokens[i], self.raw_tokens[i+1])][self.raw_tokens[i+2]] += 1
        freq = Counter(self.raw_tokens)
        for t in freq:
            self.metadata[t] = {"rar": math.log2(len(self.raw_tokens)/(freq[t]+0.5)), "xyz": self._get_multimodal_xyz(t)}
        print(f"🧬 V120-ULTRA: Cérebro de Camadas Lógicas Ativo.")

    def pensar(self, prompt):
        raw_tokens = self.normalize(prompt).split()
        foco = [t for t in raw_tokens if t in self.metadata]
        if not foco: return "Nexo offline."

        # CAMADA 1: Percepção de Intenção e Baricentro
        centroid_input = {m: tuple(sum(self.metadata[t]["xyz"][m][i] for t in foco)/len(foco) for i in range(3)) for m in ['L', 'S', 'E']}
        
        # CAMADA 2: Evocação de Realidades (Simulação)
        scores = Counter()
        for t in foco:
            for idx in self.mapa_sentencas[t]: scores[idx] += self.metadata[t]["rar"] * 10
        top_id = scores.most_common(1)[0][0] if scores else 0
        
        multiverso = []
        for _ in range(1000):
            # Gera trajetórias estocásticas baseadas no nexo inicial
            w1, w2 = self.cabecalhos.get(top_id, random.choice(list(self.matrix.keys())))
            multiverso.append(self._simular_trajetoria(w1, w2))

        # CAMADA 3: Julgamento de Similaridade (O Filtro do Cérebro)
        return self._eleger_conclusao(multiverso, foco, centroid_input)

    def _simular_trajetoria(self, w1, w2):
        caminho = [w1, w2]
        for i in range(35):
            opcoes = self.matrix.get((w1, w2), Counter())
            if not opcoes: break
            candidatos = list(opcoes.keys())
            # Escolha probabilística simples para gerar diversidade
            pesos = [opcoes[c]**2 for c in candidatos]
            w3 = random.choices(candidatos, weights=pesos, k=1)[0]
            caminho.append(w3)
            if w3 == "." and i > 5: break
            w1, w2 = w2, w3
        return caminho

    def _eleger_conclusao(self, multiverso, foco_input, centroid_input):
        def medir_similaridade(c):
            # 1. Similaridade de Identidade (Tokens do prompt na resposta)
            sinergia = sum(1 for t in c if t in foco_input)
            
            # 2. Similaridade Geométrica (Distância entre os baricentros)
            c_foco = [t for t in c if t in self.metadata]
            if not c_foco: return -1000
            centroid_c = {m: tuple(sum(self.metadata[t]["xyz"][m][i] for t in c_foco)/len(c_foco) for i in range(3)) for m in ['L', 'S', 'E']}
            
            distancia_total = 0
            for m in ['L', 'S', 'E']:
                distancia_total += math.sqrt(sum((centroid_c[m][i] - centroid_input[m][i])**2 for i in range(3)))
            
            # 3. Massa Informativa (Raridade)
            massa = sum(self.metadata[t]["rar"] for t in c if t in self.metadata)
            
            # Score Final: Sinergia forte e Distância curta
            return (sinergia * 500) + (massa * 10) - (distancia_total / 100)

        eleita = max(multiverso, key=medir_similaridade)
        return " ".join(eleita).capitalize().replace(" .", ".").replace(" ,", ",")

# --- EXECUÇÃO ---
if __name__ == "__main__":
    dataset = """
    João era cego, mas tinha o profundo desejo de conhecer as cores. Para isso, ele precisava de alguém que o ajudasse a enxergar o mundo através das palavras. 
    Ele nunca tinha visto a cor verde, mas, ainda assim, amava a camisa verde que é favorita. 
    João ficava muito triste quando alguém o ofendia, chamando-o de "cego burro". 
    No entanto, existia uma esperança, ele poderia voltar a enxergar se fizesse uma cirurgia para implantar um olho robótico.
    """
    motor = Quintikus_LGMLL_V120_Ultra(dataset)
    while True:
        p = input("\n👤 HUMANO > ").strip()
        if not p or p.lower() in ['sair', 'exit']: break
        print(f"🤖 QUINTIKUS > {motor.pensar(p)}\n" + "─"*60)
