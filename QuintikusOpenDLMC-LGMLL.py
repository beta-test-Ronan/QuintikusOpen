# -*-coding:utf8;-*-
import os, hashlib, math, random, time, sys, unicodedata
from collections import defaultdict, Counter

# HARDWARE SHIELD
os.environ["OMP_NUM_THREADS"] = "1" 

class Quintikus_dlmc_LGMLL:
    def __init__(self, raw_text):
        # 1. CONJUNTO DE MEMÓRIAS (Dataset como Long Term Memory)
        self.memoria_longo_prazo = defaultdict(Counter)
        self.metadata = {}
        self.tokens = self.normalize(raw_text).split()
        self.total_tokens = len(self.tokens)
        self.fluxo_imaginativo = []
        self._aprender()

    def normalize(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', str(t).lower().strip()) if unicodedata.category(c) != 'Mn')
        return t.replace(".", " . ").replace(",", " , ")

    def _aprender(self):
        """Fase de Consolidação de Memória"""
        freq = Counter(self.tokens)
        for i in range(len(self.tokens) - 2):
            w1, w2, w3 = self.tokens[i], self.tokens[i+1], self.tokens[i+2]
            self.memoria_longo_prazo[(w1, w2)][w3] += 1
            
        for t in freq:
            raridade = math.log2(self.total_tokens / (freq[t] + 0.5))
            self.metadata[t] = {"rar": raridade}
        print(f"🧠 Memória Consolidada: {len(self.metadata)} conceitos fixados.")

    def pensar(self, input_usuario):
        # --- PASSO 1: EVOCAÇÃO DE MEMÓRIAS ---
        foco = [t for t in self.normalize(input_usuario).split() if t in self.metadata]
        if not foco: return "Nexo perdido no silêncio."

        # --- PASSO 2: ASSOCIAÇÃO DE CONCEITOS (Expansão do Campo Semântico) ---
        print(f"\n[MONÓLOGO INTERNO]: Evocando '{' '.join(foco)}'...")
        conceitos_associados = set(foco)
        for f in foco:
            # Encontra vizinhos diretos na memória para expandir o pensamento
            vizinhos = [t for i, t in enumerate(self.tokens) if i > 0 and self.tokens[i-1] == f]
            conceitos_associados.update(vizinhos[:3]) # Associa até 3 conceitos próximos
        
        print(f"[MONÓLOGO INTERNO]: Conceitos associados: {list(conceitos_associados)}")

        # --- PASSO 3: EXPLORAR POSSIBILIDADES (Simulação Estocástica) ---
        print(f"[MONÓLOGO INTERNO]: Imaginando 1.000 futuros possíveis...")
        self.fluxo_imaginativo = []
        entradas_grafo = []
        for i in range(len(self.tokens)-1):
            if self.tokens[i] in foco:
                entradas_grafo.append((self.tokens[i], self.tokens[i+1]))
        
        if not entradas_grafo: entradas_grafo = [("joao", "era")]

        for _ in range(1000):
            self.fluxo_imaginativo.append(self._imaginar(random.choice(entradas_grafo)))

        # --- PASSO 4: CHEGAR A UMA CONCLUSÃO (Colapso por Coerência e Sinergia) ---
        conclusao = self._concluir(conceitos_associados)
        
        # --- PASSO 5: EXPRESSÃO (Output) ---
        return conclusao

    def _imaginar(self, semente, profundidade=30):
        w1, w2 = semente
        caminho = [w1, w2]
        for _ in range(profundidade):
            opcoes = self.memoria_longo_prazo.get((w1, w2))
            if not opcoes: break
            candidatos = list(opcoes.keys())
            pesos = [opcoes[c]**2 for c in candidatos]
            w3 = random.choices(candidatos, weights=pesos, k=1)[0]
            caminho.append(w3)
            if w3 == "." and len(caminho) > 5: break
            w1, w2 = w2, w3
        return caminho

    def _concluir(self, associacoes):
        def avaliar(c):
            # Sinergia com o que foi evocado e associado
            sinergia = sum(1 for t in c if t in associacoes)
            # Massa informativa (Raridade)
            massa = sum(self.metadata[t]["rar"] for t in c if t in self.metadata)
            # Estabilidade (não terminar com palavras vazias)
            estabilidade = 1.0
            if c[-1] in ["de", "o", "a", "que", "e"]: estabilidade = 0.1
            
            return (sinergia * 50) + (massa * 5) * estabilidade

        vencedora = max(self.fluxo_imaginativo, key=avaliar)
        return " ".join(vencedora).capitalize().replace(" .", ".").replace(" ,", ",")

# ==================================================================
# 🧪 EXECUÇÃO DO CICLO COGNITIVO
# ==================================================================
dataset = """
João era cego, mas tinha o profundo desejo de conhecer as cores. Para isso, ele precisava de alguém que o ajudasse a enxergar o mundo através das palavras. 
Ele nunca tinha visto a cor verde, mas, ainda assim, amava a sua camisa verde favorita. 
João ficava muito triste quando alguém o ofendia, chamando-o de "cego burro". 
No entanto, existia uma esperança, ele poderia voltar a enxergar se fizesse uma cirurgia para implantar um olho robótico.
"""

if __name__ == "__main__":
    consciencia = Quintikus_dlmc_LGMLL(dataset)
    print("\n" + "═"*60)
    print("  QUINTIKUS V120 : HUMAN-LIKE COGNITIVE FLOW")
    print("  (Memória -> Associação -> Imaginação -> Conclusão)")
    print("═"*60)
    
    while True:
        try:
            prompt = input("\n👤 HUMANO > ").strip()
            if prompt.lower() in ['sair', 'exit']: break
            
            # Conclusão do Pensamento
            resultado = consciencia.pensar(prompt)
            
            # Expressão em palavras
            sys.stdout.write("🤖 QUINTIKUS > ")
            for palavra in resultado.split():
                for char in palavra:
                    sys.stdout.write(char); sys.stdout.flush()
                    time.sleep(0.01)
                sys.stdout.write(" "); sys.stdout.flush()
                time.sleep(0.02)
            print("\n" + "─"*60)
                
        except KeyboardInterrupt: break
