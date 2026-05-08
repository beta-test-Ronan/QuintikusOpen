import numpy as np
import unicodedata
import time
import os
import math
import random # Necessário para o random.choices
from collections import defaultdict

# ==================================================================
# 1. O HARDWARE (Engine de Matrizes - v9.4)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=16):
        self.dim = dim
        self.words = {}  
        self.W = {}      
        self.lr = 0.05   
        self.temperatura = 0.0 
        self.limiar_revolucao = 0.8 
        self.memoria_recente = [] # Buffer de fatos (S, V, O)
        self.ultima_energia = {}  # Guarda a última dissonância de cada fato

    def _get_v(self, w):
        if w not in self.words:
            v = np.random.randn(self.dim) * 0.1
            self.words[w] = v / (np.linalg.norm(v) + 1e-9)
        return self.words[w]

    def _get_W(self, v):
        if v not in self.W:
            self.W[v] = np.eye(self.dim) + np.random.randn(self.dim, self.dim) * 0.01
        return self.W[v]

    def sentir_energia(self, s, v, o):
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        return np.linalg.norm((vs @ Wv) - vo)

    def pulsar(self, s, v, o, afeto_o, raridade_o, autoridade):
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        
        proj = vs @ Wv
        delta_a = proj - vo
        dissonancia = np.linalg.norm(delta_a)
        
        # Registra energia para o sono
        self.ultima_energia[(s, v, o)] = dissonancia
        if (s, v, o) not in self.memoria_recente:
            self.memoria_recente.append((s, v, o))
            if len(self.memoria_recente) > 20: self.memoria_recente.pop(0)

        # LEI 8: REVOLUÇÃO
        forca_novo = afeto_o * raridade_o * autoridade
        if dissonancia > self.limiar_revolucao and forca_novo > 3.0:
            self.temperatura = 1.0 
            self.W[v] = np.eye(self.dim) + np.random.randn(self.dim, self.dim) * 0.1
            lr_efetivo = self.lr * 5.0 
            status = "REVOLUÇÃO"
        else:
            lr_efetivo = self.lr / (1.0 + afeto_o * raridade_o)
            status = "ADAPTAÇÃO"

        self.temperatura *= 0.9
        lr_final = lr_efetivo * (1.0 + self.temperatura)

        self.W[v] -= lr_final * np.outer(vs, delta_a)
        self.words[s] -= lr_final * (Wv @ delta_a)
        self.words[o] += lr_final * delta_a
        
        self.words[s] /= (np.linalg.norm(self.words[s]) + 1e-9)
        self.words[o] /= (np.linalg.norm(self.words[o]) + 1e-9)
        
        return dissonancia, status

# ==================================================================
# 2. A MENTE E MEDULA
# ==================================================================
class Medula:
    def __init__(self): self.nervos = {}
    def reflexo_dor(self, s, v, o_novo, energia_nova, autoridade):
        antigo = self.nervos.get((s, v))
        if antigo and antigo != o_novo and energia_nova > 0.5:
            if autoridade < 5.0: return True
        return False
    def registrar(self, s, v, o): self.nervos[(s, v)] = o

class MenteBeta:
    def __init__(self, engine):
        self.engine = engine
        self.afeto = defaultdict(lambda: 0.1)      
        self.frequencia = defaultdict(int) 
        self.raridade = defaultdict(lambda: 1.0)
        
    def atualizar_raridade(self):
        for w in self.frequencia:
            self.raridade[w] = 1.0 / (math.log(self.frequencia[w] + 1.1) + 0.1)

    def processar_veredito(self, s, v):
        self.atualizar_raridade()
        candidatos = []
        for o in list(self.engine.words.keys()):
            if o == s: continue
            energia = self.engine.sentir_energia(s, v, o)
            v_decisao = ((1-energia) * self.afeto[o] * self.raridade[o]) / (energia + 0.01)
            if v_decisao > 0.01: candidatos.append((v_decisao, o, energia))
        if not candidatos: return None, 1.0
        candidatos.sort(reverse=True)
        return candidatos[0][1], candidatos[0][2]

# ==================================================================
# 3. O CÉREBRO INTEGRADO (v9.4 com Sono Ponderado)
# ==================================================================
class CerebroV9:
    def __init__(self):
        self.engine = AutoLeiEngine()
        self.medula = Medula()
        self.mente = MenteBeta(self.engine)
        self.stop_words = ["o", "a", "de", "que", "do", "da", "ele", "sobre", "um", "uma"]
        self.canon_v = {"é": "e", "sao": "e", "possui": "tem"}

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return t.replace("?", "").replace("!", "").strip()

    def dormir(self, ciclos=10):
        """
        LEI 10: Consolidação Sináptica Ponderada.
        O cérebro foca em sonhar com o que 'doeu' (maior energia).
        """
        if not self.engine.memoria_recente:
            return "Nada para consolidar."
            
        print(f"\n[SONO] Quintikus consolidando trauma ({ciclos} ciclos)...")
        
        for i in range(ciclos):
            # AJUSTE: Pesos baseados na última energia (Trauma)
            # Fatos com energia alta têm até 4x mais chance de serem escolhidos
            pesos = [1 + self.engine.ultima_energia.get((s,v,o), 0) * 4 
                     for s,v,o in self.engine.memoria_recente]
            
            # Escolha ponderada
            escolha = random.choices(self.engine.memoria_recente, weights=pesos, k=1)[0]
            s, v, o = escolha
            
            # Re-pulsar suavemente (Consolidação)
            self.engine.pulsar(s, v, o, self.mente.afeto[o], self.mente.raridade[o], autoridade=1.0)
            
            # Esfriamento rápido da temperatura (Cortisol)
            self.engine.temperatura *= 0.8
            
        return f"Sono concluído. Estabilidade restaurada."

    def escutar(self, entrada, autoridade=1.0):
        entrada_orig = entrada.strip()
        entrada_norm = self.normalizar(entrada_orig)
        eh_pergunta = entrada_orig.endswith('?') or \
                      entrada_norm.startswith(('quem', 'o que', 'qual', 'onde', 'por que'))
        tokens = [t for t in entrada_norm.split() if t not in self.stop_words]
        
        if eh_pergunta and len(tokens) >= 2:
            s, v = tokens[0], tokens[1]
            v = self.canon_v.get(v, v)
            res, e = self.mente.processar_veredito(s, v)
            if not res: return "Dissonância."
            return f"Bot: {s} {v} {res} (Confiança: {1-e:.2f})"
        
        elif len(tokens) >= 3:
            s, v, o = tokens[0], tokens[1], " ".join(tokens[2:])
            v = self.canon_v.get(v, v)
            energia_nova = self.engine.sentir_energia(s, v, o)
            if self.medula.reflexo_dor(s, v, o, energia_nova, autoridade):
                return f"REFLEXO DE DOR: Paradoxo detectado!"

            self.mente.afeto[o] += 0.1 * autoridade
            diss, status = self.engine.pulsar(s, v, o, self.mente.afeto[o], self.mente.raridade[o], autoridade)
            self.medula.registrar(s, v, o)
            return f"Integrado [{status}]: Dissonância {diss:.4f} | Temperatura {self.engine.temperatura:.2f}"
        
        return "Ruído."

# ==================================================================
# 4. EXECUÇÃO
# ==================================================================
def main():
    brain = CerebroV9()
    
    print("--- FASE 1: Aprendizado ---")
    brain.escutar("carlos e medico")
    
    print("\n--- FASE 2: Revolução (Trauma) ---")
    # Carlos agora é um robô. Isso vai gerar uma Dissonância alta.
    print(brain.escutar("carlos e robo", autoridade=10.0))

    # O sistema está "quente" (Temperatura alta)
    print(f"\nTemperatura antes do sono: {brain.engine.temperatura:.4f}")

    print("\n--- FASE 3: O Sono Consolidador ---")
    print(brain.dormir(ciclos=30))

    print(f"Temperatura após o sono: {brain.engine.temperatura:.4f} \n ")
    print(f" \n Consulta final: \n {brain.escutar('o que carlos e?')}")

if __name__ == "__main__":
    main()
