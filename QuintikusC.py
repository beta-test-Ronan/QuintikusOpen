import numpy as np
import unicodedata
import time
import os
import math
from collections import defaultdict

# ==================================================================
# 1. A MEDULA (Reflexo de Dor - Calibrada pra 16D)
# ==================================================================
class Medula:
    def __init__(self):
        self.nervos = {}

    def reflexo_dor(self, s, v, o_novo, energia_nova, autoridade):
        antigo = self.nervos.get((s, v))
        # PATCH 1: Dor limiar 0.5 (Mais sensível em 16D)
        if antigo and antigo != o_novo and energia_nova > 0.5:
            # PATCH 2: Anestesia exige autoridade >= 5.0
            if autoridade < 5.0:
                return True
        return False

    def registrar(self, s, v, o):
        self.nervos[(s, v)] = o

# ==================================================================
# 2. O CÓRTEX (Engine de Matrizes - Calibrada pra 16D)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=16):
        self.dim = dim
        self.words = {}  
        self.W = {}      
        self.lr = 0.05   
        self.temperatura = 0.0 
        # PATCH 3: Limiar de revolução proporcional ao dim
        self.limiar_revolucao = 0.8 
        
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
        
        # LEI 8: REVOLUÇÃO (Dissonância Catastrófica)
        forca_novo = afeto_o * raridade_o * autoridade
        # PATCH 4: Força 3.0 já é crise pra 16D
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
# 3. A MENTE BETA (As 7 Leis Cognitivas)
# ==================================================================
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
            if v_decisao > 0.01:
                candidatos.append((v_decisao, o, energia))
        
        if not candidatos: return None, 1.0
        candidatos.sort(reverse=True)
        
        melhor_o, melhor_e = candidatos[0][1], candidatos[0][2]
        if self.engine.sentir_energia(s, v, melhor_o) > melhor_e * 1.2:
            melhor_o = "Dúvida Meta-cognitiva"
            
        self.afeto[s] += 0.02
        self.frequencia[melhor_o] += 1
        return melhor_o, melhor_e

# ==================================================================
# 4. O CÉREBRO INTEGRADO (v9.3)
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
            if not res or res == "Dúvida Meta-cognitiva":
                return "Dissonância detectada no paradigma."
            return f"Bot: {s} {v} {res} (Confiança: {1-e:.2f})"
        
        elif len(tokens) >= 3:
            s, v, o = tokens[0], tokens[1], " ".join(tokens[2:])
            v = self.canon_v.get(v, v)
            
            energia_nova = self.engine.sentir_energia(s, v, o)
            if self.medula.reflexo_dor(s, v, o, energia_nova, autoridade):
                return f"REFLEXO DE DOR: Paradoxo detectado! '{s} {v} {o}' queima a lógica."

            self.mente.afeto[o] += 0.1 * autoridade
            self.mente.atualizar_raridade()
            
            diss, status = self.engine.pulsar(
                s, v, o, 
                afeto_o=self.mente.afeto[o], 
                raridade_o=self.mente.raridade[o],
                autoridade=autoridade
            )
            
            self.medula.registrar(s, v, o)
            self.mente.frequencia[o] += 1
            return f"Integrado [{status}]: Dissonância {diss:.4f} | Temperatura {self.engine.temperatura:.2f}"
        
        return "Ruído descartado."

# ==================================================================
# 5. EXECUÇÃO: O TESTE DO SISTEMA NERVOSO
# ==================================================================
def main():
    brain = CerebroV9()
    
    print("--- FASE 1: Estabelecendo Realidade (Carlos é médico) ---")
    print(brain.escutar("carlos e medico"))
    
    print("\n--- FASE 2: Testando o Paradoxo (Reflexo de Dor) ---")
    # Tenta dizer que carlos é uma doença sem autoridade. A Medula deve bloquear.
    print(brain.escutar("carlos e doenca")) 

    print("\n--- FASE 3: Testando a Revolução (Autoridade Galileu) ---")
    # Agora com autoridade alta, o sistema aceita a dor e muda o paradigma.
    print(brain.escutar("carlos e robo", autoridade=20.0))

    print(f"\n--- FASE 4: Consulta Pós-Crise ---")
    print(brain.escutar("o que carlos e?"))

if __name__ == "__main__":
    main()
