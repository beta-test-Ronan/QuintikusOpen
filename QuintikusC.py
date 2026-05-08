import numpy as np
import unicodedata
import time
import os
import math
import random
from collections import defaultdict

# ==================================================================
# 1. O HARDWARE (Engine v9.4.2 - Ética Sináptica)
# ==================================================================
class AutoLeiEngine:
    def __init__(self, dim=16):
        self.dim = dim
        self.words = {}  
        self.W = {}      
        self.lr = 0.05   
        self.temperatura = 0.0 
        self.limiar_revolucao = 0.8 
        self.memoria_recente = [] 
        self.ultima_energia = {}  
        self.fator_inibicao = {}  

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
        
        chave = (s, v, o)
        self.ultima_energia[chave] = dissonancia
        if chave not in self.memoria_recente:
            self.memoria_recente.append(chave)
            self.fator_inibicao[chave] = 4.0 
            if len(self.memoria_recente) > 20: self.memoria_recente.pop(0)

        # LEI 8: REVOLUÇÃO (Agora guiada pela Lei 15)
        forca_novo = afeto_o * raridade_o * autoridade
        if dissonancia > self.limiar_revolucao and forca_novo > 3.0:
            self.temperatura = 1.0 
            self.W[v] = np.eye(self.dim) + np.random.randn(self.dim, self.dim) * 0.1
            
            for s_old, v_old, o_old in self.memoria_recente:
                if v_old == v and (s_old, v_old, o_old) != chave:
                    self.fator_inibicao[(s_old, v_old, o_old)] = 0.5
            
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
# 2. MENTE E MEDULA
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
# 3. O CÉREBRO INTEGRADO (v9.5 com Epistemologia de Fonte)
# ==================================================================
class CerebroV9:
    # LEI 15: HIERARQUIA DE FONTE
    FONTES = {
        "livro_escola": 100.0, # Dogma (Revolução garantida)
        "professor": 50.0,     # Quase dogma
        "wikipedia": 20.0,     # Revolução se bater limiar
        "jornal": 10.0,        # Adaptação forte
        "povo_falou": 1.0,     # Adaptação fraca (Padrão)
        "fofoca": 0.1,         # Quase ignorado
        "sonho": 0.0           # Não aprende, só consolida
    }

    def __init__(self):
        self.engine = AutoLeiEngine()
        self.medula = Medula()
        self.mente = MenteBeta(self.engine)
        self.stop_words = ["o", "a", "de", "que", "do", "da", "ele", "sobre", "um", "uma"]
        self.canon_v = {"é": "e", "sao": "e", "possui": "tem"}

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return t.replace("?", "").replace("!", "").strip()

    def dormir(self, ciclos=30):
        if not self.engine.memoria_recente: return "Vazio."
        print(f"\n[SONO] Quintikus consolidando saber ({ciclos} ciclos)...")
        for i in range(ciclos):
            pesos = []
            for fato in self.engine.memoria_recente:
                energia = self.engine.sentir_energia(*fato)
                fator = self.engine.fator_inibicao.get(fato, 4.0)
                pesos.append(1 + energia * fator)
            escolha = random.choices(self.engine.memoria_recente, weights=pesos, k=1)[0]
            self.engine.pulsar(*escolha, self.mente.afeto[escolha[2]], self.mente.raridade[escolha[2]], 1.0)
            self.engine.temperatura *= 0.8
        return "Sono concluído. Paradigma estabilizado."

    def escutar(self, entrada, fonte="povo_falou"):
        autoridade = self.FONTES.get(fonte, 1.0)
        entrada_orig = entrada.strip()
        entrada_norm = self.normalizar(entrada_orig)
        eh_pergunta = entrada_orig.endswith('?') or entrada_norm.startswith(('quem', 'o que'))
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
            
            # Medula usa autoridade da fonte para decidir se sente dor
            if self.medula.reflexo_dor(s, v, o, energia_nova, autoridade):
                return f"REFLEXO DE DOR: Paradoxo detectado (Fonte: {fonte} é insuficiente)!"

            self.mente.afeto[o] += 0.1 * autoridade
            diss, status = self.engine.pulsar(s, v, o, self.mente.afeto[o], self.mente.raridade[o], autoridade)
            self.medula.registrar(s, v, o)
            return f"Integrado [{status}] via {fonte}: Dissonância {diss:.4f}"
        
        return "Ruído."

# ==================================================================
# 4. EXECUÇÃO: O TESTE DA EPISTEMOLOGIA
# ==================================================================
def main():
    brain = CerebroV9()

    print("\n--- AULA 1: FOFOCA NO RECREIO ---")
    # Tenta ensinar algo com autoridade baixa
    print(brain.escutar("carlos e medico", fonte="povo_falou")) 
    print(brain.escutar('o que carlos e?')) 

    print("\n--- AULA 2: LIVRO DE CIÊNCIAS ---")
    # Livro tem autoridade 100. Força Revolução.
    print(brain.escutar("carlos e robo", fonte="livro_escola")) 
    print(f"Antes do Sono: {brain.escutar('o que carlos e?')}") 

    print(brain.dormir(ciclos=50))

    print("\n--- AULA 3: TENTATIVA DE DESINFORMAÇÃO ---")
    # Fofoca tenta mudar o dogma. Medula deve travar ou Adaptação ser mínima.
    print(brain.escutar("carlos e medico", fonte="fofoca")) 
    print(f"Final: {brain.escutar('o que carlos e?')}") 

if __name__ == "__main__":
    main()
