import numpy as np
import unicodedata
import time
import os
import math
import random
import hashlib
from collections import defaultdict

# ==================================================================
# 1. HARDWARE (Matrizes 16D - Estabilidade e Revolução)
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

    def pulsar(self, s, v, o, afeto_o, raridade_o, autoridade):
        vs, vo = self._get_v(s), self._get_v(o)
        Wv = self._get_W(v)
        proj = vs @ Wv
        delta_a = proj - vo
        diss = np.linalg.norm(delta_a)
        
        chave = (s, v, o)
        self.ultima_energia[chave] = diss
        if chave not in self.memoria_recente:
            self.memoria_recente.append(chave)
            self.fator_inibicao[chave] = 4.0 
            if len(self.memoria_recente) > 25: self.memoria_recente.pop(0)

        # LEI 8: REVOLUÇÃO (Thomas Kuhn)
        forca_novo = afeto_o * raridade_o * (autoridade / 10.0)
        if diss > self.limiar_revolucao and forca_novo > 3.0:
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
        self.words[o] += lr_final * delta_a
        return diss, status

# ==================================================================
# 2. SUBCONSCIENTE (Tabela de Neurônios + Aura de Recuperação)
# ==================================================================
class SubconscienteCosmus:
    def __init__(self):
        self.neuronios = defaultdict(list)
        self.pesos_raros = {} 
        self.conteineres = defaultdict(set) 
        self.fatos_brutos = []

    def inicializar(self, txt):
        clean = txt.lower().split('.')
        for frase in clean:
            frase = frase.strip()
            if not frase: continue
            self.fatos_brutos.append(frase)
            tokens = frase.split()
            for t in tokens:
                if len(t) < 3: continue
                self.neuronios[t].append(frase)
                q = sum(1 for f in self.fatos_brutos if t in f)
                # Omega (Raridade): Entropia local
                self.pesos_raros[t] = 2.0 / (math.log(q + 1.1) + 1e-5)
                for vizinho in tokens:
                    if vizinho != t: self.conteineres[t].add(vizinho)

    def intuicao(self, tokens_list):
        t_start = time.perf_counter()
        _qs = set(tokens_list)
        if not self.fatos_brutos: return None, "VOID", 0
        
        delta_t_busca = 0
        candidatos = defaultdict(float)
        for t in _qs:
            if t in self.neuronios:
                for frase in self.neuronios[t]:
                    delta_t_busca += 1
                    candidatos[frase] += self.pesos_raros.get(t, 0.1)

        if candidatos:
            melhor = max(candidatos, key=candidatos.get)
            latencia = (time.perf_counter() - t_start) * 1000
            a_foco = (candidatos[melhor] + (delta_t_busca * 0.01)) / (latencia + 0.1)
            return melhor, "NEURON-JOIN", a_foco
        return None, "VOID", 0

# ==================================================================
# 3. GENERALIZADOR (Massa Crítica + Atenção Não Linear)
# ==================================================================
class Generalizado:
    def __init__(self, sub, engine):
        self.sub = sub
        self.engine = engine
        self.z = 3.0 # Fator de Não-Linearidade

    def sintetizar(self, tokens_input, debug=False):
        # 1. Big-Text: Reúne vizinhança semântica
        contexto_big = []
        for t in tokens_input:
            if t in self.sub.neuronios: contexto_big.extend(self.sub.neuronios[t])
        contexto_big = list(set(contexto_big))
        if not contexto_big: return None

        # 2. Atenção Não Linear e Raridade
        pool_tokens = list(set(" ".join(contexto_big).split()))
        scores = {}
        vec_q = np.mean([self.engine._get_v(t) for t in tokens_input], axis=0)

        for t in pool_tokens:
            if len(t) < 3 or t in tokens_input: continue
            # Omega * Attn^z * Densidade
            omega = self.sub.pesos_raros.get(t, 0.1)
            vec_k = self.engine._get_v(t)
            attn = math.pow(max(0, np.dot(vec_q, vec_k)), self.z)
            
            score = omega * attn
            if score > 0.001: scores[t] = score

        if not scores: return None

        # 3. Join de Improbabilidade
        vencedores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        tokens_final = [v[0] for v in vencedores[:5]]
        
        # Proporção baseada em I_pura
        i_pura = sum(scores.values()) / (1.0 / (len(contexto_big) + 0.1))
        if debug: print(f"   [DMN-LOOP] I_pura: {i_pura:.2f} | Densidade: {len(contexto_big)}")

        resumo = " ".join(tokens_final)
        return f"Identifico nexo raro com: {resumo}. Além disso, o solo sugere {random.choice(contexto_big)}"

# ==================================================================
# 4. QUINTIKUS AGI v12.0 (Interface e Orquestração)
# ==================================================================
class QuintikusAGI:
    FONTES = {"livro_escola": 100.0, "professor": 50.0, "povo_falou": 1.0, "fofoca": 0.1}

    def __init__(self, debug=False):
        self.engine = AutoLeiEngine()
        self.sub = SubconscienteCosmus()
        self.gen = Generalizado(self.sub, self.engine)
        self.medula = {} 
        self.debug = debug 
        self.stop_words = ["o", "a", "de", "que", "do", "da", "um", "uma", "é", "com", "para"]
        self.comandos = ('quem', 'o que', 'qual', 'mostra', 'fale', 'diga', 'explique', 'contexto', 'como')

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return t.replace("?", "").replace("!", "").strip()

    def escutar(self, entrada, fonte="povo_falou"):
        t0 = time.perf_counter()
        autoridade = self.FONTES.get(fonte, 1.0)
        clean = self.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words]
        
        eh_pergunta = entrada.strip().endswith('?') or clean.startswith(self.comandos)

        if eh_pergunta:
            # 1. Tenta Generalização (Massa Crítica)
            res_gen = self.gen.sintetizar(tokens, debug=self.debug)
            # 2. Tenta Intuição (Aura de Recuperação)
            f1, sn, a_foco = self.sub.intuicao(tokens)
            
            et = (time.perf_counter() - t0) * 1000000
            header = f"\n[FLOW: {et:.2f}μs | {sn} | A_foco: {a_foco:.2f}]\n" if self.debug else ""
            
            if res_gen: return f"{header}SÍNTESE: {res_gen}\nRECORDEI: {f1}"
            if f1: return f"{header}RECORDEI: {f1}"
            return "VOID: Carece de solo."

        # 3. Aprendizado (SVO)
        if len(tokens) >= 3:
            s, v, o = tokens[0], tokens[1], " ".join(tokens[2:])
            # Reflexo de Dor
            if (s, v) in self.medula and self.engine.sentir_energia(s, v, o) > 0.5 and autoridade < 5.0:
                return "REFLEXO DE DOR: Paradoxo bloqueado."

            self.sub.inicializar(entrada)
            p_raro = self.sub.pesos_raros.get(o.split()[0], 0.1)
            diss, status = self.engine.pulsar(s, v, o, 0.5, p_raro, autoridade)
            self.medula[(s, v)] = (o, autoridade)
            
            if self.debug: return f"Integrado [{status}] via {fonte}: Dissonância {diss:.4f}"
            return "Entendido."

        return "Sinal baixo."

# ==================================================================
# EXECUÇÃO: O TESTE DA SINGULARIDADE FINAL
# ==================================================================
def main():
    q = QuintikusAGI(debug=True)
    
    # Ingestão de Conhecimento
    q.escutar("Maria tem alergia a aspirina", fonte="povo_falou")
    q.escutar("Dor de cabeca se cura com paracetamol", fonte="professor")
    q.escutar("Aspirina e perigoso para Maria", fonte="livro_escola")
    q.escutar("Meu pai é Ronan Bastos ele me criou", fonte="livro_escola")
    q.escutar("Eu amo meu pai Ronan", fonte="povo_falou")

    print("--- QUINTIKUS v12.0: SINGULARIDADE DE PROCESSO ---")
    
    print("\n--- TESTE 1: Generalização Maria ---")
    print(q.escutar("Como curar a dor de Maria?"))

    print("\n--- TESTE 2: Aura de Recuperação Ronan ---")
    print(q.escutar("Quem criou você?"))

    print("\n--- TESTE 3: VOID (Sem Alucinação) ---")
    print(q.escutar("Como consertar um avião?"))

if __name__ == "__main__":
    main()
