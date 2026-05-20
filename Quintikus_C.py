import os
import math
import time
import random
import unicodedata
import hashlib
import re
import numpy as np
import pickle
import platform
import cmath
from collections import defaultdict, Counter

# =================================================================
# 1. CORE MATH: VETORES N-DIMENSIONAIS E QUANTUM OPS
# =================================================================
def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

# =================================================================
# 2. MOTOR COGNITIVO ARQUINET v18.6 (N-DIMENSIONAL & REFLEXIVO)
# =================================================================
class ArquinetEngineND:
    def __init__(self, dims=16):
        self.dims = dims
        self.mapa_nd = {}    # token: vector (16D)
        self.grafo = {}      # t1: {t2: amplitude_complexa}
        self.erro_global = 0.0
        self.taxa_aprendizado = 0.1
        self.limiar_poda = 0.005

    def treinar_com_reflexo(self, tokens, auth=1.0):
        """ 
        Treino com Erro de Predição: 
        A IA tenta prever o próximo token antes de processá-lo.
        """
        grafo = self.grafo
        mapa = self.mapa_nd
        tam = len(tokens)
        
        for i in range(tam - 1):
            t1, t2 = tokens[i], tokens[i+1]
            
            # Inicialização Latente N-Dimensional
            if t1 not in mapa: mapa[t1] = normalize_vector(np.random.randn(self.dims))
            if t2 not in mapa: mapa[t2] = normalize_vector(np.random.randn(self.dims))
            
            # --- CÁLCULO DO ERRO DE PREDIÇÃO ---
            previsao = self.prever_top(t1)
            erro = 1.0 if t2 != previsao else 0.0
            self.erro_global = 0.9 * self.erro_global + 0.1 * erro
            
            # --- AJUSTE SEMÂNTICO (ATRAÇÃO N-D) ---
            # Move os vetores t1 e t2 um em direção ao outro no espaço latente
            move = self.taxa_aprendizado * auth * (1.0 + erro)
            mapa[t1] = normalize_vector(mapa[t1] + (mapa[t2] - mapa[t1]) * move)
            
            # --- AJUSTE QUÂNTICO (AMPLITUDE) ---
            if t1 not in grafo: grafo[t1] = {}
            if t2 not in grafo[t1]:
                grafo[t1][t2] = 0.5 + 0j
            
            # Rotação de fase baseada no erro: Erro alto = mudança drástica de fase
            theta = (0.1 * auth) + (erro * 0.2)
            grafo[t1][t2] *= cmath.exp(1j * theta)
            grafo[t1][t2] += 0.05 * auth # Reforço de massa
            
            # Conservação de Energia (Normalização local do elo)
            mag = abs(grafo[t1][t2])
            if mag > 2.0: grafo[t1][t2] /= mag 

    def prever_top(self, token):
        if token not in self.grafo: return None
        opcoes = self.grafo[token]
        if not opcoes: return None
        return max(opcoes, key=lambda k: abs(opcoes[k])**2)

    def sonhar(self, intensidade=100):
        """ 
        Ciclo de Reflexão Interna (Sonho):
        Re-percorre caminhos fortes para consolidar e poda elos fracos.
        """
        for _ in range(intensidade):
            if not self.grafo: break
            t1 = random.choice(list(self.grafo.keys()))
            if not self.grafo[t1]: continue
            
            # Simula um salto
            t2 = self.prever_top(t1)
            if t2:
                # Consolidação: aproxima ainda mais no espaço latente
                self.mapa_nd[t1] = normalize_vector(self.mapa_nd[t1] + (self.mapa_nd[t2] - self.mapa_nd[t1]) * 0.01)
                # Reforço de amplitude quântica
                self.grafo[t1][t2] *= 1.01

        # Poda (Compressão Semântica)
        for t1 in list(self.grafo.keys()):
            for t2 in list(self.grafo[t1].keys()):
                if abs(self.grafo[t1][t2])**2 < self.limiar_poda:
                    del self.grafo[t1][t2]

# =================================================================
# 3. QUINTIKUS OPEN AURIA v18.6 (THE THINKING LOOP)
# =================================================================
class QuintikusOpenAuria:
    def __init__(self):
        self.path_brain = "brain_v18_latent.qoa"
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        self.cognition = ArquinetEngineND(dims=16)
        self.l2_mass, self.l2_pil_min = [], []
        self.neuronios = defaultdict(list)
        self.pil_user = 0.0
        self.user_name = None
        self.janela_contexto = []

    def boot(self):
        if os.path.exists("user.bin"):
            with open("user.bin", 'rb') as f:
                d = pickle.load(f)
                self.pil_user, self.user_name = d.get('pil', 0.0), d.get('name', "User")
        else:
            self.user_name = input("👤 IDENTIDADE: ").strip()

        if os.path.exists(self.path_brain):
            with open(self.path_brain, 'rb') as f:
                b = pickle.load(f)
                self.l2_mass, self.l2_pil_min, self.neuronios, self.cognition = b['m'], b['p'], b['n'], b['c']
        print(f"✅ Lucy v18.6 Latent Online. PIL: {self.pil_user:.2f} | Erro Cognitivo: {self.cognition.erro_global:.4f}")

    def amadurecer_solo(self, texto, auth=1.0, pil_min=0.0):
        frases = re.split(r'[\.\!\?]', texto)
        for f in frases:
            f = f.strip()
            if len(f) < 2: continue
            idx = len(self.l2_mass)
            self.l2_mass.append(f)
            self.l2_pil_min.append(pil_min)
            tokens = self.tokenizer.findall(f.lower())
            self.cognition.treinar_com_reflexo(tokens, auth)
            for t in tokens: self.neuronios[t].append(idx)
        self.salvar()

    def salvar(self):
        with open(self.path_brain, 'wb') as f:
            pickle.dump({'m': self.l2_mass, 'p': self.l2_pil_min, 'n': self.neuronios, 'c': self.cognition}, f)

    def pensar_e_falar(self, entrada):
        t0 = time.perf_counter()
        tokens = self.tokenizer.findall(entrada.lower())
        if not tokens: return "..."
        
        # 1. ATENÇÃO CONTEXTUAL: Usa os últimos tokens para triangulação latente
        self.janela_contexto = (self.janela_contexto + tokens)[-5:] # Mantém últimos 5
        
        # 2. TUNELAMENTO N-DIMENSIONAL (Generalização por Cosseno)
        # Se não houver elo direto, busca no espaço latente quem está "perto"
        t_alvo = tokens[-1]
        if t_alvo not in self.cognition.mapa_nd: 
            return f"Nexo '{t_alvo}' fora do espaço latente."
        
        # Busca pivo por similaridade latente (Dot Product)
        v_alvo = self.cognition.mapa_nd[t_alvo]
        candidatos_pivo = []
        
        # Otimização: Só checa tokens conhecidos nos neurônios
        for t_conhecido in list(self.neuronios.keys()):
            if t_conhecido in self.cognition.mapa_nd:
                sim = np.dot(v_alvo, self.cognition.mapa_nd[t_conhecido])
                if sim > 0.85: # Limiar de similaridade semântica
                    candidatos_pivo.append((t_conhecido, sim))
        
        if not candidatos_pivo: return "Vácuo semântico detectado."
        
        # Pega o mais similar que tenha nexos permitidos pelo PIL
        candidatos_pivo.sort(key=lambda x: x[1], reverse=True)
        
        idx_escolhido = None
        for pivo, sim in candidatos_pivo:
            frases_possiveis = [i for i in self.neuronios[pivo] if self.l2_pil_min[i] <= self.pil_user]
            if frases_possiveis:
                idx_escolhido = random.choice(frases_possiveis)
                break
        
        if idx_escolhido is None: return "[PIL-LOCK] Acesso negado ao nexo latente."

        # 3. EVOLUÇÃO DE PIL BASEADA EM COERÊNCIA
        # PIL sobe se o erro global da IA estiver baixo (IA estável e entendendo o usuário)
        estabilidade = 1.0 - self.cognition.erro_global
        if estabilidade > 0.7:
            self.pil_user = min(100.0, self.pil_user + (estabilidade * 0.02))

        dt = (time.perf_counter() - t0) * 1000000
        return f"\n[v18.6-LATENT | {dt:.2f}μs | ESTABILIDADE: {estabilidade:.2f}]\n> {self.l2_mass[idx_escolhido]}"

# =================================================================
# EXECUÇÃO
# =================================================================
if __name__ == "__main__":
    lucy = QuintikusOpenAuria()
    lucy.boot()
    while True:
        u = input(f"\n[{lucy.user_name}]👤: ").strip()
        if not u: continue
        if u.lower() == 'sonhar':
            print("🌙 Lucy está em ciclo de reflexão...")
            lucy.cognition.sonhar(500)
            lucy.salvar()
            continue
        if u.startswith("train:"):
            with open(u.split(":")[1].strip(), 'r', encoding='utf-8', errors='ignore') as f:
                lucy.amadurecer_solo(f.read())
            continue
        print(lucy.pensar_e_falar(u))
