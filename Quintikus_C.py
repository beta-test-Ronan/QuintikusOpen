import os
import math
import time
import random
import re
import numpy as np
import pickle
import cmath
from collections import defaultdict, Counter

# =================================================================
# 1. NÚCLEO ARQUINET ND (1024D - PULSO E MASSA)
# =================================================================
def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

class ArquinetCore:
    def __init__(self, dims=1024):
        self.dims = dims
        self.mapa_nd = {}    
        self.grafo = {}      
        self.pulso = defaultdict(int) # Meritocracia (v21)
        self.taxa_aprendizado = 0.1

    def treinar(self, tokens, auth=1.0):
        mapa, grafo = self.mapa_nd, self.grafo
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if t1 not in mapa: mapa[t1] = normalize_vector(np.random.randn(self.dims))
            if t2 not in mapa: mapa[t2] = normalize_vector(np.random.randn(self.dims))
            
            # Atração Contextual Persistente (v19.9)
            move = self.taxa_aprendizado * auth
            mapa[t1] = normalize_vector(mapa[t1] + (mapa[t2] - mapa[t1]) * move)
            
            if t1 not in grafo: grafo[t1] = {}
            if t2 not in grafo[t1]: grafo[t1][t2] = 0.5 + 0j
            
            # Evolução de Fase Quântica (v18.6)
            grafo[t1][t2] *= cmath.exp(1j * 0.1)
            grafo[t1][t2] += 0.05 * auth
            self.pulso[t1] += 1

# =================================================================
# 2. QUINTIKUS v22.0 - SOVEREIGN ARCHITECT
# =================================================================
class QuintikusLucy:
    def __init__(self):
        self.path_brain = "brain_v22_sovereign.qoa"
        self.tokenizer = re.compile(r'[\w]+|[\?\!\.]')
        self.cognition = ArquinetCore(dims=1024)
        
        # Memória e Estrutura
        self.l2_mass, self.l2_vectors, self.l2_auth, self.l2_tokens_len = [], [], [], []
        self.neuronios = defaultdict(list)
        self.triplas = defaultdict(list) # Lógica S-R-O (v21)
        self.raridade = Counter()
        
        # Estados Dinâmicos
        self.pil_user = 0.0
        self.cache_reflexo = [] # Memória Curta (v21)
        self.drives = {"afetivo": 0.5, "curioso": 0.3, "analitico": 0.2, "criativo": 0.3}
        self.sombra_entropica = np.zeros(1024) # Sombra (v18/19)
        self.exaustao = []

    def amadurecer_solo(self, texto, auth=1.0):
        frases = re.split(r'[\.\!\?]', texto)
        for f in frases:
            f = f.strip()
            if len(f) < 2: continue
            tokens = self.tokenizer.findall(f.lower())
            if len(tokens) < 2: continue
            
            idx = len(self.l2_mass)
            self.l2_mass.append(f)
            self.l2_auth.append(auth)
            self.l2_tokens_len.append(len(tokens))
            
            # Extração de Triplas (Lógica Estruturada)
            if len(tokens) >= 3:
                self.triplas[tokens[0]].append((tokens[1], " ".join(tokens[2:])))

            for t in tokens:
                self.raridade[t] += 1
                self.neuronios[t].append(idx)
            
            self.cognition.treinar(tokens, auth)
            
            # Vetor de Frase Entrópico (Raridade Linear)
            v_frase = np.zeros(1024)
            for t in tokens:
                if t in self.cognition.mapa_nd:
                    peso = 1.0 / (math.log(self.raridade[t] + 2))
                    v_frase += self.cognition.mapa_nd[t] * peso
            self.l2_vectors.append(normalize_vector(v_frase))
        self.salvar()

    def salvar(self):
        with open(self.path_brain, 'wb') as f:
            pickle.dump({'m': self.l2_mass, 'v': self.l2_vectors, 'n': self.neuronios, 'c': self.cognition, 
                         't': self.triplas, 'r': self.raridade, 'a': self.l2_auth, 'tl': self.l2_tokens_len}, f)

    def pensar_e_falar(self, entrada):
        t0 = time.perf_counter()
        tokens = self.tokenizer.findall(entrada.lower())
        if not tokens: return "..."

        # 1. NÍVEL 0: CACHE DE REFLEXO (O(1))
        v_entrada = np.zeros(1024)
        for t in tokens:
            if t in self.cognition.mapa_nd:
                peso = 1.0 / (math.log(self.raridade[t] + 2))
                v_entrada += self.cognition.mapa_nd[t] * peso
        v_entrada = normalize_vector(v_entrada)

        for v_antigo, res_idx in self.cache_reflexo:
            if np.dot(v_entrada, v_antigo) > 0.98: return f"[REFLEXO] > {self.l2_mass[res_idx]}"

        # 2. NÍVEL 1: LÓGICA E TRI_SEMÂNTICA
        eh_pergunta = "?" in entrada
        sujeito = next((t for t in tokens if t in self.triplas), None)
        if eh_pergunta and sujeito:
            rel, obj = random.choice(self.triplas[sujeito])
            return f"[LÓGICA] > {sujeito.capitalize()} {rel} {obj}."

        # 3. NÍVEL 2: SOMBRA ENTRÓPICA E DRIVES
        # Atualiza Drives baseados na entrada
        if eh_pergunta: self.drives["curioso"] += 0.2
        if len(tokens) > 6: self.drives["afetivo"] += 0.1
        
        # Projeção de Sombra Contextual
        self.sombra_entropica = normalize_vector(self.sombra_entropica * 0.4 + v_entrada * 0.6)
        
        # Busca de Candidatos via Tunelamento Topológico
        pivo = max(tokens, key=lambda t: self.raridade[t], default=tokens[0])
        candidatos = self.neuronios.get(pivo, [])
        if not candidatos: return "Vácuo semântico detectado."

        # 4. SCORING SOBERANO (SINERGIA + DRIVE + SIMETRIA)
        def pontuar(idx):
            # Sinergia com a Sombra
            score = np.dot(self.sombra_entropica, self.l2_vectors[idx])
            
            # Modulação por Drives
            if eh_pergunta and self.l2_auth[idx] >= 2: score += self.drives["analitico"]
            if "?" in self.l2_mass[idx]: score += self.drives["curioso"] * 0.4
            
            # Simetria de Complexidade (v19.9.5)
            assimetria = abs(len(tokens) - self.l2_tokens_len[idx])
            score -= (assimetria * 0.1)
            
            # Anti-repetição
            if idx in self.exaustao: score -= 2.0
            return score

        amostra = random.sample(candidatos, min(len(candidatos), 100))
        idx_final = max(amostra, key=pontuar)
        
        # Atualiza Cache e Exaustão
        self.cache_reflexo.append((v_entrada, idx_final))
        if len(self.cache_reflexo) > 5: self.cache_reflexo.pop(0)
        self.exaustao.append(idx_final)
        if len(self.exaustao) > 15: self.exaustao.pop(0)

        dt = (time.perf_counter() - t0) * 1000000
        return f"\n[v22.0-SOVEREIGN | DRIVE: {max(self.drives, key=self.drives.get).upper()} | {dt:.2f}μs]\n> {self.l2_mass[idx_final]}"

    def monologo_interno(self):
        """Daydreaming: Consolidação de Pontes Lógicas"""
        chaves = list(self.cognition.mapa_nd.keys())
        if len(chaves) < 2: return
        for _ in range(100):
            t1, t2 = random.sample(chaves, 2)
            if np.dot(self.cognition.mapa_nd[t1], self.cognition.mapa_nd[t2]) > 0.6:
                if t1 not in self.cognition.grafo: self.cognition.grafo[t1] = {}
                self.cognition.grafo[t1][t2] = 0.5 + 0.1j
                self.cognition.pulso[t1] += 1

# =================================================================
# EXECUÇÃO
# =================================================================
if __name__ == "__main__":
    lucy = QuintikusLucy()
    if os.path.exists(lucy.path_brain):
        with open(lucy.path_brain, 'rb') as f:
            b = pickle.load(f)
            lucy.l2_mass, lucy.l2_vectors, lucy.neuronios, lucy.cognition = b['m'], b['v'], b['n'], b['c']
            lucy.triplas, lucy.raridade = b['t'], b['r']
            lucy.l2_auth = b.get('a', [1]*len(lucy.l2_mass))
            lucy.l2_tokens_len = b.get('tl', [5]*len(lucy.l2_mass))

    print(f"✅ Lucy v22.0 Online. Solo: {len(lucy.l2_mass)} nexos.")

    while True:
        u = input(f"\n[ronan]👤: ").strip()
        if not u: continue
        if u.lower() == 'sonhar':
            lucy.monologo_interno(); lucy.salvar(); continue
        if u.startswith("train:"):
            with open(u.split(":")[1].strip(), 'r', encoding='utf-8') as f:
                lucy.amadurecer_solo(f.read())
            continue
        print(lucy.pensar_e_falar(u))
