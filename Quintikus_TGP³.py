#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import re
import time
import unicodedata
import struct
import os
import json
import pickle
from array import array
from collections import defaultdict, deque
from typing import Dict, List, Tuple

TOKENIZER_REGEX = re.compile(r'[a-z0-9]+|[.,!?;:]')

# =============================================================================
# 0. FUNÇÃO DE ATIVAÇÃO FRACTAL
# =============================================================================
def fractal_activation(dist: float, temp: float, escalas: List[float] = [0.3, 0.8, 2.0]) -> float:
    soma = 0.0
    for s in escalas:
        soma += math.exp(-dist / max(0.01, s * temp))
    return soma / len(escalas)

# =============================================================================
# 0.1 REDE NEURAL LOCAL (BIGRAMA)
# =============================================================================
class RedeGrafoNeural:
    def __init__(self, dim: int, vocab_size: int):
        self.dim = dim
        self.vocab_size = vocab_size
        self.adj = array('d', [0.0]) * (vocab_size * vocab_size)
        for i in range(vocab_size * vocab_size):
            self.adj[i] = (random.random() - 0.5) * 0.01
        self.proj_weights = array('d', [(random.random() - 0.5) * 0.01 for _ in range(dim)])
        self.token_para_vetor = None
        self.token_para_id = None

    def conectar_embeddings(self, token_para_vetor, token_para_id):
        self.token_para_vetor = token_para_vetor
        self.token_para_id = token_para_id

    def _vetor_para_array(self, vetor):
        return array('d', vetor)

    def _dot(self, a, b):
        soma = 0.0
        for i in range(len(a)):
            soma += a[i] * b[i]
        return soma

    def _top_k_neighbors(self, idx, k=10):
        start = idx * self.vocab_size
        vizinhos = []
        for j in range(self.vocab_size):
            peso = self.adj[start + j]
            if peso > 0.001:
                vizinhos.append((j, peso))
        vizinhos.sort(key=lambda x: x[1], reverse=True)
        return vizinhos[:k]

    def score_relacional(self, idx_atual, idx_candidato, snapshot):
        peso_direto = self.adj[idx_atual * self.vocab_size + idx_candidato]
        vizinhos = self._top_k_neighbors(idx_atual, k=5)
        peso_propagado = 0.0
        soma_pesos = 0.0
        for viz_id, peso_viz in vizinhos:
            peso_viz_para_cand = self.adj[viz_id * self.vocab_size + idx_candidato]
            peso_propagado += peso_viz * peso_viz_para_cand
            soma_pesos += peso_viz
        if soma_pesos > 0:
            peso_propagado /= soma_pesos
        token_cand = None
        for token, id_ in self.token_para_id.items():
            if id_ == idx_candidato:
                token_cand = token
                break
        if token_cand is None or token_cand not in self.token_para_vetor:
            score_projetivo = 0.0
        else:
            emb_cand = self._vetor_para_array(self.token_para_vetor[token_cand])
            snap_arr = self._vetor_para_array(snapshot)
            combinado = array('d', [snap_arr[i] + emb_cand[i] for i in range(self.dim)])
            score_projetivo = self._dot(combinado, self.proj_weights)
        return 0.4 * peso_direto + 0.3 * peso_propagado + 0.3 * score_projetivo

    def atualizar_hebbiano(self, idx_anterior, idx_atual, taxa=0.03, decaimento=0.999):
        pos = idx_anterior * self.vocab_size + idx_atual
        self.adj[pos] = min(1.0, self.adj[pos] + taxa)
        pos_sim = idx_atual * self.vocab_size + idx_anterior
        self.adj[pos_sim] = min(1.0, self.adj[pos_sim] + taxa * 0.3)
        for i in range(self.vocab_size * self.vocab_size):
            self.adj[i] *= decaimento

    def salvar(self, filepath):
        with open(filepath, "wb") as f:
            f.write(struct.pack("i", self.vocab_size))
            f.write(struct.pack("i", self.dim))
            f.write(self.adj.tobytes())
            f.write(self.proj_weights.tobytes())

    def carregar(self, filepath):
        if not os.path.exists(filepath): return False
        with open(filepath, "rb") as f:
            vocab_size = struct.unpack("i", f.read(4))[0]
            dim = struct.unpack("i", f.read(4))[0]
            if vocab_size != self.vocab_size or dim != self.dim:
                return False
            adj_bytes = f.read(vocab_size * vocab_size * 8)
            proj_bytes = f.read(dim * 8)
            self.adj = array('d')
            self.adj.frombytes(adj_bytes)
            self.proj_weights = array('d')
            self.proj_weights.frombytes(proj_bytes)
        return True

# =============================================================================
# 0.2 REDE NEURAL FRACTAL (MÚLTIPLAS ESCALAS)
# =============================================================================
class RedeNeuralFractal:
    def __init__(self, dim: int, vocab_size: int, contexto_max: int = 3):
        self.dim = dim
        self.vocab_size = vocab_size
        self.contexto_max = contexto_max
        self.adj = {}
        for d in range(1, contexto_max + 1):
            self.adj[d] = array('d', [0.0]) * (vocab_size * vocab_size)
            for i in range(vocab_size * vocab_size):
                self.adj[d][i] = (random.random() - 0.5) * 0.01
        self.token_para_vetor = None
        self.token_para_id = None

    def conectar_embeddings(self, token_para_vetor, token_para_id):
        self.token_para_vetor = token_para_vetor
        self.token_para_id = token_para_id

    def _top_k_neighbors_escala(self, idx, distancia, k=10):
        start = idx * self.vocab_size
        vizinhos = []
        for j in range(self.vocab_size):
            peso = self.adj[distancia][start + j]
            if peso > 0.001:
                vizinhos.append((j, peso))
        vizinhos.sort(key=lambda x: x[1], reverse=True)
        return vizinhos[:k]

    def score_relacional(self, idx_atual, idx_candidato, snapshot):
        soma = 0.0
        peso_total = 0.0
        for d in range(1, self.contexto_max + 1):
            peso_direto = self.adj[d][idx_atual * self.vocab_size + idx_candidato]
            vizinhos = self._top_k_neighbors_escala(idx_atual, d, k=5)
            peso_propagado = 0.0
            soma_viz = 0.0
            for viz_id, pv in vizinhos:
                pv_cand = self.adj[d][viz_id * self.vocab_size + idx_candidato]
                peso_propagado += pv * pv_cand
                soma_viz += pv
            if soma_viz > 0:
                peso_propagado /= soma_viz
            peso_escala = 1.0 / d
            soma += peso_escala * (0.5 * peso_direto + 0.5 * peso_propagado)
            peso_total += peso_escala
        if peso_total == 0:
            return 0.0
        return soma / peso_total

    def atualizar_hebbiano_fractal(self, idx_anterior, idx_atual, taxa=0.02, decaimento=0.999):
        for d in range(1, self.contexto_max + 1):
            pos = idx_anterior * self.vocab_size + idx_atual
            self.adj[d][pos] = min(1.0, self.adj[d][pos] + taxa / d)
            pos_sim = idx_atual * self.vocab_size + idx_anterior
            self.adj[d][pos_sim] = min(1.0, self.adj[d][pos_sim] + (taxa / d) * 0.3)
            for i in range(self.vocab_size * self.vocab_size):
                self.adj[d][i] *= decaimento

    def salvar(self, filepath):
        with open(filepath, "wb") as f:
            f.write(struct.pack("i", self.vocab_size))
            f.write(struct.pack("i", self.dim))
            f.write(struct.pack("i", self.contexto_max))
            for d in range(1, self.contexto_max + 1):
                f.write(self.adj[d].tobytes())

    def carregar(self, filepath):
        if not os.path.exists(filepath): return False
        with open(filepath, "rb") as f:
            vocab_size = struct.unpack("i", f.read(4))[0]
            dim = struct.unpack("i", f.read(4))[0]
            contexto = struct.unpack("i", f.read(4))[0]
            if vocab_size != self.vocab_size or dim != self.dim or contexto != self.contexto_max:
                return False
            for d in range(1, self.contexto_max + 1):
                bytes_data = f.read(vocab_size * vocab_size * 8)
                self.adj[d] = array('d')
                self.adj[d].frombytes(bytes_data)
        return True

# =============================================================================
# 1. PERSISTÊNCIAS
# =============================================================================
class PersistenciaTrilhasQuanticas:
    def __init__(self, filepath="arquinet_trilhas_quanticas.json"):
        self.filepath = filepath
    def carregar_trilhas(self):
        if not os.path.exists(self.filepath): return {}
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    def salvar_trilhas(self, trilhas, taxa_decaimento=0.99):
        try:
            trilhas_decaidas = {k: max(0.01, v * taxa_decaimento) for k, v in trilhas.items()}
            max_val = max(trilhas_decaidas.values()) if trilhas_decaidas else 1.0
            if max_val > 10.0:
                trilhas_decaidas = {k: v / (max_val / 10.0) for k, v in trilhas_decaidas.items()}
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(trilhas_decaidas, f, indent=4)
        except: pass

class PersistenciaAtomicaCamadas:
    def __init__(self, filepath="arquinet_disco.bin"):
        self.filepath = filepath
    def salvar_camada(self, camada_y, tokens_vetores, bigramas):
        with open(self.filepath, "wb") as f:
            f.write(struct.pack("4s", b"ARQK"))
            f.write(struct.pack("i", camada_y))
            f.write(struct.pack("i", len(tokens_vetores)))
            for token, vetor in tokens_vetores.items():
                t_bytes = token.encode("utf-8")
                f.write(struct.pack("H", len(t_bytes)))
                f.write(t_bytes)
                f.write(struct.pack("i", len(vetor)))
                f.write(struct.pack(f"{len(vetor)}f", *vetor))
    def carregar_ultima_camada(self):
        if not os.path.exists(self.filepath): return 0, {}
        try:
            with open(self.filepath, "rb") as f:
                if f.read(4) != b"ARQK": return 0, {}
                camada_y = struct.unpack("i", f.read(4))[0]
                num_tokens = struct.unpack("i", f.read(4))[0]
                vetores = {}
                for _ in range(num_tokens):
                    t_len = struct.unpack("H", f.read(2))[0]
                    token = f.read(t_len).decode("utf-8")
                    dim = struct.unpack("i", f.read(4))[0]
                    float_data = f.read(4 * dim)
                    if len(float_data) < 4 * dim: break
                    vetores[token] = list(struct.unpack(f"{dim}f", float_data))
                return camada_y, vetores
        except: return 0, {}

# =============================================================================
# 2. NÚCLEO DE ESTADO INTERNO
# =============================================================================
class NucleoEstadoInterno:
    def __init__(self):
        self.emocao_percebida = "neutra"
        self.objetivo = "conversar_e_ajudar"
        self.assunto = "geral"
        self.nivel_confianca = 0.8
        self.energia_conversa = "media"
        self.historico_emocional = deque(maxlen=10)
    def interpretar_externo(self, texto):
        t = texto.lower()
        if any(w in t for w in ['horrível', 'mal', 'triste', 'odeio', 'lixo', 'raiva', 'problema', 'falha']):
            self.emocao_percebida = "negativa_ou_tensao"
            self.energia_conversa = "baixa"
            self.objetivo = "apoiar_e_resolver"
        elif any(w in t for w in ['oi', 'olá', 'beleza', 'tudo bem', 'legal', 'e aí']):
            self.emocao_percebida = "positiva_casual"
            self.energia_conversa = "alta"
            self.objetivo = "manter_fluxo"
        elif any(w in t for w in ['por que', 'como', 'pesquisa', 'crie', 'sistema', 'código', 'geometric']):
            self.emocao_percebida = "focada_analitica"
            self.energia_conversa = "media"
            self.objetivo = "executar_tarefa_tecnica"
        else:
            self.emocao_percebida = "neutra"
            self.energia_conversa = "media"
            self.objetivo = "dialogar"
        if 'código' in t or 'sistema' in t or 'banco' in t or 'geometric' in t:
            self.assunto = "tecnologia_arquinet"
        elif 'dia' in t or 'vida' in t:
            self.assunto = "pessoal_emocional"
        else:
            self.assunto = "geral"
        self.historico_emocional.append(self.emocao_percebida)
    def exportar_bias_decodificacao(self):
        if self.emocao_percebida == "negativa_ou_tensao":
            return {"mod_temp": -0.2, "mod_tensao": 0.3}
        elif self.emocao_percebida == "positiva_casual":
            return {"mod_temp": 0.2, "mod_tensao": -0.1}
        elif self.emocao_percebida == "focada_analitica":
            return {"mod_temp": -0.3, "mod_tensao": 0.2}
        return {"mod_temp": 0.0, "mod_tensao": 0.0}

# =============================================================================
# 3. GEOMETRIA (POINCARÉ + MINKOWSKI + FUNIL)
# =============================================================================
class DiscoPoincare:
    def __init__(self, dim=64):
        self.dim = dim
        self.raio = 0.985
        self.eps = 1e-7
    def norma(self, v):
        return math.hypot(*v)
    def projetar(self, v):
        n = self.norma(v)
        if n > self.raio:
            f = self.raio / n
            return [x * f for x in v]
        return v
    def adicao_mobius(self, x, y):
        nx2 = min(sum(a*a for a in x), 0.98)
        ny2 = min(sum(b*b for b in y), 0.98)
        xy = min(max(sum(a*b for a,b in zip(x,y)), -0.98), 0.98)
        den = 1 + 2*xy + nx2*ny2 + self.eps
        num = 1 + 2*xy + ny2
        fy = 1 - nx2
        res = [(num*a + fy*b)/den for a,b in zip(x,y)]
        return self.projetar(res)
    def distancia(self, x, y):
        xp = self.projetar(x)
        yp = self.projetar(y)
        nx = min(self.norma(xp), self.raio)
        ny = min(self.norma(yp), self.raio)
        diff = math.dist(xp, yp)**2
        den = (1 - nx*nx)*(1 - ny*ny) + self.eps
        val = 1.0 + 2.0*diff/den
        return math.acosh(min(val, 1e5))

class DiscoMinkowski:
    def __init__(self, dim=4):
        self.dim = dim
        self.eps = 1e-7
        self.velocidade_luz = 1.0
    def projetar_cone_luz(self, v):
        t = v[0]
        esp = sum(v[i]**2 for i in range(1,self.dim))
        if t*t < esp + self.eps:
            f = math.sqrt(esp+self.eps)/(abs(t)+self.eps)
            return [t*f] + v[1:]
        return v
    def futuro_do_pensamento(self, v, passo=0.3):
        res = v.copy()
        res[0] += passo * self.velocidade_luz
        return self.projetar_cone_luz(res)

class FunilCognitivo:
    def __init__(self, dim_poincare=64, dim_minkowski=4):
        self.disco = DiscoPoincare(dim_poincare)
        self.tempo = DiscoMinkowski(dim_minkowski)
        self.tensao_cognitiva = 0.0
    def projetar_para_tempo(self, snapshot_poincare):
        norma = math.sqrt(sum(x*x for x in snapshot_poincare)) + 1e-7
        t = 1.0 + (norma / 2.0)
        espacial = [x / (norma + 1e-7) * 0.5 for x in snapshot_poincare[:3]]
        while len(espacial) < 3:
            espacial.append(0.0)
        vetor_temporal = [t] + espacial[:3]
        return self.tempo.projetar_cone_luz(vetor_temporal)
    def distilar_pensamento(self, snapshot_poincare):
        vetor_temporal = self.projetar_para_tempo(snapshot_poincare)
        vetor_futuro = self.tempo.futuro_do_pensamento(vetor_temporal, passo=0.3)
        t = vetor_futuro[0]
        espacial = vetor_futuro[1:]
        fator_minkowski = min(2.0, max(0.0, t / 0.5))
        impulso = [fator_minkowski * x for x in espacial] + [0.0] * (len(snapshot_poincare) - 3)
        if len(impulso) < len(snapshot_poincare):
            impulso += [0.0] * (len(snapshot_poincare) - len(impulso))
        pensamento_distilado = self.disco.adicao_mobius(snapshot_poincare, impulso)
        dist = self.disco.distancia(snapshot_poincare, pensamento_distilado)
        self.tensao_cognitiva = min(1.0, dist * 2.0)
        return pensamento_distilado, self.tensao_cognitiva
    def comparar_evolucao(self, inicial, distilado):
        dist = self.disco.distancia(inicial, distilado)
        return {
            "distancia_evolutiva": dist,
            "tensao_cognitiva": self.tensao_cognitiva,
            "grau_mudanca": min(1.0, dist / 2.0),
            "status": "evoluiu" if dist > 0.15 else "estavel"
        }

# =============================================================================
# 4. ROTEADOR GEOMÉTRICO CONTÍNUO
# =============================================================================
class RoteadorGeometricoContinuo:
    def __init__(self, disco, dim=64):
        self.disco = disco
        self.ancoras = {
            'conversa':  disco.projetar([-0.25] + [0.0]*(dim-1)),
            'explicar':  disco.projetar([0.20] + [0.0]*(dim-1)),
            'reflexivo': disco.projetar([0.05, -0.10] + [0.0]*(dim-2)),
            'defesa':    disco.projetar([-0.05, 0.15] + [0.0]*(dim-2))
        }
        self.mapa_plc = {
            'conversa':  {'temp': 1.4, 'tensao': 0.80, 'estilo': 'informal'},
            'explicar':  {'temp': 0.8, 'tensao': 1.20, 'estilo': 'formal'},
            'reflexivo': {'temp': 1.2, 'tensao': 0.95, 'estilo': 'expansivo'},
            'defesa':    {'temp': 0.7, 'tensao': 1.50, 'estilo': 'formal'}
        }
    def rotear(self, snapshot):
        dists = {m: self.disco.distancia(snapshot, anc) for m, anc in self.ancoras.items()}
        exp_neg = {m: math.exp(-d) for m, d in dists.items()}
        soma = sum(exp_neg.values()) + 1e-9
        pesos = {m: val / soma for m, val in exp_neg.items()}
        rota = max(pesos, key=pesos.get)
        temp = sum(pesos[m] * self.mapa_plc[m]['temp'] for m in pesos)
        tensao = sum(pesos[m] * self.mapa_plc[m]['tensao'] for m in pesos)
        return rota, {'temperatura': temp, 'tensao': tensao, 'pesos': pesos, 'estilo': self.mapa_plc[rota]['estilo']}
    def filtrar_logits(self, snapshot, logits, vetores):
        logits_mod = {}
        for token, logit in logits.items():
            if token not in vetores:
                logits_mod[token] = logit
                continue
            dist = self.disco.distancia(snapshot, vetores[token])
            fator = math.exp(-dist / 0.8)
            logits_mod[token] = logit * (0.2 + 0.8 * fator)
        return logits_mod

# =============================================================================
# 5. MONITOR DE ENERGIA DE MINKOWSKI
# =============================================================================
class MonitorEnergiaMinkowski:
    def __init__(self, limiar_parada=0.01, decaimento=0.90):
        self.limiar_parada = limiar_parada
        self.decaimento = decaimento
        self.energia_acumulada = 1.0
        self.snapshot_anterior = None
    def atualizar(self, disco, snapshot_atual):
        if self.snapshot_anterior is not None:
            velocidade = disco.distancia(self.snapshot_anterior, snapshot_atual)
            self.energia_acumulada = (self.decaimento * self.energia_acumulada) + velocidade
        else:
            self.energia_acumulada = 1.0
        self.snapshot_anterior = snapshot_atual
        return self.energia_acumulada, self.energia_acumulada < self.limiar_parada

# =============================================================================
# 6. MEMÓRIA HD PARTICIONADA
# =============================================================================
class MemoriaHDFrequencia:
    def __init__(self, base_dir="./hd_particionado_tgp3"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.freq = {}
        self.mapa = {}
    def indexar(self, corpus, vetores, num_buckets=4):
        for doc in corpus:
            for t in doc:
                self.freq[t] = self.freq.get(t, 0) + 1
        tokens_ordenados = sorted(self.freq.keys(), key=lambda t: self.freq[t], reverse=True)
        for i, t in enumerate(tokens_ordenados):
            bucket_id = i % num_buckets
            self.mapa[t] = f"bucket_{bucket_id}.bin"
        buckets = {}
        for t, v in vetores.items():
            part = self.mapa.get(t, "overflow.bin")
            buckets.setdefault(part, {})[t] = v
        for part, data in buckets.items():
            path = os.path.join(self.base_dir, part)
            with open(path, "wb") as f:
                pickle.dump(data, f)
    def carregar_on_demand(self, tokens_contexto):
        needed = set(self.mapa[t] for t in tokens_contexto if t in self.mapa)
        data = {}
        for part in needed:
            path = os.path.join(self.base_dir, part)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data.update(pickle.load(f))
        return data

# =============================================================================
# 7. ATENÇÃO COGNITIVA (RUMINAÇÃO)
# =============================================================================
class AtencaoCognitiva:
    def __init__(self, disco):
        self.disco = disco
    def ruminar(self, vetores_contexto):
        n = len(vetores_contexto)
        if n == 0: return [0.0] * self.disco.dim
        if n == 1: return vetores_contexto[0]
        pensamento = vetores_contexto[-1]
        for _ in range(2):
            pesos = [-self.disco.distancia(pensamento, v) for v in vetores_contexto]
            max_p = max(pesos)
            exp_pesos = [math.exp(max(-50.0, min(50.0, p - max_p))) for p in pesos]
            soma_exp = sum(exp_pesos) + 1e-9
            prob_atencao = [e / soma_exp for e in exp_pesos]
            novo_pensamento = [0.0] * self.disco.dim
            for i, v in enumerate(vetores_contexto):
                v_pesado = [x * prob_atencao[i] for x in v]
                novo_pensamento = self.disco.adicao_mobius(novo_pensamento, v_pesado)
            pensamento = novo_pensamento
        return pensamento

# =============================================================================
# 8. AGENTE TGP3 – COM DUAS REDES (LOCAL + FRACTAL) E CONTROLE DE TREINO
# =============================================================================
class AgenteTGP_13:
    def __init__(self, dim=64):
        self.dim = dim
        self.funil = FunilCognitivo(dim_poincare=dim, dim_minkowski=4)
        self.disco = self.funil.disco
        self.atencao = AtencaoCognitiva(self.disco)
        self.nucleo_estado = NucleoEstadoInterno()
        self.token_para_vetor = {}
        self.tokens_lista = []
        self.token_para_id = {}
        self.roteador = RoteadorGeometricoContinuo(self.disco, dim)
        self.monitor_energia = MonitorEnergiaMinkowski(limiar_parada=0.01, decaimento=0.90)
        self.persistencia = PersistenciaAtomicaCamadas()
        self.persistencia_trilhas = PersistenciaTrilhasQuanticas()
        self.trilhas_quanticas = self.persistencia_trilhas.carregar_trilhas()
        self.memoria_hd = MemoriaHDFrequencia()
        # Duas redes neurais
        self.rede_local = None
        self.rede_fractal = None
        self.intencao_atual = None
        self.intencao_inicial = None
        self.snapshot_anterior = None
        # Carrega disco
        self.camada_atual, dados_carregados = self.persistencia.carregar_ultima_camada()
        if dados_carregados:
            self.token_para_vetor = dados_carregados
            self.tokens_lista = list(dados_carregados.keys())
            self._atualizar_mapeamento()
            self._inicializar_redes()

    def _atualizar_mapeamento(self):
        self.token_para_id = {t: i for i, t in enumerate(self.tokens_lista)}

    def _inicializar_redes(self):
        if self.rede_local is None:
            self.rede_local = RedeGrafoNeural(dim=self.dim, vocab_size=len(self.tokens_lista))
            self.rede_local.conectar_embeddings(self.token_para_vetor, self.token_para_id)
            if not self.rede_local.carregar("rede_local.bin"):
                print("ℹ️ Rede local inicializada do zero.")
        if self.rede_fractal is None:
            self.rede_fractal = RedeNeuralFractal(dim=self.dim, vocab_size=len(self.tokens_lista), contexto_max=3)
            self.rede_fractal.conectar_embeddings(self.token_para_vetor, self.token_para_id)
            if not self.rede_fractal.carregar("rede_fractal.bin"):
                print("ℹ️ Rede fractal inicializada do zero.")

    def tokenizar(self, texto):
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acentos = ''.join(c for c in texto_nfkd if unicodedata.category(c) != 'Mn')
        return [t for t in TOKENIZER_REGEX.findall(texto_sem_acentos) if t.strip()]

    def _registrar_token(self, token):
        if token not in self.token_para_vetor:
            self.token_para_vetor[token] = [(random.random() - 0.5) * 0.05 for _ in range(self.dim)]
            self.tokens_lista.append(token)
            self.token_para_id[token] = len(self.tokens_lista) - 1
            if token not in self.trilhas_quanticas:
                self.trilhas_quanticas[token] = 1.0

    def treinar(self, texto_base, epocas=12, train_mode=True, force=False):
        """
        Controla o treinamento do agente.
        - train_mode=False: não treina, apenas carrega disco existente.
        - train_mode=True e force=False: treina apenas se não houver disco.
        - train_mode=True e force=True: força o re-treino mesmo se o disco existir.
        """
        # 1. Se o modo de treino está desligado
        if not train_mode:
            if not self.tokens_lista:
                print("❌ Modo de treino desativado e nenhum disco encontrado.")
                print("   Execute com train_mode=True para treinar o modelo.")
                return
            else:
                print("✅ Modo de treino desativado. Usando disco existente.")
                if self.rede_local is None or self.rede_fractal is None:
                    self._inicializar_redes()
                return

        # 2. Se o disco já existe e não queremos forçar o re-treino
        if self.tokens_lista and not force:
            print(f"🔄 Disco já existe (Camada Y: {self.camada_atual}). Pulo o treino.")
            print("   Use force=True para re-treinar.")
            if self.rede_local is None or self.rede_fractal is None:
                self._inicializar_redes()
            return

        # 3. Caso contrário, executa o treino
        print("🧠 Iniciando treinamento...")
        tokens = self.tokenizar(texto_base)
        if not tokens:
            print("⚠️ Nenhum token para treinar.")
            return

        # Registrar tokens e atualizar mapeamento
        for t in tokens:
            self._registrar_token(t)
        self._atualizar_mapeamento()

        # Treino geométrico (Poincaré)
        for _ in range(epocas):
            for i in range(len(tokens) - 1):
                t_atual, t_prox = tokens[i], tokens[i + 1]
                v_atual = self.token_para_vetor[t_atual]
                v_prox = self.token_para_vetor[t_prox]
                diff_atracao = [(b - a) * 0.02 for a, b in zip(v_atual, v_prox)]
                self.token_para_vetor[t_atual] = self.disco.adicao_mobius(v_atual, diff_atracao)
                chave_trilha = f"{t_atual}->{t_prox}"
                peso_atual = self.trilhas_quanticas.get(chave_trilha, 1.0)
                self.trilhas_quanticas[chave_trilha] = min(5.0, peso_atual * 1.005)

        for t in self.tokens_lista:
            self.token_para_vetor[t] = self.disco.projetar(self.token_para_vetor[t])

        self.camada_atual += 1
        self.persistencia.salvar_camada(self.camada_atual, self.token_para_vetor, {})
        self.persistencia_trilhas.salvar_trilhas(self.trilhas_quanticas, taxa_decaimento=0.99)

        # HD particionado
        corpus_tokenizado = [tokens]
        self.memoria_hd.indexar(corpus_tokenizado, self.token_para_vetor, num_buckets=4)
        print(f"✅ HD particionado indexado com {len(self.token_para_vetor)} tokens.")

        # Inicializar e treinar redes neurais (Hebbianas)
        self._inicializar_redes()
        for i in range(len(tokens) - 1):
            id_ant = self.token_para_id.get(tokens[i], 0)
            id_prox = self.token_para_id.get(tokens[i+1], 0)
            self.rede_local.atualizar_hebbiano(id_ant, id_prox, taxa=0.03, decaimento=0.999)
            self.rede_fractal.atualizar_hebbiano_fractal(id_ant, id_prox, taxa=0.02, decaimento=0.999)

        self.rede_local.salvar("rede_local.bin")
        self.rede_fractal.salvar("rede_fractal.bin")
        print("✅ Redes neurais (local + fractal) treinadas e salvas.")
        print("✅ Treino concluído.")

    def criar_snapshot_cognitivo(self, contexto):
        vetores_carregados = self.memoria_hd.carregar_on_demand(contexto)
        if vetores_carregados:
            vetores_ctx = [vetores_carregados[t] for t in contexto if t in vetores_carregados]
        else:
            vetores_ctx = [self.token_para_vetor[t] for t in contexto if t in self.token_para_vetor]
        if not vetores_ctx:
            zero = [0.0] * self.dim
            return zero, zero, 0.0
        snapshot_inicial = self.atencao.ruminar(vetores_ctx)
        snapshot_distilado, tensao = self.funil.distilar_pensamento(snapshot_inicial)
        return snapshot_inicial, snapshot_distilado, tensao

    def atualizar_intencao(self, token_aceito, fator_inercia=0.05):
        if token_aceito not in self.token_para_vetor:
            return
        v_token = self.token_para_vetor[token_aceito]
        if self.intencao_atual is None:
            self.intencao_atual = v_token.copy()
            return
        impulso = [fator_inercia * x for x in v_token]
        self.intencao_atual = self.disco.adicao_mobius(self.intencao_atual, impulso)
        self.intencao_atual = self.disco.projetar(self.intencao_atual)

    def coerencia_global(self, snapshot_inicial, limiar_max=1.2):
        if self.intencao_atual is None:
            return
        dist = self.disco.distancia(snapshot_inicial, self.intencao_atual)
        if dist > limiar_max:
            fator_restauracao = 0.15
            direcao = self.disco.adicao_mobius([-x for x in self.intencao_atual], snapshot_inicial)
            impulso_restauracao = [fator_restauracao * x for x in direcao]
            self.intencao_atual = self.disco.adicao_mobius(self.intencao_atual, impulso_restauracao)
            self.intencao_atual = self.disco.projetar(self.intencao_atual)

    def reacao_ambiental(self, tokens_gerados, feedback_sinal):
        if feedback_sinal == 0.0: return
        fator = 0.01 * feedback_sinal
        for t in tokens_gerados:
            if t in self.token_para_vetor:
                self.token_para_vetor[t] = self.disco.adicao_mobius(self.token_para_vetor[t], [fator] * self.dim)

# =============================================================================
# 9. MAPA NEURONAL E LIMIAR ESPACIAL
# =============================================================================
MAPA_NEURONAL = {",": 0.35, ".": 1.25, "!": 1.50, "?": 1.30}
def calcular_limiar_espacial(texto_treino):
    tokens = TOKENIZER_REGEX.findall(texto_treino.lower())
    limiar_acumulado = sum(MAPA_NEURONAL.get(token, 0.0) for token in tokens)
    qtd_pontos = sum(1 for token in tokens if token in MAPA_NEURONAL)
    return max(1.0, limiar_acumulado / max(1, qtd_pontos))

# =============================================================================
# 10. MOTOR DE DECODIFICAÇÃO COM ATIVAÇÃO FRACTAL
# =============================================================================
def arquinet_hybrid_decode(
    target, drafter, prompt, limiar_corte_base,
    max_tokens=35, block_size=3, feedback_sinal=1.0
):
    target.nucleo_estado.interpretar_externo(prompt)
    bias_interno = target.nucleo_estado.exportar_bias_decodificacao()
    tokens = target.tokenizar(prompt)
    if not tokens: tokens = [target.tokens_lista[0] if target.tokens_lista else '<s>']
    accepted = []
    stats = {'blocos': 0, 'aceitos': 0, 'rejeitados': 0, 'motivo_parada': 'max_tokens', 'relatorio_evolucao': {}}
    target.intencao_atual = None
    target.snapshot_anterior = None
    snapshot_inicial_global = None
    token_para_id = target.token_para_id
    vocab_size = len(target.tokens_lista)
    escalas_fractal = [0.3, 0.8, 2.0]

    while len(accepted) < max_tokens:
        contexto = tokens[-6:]
        snapshot_inicial, snapshot_distilado, tensao_funil = target.criar_snapshot_cognitivo(contexto)
        relatorio_ev = target.funil.comparar_evolucao(snapshot_inicial, snapshot_distilado)
        stats['relatorio_evolucao'] = relatorio_ev
        if snapshot_inicial_global is None:
            snapshot_inicial_global = snapshot_inicial
            target.intencao_atual = snapshot_distilado.copy()
            target.intencao_inicial = snapshot_distilado.copy()
        snapshot_para_rotear = target.disco.adicao_mobius(snapshot_distilado, [0.3 * x for x in target.intencao_atual])
        snapshot_para_rotear = target.disco.projetar(snapshot_para_rotear)
        rota, perfil = target.roteador.rotear(snapshot_para_rotear)
        temp_ajustada = max(0.2, perfil['temperatura'] + bias_interno['mod_temp'])
        tensao_ajustada = max(0.5, perfil['tensao'] + bias_interno['mod_tensao'])
        energia, deve_parar_energia = target.monitor_energia.atualizar(target.disco, snapshot_distilado)
        if deve_parar_energia:
            stats['motivo_parada'] = 'colapso_energetico'
            break

        # Anchor geométrico
        candidatos_anchor = []
        for token in target.tokens_lista[:5000]:
            if token in target.token_para_vetor and token not in accepted[-3:]:
                d = target.disco.distancia(target.intencao_atual, target.token_para_vetor[token])
                candidatos_anchor.append((d, token))
        candidatos_anchor.sort(key=lambda x: x[0])
        if len(candidatos_anchor) >= 3:
            idx = random.choices([0, 1, 2], weights=[0.6, 0.25, 0.15])[0]
            anchor = candidatos_anchor[idx][1]
        elif candidatos_anchor:
            anchor = candidatos_anchor[0][1]
        else:
            anchor = target.tokens_lista[0] if target.tokens_lista else '<s>'

        # Draft: combina top-K local e fractal
        id_anchor = token_para_id.get(anchor, 0)
        vizinhos_local = target.rede_local._top_k_neighbors(id_anchor, k=8) if target.rede_local else []
        vizinhos_fractal = target.rede_fractal._top_k_neighbors_escala(id_anchor, distancia=1, k=8) if target.rede_fractal else []
        todos_vizinhos = list(set([(v_id, peso) for v_id, peso in vizinhos_local + vizinhos_fractal]))
        random.shuffle(todos_vizinhos)
        candidatos_draft = [target.tokens_lista[v_id] for v_id, _ in todos_vizinhos if target.tokens_lista[v_id] not in accepted]
        if len(candidatos_draft) < block_size:
            extras = random.sample([t for t in target.tokens_lista if t not in accepted], min(block_size - len(candidatos_draft), len(target.tokens_lista)))
            candidatos_draft.extend(extras)
        draft = candidatos_draft[:block_size]

        accepted_prefix = []
        curr_anchor = anchor
        curr_penultimate = contexto[-2] if len(contexto) >= 2 else None
        parada_forcada = False
        candidatos_bloco_info = []

        for t_cand in draft:
            if t_cand in accepted or t_cand == curr_anchor:
                continue
            v_cand = target.token_para_vetor.get(t_cand, [0.0]*target.dim)
            repeticoes_recentes = accepted[-6:].count(t_cand) if accepted else 0
            penalidade_repeticao = 0.2 ** repeticoes_recentes if repeticoes_recentes > 0 else 1.0
            dist_hip = target.disco.distancia(target.intencao_atual, v_cand)

            p_nao_linear = fractal_activation(dist_hip, temp_ajustada, escalas_fractal)

            id_cur = token_para_id.get(curr_anchor, 0)
            id_cand = token_para_id.get(t_cand, 0)
            score_local = 0.0
            score_fractal = 0.0
            if target.rede_local is not None and id_cur < vocab_size and id_cand < vocab_size:
                score_local = target.rede_local.score_relacional(id_cur, id_cand, target.intencao_atual)
            if target.rede_fractal is not None and id_cur < vocab_size and id_cand < vocab_size:
                score_fractal = target.rede_fractal.score_relacional(id_cur, id_cand, target.intencao_atual)
            score_combinado = 0.6 * score_local + 0.4 * score_fractal
            p_neural = max(0.001, min(1.0, score_combinado + 0.1))

            logits_candidato = {t_cand: p_neural}
            logits_filtrados = target.roteador.filtrar_logits(target.intencao_atual, logits_candidato, target.token_para_vetor)
            p_neural_filt = logits_filtrados.get(t_cand, 0.001)

            p_val = (p_nao_linear ** 0.5) * (p_neural_filt ** 0.5) * penalidade_repeticao
            candidatos_bloco_info.append((t_cand, p_val, dist_hip))
            if p_val > 1e-5:
                accepted_prefix.append(t_cand)
                curr_penultimate = curr_anchor
                curr_anchor = t_cand
                if t_cand in [".", "!", "?"]:
                    parada_forcada = True
                    stats['motivo_parada'] = 'pontuacao_finalizadora'
                    break
            else:
                break

        if accepted_prefix:
            for i in range(len(accepted_prefix) - 1):
                id_ant = token_para_id.get(accepted_prefix[i], 0)
                id_prox = token_para_id.get(accepted_prefix[i+1], 0)
                if id_ant < vocab_size and id_prox < vocab_size:
                    target.rede_local.atualizar_hebbiano(id_ant, id_prox, taxa=0.03, decaimento=0.999)
                    target.rede_fractal.atualizar_hebbiano_fractal(id_ant, id_prox, taxa=0.02, decaimento=0.999)
            id_anc = token_para_id.get(anchor, 0)
            id_primeiro = token_para_id.get(accepted_prefix[0], 0)
            if id_anc < vocab_size and id_primeiro < vocab_size:
                target.rede_local.atualizar_hebbiano(id_anc, id_primeiro, taxa=0.03, decaimento=0.999)
                target.rede_fractal.atualizar_hebbiano_fractal(id_anc, id_primeiro, taxa=0.02, decaimento=0.999)
            for tok in accepted_prefix:
                target.atualizar_intencao(tok, fator_inercia=0.05)
            if snapshot_inicial_global is not None:
                target.coerencia_global(snapshot_inicial_global, limiar_max=1.2)

        stats['blocos'] += 1
        stats['aceitos'] += len(accepted_prefix)
        stats['rejeitados'] += (len(draft) - len(accepted_prefix))

        if accepted_prefix:
            accepted.extend(accepted_prefix)
            tokens.extend(accepted_prefix)
        else:
            fallback = random.choice([t for t in target.tokens_lista if t not in accepted])
            accepted.append(fallback)
            tokens.append(fallback)

        if parada_forcada:
            break

    if feedback_sinal != 0.0:
        target.reacao_ambiental(accepted, feedback_sinal)
    if target.rede_local is not None:
        target.rede_local.salvar("rede_local.bin")
    if target.rede_fractal is not None:
        target.rede_fractal.salvar("rede_fractal.bin")

    texto_saida = " ".join(accepted[:max_tokens])
    for p in [".", ",", "!", "?", ";", ":"]:
        texto_saida = texto_saida.replace(f" {p}", p)

    meta_info = {
        "modo_tpthink": rota,
        "estado_interno": {
            "emocao": target.nucleo_estado.emocao_percebida,
            "objetivo": target.nucleo_estado.objetivo,
            "assunto": target.nucleo_estado.assunto,
            "energia": target.nucleo_estado.energia_conversa
        },
        "estilo_plc": perfil.get('estilo', 'formal'),
        "tensao_final": tensao_ajustada,
        "temperatura_final": temp_ajustada,
        "energia_minkowski": energia,
        "stats": stats
    }
    return texto_saida, meta_info

def random_params():
    return 35, random.choice([2, 3, 4]), round(random.uniform(0.8, 1.2), 2)

# =============================================================================
# 11. DATASETS
# =============================================================================
dataset_filosofia = """
A relação entre Dioniso e a vontade de potência no pensamento de Friedrich Nietzsche não é uma simples associação metafórica, mas o eixo central de toda a sua filosofia madura. Dioniso não é apenas o deus do vinho, do êxtase e da desmedida; ele é a personificação da própria vida em seu fluxo mais intenso, criador e destruidor. A vontade de potência, por sua vez, é o princípio ontológico fundamental que Nietzsche opõe a todas as metafísicas estáticas e finalistas, desde Platão até Schopenhauer. Compreender a relação entre essas duas forças exige mergulhar na crítica nietzschiana à razão, à moralidade e à própria noção de "ser" como algo fixo.

Em primeiro lugar, é preciso dissolver o equívoco comum que reduz Dioniso a um símbolo de irracionalismo ou de embriaguez caótica. Para Nietzsche, Dioniso é a afirmação incondicional da vida em sua totalidade, incluindo o sofrimento, a destruição e o devir. A vontade de potência não é uma vontade de poder no sentido político ou psicológico vulgar; ela é a essência de toda realidade como uma força que se auto-supera constantemente, que cria e aniquila, que se expande e se contrai em ciclos. Dioniso, no nascimento da tragédia, aparece como o princípio que despedaça a individualidade e permite o contato com a unidade primordial da existência. Essa unidade não é uma substância ou um fundamento, mas o próprio jogo de forças que constitui o mundo. A vontade de potência é exatamente esse jogo: cada ser é um centro de forças que busca expandir sua influência, mas que ao mesmo tempo depende do conflito e da resistência para existir.
"""

dataset_conversa = """
Olá! Como você está hoje? Espero que bem.
Oi! Tudo bem por aí? Estou aqui para conversar.
E aí, beleza? O que você está achando do dia?
Olá, como você está? Espero que bem. A vida é cheia de perguntas, não é?
Às vezes pensamos sobre o sentido das coisas. O que é a felicidade?
Acho que a felicidade está nas pequenas coisas: um sorriso, um abraço, uma conversa.
É bom conversar com alguém. As pessoas gostam de ser ouvidas.
Quando falamos, compartilhamos um pouco de nós mesmos.
A tecnologia pode nos conectar, mas nada substitui o contato humano.
Uma conversa gentil pode aquecer o coração.
Vamos conversar sobre o que você quiser. Estou aqui para ouvir.

"""

dataset_poesia = """
O vento leva as folhas secas.
A lua brilha no céu escuro.
O mar dança com a areia.
A vida é um sopro, um instante.
O tempo passa devagar, mas nunca para.
As estrelas são olhos que nos observam.
A noite guarda segredos que o dia revela.
"""

dataset_tecnico = """
Um sistema de IA é composto por camadas de processamento.
Cada camada transforma a entrada em uma representação mais abstrata.
A geometria hiperbólica pode modelar hierarquias conceituais.
A atenção é um mecanismo que permite ao modelo focar em partes relevantes da entrada.
O aprendizado Hebbiano reforça conexões que ocorrem frequentemente.
Sistemas conversacionais precisam de fluência e coerência.
"""

# =============================================================================
# 12. EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🌿 TGP3 com Arquitetura Fractal (Local + Fractal + Ativação Fractal)")
    print("=" * 70)

    # ===== CONTROLE DE TREINO =====
    TRAIN_MODE = True      # True = treina se necessário; False = só carrega e executa
    FORCE_RE_TRAIN = False # True = força re-treino mesmo se disco existir

    dim = 64
    target = AgenteTGP_13(dim=dim)

    dataset_completo = dataset_filosofia + "\n" + (dataset_conversa * 3) + "\n" + dataset_poesia + "\n" + dataset_tecnico

    # Chama o treino com os parâmetros de controle
    target.treinar(dataset_completo, epocas=12, train_mode=TRAIN_MODE, force=FORCE_RE_TRAIN)

    # Se não há tokens e o treino foi pulado, encerra
    if not target.tokens_lista:
        print("❌ Nenhum disco carregado e treino desativado. Encerrando.")
        exit()

    prompts_teste = [
        "Oi, como você está?",
        "O que é felicidade para você?",
        "Explique a teoria da relatividade.",
        "pode ser compreendida como uma resposta à tradição metafísica que separa o 'ser' do 'devir'?",
        "Qual a implicação prática do perspectivismo para a ciência?"
    ]

    print("\n" + "=" * 70)
    print("⚡ INFERÊNCIA COM REDE FRACTAL")
    print("=" * 70 + "\n")

    for p in prompts_teste:
        t0 = time.time()
        max_tok, blk, fbk = random_params()
        print(f"📌 Prompt: {p}")
        texto_saida, meta = arquinet_hybrid_decode(
            target, None, p, limiar_corte_base=1.0,
            max_tokens=max_tok, block_size=blk, feedback_sinal=fbk
        )
        tempo = time.time() - t0

        print(f"   Rota: {meta['modo_tpthink']} | Temp: {meta['temperatura_final']:.2f} | Tensão: {meta['tensao_final']:.2f}")
        print(f"   Energia Minkowski: {meta.get('energia_minkowski', 0.0):.4f}")
        print(f"   Status Funil: {meta['stats']['relatorio_evolucao']['status']} | Tensão Cognitiva: {meta['stats']['relatorio_evolucao']['tensao_cognitiva']:.2f}")
        print(f"   Saída: '{texto_saida}'")
        print(f"   Tokens: {meta['stats']['aceitos']} | Parada: {meta['stats']['motivo_parada']} | Tempo: {tempo:.4f}s\n")
