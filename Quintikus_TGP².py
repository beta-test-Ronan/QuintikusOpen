#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TGP-2.5.13 + DSpark 2.0 (Modo Híbrido: Snapshot Cognitivo em Cache)
- Inteligência: Preserva Geometria Hiperbólica no Disco de Poincaré
- Velocidade: Validação em Lote de Baixa Latência (~0.05s)
===============================================================================
"""

import math
import os
import pickle
import random
import re
import time
import unicodedata
from collections import defaultdict, deque
from typing import Dict, List, Tuple


# =============================================================================
# BASE GEOMÉTRICA HIPERBÓLICA (POINCARÉ DISK)
# =============================================================================
class DiscoPoincare:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.raio = 0.985
        self.eps = 1e-10

    def norma(self, v: List[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def projetar(self, v: List[float]) -> List[float]:
        n = self.norma(v)
        if n > self.raio:
            fator = self.raio / n
            return [x * fator for x in v]
        return v

    def adicao_mobius(self, x: List[float], y: List[float]) -> List[float]:
        nx2 = min(sum(a * a for a in x), 0.98)
        ny2 = min(sum(b * b for b in y), 0.98)
        xy = min(max(sum(a * b for a, b in zip(x, y)), -0.98), 0.98)

        den = 1 + 2 * xy + nx2 * ny2 + self.eps
        num_fator = 1 + 2 * xy + ny2
        fator_y = 1 - nx2

        res = [(num_fator * a + fator_y * b) / den for a, b in zip(x, y)]
        return self.projetar(res)

    def distancia(self, x: List[float], y: List[float]) -> float:
        x_p = self.projetar(x)
        y_p = self.projetar(y)
        norma_x = min(self.norma(x_p), self.raio)
        norma_y = min(self.norma(y_p), self.raio)
        diff_sq = sum((a - b) ** 2 for a, b in zip(x_p, y_p))

        num = 2 * diff_sq
        den = (1 - norma_x**2) * (1 - norma_y**2) + self.eps
        val = max(1.0, 1.0 + num / den)
        return math.acosh(min(val, 1e6))


class MemoriaLinear:
    def __init__(self):
        self.bigramas = defaultdict(lambda: defaultdict(float))
        self.trigramas = defaultdict(lambda: defaultdict(float))

    def registrar_fluxo(self, tokens: List[str]):
        if len(tokens) < 2: return
        for i in range(len(tokens) - 1):
            self.bigramas[tokens[i]][tokens[i + 1]] += 1.0
        for i in range(len(tokens) - 2):
            self.trigramas[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1.0

        for t1, contagens in self.bigramas.items():
            total = sum(contagens.values())
            if total > 0:
                for t2 in contagens: contagens[t2] /= total

        for chave, contagens in self.trigramas.items():
            total = sum(contagens.values())
            if total > 0:
                for t3 in contagens: contagens[t3] /= total

    def prob_linear(self, t_atual: str, t_candidato: str, t_penultimo: str = None) -> float:
        p_bigrama = self.bigramas.get(t_atual, {}).get(t_candidato, 1e-4)
        p_trigrama = 0.0
        if t_penultimo and (t_penultimo, t_atual) in self.trigramas:
            p_trigrama = self.trigramas[(t_penultimo, t_atual)].get(t_candidato, 0.0)
        return 0.4 * p_bigrama + 0.6 * p_trigrama


class AtencaoCognitiva:
    def __init__(self, disco: DiscoPoincare):
        self.disco = disco

    def ruminar(self, vetores_contexto: List[List[float]]) -> List[float]:
        n = len(vetores_contexto)
        if n == 0: return [0.0] * self.disco.dim
        if n == 1: return vetores_contexto[0]

        pensamento = vetores_contexto[-1]
        for _ in range(2):
            pesos = [-self.disco.distancia(pensamento, v) for v in vetores_contexto]
            max_p = max(pesos)
            exp_pesos = [math.exp(p - max_p) for p in pesos]
            soma_exp = sum(exp_pesos)
            prob_atencao = [e / soma_exp for e in exp_pesos]

            novo_pensamento = [0.0] * self.disco.dim
            for i, v in enumerate(vetores_contexto):
                v_pesado = [x * prob_atencao[i] for x in v]
                novo_pensamento = self.disco.adicao_mobius(novo_pensamento, v_pesado)
            pensamento = novo_pensamento

        return pensamento

    def tensao_logica(self, vetores_contexto: List[List[float]]) -> List[float]:
        n = len(vetores_contexto)
        if n < 3: return [0.0] * self.disco.dim
        meio = n // 2
        
        polo_suj = [0.0] * self.disco.dim
        for v in vetores_contexto[:meio]:
            polo_suj = [a + b for a, b in zip(polo_suj, v)]
        polo_suj = self.disco.projetar([x / meio for x in polo_suj])

        polo_pred = [0.0] * self.disco.dim
        for v in vetores_contexto[meio:]:
            polo_pred = [a + b for a, b in zip(polo_pred, v)]
        polo_pred = self.disco.projetar([x / (n - meio) for x in polo_pred])

        return self.disco.adicao_mobius([-x for x in polo_suj], polo_pred)


# =============================================================================
# AGENTE TARGET COM SUPORTE A SNAPSHOT (CACHE COGNITIVO)
# =============================================================================
class AgenteTGP_13:
    def __init__(self, dim: int = 128, arquivo_salvamento: str = "agente_tgp13.pkl"):
        self.dim = dim
        self.disco = DiscoPoincare(dim)
        self.memoria = MemoriaLinear()
        self.atencao = AtencaoCognitiva(self.disco)
        self.token_para_vetor: Dict[str, List[float]] = {}
        self.tokens_lista: List[str] = []
        self.arquivo_salvamento = arquivo_salvamento

    def tokenizar(self, texto: str) -> List[str]:
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acentos = ''.join(c for c in texto_nfkd if unicodedata.category(c) != 'Mn')
        return [t for t in re.findall(r'[a-z0-9]+|[.,!?;:]+', texto_sem_acentos) if t.strip()]

    def _registrar_token(self, token: str):
        if token not in self.token_para_vetor:
            self.token_para_vetor[token] = [(random.random() - 0.5) * 0.08 for _ in range(self.dim)]
            self.tokens_lista.append(token)

    def treinar(self, texto_base: str, epocas: int = 35):
        tokens = self.tokenizar(texto_base)
        if not tokens: return
        self.memoria.registrar_fluxo(tokens)
        for t in tokens: self._registrar_token(t)

        num_tokens = len(self.tokens_lista)
        for _ in range(epocas):
            for i in range(len(tokens) - 1):
                t_atual, t_prox = tokens[i], tokens[i + 1]
                v_atual = self.token_para_vetor[t_atual]
                v_prox = self.token_para_vetor[t_prox]

                diff_atracao = [(b - a) * 0.04 for a, b in zip(v_atual, v_prox)]
                v_atual = self.disco.adicao_mobius(v_atual, diff_atracao)

                t_neg = self.tokens_lista[random.randint(0, num_tokens - 1)]
                if t_neg not in (t_atual, t_prox):
                    v_neg = self.token_para_vetor[t_neg]
                    diff_repulsao = [-(b - a) * 0.02 for a, b in zip(v_atual, v_neg)]
                    v_atual = self.disco.adicao_mobius(v_atual, diff_repulsao)

                self.token_para_vetor[t_atual] = v_atual

        for t in self.tokens_lista:
            self.token_para_vetor[t] = self.disco.projetar(self.token_para_vetor[t])

    def criar_snapshot_cognitivo(self, contexto: List[str]) -> List[float]:
        """Calcula o estado rumado uma única vez por bloco (Otimização Chave)"""
        vetores_ctx = [self.token_para_vetor[t] for t in contexto if t in self.token_para_vetor]
        if not vetores_ctx:
            return [0.0] * self.dim

        vetor_intencao = self.atencao.ruminar(vetores_ctx)
        vetor_tensao = self.atencao.tensao_logica(vetores_ctx)
        return self.disco.adicao_mobius(vetor_intencao, [x * 0.25 for x in vetor_tensao])

    def prob_target_fast(self, snapshot_vector: List[float], t_atual: str, token_cand: str, t_penultimo: str = None) -> float:
        """Validação ultrarrápida usando o Snapshot pré-calculado"""
        if token_cand not in self.token_para_vetor: return 1e-12
        v_cand = self.token_para_vetor[token_cand]
        
        dist_hip = self.disco.distancia(snapshot_vector, v_cand)
        p_nao_linear = math.exp(-dist_hip)
        p_linear = self.memoria.prob_linear(t_atual, token_cand, t_penultimo)

        return (p_nao_linear ** 0.6) * (p_linear ** 1.4)


# =============================================================================
# DRAFTER CACHE TEMPORÁRIO (ULTRA-FAST LATTICE)
# =============================================================================
class DrafterCacheTemporario:
    def __init__(self, target: AgenteTGP_13):
        self.target = target
        self.vocab = target.tokens_lista
        self.transition = defaultdict(lambda: defaultdict(lambda: 0.01))
        self._sincronizar_memoria()

    def _sincronizar_memoria(self):
        for t1, conexoes in self.target.memoria.bigramas.items():
            for t2, p in conexoes.items():
                self.transition[t1][t2] = max(p, 0.01)

    def draft_block(self, anchor: str, block_size: int = 6) -> List[str]:
        draft = []
        prev = anchor
        for _ in range(block_size):
            conexoes = self.transition[prev]
            cand, weights = zip(*conexoes.items()) if conexoes else (self.vocab, [1.0]*len(self.vocab))
            soma = sum(weights)
            probs = [w / soma for w in weights]
            
            # Amostrage rápida
            r = random.random()
            acum = 0.0
            chosen = cand[-1]
            for t, p in zip(cand, probs):
                acum += p
                if r <= acum:
                    chosen = t
                    break
            draft.append(chosen)
            prev = chosen
        return draft


# =============================================================================
# EXECUÇÃO HÍBRIDA (MEIO TERMO: RÁPIDO & INTELIGENTE)
# =============================================================================
def dspark_tgp13_hybrid_decode(
    target: AgenteTGP_13,
    drafter: DrafterCacheTemporario,
    prompt: str,
    max_tokens: int = 24,
    block_size: int = 6
) -> Tuple[List[str], Dict]:
    
    tokens = target.tokenizar(prompt)
    if not tokens: tokens = ['<s>']
    accepted = []

    stats = {'blocos': 0, 'aceitos': 0, 'rejeitados': 0}

    while len(accepted) < max_tokens:
        anchor = tokens[-1]
        contexto = tokens[-8:] # Janela local de contexto

        # 1. SNAPSHOT COGNITIVO (1 único cálculo geométrico por lote)
        snapshot = target.criar_snapshot_cognitivo(contexto)

        # 2. DRAFT EM MICROSEGUNDOS
        draft = drafter.draft_block(anchor, block_size=block_size)

        # 3. VALIDAÇÃO EM LOTE USANDO O SNAPSHOT
        accepted_prefix = []
        curr_anchor = anchor
        curr_penultimate = contexto[-2] if len(contexto) >= 2 else None

        for t_cand in draft:
            p_val = target.prob_target_fast(snapshot, curr_anchor, t_cand, curr_penultimate)
            
            # Limiar de aceitação
            if p_val > 1e-4:
                accepted_prefix.append(t_cand)
                curr_penultimate = curr_anchor
                curr_anchor = t_cand
            else:
                break

        stats['blocos'] += 1
        stats['aceitos'] += len(accepted_prefix)
        stats['rejeitados'] += (len(draft) - len(accepted_prefix))

        if accepted_prefix:
            accepted.extend(accepted_prefix)
            tokens.extend(accepted_prefix)
        else:
            # Fallback local se o primeiro token do rascunho falhar
            fallback = target.tokens_lista[random.randint(0, len(target.tokens_lista) - 1)]
            accepted.append(fallback)
            tokens.append(fallback)

    return accepted[:max_tokens], stats


if __name__ == "__main__":
    dataset_treino = """
Geometric intelligence maps concepts onto the Poincaré disk.
The dspark system accelerates speculative batch token generation.
Hyperbolic attraction and repulsion adjust vectors in 128-dimensional space.
The Telica JSON acts as a local syntactic cohesion manifold.
The TGP agent combines linear memory with cognitive geo-attention, without backpropagation.
Dual coexistence balances linear probability and geodesic distance.
    """

    print("🚀 1. Carregando Base TGP-13...")
    target = AgenteTGP_13(dim=128)
    target.treinar(dataset_treino, epocas=10)

    print("⚡ 2. Inicializando Drafter Cache Temporario...")
    drafter = DrafterCacheTemporario(target)

    prompt = "geometric intelligence and linear probability"
    print(f"\n💬 Estímulo (Input): '{prompt}'")

    t_inicio = time.time()
    gerado, stats = dspark_tgp13_hybrid_decode(target, drafter, prompt, max_tokens=20, block_size=6)
    t_fim = time.time() - t_inicio

    print(f"\n📄 Resultado Final: {' '.join(gerado)}")
    print(f"⏱️ Tempo Decorrido: {t_fim:.4f}s ({len(gerado)/t_fim:.1f} tokens/s)")
    print(f"📊 Estatísticas: Blocos = {stats['blocos']} | Aceitos = {stats['aceitos']} | Rejeitados = {stats['rejeitados']}")
