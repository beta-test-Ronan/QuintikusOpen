#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TGP ARQUINET v2.0: TGP-13 + DSpark 2.0 + PLC (Style vs. Semantic Engine)
===============================================================================
- Semântica: Trajetória no Disco de Poincaré + Atenção Cognitiva por Ruminação
- Estilo: Modulação de Estado (Temperatura/Tensão) via Bússola PLC
- Especulação: Drafter Cache Temporário com Aceleração DSpark
- Dinâmica: Loop de Ação e Reação (Retroalimentação de Sinal sem Backprop)
===============================================================================
"""

import math
import random
import re
import time
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# =============================================================================
# 1. BASE GEOMÉTRICA HIPERBÓLICA (POINCARÉ DISK CORE)
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


# =============================================================================
# 2. MEMÓRIA LINEAR E ATENÇÃO COGNITIVA
# =============================================================================
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
# 3. BÚSSOLA PLC & HOMEOSTASE DINÂMICA (CAMADA DE ESTILO)
# =============================================================================
class BussolaEstiloPLC:
    def __init__(self):
        self.estilos = {
            "informal": {"temperatura": 1.5, "tensao": 0.6},
            "formal": {"temperatura": 0.4, "tensao": 1.8},
            "instrucao": {"temperatura": 0.7, "tensao": 1.2},
            "expansivo": {"temperatura": 1.8, "tensao": 0.5}
        }
        self.exemplos: List[Tuple[str, str, set]] = []
        self._carregar_base()

    def _carregar_base(self):
        base = [
            ("e ai beleza suave tranquilo", "informal"),
            ("prezados senhores atenciosamente", "formal"),
            ("passo a passo como fazer instrucao", "instrucao"),
            ("geometric intelligence space poincare", "expansivo")
        ]
        for frase, tipo in base:
            tokens = set(re.findall(r'\w+', frase.lower()))
            self.exemplos.append((frase, tipo, tokens))

    def extrair_estilo(self, texto: str) -> Tuple[str, Dict[str, float]]:
        tokens_in = set(re.findall(r'\w+', texto.lower()))
        if not tokens_in:
            return "informal", self.estilos["informal"]

        melhor_sim = -1.0
        estilo_det = "expansivo"
        for _, tipo, tokens_ex in self.exemplos:
            inter = tokens_in & tokens_ex
            uniao = tokens_in | tokens_ex
            sim = len(inter) / max(len(uniao), 1)
            if sim > melhor_sim and sim > 0:
                melhor_sim = sim
                estilo_det = tipo

        return estilo_det, self.estilos.get(estilo_det, self.estilos["expansivo"])


class HomeostaseEspacial:
    def __init__(self, tensao_base: float = 1.0, temperatura_base: float = 1.0):
        self.tensao = tensao_base
        self.temperatura = temperatura_base

    def ajustar(self, candidatos_avaliados: List[Tuple[str, float, float]]):
        if not candidatos_avaliados: return

        bons = [c for c in candidatos_avaliados if c[1] > 1e-3]
        ruins = [c for c in candidatos_avaliados if c[1] <= 1e-3]

        if len(bons) >= 1 and len(ruins) <= 2:
            self.tensao = max(0.4, self.tensao * 0.88)
        elif len(ruins) > len(bons):
            self.tensao = min(2.5, self.tensao * 1.20)

        distancias = [c[2] for c in candidatos_avaliados]
        media_dist = sum(distancias) / len(distancias)
        variancia = sum((d - media_dist) ** 2 for d in distancias) / len(distancias)

        if variancia > 0.45:
            self.temperatura = max(0.2, self.temperatura * 0.82)
        elif variancia < 0.08:
            self.temperatura = min(2.2, self.temperatura * 1.25)


# =============================================================================
# 4. AGENTE TARGET TGP-13
# =============================================================================
class AgenteTGP_13:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.disco = DiscoPoincare(dim)
        self.memoria = MemoriaLinear()
        self.atencao = AtencaoCognitiva(self.disco)
        self.plc = BussolaEstiloPLC()
        self.token_para_vetor: Dict[str, List[float]] = {}
        self.tokens_lista: List[str] = []

    def tokenizar(self, texto: str) -> List[str]:
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acentos = ''.join(c for c in texto_nfkd if unicodedata.category(c) != 'Mn')
        return [t for t in re.findall(r'[a-z0-9]+|[.,!?;:]', texto_sem_acentos) if t.strip()]

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
        vetores_ctx = [self.token_para_vetor[t] for t in contexto if t in self.token_para_vetor]
        if not vetores_ctx:
            return [0.0] * self.dim

        vetor_intencao = self.atencao.ruminar(vetores_ctx)
        vetor_tensao = self.atencao.tensao_logica(vetores_ctx)
        return self.disco.adicao_mobius(vetor_intencao, [x * 0.25 for x in vetor_tensao])

    def avaliacao_completa(self, snapshot_vector: List[float], t_atual: str, token_cand: str, t_penultimo: str = None, temperatura: float = 1.0, tensao: float = 1.0) -> Tuple[float, float]:
        if token_cand not in self.token_para_vetor: return 1e-12, 10.0
        v_cand = self.token_para_vetor[token_cand]
        
        dist_hip = self.disco.distancia(snapshot_vector, v_cand)
        p_nao_linear = math.exp(-dist_hip / max(0.05, temperatura))
        p_linear = self.memoria.prob_linear(t_atual, token_cand, t_penultimo)

        prob_final = (p_nao_linear ** 0.6) * (p_linear ** (1.4 * tensao))
        return prob_final, dist_hip

    # LOOP DE REAÇÃO (Ajuste On-the-fly sem backprop)
    def reacao_ambiental(self, tokens_gerados: List[str], feedback_sinal: float):
        if feedback_sinal == 0.0: return
        fator = 0.025 * feedback_sinal
        for t in tokens_gerados:
            if t in self.token_para_vetor:
                v = self.token_para_vetor[t]
                self.token_para_vetor[t] = self.disco.adicao_mobius(v, [fator] * self.dim)


# =============================================================================
# 5. DRAFTER CACHE TEMPORÁRIO
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

    def draft_block(self, anchor: str, block_size: int = 4) -> List[str]:
        draft = []
        prev = anchor
        for _ in range(block_size):
            conexoes = self.transition[prev]
            cand, weights = zip(*conexoes.items()) if conexoes else (self.vocab, [1.0]*len(self.vocab))
            soma = sum(weights)
            probs = [w / soma for w in weights]
            
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


MAPA_NEURONAL = {",": 0.35, ".": 1.25, "!": 1.50, "?": 1.30}

def calcular_limiar_espacial(texto_treino: str) -> float:
    tokens = re.findall(r'[a-z0-9]+|[.,!?;:]', texto_treino.lower())
    limiar_acumulado = 0.0
    qtd_pontos = 0
    for token in tokens:
        if token in MAPA_NEURONAL:
            limiar_acumulado += MAPA_NEURONAL[token]
            qtd_pontos += 1
    return limiar_acumulado / max(1, qtd_pontos)


# =============================================================================
# 6. MOTOR DE DECODIFICAÇÃO UNIFICADO (DSPARK + PLC + HOMEOSTASE)
# =============================================================================
def arquinet_hybrid_decode(
    target: AgenteTGP_13,
    drafter: DrafterCacheTemporario,
    prompt: str,
    limiar_corte_base: float,
    max_tokens: int = 24,
    block_size: int = 4,
    feedback_sinal: float = 0.0
) -> Tuple[str, Dict]:
    
    tokens = target.tokenizar(prompt)
    if not tokens: tokens = ['<s>']
    accepted = []

    # Extrai o Estilo base do PLC e configura valores iniciais
    nome_estilo, perfil_estilo = target.plc.extrair_estilo(prompt)
    homeostase = HomeostaseEspacial(
        tensao_base=perfil_estilo["tensao"],
        temperatura_base=perfil_estilo["temperatura"]
    )

    stats = {'blocos': 0, 'aceitos': 0, 'rejeitados': 0}
    pos_x, pos_y = 0.0, 0.0
    angulo = 0.0
    energia_acumulada = 0.0

    while len(accepted) < max_tokens:
        anchor = tokens[-1]
        contexto = tokens[-8:]

        snapshot = target.criar_snapshot_cognitivo(contexto)
        draft = drafter.draft_block(anchor, block_size=block_size)

        accepted_prefix = []
        curr_anchor = anchor
        curr_penultimate = contexto[-2] if len(contexto) >= 2 else None

        parada_forcada = False
        candidatos_bloco_info = []

        for t_cand in draft:
            p_val, dist_hip = target.avaliacao_completa(
                snapshot, curr_anchor, t_cand, curr_penultimate,
                temperatura=homeostase.temperatura,
                tensao=homeostase.tensao
            )
            candidatos_bloco_info.append((t_cand, p_val, dist_hip))

            peso_efeito = MAPA_NEURONAL.get(t_cand, 0.0) * homeostase.tensao
            if t_cand in MAPA_NEURONAL:
                energia_acumulada += peso_efeito
                angulo += peso_efeito * (math.pi / 2.0)
            else:
                deslocamento = len(t_cand) * 0.5
                pos_x += deslocamento * math.cos(angulo)
                pos_y += deslocamento * math.sin(angulo)

            if p_val > 1e-4:
                accepted_prefix.append(t_cand)
                curr_penultimate = curr_anchor
                curr_anchor = t_cand

                limiar_dinamico = limiar_corte_base / homeostase.tensao
                if t_cand in [".", "!", "?"] or energia_acumulada >= limiar_dinamico:
                    parada_forcada = True
                    break
            else:
                break

        homeostase.ajustar(candidatos_bloco_info)

        stats['blocos'] += 1
        stats['aceitos'] += len(accepted_prefix)
        stats['rejeitados'] += (len(draft) - len(accepted_prefix))

        if accepted_prefix:
            accepted.extend(accepted_prefix)
            tokens.extend(accepted_prefix)
        else:
            fallback = target.tokens_lista[random.randint(0, len(target.tokens_lista) - 1)]
            accepted.append(fallback)
            tokens.append(fallback)

        if parada_forcada:
            break

    # REAÇÃO AMBIENTAL: Aplica a atração/repulsão no ambiente latente
    if feedback_sinal != 0.0:
        target.reacao_ambiental(accepted, feedback_sinal)

    # Formatação e limpeza interna da string de saída
    texto_saida = " ".join(accepted[:max_tokens])
    for p in [".", ",", "!", "?", ";", ":"]:
        texto_saida = texto_saida.replace(f" {p}", p)

    meta_info = {
        "estilo_detectado": nome_estilo,
        "tensao_final": homeostase.tensao,
        "temperatura_final": homeostase.temperatura,
        "posicao_espacial": (round(pos_x, 2), round(pos_y, 2)),
        "stats": stats
    }

    return texto_saida, meta_info


# =============================================================================
# EXECUTÁVEL
# =============================================================================
if __name__ == "__main__":
    dataset_treino = """
    Geometric intelligence maps concepts onto the Poincaré disk!
    The dspark system accelerates speculative batch token generation.
    Hyperbolic attraction and repulsion adjust vectors in 128-dimensional space!
    The Telica acts as a local syntactic cohesion manifold.
    The TGP agent combines linear memory with cognitive geo-attention, without backpropagation!
    Dual coexistence balances linear probability and geodesic distance.
    prezados senhores apresentamos a solucao tecnica para analise de redes.
    e ai beleza tudo certo no fluxo de dados.
    """

    print("🚀 1. Carregando Base Unificada ARQUINET v2.0...")
    target = AgenteTGP_13(dim=128)
    target.treinar(dataset_treino, epocas=20)

    print("⚡ 2. Inicializando Drafter Cache Temporario + Bússola PLC...")
    drafter = DrafterCacheTemporario(target)
    limiar = calcular_limiar_espacial(dataset_treino)

    prompts_teste = [
        "Geometric",
        "prezados senhores",
        "e ai beleza"
    ]

    print("\n------------------------------------------------------------------")
    print("3. EXECUTANDO INFERÊNCIA UNIFICADA COM DRAFTER, PLC E ESPAÇO HIPERBÓLICO")
    print("------------------------------------------------------------------")

    for p in prompts_teste:
        t_inicio = time.time()
        texto_saida, meta = arquinet_hybrid_decode(
            target, drafter, p, limiar_corte_base=limiar, max_tokens=20, block_size=4, feedback_sinal=1.0
        )
        t_fim = time.time() - t_inicio
        qtd_tokens = len(texto_saida.split())

        print(f"\n💬 Entrada: '{p}'")
        print(f"🎭 Estilo PLC: {meta['estilo_detectado']} | 🌡️ Tensão: {meta['tensao_final']:.2f} | Temp: {meta['temperatura_final']:.2f}")
        print(f"📍 Posição Espacial: X={meta['posicao_espacial'][0]}, Y={meta['posicao_espacial'][1]}")
        print(f"📄 Saída: {texto_saida}")
        print(f"⏱️ Tempo: {t_fim:.4f}s ({qtd_tokens/max(t_fim, 1e-6):.1f} t/s) | Blocos: {meta['stats']['blocos']} (Aceitos: {meta['stats']['aceitos']})")
