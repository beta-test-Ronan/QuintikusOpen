#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import re
import time
import unicodedata
import struct
import os
from array import array
from collections import defaultdict, deque
from typing import Dict, List, Tuple

# Pré-compilação do Regex para ganho de performance na tokenização
TOKENIZER_REGEX = re.compile(r'[a-z0-9]+|[.,!?;:]')

# =============================================================================
# PERSISTÊNCIA ATÔMICA POR CAMADAS (EIXO Y)
# =============================================================================
class PersistenciaAtomicaCamadas:
    def __init__(self, filepath: str = "arquinet_disco.bin"):
        self.filepath = filepath

    def salvar_camada(self, camada_y: int, tokens_vetores: Dict[str, List[float]], bigramas: dict):
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

    def carregar_ultima_camada(self) -> Tuple[int, Dict[str, List[float]]]:
        if not os.path.exists(self.filepath):
            return 0, {}

        ultima_camada = 0
        ultimos_vetores = {}

        try:
            with open(self.filepath, "rb") as f:
                if f.read(4) != b"ARQK":
                    return 0, {}
                
                camada_bytes = f.read(4)
                if len(camada_bytes) < 4:
                    return 0, {}
                camada_y = struct.unpack("i", camada_bytes)[0]
                
                num_tokens_bytes = f.read(4)
                if len(num_tokens_bytes) < 4:
                    return 0, {}
                num_tokens = struct.unpack("i", num_tokens_bytes)[0]

                vetores_temp = {}
                for _ in range(num_tokens):
                    l_bytes = f.read(2)
                    if len(l_bytes) < 2:
                        break
                    t_len = struct.unpack("H", l_bytes)[0]
                    token = f.read(t_len).decode("utf-8")
                    
                    dim_bytes = f.read(4)
                    if len(dim_bytes) < 4:
                        break
                    dim = struct.unpack("i", dim_bytes)[0]
                    
                    float_data = f.read(4 * dim)
                    if len(float_data) < 4 * dim:
                        break
                    vetores_temp[token] = list(struct.unpack(f"{dim}f", float_data))

                ultima_camada = camada_y
                ultimos_vetores = vetores_temp
        except Exception as e:
            print(f"⚠️ Erro ao ler disco, reiniciando camada: {e}")
            return 0, {}

        return ultima_camada, ultimos_vetores


# =============================================================================
# MÓDULO WERNICKE: FILTRAGEM DE CONTEXTO E ISOLAMENTO DE DOMÍNIO
# =============================================================================
class WernickeContextRouter:
    """
    Atua como o filtro de relevância e isolamento de contexto,
    evitando a contaminação entre domínios (Conversacional vs. Técnico).
    """
    def __init__(self):
        # Mapeamento estrito de vocabulário por domínio para evitar vazamento cruzado
        self.dominios = {
            "modo_conversa": ["oi", "tudo", "bem", "como", "foi", "dia", "obrigado", "ola", "e", "aí", "beleza", "tudo", "certo", "por", "aí", "tranquilo", "amigo"],
            "modo_tarefa": ["banco", "dados", "tokens", "fluxo", "geometric", "hiperbolico", "vetores", "codigo", "sistema", "analise", "modelo"]
        }

    def filtrar_vocabulario(self, rota_ativa: str, logits_brutos: dict) -> dict:
        """
        Zera ou penaliza logits de tokens que não pertencem ao contexto ativo,
        eliminando a contaminação apontada na análise de épocas.
        """
        if rota_ativa == "modo_conversa":
            for token in list(logits_brutos.keys()):
                # Se o token for estritamente técnico pesado e o modo for conversa casual, inibimos
                if token in ["banco", "dados", "hiperbolico", "vetores", "geometric", "dspark"]:
                    logits_brutos[token] = -float('inf')
                    
        return logits_brutos


# =============================================================================
# NÚCLEO DE ESTADO INTERNO (O "CÉREBRO" DA CONVERSA)
# =============================================================================
class NucleoEstadoInterno:
    """
    Mantém a identidade, emoção percebida, objetivos, foco e energia da conversa,
    garantindo continuidade e transformando o modelo de linguagem apenas na 'boca'.
    """
    def __init__(self):
        self.emocao_percebida = "neutra"
        self.objetivo = "conversar_e_ajudar"
        self.assunto = "geral"
        self.nivel_confianca = 0.8
        self.energia_conversa = "media"
        self.historico_emocional = deque(maxlen=10)

    def interpretar_externo(self, texto: str):
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

    def exportar_bias_decodificacao(self) -> Dict[str, float]:
        if self.emocao_percebida == "negativa_ou_tensao":
            return {"mod_temp": -0.2, "mod_tensao": 0.3} 
        elif self.emocao_percebida == "positiva_casual":
            return {"mod_temp": 0.2, "mod_tensao": -0.1} 
        elif self.emocao_percebida == "focada_analitica":
            return {"mod_temp": -0.3, "mod_tensao": 0.2} 
        return {"mod_temp": 0.0, "mod_tensao": 0.0}


# =============================================================================
# 0. FREIO INTELIGENTE (COM PISO DE SEGURANÇA)
# =============================================================================
class FreioInteligente:
    def __init__(self, janela_observacao: int = 5, limiar_queda: float = 0.5, piso_qualidade: float = 1e-8, max_estagnacao: int = 4):
        self.historico_qualidade = deque(maxlen=janela_observacao)
        self.janela = janela_observacao
        self.limiar_queda = limiar_queda
        self.piso_qualidade = piso_qualidade
        self.max_estagnacao = max_estagnacao

    def registrar_qualidade(self, media_qualidade: float):
        self.historico_qualidade.append(float(media_qualidade))

    def deve_parar(self) -> Tuple[bool, str]:
        if not self.historico_qualidade:
            return False, ""
        if self.historico_qualidade[-1] <= self.piso_qualidade:
            return True, "piso_qualidade_atingido"
        if len(self.historico_qualidade) < self.janela:
            return False, ""
        
        ultimos = list(self.historico_qualidade)[-self.max_estagnacao:]
        if len(ultimos) >= self.max_estagnacao and len(set(round(v, 6) for v in ultimos)) == 1:
            return True, "estagnacao_loop_detectada"
        return False, ""


# =============================================================================
# 1. ROTEADOR COMPORTAMENTAL TPTHINK
# =============================================================================
class TPThinkBehavioralRouter:
    def __init__(self):
        self.padroes_comportamento = {
            'comando_lista': ['faça', 'depois', 'crie', 'lista', 'gere', 'passo a passo', 'ordene'],
            'pergunta_direta': ['?', 'qual', 'como', 'por que', 'o que', 'descreva', 'explique'],
            'casual': ['oi', 'olá', 'tudo bem', 'boa tarde', 'beleza', 'novidades', 'fala', 'tu', 'mano', 'e aí'],
            'ofensivo': ['odeio', 'raiva', 'injusto', 'lixo', 'burro', 'ódio', 'inútil', 'horrível']
        }
        self.diagrama_perguntas = {
            'aberta': ['como', 'por que', 'o que', 'descreva'],
            'fechada': ['você fez', 'é sim', 'qual opção', 'sim ou não'],
            'reflexiva': ['faz pensar', 'evidências', 'e se', 'dia']
        }
        self.bases_de_estado = {
            'modo_tarefa':     array('B', [1, 0, 0, 0, 0, 0, 0, 1]),
            'modo_investigar': array('B', [0, 1, 0, 0, 1, 0, 0, 0]),
            'modo_binario':    array('B', [0, 1, 0, 0, 0, 1, 0, 0]),
            'modo_reflexivo':  array('B', [0, 1, 0, 0, 0, 0, 1, 0]),
            'modo_conversa':   array('B', [0, 0, 1, 0, 0, 0, 0, 0]),
            'modo_defesa':     array('B', [0, 0, 0, 1, 0, 0, 0, 0]),
            'modo_expansivo':  array('B', [0, 0, 0, 0, 0, 0, 0, 0])
        }
        self.mapa_plc = {
            'modo_tarefa':     {"estilo": "instrucao",  "temperatura": 0.7, "tensao": 1.2},
            'modo_investigar': {"estilo": "formal",     "temperatura": 0.6, "tensao": 1.1},
            'modo_binario':    {"estilo": "formal",     "temperatura": 0.4, "tensao": 1.3},
            'modo_reflexivo':  {"estilo": "expansivo",  "temperatura": 1.0, "tensao": 0.9},
            'modo_conversa':   {"estilo": "informal",   "temperatura": 1.1, "tensao": 0.9},
            'modo_defesa':     {"estilo": "formal",     "temperatura": 0.5, "tensao": 1.4},
            'modo_expansivo':  {"estilo": "expansivo",  "temperatura": 1.2, "tensao": 0.8}
        }

    def _rastrear_gatilhos(self, texto: str, dicionario: Dict) -> List[str]:
        texto = texto.lower()
        return [chave for chave, gatilhos in dicionario.items() if any(g in texto for g in gatilhos)]

    def _gerar_array_input(self, comportamentos: List[str], tipo_pergunta: List[str]) -> array:
        vetor_dinamico = array('B', [0] * 8)
        if 'comando_lista' in comportamentos: vetor_dinamico[0] = 1
        if 'pergunta_direta' in comportamentos: vetor_dinamico[1] = 1
        if 'casual' in comportamentos: vetor_dinamico[2] = 1
        if 'ofensivo' in comportamentos: vetor_dinamico[3] = 1
        if 'pergunta_direta' in comportamentos and tipo_pergunta:
            if 'aberta' in tipo_pergunta: vetor_dinamico[4] = 1
            if 'fechada' in tipo_pergunta: vetor_dinamico[5] = 1
            if 'reflexiva' in tipo_pergunta: vetor_dinamico[6] = 1
        if 'comando_lista' in comportamentos: vetor_dinamico[7] = 1
        return vetor_dinamico

    def pre_processar_estilo(self, texto: str) -> Tuple[str, Dict[str, float], array]:
        comportamentos = self._rastrear_gatilhos(texto, self.padroes_comportamento)
        tipo_pergunta = self._rastrear_gatilhos(texto, self.diagrama_perguntas) if 'pergunta_direta' in comportamentos else []
        vetor_entrada = self._gerar_array_input(comportamentos, tipo_pergunta)

        if not comportamentos and not tipo_pergunta:
            rota_escolhida = 'modo_expansivo'
        else:
            distancias = {nome: math.dist(vetor_entrada, base) for nome, base in self.bases_de_estado.items()}
            rota_escolhida = min(distancias, key=distancias.get)

        perfil_plc = self.mapa_plc[rota_escolhida]
        return rota_escolhida, perfil_plc, vetor_entrada


# =============================================================================
# 2. BASE GEOMÉTRICA HIPERBÓLICA (POINCARÉ DISK CORE)
# =============================================================================
class DiscoPoincare:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.raio = 0.985
        self.eps = 1e-7

    def norma(self, v: List[float]) -> float:
        return math.hypot(*v)

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
        
        diff_sq = math.dist(x_p, y_p) ** 2
        num = 2 * diff_sq
        den = (1 - norma_x**2) * (1 - norma_y**2) + self.eps
        val = max(1.0, 1.0 + num / den)
        return math.acosh(min(val, 1e5))


# =============================================================================
# 3. MEMÓRIA LINEAR E ATENÇÃO COGNITIVA
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

        for contagens in self.bigramas.values():
            total = sum(contagens.values())
            if total > 0:
                for t2 in contagens: contagens[t2] /= total

        for contagens in self.trigramas.values():
            total = sum(contagens.values())
            if total > 0:
                for t3 in contagens: contagens[t3] /= total

    def prob_linear(self, t_atual: str, t_candidato: str, t_penultimo: str = None) -> float:
        p_bigrama = self.bigramas.get(t_atual, {}).get(t_candidato, 1e-4)
        p_trigrama = 0.0
        if t_penultimo and (t_penultimo, t_atual) in self.trigramas:
            p_trigrama = self.trigramas[(t_penultimo, t_atual)].get(t_candidato, 0.0)
        return max(1e-5, 0.4 * p_bigrama + 0.6 * p_trigrama)


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
# 4. HOMEOSTASE ESPACIAL E AGENTE TARGET
# =============================================================================
class HomeostaseEspacial:
    def __init__(self, tensao_base: float = 1.0, temperatura_base: float = 1.0):
        self.tensao = max(0.5, min(2.0, tensao_base))
        self.temperatura = max(0.5, min(2.0, temperatura_base))

    def ajustar(self, candidatos_avaliados: List[Tuple[str, float, float]]):
        if not candidatos_avaliados: return
        bons = sum(1 for c in candidatos_avaliados if c[1] > 1e-3)
        ruins = len(candidatos_avaliados) - bons

        if bons >= 1 and ruins <= 2:
            self.tensao = max(0.7, self.tensao * 0.95)
        elif ruins > bons:
            self.tensao = min(1.5, self.tensao * 1.05)


class AgenteTGP_13:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.disco = DiscoPoincare(dim)
        self.memoria = MemoriaLinear()
        self.atencao = AtencaoCognitiva(self.disco)
        self.tpthink = TPThinkBehavioralRouter()
        self.wernicke = WernickeContextRouter()
        self.nucleo_estado = NucleoEstadoInterno()
        self.token_para_vetor: Dict[str, List[float]] = {}
        self.tokens_lista: List[str] = []
        self.persistencia = PersistenciaAtomicaCamadas()
        
        self.camada_atual, dados_carregados = self.persistencia.carregar_ultima_camada()
        if dados_carregados:
            self.token_para_vetor = dados_carregados
            self.tokens_lista = list(dados_carregados.keys())
            self.memoria.registrar_fluxo(self.tokens_lista)

    def tokenizar(self, texto: str) -> List[str]:
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acentos = ''.join(c for c in texto_nfkd if unicodedata.category(c) != 'Mn')
        return [t for t in TOKENIZER_REGEX.findall(texto_sem_acentos) if t.strip()]

    def _registrar_token(self, token: str):
        if token not in self.token_para_vetor:
            self.token_para_vetor[token] = [(random.random() - 0.5) * 0.05 for _ in range(self.dim)]
            self.tokens_lista.append(token)

    def treinar(self, texto_base: str, epocas: int = 5):
        tokens = self.tokenizar(texto_base)
        if not tokens: return
        self.memoria.registrar_fluxo(tokens)
        for t in tokens: self._registrar_token(t)

        for _ in range(epocas):
            for i in range(len(tokens) - 1):
                t_atual, t_prox = tokens[i], tokens[i + 1]
                v_atual = self.token_para_vetor[t_atual]
                v_prox = self.token_para_vetor[t_prox]
                diff_atracao = [(b - a) * 0.02 for a, b in zip(v_atual, v_prox)]
                self.token_para_vetor[t_atual] = self.disco.adicao_mobius(v_atual, diff_atracao)

        for t in self.tokens_lista:
            self.token_para_vetor[t] = self.disco.projetar(self.token_para_vetor[t])

        self.camada_atual += 1
        self.persistencia.salvar_camada(self.camada_atual, self.token_para_vetor, self.memoria.bigramas)

    def criar_snapshot_cognitivo(self, contexto: List[str]) -> List[float]:
        vetores_ctx = [self.token_para_vetor[t] for t in contexto if t in self.token_para_vetor]
        if not vetores_ctx:
            return [0.0] * self.dim
        return self.atencao.ruminar(vetores_ctx)

    def avaliacao_completa(self, snapshot_vector: List[float], t_atual: str, token_cand: str, t_penultimo: str = None, temperatura: float = 1.0, tensao: float = 1.0) -> Tuple[float, float]:
        if token_cand not in self.token_para_vetor: return 1e-12, 10.0
        v_cand = self.token_para_vetor[token_cand]
        
        dist_hip = self.disco.distancia(snapshot_vector, v_cand)
        p_nao_linear = math.exp(-dist_hip / max(0.1, temperatura))
        p_linear = self.memoria.prob_linear(t_atual, token_cand, t_penultimo)

        prob_final = math.sqrt(p_nao_linear * p_linear)
        return prob_final, dist_hip

    def reacao_ambiental(self, tokens_gerados: List[str], feedback_sinal: float):
        if feedback_sinal == 0.0: return
        fator = 0.01 * feedback_sinal
        for t in tokens_gerados:
            if t in self.token_para_vetor:
                self.token_para_vetor[t] = self.disco.adicao_mobius(self.token_para_vetor[t], [fator] * self.dim)


# =============================================================================
# CORRETOR DINÂMICO DE ÂNCORA
# =============================================================================
class CorretorDinamicoAncora:
    def __init__(self, target: AgenteTGP_13):
        self.target = target

    def selecionar_melhor_ancora(self, tokens_contexto: List[str], tokens_ja_gerados: List[str]) -> str:
        for t in reversed(tokens_contexto):
            if t in self.target.token_para_vetor and tokens_ja_gerados.count(t) < 2:
                return t
        if self.target.tokens_lista:
            return self.target.tokens_lista[0]
        return '<s>'


# =============================================================================
# 5. DRAFTER CACHE TEMPORÁRIO COM SUPORTE A WERNICKE
# =============================================================================
class DrafterCacheTemporario:
    def __init__(self, target: AgenteTGP_13):
        self.target = target
        self.vocab = target.tokens_lista if target.tokens_lista else ['<s>']
        self.transition = defaultdict(lambda: defaultdict(lambda: 0.01))
        self._sincronizar_memoria()

    def _sincronizar_memoria(self):
        for t1, conexoes in self.target.memoria.bigramas.items():
            for t2, p in conexoes.items():
                self.transition[t1][t2] = max(p, 0.01)

    def draft_block(self, anchor: str, block_size: int = 4, rota_ativa: str = None) -> List[str]:
        draft = []
        prev = anchor
        for _ in range(block_size):
            conexoes = dict(self.transition[prev])
            
            # Aplicação do Filtro Wernicke diretamente no Drafter para evitar contaminação
            if rota_ativa:
                conexoes = self.target.wernicke.filtrar_vocabulario(rota_ativa, conexoes)
                conexoes = {k: v for k, v in conexoes.items() if v != -float('inf')}

            if not conexoes:
                chosen = random.choice(self.vocab)
            else:
                cand, weights = zip(*conexoes.items())
                soma = sum(weights)
                if soma <= 0:
                    chosen = random.choice(self.vocab)
                else:
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
    tokens = TOKENIZER_REGEX.findall(texto_treino.lower())
    limiar_acumulado = sum(MAPA_NEURONAL.get(token, 0.0) for token in tokens)
    qtd_pontos = sum(1 for token in tokens if token in MAPA_NEURONAL)
    return max(1.0, limiar_acumulado / max(1, qtd_pontos))


# =============================================================================
# 6. MOTOR DE DECODIFICAÇÃO UNIFICADO
# =============================================================================
def arquinet_hybrid_decode(
    target: AgenteTGP_13,
    drafter: DrafterCacheTemporario,
    prompt: str,
    limiar_corte_base: float,
    max_tokens: int = 16,
    block_size: int = 4,
    feedback_sinal: float = 0.0
) -> Tuple[str, Dict]:
    
    target.nucleo_estado.interpretar_externo(prompt)
    bias_interno = target.nucleo_estado.exportar_bias_decodificacao()

    corretor_ancora = CorretorDinamicoAncora(target)
    tokens = target.tokenizar(prompt)
    if not tokens: tokens = [target.tokens_lista[0] if target.tokens_lista else '<s>']
    accepted = []

    modo_tpthink, perfil_plc, bit_array = target.tpthink.pre_processar_estilo(prompt)
    
    temp_ajustada = max(0.2, perfil_plc["temperatura"] + bias_interno["mod_temp"])
    tensao_ajustada = max(0.5, perfil_plc["tensao"] + bias_interno["mod_tensao"])

    homeostase = HomeostaseEspacial(
        tensao_base=tensao_ajustada,
        temperatura_base=temp_ajustada
    )

    freio = FreioInteligente(janela_observacao=6, limiar_queda=0.4, piso_qualidade=1e-7, max_estagnacao=3)

    stats = {'blocos': 0, 'aceitos': 0, 'rejeitados': 0, 'motivo_parada': 'max_tokens'}

    while len(accepted) < max_tokens:
        anchor = corretor_ancora.selecionar_melhor_ancora(tokens, accepted)
        contexto = tokens[-4:]

        snapshot = target.criar_snapshot_cognitivo(contexto)
        
        # O Roteador Wernicke define a rota ativa para limpar o vocabulário indesejado no Drafter
        draft = drafter.draft_block(anchor, block_size=block_size, rota_ativa=modo_tpthink)

        accepted_prefix = []
        curr_anchor = anchor
        curr_penultimate = contexto[-2] if len(contexto) >= 2 else None

        parada_forcada = False
        candidatos_bloco_info = []

        for t_cand in draft:
            if t_cand in accepted or t_cand == curr_anchor:
                continue

            p_val, dist_hip = target.avaliacao_completa(
                snapshot, curr_anchor, t_cand, curr_penultimate,
                temperatura=homeostase.temperatura,
                tensao=homeostase.tensao
            )
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

        if candidatos_bloco_info:
            qualidade_media_bloco = sum(c[1] for c in candidatos_bloco_info) / len(candidatos_bloco_info)
            freio.registrar_qualidade(qualidade_media_bloco)

        parar_freio, motivo_freio = freio.deve_parar()
        if parar_freio:
            stats['motivo_parada'] = f'freio_{motivo_freio}'
            break

        homeostase.ajustar(candidatos_bloco_info)

        stats['blocos'] += 1
        stats['aceitos'] += len(accepted_prefix)
        stats['rejeitados'] += (len(draft) - len(accepted_prefix))

        if accepted_prefix:
            accepted.extend(accepted_prefix)
            tokens.extend(accepted_prefix)
        else:
            candidatos_seguros = [t for t in target.tokens_lista if t not in accepted]
            if candidatos_seguros:
                fallback = random.choice(candidatos_seguros)
                accepted.append(fallback)
                tokens.append(fallback)
            else:
                break

        if parada_forcada:
            break

    if feedback_sinal != 0.0:
        target.reacao_ambiental(accepted, feedback_sinal)

    texto_saida = " ".join(accepted[:max_tokens])
    for p in [".", ",", "!", "?", ";", ":"]:
        texto_saida = texto_saida.replace(f" {p}", p)

    meta_info = {
        "modo_tpthink": modo_tpthink,
        "estado_interno": {
            "emocao": target.nucleo_estado.emocao_percebida,
            "objetivo": target.nucleo_estado.objetivo,
            "assunto": target.nucleo_estado.assunto,
            "energia": target.nucleo_estado.energia_conversa
        },
        "estilo_plc": perfil_plc["estilo"],
        "tensao_final": homeostase.tensao,
        "temperatura_final": homeostase.temperatura,
        "stats": stats
    }

    return texto_saida, meta_info


if __name__ == "__main__":
    dataset_treino = """
    faça uma pesquisa no banco e crie a lista de dados atualizados.
    crie a lista de passos para executar o sistema com sucesso.
    gere um relatorio completo e ordene os resultados por prioridade.
    
    por que voce fez essa alteracao no codigo do sistema?
    como funciona a analise de dados no modelo hiperbolico?
    o que desencadeou essa resposta no banco de dados?
    
    oi tudo bem como voce esta mano?
    fala tu beleza tudo certo no fluxo de dados.
    ola boa tarde tranquilo por ai meu amigo?
    
    hoje foi um dia horrível e difícil no trabalho.
    mantenha a calma e apresente a solucao formal para a equipe.
    analise os erros com cuidado e execute o protocolo de defesa.
    
    geometric intelligence mapeia conceitos no disco de poincare.
    o sistema dspark acelera a geracao especulativa de tokens em lote.
    a atracao hiperbolica ajusta os vetores no espaco multidimensional.
    o agente tgp combina memoria linear com atencao geometrica sem backpropagation.
    faça uma pesquisa no banco e crie a lista de dados atualizados.
    crie a lista de passos para executar o sistema com sucesso.
    gere um relatorio completo e ordene os resultados por prioridade.
    
    por que voce fez essa alteracao no codigo do sistema?
    como funciona a analise de dados no modelo hiperbolico?
    o que desencadeou essa resposta no banco de dados?
    
    oi tudo bem como voce esta mano?
    fala tu beleza tudo certo no fluxo de dados.
    ola boa tarde tranquilo por ai meu amigo?
    Oi! Tudo bem com você?
    Olá! Como está seu dia?
    E aí! Como vão as coisas?
    Oi! Em que posso ajudar hoje?
    Olá! Que bom falar com você.
    E aí, tudo certo por aí?
    Oi! Espero que esteja tudo bem.
    Olá! Como você está hoje?
    Opa! Tudo tranquilo?
    Oi! É um prazer conversar com você.
    
    esse codigo esta ruim mas vamos corrigir a falha técnica com calma.
    mantenha a calma e apresente a solucao formal para a equipe.
    analise os erros com cuidado e execute o protocolo de defesa.
    
    geometric intelligence mapeia conceitos no disco de poincare.
    o sistema dspark acelera a geracao especulativa de tokens em lote.
    a atracao hiperbolica ajusta os vetores no espaco multidimensional.
    o agente tgp combina memoria linear com atencao geometrica sem backpropagation.
    """

    print("🚀 1. Inicializando Agente Arquinet com Núcleo de Estado Interno e Wernicke Router...")
    target = AgenteTGP_13(dim=512)
    
    if not target.tokens_lista:
        target.treinar(dataset_treino, epocas=1000)
    else:
        print(f"🔄 Disco carregado com sucesso na Camada Y: {target.camada_atual}")
    
    drafter = DrafterCacheTemporario(target)
    limiar = calcular_limiar_espacial(dataset_treino)

    prompts_teste = [
        "Oi, tudo beleza por aí?"
    ]

    print("\n------------------------------------------------------------------")
    print("3. EXECUTANDO INFERÊNCIA COM ISOLAMENTO DE DOMÍNIO (WERNICKE)")
    print("------------------------------------------------------------------")

    for p in prompts_teste:
        t_inicio = time.time()
        texto_saida, meta = arquinet_hybrid_decode(
            target, drafter, p, limiar_corte_base=limiar, max_tokens=20, block_size=10, feedback_sinal=1.0
        )
        t_fim = time.time() - t_inicio

        print(f"\n💬 Entrada Externa: '{p}'")
        print(f"🧠 Estado Interno -> Emoção: {meta['estado_interno']['emocao']} | Assunto: {meta['estado_interno']['assunto']} | Objetivo: {meta['estado_interno']['objetivo']}")
        print(f"⚙️ Ajuste Boca -> Rota: {meta['modo_tpthink']} | Tensão: {meta['tensao_final']:.2f} | Temp: {meta['temperatura_final']:.2f}")
        print(f"🛑 Parada: {meta['stats']['motivo_parada']} | Blocos: {meta['stats']['blocos']} | Aceitos: {meta['stats']['aceitos']}")
        print(f"📤 Saída Gerada: '{texto_saida}'")
        print(f"⏱️ Tempo de Inferência: {t_fim:.4f}s")
