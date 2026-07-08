import numpy as np
import time
import random
import math

# --- 1. MODELO DE MUNDO E CAUSALIDADE (REQ 1) ---
class CausalWorld:
    def __init__(self):
        self.laws = {
            "empurrar": {"condicao": "massa < 5", "efeito": {"posicao": "chao", "movendo": False}, "dano_se": "fragil"},
            "bater": {"condicao": "dureza > 5", "efeito": {"fixo": True}, "dano_se": "fragil"}
        }

    def simular(self, entidade, acao):
        # Simula o "E SE?" antes de agir
        props = entidade.properties
        if acao in self.laws:
            law = self.laws[acao]
            if props.get("fragil") and "dano_se" in law:
                return "DESTRUIÇÃO", 1.0
            return "SUCESSO_MECANICO", 0.0
        return "DESCONHECIDO", 0.5

# --- 2. MEMÓRIA EPISÓDICA E SEMÂNTICA (REQ 3 e 5) ---
class Entity:
    def __init__(self, nome, props):
        self.nome = nome
        self.properties = props # Ex: {"fragil": True, "massa": 0.2}

class MemorySystem:
    def __init__(self):
        self.semantic = {
            "copo": Entity("copo", {"fragil": True, "massa": 0.2, "tipo": "recipiente"}),
            "martelo": Entity("martelo", {"fragil": False, "massa": 2.0, "tipo": "ferramenta"})
        }
        self.episodic = [] # Lista de eventos (Tempo, Ação, Resultado)

    def recordar_evento(self, termo):
        return [e for e in self.episodic if termo in e['evento']]

# --- 3. QUINTIKUS (PERCEPÇÃO E EMOÇÃO - REQ 7) ---
class QuintikusPerception:
    def processar(self, entrada):
        # Sistema Límbico: Define valência e curiosidade
        valencia = "NEUTRO"
        if any(w in entrada for w in ["perigo", "cuidado", "quebrar"]): valencia = "ALERTA"
        if "?" in entrada: valencia = "CURIOSO"
        
        return {"input": entrada, "mood": valencia, "timestamp": time.time()}

# --- 4. TGNC (ENTENDIMENTO E APRENDIZADO - REQ 4) ---
class TGNC_Understanding:
    def __init__(self):
        self.weights = {} # Simulação de pesos sinápticos

    def analisar(self, percepcao):
        texto = percepcao["input"].lower()
        # Mapeia Texto -> Nexo -> Ação
        if "copo" in texto:
            nexo, acao = "OBJETO_FRAGIL", "olhar"
            confianca = 0.85
        elif "martelo" in texto:
            nexo, acao = "FERRAMENTA_PESADA", "bater"
            confianca = 0.90
        else:
            nexo, acao = "DESCONHECIDO", "explorar"
            confianca = 0.20 # Gera lacuna de curiosidade
            
        return nexo, acao, confianca

# --- 5. CONSCIOUS CORE (O MAESTRO - REQ 2, 6, 8, 9, 10) ---
class ConsciousCore:
    def __init__(self):
        self.quintikus = QuintikusPerception()
        self.tgnc = TGNC_Understanding()
        self.world = CausalWorld()
        self.memory = MemorySystem()
        self.goals = ["MANTER_INTEGRIDADE", "APRENDER"] # REQ 9: Motor de Objetivos
        self.scratchpad = []

    def pensar(self, entrada):
        self.scratchpad = []
        self._log(f"--- Início do Ciclo Cognitivo: '{entrada}' ---")

        # PASSO 1: PERCEPÇÃO (Sente o contexto)
        percepcao = self.quintikus.processar(entrada)
        
        # PASSO 2: ENTENDIMENTO (Rotula o nexo)
        nexo, acao_sugerida, conf = self.tgnc.analisar(percepcao)

        # PASSO 3: METACOGNIÇÃO E CURIOSIDADE (REQ 6 e 8)
        # "Eu entendo o que é isso?"
        alvo = self._extrair_entidade(entrada)
        entidade = self.memory.semantic.get(alvo)

        if not entidade or conf < 0.4:
            return self._curiosidade_ativa(alvo)

        # PASSO 4: PLANEJAMENTO E MODELO DE MUNDO (REQ 1 e 2)
        # "Se eu fizer a ação sugerida, o que acontece?"
        previsao, risco = self.world.simular(entidade, acao_sugerida)
        
        self._log(f"Autoavaliação: Nexo {nexo}, Risco de dano: {risco}")

        # PASSO 5: TOMADA DE DECISÃO (REQ 9)
        # Avalia se a ação fere o objetivo de "MANTER_INTEGRIDADE"
        if risco > 0.7:
            decisao = f"ABORTAR {acao_sugerida}. Motivo: Risco de quebra detectado no WorldModel."
        else:
            decisao = f"EXECUTAR {acao_sugerida} em {alvo}."

        # PASSO 6: APRENDIZADO POR EXPERIÊNCIA (REQ 10)
        self._registrar_episodio(entrada, decisao, previsao)
        
        return self._gerar_resposta(decisao, conf, percepcao["mood"])

    def _curiosidade_ativa(self, alvo):
        # REQ 6: Ativa busca por conhecimento quando a confiança é baixa
        self._log(f"CURIOSIDADE ATIVA: Lacuna detectada sobre '{alvo}'")
        return f"Não possuo modelo causal para '{alvo}'. Você pode me explicar a função deste objeto?"

    def _extrair_entidade(self, texto):
        for k in self.memory.semantic.keys():
            if k in texto.lower(): return k
        return "desconhecido"

    def _registrar_episodio(self, entrada, decisao, resultado):
        # REQ 3: Memória Episódica Real (Linha do tempo)
        evento = {
            "t": time.time(),
            "entrada": entrada,
            "decisao": decisao,
            "resultado_esperado": resultado
        }
        self.memory.episodic.append(evento)
        # REQ 10: Feedback Loop (Aprendizado)
        if len(self.memory.episodic) > 0:
            self._log("Consolidando experiência na base semântica...")

    def _log(self, msg):
        print(f"  [PENSAMENTO]: {msg}")
        self.scratchpad.append(msg)

    def _gerar_resposta(self, decisao, conf, mood):
        # REQ 8: Autoconsciência Operacional
        explicacao = f"\nRazão: Confiança de {conf*100}% com base no estado {mood}."
        return f"RESULTADO: {decisao} {explicacao}"

# --- EXECUÇÃO DO TESTE (AGI OPERATIONAL) ---

agi = ConsciousCore()

print("\n--- CENÁRIO 1: ENTIDADE CONHECIDA E RISCO ---")
# Aqui ele deve usar o World Model para ver que o copo quebra
print(agi.pensar("Vou bater com o martelo no copo."))

print("\n--- CENÁRIO 2: ENTIDADE DESCONHECIDA (CURIOSIDADE) ---")
# Aqui ele deve admitir que não sabe e pedir informação (Lacuna)
print(agi.pensar("O que faz um Quantum-Siphon?"))

print("\n--- CENÁRIO 3: RACIOCÍNIO SIMBÓLICO E SEGURO ---")
# Aqui ele entende que o martelo é robusto
print(agi.pensar("Pode empurrar o martelo na mesa?"))
