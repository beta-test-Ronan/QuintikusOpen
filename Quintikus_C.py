import os
import numpy as np
import unicodedata
import time
import math
import threading
import random
from collections import defaultdict

# ==================================================================
#    ❄️ ESCUDO TÉRMICO E FONTES DE AUTORIDADE
# ==================================================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 

# ==================================================================
# 1. ÁREA DE BROCA (Síntese Emocional e Térmica)
# ==================================================================
class AreaDeBroca:
    def __init__(self):
        # Estados: [0]: Pressão (Calor/Erro), [1]: Sinergia (Harmonia/Paz)
        self.st = [0.5, 0.5]
        # Mapa Térmico Minimalista (Extraído do seu código de análise)
        self.th = {
            'bom': 0.1, 'otimo': 0.2, 'sinergia': 0.3, 'paz': 0.2, 
            'erro': -0.2, 'urgente': -0.3, 'falha': -0.2, 'ruido': -0.1
        }
        self.tons = {
            "quente": ["Sob pressão dos dados, ", "Em estado de alerta, ", "Detectando urgência, "],
            "harmonia": ["Em sinergia plena, ", "Com clareza, ", "De forma harmoniosa, "],
            "neutro": ["Pela razão do solo, ", "Observando os nexos, ", "No fluxo dos fatos, "]
        }

    def sentir(self, tokens):
        """TSPLS: Atualização Procedural do Estado Emocional"""
        pos, neg = 0, 0
        for t in tokens:
            if t in self.th:
                val = self.th[t]
                if val > 0: pos += val
                else: neg += abs(val)
        
        # Equação Térmica: T = T * 0.95 + (N * 0.1) | S = S * 0.95 + (P * 0.1)
        self.st[0] = self.st[0] * 0.95 + (neg * 0.1)
        self.st[1] = self.st[1] * 0.95 + (pos * 0.1)
        self.st = [max(0, min(1, s)) for s in self.st]

    def modular_voz(self):
        """Define o prefixo da frase com base na temperatura interna"""
        if self.st[0] > 0.6: return random.choice(self.tons["quente"])
        if self.st[1] > 0.6: return random.choice(self.tons["harmonia"])
        return random.choice(self.tons["neutro"])

# ==================================================================
# 2. ESPINHA DORSAL (Orquestrador QuintikusAGI)
# ==================================================================
class QuintikusAGI:
    FONTES = {"livro_escola": 100.0, "professor": 50.0, "povo_falou": 5.0}

    def __init__(self, debug=False):
        self.stop_words = {
            "o", "a", "de", "que", "do", "da", "é", "em", "no", "na", "com", 
            "tem", "um", "uma", "sobre", "saber", "fale", "qual", "quem", "como"
        }
        self.engine = AutoLeiEngine()
        self.sub = SubconscienteCosmus(stop_words=self.stop_words)
        self.broca = AreaDeBroca() # Novo Lobo Frontal
        self.void = VoidMeritocratico(self.engine, self.sub)
        self.oculzer = Oculzer(self.sub)
        self.gen = Generalizado(self.sub, self.broca)
        self.debug = debug

    def normalizar(self, t):
        return "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn').replace("?", "").strip()

    def escutar(self, entrada, fonte="povo_falou"):
        t_start = time.perf_counter()
        clean = self.normalizar(entrada)
        tokens = [t for t in clean.split() if t not in self.stop_words and len(t) > 2]
        tokens_set = set(tokens)
        
        # A Broca "sente" o impacto térmico do input
        self.broca.sentir(tokens)

        eh_pergunta = entrada.strip().endswith('?') or any(clean.startswith(c) for c in ('quem', 'qual', 'como', 'sobre', 'fale'))

        if eh_pergunta:
            status, x_nec, x_apr = self.void.meditar(tokens)
            if status == "ZONA_SILENCIO":
                return "O nexo carece de solo no meu subconsciente."

            # Oculzer foca no sujeito e extrai predicados
            sujeito, predicados_brutos = self.oculzer.olhar_detalhes(tokens)
            
            # Generalizado sintetiza usando a modulação da Broca
            res = self.gen.sintetizar_agregado(sujeito, tokens_set, predicados_brutos, self.debug)
            
            if self.debug:
                t_end = time.perf_counter()
                st_info = f"T:{self.broca.st[0]:.1f}|S:{self.broca.st[1]:.1f}"
                return f"[FLOW: {(t_end-t_start)*1e6:.2f}μs | {st_info}]\n{res}"
            return res

        # Ingestão (Temporal/Medula)
        if len(tokens) >= 1:
            auth_val = self.FONTES.get(fonte, 5.0)
            self.sub.inicializar(entrada, autoridade=auth_val)
            for t in tokens: self.engine.pulsar(t, auth_val)
            return None if not self.debug else f"[Ingerido: {clean}]"
        
        return None

# ==================================================================
# 3. OCULZER & GENERALIZADOR (Análise e Síntese de Broca)
# ==================================================================
class Oculzer:
    def __init__(self, sub): self.sub = sub
    def olhar_detalhes(self, tokens_pergunta):
        if not tokens_pergunta: return None, {}
        # Pivota no token de maior massa crítica (Raridade + Conexão)
        sujeito = max(tokens_pergunta, key=lambda t: self.sub.pesos_raros.get(t, 0) + (0.1 * len(self.sub.neuronios.get(t, []))))
        frases = self.sub.neuronios.get(sujeito, [])
        predicados = {}
        for f in frases:
            auth = self.sub.autoridade_frase.get(f, 1.0)
            for t in self.sub.fatos_tokens.get(f, []):
                if t != sujeito and t not in tokens_pergunta:
                    peso = self.sub.pesos_raros.get(t, 1.0) * auth
                    if peso > predicados.get(t, 0): predicados[t] = peso
        return sujeito, predicados

class Generalizado:
    def __init__(self, sub, broca):
        self.sub = sub
        self.broca = broca

    def sintetizar_agregado(self, sujeito, tokens_set, predicados_dict, debug):
        if not sujeito: return "Dados insuficientes."
        
        # Pega a frase de maior autoridade (Espinha Dorsal)
        frases = self.sub.neuronios.get(sujeito, [])
        melhor_f = max(frases, key=lambda f: self.sub.autoridade_frase.get(f, 0))
        
        # Adjetivos extras (Nexos secundários)
        ordenados = sorted(predicados_dict.items(), key=lambda x: x[1], reverse=True)
        detalhes = [p[0] for p in ordenados[:3] if p[0] not in melhor_f]
        
        # Modulação da Broca (Voz Emocional)
        prefixo = self.broca.modular_voz()
        
        res = f"{prefixo}{melhor_f.capitalize()}."
        if detalhes:
            res += f" Notei conexões com {', '.join(detalhes[:-1]) + ' e ' + detalhes[-1] if len(detalhes)>1 else detalhes[0]}."
            
        return res

# ==================================================================
# 4. INFRAESTRUTURA (Subconsciente e Motor)
# ==================================================================
class SubconscienteCosmus:
    def __init__(self, stop_words):
        self.neuronios = defaultdict(list)
        self.autoridade_frase = {} 
        self.pesos_raros = {} 
        self.fatos_brutos = []
        self.fatos_tokens = {} 
        self.stop_words = stop_words

    def inicializar(self, txt, autoridade=1.0):
        f = txt.lower().replace(",", "").replace(".", "").strip()
        if f not in self.autoridade_frase:
            self.fatos_brutos.append(f)
            self.autoridade_frase[f] = autoridade
        else: self.autoridade_frase[f] += autoridade
        
        tokens = [t for t in f.split() if t not in self.stop_words and len(t) > 2]
        self.fatos_tokens[f] = tokens 
        for t in tokens:
            if f not in self.neuronios[t]: self.neuronios[t].append(f)
            q = sum(1 for fb in self.fatos_brutos if t in fb)
            self.pesos_raros[t] = 2.0 / (math.log(q + 1.2) + 1e-5)

class AutoLeiEngine:
    def __init__(self): self.frequencia_pulso = defaultdict(int)
    def pulsar(self, t, auth): self.frequencia_pulso[t] += int(auth)

class VoidMeritocratico:
    def __init__(self, engine, sub): self.engine, self.sub = engine, sub
    def meditar(self, tokens):
        if not tokens: return "VOID", 0, 1
        x_apr = sum(self.engine.frequencia_pulso.get(t, 0) for t in tokens) / (len(tokens) + 1e-5)
        return ("LIBERADO", 1, x_apr) if x_apr >= 1.0 else ("ZONA_SILENCIO", 0, x_apr)

# ==================================================================
# TESTE OPERACIONAL
# ==================================================================
if __name__ == "__main__":
    q = QuintikusAGI(debug=True)
    
    # Ingestão de conhecimento
    q.escutar("Maria tem cabelo preto", fonte="povo_falou")
    q.escutar("Maria tem cabelo andulado e cheiroso", fonte="povo_falou")
    q.escutar("Maria tem cheiro de rosas", fonte="professor")
    q.escutar("Maria é uma excelente programadora", fonte="livro_escola")
    
    print("\n--- Pergunta Normal ---")
    print(q.escutar("fale sobre a maria?"))

    print("\n--- Pergunta com 'Calor' (Erro) ---")
    # A palavra 'erro' deve subir a pressão térmica e mudar o prefixo
    print(q.escutar("qual cor do cabelo de maria?"))
