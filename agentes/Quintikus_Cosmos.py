import math
import random
import re
import os
import json
import unicodedata
from collections import Counter, defaultdict

# =====================================================================
# SETOR 1: UTILITÁRIOS E NORMALIZAÇÃO DE TEXTO
# =====================================================================
def normalizar(texto):
    texto_limpo = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return re.sub(r"\s+", " ", texto_limpo.lower().strip())

def dot(a, b): 
    return sum(x * y for x, y in zip(a, b))

def scale(a, s): 
    return [x * s for x in a]


# =====================================================================
# SETOR 2: GRAFO DE CORRELAÇÃO E APRENDIZADO DINÂMICO
# =====================================================================
class GrafoCorrelacao:
    def __init__(self):
        self.adjacencias = defaultdict(lambda: defaultdict(float))
        self.frequencia_termos = Counter()

    def aprender_sentenca(self, texto):
        tokens = [t for t in re.findall(r'\b\w+\b', normalizar(texto)) if len(t) > 2]
        for i, t1 in enumerate(tokens):
            self.frequencia_termos[t1] += 1
            for j in range(max(0, i-3), min(len(tokens), i+4)):
                if i != j:
                    t2 = tokens[j]
                    self.adjacencias[t1][t2] += 1.0

    def expandir_contexto(self, texto):
        tokens = [t for t in re.findall(r'\b\w+\b', normalizar(texto)) if len(t) > 2]
        expansao = list(tokens)
        for t in tokens:
            if t in self.adjacencias:
                vizinhos_ordenados = sorted(self.adjacencias[t].items(), key=lambda x: x[1], reverse=True)
                for vizinho, peso in vizinhos_ordenados[:2]:
                    if peso > 1.0 and vizinho not in expansao:
                        expansao.append(vizinho)
        return " ".join(expansao)


# =====================================================================
# SETOR 3: VETORIZADOR ESPACIAL COM SUPORTE AO GRAFO
# =====================================================================
class VetorizadorEspacial:
    def __init__(self, grafo, dimensao=256):
        self.dimensao = dimensao
        self.grafo = grafo

    def vetorizar(self, texto, usar_grafo=True):
        texto_processado = self.grafo.expandir_contexto(texto) if usar_grafo else texto
        v = [0.0] * self.dimensao
        s = normalizar(texto_processado)
        tokens = re.findall(r"[a-záéíóúãõâêôç0-9]+", s)
        unidades = tokens + [s[i:i+3] for i in range(max(0, len(s)-2))]
        
        for u in unidades:
            h = 2166136261
            for c in u:
                h = ((h ^ ord(c)) * 16777619) & 0xffffffff
            i = h % self.dimensao
            v[i] += 1.0
            j = (h // self.dimensao) % self.dimensao
            v[j] -= 0.35
            
        norma = math.sqrt(dot(v, v)) or 1.0
        return scale(v, 1.0 / norma)


# =====================================================================
# SETOR 4: MOTOR LÓGICO-MATEMÁTICO (Sistema 1 Exato)
# =====================================================================
class ProcessadorLogicoMatematico:
    def __init__(self):
        self.palavras_num = {
            'um': 1, 'uma': 1, 'dois': 2, 'duas': 2, 'tres': 3, 'três': 3,
            'quatro': 4, 'cinco': 5, 'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9, 'dez': 10,
            'quinze': 15, 'vinte': 20, 'trinta': 30, 'quarenta': 40, 'cinquenta': 50
        }
        self.subtracao_termos = {'perdi', 'perdeu', 'tirou', 'menos', 'morreu', 'gastou', 'vendeu', 'vendi'}
        self.adicao_termos = {'ganhou', 'comprou', 'comprei', 'ganhei', 'mais', 'somou', 'juntou', 'adotou', 'adotei'}

    def avaliar(self, pergunta):
        p_lower = pergunta.lower()
        tokens = re.findall(r'\b\w+\b', p_lower)
        valores = []
        for t in tokens:
            if t.isdigit():
                valores.append(int(t))
            elif t in self.palavras_num:
                valores.append(self.palavras_num[t])

        if len(valores) >= 2:
            tem_sub = any(termo in tokens for termo in self.subtracao_termos)
            tem_add = any(termo in tokens for termo in self.adicao_termos)
            if tem_sub and not tem_add:
                res = valores[0] - valores[1]
                return True, f"Cálculo lógico aplicado (subtração): {valores[0]} - {valores[1]} = {res}.", res
            elif tem_add and not tem_sub:
                res = valores[0] + valores[1]
                return True, f"Cálculo lógico aplicado (adição): {valores[0]} + {valores[1]} = {res}.", res
        return False, "", None


# =====================================================================
# SETOR 5: MOTOR COSMOS (Camada 1 Linear & Camada 2 Não-Linear)
# =====================================================================
class Cosmos:
    def __init__(self, peso_risco=0.20, limiar_aceitacao=0.30):
        self.feedbacks = []
        self.peso_linear = 0.65
        self.bias_linear = 0.10
        self.peso_risco = peso_risco
        self.limiar_aceitacao = limiar_aceitacao

    def camada_1_linear(self, similaridade):
        score_linear = (similaridade * self.peso_linear) + self.bias_linear
        return min(1.0, max(0.0, score_linear))

    def camada_2_nao_linear(self, score_linear, margem, qtd_candidatos, entropia_media):
        peso_fb = sum(self.feedbacks) * 0.04
        risco_divergente = entropia_media * self.peso_risco
        densidade_dados = min(1.0, qtd_candidatos / 10.0)
        
        conf_final = score_linear + peso_fb + (densidade_dados * 0.1) - risco_divergente
        conf_final = min(1.0, max(0.0, conf_final))
        
        if conf_final >= self.limiar_aceitacao:
            status = "cosmos_aceito"
        elif conf_final >= (self.limiar_aceitacao - 0.08):
            status = "cosmos_incerto"
        else:
            status = "rejeitado_por_entropia"
            
        return status, conf_final, risco_divergente

    def processar(self, similaridade, margem, qtd_candidatos, entropia_media):
        s_linear = self.camada_1_linear(similaridade)
        status, conf_final, risco = self.camada_2_nao_linear(s_linear, margem, qtd_candidatos, entropia_media)
        return status, conf_final, s_linear, risco

    def registrar_feedback(self, positivo):
        self.feedbacks.append(1 if positivo else -1)


# =====================================================================
# SETOR 6: CONTEXTO ENTRÓPICO E MEMÓRIA GLOBAL
# =====================================================================
class ContextoEntropico:
    def __init__(self):
        self.documentos = []
        self.freq_palavras = Counter()
        self.total_documentos = 0

    def adicionar(self, texto):
        palavras = [p for p in re.findall(r'\b\w+\b', texto.lower()) if len(p) > 1]
        if not palavras: return
        self.documentos.append(set(palavras))
        self.total_documentos += 1
        self.freq_palavras.update(palavras)

    def entropia_palavra(self, palavra):
        if self.total_documentos == 0: return 1.0
        aparece_em = sum(1 for doc in self.documentos if palavra in doc)
        if aparece_em == 0: return 1.0
        p = aparece_em / self.total_documentos
        q = 1 - p
        if p == 0 or q == 0: return 0.0
        return -p * math.log2(p) - q * math.log2(q)


class MemoriaGlobal:
    def __init__(self):
        self.dados = []
        self.indice = defaultdict(list)

    def adicionar(self, entrada, resposta):
        idx = len(self.dados)
        self.dados.append({'entrada': entrada, 'resposta': resposta})
        for palavra in set(re.findall(r'\b\w+\b', entrada.lower())):
            self.indice[palavra].append(idx)

    def buscar_fallback(self, pergunta):
        tokens = [t for t in re.findall(r'\b\w+\b', pergunta.lower()) if len(t) > 2]
        if not tokens or not self.dados: return None, 0.0
        scores = defaultdict(float)
        total_docs = len(self.dados)
        for palavra in tokens:
            idxs = self.indice.get(palavra, [])
            if not idxs: continue
            idf = math.log((total_docs + 1) / (len(idxs) + 1)) + 1
            for idx in idxs: scores[idx] += idf
        if not scores: return None, 0.0
        melhor_idx = max(scores, key=scores.get)
        score_norm = min(1.0, scores[melhor_idx] / (len(tokens) * 2.0))
        return self.dados[melhor_idx]['resposta'], score_norm


# =====================================================================
# SETOR 7: PERCEPÇÃO E CONSCIÊNCIA (Sem camada de saudação)
# =====================================================================
class Percepcao:
    def processar(self, texto):
        t_lower = texto.lower()
        if any(w in t_lower for w in ['perdi', 'ganhou', 'menos', 'mais', 'somou', 'perdeu', 'tirou', 'gastou']) and bool(re.search(r'\d+', t_lower)):
            return "operacional_matematico"
        elif "?" in t_lower or any(w in t_lower for w in ['qual', 'quem', 'como', 'onde', 'quanto']):
            return "pergunta_direta"
        return "declarativo"


class Consciencia:
    def arbitrar(self, confianca_cosmos, risco, intencao):
        if intencao == "operacional_matematico": return "interceptacao_matematica"
        
        # Proteção rigorosa de risco (> 0.15 recua)
        if risco > 0.15 or confianca_cosmos < 0.35: 
            return "defensivo_recuar"
            
        return "deliberativo_aceito"


# =====================================================================
# SETOR 8: CAIXA DE VIDRO (Auditoria Cognitiva em Tempo Real)
# =====================================================================
class CaixaDeVidro:
    def __init__(self):
        self.resetar()

    def resetar(self):
        self.entrada = ""
        self.intencao = ""
        self.modo = ""
        self.decisao = ""
        self.confianca = 0.0
        self.risco = 0.0
        self.comparacoes = []

    def exibir(self):
        print("\n" + "="*70)
        print(" CAIXA DE VIDRO (IPE v9 — Risco Rigoroso & Cosmos Dual-Layer)")
        print("="*70)
        print(f"Entrada: {self.entrada}")
        print(f"Intenção: {self.intencao} | Modo Consciência: {self.modo}")
        print(f"Decisão: {self.decisao} (Confiança Cosmos: {self.confianca*100:.1f}%, Risco: {self.risco*100:.1f}%)")
        print("Comparação de evidências:")
        for score, texto in self.comparacoes:
            print(f"  - {score*100:.1f}%: {texto}")
        print("="*70)


# =====================================================================
# SETOR 9: NÚCLEO IPE E ROTEADOR DE FLUXO
# =====================================================================
class GeoCache:
    def __init__(self, arquivo='mapa_mental.json'):
        self.arquivo = arquivo
        self.entradas = []
        self.carregar()

    def carregar(self):
        if os.path.exists(self.arquivo):
            with open(self.arquivo, 'r', encoding='utf-8') as f:
                self.entradas = json.load(f)
        else:
            self.entradas = []

    def salvar(self):
        with open(self.arquivo, 'w', encoding='utf-8') as f:
            json.dump(self.entradas, f, ensure_ascii=False, indent=2)

    def adicionar(self, texto):
        if not any(e['texto'] == texto for e in self.entradas):
            self.entradas.append({'texto': texto})
            self.salvar()

    def todas_frases(self):
        return [e['texto'] for e in self.entradas]


class IPE:
    def __init__(self):
        self.geocache = GeoCache()
        self.grafo = GrafoCorrelacao()
        self.vetorizador = VetorizadorEspacial(self.grafo, dimensao=256)
        self.cosmos = Cosmos()
        self.logica = ProcessadorLogicoMatematico()
        self.contexto = ContextoEntropico()
        self.memoria = MemoriaGlobal()
        self.percepcao = Percepcao()
        self.consciencia = Consciencia()
        self.box = CaixaDeVidro()

    def aprender(self, texto_grande):
        for frase in re.split(r'(?<=[.!?])\s+', texto_grande):
            f = frase.strip()
            if len(f) >= 5:
                if f not in self.geocache.todas_frases():
                    self.geocache.adicionar(f)
                self.grafo.aprender_sentenca(f)
                self.contexto.adicionar(f)


class Rote:
    def __init__(self, ipe):
        self.ipe = ipe

    def processar(self, pergunta):
        intencao = self.ipe.percepcao.processar(pergunta)
        self.ipe.contexto.adicionar(pergunta)
        self.ipe.grafo.aprender_sentenca(pergunta)

        # Intercepção do Sistema 1 (Matemática Exata)
        resolvido, texto_calc, resultado = self.ipe.logica.avaliar(pergunta)
        if resolvido:
            self.ipe.box.resetar()
            self.ipe.box.entrada = pergunta
            self.ipe.box.intencao = intencao
            self.ipe.box.modo = "interceptacao_matematica"
            self.ipe.box.confianca = 1.0
            self.ipe.box.risco = 0.0
            self.ipe.box.decisao = "motor_matematico"
            self.ipe.box.exibir()
            return f"🧮 **Sistema 1 (Exato):**\n{texto_calc}"

        # Consulta Vetorial Baseada em Grafos Dinâmicos e Cosmos
        frases = self.ipe.geocache.todas_frases()
        v_perg = self.ipe.vetorizador.vetorizar(pergunta, usar_grafo=True)
        candidatos = []
        for f in frases:
            v_f = self.ipe.vetorizador.vetorizar(f, usar_grafo=True)
            score = dot(v_perg, v_f)
            candidatos.append((score, f))

        candidatos.sort(key=lambda x: x[0], reverse=True)
        melhor_score, melhor_frase = candidatos[0] if candidatos else (0.0, "")
        margem = (melhor_score - candidatos[1][0]) if len(candidatos) > 1 else 0.0
        
        entropia_media = sum(self.ipe.contexto.entropia_palavra(p) for p in pergunta.split()) / max(1, len(pergunta.split()))

        status_cosmos, conf_cosmos, s_linear, risco = self.ipe.cosmos.processar(
            melhor_score, margem, len(candidatos), entropia_media
        )
        modo = self.ipe.consciencia.arbitrar(conf_cosmos, risco, intencao)

        self.ipe.box.resetar()
        self.ipe.box.entrada = pergunta
        self.ipe.box.intencao = intencao
        self.ipe.box.modo = modo
        self.ipe.box.confianca = conf_cosmos
        self.ipe.box.risco = risco
        self.ipe.box.comparacoes = candidatos[:3]
        self.ipe.box.exibir()

        # Validação restrita: Se o modo for defensivo, bloqueia o aceite direto
        if status_cosmos == "cosmos_aceito" and modo != "defensivo_recuar":
            self.ipe.memoria.adicionar(pergunta, melhor_frase)
            return f"Evidência mais próxima (Cosmos + Grafo): [{melhor_frase}]"
        else:
            resp_mem, score_mem = self.ipe.memoria.buscar_fallback(pergunta)
            if resp_mem and score_mem > 0.18:
                return f"🧠 **Memória Global:** {resp_mem}"
            return "🤔 **Cosmos (Defensivo):** Risco elevado ou incerteza detectada. Ação bloqueada por segurança cognitiva."


# =====================================================================
# SETOR 10: EXECUÇÃO DO SISTEMA
# =====================================================================
if __name__ == "__main__":
    random.seed(42)
    BASE_INICIAL = """
    se tiver inimigo perto e vida alta eu atiro.
    se tiver inimigo perto e vida baixa eu recuo.
    se tiver inimigo longe e vida alta eu avanço.
    se tiver inimigo longe e vida baixa eu me escondo.
    se tiver dois inimigos eu ataco o mais fraco primeiro.
    se tiver três inimigos eu uso granada.
    se tiver muitos inimigos eu corro.
    se tiver inimigo armado eu procuro cobertura.
    se tiver inimigo desarmado eu ataco direto.
    se tiver inimigo rápido eu atiro de longe.
    se tiver inimigo lento eu ataco de perto.
    se tiver chefe e vida cheia eu ataco.
    se tiver chefe e vida baixa eu procuro cura.
    se tiver munição cheia eu atiro sem parar.
    se tiver munição baixa eu uso pistola.
    se tiver sem munição eu fujo.
    se tiver arma forte eu uso no chefe.
    se tiver arma fraca eu uso nos fracos.
    se tiver granada e inimigo agrupado eu jogo.
    se tiver bomba e caminho bloqueado eu explodo.
    se tiver armadura cheia eu avanço.
    se tiver armadura baixa eu recuo.
    se tiver vida cheia e armadura cheia eu ataco tudo.
    se tiver vida baixa e armadura baixa eu fujo.
    se tiver poção eu guardo pra emergência.
    se tiver vida crítica eu bebo poção.
    se tiver mana cheia eu uso habilidade.
    """
    
    ipe = IPE()
    ipe.aprender(BASE_INICIAL)
    roteador = Rote(ipe)

    print("\nIPE Sinergia v9.2 (Sem Saudações — Lógica Pura & Risco Ativo).")
    print("Comandos: :cosmos | :feedback sim/nao | sair")
    
    while True:
        try:
            perq = input("\nVocê > ").strip()
        except:
            break
        if not perq or perq.lower() == 'sair': break
        
        if perq.lower().startswith(":feedback"):
            partes = perq.split()
            positivo = len(partes) > 1 and partes[1].lower() in ['sim', 's', 'true']
            ipe.cosmos.registrar_feedback(positivo)
            print("Feedback registrado no motor Cosmos.")
            continue
            
        print(f"IPE > {roteador.processar(perq)}")
