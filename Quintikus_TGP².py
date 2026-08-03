import numpy as np
import re
import sys
import time
import math
from collections import defaultdict, Counter

# ============================================================
# 1. GEOMETRIA HIPERBÓLICA (NÃO-LINEAR)
# ============================================================
class DiscoPoincare:
    def __init__(self, dim=128):
        self.dim = dim

    def mobius_add(self, x, y):
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return np.zeros(self.dim)
        nx2 = np.clip(np.dot(x, x), 0.0, 0.99)
        ny2 = np.clip(np.dot(y, y), 0.0, 0.99)
        xy  = np.clip(np.dot(x, y), -0.99, 0.99)
        num = (1 + 2 * xy + ny2) * x + (1 - nx2) * y
        den = 1 + 2 * xy + nx2 * ny2 + 1e-10
        res = num / den
        res_sq = np.dot(res, res)
        if res_sq >= 0.9801:
            res = (res / (np.sqrt(res_sq) + 1e-10)) * 0.985
        return res if np.isfinite(res).all() else np.zeros(self.dim)

    def distancia(self, x, y):
        norma_x = np.clip(np.linalg.norm(x), 0, 0.985)
        norma_y = np.clip(np.linalg.norm(y), 0, 0.985)
        diff_sq = np.dot(x - y, x - y)
        num = 2 * diff_sq
        den = (1 - norma_x**2) * (1 - norma_y**2) + 1e-10
        return np.arccosh(np.clip(1 + num / den, 1.0, 1e6))

# ============================================================
# 2. PROBABILIDADE LINEAR (CADEIA DE MARKOV / REGÊNCIA)
# ============================================================
class ConexaoLinear:
    """Mapeia a probabilidade de transição sequencial entre palavras (Linha Temporal)."""
    def __init__(self):
        self.transicoes = defaultdict(lambda: defaultdict(float))
        self.trigramas = defaultdict(list)

    def registrar_fluxo(self, tokens):
        for i in range(len(tokens) - 1):
            atual, prox = tokens[i], tokens[i+1]
            self.transicoes[atual][prox] += 1.0
            if i < len(tokens) - 2:
                self.trigramas[(atual, prox)].append(tokens[i+2])

        # Normaliza probabilidades lineares
        for t1, contagens in self.transicoes.items():
            soma = sum(contagens.values())
            for t2 in contagens:
                self.transicoes[t1][t2] /= soma

    def prob_linear(self, t_atual, t_candidato, t_penultimo=None):
        p_bigrama = self.transicoes[t_atual].get(t_candidato, 0.0001)
        p_trigrama = 0.0
        if t_penultimo and (t_penultimo, t_atual) in self.trigramas:
            if t_candidato in self.trigramas[(t_penultimo, t_atual)]:
                p_trigrama = 0.5
        return p_bigrama + p_trigrama

# ============================================================
# 3. ATENÇÃO COGNITIVA + DUALIDADE PROBABILÍSTICA
# ============================================================
class GeoAtencaoCognitivaRefinada:
    def __init__(self, disco):
        self.disco = disco

    def focar_pensamento(self, tokens_contexto, token_para_vetor):
        n = len(tokens_contexto)
        if n == 0: return np.zeros(self.disco.dim)

        vetores_contexto = [token_para_vetor.get(t, np.zeros(self.disco.dim)) for t in tokens_contexto]
        pensamento = vetores_contexto[-1]
        
        # Ruminação hiperbólica
        for _ in range(2):
            pesos = np.array([-self.disco.distancia(pensamento, v) for v in vetores_contexto])
            exp_pesos = np.exp(pesos - np.max(pesos))
            prob_atencao = exp_pesos / np.sum(exp_pesos)
            
            novo_pensamento = np.zeros(self.disco.dim)
            for i, v in enumerate(vetores_contexto):
                novo_pensamento = self.disco.mobius_add(novo_pensamento, v * prob_atencao[i])
            pensamento = novo_pensamento

        meio = max(1, n // 2)
        polo_sujeito = np.zeros(self.disco.dim)
        for v in vetores_contexto[:meio]:
            polo_sujeito = self.disco.mobius_add(polo_sujeito, v * (1.0 / len(vetores_contexto[:meio])))

        polo_predicativo = np.zeros(self.disco.dim)
        for v in vetores_contexto[meio:]:
            polo_predicativo = self.disco.mobius_add(polo_predicativo, v * (1.0 / len(vetores_contexto[meio:])))

        tensao_logica = self.disco.mobius_add(-polo_sujeito, polo_predicativo)
        fator_tensao = 0.20 if tokens_contexto[-1] in {'?', 'que', 'qual'} else 0.40
        return self.disco.mobius_add(pensamento, tensao_logica * fator_tensao)

# ============================================================
# 4. AGENTE NEURAL COM COEXISTÊNCIA LINEAR E NÃO-LINEAR
# ============================================================
class AgenteNeuralGeometrico:
    def __init__(self, dim_espaco=128):
        self.dim_espaco = dim_espaco
        self.disco = DiscoPoincare(dim_espaco)
        self.conexao_linear = ConexaoLinear()
        self.geo_atencao = GeoAtencaoCognitivaRefinada(self.disco)
        self.token_para_vetor = {}
        self.tokens_lista = []

    def _registrar_token(self, token):
        if token not in self.token_para_vetor:
            self.token_para_vetor[token] = (np.random.rand(self.dim_espaco) - 0.5) * 0.05
            self.tokens_lista.append(token)

    def tokenizar(self, texto):
        import unicodedata
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acento = ''.join([c for c in texto_nfkd if unicodedata.category(c) != 'Mn'])
        tokens = re.findall(r'[a-z0-9]+|[.,!?;:]+', texto_sem_acento)
        return [t for t in tokens if t.strip()]

    def aprender_e_devorar(self, base_texto, epocas=35, taxa_aprendizado=0.04):
        tokens = self.tokenizar(base_texto)
        self.conexao_linear.registrar_fluxo(tokens)
        
        for t in tokens:
            self._registrar_token(t)

        print(f"🧠 Treinando Espaço Vetorial e Conexões Lineares ({epocas} épocas)...")
        vocab_size = len(self.tokens_lista)

        for epoca in range(epocas):
            for i in range(len(tokens) - 1):
                t_atual, t_prox = tokens[i], tokens[i+1]
                v_atual = self.token_para_vetor[t_atual]
                v_prox = self.token_para_vetor[t_prox]

                direcao_atracao = (v_prox - v_atual) * taxa_aprendizado
                self.token_para_vetor[t_atual] = self.disco.mobius_add(v_atual, direcao_atracao)

                idx_neg = np.random.randint(0, vocab_size)
                t_neg = self.tokens_lista[idx_neg]
                if t_neg not in (t_atual, t_prox):
                    v_neg = self.token_para_vetor[t_neg]
                    direcao_repulsao = -(v_neg - v_atual) * (taxa_aprendizado * 0.5)
                    self.token_para_vetor[t_atual] = self.disco.mobius_add(
                        self.token_para_vetor[t_atual], direcao_repulsao
                    )

        print(f"✅ Consciência Espacial + Regência Linear Prontas!\n")

    def decidir_e_responder(self, estimulo, max_tokens=18, temperatura=0.25):
        tokens_iniciais = self.tokenizar(estimulo)
        if not tokens_iniciais: return

        contexto = tokens_iniciais.copy()
        gerados = []

        # Avaliação Inicial de Entropia Global
        vetor_intencao = self.geo_atencao.focar_pensamento(contexto, self.token_para_vetor)
        norma = np.linalg.norm(vetor_intencao)
        prob_coerencia_global = 1.0 / (1.0 + math.exp(-norma * 4.0))
        h_entropia = -prob_coerencia_global * math.log2(prob_coerencia_global) - (1-prob_coerencia_global)*math.log2(1-prob_coerencia_global)

        print(f"📊 [ESTADO DE COEXISTÊNCIA DAS CAMADAS]")
        print(f"   ├─ Entropia Hiperbólica (Dúvida H): {h_entropia:.4f}")
        print(f"   └─ Coerência Semântica Não-Linear:  {prob_coerencia_global*100:.2f}%")

        for step in range(max_tokens):
            t_ultimo = contexto[-1]
            t_penultimo = contexto[-2] if len(contexto) >= 2 else None

            if step >= 3 and t_ultimo in {'.', '!', '?'}:
                break

            vetor_intencao = self.geo_atencao.focar_pensamento(contexto, self.token_para_vetor)
            candidatos, logits = [], []

            for token, v_ref in self.token_para_vetor.items():
                if token == t_ultimo: continue

                # 1. PROBABILIDADE NÃO-LINEAR (Geometria Hiperbólica Global)
                dist_hiperbolica = self.disco.distancia(vetor_intencao, v_ref)
                p_nao_linear = math.exp(-dist_hiperbolica)

                # 2. PROBABILIDADE LINEAR (Regência de Transição Local)
                p_linear = self.conexao_linear.prob_linear(t_ultimo, token, t_penultimo)

                # 🧠 COEXISTÊNCIA DUAL: O token só é forte se LINEAR e NÃO-LINEAR concordarem!
                # Multiplicação das Probabilidades (Interseção das Probabilidades de Sentido)
                p_coexistencia = (p_nao_linear ** 0.6) * (p_linear ** 1.4)

                logit = math.log(p_coexistencia + 1e-9)

                if token in gerados:
                    logit -= 2.5

                candidatos.append(token)
                logits.append(logit)

            if not candidatos: break

            logits = np.array(logits) / temperatura
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            proximo_token = np.random.choice(candidatos, p=probs)
            contexto.append(proximo_token)
            gerados.append(proximo_token)
            yield proximo_token

# ============================================================
# EXECUÇÃO DO NOVO SISTEMA DUAL
# ============================================================
if __name__ == "__main__":
    agente = AgenteNeuralGeometrico(dim_espaco=128)

    dataset = """
    A inteligencia artificial e um campo da ciencia da computacao. O aprendizado de maquina 
    permite que os sistemas identifiquem padroes em grandes volumes de dados. No espaco 
    quantico, a informacao existe em multiplos estados simultaneamente.
    Pra beber muita agua precisa de copo grande. Pra beber pouca agua usa copo pequeno.
    eu nao sei o seu nome, mas o meu nome e tgp2.
    você for o usuario ronan,seu nome é ronan?
    """

    agente.aprender_e_devorar(dataset, epocas=35)

    testes = [
        "qual e o meu nome",
        "pra beber muita agua",
        "No espaco quantico"
    ]

    for t in testes:
        print(f"🙋 Estímulo (Input): '{t}'")
        print("🤖 Agente TGP-2: ", end="")
        for tok in agente.decidir_e_responder(t):
            sys.stdout.write(tok + " ")
            sys.stdout.flush()
            time.sleep(0.03)
        print("\n" + "=" * 50)
