import numpy as np
import re
import sys
import time
from collections import defaultdict

# ============================================================
# GEO-ESPAÇO PROBABILÍSTICO VETORIAL QUÂNTICO (TGP-2 COGNITIVO)
# ============================================================

class DiscoPoincare:
    """Geometria Hiperbólica de Alta Estabilidade e Desempenho."""
    def __init__(self, dim=128): # <-- Dimensão aumentada para 128
        self.dim = dim

    def mobius_add(self, x, y):
        """
        Adição de Möbius no Disco de Poincaré.
        Otimizada para evitar NaN e overflow numérico.
        """
        # 1. Validação de entrada
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return np.zeros(self.dim)

        # 2. Produtos internos com clipping para evitar overflow
        nx2 = np.clip(np.dot(x, x), 0.0, 0.99)
        ny2 = np.clip(np.dot(y, y), 0.0, 0.99)
        xy  = np.clip(np.dot(x, y), -0.99, 0.99)

        # 3. Numerador e denominador (com epsilon extra)
        num = (1 + 2 * xy + ny2) * x + (1 - nx2) * y
        den = 1 + 2 * xy + nx2 * ny2 + 1e-10

        # 4. Divisão
        res = num / den

        # 5. Trava geométrica (com checagem de norma)
        res_sq = np.dot(res, res)
        if res_sq >= 0.9801:  # 0.99²
            res = (res / (np.sqrt(res_sq) + 1e-10)) * 0.985

        # 6. Validação final
        if not np.isfinite(res).all():
            return np.zeros(self.dim)

        return res

    def distancia(self, x, y):
        norma_x = np.clip(np.linalg.norm(x), 0, 0.985)
        norma_y = np.clip(np.linalg.norm(y), 0, 0.985)
        
        diff_sq = np.dot(x - y, x - y)
        num = 2 * diff_sq
        den = (1 - norma_x**2) * (1 - norma_y**2) + 1e-10
        
        return np.arccosh(np.clip(1 + num / den, 1.0, 1e6))


class GeoAtencaoCognitiva:
    """Atenção Hiperbólica em 3 Camadas de Raciocínio (Ruminação, Estrutura e Síntese)."""
    def __init__(self, disco):
        self.disco = disco

    def focar_pensamento(self, tokens_contexto, token_para_vetor, iteracoes_ruminacao=2):
        n = len(tokens_contexto)
        if n == 0:
            return np.zeros(self.disco.dim)

        vetores_contexto = [token_para_vetor.get(t, np.zeros(self.disco.dim)) for t in tokens_contexto]
        
        # ========================================================
        # CAMADA 1: REPETIÇÃO (Ruminação Geométrica)
        # ========================================================
        pensamento = vetores_contexto[-1]
        
        for _ in range(iteracoes_ruminacao):
            pesos = np.zeros(n)
            for i, v in enumerate(vetores_contexto):
                dist = self.disco.distancia(pensamento, v)
                fator_volume = 1.0 + (tokens_contexto.count(tokens_contexto[i]) * 0.2) + (i / n)
                pesos[i] = -(dist / fator_volume)
            
            exp_pesos = np.exp(pesos - np.max(pesos))
            prob_atencao = exp_pesos / np.sum(exp_pesos)
            
            novo_pensamento = np.zeros(self.disco.dim)
            for i, v in enumerate(vetores_contexto):
                v_ponderado = v * prob_atencao[i]
                novo_pensamento = self.disco.mobius_add(novo_pensamento, v_ponderado)
            
            pensamento = novo_pensamento

        # ========================================================
        # CAMADA 2: SEPARAÇÃO DE CONCEITO (Sujeito vs Predicativo)
        # ========================================================
        meio = max(1, n // 2)
        
        polo_sujeito = np.zeros(self.disco.dim)
        for v in vetores_contexto[:meio]:
            polo_sujeito = self.disco.mobius_add(polo_sujeito, v * (1.0 / len(vetores_contexto[:meio])))
            
        polo_predicativo = np.zeros(self.disco.dim)
        for v in vetores_contexto[meio:]:
            polo_predicativo = self.disco.mobius_add(polo_predicativo, v * (1.0 / len(vetores_contexto[meio:])))

        tensao_logica = self.disco.mobius_add(-polo_sujeito, polo_predicativo)

        # ========================================================
        # CAMADA 3: AGRUPAMENTO COM BASE NO INPUT (Síntese)
        # ========================================================
        vetor_intencao_final = self.disco.mobius_add(pensamento, tensao_logica * 0.4)

        return vetor_intencao_final


class ProbabilidadeLinearPares:
    def __init__(self):
        self.pares_diretos = defaultdict(lambda: defaultdict(float))
        self.trilhas_contexto = defaultdict(list)

    def registrar_fluxo(self, tokens):
        for i in range(len(tokens) - 1):
            atual, prox = tokens[i], tokens[i+1]
            self.pares_diretos[atual][prox] += 1.0
            
            if i < len(tokens) - 2:
                self.trilhas_contexto[(atual, prox)].append(tokens[i+2])

        for t1, contagens in self.pares_diretos.items():
            soma = sum(contagens.values())
            for t2 in contagens:
                self.pares_diretos[t1][t2] /= soma


class TGP2:
    def __init__(self, dim_espaco=128): # <-- Dimensão aumentada para 128
        self.dim_espaco = dim_espaco
        self.disco = DiscoPoincare(dim_espaco)
        self.geo_atencao = GeoAtencaoCognitiva(self.disco) # <-- Arquitetura Cognitiva de 3 Camadas
        self.prob_pares = ProbabilidadeLinearPares()
        self.token_para_vetor = {}
        self.tokens_lista = []
        
    def _registrar_token(self, token):
        if token not in self.token_para_vetor:
            vetor = (np.random.rand(self.dim_espaco) - 0.5) * 0.05
            self.token_para_vetor[token] = vetor
            self.tokens_lista.append(token)

    def tokenizar(self, texto):
        import unicodedata
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acento = ''.join([c for c in texto_nfkd if unicodedata.category(c) != 'Mn'])
        tokens = re.findall(r'[a-z0-9]+|[.,!?;:]+', texto_sem_acento)
        return [t for t in tokens if t.strip()]

    def devorar_texto(self, texto_bruto, epocas=30, taxa_aprendizado=0.04, k_negativos=4):
        tokens = self.tokenizar(texto_bruto)
        for t in tokens:
            self._registrar_token(t)
            
        print(f"📚 Treinando espaço hiperbólico blindado ({epocas} épocas)...")
        self.prob_pares.registrar_fluxo(tokens)
        vocab_size = len(self.tokens_lista)

        for epoca in range(epocas):
            for i in range(len(tokens) - 1):
                t_atual, t_prox = tokens[i], tokens[i+1]
                v_atual = self.token_para_vetor[t_atual]
                v_prox = self.token_para_vetor[t_prox]
                
                # Atração Hiperbólica
                direcao_atracao = (v_prox - v_atual) * taxa_aprendizado
                self.token_para_vetor[t_atual] = self.disco.mobius_add(v_atual, direcao_atracao)

                # Repulsão (Negative Sampling)
                for _ in range(k_negativos):
                    idx_neg = np.random.randint(0, vocab_size)
                    t_negativ = self.tokens_lista[idx_neg]
                    
                    if t_negativ != t_prox and t_negativ != t_atual:
                        v_neg = self.token_para_vetor[t_negativ]
                        direcao_repulsao = -(v_neg - v_atual) * (taxa_aprendizado * 0.5)
                        self.token_para_vetor[t_atual] = self.disco.mobius_add(
                            self.token_para_vetor[t_atual], direcao_repulsao
                        )

        print(f"✅ Concluído! {vocab_size} tokens organizados topologicamente.\n")

    def medir_distancia(self, token_a, token_b):
        tok_a = self.tokenizar(token_a)[0]
        tok_b = self.tokenizar(token_b)[0]
        if tok_a in self.token_para_vetor and tok_b in self.token_para_vetor:
            v_a = self.token_para_vetor[tok_a]
            v_b = self.token_para_vetor[tok_b]
            return self.disco.distancia(v_a, v_b)
        return None

    def gerar_stream(self, texto_entrada, max_tokens=18, temperatura=0.35, penalidade_repeticao=2.5):
        tokens_iniciais = self.tokenizar(texto_entrada)
        if not tokens_iniciais: 
            return
        
        contexto = tokens_iniciais.copy()
        gerados = []
        pontuacoes = {'.', ',', '!', '?', ';', ':'}
        pontos_finais = {'.', '!', '?'}

        for step in range(max_tokens):
            t_ultimo = contexto[-1]
            t_penultimo = contexto[-2] if len(contexto) >= 2 else None
            
            if step >= 3 and t_ultimo in pontos_finais:
                break

            # 🧠 O MODELO PENSA: Gera o Vetor de Intenção Sintética
            vetor_pensamento = self.geo_atencao.focar_pensamento(contexto, self.token_para_vetor)

            candidatos, logits = [], []

            for token, v_ref in self.token_para_vetor.items():
                if token == t_ultimo:
                    continue
                if token in pontuacoes and t_ultimo in pontuacoes:
                    continue

                # 1. Score de Atenção Geométrico (Distância até o PENSAMENTO GLOBAL SINTÉTICO)
                dist_pensamento = self.disco.distancia(vetor_pensamento, v_ref)
                score_geom = -dist_pensamento * 1.5

                # 2. Score Probabilístico de Regência (Bigrama)
                score_par = self.prob_pares.pares_diretos[t_ultimo].get(token, 0.0) * 4.0
                
                # 3. Score de Trilha (Trigrama)
                score_trilha = 0.0
                if t_penultimo and (t_penultimo, t_ultimo) in self.prob_pares.trilhas_contexto:
                    if token in self.prob_pares.trilhas_contexto[(t_penultimo, t_ultimo)]:
                        score_trilha = 6.0

                logit_total = score_geom + score_par + score_trilha

                # 4. Penalidade de Repetição Dinâmica
                if token in gerados:
                    logit_total -= (penalidade_repeticao * gerados.count(token))

                candidatos.append(token)
                logits.append(logit_total)

            if not candidatos: 
                break

            logits = np.array(logits) / temperatura
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            proximo_token = np.random.choice(candidatos, p=probs)
            
            contexto.append(proximo_token)
            gerados.append(proximo_token)
            yield proximo_token

# ============================================================
# EXECUÇÃO E TESTES
# ============================================================
if __name__ == "__main__":
    modelo = TGP2(dim_espaco=128) # <-- Dimensão aumentada no setup de inicialização

    base_conhecimento = """
    A inteligencia artificial e um campo da ciencia da computacao. O aprendizado de maquina 
    permite que os sistemas identifiquem padroes em grandes volumes de dados. No espaco 
    quantico, a informacao existe em multiplos estados simultaneamente. Quando uma medicao 
    ocorre, a funcao de onda entra em colapso. Modelos vetoriais hiperbolicos organizam a 
    linguagem humana como uma arvore, onde conceitos centrais ficam no meio e conceitos 
    especificos habitam as bordas. A arquitetura tgp tenta unir a fisica quantica e a 
    geometria nao euclidiana para processamento natural de texto em dispositivos limitados.
    oi, tudo bem, estou bem, espero que voce esteja bem hoje.
    qual meu nome? meu nome e tgp2, eu sou modelo de linguagem geometrica!
    eu sou o tgp2, esse meu nome tgp2.caso um pergutando for qual é meu nome? eu devo fala eu não sei o seu nome,mas meu nome é tgp2
    """

    print("🧠 Inicializando treinamento...")
    modelo.devorar_texto(base_conhecimento, epocas=30, k_negativos=4)

    print("📐 --- [INSPEÇÃO DA GEOMETRIA HIPERBÓLICA] ---")
    d1 = modelo.medir_distancia("inteligencia", "artificial")
    d2 = modelo.medir_distancia("funcao", "onda")
    d3 = modelo.medir_distancia("inteligencia", "hoje")
    d4 = modelo.medir_distancia("onda", "nome")

    print(f"Distância ('inteligencia' <-> 'artificial'): {d1:.4f}  (Esperado: Baixa)")
    print(f"Distância ('funcao' <-> 'onda'):             {d2:.4f}  (Esperado: Baixa)")
    print(f"Distância ('inteligencia' <-> 'hoje'):        {d3:.4f}  (Esperado: Alta)")
    print(f"Distância ('onda' <-> 'nome'):               {d4:.4f}  (Esperado: Alta)")
    print("-" * 50 + "\n")

    print("🤖 --- [GERAÇÃO COM GEO-ATENÇÃO COGNITIVA] ---")
    gatilhos = [
        "A inteligencia artificial",
        "No espaco quantico",
        "oi, tudo bem",
        "qual meu nome"
    ]

    for g in gatilhos:
        print(f"🙋 Gatilho: {g}")
        print("🤖 TGP-2: ", end="")
        for token in modelo.gerar_stream(g, max_tokens=18, temperatura=0.35):
            sys.stdout.write(token + " ")
            sys.stdout.flush()
            time.sleep(0.04)
        print("\n" + "-" * 40)
