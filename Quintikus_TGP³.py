import math
import re
import pickle
import os
import random
from array import array

# ============================================================
#  CLASSE VETOR (substitui numpy)
# ============================================================
class Vetor:
    """Vetor de dimensão fixa usando array.array('d')."""
    __slots__ = ('dados', 'dim')

    def __init__(self, dim, valores=None):
        self.dim = dim
        if valores is None:
            self.dados = array('d', [0.0] * dim)
        else:
            if len(valores) != dim:
                raise ValueError("Dimensão inválida")
            self.dados = array('d', valores)

    @classmethod
    def aleatorio(cls, dim, raio=0.1):
        """Vetor aleatório com norma até raio."""
        v = cls(dim)
        for i in range(dim):
            v.dados[i] = random.uniform(-1, 1)
        norma = v.norma()
        if norma > 0:
            v = v * (raio / norma)
        return v

    def __add__(self, outro):
        res = Vetor(self.dim)
        for i in range(self.dim):
            res.dados[i] = self.dados[i] + outro.dados[i]
        return res

    def __sub__(self, outro):
        res = Vetor(self.dim)
        for i in range(self.dim):
            res.dados[i] = self.dados[i] - outro.dados[i]
        return res

    def __mul__(self, escalar):
        res = Vetor(self.dim)
        for i in range(self.dim):
            res.dados[i] = self.dados[i] * escalar
        return res

    def __neg__(self):
        return self * (-1.0)

    def dot(self, outro):
        s = 0.0
        for i in range(self.dim):
            s += self.dados[i] * outro.dados[i]
        return s

    def norma(self):
        return math.sqrt(self.dot(self))

    def normalizar(self):
        n = self.norma()
        if n > 0:
            return self * (1.0 / n)
        return Vetor(self.dim)

    def projetar_no_disco(self, max_norma=0.999):
        n = self.norma()
        if n > max_norma:
            return self * (max_norma / n)
        return self

    def copiar(self):
        return Vetor(self.dim, list(self.dados))

    def media(self, lista_vetores):
        """Retorna a média de uma lista de Vetor."""
        if not lista_vetores:
            return Vetor(self.dim)
        res = Vetor(self.dim)
        for v in lista_vetores:
            res = res + v
        return res * (1.0 / len(lista_vetores))

# ============================================================
#  DISCO DE POINCARÉ
# ============================================================
class DiscoPoincare:
    def __init__(self, dim=128):
        self.dim = dim

    def adicao_mobius(self, u, v):
        """Adição de Möbius no disco."""
        uu = u.dot(u)
        vv = v.dot(v)
        uv = u.dot(v)
        denom = 1 + 2 * uv + uu * vv
        if abs(denom) < 1e-12:
            denom = 1e-12
        coef_u = (1 + 2 * uv + vv) / denom
        coef_v = (1 - uu) / denom
        return u * coef_u + v * coef_v

    def distancia(self, u, v):
        """Distância hiperbólica."""
        diff = u - v
        num = 2 * diff.dot(diff)
        denom = (1 - u.dot(u)) * (1 - v.dot(v))
        if denom <= 0:
            return 50.0
        arg = 1 + num / denom
        if arg < 1.0:
            arg = 1.0
        return math.acosh(arg)

    def projetar(self, vetor, max_norma=0.999):
        return vetor.projetar_no_disco(max_norma)

# ============================================================
#  REDE TRIANGULAR (aprendizado de coerência A→B→C)
# ============================================================
class RedeTriangular:
    """
    Aprende a avaliar triângulos (A,B,C) no espaço hiperbólico.
    Features: dAB, dBC, dAC, cos_angle_B.
    Saída: score (logit) via combinação linear com pesos.
    """
    def __init__(self):
        # Pesos iniciais pequenos (4 features + bias)
        self.pesos = [random.uniform(-0.01, 0.01) for _ in range(4)]
        self.bias = 0.0

    def features(self, A, B, C, disco):
        dAB = disco.distancia(A, B)
        dBC = disco.distancia(B, C)
        dAC = disco.distancia(A, C)
        try:
            cos_angle = (math.cosh(dAB) * math.cosh(dBC) - math.cosh(dAC)) / \
                        (math.sinh(dAB) * math.sinh(dBC) + 1e-12)
            cos_angle = max(-1.0, min(1.0, cos_angle))
        except OverflowError:
            cos_angle = 0.0
        return [dAB, dBC, dAC, cos_angle]

    def pontuar(self, features):
        """Score = sum(peso_i * feat_i) + bias."""
        return sum(p * f for p, f in zip(self.pesos, features)) + self.bias

    def treinar_passos(self, exemplos_pos, exemplos_neg, lr=0.01, epocas=5):
        """Treina por regressão logística com gradiente descendente."""
        for epoca in range(epocas):
            todos = [(f, 1.0) for f in exemplos_pos] + [(f, 0.0) for f in exemplos_neg]
            random.shuffle(todos)
            for features, alvo in todos:
                score = self.pontuar(features)
                pred = 1.0 / (1.0 + math.exp(-score))
                erro = alvo - pred
                for i in range(len(self.pesos)):
                    self.pesos[i] += lr * erro * features[i]
                self.bias += lr * erro

# ============================================================
#  AGENTE COGNITIVO HÍBRIDO COM REDE TRIANGULAR
# ============================================================
class AgenteCognitivo:
    def __init__(self, arquivo_salvamento='agente_triangular.pkl', dim=64):
        self.arquivo_salvamento = arquivo_salvamento
        self.dim = dim
        self.disco = DiscoPoincare(dim)
        self.rede_triangular = RedeTriangular()

        self.token_para_vetor = {}
        self.vocab = set()
        self.bigramas = {}
        self.trigramas = {}
        self.contagem_total = {}

        self._carregar()

    # --------------------------------------------------------
    #  PERSISTÊNCIA
    # --------------------------------------------------------
    def _salvar(self):
        dados = {
            'tv': self.token_para_vetor,
            'vocab': list(self.vocab),
            'bigramas': self.bigramas,
            'trigramas': self.trigramas,
            'contagem_total': self.contagem_total,
            'pesos_triang': self.rede_triangular.pesos,
            'bias_triang': self.rede_triangular.bias,
        }
        with open(self.arquivo_salvamento, 'wb') as f:
            pickle.dump(dados, f)
        print(f"Estado salvo em {self.arquivo_salvamento}")

    def _carregar(self):
        if os.path.exists(self.arquivo_salvamento):
            with open(self.arquivo_salvamento, 'rb') as f:
                dados = pickle.load(f)
            self.token_para_vetor = dados.get('tv', {})
            self.vocab = set(dados.get('vocab', []))
            self.bigramas = dados.get('bigramas', {})
            self.trigramas = dados.get('trigramas', {})
            self.contagem_total = dados.get('contagem_total', {})
            self.rede_triangular.pesos = dados.get('pesos_triang', self.rede_triangular.pesos)
            self.rede_triangular.bias = dados.get('bias_triang', self.rede_triangular.bias)
            print(f"Estado carregado de {self.arquivo_salvamento}")
        else:
            print("Nenhum estado salvo encontrado. Inicializando do zero.")

    # --------------------------------------------------------
    #  TOKENIZAÇÃO (mantém acentos e pontuação)
    # --------------------------------------------------------
    def tokenizar(self, texto):
        # Converte para minúsculas sem remover acentos
        texto = texto.lower()
        # Captura palavras com letras Unicode (incluindo acentos) e pontuação
        return re.findall(r'[^\W\d_]+|[.,;:!?()\-"]', texto)

    def _inicializar_vetor(self, token):
        if token not in self.token_para_vetor:
            self.token_para_vetor[token] = Vetor.aleatorio(self.dim, raio=0.1)
            self.vocab.add(token)

    # --------------------------------------------------------
    #  TREINAMENTO LINEAR (MARKOV)
    # --------------------------------------------------------
    def registrar_fluxo(self, tokens):
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a not in self.bigramas:
                self.bigramas[a] = {}
            self.bigramas[a][b] = self.bigramas[a].get(b, 0) + 1
            self.contagem_total[a] = self.contagem_total.get(a, 0) + 1

            if i < len(tokens) - 2:
                c = tokens[i + 2]
                chave = (a, b)
                if chave not in self.trigramas:
                    self.trigramas[chave] = {}
                self.trigramas[chave][c] = self.trigramas[chave].get(c, 0) + 1

    # --------------------------------------------------------
    #  TREINAMENTO ESPACIAL (POINCARÉ) + REDE TRIANGULAR
    # --------------------------------------------------------
    def treinar_em_arquivo(self, caminho, epocas_espacial=3, lr_espacial=0.02,
                          epocas_rede=10, lr_rede=0.01):
        with open(caminho, 'r', encoding='utf-8') as f:
            texto = f.read()
        tokens = self.tokenizar(texto)
        print(f"Tokenizando e treinando com {len(tokens)} tokens...")

        for tok in set(tokens):
            self._inicializar_vetor(tok)

        self.registrar_fluxo(tokens)

        vocab_list = list(self.vocab)
        for epoca in range(epocas_espacial):
            print(f"  Época espacial {epoca+1}/{epocas_espacial}...")
            for i, token in enumerate(tokens):
                inicio = max(0, i - 3)
                fim = min(len(tokens), i + 4)
                contexto = [tokens[j] for j in range(inicio, fim) if j != i]
                if not contexto:
                    continue
                u = self.token_para_vetor[token]
                for ctx in contexto:
                    v = self.token_para_vetor[ctx]
                    # Atração
                    self.token_para_vetor[token] = self.disco.projetar(u + (v - u) * lr_espacial)
                    self.token_para_vetor[ctx] = self.disco.projetar(v + (u - v) * lr_espacial)
                    # Repulsão negativa
                    for _ in range(2):
                        neg = random.choice(vocab_list)
                        if neg == token or neg in contexto:
                            continue
                        w = self.token_para_vetor[neg]
                        self.token_para_vetor[token] = self.disco.projetar(u - (w - u) * (lr_espacial * 0.5))

        print("Preparando exemplos para a rede triangular...")
        exemplos_pos = []
        exemplos_neg = []
        for i in range(len(tokens) - 2):
            A, B, C = tokens[i], tokens[i+1], tokens[i+2]
            va = self.token_para_vetor[A]
            vb = self.token_para_vetor[B]
            vc = self.token_para_vetor[C]
            feat_pos = self.rede_triangular.features(va, vb, vc, self.disco)
            exemplos_pos.append(feat_pos)

            C_neg = random.choice(vocab_list)
            while C_neg == C:
                C_neg = random.choice(vocab_list)
            vc_neg = self.token_para_vetor[C_neg]
            feat_neg = self.rede_triangular.features(va, vb, vc_neg, self.disco)
            exemplos_neg.append(feat_neg)

        print("Treinando rede triangular...")
        self.rede_triangular.treinar_passos(exemplos_pos, exemplos_neg, lr=lr_rede, epocas=epocas_rede)

        self._salvar()
        print("Treinamento concluído.")

    # --------------------------------------------------------
    #  PROBABILIDADE LINEAR (MARKOV)
    # --------------------------------------------------------
    def prob_linear(self, contexto_tokens):
        vocab_list = list(self.vocab)
        if not contexto_tokens:
            uniforme = 1.0 / len(vocab_list)
            return {tok: uniforme for tok in vocab_list}

        if len(contexto_tokens) >= 2:
            chave = (contexto_tokens[-2], contexto_tokens[-1])
            if chave in self.trigramas:
                cont = self.trigramas[chave]
                total = sum(cont.values())
                return {tok: (cont.get(tok, 0) + 0.01) / (total + 0.01 * len(vocab_list))
                        for tok in vocab_list}

        ultimo = contexto_tokens[-1]
        if ultimo in self.bigramas:
            cont = self.bigramas[ultimo]
            total = sum(cont.values())
            return {tok: (cont.get(tok, 0) + 0.01) / (total + 0.01 * len(vocab_list))
                    for tok in vocab_list}

        uniforme = 1.0 / len(vocab_list)
        return {tok: uniforme for tok in vocab_list}

    # --------------------------------------------------------
    #  PROBABILIDADE NÃO‑LINEAR (REDE TRIANGULAR)
    # --------------------------------------------------------
    def prob_triangular(self, contexto_tokens):
        vocab_list = list(self.vocab)
        if len(contexto_tokens) < 2:
            vetor_ref = self.vetor_intencao(contexto_tokens)
            scores = {}
            for tok in vocab_list:
                v = self.token_para_vetor[tok]
                d = self.disco.distancia(vetor_ref, v)
                scores[tok] = -d
            return scores

        A_tok, B_tok = contexto_tokens[-2], contexto_tokens[-1]
        va = self.token_para_vetor.get(A_tok)
        vb = self.token_para_vetor.get(B_tok)
        if va is None or vb is None:
            return {tok: 0.0 for tok in vocab_list}

        scores = {}
        for tok in vocab_list:
            vc = self.token_para_vetor.get(tok)
            if vc is None:
                vc = Vetor(self.dim)
            feat = self.rede_triangular.features(va, vb, vc, self.disco)
            score = self.rede_triangular.pontuar(feat)
            scores[tok] = score
        return scores

    def vetor_intencao(self, contexto_tokens):
        if not contexto_tokens:
            return Vetor(self.dim)
        vetores = [self.token_para_vetor[t] for t in contexto_tokens if t in self.token_para_vetor]
        if not vetores:
            return Vetor(self.dim)
        media = vetores[0]
        for v in vetores[1:]:
            media = media + v
        media = media * (1.0 / len(vetores))
        return self.disco.projetar(media)

    # --------------------------------------------------------
    #  COEXISTÊNCIA DUAL E GERAÇÃO
    # --------------------------------------------------------
    def prever_proximo_token(self, contexto_tokens, temperatura=0.8,
                             penalidade_repeticao=2.5, historico_recente=None):
        if historico_recente is None:
            historico_recente = []

        vocab_list = list(self.vocab)
        if not contexto_tokens or len(vocab_list) == 0:
            probs = [1.0 / len(vocab_list)] * len(vocab_list)
            idx = random.choices(range(len(vocab_list)), weights=probs)[0]
            return vocab_list[idx]

        p_lin = self.prob_linear(contexto_tokens)
        p_tri = self.prob_triangular(contexto_tokens)

        max_score = max(p_tri.values()) if p_tri else 0.0
        exps = {tok: math.exp(score - max_score) for tok, score in p_tri.items()}
        soma_exp = sum(exps.values())
        p_tri_norm = {tok: e / soma_exp for tok, e in exps.items()}

        logits = []
        for tok in vocab_list:
            lp = math.log(p_lin.get(tok, 1e-12))
            ltp = math.log(p_tri_norm.get(tok, 1e-12))
            logit = 1.4 * lp + 0.6 * ltp
            logits.append(logit)

        for i, tok in enumerate(vocab_list):
            if tok in historico_recente:
                logits[i] -= penalidade_repeticao * historico_recente.count(tok)

        logits = [l / max(temperatura, 1e-6) for l in logits]
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        soma = sum(exps)
        probs = [e / soma for e in exps]

        idx = random.choices(range(len(vocab_list)), weights=probs)[0]
        return vocab_list[idx]

    def gerar(self, estimulo, max_tokens=50, temperatura=0.8):
        contexto = self.tokenizar(estimulo)
        for tok in contexto:
            self._inicializar_vetor(tok)

        gerados = list(contexto)
        historico_recente = []

        for _ in range(max_tokens):
            if len(gerados) > len(contexto) and gerados[-1] in ['.', '!', '?']:
                break
            contexto_atual = gerados[-5:]
            prox = self.prever_proximo_token(
                contexto_atual,
                temperatura=temperatura,
                historico_recente=historico_recente
            )
            gerados.append(prox)
            historico_recente.append(prox)
            if len(historico_recente) > 10:
                historico_recente.pop(0)

        resultado = ''
        for tok in gerados:
            if tok in ['.', ',', ';', ':', '!', '?', ')', '(', '-', '"']:
                resultado = resultado.rstrip() + tok + ' '
            else:
                resultado += tok + ' '
        return resultado.strip()

    # --------------------------------------------------------
    #  INTERFACE CLI
    # --------------------------------------------------------
    def loop_cli(self):
        print("\nAgente Cognitivo Híbrido com Rede Triangular")
        print("Comandos:")
        print("  train:<arquivo>  -> treina com o arquivo de texto")
        print("  <texto>          -> gera uma continuação")
        print("  sair             -> encerra")
        while True:
            entrada = input("\n>>> ").strip()
            if entrada == 'sair':
                self._salvar()
                print("Encerrado.")
                break
            elif entrada.startswith('train:'):
                arquivo = entrada.split(':', 1)[1]
                if os.path.exists(arquivo):
                    self.treinar_em_arquivo(arquivo)
                else:
                    print(f"Arquivo não encontrado: {arquivo}")
            elif entrada:
                resposta = self.gerar(entrada, max_tokens=50)
                print(f"\n{resposta}")

# ============================================================
#  EXECUÇÃO
# ============================================================
if __name__ == '__main__':
    agente = AgenteCognitivo()
    agente.loop_cli()
