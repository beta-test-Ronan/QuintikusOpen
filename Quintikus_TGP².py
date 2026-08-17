import numpy as np
import re
import unicodedata
import pickle
import os
import math
import random

# ============================================================
#  DISCO DE POINCARÉ
# ============================================================
class DiscoPoincare:
    """Operações básicas no disco de Poincaré."""

    def __init__(self, dim=128):
        self.dim = dim

    def adicao_mobius(self, u, v):
        """Adição de Möbius no disco de Poincaré."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)
        denom = 1 + 2 * uv + uu * vv
        if abs(denom) < 1e-12:
            denom = 1e-12
        return ((1 + 2 * uv + vv) * u + (1 - uu) * v) / denom

    def distancia(self, u, v):
        """Distância hiperbólica entre dois pontos no disco."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        diff = u - v
        num = 2 * np.dot(diff, diff)
        denom = (1 - np.dot(u, u)) * (1 - np.dot(v, v))
        if denom <= 0:
            return 50.0  # penalidade alta
        arg = 1 + num / denom
        arg = max(1.0, arg)  # acosh precisa de arg >= 1
        return math.acosh(arg)

    def projetar(self, u, max_norm=0.999):
        """Garante que o vetor permaneça dentro do disco aberto."""
        u = np.asarray(u, dtype=np.float64)
        norma = np.linalg.norm(u)
        if norma >= max_norm:
            return u * (max_norm / norma)
        return u

# ============================================================
#  AGENTE COGNITIVO HÍBRIDO
# ============================================================
class AgenteCognitivo:
    def __init__(self, arquivo_salvamento='agente_cognitivo.pkl', dim=128):
        self.arquivo_salvamento = arquivo_salvamento
        self.dim = dim
        self.disco = DiscoPoincare(dim)

        self.token_para_vetor = {}   # token -> vetor no disco
        self.vocab = set()
        self.bigramas = {}           # token -> {prox: contagem}
        self.trigramas = {}          # (tok1, tok2) -> {prox: contagem}
        self.contagem_total = {}     # token -> total de ocorrências

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
            print(f"Estado carregado de {self.arquivo_salvamento}")
        else:
            print("Nenhum estado salvo encontrado. Inicializando do zero.")

    # --------------------------------------------------------
    #  TOKENIZAÇÃO
    # --------------------------------------------------------
    def tokenizar(self, texto):
        texto = texto.lower()
        texto = unicodedata.normalize('NFD', texto)
        texto = ''.join(ch for ch in texto if unicodedata.category(ch) != 'Mn')
        return re.findall(r'[a-z0-9]+|[.,;:!?()\-"]', texto)

    def _inicializar_vetor(self, token):
        if token not in self.token_para_vetor:
            direcao = np.random.randn(self.dim)
            direcao /= (np.linalg.norm(direcao) + 1e-12)
            raio = np.random.uniform(0, 0.1)
            self.token_para_vetor[token] = self.disco.projetar(direcao * raio)
            self.vocab.add(token)

    # --------------------------------------------------------
    #  TREINAMENTO LINEAR (MARKOV)
    # --------------------------------------------------------
    def registrar_fluxo(self, tokens):
        """Mapeia bigramas e trigramas."""
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
    #  TREINAMENTO ESPACIAL (POINCARÉ)
    # --------------------------------------------------------
    def treinar_espacial(self, tokens, epocas=5, lr=0.02, janela=3, negativas=2):
        """Atração co-ocorrência / repulsão negativa no disco."""
        for tok in set(tokens):
            self._inicializar_vetor(tok)

        vocab_list = list(self.vocab)
        for epoca in range(epocas):
            for i, token in enumerate(tokens):
                inicio = max(0, i - janela)
                fim = min(len(tokens), i + janela + 1)
                contexto = [tokens[j] for j in range(inicio, fim) if j != i]
                if not contexto:
                    continue

                for ctx in contexto:
                    u = self.token_para_vetor[token]
                    v = self.token_para_vetor[ctx]

                    # atração: move token em direção ao contexto
                    novo_u = u + lr * (v - u)
                    self.token_para_vetor[token] = self.disco.projetar(novo_u)

                    # move contexto em direção ao token
                    novo_v = v + lr * (u - v)
                    self.token_para_vetor[ctx] = self.disco.projetar(novo_v)

                    # repulsão: amostras negativas
                    for _ in range(negativas):
                        neg = random.choice(vocab_list)
                        if neg == token or neg in contexto:
                            continue
                        w = self.token_para_vetor[neg]
                        novo_u = self.token_para_vetor[token] - lr * 0.5 * (w - self.token_para_vetor[token])
                        self.token_para_vetor[token] = self.disco.projetar(novo_u)

    # --------------------------------------------------------
    #  TENSÃO LÓGICA (SUJEITO vs PREDICADO)
    # --------------------------------------------------------
    def tensao_logica(self, vetores_ctx):
        if len(vetores_ctx) < 2:
            return np.zeros(self.dim)
        meio = len(vetores_ctx) // 2
        polo_suj = np.mean(vetores_ctx[:meio], axis=0) if meio > 0 else np.zeros(self.dim)
        polo_pred = np.mean(vetores_ctx[meio:], axis=0) if meio < len(vetores_ctx) else np.zeros(self.dim)
        return self.disco.adicao_mobius(-polo_suj, polo_pred)

    # --------------------------------------------------------
    #  VETOR DE INTENÇÃO (RUMINAÇÃO)
    # --------------------------------------------------------
    def vetor_intencao(self, contexto_tokens):
        if not contexto_tokens:
            return np.zeros(self.dim)

        for tok in contexto_tokens:
            self._inicializar_vetor(tok)

        vetores = np.array([self.token_para_vetor[tok] for tok in contexto_tokens])
        media = np.mean(vetores, axis=0)

        # atenção: pesos por distância hiperbólica à média
        pesos = []
        for v in vetores:
            d = self.disco.distancia(media, v)
            pesos.append(math.exp(-d))
        pesos = np.array(pesos) + 1e-12
        pesos /= pesos.sum()

        intencao = np.sum(vetores * pesos[:, None], axis=0)

        # fusão com tensão lógica (peso 0.25)
        tensao = self.tensao_logica(vetores)
        intencao = 0.75 * intencao + 0.25 * tensao
        return self.disco.projetar(intencao)

    def entropia_do_contexto(self, contexto_tokens):
        v = self.vetor_intencao(contexto_tokens)
        return float(np.linalg.norm(v))  # 0 = baixa dúvida, ~1 = alta dúvida

    # --------------------------------------------------------
    #  PROBABILIDADES
    # --------------------------------------------------------
    def prob_linear(self, contexto_tokens):
        """Probabilidade baseada apenas em transições Markov (bigramas/trigramas)."""
        vocab_list = list(self.vocab)
        if not contexto_tokens:
            uniforme = 1.0 / len(vocab_list)
            return {tok: uniforme for tok in vocab_list}

        # tenta trigrama
        if len(contexto_tokens) >= 2:
            chave = (contexto_tokens[-2], contexto_tokens[-1])
            if chave in self.trigramas:
                cont = self.trigramas[chave]
                total = sum(cont.values())
                # suavização add-one leve
                return {tok: (cont.get(tok, 0) + 0.01) / (total + 0.01 * len(vocab_list))
                        for tok in vocab_list}

        # bigrama
        ultimo = contexto_tokens[-1]
        if ultimo in self.bigramas:
            cont = self.bigramas[ultimo]
            total = sum(cont.values())
            return {tok: (cont.get(tok, 0) + 0.01) / (total + 0.01 * len(vocab_list))
                    for tok in vocab_list}

        # fallback uniforme
        uniforme = 1.0 / len(vocab_list)
        return {tok: uniforme for tok in vocab_list}

    def prob_nao_linear(self, vetor_intencao):
        """Probabilidade baseada na distância hiperbólica ao vetor de intenção."""
        vocab_list = list(self.vocab)
        dists = []
        for tok in vocab_list:
            v = self.token_para_vetor.get(tok)
            if v is None:
                self._inicializar_vetor(tok)
                v = self.token_para_vetor[tok]
            d = self.disco.distancia(vetor_intencao, v)
            dists.append(d)

        dists = np.array(dists)
        logits = -dists * 2.0
        logits -= np.max(logits)
        exps = np.exp(logits)
        probs = exps / exps.sum()
        return dict(zip(vocab_list, probs))

    # --------------------------------------------------------
    #  COEXISTÊNCIA DUAL E GERAÇÃO
    # --------------------------------------------------------
    def prever_proximo_token(self, contexto_tokens, temperatura=0.8,
                             penalidade_repeticao=2.5, historico_recente=None):
        if historico_recente is None:
            historico_recente = []

        vocab_list = list(self.vocab)
        if not contexto_tokens or len(vocab_list) == 0:
            probs = np.ones(len(vocab_list)) / len(vocab_list)
        else:
            p_lin = self.prob_linear(contexto_tokens)
            intencao = self.vetor_intencao(contexto_tokens)
            p_nlin = self.prob_nao_linear(intencao)

            logits = []
            for tok in vocab_list:
                lp = math.log(p_lin.get(tok, 1e-12))
                lnp = math.log(p_nlin.get(tok, 1e-12))
                # fusão: P_final = (P_não_linear^0.6) * (P_linear^1.4)
                logit = 0.6 * lnp + 1.4 * lp
                logits.append(logit)

            logits = np.array(logits)

            # penalidade de repetição (janela recente)
            for i, tok in enumerate(vocab_list):
                if tok in historico_recente:
                    logits[i] -= penalidade_repeticao * historico_recente.count(tok)

            # temperatura
            logits = logits / max(temperatura, 1e-6)
            logits -= np.max(logits)
            exps = np.exp(logits)
            probs = exps / exps.sum()

        idx = np.random.choice(len(vocab_list), p=probs)
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

            contexto_atual = gerados[-5:]  # janela de contexto
            prox = self.prever_proximo_token(
                contexto_atual,
                temperatura=temperatura,
                historico_recente=historico_recente
            )
            gerados.append(prox)
            historico_recente.append(prox)
            if len(historico_recente) > 10:
                historico_recente.pop(0)

        # reconstrução do texto com pontuação
        resultado = ''
        for tok in gerados:
            if tok in ['.', ',', ';', ':', '!', '?', ')', '(', '-', '"']:
                resultado = resultado.rstrip() + tok + ' '
            else:
                resultado += tok + ' '
        return resultado.strip()

    # --------------------------------------------------------
    #  TREINAMENTO A PARTIR DE ARQUIVO
    # --------------------------------------------------------
    def treinar_em_arquivo(self, caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            texto = f.read()
        tokens = self.tokenizar(texto)
        print(f"Tokenizando e treinando com {len(tokens)} tokens...")
        self.registrar_fluxo(tokens)
        self.treinar_espacial(tokens, epocas=5, lr=0.02, janela=3, negativas=2)
        self._salvar()
        print("Treinamento concluído.")

    # --------------------------------------------------------
    #  INTERFACE CLI
    # --------------------------------------------------------
    def loop_cli(self):
        print("\nAgente Cognitivo Híbrido")
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
