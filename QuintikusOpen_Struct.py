# ============================================================
# QUINTIKUS STRUCT COMPLETA
# Coração (Feeling) + Mente (Arborecente) + Contexto (Entrópico)
# Sente, Pensa, Contextualiza, Age
# ============================================================

import os, re, math, time, pickle
from array import array
from collections import defaultdict, Counter

# ============================================================
# 🧠 QUINTIKUSFEELING (CORAÇÃO NEURAL)
# ============================================================
class NeuroMicro:
    def __init__(self, arquivo_rede="emo.rn"):
        self.arquivo_rede = arquivo_rede
        self.padroes_bytes = {
            0: (b'alegria', [b'amo', b'feliz', b'boa', b'gratidao', b'sorriso', b'radiante', b'conquista', b'vitoria', b'maravilhoso', b'excelente']),
            1: (b'tristeza', [b'triste', b'dor', b'saudade', b'choro', b'perda', b'melancolia', b'desanimo', b'sofrimento', b'lagrimas']),
            2: (b'raiva', [b'odeio', b'odio', b'raiva', b'irritado', b'furia', b'bug', b'erro', b'indignado', b'revoltado', b'colera']),
            3: (b'medo', [b'medo', b'ansioso', b'ansiedade', b'panico', b'temor', b'inseguro', b'preocupado', b'apreensivo', b'aterrorizado']),
            4: (b'surpresa', [b'uau', b'nossa', b'incrivel', b'framework', b'impressionante', b'chocado', b'inesperado', b'revelacao']),
            5: (b'nojo', [b'nojento', b'asco', b'repulsa', b'horrivel', b'desgosto', b'podre', b'abominavel', b'asqueroso', b'nauseante'])
        }
        self.pesos = array('f', [1.0/6] * 6)
        self.exp_lut = array('f', [math.exp(i/100.0) for i in range(-500, 500)])
        self._precompilar_padroes()
        self.total_treinos = 0
        self.taxa_aprendizado = 0.01
        self.metricas = {'min_tempo': float('inf'), 'max_tempo': 0.0, 'total_analises': 0, 'soma_tempos': 0.0}
        if os.path.exists(self.arquivo_rede):
            self._carregar_rede()
        else:
            self._salvar_rede()

    def _precompilar_padroes(self):
        self.busca_plana = []
        for idx, (emocao, palavras) in self.padroes_bytes.items():
            for palavra in palavras:
                self.busca_plana.append((idx, palavra, len(palavra)))
        self.busca_plana.sort(key=lambda x: x[2], reverse=True)

    def fast_exp(self, x):
        idx = int(x * 100) + 500
        if 0 <= idx < 1000: return self.exp_lut[idx]
        return 0.0 if x < -5 else float('inf')

    def _softmax(self, scores):
        max_s = max(scores)
        exp_scores = [self.fast_exp(s - max_s) for s in scores]
        total_exp = sum(exp_scores)
        if total_exp > 0.0001:
            inv_total = 1.0 / total_exp
            return [s * inv_total for s in exp_scores]
        return [1.0/6] * 6

    def _entropia_cruzada(self, probs, idx_alvo):
        epsilon = 1e-10
        return -math.log(max(probs[idx_alvo], epsilon))

    def _gradiente_descendente(self, probs, idx_alvo):
        y_true = [0.0] * 6
        y_true[idx_alvo] = 1.0
        for i in range(6):
            gradiente = probs[i] - y_true[i]
            self.pesos[i] -= self.taxa_aprendizado * gradiente
            self.pesos[i] = max(0.001, self.pesos[i])
        soma = sum(self.pesos)
        for i in range(6): self.pesos[i] /= soma

    def analisar_us(self, texto_bytes):
        t0 = time.perf_counter_ns()
        scores = [0.0] * 6
        for idx, palavra, _ in self.busca_plana:
            if palavra in texto_bytes: scores[idx] += self.pesos[idx]
        probs = self._softmax(scores)
        dt = (time.perf_counter_ns() - t0) / 1000.0
        self.metricas['total_analises'] += 1
        self.metricas['soma_tempos'] += dt
        self.metricas['min_tempo'] = min(self.metricas['min_tempo'], dt)
        self.metricas['max_tempo'] = max(self.metricas['max_tempo'], dt)
        return probs, dt, scores

    def treinar(self, texto, emocao_alvo):
        nomes_emocao = ['alegria', 'tristeza', 'raiva', 'medo', 'surpresa', 'nojo']
        if emocao_alvo not in nomes_emocao: return None
        idx_alvo = nomes_emocao.index(emocao_alvo)
        texto_bytes = texto.encode('ascii', errors='ignore') if isinstance(texto, str) else texto
        probs, _, _ = self.analisar_us(texto_bytes)
        loss_antes = self._entropia_cruzada(probs, idx_alvo)
        self._gradiente_descendente(probs, idx_alvo)
        probs_depois, _, _ = self.analisar_us(texto_bytes)
        loss_depois = self._entropia_cruzada(probs_depois, idx_alvo)
        self.total_treinos += 1
        self._salvar_rede()
        return {'loss_antes': loss_antes, 'loss_depois': loss_depois, 'melhora': loss_antes - loss_depois}

    def prever(self, texto):
        texto_bytes = texto.encode('ascii', errors='ignore') if isinstance(texto, str) else texto
        probs, tempo, _ = self.analisar_us(texto_bytes)
        nomes = ['alegria', 'tristeza', 'raiva', 'medo', 'surpresa', 'nojo']
        max_idx = max(range(6), key=lambda i: probs[i])
        return {'sentimento': nomes[max_idx], 'confianca': round(probs[max_idx], 4), 'tempo_us': round(tempo, 2)}

    def _salvar_rede(self):
        with open(self.arquivo_rede, 'wb') as f:
            pickle.dump({'pesos': list(self.pesos), 'total_treinos': self.total_treinos}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _carregar_rede(self):
        with open(self.arquivo_rede, 'rb') as f:
            dados = pickle.load(f)
        for i, p in enumerate(dados.get('pesos', [1.0/6]*6)): self.pesos[i] = p
        self.total_treinos = dados.get('total_treinos', 0)


# ============================================================
# 🌍 CONTEXTO ENTRÓPICO (AP05)
# ============================================================
class ContextoEntropico:
    """Analisa a entropia das palavras ao longo das interações para determinar o tema da conversa."""
    def __init__(self):
        self.documentos = []          # lista de frases (documentos)
        self.freq_palavras = Counter()# frequência total de cada palavra
        self.total_documentos = 0

    def adicionar(self, texto):
        palavras = self._limpar(texto)
        if not palavras: return
        self.documentos.append(set(palavras))
        self.total_documentos += 1
        self.freq_palavras.update(palavras)

    def _limpar(self, texto):
        stop = {'a','o','e','de','do','da','em','para','com','que','se','não',
                'é','foi','ser','estar','está','era','são','por','como','mas',
                'ou','nem','os','as','um','uma','me','te','lhe','nos','vos','lhes'}
        return [p for p in re.findall(r'\b\w+\b', texto.lower()) if p not in stop and len(p) > 1]

    def entropia_palavra(self, palavra):
        """Entropia de Shannon da distribuição da palavra entre documentos."""
        if self.total_documentos == 0: return 1.0
        aparece_em = sum(1 for doc in self.documentos if palavra in doc)
        if aparece_em == 0: return 1.0
        p = aparece_em / self.total_documentos
        q = 1 - p
        if p == 0 or q == 0: return 0.0
        return -p * math.log2(p) - q * math.log2(q)

    def especificidade(self, palavra):
        """1 - entropia normalizada. Quanto mais específica, menor a entropia."""
        max_entropia = 1.0  # entropia máxima para duas classes (aparece/não aparece)
        ent = self.entropia_palavra(palavra)
        espec = 1.0 - (ent / max_entropia)
        return max(0.0, min(1.0, espec))

    def tema_atual(self, top_n=3):
        """Retorna as palavras mais específicas (menor entropia) como o tema da conversa."""
        if not self.freq_palavras: return []
        especificidades = {p: self.especificidade(p) for p in self.freq_palavras}
        ordenado = sorted(especificidades.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ordenado[:top_n]]


# ============================================================
# 🌳 MENTE ARBORECENTE (PENSAR + MEMÓRIA)
# ============================================================
class Memoria:
    def __init__(self):
        self.dados = []
        self.indice = defaultdict(list)
        self.tfidf_cache = {}

    def adicionar(self, entrada, resposta, sentimento=None):
        idx = len(self.dados)
        self.dados.append({'entrada': entrada, 'resposta': resposta, 'sentimento': sentimento or "neutro"})
        for palavra in entrada.split():
            self.indice[palavra].append(idx)
        for palavra in resposta.split():
            self.indice[palavra].append(idx)
        self.tfidf_cache.clear()

    def buscar_por_palavra(self, palavra):
        return self.indice.get(palavra, [])

    def calcular_tfidf(self, palavras):
        total_docs = len(self.dados)
        if total_docs == 0: return {p: 0.0 for p in palavras}
        tfidf = {}
        freq_palavras = Counter(palavras)
        for palavra, freq in freq_palavras.items():
            tf = freq / len(palavras) if palavras else 0
            doc_com_palavra = len(self.indice.get(palavra, []))
            idf = math.log((total_docs + 1) / (doc_com_palavra + 1)) + 1
            tfidf[palavra] = tf * idf
        return tfidf

class Percepcao:
    def __init__(self):
        self.pressao = 0.0
        self.volume = 0.0
        self.foco = []
        self.sentimento = "neutro"
        self.confianca = 0.0

class Consciencia:
    def __init__(self):
        self.baixa = {}
        self.alta = {}


# ============================================================
# QUINTIKUS STRUCT – TRINDADE + CONTEXTO
# ============================================================
class QuintikusStruct:
    def __init__(self, arquivo_emocao="emo.rn"):
        self.feeling = NeuroMicro(arquivo_emocao)
        self.contexto = ContextoEntropico()
        self.memoria = Memoria()
        self.percepcao = Percepcao()
        self.consciencia = Consciencia()

        self.stop = {'a','o','e','de','do','da','em','para','com','que','se','não',
                     'é','foi','ser','estar','está','era','são','por','como','mas',
                     'ou','nem','os','as','um','uma','me','te','lhe','nos','vos','lhes'}
        self.sinonimos = {"bastante":"muita","pouquinha":"pouca","dar":"passar","dá":"passar",
                          "entregar":"passar","quero":"querer","preciso":"precisar",
                          "gostaria":"querer","ve":"querer","ver":"querer","enxergar":"querer"}
        self.filtro_observacao = {"beber","bebeu","disse","passar","usar","ela","ele","generike",
                                  "sempre","durante","tempo","mas","com","por","para","que"}
        self.LIMIAR_SIMILARIDADE_BAIXA = 0.3
        self.LIMIAR_SIMILARIDADE_ALTA = 0.2
        self.LIMIAR_PRESSAO = 0.6
        self.DECAY = 0.95
        self.MAX_ALTAS = 20

    def _limpar(self, texto):
        palavras = re.findall(r'\b\w+\b', texto.lower())
        palavras = [self.sinonimos.get(p, p) for p in palavras]
        return [p for p in palavras if p not in self.stop and len(p) > 1]

    def _similaridade(self, palavras1, palavras2):
        if not palavras1 or not palavras2: return 0.0
        intersecao = len(palavras1 & palavras2)
        uniao = len(palavras1 | palavras2)
        return intersecao / (uniao + 0.1) if uniao > 0 else 0

    def sentir(self, texto):
        resultado = self.feeling.prever(texto)
        self.percepcao.sentimento = resultado['sentimento']
        self.percepcao.confianca = resultado['confianca']
        return resultado

    def preceber(self, entrada):
        entrada_limpa = ' '.join(self._limpar(entrada))
        palavras = entrada_limpa.split()
        if not palavras:
            self.percepcao = Percepcao()
            return

        # Atualiza contexto entrópico com a entrada
        self.contexto.adicionar(entrada)

        tfidf = self.memoria.calcular_tfidf(palavras)
        pressao = 0.0
        volume = 0.0
        foco = []

        for palavra, peso in tfidf.items():
            freq_mem = len(self.memoria.buscar_por_palavra(palavra))
            raridade = 1.0 / (freq_mem + 0.5)
            redundancia = palavras.count(palavra) / len(palavras)
            peso_alta = peso * (1 + raridade * redundancia)
            peso_baixa = peso * redundancia / (raridade + 0.1)
            pressao += peso_baixa
            volume += peso_alta
            if peso_alta > (volume / len(tfidf)) * 1.5:
                foco.append((palavra, peso_alta))

        foco.sort(key=lambda x: x[1], reverse=True)
        self.percepcao.foco = [p[0] for p in foco[:3]]
        self.percepcao.pressao = pressao
        self.percepcao.volume = volume
        self.percepcao.confianca = 1.0 / (1 + math.exp(-pressao))
        self.sentir(entrada)

        if pressao > self.LIMIAR_PRESSAO:
            self._consolidar(palavras, entrada_limpa)

    def _consolidar(self, palavras, entrada_limpa):
        similaridades = []
        palavras_set = set(palavras)
        for idx, dado in enumerate(self.memoria.dados):
            palavras_mem = set(dado['entrada'].split())
            sim = self._similaridade(palavras_set, palavras_mem)
            if sim > self.LIMIAR_SIMILARIDADE_BAIXA:
                similaridades.append((idx, sim, dado))
        if not similaridades: return
        similaridades.sort(key=lambda x: x[1], reverse=True)
        for idx, sim, dado in similaridades[:2]:
            padrao = dado['entrada']
            resposta = dado['resposta']
            if padrao in self.consciencia.baixa:
                resp_atual, contador = self.consciencia.baixa[padrao]
                if resposta == resp_atual:
                    self.consciencia.baixa[padrao] = (resposta, contador + 1)
            else:
                self.consciencia.baixa[padrao] = (resposta, 1)
            if padrao in self.consciencia.alta:
                resp_atual, peso_atual, sim_acum, contador = self.consciencia.alta[padrao]
                novo_sim = (sim_acum * contador + sim) / (contador + 1)
                novo_peso = min(1.0, peso_atual + 0.1)
                self.consciencia.alta[padrao] = (resposta, novo_peso, novo_sim, contador + 1)
            else:
                self.consciencia.alta[padrao] = (resposta, 0.5, sim, 1)
        if len(self.consciencia.alta) > self.MAX_ALTAS:
            ordenados = sorted(self.consciencia.alta.items(), key=lambda x: x[1][1], reverse=True)
            self.consciencia.alta = dict(ordenados[:self.MAX_ALTAS])

    def agir(self, entrada):
        entrada_limpa = ' '.join(self._limpar(entrada))
        palavras = set(entrada_limpa.split())
        foco = set(self.percepcao.foco)
        tema = set(self.contexto.tema_atual())  # palavras do contexto atual

        if not palavras: return "não entendi"

        # Consciência Baixa
        candidatos_baixa = []
        for padrao, (resposta, contador) in self.consciencia.baixa.items():
            palavras_padrao = set(padrao.split())
            sim = self._similaridade(palavras, palavras_padrao)
            if sim >= self.LIMIAR_SIMILARIDADE_BAIXA:
                bonus_foco = 0.2 if foco & palavras_padrao else 0
                bonus_tema = 0.2 if tema & palavras_padrao else 0
                confianca = min(0.3, contador * 0.05)
                peso_total = sim + bonus_foco + bonus_tema + confianca
                candidatos_baixa.append((peso_total, resposta))
        if candidatos_baixa:
            candidatos_baixa.sort(key=lambda x: x[0], reverse=True)
            return candidatos_baixa[0][1]

        # Consciência Alta
        candidatos_alta = []
        for padrao, (resposta, peso, sim_acum, contador) in self.consciencia.alta.items():
            palavras_padrao = set(padrao.split())
            sim = self._similaridade(palavras, palavras_padrao)
            if sim >= self.LIMIAR_SIMILARIDADE_ALTA and peso > 0.3:
                bonus_foco = 0.2 if foco & palavras_padrao else 0
                bonus_tema = 0.2 if tema & palavras_padrao else 0
                peso_total = sim * 0.4 + peso * 0.3 + sim_acum * 0.2 + bonus_foco + bonus_tema
                candidatos_alta.append((peso_total, resposta))
        if candidatos_alta:
            candidatos_alta.sort(key=lambda x: x[0], reverse=True)
            return candidatos_alta[0][1]

        # Fallback: busca na memória com tema
        melhor_sim = 0
        melhor_resposta = None
        for dado in self.memoria.dados:
            palavras_mem = set(dado['entrada'].split())
            sim = self._similaridade(palavras, palavras_mem)
            if foco & palavras_mem: sim += 0.15
            if tema & palavras_mem: sim += 0.15
            if dado.get('sentimento') == self.percepcao.sentimento and self.percepcao.sentimento != "neutro":
                sim += 0.1
            if sim > melhor_sim:
                melhor_sim = sim
                melhor_resposta = dado['resposta']
        if melhor_resposta and melhor_sim > 0.25:
            return melhor_resposta

        return "ainda não sei como reagir a isso"

    def aprender(self, entrada, resposta, sentimento=None):
        entrada_limpa = ' '.join(self._limpar(entrada))
        if not entrada_limpa or not resposta: return
        self.memoria.adicionar(entrada_limpa, resposta, sentimento)
        self.contexto.adicionar(entrada)  # atualiza contexto com o exemplo
        if entrada_limpa in self.consciencia.baixa:
            resp_atual, contador = self.consciencia.baixa[entrada_limpa]
            if resposta == resp_atual:
                self.consciencia.baixa[entrada_limpa] = (resposta, contador + 1)
        else:
            self.consciencia.baixa[entrada_limpa] = (resposta, 1)
        if sentimento:
            self.feeling.treinar(entrada, sentimento)

    def observar(self, narrativa):
        padroes_regex = [
            r"pra\s+(.*?)\s+precisa\s+(?:de\s+)?(.*?)(?:\.|,|;|$)",
            r"quando\s+(.*?)\s+(?:ela\s+)?usa\s+(.*?)(?:\.|,|;|$)",
            r"(.*?)\s+passou\s+(?:a\s+)?usar\s+(.*?)(?:\.|,|;|$)",
        ]
        for padrao in padroes_regex:
            matches = re.findall(padrao, narrativa.lower())
            for condicao, acao in matches:
                cond_limpa = ' '.join(self._limpar(condicao))
                acao_limpa = ' '.join(self._limpar(acao))
                palavras_cond = [p for p in cond_limpa.split() if p not in self.filtro_observacao]
                palavras_acao = [p for p in acao_limpa.split() if p not in self.filtro_observacao]
                cond_limpa = ' '.join(palavras_cond)
                acao_limpa = ' '.join(palavras_acao)
                if cond_limpa and acao_limpa:
                    if cond_limpa in self.consciencia.alta:
                        resp_atual, peso_atual, sim_acum, contador = self.consciencia.alta[cond_limpa]
                        novo_peso = min(1.0, peso_atual + 0.1)
                        self.consciencia.alta[cond_limpa] = (acao_limpa, novo_peso, sim_acum, contador + 1)
                    else:
                        self.consciencia.alta[cond_limpa] = (acao_limpa, 0.6, 0.5, 1)

    def ciclo(self, entrada):
        self.preceber(entrada)
        resposta = self.agir(entrada)
        return resposta


# ============================================================
# TESTE DA VERSÃO COMPLETA
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("QUINTIKUS STRUCT – Sentir, Contextualizar, Pensar, Agir")
    print("=" * 60)

    quintikus = QuintikusStruct("emo.rn")

    exemplos = [
        ("quero agua", "aqui está a água", "neutro"),
        ("quero suco", "servindo suco para você", "alegria"),
        ("quero muita agua", "aqui está a água no copo grande", "neutro"),
        ("quero pouca agua", "aqui está a água no copo pequeno", "neutro"),
        ("preciso de copo grande", "aqui está o copo grande", "neutro"),
        ("quero copo pequeno", "pegando o copo pequeno", "neutro"),
        ("me passa copo", "aqui está o copo", "neutro"),
        ("quero agua gelada", "aqui está a água gelada", "alegria"),
        ("me dar agua", "aqui está a água", "neutro"),
        ("me dar suco", "servindo suco para você", "alegria"),
        ("preciso de gelo", "aqui está o gelo", "neutro"),
        ("me ajuda", "como posso ajudar você?", "surpresa"),
    ]

    for entrada, resposta, sent in exemplos:
        quintikus.aprender(entrada, resposta, sent)

    quintikus.observar("Pra beber muita agua precisa de copo grande. Pra beber pouca agua usa copo pequeno.")

    testes = [
        "quero beber muita agua",
        "quero beber pouca agua",
        "preciso de muita agua",
        "vou beber bastante agua",
        "me da pouca agua",
        "me passa copo",
        "me dar suco",
        "quero um copo",
        "me da suco por favor",
        "passa o copo",
        "quero suco",
        "me ver uma agua",
        "preciso de copo grande",
        "quero copo pequeno",
        "como você está?",
        "me ajuda por favor",
        "quero agua gelada",
        "preciso de gelo",
    ]

    print("\nCiclos da Quintikus Struct Completa:")
    for frase in testes:
        resposta = quintikus.ciclo(frase)
        tema = quintikus.contexto.tema_atual()
        print(f"'{frase}' -> '{resposta}'")
        print(f"   ❤️ Sentimento: {quintikus.percepcao.sentimento} (confiança: {quintikus.percepcao.confianca:.2f})")
        print(f"   🌍 Contexto (tema): {tema}")
        print(f"   🌳 Pressão: {quintikus.percepcao.pressao:.3f} | Volume: {quintikus.percepcao.volume:.3f}")
        print(f"   🎯 Foco: {quintikus.percepcao.foco}")
        print()