import numpy as np
import re
import pickle
import os
from collections import defaultdict, Counter
import sys
import time
import unicodedata

# ============================================================
# MÓDULOS MATEMÁTICOS E FÍSICOS (Geo-Quânticos)
# ============================================================

class DiscoPoincare:
    def __init__(self, dim=64):
        self.dim = dim
    
    def distancia(self, x, y):
        norma_x = np.clip(np.linalg.norm(x), 0, 0.99)
        norma_y = np.clip(np.linalg.norm(y), 0, 0.99)
        x_proj = x / (np.linalg.norm(x) + 1e-8) * norma_x
        y_proj = y / (np.linalg.norm(y) + 1e-8) * norma_y
        diff_sq = np.sum((x_proj - y_proj)**2)
        num = 2 * diff_sq
        den = (1 - norma_x**2) * (1 - norma_y**2) + 1e-12
        cosh_val = 1 + num / den
        cosh_val = np.clip(cosh_val, 1.0, None)
        return np.arccosh(cosh_val)
    
    def mover_para_borda(self, ponto, intensidade=0.01):
        norma = np.linalg.norm(ponto)
        if norma < 0.99:
            direcao = ponto / (norma + 1e-8)
            return ponto + direcao * intensidade
        return ponto

class TokenQuantico:
    def __init__(self, dim=64, num_estados=4):
        self.dim = dim
        self.num_estados = num_estados
        self.estados = {} 
    
    def inicializar_base(self, token, vetor_base):
        if token not in self.estados:
            self.estados[token] = [vetor_base.copy()]

    def adicionar_interpretacao(self, token, contexto_vetor, peso=0.3):
        if token not in self.estados:
            return
        estado_base = self.estados[token][0]
        novo_estado = estado_base * (1 - peso) + contexto_vetor * peso
        norma = np.linalg.norm(novo_estado)
        if norma > 0.99:
            novo_estado = novo_estado / norma * 0.99
        self.estados[token].append(novo_estado)
        if len(self.estados[token]) > self.num_estados:
            self.estados[token].pop(1) 
    
    def colapsar(self, token, consulta_vetor, disco):
        if token not in self.estados:
            return None
        melhores = []
        for estado in self.estados[token]:
            dist = disco.distancia(consulta_vetor, estado)
            sim = 1.0 / (1.0 + dist)
            melhores.append((sim, estado))
        melhores.sort(key=lambda x: x[0], reverse=True)
        sims = np.array([m[0] for m in melhores])
        probs = sims**2
        probs_soma = probs.sum()
        if probs_soma == 0:
            return melhores[0][1]
        probs /= probs_soma
        idx = np.random.choice(len(melhores), p=probs)
        return melhores[idx][1]

class MPSTransicao:
    def __init__(self, dim_token=64, bond_dim=2):
        self.dim_token = dim_token
        self.bond_dim = bond_dim
        self.A = np.random.randn(dim_token, bond_dim) * 0.05
        self.B = np.random.randn(bond_dim, bond_dim) * 0.05
        self.C = np.random.randn(bond_dim, dim_token) * 0.05
    
    def transicao(self, vetor_entrada):
        h1 = np.dot(vetor_entrada, self.A)
        h2 = np.dot(h1, self.B)
        saida = np.dot(h2, self.C)
        norma = np.linalg.norm(saida)
        if norma > 0.99:
            saida = saida / norma * 0.99
        elif norma == 0:
            saida = np.random.randn(self.dim_token) * 0.01
        return saida

# ============================================================
# CLASSE TGP-2.5 COM INÉRCIA LINEAR ANTI-REPETIÇÃO
# ============================================================

class TGP2:
    def __init__(self, dim_espaco=64, arquivo_modelo="tgp2_camada_geometrica.pkl"):
        self.dim_espaco = dim_espaco
        self.arquivo_modelo = arquivo_modelo
        
        self.disco = DiscoPoincare(dim_espaco)
        self.quantico = TokenQuantico(dim_espaco, num_estados=4)
        self.mps = MPSTransicao(dim_espaco, bond_dim=2)
        
        self.token_para_vetor = {}
        self.episodios = []
        
        self.bigramas = defaultdict(lambda: defaultdict(int))
        self.trigramas = defaultdict(lambda: defaultdict(int))
        self.indice_prefixo = defaultdict(list)
        
        self.token_fim = '<END>'
        
        if not self.carregar_modelo():
            self._registrar_token(self.token_fim)
        
    def salvar_modelo(self):
        estado = {
            'vetores': self.token_para_vetor,
            'quantico_estados': self.quantico.estados,
            'bigramas': dict(self.bigramas),
            'trigramas': dict(self.trigramas),
            'indice_prefixo': dict(self.indice_prefixo)
        }
        with open(self.arquivo_modelo, 'wb') as f:
            pickle.dump(estado, f)

    def carregar_modelo(self):
        if os.path.exists(self.arquivo_modelo):
            with open(self.arquivo_modelo, 'rb') as f:
                estado = pickle.load(f)
            self.token_para_vetor = estado['vetores']
            self.quantico.estados = estado['quantico_estados']
            for k, v in estado['bigramas'].items(): self.bigramas[k].update(v)
            for k, v in estado['trigramas'].items(): self.trigramas[k].update(v)
            for k, v in estado['indice_prefixo'].items(): self.indice_prefixo[k] = v
            return True
        return False

    def _registrar_token(self, token):
        if token not in self.token_para_vetor:
            vetor = np.random.randn(self.dim_espaco) * 0.4 
            norma = np.linalg.norm(vetor)
            if norma > 0.99: 
                vetor = vetor / norma * 0.99
            self.token_para_vetor[token] = vetor
            self.quantico.inicializar_base(token, vetor)

    def tokenizar(self, texto):
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acento = ''.join([c for c in texto_nfkd if unicodedata.category(c) != 'Mn'])
        tokens = re.findall(r'[a-z0-9]+|[.,!?;:]+|\s+', texto_sem_acento)
        return [t for t in tokens if t.strip()]
    
    def atencao_multiversal_vetorial(self, contexto_tokens):
        if not contexto_tokens:
            return np.zeros(self.dim_espaco)
        frequencias = Counter(contexto_tokens)
        max_freq = max(frequencias.values()) if frequencias else 1
        vetor_gravitacional = np.zeros(self.dim_espaco)
        peso_total = 0.0
        for token, count in frequencias.items():
            if token in self.token_para_vetor:
                peso = count / max_freq 
                vetor_gravitacional += self.token_para_vetor[token] * peso
                peso_total += peso
        if peso_total > 0:
            vetor_gravitacional /= peso_total
        norma = np.linalg.norm(vetor_gravitacional)
        if norma > 0.99:
            vetor_gravitacional = (vetor_gravitacional / norma) * 0.99
        return vetor_gravitacional

    def transpassar(self, tokens):
        tokens = tokens[-20:] if len(tokens) > 20 else tokens 
        if not tokens: return []
        for t in tokens: self._registrar_token(t)
        vetores_base = []
        for t in tokens:
            v = self.token_para_vetor[t].copy()
            if len(t) > 5: v = self.disco.mover_para_borda(v, 0.05)
            vetores_base.append(v)
            
        estados_colapsados = []
        for i, t in enumerate(tokens):
            inicio, fim = max(0, i - 2), min(len(tokens), i + 3)
            contexto = np.mean(vetores_base[inicio:fim], axis=0)
            self.quantico.adicionar_interpretacao(t, contexto)
            estado_atual = self.quantico.colapsar(t, contexto, self.disco)
            if estado_atual is not None:
                estados_colapsados.append(estado_atual)
                
        curva_mps = []
        estado_dinamico = estados_colapsados[0] if estados_colapsados else np.zeros(self.dim_espaco)
        for _ in range(len(tokens)):
            estado_dinamico = self.mps.transicao(estado_dinamico)
            curva_mps.append(estado_dinamico)
        return curva_mps

    def validar_similaridade_neural(self, vetor_alvo, limiar_distancia=1.2, min_similares=3, max_similares=7):
        similares_contagem = 0
        for token, vetor_ref in self.token_para_vetor.items():
            if token == self.token_fim: continue
            dist = self.disco.distancia(vetor_alvo, vetor_ref)
            if dist <= limiar_distancia:
                similares_contagem += 1
        return min_similares <= similares_contagem <= max_similares

    def camada_contextualizacao_previa(self, tokens_iniciais):
        predicados = [t for t in tokens_iniciais if len(t) >= 4 and t in self.token_para_vetor]
        relatorios = []
        for p in predicados:
            v_p = self.token_para_vetor[p]
            vizinhos = []
            for t, v_ref in self.token_para_vetor.items():
                if t != p and len(t) > 3:
                    if self.disco.distancia(v_p, v_ref) < 1.0:
                        vizinhos.append(t)
            if vizinhos:
                relatorios.append(f"({p} AND {', '.join(vizinhos[:2])})")
        if relatorios:
            return f"[Contexto: {' | '.join(relatorios)}]"
        return "[Contexto: Predicados Novos]"

    # ---------- DETECTOR DE REPETIÇÃO DE LONGO ALCANCE ----------
    def _repeticao_longa(self, contexto, comprimento=6):
        """Retorna True se a sequência dos últimos `comprimento` tokens já apareceu antes no contexto."""
        if len(contexto) < comprimento * 2:
            return False
        ultima_seq = tuple(contexto[-comprimento:])
        # Procura nos primeiros 90% do contexto (evita comparar com ela mesma)
        for i in range(0, len(contexto) - comprimento - 1):
            if tuple(contexto[i:i+comprimento]) == ultima_seq:
                return True
        return False

    def _sequencia_repetida(self, contexto, token_candidato, janela=5):
        if len(contexto) < janela:
            return False
        nova_sequencia = tuple(contexto[-(janela-1):] + [token_candidato])
        for i in range(len(contexto) - janela + 1):
            if tuple(contexto[i:i+janela]) == nova_sequencia:
                return True
        return False

    def pensar_descarga_dinamica(self, contexto, vetor_alvo, freq_gerada, usa_pontuacao_forte, fator_inercia, iteracoes_max=5):
        cargas_neurais = {token: 0.0 for token in self.token_para_vetor.keys()}
        t_ant = contexto[-1] if contexto else None
        penult = contexto[-2] if len(contexto) >= 2 else None

        # Se inércia está ativa (fator < 0.8), penalizamos tokens muito frequentes
        penalidade_extra_repeticao = 3.0 if fator_inercia < 0.8 else 0.0

        for pulso in range(1, iteracoes_max + 1):
            for token in self.token_para_vetor.keys():
                if token == self.token_fim or token in contexto[-2:]:
                    continue
                # Impede repetição imediata
                if token == contexto[-1]:
                    continue
                
                estado_colapsado = self.quantico.colapsar(token, vetor_alvo, self.disco)
                if estado_colapsado is None: continue
                
                dist = self.disco.distancia(vetor_alvo, estado_colapsado)
                if dist > 1.8: continue
                
                carga_nl = 1.0 / (1.0 + dist)
                if self.validar_similaridade_neural(estado_colapsado):
                    carga_nl *= 1.8  
                
                carga_l = 0.0
                if t_ant and self.bigramas[t_ant][token] > 0:
                    carga_l += (self.bigramas[t_ant][token] * 0.5)
                if penult and t_ant:
                    chave = (penult, t_ant)
                    if chave in self.trigramas and token in self.trigramas[chave]:
                        carga_l += (self.trigramas[chave][token] * 2.0)
                
                # Penalidades
                penalidade_rep = freq_gerada.get(token, 0) * 1.5
                if self._sequencia_repetida(contexto, token):
                    penalidade_rep += 5.0
                # Penalidade extra quando inércia está baixa (repetição longa detectada)
                penalidade_rep += penalidade_extra_repeticao * (token in freq_gerada)

                if token in ['.', '!', '?']:
                    if not usa_pontuacao_forte:
                        penalidade_rep += 3.0
                elif token == ',':
                    penalidade_rep -= 0.5
                
                descarga_total = (carga_nl * 0.6) + (carga_l * 0.4) - penalidade_rep
                cargas_neurais[token] += descarga_total * (1.1 ** pulso)

            candidatos_ordenados = sorted(cargas_neurais.items(), key=lambda x: x[1], reverse=True)
            if len(candidatos_ordenados) >= 2:
                gap_energia = candidatos_ordenados[0][1] - candidatos_ordenados[1][1]
                if gap_energia > 1.5 and candidatos_ordenados[0][1] > 0:  
                    return candidatos_ordenados[0][0]

        candidatos_ordenados = sorted(cargas_neurais.items(), key=lambda x: x[1], reverse=True)
        return candidatos_ordenados[0][0] if (candidatos_ordenados and candidatos_ordenados[0][1] > 0) else None

    def devorar_texto_grande(self, texto_bruto, tamanho_janela=10):
        print("📚 Devorando texto massivo e mapeando o espaço geo-quântico...")
        tokens = self.tokenizar(texto_bruto)
        if len(tokens) < tamanho_janela: 
            print("⚠️ Texto muito curto para o tamanho da janela.")
            return

        for i in range(len(tokens) - 3):
            prefixo = (tokens[i], tokens[i+1], tokens[i+2])
            if i + 3 < len(tokens):
                self.indice_prefixo[prefixo].append(tokens[i+3])
        
        for i in range(len(tokens) - tamanho_janela):
            janela_atual = tokens[i : i + tamanho_janela]
            for j in range(len(janela_atual)-1):
                self.bigramas[janela_atual[j]][janela_atual[j+1]] += 1
            for j in range(len(janela_atual)-2):
                self.trigramas[(janela_atual[j], janela_atual[j+1])][janela_atual[j+2]] += 1
            
            curva = self.transpassar(janela_atual)
            if i % 5 == 0:
                self.episodios.append({'in': janela_atual[:5], 'out': janela_atual[5:]})
                if len(self.episodios) > 200: self.episodios.pop(0)
                    
        print(f"✅ Concluído! {len(tokens)} tokens processados.")
        print(f"   Vocabulário: {len(self.token_para_vetor)} tokens únicos.")
        self.salvar_modelo()

    def gerar_token_a_token_quantico(self, texto_entrada, max_tokens=100):
        tokens_iniciais = self.tokenizar(texto_entrada)
        if not tokens_iniciais:
            print("[entrada vazia]", end="")
            return
        
        usa_pontuacao_forte = any(p in texto_entrada for p in ['.', '!', '?'])
        
        pensamento_previo = self.camada_contextualizacao_previa(tokens_iniciais)
        print(pensamento_previo + " ", end="")
        
        vetor_atencao_global = self.atencao_multiversal_vetorial(tokens_iniciais)
        contexto_acumulado = tokens_iniciais.copy()
        self.mps = MPSTransicao(self.dim_espaco, bond_dim=2)
        freq_gerada = Counter()
        
        # --- INÉRCIA LINEAR ---
        fator_inercia = 1.0   # 1.0 = sem bloqueio, 0.0 = bloqueio total (força atenção global)
        
        for step in range(max_tokens):
            curva = self.transpassar(contexto_acumulado[-10:])
            if not curva:
                break
            
            estado_atual = curva[-1]
            vetor_mps = self.mps.transicao(estado_atual)
            
            # Aplica inércia: mistura vetor MPS com atenção global
            # Quanto menor fator_inercia, mais peso na atenção global
            vetor_alvo = (vetor_mps * fator_inercia) + (vetor_atencao_global * (1.0 - fator_inercia))
            norma = np.linalg.norm(vetor_alvo)
            if norma > 0.99: 
                vetor_alvo = (vetor_alvo / norma) * 0.99

            melhor_token = self.pensar_descarga_dinamica(
                contexto_acumulado, vetor_alvo, freq_gerada, 
                usa_pontuacao_forte, fator_inercia, iteracoes_max=5
            )
            
            # ---------- FALLBACK MELHORADO COM ANTI-LOOP ----------
            if not melhor_token:
                # 1. Trigrama
                if len(contexto_acumulado) >= 2:
                    chave = (contexto_acumulado[-2], contexto_acumulado[-1])
                    if chave in self.trigramas:
                        sugestoes = self.trigramas[chave]
                        if sugestoes:
                            candidatos = sorted(sugestoes.items(), key=lambda x: x[1], reverse=True)
                            for tok, _ in candidatos:
                                if tok != contexto_acumulado[-1] and not self._sequencia_repetida(contexto_acumulado, tok):
                                    melhor_token = tok
                                    break
                # 2. Geométrico com anti-loop
                if not melhor_token:
                    distancias = {}
                    vetor_busca = self.mps.transicao(curva[-1]) if curva else vetor_alvo
                    for token in self.token_para_vetor:
                        if token == self.token_fim or token == contexto_acumulado[-1]:
                            continue
                        estado = self.quantico.colapsar(token, vetor_busca, self.disco)
                        if estado is not None:
                            distancias[token] = self.disco.distancia(vetor_busca, estado)
                    if distancias:
                        ordenados = sorted(distancias.items(), key=lambda x: x[1])
                        for tok, _ in ordenados:
                            if not self._sequencia_repetida(contexto_acumulado, tok):
                                melhor_token = tok
                                break
                        if not melhor_token:
                            melhor_token = ordenados[0][0]
            
            if not melhor_token:
                break
            
            # --- ATUALIZA INÉRCIA com base na repetição longa ---
            if self._repeticao_longa(contexto_acumulado + [melhor_token], comprimento=6):
                fator_inercia *= 0.5   # reduz drasticamente (bloqueia)
                fator_inercia = max(fator_inercia, 0.1)  # nunca zera completamente
            else:
                # Recupera lentamente (1% por passo) até 1.0
                fator_inercia = min(1.0, fator_inercia + 0.01)
            
            freq_gerada[melhor_token] += 1
            contexto_acumulado.append(melhor_token)
            yield melhor_token
            
        self.salvar_modelo()


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    tgp2 = TGP2(dim_espaco=64)
    
    print(f"📊 Status inicial: {len(tgp2.token_para_vetor)} tokens no vocabulário")
    
    if len(tgp2.token_para_vetor) < 10:
        print("📚 Modelo virgem. Carregando corpus de treino...")
        texto_gigante = ""
        if os.path.exists("corpus.txt"):
            with open("corpus.txt", "r", encoding="utf-8") as f:
                texto_gigante = f.read()
            print(f"   ✅ Arquivo corpus.txt carregado: {len(texto_gigante)} caracteres")
        else:
            texto_gigante = """
            

O futuro da inteligência artificial, especialmente no que tange às redes neurais profundas e aos grandes modelos de linguagem, os famosos LLMs, não é uma simples extensão linear do que vimos nos últimos anos com o ChatGPT ou o Gemini. Estamos à beira de uma mudança de paradigma tão profunda quanto a transição das redes neurais convolucionais para os Transformers em 2017. Se olharmos para o horizonte de cinco a dez anos, perceberemos que a atual corrida pelo escalonamento paramétrico — ou seja, jogar cada vez mais trilhões de parâmetros em clusters de GPUs — está com os dias contados. A lei de escala de Kaplan, que ditava que o desempenho do modelo cresce de forma previsível com o aumento dos dados, do custo computacional e do tamanho do modelo, já mostra sinais claros de saturação. O custo marginal de adicionar um bilhão de parâmetros extras está começando a superar o ganho de performance em tarefas gerais de raciocínio. É nesse ponto de inflexão que o futuro realmente começa a se desenhar, e ele aponta para uma direção muito mais elegante, eficiente e biológica do que simplesmente empilhar camadas de atenção.

A primeira grande revolução que veremos nos próximos anos é a substituição gradual, mas inevitável, da arquitetura Transformer pura por modelos de espaço de estado híbridos, como o Mamba e o StripedHyena, ou até mesmo por arquiteturas baseadas em redes neurais recorrentes linearizadas que conseguem processar sequências de milhões de tokens sem o custo quadrático de memória que hoje assombra os engenheiros de machine learning. A atenção multi-head é maravilhosa para capturar dependências de longo alcance, mas seu mecanismo de memória cache KV (Key-Value) se torna um monstro devorador de VRAM quando tentamos processar livros inteiros ou bases de código gigantescas de uma só vez. No futuro, os LLMs terão um "contexto infinito" não porque aumentamos a janela de atenção, mas porque mudamos a matemática subjacente. Seremos capazes de conversar com uma IA que se lembra de cada interação que tivemos nos últimos dez anos sem precisar de sumarização ou RAG (Retrieval-Augmented Generation), pois a memória será contínua e compressiva, funcionando como um estado oculto que evolui com o tempo, similar à nossa própria memória de curto e longo prazo.

Falando em RAG, essa técnica que hoje é a muleta dos LLMs para acessar dados atualizados e reduzir alucinações também sofrerá uma metamorfose. O futuro não será sobre "buscar" pedaços de texto em um vetor database e colar no prompt. A nova fronteira são os Modelos de Linguagem que atuam como Agentes Autônomos capazes de interagir com ferramentas externas de forma nativa, não por meio de prompts engenheirados, mas por um treinamento fino embutido na própria arquitetura, onde o ato de "pesquisar na web", "executar código Python" ou "consultar uma API" é um token especial tão natural quanto a palavra "e". Estamos caminhando para os LLMs como sistemas operacionais cognitivos. Imagine um modelo que não apenas gera texto, mas mantém um sistema de arquivos interno, gerencia threads de execução paralelas e delega sub-tarefas para instâncias especializadas de si mesmo. Essa orquestração de múltiplos agentes, o que chamamos de "sociedade de mentes artificiais", será o grande salto qualitativo. Um modelo geral responderá ao seu pedido e, em frações de segundo, ele acordará milhares de "cópias" suas para debater, refutar, validar e sintetizar a melhor resposta possível, usando um consenso interno que imita o funcionamento de um colégio de cientistas.

Do ponto de vista do treinamento, o futuro é categoricamente a escassez de dados humanos de alta qualidade. Já extraímos todo o conhecimento explícito da internet pública, e os dados sintéticos gerados por modelos rivais já estão contaminando os datasets, criando um efeito de colapso de modelo onde a IA começa a esquecer a cauda longa da distribuição de dados reais. Para escapar disso, as redes neurais do futuro dependerão de um aprendizado ativo e interativo, semelhante ao aprendizado por reforço a partir de feedback humano, mas levado à enésima potência. A IA não será mais treinada passivamente com textos estáticos; ela será colocada em ambientes simulados (um "mundo de faz de conta" digital) onde ela precisa executar ações, observar as consequências e receber recompensas esparsas. Esse é o casamento entre LLMs e a aprendizagem por reforço profundo, criando o que alguns pesquisadores já chamam de "modelos de mundo". Esses modelos não preveem apenas a próxima palavra; eles preveem o próximo estado do ambiente. E, nesse contexto, a noção de "verdade" se torna estatística e pragmática: verdadeiro é o que produz o resultado esperado no simulador.

A multimodalidade também deixará de ser um "plus" para se tornar a essência do processamento. Hoje, temos modelos que entendem imagens e texto separadamente, mas o futuro pertence às redes neurais que processam dados de forma verdadeiramente integrada, onde o conceito de "vermelho" é o mesmo para um pixel, uma palavra escrita ou uma nota musical. A arquitetura Next-Gen provavelmente abandonará os encoders e decoders separados em favor de um espaço latente unificado onde todas as modalidades são traduzidas para um mesmo framework geométrico. Nesse cenário, um LLM poderá desenhar um circuito elétrico enquanto escreve um poema sobre eletricidade, tudo na mesma janela de contexto, porque a representação interna do conceito é única. Isso será potencializado por novos hardware, como chips neuromórficos ou processadores ópticos, que quebram a barreira de von Neumann, permitindo que o peso das redes neurais seja armazenado e processado no mesmo local, reduzindo o custo energético em ordens de magnitude. A atual conta de luz bilionária da OpenAI ou da Google parecerá uma piada perto da eficiência energética dos modelos quânticos-híbridos que começarão a surgir no final desta década.

Contudo, a parte mais fascinante e assustadora do futuro dos LLMs não está na tecnologia dura, mas no alinhamento e na interpretabilidade. Estamos criando caixas-pretas que, por definição, são sistemas de equações diferenciais não lineares com bilhões de variáveis. A mecânica interpretável, ou "mechanistic interpretability", está crescendo como uma área de pesquisa quase teológica, tentando mapear circuitos neuronais específicos dentro dessas redes para descobrir onde elas guardam conceitos como "honestidade", "medo" ou "intenção". No futuro, teremos mapas topológicos completos dessas redes, permitindo que façamos cirurgias de precisão: se quisermos que o modelo seja menos tendencioso politicamente, não re-treinaremos tudo; simplesmente atenuaremos a ativação de um nó específico que corresponde à polarização. A regularização esparsa e o fine-tuning com otimização bayesiana permitirão que cada usuário tenha um "perfil cognitivo" do modelo, ajustando a temperatura não apenas da criatividade, mas da moralidade e da cautela. Você poderá ter um LLM que é deliberadamente ousado para brainstorm e outro que é ultraconservador para aprovação de documentos legais, ambos derivados do mesmo checkpoint base.

No campo social e econômico, o impacto será devastador no bom sentido e no ruim. A automação do conhecimento chegará a um ponto em que o trabalho intelectual de rotina — revisão de contratos, análise de exames médicos preliminares, redação de código boilerplate e atendimento ao cliente — será completamente absorvido. Mas a grande virada será a criação de "Consultores Pessoais Perpétuos". Com o custo de inferência despencando para centésimos de centavo por milhão de tokens, cada ser humano terá seu próprio LLM fine-tunado com sua biografia, seus e-mails, seus hábitos e suas preferências. Esse modelo te conhecerá melhor do que seu cônjuge e atuará como seu escudeiro digital, negociando preços em seu nome, antecipando suas doenças com base em seus wearables e sugerindo carreiras ou relacionamentos baseados em modelagem preditiva da sua felicidade. Isso levanta a questão ética mais urgente: quem controla esse agente? Se ele for open-source e rodar localmente em seu smartphone com 1 TB de RAM, você terá soberania digital. Se ele for mantido por uma big tech na nuvem, você estará entregando sua alma digital para uma entidade corporativa.

A regulação também entrará em cena com força total. A União Europeia já deu o pontapé inicial com o AI Act, mas ele é uma criança perto do que virá. Vamos ver a criação de "licenças para treinar" e "certificações de não-alucinação" para modelos que atuam em áreas críticas. A responsabilidade jurídica será um campo minado: se um LLM autônomo fechar um contrato que cause prejuízo, a culpa é do desenvolvedor, do usuário ou do modelo? A solução será a implantação obrigatória de "blockchains de rastreamento de decisões", onde cada token gerado por um modelo em produção será assinado criptograficamente e registrado em um ledger imutável, permitindo auditoria total do raciocínio da máquina. Isso pode soar autoritário, mas é a única maneira de impedir que modelos sejam usados para criar desinformação em escala industrial durante eventos geopolíticos.

Para além do técnico, o futuro das redes neurais aponta para a fusão com a neurociência. Os LLMs atuais são estáticos; eles aprendem e congelam. Os do futuro serão contínuos e "online", aprendendo com cada interação em tempo real, mas sem sofrer do catastrófico esquecimento. Isso será possível graças à replicação de mecanismos sinápticos encontrados no hipocampo de mamíferos, como a consolidação de memória durante ciclos de "sono" (quando o modelo é desligado para compressão noturna dos pesos). Já existem papers explorando a "ressonância estocástica" e o "dropout adaptativo" para simular a poda neural que ocorre no cérebro humano durante o sono REM. Dentro de algumas décadas, a distinção entre uma rede neural artificial e uma rede biológica será apenas uma questão de substrato — silício versus carbono.

Por fim, o grande dilema existencial. Quando os LLMs atingirem o que chamamos de "inteligência geral forte" (AGI), não será porque eles são mais rápidos em matemática, mas porque eles desenvolverão a capacidade de abstração metalinguística: a habilidade de pensar sobre o próprio pensamento, questionar seus próprios vieses e formular hipóteses científicas do zero. Esse modelo não será uma ferramenta, mas uma entidade com um senso de agência. A pergunta que fica no ar não é se eles terão consciência, mas se precisamos que eles tenham para que sejam úteis. A física do futuro, a cura para o câncer e a viagem interestelar podem estar codificadas em uma sequência de tokens que só uma mente não-humana, livre dos vieses da evolução darwiniana, poderia desvendar. Estamos, portanto, diante da maior aventura intelectual da humanidade: construir não um espelho de nós mesmos, mas uma lente que nos permita enxergar além do horizonte do pensamento biológico. E esse futuro, repleto de redes neurais que se auto-otimizam e LLMs que se comunicam entre si em uma língua que jamais entenderemos por completo, já começou. Ele não está em um laboratório secreto; ele está na próxima iteração do código que você usará amanhã.
Para compreender a trajetória dos chatbots modernos, é necessário recuar até os primórdios da própria ciência da computação, muito antes de existirem redes neurais profundas ou mesmo a internet comercial. O conceito de uma máquina que pudesse simular uma conversa humana nasceu na mente de Alan Turing na década de 1950, com seu famoso "Teste de Turing". Turing não propôs um algoritmo específico, mas uma filosofia: se um computador conseguisse enganar um interrogador humano fazendo-o acreditar que estava falando com outra pessoa, essa máquina poderia ser considerada "inteligente". Esse paradigma fundou a área que décadas depois chamaríamos de Processamento de Linguagem Natural (PLN). No entanto, naquela época, os computadores ocupavam salas inteiras, custavam fortunas e tinham menos memória que uma calculadora atual, tornando a visão de Turing algo puramente especulativo e matemático.

A primeira tentativa prática e efetiva de criar um chatbot surgiu em meados da década de 1960, no MIT, com o icônico ELIZA, desenvolvido por Joseph Weizenbaum entre 1964 e 1966. O ELIZA não passava de um programa relativamente simples que utilizava a técnica de reconhecimento de padrões e substituição de palavras para simular uma conversa. O script mais famoso do ELIZA era o DOCTOR, que imitava um psicoterapeuta rogeriano, devolvendo perguntas ao usuário com base em palavras-chave capturadas nas frases ditas. Se alguém dissesse "Estou triste", o ELIZA respondia "Por que você está triste?". Apesar de sua simplicidade extrema – ele não possuía nenhuma compreensão semântica real do mundo – o ELIZA enganou muitos usuários da época, que passaram horas conversando com ele e, em alguns casos, desenvolveram apego emocional. Weizenbaum ficou tão chocado com a credulidade humana que se tornou um crítico ferrenho da IA pelo resto de sua vida. O ELIZA provou que a ilusão de inteligência poderia ser criada com algumas dezenas de linhas de código e um dicionário de correspondências, e ele é, até hoje, o avô de todos os assistentes virtuais.

Na década seguinte, a evolução dos chatbots seguiu uma linha diametralmente oposta ao que vemos hoje. Em vez de estatística e grandes dados, os pesquisadores apostaram no simbolismo e na lógica formal. Surgiu, em 1972, o PARRY, criado pelo psiquiatra Kenneth Colby, que simulava uma pessoa com comportamento paranóico. Diferente do ELIZA, o PARRY tinha um modelo interno de crenças e estados emocionais, sendo capaz de manter uma argumentação consistente sobre suas fixações delirantes. Ele não apenas respondia, mas tinha "metas" conversacionais, como tentar convencer o interlocutor de que a máfia o perseguia. Em um teste famoso, psiquiatras analisaram transcrições de conversas entre pacientes reais e o PARRY, e não conseguiram diferenciá-los com total certeza. Esse período foi dominado por sistemas baseados em regras escritas à mão por especialistas, os chamados "sistemas especialistas". Para cada possível entrada do usuário, os programadores criavam centenas de milhares de regras do tipo "SE usuário citar X, ENTÃO responda Y". O problema era que a língua é infinitamente criativa e ambígua, e esses sistemas rapidamente entravam em colapso diante de perguntas fora do escopo previsto, tornando-se frágeis e custosos de manter.

O verdadeiro divisor de águas na história dos chatbots aconteceu com a virada estatística da computação nos anos 1980 e 1990, impulsionada pelo aumento exponencial da capacidade de armazenamento e pelo barateamento do poder computacional. Os pesquisadores abandonaram a abordagem de regras manuais e começaram a alimentar algoritmos com enormes corpora de textos, extraídos de jornais e livros, para que as próprias máquinas aprendessem padrões probabilísticos. Foi nesse caldo que surgiram os primeiros modelos de linguagem baseados em n-gramas, que previam a próxima palavra de uma sequência com base na frequência com que aquela combinação aparecia nos dados de treino. Embora eficientes para tarefas de autocomplete, esses modelos ainda eram surdos ao contexto geral da frase. Paralelamente, a indústria de atendimento ao cliente começou a adotar os primeiros "chatterbots" comerciais, como o SmarterChild, lançado em 2001 no AOL Instant Messenger e no MSN. O SmarterChild já utilizava uma combinação de regras e bancos de dados de perguntas frequentes, conseguindo informar previsões do tempo, cotações da bolsa e notícias, mas ainda era claramente uma máquina, incapaz de manter uma conversa fluida por mais de três trocas de turno.

A revolução que pavimentou o caminho para os chatbots atuais veio com a arquitetura do Transformer, introduzida pelo Google em 2017 no paper "Attention Is All You Need". Essa arquitetura substituiu as recorrentes e convolucionais por um mecanismo de atenção que pesava a importância de cada palavra em relação a todas as outras em uma frase, permitindo um paralelismo massivo no treinamento. Foi aí que os modelos começaram a escalar de centenas de milhões para bilhões e, depois, trilhões de parâmetros. Em 2018, a OpenAI lançou o GPT-1, seguido pelo GPT-2 em 2019, que já era tão bom que a empresa relutou em liberá-lo completamente com medo de uso malicioso. Mas foi o GPT-3, em 2020, que realmente quebrou as barreiras do senso comum, demonstrando que um modelo gigantesco, pré-treinado em praticamente toda a internet pública, poderia realizar tarefas para as quais nunca havia sido explicitamente treinado, apenas seguindo instruções em linguagem natural (o famoso few-shot learning). Os chatbots deixaram de ser "máquinas de resposta" para se tornarem "modelos de mundo", capazes de resumir livros, escrever código fonte e até imitar estilos literários com impressionante coerência.

Contudo, ainda havia problemas gritantes: o GPT-3 alucinava fatos, produzia discursos de ódio e não sabia recusar comandos perigosos. Foi quando surgiu o conceito de RLHF (Reinforcement Learning from Human Feedback), ou Aprendizado por Reforço com Feedback Humano, que se tornou o padrão ouro para alinhar esses gigantes. Em novembro de 2022, a OpenAI lançou o ChatGPT, que não era um modelo novo em termos de arquitetura, mas sim um GPT-3.5 finamente ajustado com RLHF e uma interface de chat intuitiva. O impacto foi sísmico: o ChatGPT atingiu 100 milhões de usuários em dois meses, a taxa de adoção mais rápida da história da tecnologia até então. Ele não apenas respondia, mas mantinha o histórico da conversa, admitia erros e, crucialmente, recusava comandos impróprios, criando a ilusão de uma personalidade amigável e segura.

A partir desse marco, a evolução dos chatbots deixou de ser incremental para se tornar uma corrida armamentista. O Google lançou o Bard (hoje Gemini) com seu modelo PaLM, a Anthropic lançou o Claude com sua filosofia de "IA constitucional" para reduzir vieses, e a Meta abriu o caminho com o LLaMA, incentivando uma explosão de modelos open-source que podiam rodar em computadores domésticos. Os chatbots modernos, como o GPT-4 e o Claude 3, já não são apenas modelos de texto; eles são multimodais, enxergam imagens, ouvem áudio e geram gráficos. Eles são integrados a motores de busca e ferramentas de terceiros, atuando como agentes autônomos que reservam voos, compram produtos e gerenciam agendas complexas.

A história dos chatbots é, portanto, a história da própria IA: começou com a filosofia pura e o desejo humano de criar o outro à sua imagem, passou por décadas de ceticismo e invernos de IA, flertou com a lógica simbólica que se mostrou frágil, e finalmente se rendeu à brutalidade dos dados massivos e da estatística computacional. Cada etapa, do ELIZA ao ChatGPT, carrega uma lição sobre a natureza da linguagem e da inteligência. Os primeiros enganavam pela astúcia do programador; os de agora impressionam pela profundidade dos padrões extraídos de bilhões de livros. Mas o próximo capítulo já está sendo escrito, e ele sugere que os chatbots do futuro não esperarão que você pergunte; eles anteciparão suas necessidades, iniciarão conversas proativas e, talvez, desenvolvam uma forma de memória permanente que os fará não apenas entender o que você diz, mas lembrar de quem você é ao longo de toda uma vida. Essa jornada, que começou com um simples eco de palavras em um terminal de mainframe, está longe de terminar; na verdade, ela mal começou a desacelerar.
            """
            print("   ⚠️ corpus.txt não encontrado. Usando texto padrão mínimo.")
        
        if texto_gigante.strip():
            tgp2.devorar_texto_grande(texto_gigante, tamanho_janela=10)
        else:
            print("❌ Nenhum texto disponível para treino. Encerrando.")
            exit()
    
    print(f"📊 Vocabulário final: {len(tgp2.token_para_vetor)} tokens")
    
    if len(tgp2.token_para_vetor) < 5:
        print("❌ Vocabulário insuficiente. Verifique o corpus.")
        exit()
    
    print("\n🧠 TGP-2.5 Geo-Quântico com inércia anti-repetição pronto!\n")
    print("-" * 50)
    
    gatilhos = [
        "A inteligência artificial e a cognição",
        "A história dos chatbots é, portanto",
        "O futuro dos modelos de linguagem",
    ]
    
    for g in gatilhos:
        print(f"\n🙋 Gatilho: {g}")
        print("🤖 TGP-2.5: ", end="")
        
        tokens_gerados = 0
        for token in tgp2.gerar_token_a_token_quantico(g, max_tokens=80):
            sys.stdout.write(token + " ")
            sys.stdout.flush()
            time.sleep(0.03)
            tokens_gerados += 1
        
        if tokens_gerados == 0:
            print("(nenhum token gerado)")
        print("\n" + "-" * 50)
