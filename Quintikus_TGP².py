import numpy as np
import re
import pickle
import os
from collections import defaultdict, Counter
import sys
import time

# ============================================================
# MÓDULOS MATEMÁTICOS E FÍSICOS (Geo-Quânticos)
# ============================================================

class DiscoPoincare:
    """Espaço hiperbólico para mapeamento hierárquico de tokens."""
    def __init__(self, dim=64):
        self.dim = dim
    
    def distancia(self, x, y):
        norma_x = np.clip(np.linalg.norm(x), 0, 0.99)
        norma_y = np.clip(np.linalg.norm(y), 0, 0.99)
        
        x_proj = x / (np.linalg.norm(x) + 1e-8) * norma_x
        y_proj = y / (np.linalg.norm(y) + 1e-8) * norma_y
        
        diff_sq = np.sum((x_proj - y_proj)**2)
        num = 2 * diff_sq
        den = (1 - norma_x**2) * (1 - norma_y**2)
        
        return np.arccosh(1 + num / den + 1e-8)
    
    def mover_para_borda(self, ponto, intensidade=0.01):
        norma = np.linalg.norm(ponto)
        if norma < 0.99:
            direcao = ponto / (norma + 1e-8)
            return ponto + direcao * intensidade
        return ponto
    
    def mover_para_centro(self, ponto, intensidade=0.01):
        return ponto * (1 - intensidade)


class TokenQuantico:
    """Superposição semântica: tokens habitam múltiplos estados."""
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
        probs = sims**2  # Regra de Born
        probs_soma = probs.sum()
        if probs_soma == 0:
            return melhores[0][1]
            
        probs /= probs_soma
        idx = np.random.choice(len(melhores), p=probs)
        return melhores[idx][1]


class MPSTransicao:
    """Matrix Product State para compressão de transições de estados."""
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
# CLASSE TGP-2 (Geo-Quântica Completa + Consciência Neural)
# ============================================================

class TGP2:
    def __init__(self, dim_espaco=64):
        self.dim_espaco = dim_espaco
        
        self.disco = DiscoPoincare(dim_espaco)
        self.quantico = TokenQuantico(dim_espaco, num_estados=4)
        self.mps = MPSTransicao(dim_espaco, bond_dim=2)
        
        self.token_para_vetor = {}
        self.episodios = []
        
        self.bigramas = defaultdict(lambda: defaultdict(int))
        self.trigramas = defaultdict(lambda: defaultdict(int))
        self.indice_prefixo = defaultdict(list)
        
        self.token_fim = '<END>'
        self._registrar_token(self.token_fim)
        
    def _registrar_token(self, token):
        if token not in self.token_para_vetor:
            vetor = np.random.randn(self.dim_espaco) * 0.4 
            norma = np.linalg.norm(vetor)
            if norma > 0.99: 
                vetor = vetor / norma * 0.99
            
            self.token_para_vetor[token] = vetor
            self.quantico.inicializar_base(token, vetor)

    def tokenizar(self, texto):
        import unicodedata
        texto_nfkd = unicodedata.normalize('NFD', texto.lower())
        texto_sem_acento = ''.join([c for c in texto_nfkd if unicodedata.category(c) != 'Mn'])
        tokens = re.findall(r'[a-z0-9]+|[.,!?;:]+|\s+', texto_sem_acento)
        return [t for t in tokens if t.strip()]
    
    def transpassar(self, tokens):
        tokens = tokens[-20:] if len(tokens) > 20 else tokens 
        if not tokens: 
            return []
        
        for t in tokens:
            self._registrar_token(t)
            
        vetores_base = []
        for t in tokens:
            v = self.token_para_vetor[t].copy()
            if len(t) > 5:
                v = self.disco.mover_para_borda(v, 0.05)
            vetores_base.append(v)
            
        estados_colapsados = []
        for i, t in enumerate(tokens):
            inicio = max(0, i - 2)
            fim = min(len(tokens), i + 3)
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
        """
        Consciência Neural Comparativa:
        Compara o vetor alvo com todos os pesos gravados na memória.
        Valida se existem entre 3 e 7 pesos com alta similaridade geométrica próxima.
        """
        similares_contagem = 0
        for token, vetor_ref in self.token_para_vetor.items():
            if token == self.token_fim:
                continue
            dist = self.disco.distancia(vetor_alvo, vetor_ref)
            if dist <= limiar_distancia:
                similares_contagem += 1
                
        # Retorna True se a densidade neural estiver na faixa desejada (3 a 7)
        return min_similares <= similares_contagem <= max_similares

    def devorar_texto_grande(self, texto_bruto, tamanho_janela=10):
        print("📚 Devorando texto massivo e mapeando o espaço geo-quântico...")
        tokens = self.tokenizar(texto_bruto)
        
        if len(tokens) < tamanho_janela:
            print("Texto muito curto para o tamanho da janela.")
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
                self.episodios.append({
                    'in': janela_atual[:5], 
                    'out': janela_atual[5:], 
                    'c_in': curva[:5], 
                    'c_out': curva[5:]
                })
                if len(self.episodios) > 200:
                    self.episodios.pop(0)
                    
        print(f"✅ Concluído! {len(tokens)} palavras absorvidas e estruturadas.")
        print(f"   Índice de prefixos: {len(self.indice_prefixo)} entradas.")

    def gerar_token_a_token_quantico(self, texto_entrada, max_tokens=20):
        """Geração integrada com busca literal de prefixo e Consciência Neural Comparativa."""
        tokens_iniciais = self.tokenizar(texto_entrada)
        if not tokens_iniciais: 
            return
        
        contexto_acumulado = tokens_iniciais.copy()
        self.mps = MPSTransicao(self.dim_espaco, bond_dim=2)
        
        # 🔍 FASE 1: Busca literal de prefixo flexível
        chave_encontrada = None
        melhor_match_len = 0
        for prefixo_cadastrado in self.indice_prefixo.keys():
            coincidencias = sum(1 for a, b in zip(tokens_iniciais, prefixo_cadastrado) if a == b)
            if coincidencias > melhor_match_len and coincidencias >= 2:
                melhor_match_len = coincidencias
                chave_encontrada = prefixo_cadastrado

        if chave_encontrada:
            contexto_acumulado = list(chave_encontrada)
            for _ in range(max_tokens):
                if len(contexto_acumulado) >= len(tokens_iniciais) + max_tokens:
                    break
                
                if len(contexto_acumulado) >= 3:
                    prefixo_atual = (contexto_acumulado[-3], contexto_acumulado[-2], contexto_acumulado[-1])
                    if prefixo_atual in self.indice_prefixo:
                        sugestoes = self.indice_prefixo[prefixo_atual]
                        proximo_token = Counter(sugestoes).most_common(1)[0][0]
                        contexto_acumulado.append(proximo_token)
                        yield proximo_token
                    else:
                        break
                else:
                    break
            return
        
        # 🔄 FASE 2: Fallback com Consciência Neural Comparativa
        for step in range(max_tokens):
            curva = self.transpassar(contexto_acumulado[-10:])
            if not curva: 
                break
            
            estado_atual = curva[-1]
            vetor_alvo = self.mps.transicao(estado_atual)

            candidatos = []
            scores = []
            t_ant = contexto_acumulado[-1] if contexto_acumulado else None
            penult = contexto_acumulado[-2] if len(contexto_acumulado) >= 2 else None
            
            for token in self.token_para_vetor.keys():
                if token == self.token_fim or token in contexto_acumulado[-3:]: 
                    continue
                    
                estado_colapsado = self.quantico.colapsar(token, vetor_alvo, self.disco)
                if estado_colapsado is None: 
                    continue
                
                dist = self.disco.distancia(vetor_alvo, estado_colapsado)
                sim_base = 1.0 / (1.0 + dist)
                
                # Aplica Consciência Neural Comparativa (validação 3 a 7 similares)
                if self.validar_similaridade_neural(estado_colapsado, limiar_distancia=1.2, min_similares=3, max_similares=7):
                    sim_base *= 2.0  # Bonifica o token se estiver na densidade neural ideal
                
                if penult and t_ant:
                    chave = (penult, t_ant)
                    if chave in self.trigramas and token in self.trigramas[chave]:
                        sim_base *= 3.0
                    
                candidatos.append(token)
                scores.append(sim_base)
            
            if not candidatos: 
                break
            
            scores = np.array(scores)
            probs = scores ** 2
            soma_probs = probs.sum()
            if soma_probs == 0: 
                break
            probs /= soma_probs
            
            melhor_token = np.random.choice(candidatos, p=probs)
            contexto_acumulado.append(melhor_token)
            yield melhor_token

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    tgp2 = TGP2(dim_espaco=512)
    
    texto_gigante = """
    A inteligência artificial é um campo da ciência da computação. O aprendizado de máquina 
    permite que os sistemas identifiquem padrões em grandes volumes de dados. No espaço 
    quântico, a informação existe em múltiplos estados simultaneamente. Quando uma medição 
    ocorre, a função de onda entra em colapso. Modelos vetoriais hiperbólicos organizam a 
    linguagem humana como uma árvore, onde conceitos centrais ficam no meio e conceitos 
    específicos habitam as bordas. A arquitetura TGP tenta unir a física quântica e a 
    geometria não-euclidiana para processamento natural de texto em dispositivos limitados.
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
    
    
    tgp2.devorar_texto_grande(texto_gigante, tamanho_janela=50)
    
    print("\n🧠 TGP-2 Geo-Quântico pronto para interagir!\n")
    print("-" * 50)
    
    gatilhos = [
      
        "Oi, tudo beleza por aí?"
    ]
    
    for g in gatilhos:
        print(f"🙋 Gatilho: {g}")
        print("🤖 TGP-2: ", end="")
        
        for token in tgp2.gerar_token_a_token_quantico(g, max_tokens=12):
            sys.stdout.write(token + " ")
            sys.stdout.flush()
            time.sleep(0.12)
            
        print("\n" + "-" * 50)
