import os
import sys
import math
import time
import pickle
import struct
from array import array


# ============================================
# 🧠 QUINTIKUSFEELING - API DE SENTIMENTOS
# ============================================

class QuintikusFeeling:
    """
    QUINTIKUSFEELING SDK
    Uso:
        model = QuintikusFeeling("visual")    # Interface completa
        model = QuintikusFeeling("terminal")  # Apenas previsão
        
        model.predict("emo.rn", "estou feliz hoje")
        model.trainer("estou triste", "tristeza")
    """
    
    def __init__(self, modo="visual"):
        """
        Inicializa o modelo
        modo: "visual" (interface completa) ou "terminal" (apenas previsão)
        """
        self.modo = modo.lower()
        self.modelo = None
        self.arquivo_padrao = "emo.rn"
        
        # Garante que o arquivo .rn existe
        self._garantir_arquivo()
        
        if self.modo == "visual":
            print("🖥️  QUINTIKUSFEELING - Modo VISUAL ativado")
            print("   Interface completa: treino + previsão + métricas")
        elif self.modo == "terminal":
            print("💻 QUINTIKUSFEELING - Modo TERMINAL ativado")
            print("   Apenas previsão rápida")
        else:
            print(f"⚠️  Modo '{modo}' não reconhecido. Use 'visual' ou 'terminal'")
            print("   Usando 'visual' como padrão")
            self.modo = "visual"
    
    def _garantir_arquivo(self, arquivo=None):
        """Garante que o arquivo .rn existe, cria se não existir"""
        if arquivo is None:
            arquivo = self.arquivo_padrao
        
        if not os.path.exists(arquivo):
            print(f"📄 Arquivo '{arquivo}' não encontrado. Criando novo modelo...")
            try:
                neuro_temp = NeuroMicro(arquivo)
                neuro_temp._salvar_rede()
                print(f"✅ Arquivo '{arquivo}' criado com sucesso!")
            except Exception as e:
                print(f"❌ Erro ao criar '{arquivo}': {e}")
    
    def predict(self, arquivo_rn, texto):
        """
        Faz previsão de sentimento
        Uso: resultado = model.predict("emo.rn", "estou feliz hoje")
        """
        # Garante que arquivo existe
        self._garantir_arquivo(arquivo_rn)
        
        try:
            # Carrega modelo do arquivo .rn
            neuro = NeuroMicro(arquivo_rn)
            
            # Faz previsão
            resultado = neuro.prever(texto)
            
            # Emojis
            emojis = {
                'alegria': '😊', 'tristeza': '😢', 'raiva': '😡',
                'medo': '😨', 'surpresa': '😲', 'nojo': '🤢'
            }
            
            sentimento = resultado['sentimento']
            emoji = emojis.get(sentimento, '🤔')
            
            print(f"\n  📝 Texto: '{texto[:60]}'")
            print(f"  🎯 Sentimento: {sentimento.upper()} {emoji}")
            print(f"  📊 Confiança: {resultado['confianca']:.4f}")
            print(f"  ⚡ Tempo: {resultado['tempo_us']:.2f} μs")
            
            return resultado
        
        except FileNotFoundError:
            print(f"\n  ❌ Arquivo '{arquivo_rn}' não encontrado!")
            return {'sentimento': 'erro', 'confianca': 0.0, 'erro': 'arquivo nao encontrado'}
        except Exception as e:
            print(f"\n  ❌ Erro na previsão: {str(e)}")
            return {'sentimento': 'erro', 'confianca': 0.0, 'erro': str(e)}
    
    def trainer(self, frase, emocao, arquivo_rn=None):
        """
        Treina o modelo com uma frase e emoção
        Uso: model.trainer("estou triste", "tristeza")
             model.trainer("estou feliz", "alegria", "meu_modelo.rn")
        """
        if arquivo_rn is None:
            arquivo_rn = self.arquivo_padrao
        
        # Garante que arquivo existe
        self._garantir_arquivo(arquivo_rn)
        
        try:
            # Carrega modelo
            neuro = NeuroMicro(arquivo_rn)
            
            # Treina
            resultado = neuro.treinar(frase, emocao)
            
            if resultado:
                print(f"\n  ✅ TREINO CONCLUÍDO!")
                print(f"  📝 Frase: '{frase[:50]}'")
                print(f"  🎯 Emoção: {emocao}")
                print(f"  📉 Loss: {resultado['loss_antes']:.4f} → {resultado['loss_depois']:.4f}")
                print(f"  📈 Melhora: {resultado['melhora']:.6f}")
                print(f"  💾 Salvo em: {arquivo_rn}")
            
            return resultado
        
        except PermissionError:
            print(f"\n  ❌ Sem permissão para salvar em '{arquivo_rn}'!")
            return None
        except Exception as e:
            print(f"\n  ❌ Erro no treino: {str(e)}")
            return None
    
    def trainer_lote(self, exemplos, arquivo_rn=None, intensivo=False):
        """
        Treina com múltiplos exemplos
        Uso: model.trainer_lote([
                ("frase 1", "alegria"),
                ("frase 2", "tristeza")
             ])
        """
        if arquivo_rn is None:
            arquivo_rn = self.arquivo_padrao
        
        # Garante que arquivo existe
        self._garantir_arquivo(arquivo_rn)
        
        try:
            neuro = NeuroMicro(arquivo_rn)
            
            if intensivo:
                resultados = neuro.treinar_lote_adaptativo(exemplos, repeticoes=3)
            else:
                resultados = neuro.treinar_lote(exemplos)
            
            print(f"\n  ✅ LOTE CONCLUÍDO!")
            print(f"  📚 {len(resultados)}/{len(exemplos)} exemplos processados")
            print(f"  💾 Salvo em: {arquivo_rn}")
            
            return resultados
        
        except Exception as e:
            print(f"\n  ❌ Erro no treino em lote: {str(e)}")
            return []
    
    def info(self, arquivo_rn=None):
        """Mostra informações do modelo"""
        if arquivo_rn is None:
            arquivo_rn = self.arquivo_padrao
        
        # Garante que arquivo existe
        self._garantir_arquivo(arquivo_rn)
        
        try:
            neuro = NeuroMicro(arquivo_rn)
            neuro.mostrar_estado()
            neuro.mostrar_metricas()
        except Exception as e:
            print(f"\n  ❌ Erro ao carregar informações: {str(e)}")


# ============================================
# 🧠 CLASSE NEUROMICRO (MANTIDA IGUAL)
# ============================================

class NeuroMicro:
    def __init__(self, arquivo_rede="emo.rn"):
        self.arquivo_rede = arquivo_rede
        
        # BYTES puro - zero alocação dinâmica
        self.padroes_bytes = {
            0: (b'alegria', [b'amo', b'feliz', b'boa', b'gratidao', b'sorriso', b'radiante', b'conquista', b'vitoria', b'maravilhoso', b'excelente']),
            1: (b'tristeza', [b'triste', b'dor', b'saudade', b'choro', b'perda', b'melancolia', b'desanimo', b'sofrimento', b'lagrimas']),
            2: (b'raiva', [b'odeio', b'odio', b'raiva', b'irritado', b'furia', b'bug', b'erro', b'indignado', b'revoltado', b'colera']),
            3: (b'medo', [b'medo', b'ansioso', b'ansiedade', b'panico', b'temor', b'inseguro', b'preocupado', b'apreensivo', b'aterrorizado']),
            4: (b'surpresa', [b'uau', b'nossa', b'incrivel', b'framework', b'impressionante', b'chocado', b'inesperado', b'revelacao']),
            5: (b'nojo', [b'nojento', b'asco', b'repulsa', b'horrivel', b'desgosto', b'podre', b'abominavel', b'asqueroso', b'nauseante'])
        }
        
        # Array C-contíguo pra cache line (64 bytes)
        self.pesos = array('f', [1.0/6] * 6)
        
        # Tabela de lookup pra exp() - 4KB em L1 cache
        self.exp_lut = array('f', [math.exp(i/100.0) for i in range(-500, 500)])
        
        # Pré-compilação de padrões
        self._precompilar_padroes()
        
        # Histórico de treinamento
        self.historico_treino = []
        self.total_treinos = 0
        self.taxa_aprendizado = 0.01
        
        # Métricas de performance
        self.metricas = {
            'min_tempo': float('inf'),
            'max_tempo': 0.0,
            'total_analises': 0,
            'soma_tempos': 0.0
        }
        
        # Carrega rede salva se existir
        self._carregar_rede()
    
    def _precompilar_padroes(self):
        """Pré-compila padrões pra busca otimizada"""
        self.busca_plana = []
        for idx, (emocao, palavras) in self.padroes_bytes.items():
            for palavra in palavras:
                self.busca_plana.append((idx, palavra, len(palavra)))
        self.busca_plana.sort(key=lambda x: x[2], reverse=True)
    
    def fast_exp(self, x):
        """exp() com lookup table - 20x mais rápido"""
        idx = int(x * 100) + 500
        if 0 <= idx < 1000:
            return self.exp_lut[idx]
        return 0.0 if x < -5 else float('inf')
    
    def _softmax(self, scores):
        """Softmax com fast_exp"""
        max_s = max(scores)
        exp_scores = [self.fast_exp(s - max_s) for s in scores]
        total_exp = sum(exp_scores)
        
        if total_exp > 0.0001:
            inv_total = 1.0 / total_exp
            return [s * inv_total for s in exp_scores]
        return [1.0/6] * 6
    
    def _entropia_cruzada(self, probs, idx_alvo):
        """Loss de entropia cruzada"""
        epsilon = 1e-10
        p = max(probs[idx_alvo], epsilon)
        return -math.log(p)
    
    def _gradiente_descendente(self, probs, idx_alvo):
        """Atualiza pesos com gradiente de entropia"""
        y_true = [0.0] * 6
        y_true[idx_alvo] = 1.0
        
        for i in range(6):
            gradiente = probs[i] - y_true[i]
            self.pesos[i] -= self.taxa_aprendizado * gradiente
            self.pesos[i] = max(0.001, self.pesos[i])
        
        soma = sum(self.pesos)
        for i in range(6):
            self.pesos[i] /= soma
    
    def analisar_us(self, texto_bytes):
        """Análise de sentimento em microsegundos"""
        t0 = time.perf_counter_ns()
        
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        for idx, palavra, tamanho in self.busca_plana:
            if palavra in texto_bytes:
                scores[idx] += self.pesos[idx]
        
        probs = self._softmax(scores)
        
        dt = (time.perf_counter_ns() - t0) / 1000.0
        
        self.metricas['total_analises'] += 1
        self.metricas['soma_tempos'] += dt
        self.metricas['min_tempo'] = min(self.metricas['min_tempo'], dt)
        self.metricas['max_tempo'] = max(self.metricas['max_tempo'], dt)
        
        return probs, dt, scores
    
    def treinar(self, texto, emocao_alvo):
        """Treina a rede com um exemplo"""
        if isinstance(texto, str):
            texto_bytes = texto.encode('ascii', errors='ignore')
        else:
            texto_bytes = texto
        
        nomes_emocao = ['alegria', 'tristeza', 'raiva', 'medo', 'surpresa', 'nojo']
        if emocao_alvo not in nomes_emocao:
            print(f"❌ Emoção '{emocao_alvo}' não reconhecida")
            return None
        
        idx_alvo = nomes_emocao.index(emocao_alvo)
        
        probs, tempo, scores = self.analisar_us(texto_bytes)
        loss_antes = self._entropia_cruzada(probs, idx_alvo)
        
        self._gradiente_descendente(probs, idx_alvo)
        
        probs_depois, _, _ = self.analisar_us(texto_bytes)
        loss_depois = self._entropia_cruzada(probs_depois, idx_alvo)
        
        self.total_treinos += 1
        registro = {
            'id': self.total_treinos,
            'texto': texto_bytes.decode('ascii', errors='ignore')[:50],
            'alvo': emocao_alvo,
            'loss_antes': round(loss_antes, 6),
            'loss_depois': round(loss_depois, 6),
            'melhora': round(loss_antes - loss_depois, 6),
            'pesos': [round(p, 4) for p in self.pesos],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.historico_treino.append(registro)
        
        self._salvar_rede()
        
        return {
            'loss_antes': loss_antes,
            'loss_depois': loss_depois,
            'melhora': loss_antes - loss_depois,
            'emocao_predita': nomes_emocao[probs.index(max(probs))]
        }
    
    def treinar_lote(self, exemplos):
        """Treina com lote de exemplos"""
        resultados = []
        for texto, emocao in exemplos:
            r = self.treinar(texto, emocao)
            if r:
                resultados.append(r)
        return resultados
    
    def treinar_lote_adaptativo(self, exemplos, repeticoes=3):
        """Treina com lote repetindo pra reforçar padrões"""
        resultados_finais = []
        for epoca in range(repeticoes):
            for texto, emocao in exemplos:
                r = self.treinar(texto, emocao)
                if r and epoca == repeticoes - 1:
                    resultados_finais.append(r)
        return resultados_finais
    
    def prever(self, texto):
        """Prevê sentimento de um texto"""
        if isinstance(texto, str):
            texto_bytes = texto.encode('ascii', errors='ignore')
        else:
            texto_bytes = texto
        
        probs, tempo, scores = self.analisar_us(texto_bytes)
        
        nomes = ['alegria', 'tristeza', 'raiva', 'medo', 'surpresa', 'nojo']
        max_idx = max(range(6), key=lambda i: probs[i])
        
        entropia = 0.0
        for p in probs:
            if p > 0.001:
                entropia -= p * math.log2(p)
        entropia_max = math.log2(6)
        certeza = 1.0 - (entropia / entropia_max if entropia_max > 0 else 0)
        
        return {
            'sentimento': nomes[max_idx],
            'confianca': round(probs[max_idx], 4),
            'certeza': round(certeza, 4),
            'entropia': round(entropia, 4),
            'probabilidades': dict(zip(nomes, [round(p, 4) for p in probs])),
            'tempo_us': round(tempo, 2)
        }
    
    def sentiment_string(self, texto_str):
        """Interface amigável: string → sentimento"""
        return self.prever(texto_str)
    
    def _salvar_rede(self):
        """Salva rede neural em arquivo .rn"""
        dados = {
            'pesos': list(self.pesos),
            'total_treinos': self.total_treinos,
            'historico_treino': self.historico_treino[-1000:],
            'taxa_aprendizado': self.taxa_aprendizado,
            'metricas': {
                'total_analises': self.metricas['total_analises'],
                'soma_tempos': self.metricas['soma_tempos']
            }
        }
        try:
            with open(self.arquivo_rede, 'wb') as f:
                pickle.dump(dados, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"⚠️ Erro ao salvar rede: {e}")
    
    def _carregar_rede(self):
        """Carrega rede neural do arquivo .rn"""
        if not os.path.exists(self.arquivo_rede):
            print(f"📄 Arquivo '{self.arquivo_rede}' não encontrado. Rede zerada.")
            return False
        try:
            with open(self.arquivo_rede, 'rb') as f:
                dados = pickle.load(f)
            pesos_carregados = dados.get('pesos', [1.0/6]*6)
            for i, p in enumerate(pesos_carregados):
                self.pesos[i] = p
            self.total_treinos = dados.get('total_treinos', 0)
            self.historico_treino = dados.get('historico_treino', [])
            self.taxa_aprendizado = dados.get('taxa_aprendizado', 0.01)
            print(f"✅ Rede carregada de '{self.arquivo_rede}'")
            print(f"   Treinos anteriores: {self.total_treinos}")
            print(f"   Pesos: {[round(p, 4) for p in self.pesos]}")
            return True
        except Exception as e:
            print(f"⚠️ Erro ao carregar rede: {e}")
            return False
    
    def mostrar_estado(self):
        """Mostra estado completo da rede"""
        print("\n" + "="*60)
        print("🧠 ESTADO DA REDE NEURAL")
        print("="*60)
        nomes = ['alegria', 'tristeza', 'raiva', 'medo', 'surpresa', 'nojo']
        for i, (nome, peso) in enumerate(zip(nomes, self.pesos)):
            barra = "█" * int(peso * 50)
            print(f"  {nome:12} | {peso:.4f} ({peso*100:5.1f}%) | {barra}")
        print(f"\n  📚 Total treinos: {self.total_treinos}")
        print(f"  📐 Taxa aprendizado: {self.taxa_aprendizado}")
        print(f"  💾 Arquivo: {self.arquivo_rede}")
        if self.historico_treino:
            print(f"\n  📜 Últimos 3 treinos:")
            for t in self.historico_treino[-3:]:
                print(f"    [{t['id']}] '{t['texto']}' → {t['alvo']}")
                print(f"        Loss: {t['loss_antes']:.4f} → {t['loss_depois']:.4f} (Δ{t['melhora']:.4f})")
    
    def mostrar_metricas(self):
        """Mostra estatísticas de performance"""
        m = self.metricas
        if m['total_analises'] > 0:
            media = m['soma_tempos'] / m['total_analises']
            print(f"\n⚡ MÉTRICAS DE PERFORMANCE:")
            print(f"  Análises: {m['total_analises']}")
            print(f"  Média: {media:.2f} μs")
            print(f"  Mínimo: {m['min_tempo']:.2f} μs")
            print(f"  Máximo: {m['max_tempo']:.2f} μs")
            print(f"  Throughput: {1_000_000/media:.0f} textos/s")
    
    def analisar_lote(self, lista_textos_bytes):
        """Processa lote de textos"""
        t0 = time.perf_counter_ns()
        resultados = []
        for texto in lista_textos_bytes:
            probs, _, _ = self.analisar_us(texto)
            resultados.append(probs)
        dt_total = (time.perf_counter_ns() - t0) / 1000.0
        return resultados, dt_total


# ============================================
# SISTEMA DE TREINAMENTO (MENU VISUAL)
# ============================================

def menu_treino():
    """Interface de treinamento interativo"""
    print("\n" + "="*60)
    print("🎓 MENU DA REDE NEURAL")
    print("="*60)
    print("  1. Treinar com 1 exemplo")
    print("  2. Treinar com vários exemplos")
    print("  3. Testar predição de texto")
    print("  4. Ver estado da rede")
    print("  5. Ver métricas de performance")
    print("  6. Fazer benchmark de velocidade")
    print("  7. Resetar rede (apaga treinos)")
    print("  8. Salvar e sair")
    return input("  Escolha: ")

def escolher_emocao():
    """Interface amigável pra escolher emoção"""
    print("  🎯 EMOÇÕES DISPONÍVEIS:")
    print("    1 - 😊 alegria")
    print("    2 - 😢 tristeza")
    print("    3 - 😡 raiva")
    print("    4 - 😨 medo")
    print("    5 - 😲 surpresa")
    print("    6 - 🤢 nojo")
    
    mapa = {
        '1': 'alegria', '2': 'tristeza', '3': 'raiva',
        '4': 'medo', '5': 'surpresa', '6': 'nojo',
        'alegria': 'alegria', 'tristeza': 'tristeza', 'raiva': 'raiva',
        'medo': 'medo', 'surpresa': 'surpresa', 'nojo': 'nojo'
    }
    
    while True:
        escolha = input("  Escolha (1-6 ou nome): ").lower().strip()
        if escolha in mapa:
            return mapa[escolha]
        if escolha in ['sair', 'cancelar', 'exit']:
            return 'cancelar'
        print("  ❌ Inválido! Use 1-6 ou nome da emoção.")

def start():
     # Verifica argumentos de linha de comando
    if len(sys.argv) > 1:
        modo = sys.argv[1]  # "visual" ou "terminal"
    else:
        modo = "visual"  # padrão
    
    # ===== MODO TERMINAL (APENAS PREVISÃO) =====
    if modo == "terminal":
        print("="*60)
        print("💻 QUINTIKUSFEELING - Modo Terminal")
        print("="*60)
        
        model = QuintikusFeeling("terminal")
        
        if len(sys.argv) > 2:
            # Uso: python script.py terminal "texto para analisar"
            texto = sys.argv[2]
            resultado = model.predict("emo.rn", texto)
        else:
            # Modo interativo simples
            print("  Digite 'sair' para encerrar\n")
            while True:
                texto = input("  📝 Texto: ")
                if texto.lower() in ['sair', 'exit', 'quit', '']:
                    print("  👋 Até mais!")
                    break
                model.predict("emo.rn", texto)
    
    # ===== MODO VISUAL (INTERFACE COMPLETA) =====
    else:
        print("="*60)
        print("🖥️  QUINTIKUSFEELING - Modo Visual")
        print("="*60)
        
        # Exemplo de uso da API
        print("\n📚 EXEMPLO DE USO DA API:")
        print("  model = QuintikusFeeling('visual')")
        print("  model.predict('emo.rn', 'estou feliz hoje')")
        print("  model.trainer('estou triste', 'tristeza')\n")
        
        # Inicializa rede
        neuro = NeuroMicro("emo.rn")
        
        # Treino base automático se primeiro uso
        if neuro.total_treinos == 0:
            print("\n📚 PRIMEIRO USO - TREINANDO COM EXEMPLOS BASE...")
            exemplos_base = [
                ("estou muito feliz e radiante hoje", "alegria"),
                ("amo programar e estou grato", "alegria"),
                ("conquista e vitória maravilhosa", "alegria"),
                ("que tristeza profunda sinto dor", "tristeza"),
                ("estou com saudade e melancolia", "tristeza"),
                ("perda e desanimo total", "tristeza"),
                ("odeio bugs e me dá raiva", "raiva"),
                ("que ódio desse erro irritante", "raiva"),
                ("estou furioso e indignado", "raiva"),
                ("estou ansioso e com medo", "medo"),
                ("sinto pânico e insegurança", "medo"),
                ("preocupado e temeroso", "medo"),
                ("uau que framework incrível", "surpresa"),
                ("nossa que impressionante", "surpresa"),
                ("estou chocado com isso", "surpresa"),
                ("isso é nojento e horrível", "nojo"),
                ("me dá asco e repulsa", "nojo"),
                ("código podre e abominável", "nojo"),
            ]
            
            for texto, emocao in exemplos_base:
                r = neuro.treinar(texto, emocao)
                if r:
                    print(f"  ✅ '{texto[:40]}...' → {emocao}")
            
            print(f"\n✅ Treino base concluído! {neuro.total_treinos} exemplos.")
        
        # Mostra estado
        neuro.mostrar_estado()
        neuro.mostrar_metricas()
        
        # Testes rápidos
        print("\n🧪 TESTES RÁPIDOS:")
        testes = [
            "estou feliz com meu código",
            "que raiva desse bug",
            "sinto ansiedade com prazos",
            "nossa que incrível essa lib",
            "código nojento e podre",
            "estou triste mas vai melhorar",
        ]
        
        for texto in testes:
            r = neuro.prever(texto)
            print(f"  '{texto}' → {r['sentimento']} ({r['confianca']:.2f})")
        
        # Loop principal
        while True:
            opcao = menu_treino()
            
            if opcao == '1':
                print("\n  📝 TREINAR COM 1 EXEMPLO")
                print("  (Digite 'sair' para cancelar)")
                texto = input("  Digite o texto: ")
                if not texto or texto.lower() in ['sair', 'exit', 'quit', 'fim', 'cancelar']:
                    print("  👋 Operação cancelada.")
                    continue
                emocao = escolher_emocao()
                if emocao == 'cancelar':
                    print("  👋 Operação cancelada.")
                    continue
                r = neuro.treinar(texto, emocao)
                if r:
                    print(f"\n  ✅ TREINO CONCLUÍDO!")
                    print(f"  Texto: '{texto[:50]}...'")
                    print(f"  Emoção alvo: {emocao}")
                    print(f"  Loss antes: {r['loss_antes']:.6f}")
                    print(f"  Loss depois: {r['loss_depois']:.6f}")
                    print(f"  Melhora: {r['melhora']:.6f}")
            
            elif opcao == '2':
                print("\n  📝 TREINAR COM VÁRIOS EXEMPLOS")
                print("  ═══════════════════════════════════════")
                print("  📌 COMANDOS: Enter vazio = treinar | 'sair' = sair")
                print("  'finalizar','fim','terminar','treinar','go','ok' = treinar agora")
                print("  'cancelar','cancela' = cancelar | 'pular','skip' = pular")
                print("  ═══════════════════════════════════════\n")
                
                exemplos = []
                num = 1
                
                while True:
                    print(f"  ─── Exemplo {num} ───")
                    texto = input("  📝 Texto: ")
                    texto_lower = texto.lower().strip()
                    
                    if not texto or texto_lower in ['finalizar', 'fim', 'terminar', 'treinar', 'train', 'go', 'ok', 'processar', 'feito', 'concluir']:
                        if exemplos:
                            print(f"\n  ⚡ Finalizando e treinando com {len(exemplos)} exemplos!")
                        else:
                            print(f"\n  ⚠️ Nenhum exemplo adicionado.")
                        break
                    
                    if texto_lower in ['sair', 'exit', 'quit', 'abort', 'abortar']:
                        print(f"  🗑️ Saindo sem treinar.")
                        exemplos = []
                        break
                    
                    if texto_lower in ['cancelar', 'cancel', 'cancela']:
                        print(f"  ❌ Operação cancelada.")
                        exemplos = []
                        break
                    
                    if texto_lower in ['pular', 'pula', 'skip', 'pass', 'proximo', 'next']:
                        print(f"  ⏭️ Exemplo {num} pulado.\n")
                        continue
                    
                    emocao = escolher_emocao()
                    if emocao == 'cancelar':
                        print(f"  ↩️ Exemplo cancelado.\n")
                        continue
                    
                    exemplos.append((texto, emocao))
                    print(f"  ✅ Exemplo {num} adicionado! ({len(exemplos)} na fila)\n")
                    num += 1
                
                if exemplos:
                    print(f"\n  🧠 TREINANDO REDE NEURAL...")
                    print(f"  Total de exemplos: {len(exemplos)}")
                    
                    print(f"\n  📐 Modo: 1-Normal 2-Intensivo(3x)")
                    modo = input("  Escolha (Enter=1): ").strip()
                    
                    if modo == '2':
                        print(f"\n  🔥 MODO INTENSIVO...")
                        neuro.taxa_aprendizado = 0.03
                        resultados = neuro.treinar_lote_adaptativo(exemplos, repeticoes=3)
                        neuro.taxa_aprendizado = 0.01
                    else:
                        resultados = neuro.treinar_lote(exemplos)
                    
                    print(f"\n  📊 RESULTADOS:")
                    melhora_total = 0
                    for i, (r, (texto, emocao)) in enumerate(zip(resultados, exemplos), 1):
                        melhora_total += r['melhora']
                        print(f"  [{i}] '{texto[:45]}...' → {emocao}")
                        print(f"      Loss: {r['loss_antes']:.4f} → {r['loss_depois']:.4f} (Δ{r['melhora']:.4f})")
                    
                    print(f"  ✅ TREINAMENTO CONCLUÍDO! Melhora total: {melhora_total:.6f}")
                else:
                    print("  ⚠️ Nenhum exemplo para treinar.")
            
            elif opcao == '3':
                print("\n  🔮 TESTAR PREDIÇÃO")
                texto = input("  Digite o texto: ")
                if not texto:
                    print("  ⚠️ Texto vazio.")
                    continue
                
                r = neuro.prever(texto)
                emojis = {'alegria': '😊', 'tristeza': '😢', 'raiva': '😡', 
                          'medo': '😨', 'surpresa': '😲', 'nojo': '🤢'}
                
                print(f"\n  📊 Sentimento: {r['sentimento'].upper()} {emojis.get(r['sentimento'], '')}")
                print(f"  Confiança: {r['confianca']:.4f} | Certeza: {r['certeza']:.4f}")
                print(f"  Tempo: {r['tempo_us']:.2f} μs")
                
                print("\n  📈 PROBABILIDADES:")
                for emocao, prob in sorted(r['probabilidades'].items(), key=lambda x: x[1], reverse=True):
                    barra = "█" * int(prob * 40)
                    print(f"    {emocao:12} | {prob:.4f} | {barra}")
            
            elif opcao == '4':
                neuro.mostrar_estado()
            
            elif opcao == '5':
                neuro.mostrar_metricas()
            
            elif opcao == '6':
                print("\n  ⚡ BENCHMARK DE VELOCIDADE")
                textos_bench = [b'teste feliz ' + str(i).encode() for i in range(1000)]
                t0 = time.perf_counter_ns()
                for texto in textos_bench:
                    neuro.analisar_us(texto)
                dt = (time.perf_counter_ns() - t0) / 1000.0
                print(f"  Tempo total: {dt:.1f} μs")
                print(f"  Média por texto: {dt/1000:.2f} μs")
                print(f"  Throughput: {1_000_000/(dt/1000):.0f} textos/s")
            
            elif opcao == '7':
                print("\n  ⚠️ RESETAR REDE")
                confirma = input("  Digite 'SIM' para confirmar: ")
                if confirma == 'SIM':
                    if os.path.exists(neuro.arquivo_rede):
                        os.remove(neuro.arquivo_rede)
                    neuro = NeuroMicro("emo.rn")
                    print("  ✅ Rede resetada com sucesso!")
                else:
                    print("  Cancelado.")
            
            elif opcao == '8':
                print(f"\n  💾 Rede salva em '{neuro.arquivo_rede}'")
                print(f"  📚 Total de treinos: {neuro.total_treinos}")
                print("  👋 Até mais!\n")
                break
            
            else:
                print("  ⚠️ Opção inválida! Escolha 1-8.")

# ============================================
# MAIN - PONTO DE ENTRADA
# ============================================

if __name__ == "__main__":
    #start()
   model = QuintikusFeeling("terminal") 
   model.predict("emo.rn", "estou feliz hoje")
