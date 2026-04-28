#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import time
import re
import shutil
import string
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple

# =============================================================================
#  MINI REDE NEURAL
# =============================================================================
class CamadaLinear:
    def __init__(self, n_entrada, n_saida):
        limite = np.sqrt(6 / (n_entrada + n_saida))
        self.pesos = np.random.uniform(-limite, limite, (n_entrada, n_saida))
        self.vies = np.zeros((1, n_saida))
        self.grad_pesos = None
        self.grad_vies = None
        self.entrada = None

    def forward(self, x):
        self.entrada = x
        return np.dot(x, self.pesos) + self.vies

    def backward(self, grad_saida, taxa_aprendizado=0.01):
        self.grad_pesos = np.dot(self.entrada.T, grad_saida)
        self.grad_vies = np.sum(grad_saida, axis=0, keepdims=True)
        grad_entrada = np.dot(grad_saida, self.pesos.T)
        self.pesos -= taxa_aprendizado * self.grad_pesos
        self.vies  -= taxa_aprendizado * self.grad_vies
        return grad_entrada


class RedeNeural:
    def __init__(self, n_entrada, n_saida):
        self.camadas = [CamadaLinear(n_entrada, n_saida)]

    def forward(self, x):
        ativacao = x
        for camada in self.camadas:
            ativacao = camada.forward(ativacao)
        return ativacao

    def treinar_passo(self, x, y, taxa_aprendizado=0.01):
        saida = self.forward(x)
        perda = np.mean((saida - y) ** 2)
        grad_saida = 2 * (saida - y) / y.shape[0]
        for camada in reversed(self.camadas):
            grad_saida = camada.backward(grad_saida, taxa_aprendizado)
        return perda

    def prever(self, x):
        return self.forward(x)

    def carregar_pesos(self, estado):
        for c, e in zip(self.camadas, estado):
            c.pesos = np.array(e['pesos'])
            c.vies = np.array(e['vies'])

    def obter_pesos(self):
        return [{'pesos': c.pesos.tolist(), 'vies': c.vies.tolist()} for c in self.camadas]


# =============================================================================
#  PARSE DO NOME DA TAREFA
# =============================================================================
def parse_task_name(raw_name: str) -> Tuple[str, bool, Optional[int], Optional[int]]:
    clean = raw_name.strip()
    priority = False
    group = None
    order = None

    if '-P' in clean:
        priority = True
        clean = clean.replace('-P', '')

    match = re.search(r'id\s*=\s*(\d+)\.(\d+)', clean)
    if match:
        group = int(match.group(1))
        order = int(match.group(2))
        clean = re.sub(r'id\s*=\s*\d+\.\d+', '', clean)

    clean = ' '.join(clean.split())
    return clean, priority, group, order


# =============================================================================
#  AGENTE CASSINUS (MELHOR DE 3 + EXPLORAÇÃO REFORÇADA + LOG)
# =============================================================================
class Cassinus:
    def __init__(self,
                 tarefas: List[Dict[str, object]],
                 taxa_exploracao: float = 0.25,
                 repeticoes: int = 3,
                 max_consecutivas: int = 2):
        self.tarefas = []
        for t in tarefas:
            func = t['func']
            raw_name = t['name']
            clean_name, priority, group, order = parse_task_name(raw_name)
            self.tarefas.append({
                'func': func,
                'nome_limpo': clean_name,
                'prioritaria': priority,
                'grupo': group,
                'ordem': order,
                'executada': False,
            })

        self.num_tarefas = len(self.tarefas)
        self.taxa_exploracao = taxa_exploracao
        self.repeticoes = repeticoes
        self.max_consecutivas = max_consecutivas
        self.ultimo_indice = None
        self.consecutivas = 0
        self.confianca = 0.5
        self.ciclo = 0  
        self.rede = RedeNeural(self.num_tarefas, 1)
        self.primeira_vez = [True] * self.num_tarefas
        self.ultimos_tempos = [0.0] * self.num_tarefas
        self.modelo_arquivo = "rotas.model.json"
        self.log_arquivo = "cassinus_log.json"
        self.log_txt_arquivo = "cassinus_log.txt"
        self._carregar_modelo()

    def _carregar_modelo(self):
        if os.path.exists(self.modelo_arquivo):
            try:
                with open(self.modelo_arquivo, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                if dados.get('num_rotas') == self.num_tarefas:
                    self.rede.carregar_pesos(dados['modelo'])
                    self.primeira_vez = dados.get('primeira_vez', [True]*self.num_tarefas)
                    self.confianca = dados.get('confianca', 0.5)
                    print("🎩 Cassinus: Carreguei meu cérebro e memórias. Pronto para brilhar!\n")
                else:
                    print("🎩 Cassinus: O número de tarefas mudou, começando do zero.\n")
            except (json.JSONDecodeError, KeyError, TypeError):
                print("🎩 Cassinus: Arquivo de modelo corrompido/incompatível. Recomeçando.\n")
        else:
            print("🎩 Cassinus: Nenhum modelo encontrado. Iniciarei minha jornada.\n")

    def _salvar_modelo(self):
        dados = {
            'num_rotas': self.num_tarefas,
            'modelo': self.rede.obter_pesos(),
            'primeira_vez': self.primeira_vez,
            'confianca': self.confianca
        }
        with open(self.modelo_arquivo, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2)

    def _one_hot(self, indice):
        v = np.zeros((1, self.num_tarefas))
        v[0, indice] = 1.0
        return v

    def _prever_tempo(self, indice):
        x = self._one_hot(indice)
        pred = self.rede.prever(x)[0, 0]
        return max(0.0001, pred)

    def _tarefas_prontas_prioritarias(self):
        prontas = []
        for i, t in enumerate(self.tarefas):
            if not t['prioritaria'] or self.tarefas[i]['executada']:
                continue
            if t['grupo'] is not None:
                deps = [j for j, t2 in enumerate(self.tarefas)
                        if t2['prioritaria'] and t2['grupo'] == t['grupo'] and t2['ordem'] < t['ordem']]
                if not all(self.tarefas[j]['executada'] for j in deps):
                    continue
            prontas.append(i)
        return prontas

    def _tarefas_nao_prioritarias(self):
        return [i for i, t in enumerate(self.tarefas) if not t['prioritaria']]

    def _forcar_exploracao(self, opcoes):
        """Escolhe aleatoriamente, excluindo a última tarefa se possível."""
        if len(opcoes) == 1:
            return opcoes[0]
        if self.ultimo_indice is not None and self.ultimo_indice in opcoes and len(opcoes) > 1:
            outras = [i for i in opcoes if i != self.ultimo_indice]
            return random.choice(outras)
        return random.choice(opcoes)

    def _salvar_log_txt(self, indice, previsto, real, tempos, perda):
        """Escreve uma linha no log TXT com os dados da execução."""
        tarefa = self.tarefas[indice]
        linha = (
            f"Ciclo {self.ciclo:3d} | "
            f"{'PRI' if tarefa['prioritaria'] else 'CAS'} | "
            f"{tarefa['nome_limpo']:<25s} | "
            f"Melhor: {real:.4f}s | "
            f"Previsto: {previsto:.4f}s | "
            f"Confiança: {self.confianca:.0%} | "
            f"{'ACERTOU 💪' if real < previsto else 'ERROU 😢'}\n"
        )
        with open(self.log_txt_arquivo, 'a', encoding='utf-8') as f:
            f.write(linha)

    def gerar_resumo_txt(self):
        """Adiciona um resumo final ao log TXT."""
        with open(self.log_txt_arquivo, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*70 + "\n")
            f.write("🏆 MELHORES TEMPOS ABSOLUTOS POR TAREFA 🏆\n")
            f.write("="*70 + "\n")
            for i, t in enumerate(self.tarefas):
                f.write(f"{t['nome_limpo']:<30s} → {self.ultimos_tempos[i]:.4f}s\n")
            f.write(f"\n✨ Confiança final: {self.confianca:.0%}\n")

    def escolher_proxima_tarefa(self):
        prontas_prio = self._tarefas_prontas_prioritarias()
        if prontas_prio:
            if random.random() < self.taxa_exploracao:
                escolhido = random.choice(prontas_prio)
                print(f"🎲 Cassinus (prioridade): Explorando! Escolhi '{self.tarefas[escolhido]['nome_limpo']}'.")
            else:
                previsoes = {i: self._prever_tempo(i) for i in prontas_prio}
                escolhido = min(previsoes, key=previsoes.get)
                print(f"🎯 Cassinus (prioridade): Apostei em '{self.tarefas[escolhido]['nome_limpo']}' "
                      f"(previsto {previsoes[escolhido]:.3f}s).")
            return escolhido

        cassino = self._tarefas_nao_prioritarias()
        if not cassino:
            return None

        if self.consecutivas >= self.max_consecutivas:
            print(f"🚫 Cassinus: Já executei '{self.tarefas[self.ultimo_indice]['nome_limpo']}' "
                  f"{self.consecutivas}x seguidas. Hora de variar!")
            escolhido = self._forcar_exploracao(cassino)
            print(f"🎲 Cassinus (cassino): Forçando variedade. Escolhi '{self.tarefas[escolhido]['nome_limpo']}'.")
            return escolhido

        if random.random() < self.taxa_exploracao:
            escolhido = random.choice(cassino)
            print(f"🎲 Cassinus (cassino): Explorando! Escolhi '{self.tarefas[escolhido]['nome_limpo']}'.")
        else:
            previsoes = {i: self._prever_tempo(i) for i in cassino}
            escolhido = min(previsoes, key=previsoes.get)
            print(f"🎯 Cassinus (cassino): Minha aposta é '{self.tarefas[escolhido]['nome_limpo']}' "
                  f"(previsto {previsoes[escolhido]:.3f}s).")
        return escolhido

    def executar_por_nome(self, nome_tarefa):
        """Executa uma tarefa pelo nome (se existir) e atualiza o modelo."""
        for i, t in enumerate(self.tarefas):
            if t['nome_limpo'] == nome_tarefa:
                print(f"🔍 Cassinus: Tarefa '{nome_tarefa}' encontrada. Executando...")
                self.executar_tarefa(i)
                return
        print(f"❌ Cassinus: Não conheço nenhuma tarefa chamada '{nome_tarefa}'.")
    def executar_tarefa(self, indice):
        tarefa = self.tarefas[indice]
        if self.primeira_vez[indice]:
            print(f"📝 Cassinus: Primeira vez com '{tarefa['nome_limpo']}'. Vamos conhecer essa rota.")
            self.primeira_vez[indice] = False
        else:
            print(f"📋 Cassinus: Executando '{tarefa['nome_limpo']}' "
                  f"({'prioritária' if tarefa['prioritaria'] else 'cassino'}).")

        tempos_rodada = []
        for tentativa in range(1, self.repeticoes + 1):
            inicio = time.perf_counter()
            tarefa['func']()
            duracao = time.perf_counter() - inicio
            tempos_rodada.append(duracao)
            print(f"   🏁 Tentativa {tentativa}/{self.repeticoes}: {duracao:.4f}s")

        tempo_real = min(tempos_rodada)
        self.ultimos_tempos[indice] = tempo_real
        print(f"⏱️  Melhor tempo (de {self.repeticoes}): {tempo_real:.4f}s")

        tempo_previsto = self._prever_tempo(indice)

        if tempo_real < tempo_previsto:
            print(f"😎 Cassinus: '{tarefa['nome_limpo']}' foi mais rápida que o previsto "
                  f"({tempo_real:.4f}s < {tempo_previsto:.4f}s). Sou um super agente! 💪")
            self.confianca = min(1.0, self.confianca + 0.12)
        else:
            print(f"😢 Cassinus: '{tarefa['nome_limpo']}' demorou mais que o esperado "
                  f"({tempo_real:.4f}s >= {tempo_previsto:.4f}s). Que tristeza...")
            self.confianca = max(0.0, self.confianca - 0.05)

        x = self._one_hot(indice)
        perda = self.rede.treinar_passo(x, np.array([[tempo_real]]), taxa_aprendizado=0.08)
        print(f"📉 Perda MSE: {perda:.6f}")

        if tarefa['prioritaria']:
            self.tarefas[indice]['executada'] = True

        if self.ultimo_indice == indice:
            self.consecutivas += 1
        else:
            self.consecutivas = 1
        self.ultimo_indice = indice

        # Salvar logs (JSON e TXT)
        self._salvar_log(indice, tempo_previsto, tempo_real, tempos_rodada, perda)
        self._salvar_log_txt(indice, tempo_previsto, tempo_real, tempos_rodada, perda)

        self._salvar_modelo()
        print(f"💾 Modelo salvo em '{self.modelo_arquivo}'.\n")

    def _salvar_log(self, indice, previsto, real, tempos, perda):
        """Log JSON (já existente)."""
        entrada = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ciclo": self.ciclo if hasattr(self, 'ciclo') else 0,
            "tarefa": self.tarefas[indice]['nome_limpo'],
            "tipo": "prioritaria" if self.tarefas[indice]['prioritaria'] else "cassino",
            "previsao": round(previsto, 6),
            "melhor_tempo": round(real, 6),
            "tentativas": [round(t, 6) for t in tempos],
            "perda_mse": round(float(perda), 6),
            "confianca": round(self.confianca, 4)
        }
        log = []
        if os.path.exists(self.log_arquivo):
            try:
                with open(self.log_arquivo, 'r', encoding='utf-8') as f:
                    log = json.load(f)
            except:
                pass
        log.append(entrada)
        with open(self.log_arquivo, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    def exibir_estado_interno(self):
        print("🧠 Cassinus: Minhas previsões e status:")
        for i, t in enumerate(self.tarefas):
            pred = self._prever_tempo(i)
            if t['prioritaria']:
                estado = "executada" if t['executada'] else ("pronta" if i in self._tarefas_prontas_prioritarias() else "pendente")
            else:
                estado = "sempre disponível"
            dep = f" (id={t['grupo']}.{t['ordem']})" if t['grupo'] is not None else ""
            tipo = "PRI" if t['prioritaria'] else "CAS"
            print(f"  [{tipo}] {t['nome_limpo']}{dep}: prev {pred:.4f}s | último {self.ultimos_tempos[i]:.4f}s | {estado}")
        print(f"   Confiança atual: {self.confianca:.0%}\n")


# =============================================================================
#  EXECUÇÃO DO CICLO COMPLETO (agora com aquecimento)
# =============================================================================
def executar_cassino_prioritario(tarefas: List[Dict],
                                 num_extra: int = 20,
                                 taxa_exploracao: float = 0.25,
                                 repeticoes: int = 3,
                                 max_consecutivas: int = 2,
                                 aquecer: bool = True):
    print("\n" + "=" * 60)
    print("🎰 CASSINO INTELIGENTE (Melhor de 3 + Exploração Reforçada) 🎰".center(60))
    print("=" * 60 + "\n")

    agente = Cassinus(tarefas,
                      taxa_exploracao=taxa_exploracao,
                      repeticoes=repeticoes,
                      max_consecutivas=max_consecutivas)

    # ----- AQUECIMENTO: roda cada tarefa uma vez para ter dados iniciais -----
    if aquecer:
        print("🔥 AQUECENDO O CASSINUS: executando todas as tarefas pela primeira vez...\n")
        for i in range(agente.num_tarefas):
            agente.ciclo = 0  # ciclo 0 para não atrapalhar a contagem
            print(f"--- Aquecimento: conhecendo '{agente.tarefas[i]['nome_limpo']}' ---")
            agente.executar_tarefa(i)
        print("✅ Aquecimento concluído!\n")

    total_prioritarias = sum(1 for t in tarefas if parse_task_name(t['name'])[1])
    total_ciclos = total_prioritarias + num_extra

    for ciclo in range(1, total_ciclos + 1):
        agente.ciclo = ciclo
        print(f"\n--- Ciclo {ciclo}/{total_ciclos} ---")
        indice = agente.escolher_proxima_tarefa()
        if indice is None:
            print("🎉 Nenhuma tarefa disponível. Encerrando.\n")
            break
        agente.executar_tarefa(indice)
        if ciclo % 4 == 0:
            agente.exibir_estado_interno()

    agente.gerar_resumo_txt()

    print("\n" + "=" * 60)
    print("🎉 Cassinus encerrou o turno com maestria. 🎉".center(60))
    print("=" * 60)
    print(f"\n📁 Log detalhado salvo em '{agente.log_arquivo}' e '{agente.log_txt_arquivo}'.")


# =============================================================================
#  EXEMPLO DE USO – FUNÇÕES QUE VOCÊ VAI VER ACONTECER DE VERDADE
# =============================================================================

PASTA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teste_cassinus")

def criar_txt_numerados():
    """Cria 10 arquivos (1.txt .. 10.txt) dentro de teste_cassinus/txt_numerados/."""
    destino = os.path.join(PASTA_BASE, "txt_numerados")
    os.makedirs(destino, exist_ok=True)
    for i in range(1, 11):
        with open(os.path.join(destino, f"{i}.txt"), 'w') as f:
            f.write("conteudo")

def criar_txt_letras():
    """Cria 10 arquivos (a.txt .. j.txt) dentro de teste_cassinus/txt_letras/."""
    destino = os.path.join(PASTA_BASE, "txt_letras")
    os.makedirs(destino, exist_ok=True)
    for letra in string.ascii_lowercase[:10]:
        with open(os.path.join(destino, f"{letra}.txt"), 'w') as f:
            f.write("conteudo")

def criar_subpasta_e_mover():
    """Dentro de teste_cassinus, cria 'test/test', gera um arquivo e move."""
    raiz = os.path.join(PASTA_BASE, "mover")
    os.makedirs(os.path.join(raiz, "test", "test"), exist_ok=True)
    origem = os.path.join(raiz, "arquivo_original.txt")
    with open(origem, 'w') as f:
        f.write("dados")
    destino = os.path.join(raiz, "test", "test", "arquivo_renomeado.txt")
    shutil.move(origem, destino)


if __name__ == "__main__":
    # Cria a pasta base (ou limpa se já existir)
    if os.path.exists(PASTA_BASE):
        shutil.rmtree(PASTA_BASE)
    os.makedirs(PASTA_BASE, exist_ok=True)

    lista_tarefas = [
        {'func': criar_txt_numerados,      'name': 'Criar TXT numerados'},
        {'func': criar_txt_letras,         'name': 'Criar TXT letras'},
        {'func': criar_subpasta_e_mover,   'name': 'Mover e Renomear'},
    ]

    # Executa com 20 ciclos extras, sem repetição interna para rodar rápido
   # executar_cassino_prioritario(lista_tarefas,  num_extra=5,   repeticoes=1,  aquecer=True)        # roda todas primeiro

    print(f"\n🔍 Verifique a pasta '{PASTA_BASE}' para ver os arquivos criados!\n")
    agente = Cassinus(lista_tarefas)
    agente.executar_por_nome("Criar TXT letras")
