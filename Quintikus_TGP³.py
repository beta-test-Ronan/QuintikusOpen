#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TGP3 – Agente Cognitivo Híbrido com Rede Triangular
Versão unificada:
- Tokenização TGP2.5 (preserva acentos)
- Geometria hiperbólica (Poincaré) com Vetor(array)
- Rede triangular treinada com exemplos positivos/negativos
- Combinação de bigramas/trigramas (linear) + rede (não‑linear)
- Persistência por camadas (salvamento incremental)
- Sem dependências pesadas (numpy não é necessário)
- Otimizado para velocidade e coerência
"""

import math
import re
import pickle
import os
import random
from array import array
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# =============================================================================
# 1. VETOR (array puro)
# =============================================================================
class Vetor:
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
        v = cls(dim)
        for i in range(dim):
            v.dados[i] = random.uniform(-1, 1)
        n = v.norma()
        if n > 0:
            v = v * (raio / n)
        return v

    def __add__(self, o):
        r = Vetor(self.dim)
        for i in range(self.dim):
            r.dados[i] = self.dados[i] + o.dados[i]
        return r

    def __sub__(self, o):
        r = Vetor(self.dim)
        for i in range(self.dim):
            r.dados[i] = self.dados[i] - o.dados[i]
        return r

    def __mul__(self, e):
        r = Vetor(self.dim)
        for i in range(self.dim):
            r.dados[i] = self.dados[i] * e
        return r

    def __neg__(self):
        return self * (-1.0)

    def dot(self, o):
        s = 0.0
        for i in range(self.dim):
            s += self.dados[i] * o.dados[i]
        return s

    def norma(self):
        return math.sqrt(self.dot(self))

    def projetar_no_disco(self, max_norma=0.999):
        n = self.norma()
        if n > max_norma:
            return self * (max_norma / n)
        return self

    def copiar(self):
        return Vetor(self.dim, list(self.dados))

    @staticmethod
    def media(lista):
        if not lista:
            return Vetor(len(lista[0].dados) if lista else 64)
        dim = lista[0].dim
        res = Vetor(dim)
        for v in lista:
            for i in range(dim):
                res.dados[i] += v.dados[i]
        for i in range(dim):
            res.dados[i] /= len(lista)
        return res

# =============================================================================
# 2. DISCO DE POINCARÉ
# =============================================================================
class DiscoPoincare:
    def __init__(self, dim=64):
        self.dim = dim

    def adicao_mobius(self, u, v):
        uu = u.dot(u); vv = v.dot(v); uv = u.dot(v)
        denom = 1 + 2*uv + uu*vv
        if abs(denom) < 1e-12:
            denom = 1e-12
        cu = (1 + 2*uv + vv) / denom
        cv = (1 - uu) / denom
        return u * cu + v * cv

    def distancia(self, u, v):
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

# =============================================================================
# 3. REDE TRIANGULAR
# =============================================================================
class RedeTriangular:
    def __init__(self):
        self.pesos = [random.uniform(-0.01, 0.01) for _ in range(4)]
        self.bias = 0.0

    def features(self, A, B, C, disco):
        dAB = disco.distancia(A, B)
        dBC = disco.distancia(B, C)
        dAC = disco.distancia(A, C)
        try:
            cos_angle = (math.cosh(dAB)*math.cosh(dBC) - math.cosh(dAC)) / \
                        (math.sinh(dAB)*math.sinh(dBC) + 1e-12)
            cos_angle = max(-1.0, min(1.0, cos_angle))
        except OverflowError:
            cos_angle = 0.0
        return [dAB, dBC, dAC, cos_angle]

    def pontuar(self, features):
        s = self.bias
        for p, f in zip(self.pesos, features):
            s += p * f
        return s

    def treinar_batch(self, batch_pos, batch_neg, lr=0.01):
        """Treina com um batch (listas de features)."""
        for feat in batch_pos:
            score = self.pontuar(feat)
            pred = 1.0 / (1.0 + math.exp(-score))
            erro = 1.0 - pred
            for i in range(4):
                self.pesos[i] += lr * erro * feat[i]
            self.bias += lr * erro
        for feat in batch_neg:
            score = self.pontuar(feat)
            pred = 1.0 / (1.0 + math.exp(-score))
            erro = 0.0 - pred
            for i in range(4):
                self.pesos[i] += lr * erro * feat[i]
            self.bias += lr * erro

# =============================================================================
# 4. AGENTE PRINCIPAL (UNIFICADO)
# =============================================================================
class AgenteTGP3:
    def __init__(self, arquivo_base='agente_camada', dim=64):
        self.dim = dim
        self.disco = DiscoPoincare(dim)
        self.rede = RedeTriangular()

        self.token_para_vetor = {}   # token -> Vetor
        self.vocab = set()
        self.bigramas = defaultdict(lambda: defaultdict(int))
        self.trigramas = defaultdict(lambda: defaultdict(int))
        self.contagem_total = defaultdict(int)

        self.arquivo_base = arquivo_base
        self.camada_atual = 0
        self._carregar_estado()

    # ---------- Persistência ----------
    def _carregar_estado(self):
        base = f"{self.arquivo_base}_{self.camada_atual}.pkl"
        if os.path.exists(base):
            with open(base, 'rb') as f:
                dados = pickle.load(f)
            self.token_para_vetor = dados.get('tv', {})
            self.vocab = set(dados.get('vocab', []))
            self.bigramas = defaultdict(lambda: defaultdict(int), dados.get('bigramas', {}))
            self.trigramas = defaultdict(lambda: defaultdict(int), dados.get('trigramas', {}))
            self.contagem_total = defaultdict(int, dados.get('contagem_total', {}))
            self.rede.pesos = dados.get('pesos_triang', self.rede.pesos)
            self.rede.bias = dados.get('bias_triang', self.rede.bias)
            print(f"🔄 Estado carregado da camada {self.camada_atual}.")
        else:
            print("🆕 Nenhum estado encontrado. Inicializando do zero.")

    def _salvar_estado(self):
        self.camada_atual += 1
        base = f"{self.arquivo_base}_{self.camada_atual}.pkl"
        dados = {
            'tv': self.token_para_vetor,
            'vocab': list(self.vocab),
            'bigramas': dict(self.bigramas),
            'trigramas': dict(self.trigramas),
            'contagem_total': dict(self.contagem_total),
            'pesos_triang': self.rede.pesos,
            'bias_triang': self.rede.bias,
        }
        with open(base, 'wb') as f:
            pickle.dump(dados, f)
        print(f"💾 Estado salvo na camada {self.camada_atual} (arquivo {base}).")

    # ---------- Tokenização ----------
    @staticmethod
    def tokenizar(texto):
        texto = texto.lower()
        return re.findall(r'[^\W\d_]+|[.,;:!?()\-"]', texto)

    def _inicializar_vetor(self, token):
        if token not in self.token_para_vetor:
            self.token_para_vetor[token] = Vetor.aleatorio(self.dim, raio=0.1)
            self.vocab.add(token)

    # ---------- Treinamento ----------
    def treinar(self, texto, epocas_espacial=3, epocas_rede=10,
                lr_espacial=0.02, lr_rede=0.01, batch_size=32, salvar=True):
        print("🧠 Iniciando treinamento...")
        tokens = self.tokenizar(texto)
        if not tokens:
            return
        print(f"   {len(tokens)} tokens extraídos.")

        # Registrar tokens e vetores
        for t in set(tokens):
            self._inicializar_vetor(t)

        # Registra bigramas/trigramas
        self._registrar_fluxo(tokens)

        vocab_list = list(self.vocab)
        # Treino espacial (Poincaré)
        for epoca in range(epocas_espacial):
            print(f"   Época espacial {epoca+1}/{epocas_espacial}...")
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
                    # Repulsão (amostragem uniforme)
                    for _ in range(2):
                        neg = random.choice(vocab_list)
                        if neg == token or neg in contexto:
                            continue
                        w = self.token_para_vetor[neg]
                        self.token_para_vetor[token] = self.disco.projetar(u - (w - u) * (lr_espacial * 0.5))

        # Construir dataset para rede triangular (amostragem reduzida)
        print("   Construindo dataset para rede triangular...")
        exemplos_pos = []
        exemplos_neg = []
        max_exemplos = min(3000, len(tokens) - 2)  # limite para velocidade
        indices = random.sample(range(len(tokens)-2), max_exemplos) if len(tokens)-2 > max_exemplos else range(len(tokens)-2)
        for i in indices:
            A, B, C = tokens[i], tokens[i+1], tokens[i+2]
            va = self.token_para_vetor[A]
            vb = self.token_para_vetor[B]
            vc = self.token_para_vetor[C]
            feat_pos = self.rede.features(va, vb, vc, self.disco)
            exemplos_pos.append(feat_pos)
            # Negativo: escolhe outro token aleatório
            C_neg = random.choice(vocab_list)
            while C_neg == C:
                C_neg = random.choice(vocab_list)
            vc_neg = self.token_para_vetor[C_neg]
            feat_neg = self.rede.features(va, vb, vc_neg, self.disco)
            exemplos_neg.append(feat_neg)

        print(f"   Dataset: {len(exemplos_pos)} exemplos positivos/negativos.")
        print(f"   Treinando rede triangular ({epocas_rede} épocas, batch={batch_size})...")

        # Treino da rede com batches
        for epoca in range(epocas_rede):
            print(f"      Época rede {epoca+1}/{epocas_rede}")
            # Embaralha índices
            idx = list(range(len(exemplos_pos)))
            random.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start:start+batch_size]
                batch_pos = [exemplos_pos[i] for i in batch_idx]
                batch_neg = [exemplos_neg[i] for i in batch_idx]
                self.rede.treinar_batch(batch_pos, batch_neg, lr=lr_rede)

        if salvar:
            self._salvar_estado()
        print(f"✅ Treino concluído. {len(self.vocab)} tokens.")

    def _registrar_fluxo(self, tokens):
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i+1]
            self.bigramas[a][b] += 1
            self.contagem_total[a] += 1
            if i < len(tokens) - 2:
                c = tokens[i+2]
                self.trigramas[(a, b)][c] += 1

    # ---------- Inferência ----------
    def vetor_intencao(self, contexto_tokens):
        if not contexto_tokens:
            return Vetor(self.dim)
        vetores = [self.token_para_vetor[t] for t in contexto_tokens if t in self.token_para_vetor]
        if not vetores:
            return Vetor(self.dim)
        return Vetor.media(vetores).projetar_no_disco()

    def prob_linear(self, contexto_tokens):
        vocab_list = list(self.vocab)
        if not contexto_tokens:
            return {t: 1.0/len(vocab_list) for t in vocab_list}
        if len(contexto_tokens) >= 2:
            chave = (contexto_tokens[-2], contexto_tokens[-1])
            if chave in self.trigramas:
                cont = self.trigramas[chave]
                total = sum(cont.values())
                return {t: (cont.get(t, 0) + 0.01) / (total + 0.01*len(vocab_list))
                        for t in vocab_list}
        ultimo = contexto_tokens[-1]
        if ultimo in self.bigramas:
            cont = self.bigramas[ultimo]
            total = sum(cont.values())
            return {t: (cont.get(t, 0) + 0.01) / (total + 0.01*len(vocab_list))
                    for t in vocab_list}
        return {t: 1.0/len(vocab_list) for t in vocab_list}

    def prob_triangular(self, contexto_tokens):
        vocab_list = list(self.vocab)
        if len(contexto_tokens) < 2:
            ref = self.vetor_intencao(contexto_tokens)
            return {t: -self.disco.distancia(ref, self.token_para_vetor[t]) for t in vocab_list}
        A, B = contexto_tokens[-2], contexto_tokens[-1]
        va = self.token_para_vetor.get(A)
        vb = self.token_para_vetor.get(B)
        if va is None or vb is None:
            return {t: 0.0 for t in vocab_list}
        scores = {}
        for t in vocab_list:
            vc = self.token_para_vetor.get(t)
            if vc is None:
                continue
            feat = self.rede.features(va, vb, vc, self.disco)
            scores[t] = self.rede.pontuar(feat)
        return scores

    def prever_proximo_token(self, contexto_tokens, temperatura=0.8,
                             penalidade_repeticao=2.5, historico_recente=None):
        if historico_recente is None:
            historico_recente = []
        vocab_list = list(self.vocab)
        if not contexto_tokens or not vocab_list:
            return random.choice(vocab_list) if vocab_list else None

        p_lin = self.prob_linear(contexto_tokens)
        p_tri = self.prob_triangular(contexto_tokens)

        # Normaliza p_tri com softmax
        max_score = max(p_tri.values()) if p_tri else 0.0
        exps = {t: math.exp(score - max_score) for t, score in p_tri.items()}
        soma_exp = sum(exps.values())
        p_tri_norm = {t: e / soma_exp for t, e in exps.items()}

        logits = []
        for t in vocab_list:
            lp = math.log(p_lin.get(t, 1e-12))
            ltp = math.log(p_tri_norm.get(t, 1e-12))
            logit = 1.4 * lp + 0.6 * ltp   # pesos ajustados para boa coerência
            logits.append(logit)

        # Penalidade de repetição
        for i, t in enumerate(vocab_list):
            if t in historico_recente:
                logits[i] -= penalidade_repeticao * historico_recente.count(t)

        # Softmax com temperatura
        logits = [l / max(temperatura, 1e-6) for l in logits]
        max_l = max(logits)
        probs = [math.exp(l - max_l) for l in logits]
        soma = sum(probs)
        probs = [p / soma for p in probs]

        idx = random.choices(range(len(vocab_list)), weights=probs)[0]
        return vocab_list[idx]

    # ---------- Geração ----------
    def gerar(self, estimulo, max_tokens=50, temperatura=0.8):
        contexto = self.tokenizar(estimulo)
        for t in contexto:
            self._inicializar_vetor(t)

        gerados = list(contexto)
        historico = []

        for _ in range(max_tokens):
            if len(gerados) > len(contexto) and gerados[-1] in '.!?':
                break
            contexto_atual = gerados[-5:]
            prox = self.prever_proximo_token(
                contexto_atual,
                temperatura=temperatura,
                historico_recente=historico
            )
            if prox is None:
                break
            gerados.append(prox)
            historico.append(prox)
            if len(historico) > 10:
                historico.pop(0)

        # Formata saída
        resultado = ''
        for tok in gerados:
            if tok in '.,;:!?()-"':
                resultado = resultado.rstrip() + tok + ' '
            else:
                resultado += tok + ' '
        return resultado.strip()

    # ---------- Interface CLI ----------
    def loop_cli(self):
        print("\n🧠 TGP3 – Agente Cognitivo Híbrido (Rede Triangular + Poincaré)")
        print("Comandos:")
        print("  train:<arquivo>  -> treina com o arquivo de texto")
        print("  <texto>          -> gera uma continuação")
        print("  salvar           -> salva o estado atual (nova camada)")
        print("  sair             -> encerra")
        while True:
            entrada = input("\n>>> ").strip()
            if entrada == 'sair':
                self._salvar_estado()
                print("Encerrado.")
                break
            elif entrada.startswith('train:'):
                arquivo = entrada.split(':', 1)[1]
                if os.path.exists(arquivo):
                    with open(arquivo, 'r', encoding='utf-8') as f:
                        texto = f.read()
                    self.treinar(texto, epocas_espacial=3, epocas_rede=10,
                                 batch_size=32, salvar=True)
                else:
                    print(f"Arquivo não encontrado: {arquivo}")
            elif entrada == 'salvar':
                self._salvar_estado()
            elif entrada:
                resposta = self.gerar(entrada, max_tokens=50, temperatura=0.8)
                print(f"\n{resposta}")

# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == '__main__':
    agente = AgenteTGP3()
    agente.loop_cli()
