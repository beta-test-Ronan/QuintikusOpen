#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import math
import random
import time
import pickle
import os
from collections import defaultdict, deque

# ============================================
# UTILITÁRIOS
# ============================================
def sha256(msg: str) -> str:
    return hashlib.sha256(msg.encode()).hexdigest()

def cosine_sim(v1, v2):
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    return dot / (n1 * n2)

def sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def softplus(x: float) -> float:
    x = min(x, 20.0)
    return math.log1p(math.exp(x))

# ============================================
# REDE NEURAL COM ADAM
# ============================================
class MiniRedeAdam:
    def __init__(self, input_dim=10, hidden_dim=16, output_dim=8):
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)
        
        self.W1 = [[(random.random() * 2 - 1) * scale1 for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[(random.random() * 2 - 1) * scale2 for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        self.lr = 0.005
        
        self.mW1 = [[0.0] * hidden_dim for _ in range(input_dim)]
        self.vW1 = [[0.0] * hidden_dim for _ in range(input_dim)]
        self.mb1 = [0.0] * hidden_dim
        self.vb1 = [0.0] * hidden_dim
        self.mW2 = [[0.0] * output_dim for _ in range(hidden_dim)]
        self.vW2 = [[0.0] * output_dim for _ in range(hidden_dim)]
        self.mb2 = [0.0] * output_dim
        self.vb2 = [0.0] * output_dim

    def forward(self, x):
        self.x = x
        self.h = [0.0] * len(self.b1)
        for i in range(len(self.b1)):
            s = self.b1[i]
            for j in range(len(x)):
                s += x[j] * self.W1[j][i]
            self.h[i] = max(0.0, s)
        
        self.out_raw = [0.0] * len(self.b2)
        for i in range(len(self.b2)):
            s = self.b2[i]
            for j in range(len(self.h)):
                s += self.h[j] * self.W2[j][i]
            self.out_raw[i] = s
        
        return [softplus(v) for v in self.out_raw]

    def backward(self, target):
        self.t += 1
        sig_out_raw = [sigmoid(v) for v in self.out_raw]
        delta2 = [(self.out_raw[i] - target[i]) * sig_out_raw[i] for i in range(len(target))]
        dW2 = [[delta2[j] * self.h[i] for j in range(len(delta2))] for i in range(len(self.h))]
        db2 = list(delta2)
        delta1 = [0.0] * len(self.h)
        for i in range(len(self.h)):
            if self.h[i] <= 0:
                continue
            s = 0.0
            for j in range(len(delta2)):
                s += delta2[j] * self.W2[i][j]
            delta1[i] = s
        dW1 = [[delta1[j] * self.x[i] for j in range(len(delta1))] for i in range(len(self.x))]
        db1 = list(delta1)
        self._adam_update(self.W1, dW1, self.mW1, self.vW1)
        self._adam_update_1d(self.b1, db1, self.mb1, self.vb1)
        self._adam_update(self.W2, dW2, self.mW2, self.vW2)
        self._adam_update_1d(self.b2, db2, self.mb2, self.vb2)

    def _adam_update(self, param, grad, m, v):
        for i in range(len(param)):
            for j in range(len(param[i])):
                m[i][j] = self.beta1 * m[i][j] + (1 - self.beta1) * grad[i][j]
                v[i][j] = self.beta2 * v[i][j] + (1 - self.beta2) * grad[i][j] * grad[i][j]
                m_hat = m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = v[i][j] / (1 - self.beta2 ** self.t)
                param[i][j] -= self.lr * m_hat / (math.sqrt(abs(v_hat)) + self.eps)

    def _adam_update_1d(self, param, grad, m, v):
        for i in range(len(param)):
            m[i] = self.beta1 * m[i] + (1 - self.beta1) * grad[i]
            v[i] = self.beta2 * v[i] + (1 - self.beta2) * grad[i] * grad[i]
            m_hat = m[i] / (1 - self.beta1 ** self.t)
            v_hat = v[i] / (1 - self.beta2 ** self.t)
            param[i] -= self.lr * m_hat / (math.sqrt(abs(v_hat)) + self.eps)


# ============================================
# MOTOR PRINCIPAL V82.1
# ============================================
class QuintikusDLMC:
    def __init__(self, texto: str = "", arquivo_bin: str = "cerebro_v82.bin"):
        self.texto = texto
        self.arquivo_bin = arquivo_bin
        self.matrix = {}
        self.blocos = []
        self.estados = [0.3, 0.7]
        self.rastro = []
        self.coords = {}
        self.inicios = []
        self.frases_originais = []
        self.memoria_curto_prazo = deque(maxlen=10)
        self.topicos = defaultdict(list)
        self.pronto = False
        self.rede = MiniRedeAdam(10, 16, 8)
        
        # Parâmetros ajustáveis
        self.max_tokens = 40
        self.temperatura = 0.7
        self.debug = False
        
        # Contador de interações para treino de consolidação
        self.interacoes = 0
        self.ultimo_treino_consolidacao = 0

    def _embed_palavra(self, token: str):
        if token not in self.matrix:
            return [0.0] * 10
        coord = self.coords.get(token, [0, 0, 0])
        freq = len(self.matrix[token].get("links", {}))
        massa = self.matrix[token].get("m", 0)
        num_nexts = len(self.matrix[token].get("nexts", []))
        return [
            coord[0] / 100, coord[1] / 100, coord[2] / 100,
            math.log1p(freq) / 10, massa * 10, num_nexts / 100,
            random.random() * 0.01,
            self.estados[0], self.estados[1],
            len(self.rastro) / 100
        ]

    def _embed_frase(self, tokens):
        if not tokens:
            return [0.0] * 10
        embeddings = [self._embed_palavra(t) for t in tokens]
        soma = [0.0] * 10
        for emb in embeddings:
            for i in range(10):
                soma[i] += emb[i]
        n = len(embeddings)
        return [v / n for v in soma]

    def _similaridade_prompt_frase(self, prompt_tokens, frase_tokens):
        if not prompt_tokens or not frase_tokens:
            return 0.0
        set_p = set(prompt_tokens)
        set_f = set(frase_tokens)
        intersecao = len(set_p & set_f)
        uniao = len(set_p | set_f)
        jaccard = intersecao / uniao if uniao > 0 else 0.0
        coord_p = self._embed_frase(prompt_tokens)
        coord_f = self._embed_frase(frase_tokens)
        cos_sim = cosine_sim(coord_p, coord_f)
        return jaccard * 0.6 + ((cos_sim + 1) / 2) * 0.4

    # ------------------------------------------------------------
    # COMPORTAMENTO EMERGENTE: decide sozinho como responder
    # ------------------------------------------------------------
    def _comportamento_emergente(self, ql):
        """O motor decide por vontade própria qual estratégia usar"""
        if not self.frases_originais:
            return "criativo", 0.0
        
        # Calcula similaridade com dataset
        sims = []
        for frase in self.frases_originais[:50]:
            sim = self._similaridade_prompt_frase(ql, frase)
            sims.append(sim)
        
        sim_max = max(sims) if sims else 0
        sim_media = sum(sims) / len(sims) if sims else 0
        
        # Fatores de decisão
        fator_dataset = sim_max  # Quão próximo está do dataset original
        fator_emocional = abs(self.estados[0] - self.estados[1])  # Instabilidade emocional
        fator_comprimento = min(1.0, len(ql) / 10)  # Prompt mais longo = mais contexto
        fator_novidade = 1.0 - sim_media  # Quão novo é o prompt
        
        # Decisão ponderada
        score_linear = fator_dataset * 0.5 + fator_comprimento * 0.3 + (1 - fator_emocional) * 0.2
        score_criativo = fator_novidade * 0.4 + fator_emocional * 0.4 + (1 - fator_comprimento) * 0.2
        
        if self.debug:
            print(f"   [DEBUG] Score Linear: {score_linear:.3f} | Score Criativo: {score_criativo:.3f}")
            print(f"   [DEBUG] Dataset: {fator_dataset:.3f} | Emoção: {fator_emocional:.3f} | Novidade: {fator_novidade:.3f}")
        
        if score_linear >= score_criativo:
            return "linear", sim_max
        return "criativo", fator_novidade

    # ------------------------------------------------------------
    # RESPOSTA LINEAR (baseada no dataset)
    # ------------------------------------------------------------
    def _responder_linear(self, ql, sim_max):
        scores = []
        for idx, frase in enumerate(self.frases_originais):
            score = self._similaridade_prompt_frase(ql, frase)
            scores.append((score, idx, frase))
        scores.sort(key=lambda x: x[0], reverse=True)
        
        top = [s for s in scores[:5] if s[0] > 0.02]
        if not top:
            return self._responder_criativo(ql, 0.5)
        
        melhor_frase = top[0][2]
        inicio = 0
        for i, token in enumerate(melhor_frase):
            if any(p in token for p in ql):
                inicio = i
                break
        
        trecho = melhor_frase[inicio:inicio + self.max_tokens]
        if len(trecho) < 3 and len(top) > 1:
            trecho = top[1][2][:self.max_tokens]
        
        resposta = " ".join(trecho)
        if resposta and resposta[-1] not in ".!?":
            resposta += "."
        return resposta[0].upper() + resposta[1:] if resposta else "..."

    # ------------------------------------------------------------
    # RESPOSTA CRIATIVA (Markov + Rede Neural)
    # ------------------------------------------------------------
    def _responder_criativo(self, ql, fator_novidade):
        if not self.matrix:
            return "Preciso de mais dados."
        
        emb_prompt = self._embed_frase(ql)
        pesos_geracao = self.rede.forward(emb_prompt)
        w_comprimento, w_palavras_longas, w_pontuacao, w_criatividade, w_estado0, w_estado1, w_coerencia, w_variabilidade = pesos_geracao
        
        # Aprende com o input do usuário (sempre!)
        target_aprendizado = [
            min(1.0, len(ql) / 20),
            sum(1 for t in ql if len(t) > 5) / max(1, len(ql)),
            sum(1 for t in ql if t == ',') / max(1, len(ql)),
            0.6,
            self.estados[0],
            self.estados[1],
            0.7,
            0.5
        ]
        self.rede.backward(target_aprendizado)
        
        # Atualiza estados emocionais
        pos = ['amo', 'amor', 'bem', 'feliz', 'bom', 'gosto', 'lindo', 'maravilhoso', 'obrigado']
        neg = ['odeio', 'triste', 'mal', 'raiva', 'feio', 'horrivel', 'chateado']
        for t in ql:
            if any(e in t for e in pos):
                self.estados[1] = min(1.0, self.estados[1] + 0.15)
            if any(e in t for e in neg):
                self.estados[0] = min(1.0, self.estados[0] + 0.15)
        self.estados = [s * 0.95 for s in self.estados]
        
        # Palavra inicial
        atual = None
        for t in ql:
            if t in self.matrix and self.matrix[t].get("nexts"):
                atual = t
                break
        if not atual and self.inicios:
            atual = random.choice(self.inicios)
        if not atual or not self.matrix.get(atual, {}).get("nexts"):
            candidatas = [k for k in self.matrix if self.matrix[k].get("nexts")]
            if not candidatas:
                return "Preciso de mais dados."
            atual = random.choice(candidatas)
        
        resultado = []
        ultimos_tokens = []
        comprimento_alvo = max(5, min(self.max_tokens, round((w_comprimento or 0.5) * 30)))
        
        for _ in range(comprimento_alvo):
            if atual not in self.matrix or not self.matrix[atual].get("nexts"):
                break
            
            if self.debug:
                print(f"   [DEBUG] Token: '{atual}'")
            
            resultado.append(atual)
            ultimos_tokens.append(atual)
            if len(ultimos_tokens) > 8:
                ultimos_tokens.pop(0)
            
            candidatos = self.matrix[atual]["nexts"]
            links = self.matrix[atual]["links"]
            pesos = []
            for prox in candidatos:
                peso = links.get(prox, 1)
                if prox in ultimos_tokens:
                    peso *= 0.001
                if not self.matrix.get(prox, {}).get("nexts"):
                    peso *= 0.1
                if prox in ql:
                    peso *= (1.5 + (w_coerencia or 0.5))
                if len(prox) > 5 and (w_palavras_longas or 0.5) > 0.5:
                    peso *= 1.5
                if prox in (',', '.') and (w_pontuacao or 0.3) > 0.5:
                    peso *= 2.0
                if self.coords.get(prox):
                    for tp in ql:
                        if self.coords.get(tp):
                            sim = (cosine_sim(self.coords[prox], self.coords[tp]) + 1) / 2
                            peso *= (1 + sim * (w_variabilidade or 0.5))
                if len(resultado) % 15 == 14 and prox == '.':
                    peso *= 5.0
                pesos.append(peso)
            
            # Aplica temperatura
            if self.temperatura != 1.0 and self.temperatura > 0:
                pesos = [p ** (1.0 / max(0.1, self.temperatura)) for p in pesos]
            
            soma = sum(pesos) + 1e-8
            probs = [p / soma for p in pesos]
            r = random.random()
            acum = 0.0
            escolhido = candidatos[0]
            for j, p in enumerate(probs):
                acum += p
                if r <= acum:
                    escolhido = candidatos[j]
                    break
            atual = escolhido
            if atual == '.' and len(resultado) >= 3:
                break
        
        if resultado and resultado[-1] == '.':
            resultado.pop()
        
        self.rastro.extend(resultado)
        if len(self.rastro) > 100:
            self.rastro = self.rastro[-100:]
        
        resposta = " ".join(resultado).replace(" ,", ",").replace(" .", ".")
        if resposta and resposta[-1] not in ".!?":
            resposta += "."
        return resposta[0].upper() + resposta[1:] if resposta else "..."

    # ------------------------------------------------------------
    # PENSAR (comportamento 100% emergente)
    # ------------------------------------------------------------
    def pensar(self, prompt: str) -> str:
        self.interacoes += 1
        ql = prompt.lower().split()
        if not ql:
            return "..."
        if not self.matrix:
            return "Preciso de mais dados. Use train:arquivo.txt"
        
        self.memoria_curto_prazo.append(("user", ql))
        
        # O motor decide sozinho como responder
        estrategia, confianca = self._comportamento_emergente(ql)
        
        if self.debug:
            print(f"   [DEBUG] Estratégia emergente: {estrategia} (confiança: {confianca:.3f})")
        
        if estrategia == "linear":
            resposta = self._responder_linear(ql, confianca)
        else:
            resposta = self._responder_criativo(ql, confianca)
        
        # Limite de tokens
        tokens = resposta.split()
        if len(tokens) > self.max_tokens + 10:
            resposta = " ".join(tokens[:self.max_tokens]) + "."
        
        return resposta

    # ------------------------------------------------------------
    # TREINO DE CONSOLIDAÇÃO (executado ao sair)
    # ------------------------------------------------------------
    def treino_consolidacao(self):
        """Reforça o aprendizado com todo o histórico da sessão"""
        if not self.frases_originais:
            return
        
        print("⚙️ Consolidando aprendizado da sessão...")
        t0 = time.time()
        
        # Treina com frases do dataset + respostas geradas
        num_treino = min(len(self.frases_originais), 200)
        for epoca in range(3):  # 3 épocas de reforço
            for i in range(num_treino):
                frase = self.frases_originais[i % len(self.frases_originais)]
                if len(frase) < 2:
                    continue
                emb = self._embed_frase(frase)
                self.rede.forward(emb)
                target = [
                    min(1.0, len(frase) / 20),
                    sum(1 for t in frase if len(t) > 5) / max(1, len(frase)),
                    sum(1 for t in frase if t in (',', '.')) / max(1, len(frase)),
                    0.5 + random.random() * 0.3,
                    self.estados[0],
                    self.estados[1],
                    0.5,
                    0.5
                ]
                self.rede.backward(target)
        
        print(f"✅ Consolidação concluída em {time.time() - t0:.2f}s.")
        self.salvar()

    # ------------------------------------------------------------
    # INICIALIZAÇÃO
    # ------------------------------------------------------------
    def inicializar(self):
        if os.path.exists(self.arquivo_bin):
            try:
                with open(self.arquivo_bin, "rb") as f:
                    saved = pickle.load(f)
                self.matrix = saved.get("matrix", {})
                self.blocos = saved.get("blocos", [])
                self.estados = saved.get("estados", [0.3, 0.7])
                self.rastro = saved.get("rastro", [])
                self.coords = saved.get("coords", {})
                self.inicios = saved.get("inicios", [])
                self.frases_originais = saved.get("frases_originais", [])
                self.topicos = defaultdict(list, saved.get("topicos", {}))
                self.max_tokens = saved.get("max_tokens", 40)
                self.temperatura = saved.get("temperatura", 0.7)
                self.interacoes = saved.get("interacoes", 0)
                rede_data = saved.get("rede", {})
                if rede_data:
                    self.rede.W1 = rede_data.get("W1", self.rede.W1)
                    self.rede.b1 = rede_data.get("b1", self.rede.b1)
                    self.rede.W2 = rede_data.get("W2", self.rede.W2)
                    self.rede.b2 = rede_data.get("b2", self.rede.b2)
                    self.rede.t = rede_data.get("t", 0)
                if not self.inicios:
                    self.inicios = [k for k in self.matrix if self.matrix[k].get("nexts")]
                print(f"🧠 Cérebro carregado. Palavras: {len(self.matrix)} | Frases: {len(self.frases_originais)}")
                print(f"   Interações anteriores: {self.interacoes} | Rede neural: {self.rede.t} ciclos")
                self.pronto = True
                return
            except Exception as e:
                print(f"⚠️ Erro ao carregar: {e}")

        if not self.texto.strip():
            print("⚠️ Nenhum texto. Cérebro vazio.")
            self.pronto = True
            return

        print("🔄 Processando dataset...")
        frases_raw = [f.strip() for f in self.texto.replace("!", ".").replace("?", ".").replace(";", ".").split(".") if len(f.strip()) > 0]
        frases = []
        for f in frases_raw:
            tokens = f.split()
            if len(tokens) > 40:
                for i in range(0, len(tokens), 40):
                    sub = tokens[i:i+40]
                    if len(sub) >= 2:
                        frases.append(" ".join(sub))
            else:
                frases.append(f)

        todas_frases = []
        all_tokens = []
        for frase in frases:
            tokens = frase.lower().replace(",", " , ").split()
            if len(tokens) >= 2:
                self.inicios.append(tokens[0])
                all_tokens.extend(tokens)
                all_tokens.append(".")
                todas_frases.append(tokens)
                self.frases_originais.append(tokens)
                for t in tokens:
                    self.topicos[t].append(len(self.frases_originais) - 1)

        if not self.inicios and all_tokens:
            self.inicios = list(set(t for t in all_tokens if t != "."))

        freq = {}
        for t in all_tokens:
            freq[t] = freq.get(t, 0) + 1
        for t, f in freq.items():
            self.matrix[t] = {"m": 1.5 / (f + 1e-5), "links": {}, "nexts": []}
        for i in range(len(all_tokens) - 1):
            a, b = all_tokens[i], all_tokens[i+1]
            if a in self.matrix and b in self.matrix:
                self.matrix[a]["links"][b] = self.matrix[a]["links"].get(b, 0) + 1
                if b not in self.matrix[a]["nexts"]:
                    self.matrix[a]["nexts"].append(b)

        temp_coords = defaultdict(list)
        for i in range(0, len(all_tokens), 256):
            bloco = all_tokens[i:i+256]
            if not bloco:
                continue
            h = sha256(" ".join(bloco))
            xyz = [(int(h[0:4], 16) % 200) - 100, (int(h[4:8], 16) % 200) - 100, (int(h[8:12], 16) % 200) - 100]
            self.blocos.append({"xyz": xyz, "txt": bloco})
            for t in bloco:
                temp_coords[t].append(xyz)
        for t, lista in temp_coords.items():
            xs = [p[0] for p in lista]
            ys = [p[1] for p in lista]
            zs = [p[2] for p in lista]
            self.coords[t] = [sum(xs)/len(lista), sum(ys)/len(lista), sum(zs)/len(lista)]

        print("⚙️ Treinando rede neural inicial...")
        t0 = time.time()
        num_treino = min(len(todas_frases) or len(all_tokens), 500)
        for _ in range(5):
            for i in range(num_treino):
                frase = todas_frases[i % max(1, len(todas_frases))] if todas_frases else [all_tokens[i % len(all_tokens)]]
                if len(frase) < 1:
                    continue
                emb = self._embed_frase(frase)
                self.rede.forward(emb)
                target = [
                    min(1.0, len(frase) / 20),
                    sum(1 for t in frase if len(t) > 5) / max(1, len(frase)),
                    sum(1 for t in frase if t in (',', '.')) / max(1, len(frase)),
                    0.5 + random.random() * 0.3,
                    self.estados[0],
                    self.estados[1],
                    0.5,
                    0.5
                ]
                self.rede.backward(target)
        print(f"✅ Treino inicial concluído em {time.time() - t0:.2f}s.")
        print(f"✅ Motor pronto! Palavras: {len(self.matrix)} | Frases: {len(self.frases_originais)}")
        self.pronto = True
        self.salvar()

    def salvar(self):
        try:
            with open(self.arquivo_bin, "wb") as f:
                pickle.dump({
                    "matrix": self.matrix, "blocos": self.blocos,
                    "estados": self.estados, "rastro": self.rastro,
                    "coords": self.coords, "inicios": self.inicios,
                    "frases_originais": self.frases_originais,
                    "topicos": dict(self.topicos),
                    "max_tokens": self.max_tokens,
                    "temperatura": self.temperatura,
                    "interacoes": self.interacoes,
                    "rede": {"W1": self.rede.W1, "b1": self.rede.b1,
                             "W2": self.rede.W2, "b2": self.rede.b2, "t": self.rede.t}
                }, f)
        except Exception as e:
            print(f"⚠️ Erro ao salvar: {e}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("🧬 Quintikus DLMC V82.1 – Comportamento Emergente")
    print("=" * 55)
    
    texto_inicial = ""
    if os.path.exists("amor.txt"):
        with open("amor.txt", "r", encoding="utf-8") as f:
            texto_inicial = f.read()
        print(f"📁 treino.txt carregado ({len(texto_inicial)} caracteres).")
    
    motor = QuintikusDLMC(texto_inicial)
    motor.inicializar()
    
    print("\n💬 Comandos:")
    print("   tokens:30  | temp:0.5  | debug:on/off")
    print("   train:arquivo.txt | sair")
    print("   (O motor decide sozinho como responder)\n")
    
    while True:
        try:
            entrada = input("👤: ").strip()
            if not entrada:
                continue
            
            if entrada.lower() == "sair":
                print("⚙️ Executando treino de consolidação...")
                motor.treino_consolidacao()
                print("💤 Cérebro salvo. Até mais!")
                break
            
            if entrada.lower().startswith("tokens:"):
                try:
                    val = int(entrada.split(":", 1)[1].strip())
                    motor.max_tokens = max(5, min(100, val))
                    print(f"✅ max_tokens = {motor.max_tokens}")
                except:
                    print("❌ Use: tokens:30")
                continue
            
            if entrada.lower().startswith("temp:"):
                try:
                    val = float(entrada.split(":", 1)[1].strip())
                    motor.temperatura = max(0.1, min(2.0, val))
                    print(f"✅ temperatura = {motor.temperatura}")
                except:
                    print("❌ Use: temp:0.7")
                continue
            
            if entrada.lower().startswith("debug:"):
                modo = entrada.split(":", 1)[1].strip()
                motor.debug = (modo == "on")
                print(f"✅ debug = {motor.debug}")
                continue
            
            if entrada.lower().startswith("train:"):
                arquivo = entrada.split(":", 1)[1].strip()
                if os.path.exists(arquivo):
                    with open(arquivo, "r", encoding="utf-8") as f:
                        texto = f.read()
                    motor = QuintikusDLMC(texto)
                    motor.inicializar()
                else:
                    print(f"❌ Arquivo '{arquivo}' não encontrado.")
                continue
            
            resposta = motor.pensar(entrada)
            print(f"🧠: {resposta}")
            
        except KeyboardInterrupt:
            print("\n⚙️ Executando treino de consolidação...")
            motor.treino_consolidacao()
            print("💤 Interrompido. Cérebro salvo.")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
