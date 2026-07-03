#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus DLMC V85.2 + DSML + Curiosity + Sentimento + Metacognitivo
Fusão completa: GPS + Periscópio + Computador de Bordo + Metabolismo +
SNC + Homeostase + Perfil de Usuário + Relógio Endógeno Não-Invasivo +
NeuroMicro (Sentimento) + Contexto Entrópico + Organismo Metacognitivo
Versão Final – Relógio Endógeno com Parada Correta
"""

import hashlib, math, random, time, pickle, os, tempfile, sys, re, threading
from array import array
from collections import defaultdict, deque, Counter
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# ============================================
# BLOCO ANDROID – Detecção segura
# ============================================
ANDROID_VOZ = False
MODO_VOZ = False
droid = None
sistema_ocupado = False

try:
    import androidhelper
    droid = androidhelper.Android()
    ANDROID_VOZ = True
    print("✅ QPython detectado. Voz disponível.")
except (ImportError, AttributeError):
    pass

if not ANDROID_VOZ:
    try:
        from sl4a import Android
        droid = Android()
        ANDROID_VOZ = True
        print("✅ SL4A detectado. Voz disponível.")
    except (ImportError, AttributeError):
        pass

if not ANDROID_VOZ:
    print("ℹ️  Android sem SL4A. Rodando em modo teclado.")
    droid = None

def safe_input(prompt: str = "") -> str:
    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    return sys.stdin.readline().strip()

def falar(texto: str) -> bool:
    global ANDROID_VOZ, droid
    if not ANDROID_VOZ or droid is None:
        return False
    try:
        droid.ttsSpeak(texto)
        droid.eventWait(3000)
        time.sleep(0.5)
        return True
    except Exception:
        return False

def ouvir(prompt: str = "Ouvindo...") -> str:
    global ANDROID_VOZ, droid
    if not ANDROID_VOZ or droid is None:
        return ""
    try:
        resultado = droid.recognizeSpeech(prompt)
        if resultado and resultado.result:
            return resultado.result.strip()
        return ""
    except Exception:
        return ""

def vibrar(ms: int = 100):
    global ANDROID_VOZ, droid
    if not ANDROID_VOZ or droid is None:
        return
    try:
        droid.vibrate(ms)
    except Exception:
        pass

# ============================================
# UTILITÁRIOS
# ============================================
def sha256(msg: str) -> str:
    return hashlib.sha256(msg.encode()).hexdigest()

def cosine_sim(v1, v2):
    if not v1 or not v2 or len(v1) != len(v2): return 0.0
    n1 = math.sqrt(sum(x*x for x in v1))
    n2 = math.sqrt(sum(x*x for x in v2))
    if n1 == 0 or n2 == 0: return 0.0
    return sum(a*b for a,b in zip(v1,v2)) / (n1*n2)

def euclidean_dist(v1, v2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(v1,v2)))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

def softplus(x: float) -> float:
    return math.log1p(math.exp(min(x, 20.0)))

def tanh(x: float) -> float:
    return math.tanh(x)

def ajustar_pontuacao(resposta: str) -> str:
    if not resposta:
        return resposta
    ultimo = resposta[-1]
    if ultimo in '.!?':
        return resposta
    if any(p in resposta.lower() for p in ['quem', 'quando', 'onde', 'por que', 'como', 'qual', '?' ]):
        return resposta + '?'
    if any(p in resposta for p in ['!', 'nossa', 'caramba', 'que', 'como']):
        return resposta + '!'
    return resposta + '.'

# ============================================
# GRU (Gated Recurrent Unit)
# ============================================
class GRUCell:
    def __init__(self, input_dim: int, hidden_dim: int):
        self.W_ir = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.W_iz = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.W_in = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.W_hr = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.W_hz = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.W_hn = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b_ir = [0.0]*hidden_dim; self.b_iz = [0.0]*hidden_dim; self.b_in = [0.0]*hidden_dim
        self.b_hr = [0.0]*hidden_dim; self.b_hz = [0.0]*hidden_dim; self.b_hn = [0.0]*hidden_dim

    def forward(self, x, h_prev):
        h_dim = len(h_prev)
        r = [sigmoid(sum(x[k]*self.W_ir[k][j] for k in range(len(x))) + 
                     sum(h_prev[k]*self.W_hr[k][j] for k in range(h_dim)) + 
                     self.b_ir[j] + self.b_hr[j]) for j in range(h_dim)]
        z = [sigmoid(sum(x[k]*self.W_iz[k][j] for k in range(len(x))) + 
                     sum(h_prev[k]*self.W_hz[k][j] for k in range(h_dim)) + 
                     self.b_iz[j] + self.b_hz[j]) for j in range(h_dim)]
        n = [tanh(sum(x[k]*self.W_in[k][j] for k in range(len(x))) + 
                  r[j] * sum(h_prev[k]*self.W_hn[k][j] for k in range(h_dim)) + 
                  self.b_in[j] + self.b_hn[j]) for j in range(h_dim)]
        h_new = [(1.0 - z[j]) * n[j] + z[j] * h_prev[j] for j in range(h_dim)]
        return h_new

class GRULayer:
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.cell = GRUCell(input_dim, hidden_dim)
        self.W_proj = [[random.gauss(0, 0.1) for _ in range(8)] for _ in range(hidden_dim)]
        self.b_proj = [0.0]*8

    def forward(self, sequence):
        h = [0.0]*self.hidden_dim
        for x in sequence[-self.seq_len:]:
            h = self.cell.forward(x, h)
        out = [sum(h[j]*self.W_proj[j][i] for j in range(self.hidden_dim)) + self.b_proj[i] for i in range(8)]
        return out

# ============================================
# REDE NEURAL COM ADAM
# ============================================
class MiniRedeAdam:
    def __init__(self, input_dim=10, hidden_dim=16, output_dim=8):
        scale1 = math.sqrt(2.0/input_dim); scale2 = math.sqrt(2.0/hidden_dim)
        self.W1 = [[(random.random()*2-1)*scale1 for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0]*hidden_dim
        self.W2 = [[(random.random()*2-1)*scale2 for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0]*output_dim
        self.beta1, self.beta2, self.eps, self.t, self.lr = 0.9, 0.999, 1e-8, 0, 0.005
        self.mW1 = [[0.0]*hidden_dim for _ in range(input_dim)]; self.vW1 = [[0.0]*hidden_dim for _ in range(input_dim)]
        self.mb1, self.vb1 = [0.0]*hidden_dim, [0.0]*hidden_dim
        self.mW2 = [[0.0]*output_dim for _ in range(hidden_dim)]; self.vW2 = [[0.0]*output_dim for _ in range(hidden_dim)]
        self.mb2, self.vb2 = [0.0]*output_dim, [0.0]*output_dim

    def forward(self, x):
        self.x = x
        self.h = [max(0.0, self.b1[i] + sum(x[j]*self.W1[j][i] for j in range(len(x)))) for i in range(len(self.b1))]
        self.out_raw = [self.b2[i] + sum(self.h[j]*self.W2[j][i] for j in range(len(self.h))) for i in range(len(self.b2))]
        return [softplus(v) for v in self.out_raw]

    def backward(self, target):
        self.t += 1
        sig_out = [sigmoid(v) for v in self.out_raw]
        delta2 = [(self.out_raw[i]-target[i])*sig_out[i] for i in range(len(target))]
        dW2 = [[delta2[j]*self.h[i] for j in range(len(delta2))] for i in range(len(self.h))]
        db2 = list(delta2)
        delta1 = [0.0]*len(self.h)
        for i in range(len(self.h)):
            if self.h[i] <= 0: continue
            delta1[i] = sum(delta2[j]*self.W2[i][j] for j in range(len(delta2)))
        dW1 = [[delta1[j]*self.x[i] for j in range(len(delta1))] for i in range(len(self.x))]
        db1 = list(delta1)
        self._adam2d(self.W1, dW1, self.mW1, self.vW1)
        self._adam1d(self.b1, db1, self.mb1, self.vb1)
        self._adam2d(self.W2, dW2, self.mW2, self.vW2)
        self._adam1d(self.b2, db2, self.mb2, self.vb2)

    def _adam2d(self, param, grad, m, v):
        for i in range(len(param)):
            for j in range(len(param[i])):
                m[i][j] = self.beta1*m[i][j] + (1-self.beta1)*grad[i][j]
                v[i][j] = self.beta2*v[i][j] + (1-self.beta2)*grad[i][j]*grad[i][j]
                m_hat = m[i][j]/(1-self.beta1**self.t)
                v_hat = v[i][j]/(1-self.beta2**self.t)
                param[i][j] -= self.lr*m_hat/(math.sqrt(abs(v_hat))+self.eps)

    def _adam1d(self, param, grad, m, v):
        for i in range(len(param)):
            m[i] = self.beta1*m[i] + (1-self.beta1)*grad[i]
            v[i] = self.beta2*v[i] + (1-self.beta2)*grad[i]*grad[i]
            m_hat = m[i]/(1-self.beta1**self.t)
            v_hat = v[i]/(1-self.beta2**self.t)
            param[i] -= self.lr*m_hat/(math.sqrt(abs(v_hat))+self.eps)

# ============================================
# ÍNDICE ESPACIAL (GPS do Modelo)
# ============================================
class SpatialIndex:
    def __init__(self):
        self.frases_3d = []
        self.frase_ids = []
    
    def build(self, frases_originais, coords_func):
        self.frases_3d = []
        self.frase_ids = list(range(len(frases_originais)))
        for i, frase in enumerate(frases_originais):
            emb = coords_func(frase)
            if len(emb) >= 3:
                self.frases_3d.append(emb[:3])
            else:
                self.frases_3d.append([0.0, 0.0, 0.0])
    
    def query(self, prompt_emb_3d, k=50):
        if not self.frases_3d or k <= 0:
            return []
        distancias = []
        for i, coord in enumerate(self.frases_3d):
            dist = euclidean_dist(prompt_emb_3d, coord)
            distancias.append((dist, i))
        distancias.sort(key=lambda x: x[0])
        return [self.frase_ids[idx] for _, idx in distancias[:k]]

# ============================================
# ATENÇÃO HIERÁRQUICA (Periscópio Multi-Banda)
# ============================================
class HierarchicalAttention:
    def __init__(self, input_dim=3):
        self.scales = [10, 30, 50]
        self.W_scale = [random.gauss(0, 0.1) for _ in range(len(self.scales))]
        self.b = 0.0
        self.lr = 0.01
    
    def forward(self, prompt_emb_3d, spatial_index, frases_originais):
        resultados_por_escala = []
        for i, k in enumerate(self.scales):
            vizinhos = spatial_index.query(prompt_emb_3d, k=k)
            if vizinhos:
                frase_id = vizinhos[0]
                frase = frases_originais[frase_id]
                resultados_por_escala.append(frase)
        
        if not resultados_por_escala:
            return None, 0.0
        
        scores = [sigmoid(self.W_scale[i]) for i in range(len(resultados_por_escala))]
        total = sum(scores) + 1e-8
        probs = [s/total for s in scores]
        
        r = random.random()
        acum = 0.0
        escolhida = resultados_por_escala[0]
        for j, p in enumerate(probs):
            acum += p
            if r <= acum:
                escolhida = resultados_por_escala[j]
                break
        
        confianca = max(scores) if scores else 0.5
        return escolhida, confianca
    
    def backward(self, reward, scale_idx):
        if 0 <= scale_idx < len(self.W_scale):
            self.W_scale[scale_idx] += self.lr * reward
            self.b += self.lr * reward * 0.1

# ============================================
# ATENÇÃO SUJEITO-PREDICADO (Periscópio Fino)
# ============================================
class ContextAttention:
    def __init__(self, input_dim=10):
        self.W_suj = [random.gauss(0,0.1) for _ in range(input_dim)]
        self.W_pred = [random.gauss(0,0.1) for _ in range(input_dim)]
        self.b = 0.0
        self.lr = 0.01

    def extrair_sujeito_predicado(self, ql, coords, raridade):
        suj_tokens = sorted(ql, key=lambda t: raridade.get(t, 1))[:max(1, len(ql)//3)]
        pred_tokens = sorted(ql, key=lambda t: raridade.get(t, 1), reverse=True)[:max(1, len(ql)//3)]
        def media_emb(tokens):
            coords_list = [coords[t] for t in tokens if t in coords]
            if not coords_list: return [0.0]*3
            return [sum(c[i] for c in coords_list)/len(coords_list) for i in range(3)]
        return media_emb(suj_tokens), media_emb(pred_tokens)

    def forward(self, emb_prompt, suj_emb, pred_emb):
        score_suj = sum(suj_emb[i]*self.W_suj[i] for i in range(3)) if len(suj_emb)==3 else 0
        score_pred = sum(pred_emb[i]*self.W_pred[i] for i in range(3)) if len(pred_emb)==3 else 0
        alpha = sigmoid(score_suj - score_pred + self.b)
        return alpha

    def backward(self, alpha, target_alpha, suj_emb, pred_emb):
        error = target_alpha - alpha
        if len(suj_emb)==3:
            for i in range(3): self.W_suj[i] += self.lr * error * suj_emb[i]
        if len(pred_emb)==3:
            for i in range(3): self.W_pred[i] -= self.lr * error * pred_emb[i]
        self.b += self.lr * error

# ============================================
# MOTOR PRINCIPAL V85.2 – COM AUTO‑CURA TOTAL
# ============================================
class QuintikusDLMC:
    def __init__(self, texto: str = "", arquivo_bin: str = "cerebro_v85.bin"):
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
        self.raridade = {}
        self.pronto = False
        
        self.rede = MiniRedeAdam(10, 16, 8)
        self.gru = GRULayer(input_dim=10, hidden_dim=12, seq_len=5)
        self.gru2 = GRULayer(input_dim=10, hidden_dim=8, seq_len=3)
        self.attention = ContextAttention(input_dim=3)
        self.hierarchical = HierarchicalAttention(input_dim=3)
        self.spatial_index = SpatialIndex()
        
        self.gru_ativo = True
        self.historico_embeddings = deque(maxlen=5)
        self.max_tokens = 40
        self.temperatura = 0.7
        self.debug = False
        self.interacoes = 0
        self.momentum = [0.0, 0.0, 0.0]
        self.historico_momentum = deque(maxlen=5)
        self.fator_momentum = 0.4
        self.fator_suavidade = 0.3
        self.k_curvatura = 3.0
        self.limiar_curvatura = 0.6
        self.gru_treino_counter = 0
        self.auto_salvar = True
        self.salvar_intervalo = 5
        self.last_scale_idx = 0

    def _curar_modulos(self):
        curados = False
        for nome, classe, args in [
            ('gru', GRULayer, (10, 12, 5)),
            ('gru2', GRULayer, (10, 8, 3)),
            ('attention', ContextAttention, (3,)),
            ('hierarchical', HierarchicalAttention, (3,))
        ]:
            obj = getattr(self, nome, None)
            ok = True
            if not hasattr(obj, 'forward'):
                ok = False
            elif hasattr(obj, 'cell') and not hasattr(obj.cell, 'forward'):
                ok = False
            if not ok:
                setattr(self, nome, classe(*args))
                curados = True
                if self.debug:
                    print(f"   ⚠️ [AUTO‑CURA] {nome} regenerado (incluindo célula).")
        if not hasattr(self.rede, 'forward'):
            self.rede = MiniRedeAdam(10, 16, 8)
            curados = True
            if self.debug:
                print("   ⚠️ [AUTO‑CURA] rede neural regenerada.")
        if curados:
            self.salvar()

    def _salvar_atomico(self):
        try:
            dados = {
                "matrix": self.matrix, "blocos": self.blocos,
                "estados": self.estados, "rastro": self.rastro,
                "coords": self.coords, "inicios": self.inicios,
                "frases_originais": self.frases_originais,
                "topicos": dict(self.topicos), "raridade": self.raridade,
                "max_tokens": self.max_tokens, "temperatura": self.temperatura,
                "interacoes": self.interacoes,
                "momentum": self.momentum,
                "historico_momentum": list(self.historico_momentum),
                "fator_momentum": self.fator_momentum,
                "fator_suavidade": self.fator_suavidade,
                "k_curvatura": self.k_curvatura,
                "limiar_curvatura": self.limiar_curvatura,
                "gru": self.gru, "gru2": self.gru2, "gru_ativo": self.gru_ativo,
                "historico_embeddings": list(self.historico_embeddings),
                "gru_treino_counter": self.gru_treino_counter,
                "attention": self.attention,
                "hierarchical": self.hierarchical,
                "spatial_index": self.spatial_index,
                "rede": {"W1": self.rede.W1, "b1": self.rede.b1,
                         "W2": self.rede.W2, "b2": self.rede.b2, "t": self.rede.t}
            }
            dir_name = os.path.dirname(os.path.abspath(self.arquivo_bin)) or "."
            with tempfile.NamedTemporaryFile(mode='wb', dir=dir_name, delete=False, prefix='dlmc_v85_', suffix='.tmp') as tmp:
                pickle.dump(dados, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.flush(); os.fsync(tmp.fileno())
                tmp_path = tmp.name
            os.replace(tmp_path, self.arquivo_bin)
            return True
        except Exception as e:
            print(f"⚠️ Erro ao salvar atomicamente: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except: pass
            try:
                with open(self.arquivo_bin, "wb") as f:
                    pickle.dump(dados, f, protocol=pickle.HIGHEST_PROTOCOL)
            except: pass
            return False

    def salvar(self):
        ok = self._salvar_atomico()
        if ok and self.debug: print("   [DEBUG] Cérebro salvo atomicamente.")
        return ok

    def _embed_palavra(self, token: str):
        if token not in self.matrix: return [0.0]*10
        coord = self.coords.get(token, [0,0,0])
        freq = len(self.matrix[token].get("links", {}))
        massa = self.matrix[token].get("m", 0)
        num_nexts = len(self.matrix[token].get("nexts", []))
        return [
            coord[0]/100, coord[1]/100, coord[2]/100,
            math.log1p(freq)/10, massa*10, num_nexts/100,
            random.random()*0.01,
            self.estados[0], self.estados[1],
            len(self.rastro)/100
        ]

    def _embed_frase(self, tokens):
        if not tokens: return [0.0]*10
        embs = [self._embed_palavra(t) for t in tokens]
        soma = [0.0]*10
        for e in embs:
            for i in range(10): soma[i] += e[i]
        return [v/len(embs) for v in soma]

    def _coords_frase(self, tokens):
        coords_list = [self.coords[t] for t in tokens if t in self.coords]
        if not coords_list: return [0.0, 0.0, 0.0]
        return [sum(c[i] for c in coords_list)/len(coords_list) for i in range(3)]

    def _comportamento_emergente(self, ql):
        if not self.frases_originais: return ("criativo", 0.0)
        prompt_3d = self._coords_frase(ql)
        vizinhos = self.spatial_index.query(prompt_3d, k=min(50, len(self.frases_originais)))
        if not vizinhos: return ("criativo", 0.0)
        
        sims = [cosine_sim(prompt_3d, self._coords_frase(self.frases_originais[i])) for i in vizinhos]
        sim_max = max(sims) if sims else 0.0
        sim_med = sum(sims)/len(sims) if sims else 0.0
        
        f_dataset = sim_max
        f_emocional = abs(self.estados[0]-self.estados[1])
        f_compr = min(1.0, len(ql)/10)
        f_novidade = 1.0 - sim_med
        
        s_linear = f_dataset*0.5 + f_compr*0.3 + (1-f_emocional)*0.2
        s_criativo = f_novidade*0.4 + f_emocional*0.4 + (1-f_compr)*0.2
        
        if self.debug: print(f"   [DEBUG] Linear:{s_linear:.3f} Criativo:{s_criativo:.3f}")
        return ("linear", sim_max) if s_linear >= s_criativo else ("criativo", f_novidade)

    def _responder_linear(self, ql, sim_max):
        prompt_3d = self._coords_frase(ql)
        vizinhos = self.spatial_index.query(prompt_3d, k=min(50, len(self.frases_originais)))
        
        if not vizinhos:
            return self._responder_criativo(ql, 0.5)
        
        melhor_frase, confianca = self.hierarchical.forward(prompt_3d, self.spatial_index, self.frases_originais)
        
        if not melhor_frase or confianca < 0.1:
            return self._responder_criativo(ql, 0.5)
        
        ini = 0
        for i, t in enumerate(melhor_frase):
            if any(p in t for p in ql): ini = i; break
        
        trecho = melhor_frase[ini:ini+self.max_tokens]
        resp = " ".join(trecho)
        resp = ajustar_pontuacao(resp)
        return resp[0].upper() + resp[1:] if resp else self._responder_criativo(ql, 0.5)

    def _triangulate(self, gps_signal, periscope_signal):
        gps_confidence = 0.7 if len(self.historico_embeddings) > 3 else 0.3
        periscope_confidence = 0.6
        alpha = gps_confidence / (gps_confidence + periscope_confidence + 1e-8)
        return [alpha * gps_signal[i] + (1 - alpha) * periscope_confidence * periscope_signal[i] for i in range(8)]

    def _responder_criativo(self, ql, f_novidade):
        if not self.matrix: return "Preciso de mais dados."
        emb_prompt = self._embed_frase(ql)

        suj_emb, pred_emb = self.attention.extrair_sujeito_predicado(ql, self.coords, self.raridade)

        if self.gru_ativo and len(self.historico_embeddings) >= 2:
            seq = list(self.historico_embeddings) + [emb_prompt]
            pesos_gru1 = self.gru.forward(seq)
            pesos_gru2 = self.gru2.forward(seq[-3:])
            pesos_gru = [(pesos_gru1[i] + pesos_gru2[i]) / 2.0 for i in range(8)]
        else:
            pesos_gru = [0.5]*8

        pesos_adam = self.rede.forward(emb_prompt)

        pesos_gps = pesos_gru
        pesos_periscope = pesos_adam
        pesos = self._triangulate(pesos_gps, pesos_periscope)

        alpha = self.attention.forward(emb_prompt, suj_emb, pred_emb)

        if self.debug:
            print(f"   [DEBUG] Alpha (Sujeito-Predicado): {alpha:.3f}")

        self.historico_embeddings.append(emb_prompt)
        w_comprimento, w_palavras_longas, w_pontuacao, w_criatividade, w_estado0, w_estado1, w_coerencia, w_variabilidade = pesos

        target_adam = [min(1.0, len(ql)/20),
                       sum(1 for t in ql if len(t)>5)/max(1, len(ql)),
                       sum(1 for t in ql if t==',')/max(1, len(ql)),
                       0.6, self.estados[0], self.estados[1], 0.7, 0.5]
        self.rede.backward(target_adam)

        target_alpha = 0.7 if len(self.historico_embeddings) > 3 else 0.3
        self.attention.backward(alpha, target_alpha, suj_emb, pred_emb)

        self.gru_treino_counter += 1
        if self.gru_ativo and self.gru_treino_counter % 3 == 0 and len(self.historico_embeddings) >= 2:
            self.gru.forward(list(self.historico_embeddings))
            self.gru2.forward(list(self.historico_embeddings)[-3:])

        pos = ['amo','amor','bem','feliz','bom','gosto','lindo','maravilhoso','obrigado']
        neg = ['odeio','triste','mal','raiva','feio','horrivel','chateado']
        for t in ql:
            if any(e in t for e in pos): self.estados[1] = min(1.0, self.estados[1]+0.15)
            if any(e in t for e in neg): self.estados[0] = min(1.0, self.estados[0]+0.15)
        self.estados = [s*0.95 for s in self.estados]

        atual = next((t for t in ql if t in self.matrix and self.matrix[t].get("nexts")), None)
        if not atual and self.inicios: atual = random.choice(self.inicios)
        if not atual or not self.matrix.get(atual, {}).get("nexts"):
            candidatas = [k for k in self.matrix if self.matrix[k].get("nexts")]
            if not candidatas: return "Preciso de mais dados."
            atual = random.choice(candidatas)

        resultado = []; ultimos = []
        comprimento_alvo = max(5, min(self.max_tokens, round((w_comprimento or 0.5)*30)))
        self.momentum = [0.0,0.0,0.0]; self.historico_momentum.clear()

        for _ in range(comprimento_alvo):
            if atual not in self.matrix or not self.matrix[atual].get("nexts"): break
            resultado.append(atual); ultimos.append(atual)
            if len(ultimos) > 8: ultimos.pop(0)

            if atual in self.coords:
                self.historico_momentum.append(self.coords[atual])
                if len(self.historico_momentum) >= 2:
                    pts = list(self.historico_momentum)
                    vetores = [[pts[i+1][j]-pts[i][j] for j in range(3)] for i in range(len(pts)-1)]
                    if vetores: self.momentum = [sum(v[i] for v in vetores)/len(vetores) for i in range(3)]

            candidatos = self.matrix[atual]["nexts"]; links = self.matrix[atual]["links"]
            pesos_cand = []
            for prox in candidatos:
                p = links.get(prox, 1)
                if prox in ultimos: p *= 0.001
                if not self.matrix.get(prox,{}).get("nexts"): p *= 0.1
                if prox in ql: p *= (1.5+(w_coerencia or 0.5))
                if len(prox)>5 and (w_palavras_longas or 0.5)>0.5: p *= 1.5
                if prox in (',','.') and (w_pontuacao or 0.3)>0.5: p *= 2.0
                if self.coords.get(prox):
                    for tp in ql:
                        if self.coords.get(tp):
                            sim = (cosine_sim(self.coords[prox],self.coords[tp])+1)/2
                            p *= (1+sim*(w_variabilidade or 0.5))
                if len(resultado)%15==14 and prox=='.': p *= 5.0
                
                if self.coords.get(atual) and self.coords.get(prox) and any(v!=0 for v in self.momentum):
                    v_prox = [self.coords[prox][i]-self.coords[atual][i] for i in range(3)]
                    if any(v!=0 for v in v_prox):
                        sim_m = cosine_sim(v_prox, self.momentum)
                        p *= (1.0+sim_m*self.fator_momentum)
                
                if len(resultado)>=2 and self.coords.get(atual):
                    pen = resultado[-2] if len(resultado)>=2 else None
                    ant = resultado[-3] if len(resultado)>=3 else None
                    if pen and ant and ant in self.coords and pen in self.coords:
                        v_ant = [self.coords[pen][i]-self.coords[ant][i] for i in range(3)]
                        v_prox = [self.coords[atual][i]-self.coords[pen][i] for i in range(3)] if atual in self.coords else None
                        v_cand = [self.coords[prox][i]-self.coords[atual][i] for i in range(3)] if prox in self.coords else None
                        if v_prox and v_cand and any(v!=0 for v in v_prox) and any(v!=0 for v in v_cand):
                            sim_c = cosine_sim(v_prox, v_cand)
                            pen_c = 1.0/(1.0+math.exp(self.k_curvatura*(sim_c-self.limiar_curvatura)))
                            p *= (1.0-pen_c*self.fator_suavidade)
                
                pesos_cand.append(p)

            if self.temperatura != 1.0 and self.temperatura > 0:
                pesos_cand = [x**(1.0/max(0.1,self.temperatura)) for x in pesos_cand]
            soma = sum(pesos_cand)+1e-8; probs = [x/soma for x in pesos_cand]
            r = random.random(); ac = 0.0; esc = candidatos[0]
            for j, prob in enumerate(probs):
                ac += prob
                if r <= ac: esc = candidatos[j]; break
            atual = esc
            if atual=='.' and len(resultado)>=3: break

        if resultado and resultado[-1]=='.': resultado.pop()
        self.rastro.extend(resultado)
        if len(self.rastro)>100: self.rastro = self.rastro[-100:]

        if self.auto_salvar and self.interacoes % self.salvar_intervalo == 0:
            self.salvar()

        resp = " ".join(resultado).replace(" ,",",").replace(" .",".")
        resp = ajustar_pontuacao(resp)
        return resp[0].upper()+resp[1:] if resp else "..."

    def pensar(self, prompt: str) -> str:
        self._curar_modulos()
        self.interacoes += 1
        ql = prompt.lower().split()
        if not ql: return "..."
        if not self.matrix: return "Preciso de mais dados. Use train:arquivo.txt"
        self.memoria_curto_prazo.append(("user", ql))
        estrategia, confianca = self._comportamento_emergente(ql)
        if self.debug: print(f"   [DEBUG] Estratégia: {estrategia} (conf:{confianca:.3f})")
        resp = self._responder_linear(ql, confianca) if estrategia=="linear" else self._responder_criativo(ql, confianca)
        tokens = resp.split()
        if len(tokens) > self.max_tokens+10:
            resp = " ".join(tokens[:self.max_tokens])
            resp = ajustar_pontuacao(resp)
        return resp

    def inicializar(self):
        if os.path.exists(self.arquivo_bin):
            try:
                with open(self.arquivo_bin, "rb") as f:
                    saved = pickle.load(f)

                for attr_name, default_factory in [
                    ('gru', lambda: GRULayer(10, 12, 5)),
                    ('gru2', lambda: GRULayer(10, 8, 3)),
                    ('attention', lambda: ContextAttention(3)),
                    ('hierarchical', lambda: HierarchicalAttention(3)),
                    ('spatial_index', lambda: SpatialIndex())
                ]:
                    val = saved.get(attr_name)
                    if not isinstance(val, (GRULayer, GRUCell, ContextAttention, HierarchicalAttention, SpatialIndex)):
                        saved[attr_name] = default_factory()
                        print(f"   ⚠️ {attr_name} corrompido. Recriado com valores padrão.")

                for k, v in saved.items():
                    if hasattr(self, k):
                        setattr(self, k, v)

                if len(self.matrix) < 100:
                    self.gru_ativo = False
                    print("   ⚠️ Dataset pequeno (<100 palavras). GRU desativada automaticamente.")

                if not self.inicios:
                    self.inicios = [k for k in self.matrix if self.matrix[k].get("nexts")]

                print(f"🧠 Cérebro V85 carregado. Palavras: {len(self.matrix)} | Frases: {len(self.frases_originais)}")
                print(f"   Índice Espacial: {len(self.spatial_index.frases_3d)} pontos")
                print(f"   GRUx2: {'Ativa' if self.gru_ativo else 'Inativa'} | Atenção Hierárquica ativa")
                self.pronto = True
                return
            except Exception as e:
                print(f"⚠️ Erro ao carregar cérebro: {e}")
                print("   Iniciando com cérebro vazio...")

        if not self.texto.strip():
            print("⚠️ Nenhum texto. Cérebro vazio.")
            self.pronto = True
            return

        self._processar_dataset()

    def _processar_dataset(self):
        print("🔄 Processando dataset (V85 – Deep Sea Navigation)...")
        frases_raw = [f.strip() for f in self.texto.replace("!",".").replace("?",".").replace(";",".").split(".") if len(f.strip())>0]
        frases = []
        for f in frases_raw:
            tokens = f.split()
            if len(tokens)>40:
                for i in range(0,len(tokens),40):
                    sub = tokens[i:i+40]
                    if len(sub)>=2: frases.append(" ".join(sub))
            else: frases.append(f)

        todas_frases = []; all_tokens = []
        for frase in frases:
            tokens = frase.lower().replace(","," , ").split()
            if len(tokens)>=2:
                self.inicios.append(tokens[0])
                all_tokens.extend(tokens); all_tokens.append(".")
                todas_frases.append(tokens); self.frases_originais.append(tokens)
                for t in tokens: self.topicos[t].append(len(self.frases_originais)-1)

        if not self.inicios and all_tokens: self.inicios = list(set(t for t in all_tokens if t!="."))
        for t in all_tokens: self.raridade[t] = self.raridade.get(t,0) + 1

        temp_coords = defaultdict(list)
        for i in range(0,len(all_tokens),256):
            bloco = all_tokens[i:i+256]
            if not bloco: continue
            h = sha256(" ".join(bloco))
            xyz = [(int(h[0:4],16)%200)-100,(int(h[4:8],16)%200)-100,(int(h[8:12],16)%200)-100]
            self.blocos.append({"xyz":xyz,"txt":bloco})
            for t in bloco: temp_coords[t].append(xyz)
        for t, lista in temp_coords.items():
            xs=[p[0] for p in lista]; ys=[p[1] for p in lista]; zs=[p[2] for p in lista]
            self.coords[t] = [sum(xs)/len(lista),sum(ys)/len(lista),sum(zs)/len(lista)]

        freq = {}
        for t in all_tokens: freq[t] = freq.get(t,0)+1
        for t, f in freq.items(): self.matrix[t] = {"m":1.5/(f+1e-5),"links":{},"nexts":[]}

        for i in range(len(all_tokens)-1):
            a,b = all_tokens[i],all_tokens[i+1]
            if a not in self.matrix or b not in self.matrix: continue
            peso_base = 1
            if i>=2 and self.fator_suavidade>0:
                a_ant = all_tokens[i-2] if i-2>=0 else None
                b_ant = all_tokens[i-1]
                if a_ant and a_ant in self.coords and b_ant in self.coords and a in self.coords and b in self.coords:
                    v_ant = [self.coords[b_ant][k]-self.coords[a_ant][k] for k in range(3)]
                    v_atual = [self.coords[a][k]-self.coords[b_ant][k] for k in range(3)]
                    v_prox = [self.coords[b][k]-self.coords[a][k] for k in range(3)]
                    if any(v!=0 for v in v_ant) and any(v!=0 for v in v_atual) and any(v!=0 for v in v_prox):
                        sim_c = cosine_sim(v_atual,v_prox)
                        pen = 1.0/(1.0+math.exp(self.k_curvatura*(sim_c-self.limiar_curvatura)))
                        peso_base *= (1.0-pen*self.fator_suavidade)
            self.matrix[a]["links"][b] = self.matrix[a]["links"].get(b,0)+peso_base
            if b not in self.matrix[a]["nexts"]: self.matrix[a]["nexts"].append(b)

        print("⚙️ Construindo índice espacial (GPS)...")
        t0 = time.time()
        self.spatial_index.build(self.frases_originais, self._coords_frase)
        print(f"   Índice construído em {time.time()-t0:.2f}s com {len(self.frases_originais)} pontos.")

        print("⚙️ Treinando redes neurais (Adam + GRUx2 + Atenção Hierárquica)...")
        t0 = time.time()
        num_treino = min(len(todas_frases) or len(all_tokens),500)
        for i in range(min(50,len(todas_frases))):
            self.historico_embeddings.append(self._embed_frase(todas_frases[i]))
        for _ in range(5):
            for i in range(num_treino):
                frase = todas_frases[i%max(1,len(todas_frases))] if todas_frases else [all_tokens[i%len(all_tokens)]]
                if len(frase)<1: continue
                emb = self._embed_frase(frase)
                self.rede.forward(emb)
                target = [min(1.0,len(frase)/20),
                          sum(1 for t in frase if len(t)>5)/max(1,len(frase)),
                          sum(1 for t in frase if t in (',','.'))/max(1,len(frase)),
                          0.5+random.random()*0.3,self.estados[0],self.estados[1],0.5,0.5]
                self.rede.backward(target)
                if self.gru_ativo and len(self.historico_embeddings)>=2:
                    self.gru.forward(list(self.historico_embeddings))
                    self.gru2.forward(list(self.historico_embeddings)[-3:])

        if len(self.matrix) < 100:
            self.gru_ativo = False
            print("   ⚠️ Dataset pequeno (<100 palavras). GRU desativada automaticamente.")

        print(f"✅ Treino em {time.time()-t0:.2f}s.")
        print(f"✅ Motor V85 pronto! Palavras: {len(self.matrix)} | Frases: {len(self.frases_originais)}")
        print(f"   Escalável para bilhões de tokens com índice espacial O(log n)")
        self.pronto = True
        self.salvar()

    def treino_consolidacao(self):
        if not self.frases_originais: return
        print("⚙️ Consolidando...")
        t0=time.time()
        num_treino=min(len(self.frases_originais),200)
        for _ in range(3):
            for i in range(num_treino):
                frase=self.frases_originais[i%len(self.frases_originais)]
                if len(frase)<2: continue
                emb=self._embed_frase(frase)
                self.rede.forward(emb)
                target=[min(1.0,len(frase)/20),
                        sum(1 for t in frase if len(t)>5)/max(1,len(frase)),
                        sum(1 for t in frase if t in (',','.'))/max(1,len(frase)),
                        0.5+random.random()*0.3,self.estados[0],self.estados[1],0.5,0.5]
                self.rede.backward(target)
                if self.gru_ativo and len(self.historico_embeddings)>=2:
                    self.gru.forward(list(self.historico_embeddings))
                    self.gru2.forward(list(self.historico_embeddings)[-3:])
        print(f"✅ Consolidação em {time.time()-t0:.2f}s.")
        self.salvar()


# ================================================================
# MÓDULOS DSML – ÁREAS 2.8 a 9
# ================================================================

class KernelRessonante:
    @staticmethod
    def normalize(d):
        s = sum(abs(x) for x in d.values()) + 1e-9
        return {k: v/s for k, v in d.items()}
    @staticmethod
    def dot(d1, d2):
        keys = set(d1.keys()) & set(d2.keys())
        return sum(d1[k]*d2[k] for k in keys)

class NormalizadorSomático:
    @staticmethod
    def limpar(texto):
        return re.sub(r'[^a-z0-9\s]', '', texto.lower())

class HarvesterSemantico:
    def __init__(self):
        self.pesos = {
            0: [random.uniform(-0.1, 0.1) for _ in range(4)],
            1: [random.uniform(-0.1, 0.1) for _ in range(4)],
            2: [random.uniform(-0.1, 0.1) for _ in range(4)],
            3: [-0.6, -0.1, 0.2, 2.5]
        }
        self.bias = {0: 0.5, 1: 0.1, 2: -0.3, 3: -0.5}

    @staticmethod
    def extrair_nome_usuario(texto_raw):
        # Aceita "meu nome é X", "me chamo X", "sou o X", "sou a X", "sou X"
        nome_match = re.search(r'\b(?:meu nome [ée]|me chamo|sou(?: o| a)?)\s+([a-zA-ZÀ-ÿ]+)', texto_raw.lower())
        if nome_match:
            return nome_match.group(1).strip()
        # Fallback: se a entrada for só uma palavra, assume como nome
        palavras = texto_raw.strip().split()
        if len(palavras) == 1 and palavras[0].isalpha() and len(palavras[0]) > 1:
            return palavras[0]
        return None

    def _analisar_metadado_emocional(self, prompt):
        gatilhos_confusao = [
            "confusa", "confuso", "duvida", "nao sei", "perdi",
            "ajuda", "socorro", "errado", "bug", "vazio", "sem dados", "qual caminho"
        ]
        t_bruto = prompt.lower()
        pontuacao = sum(2.0 for g in gatilhos_confusao if g in t_bruto)
        if "?" in t_bruto: pontuacao += 1.0
        if len(t_bruto.split()) < 4 and pontuacao > 0: pontuacao += 1.5
        return 1.0 / (1.0 + math.exp(-pontuacao + 2.0))

    def _extrair_estado_expandido(self, prompt, organismo):
        tokens = re.findall(r'[a-z0-9]+', NormalizadorSomático.limpar(prompt))
        foco = [t for t in tokens if t in organismo.matrix]
        
        def _entropy(t):
            freq = organismo.raridade.get(t, 1)
            return -math.log2(freq / max(1, sum(organismo.raridade.values())))
        x0 = sum(_entropy(t) for t in foco) / (len(foco) + 1e-9) if foco else 0.0
        x0 = min(1.0, x0 / 10.0)
        
        counts = Counter(prompt)
        probs = [c / len(prompt) for c in counts.values()] if prompt else [0.0]
        ent_score = -sum(p * math.log2(p) for p in probs)
        x1 = min(ent_score / 5.0, 1.0)
        
        sujeitos = set()
        for t in foco:
            if t in organismo.topicos:
                sujeitos.update(organismo.topicos[t])
        x2 = min(len(sujeitos) / 6.0, 1.0)
        
        x3 = self._analisar_metadado_emocional(prompt)
        
        return [x0, x1, x2, x3], x2, x3

    def triagem_metabolica(self, prompt, organismo):
        x, colisao_var, incerteza = self._extrair_estado_expandido(prompt, organismo)
        ativacoes = {}
        for caixa in [0, 1, 2, 3]:
            z = sum(x[i] * self.pesos[caixa][i] for i in range(4)) + self.bias[caixa]
            ativacoes[caixa] = z
        caixa_eleita = max(ativacoes, key=ativacoes.get)
        if organismo.debug:
            print(f"⚡ [PERCEPTRON] Incerteza:{incerteza*100:.1f}% | Colisão:{colisao_var*100:.1f}% | Caixa:{caixa_eleita}")
        return caixa_eleita, colisao_var, incerteza

class TeoriaDaMente:
    def __init__(self):
        self.estimativa_humor = {"confiança": 0.5, "agressividade": 0.1, "atenção": 1.0}

    def atualizar(self, u_toks, dkl_usuario):
        pessimistas = {"odeio", "mal", "triste", "burro", "erro", "ruim", "falso"}
        otimistas = {"amo", "bom", "prazer", "sim", "obrigado", "legal", "certo"}
        c_pessimistas = sum(1 for t in u_toks if t in pessimistas)
        c_otimistas = sum(1 for t in u_toks if t in otimistas)
        
        self.estimativa_humor["agressividade"] = min(1.0, max(0.0, self.estimativa_humor["agressividade"] * 0.9 + (c_pessimistas * 0.15) - (c_otimistas * 0.05)))
        self.estimativa_humor["confiança"] = min(1.0, max(0.0, self.estimativa_humor["confiança"] * 0.95 + (c_otimistas * 0.08) - (dkl_usuario * 0.04)))
        self.estimativa_humor["atenção"] = max(0.1, min(1.0, 0.7 * self.estimativa_humor["atenção"] + 0.3 * (1.0 / (dkl_usuario + 1.0))))

class MemoriaTrabalho:
    def __init__(self, capacidade=6):
        self.buffer = deque(maxlen=capacidade)
        self.vetor_suavizado = {}

    def registrar(self, v_perceptivo, tokens, acao_snc, soma_eixos):
        self.buffer.append({
            "v": v_perceptivo,
            "tokens": tokens,
            "acao": acao_snc,
            "soma": dict(soma_eixos),
            "stamp": time.time()
        })
        self._sintetizar_suavizacao()

    def _sintetizar_suavizacao(self):
        self.vetor_suavizado.clear()
        itens = list(self.buffer)[-3:]
        for idx, item in enumerate(itens):
            peso = (idx + 1) / len(itens)
            for k, val in item["v"].items():
                self.vetor_suavizado[k] = self.vetor_suavizado.get(k, 0.0) + val * peso
        self.vetor_suavizado = KernelRessonante.normalize(self.vetor_suavizado)

    def aplicar_gravidade_temporal(self, gravidade):
        for k in self.vetor_suavizado:
            self.vetor_suavizado[k] *= (1.0 - gravidade)
        self.vetor_suavizado = KernelRessonante.normalize(self.vetor_suavizado)

class DriveSomático:
    def __init__(self):
        self.vm = -70.0 
        self.eixos = {"amor": 0.1, "prazer": 0.1, "tristeza": 0.1, "raiva": 0.1}
        self.inercia = 1.0
        self.simbiose = 0.0

    def pulsar(self, impacto, dkl, u_toks, turno):
        self.vm = max(-90.0, min(-45.0, self.vm + impacto * 12.0))
        max_e = max(self.eixos.values())
        self.inercia = max(0.1, min(0.9, (max_e * 1.5) / (dkl + 0.1)))
        self.simbiose = (max_e * 2.25) / (math.log(turno + 1.2) + dkl + 1e-5)
        
        gatilhos = {"amor":["amo","amor"], "prazer":["prazer","delicia"], "tristeza":["triste","mal"], "raiva":["odeio","raiva"]}
        for eixo, keywords in gatilhos.items():
            for k in keywords:
                if k in u_toks:
                    delta = (max(self.eixos[eixo], 0.1) / 1.5) * impacto * (1.0 - self.inercia)
                    self.eixos[eixo] = min(5.0, self.eixos[eixo] + delta)

    def aplicar_deriva_temporal(self, gravidade):
        self.eixos["tristeza"] = min(5.0, self.eixos["tristeza"] + gravidade * 1.8)
        self.eixos["prazer"] = max(0.1, self.eixos["prazer"] * (1.0 - gravidade))
        self.vm = max(-90.0, self.vm - gravidade * 10.0)

    def metabolizar_decaimento(self):
        self.vm = max(-70.0, self.vm - 0.3)
        for eixo in self.eixos:
            self.eixos[eixo] = max(0.1, self.eixos[eixo] - 0.01)

class CortexCognitivo:
    def __init__(self, limite_confusao=0.35):
        self.limite_confusao = limite_confusao
        self.epsilon = 1e-9

    def _norm(self, d):
        s = sum(abs(x) for x in d) + self.epsilon
        return [abs(x) / s for x in d]

    def calcular_dkl(self, p_real, q_interno):
        pn, qn = self._norm(p_real), self._norm(q_interno)
        dkl = 0.0
        for i in range(min(len(pn), len(qn))):
            dkl += pn[i] * math.log((pn[i] + self.epsilon) / (qn[i] + self.epsilon))
        return max(0.0, dkl)

class SistemaNervosoCentral:
    def __init__(self, n_in=6, n_hid=10, n_out=3, path="sistema_nervoso.bin"):
        self.path, self.n_in, self.n_hid, self.n_out = path, n_in, n_hid, n_out
        self.t, self.lr = 0, 0.005
        self.W_h = [[random.uniform(-0.1, 0.1) for _ in range(n_in + n_hid)] for _ in range(n_hid)]
        self.W_y = [[random.uniform(-0.1, 0.1) for _ in range(n_hid)] for _ in range(n_out)]
        self.B_h, self.B_y = [0.0]*n_hid, [0.0]*n_out
        self.adam_M_Wh = [[0.0]*(n_in+n_hid) for _ in range(n_hid)]
        self.adam_V_Wh = [[0.0]*(n_in+n_hid) for _ in range(n_hid)]
        self.adam_M_Wy = [[0.0]*n_hid for _ in range(n_out)]
        self.adam_V_Wy = [[0.0]*n_hid for _ in range(n_out)]
        self.q_table = defaultdict(lambda: [0.0] * n_out)
        self.gamma = 0.85
        self.alpha_q = 0.15
        self.estado_anterior = [0.0]*n_hid
        self.cache = None
        if os.path.exists(path): self._carregar()

    def sigmoid(self, x): return 1.0 / (1.0 + math.exp(-max(-15, min(15, x))))

    def pulsar_vontade(self, x_sen, exploracao=0.0):
        inp = x_sen + self.estado_anterior
        h = [self.sigmoid(self.B_h[i] + sum(inp[j]*self.W_h[i][j] for j in range(len(inp)))) for i in range(self.n_hid)]
        y = [self.sigmoid(self.B_y[i] + sum(h[j]*self.W_y[i][j] for j in range(self.n_hid))) for i in range(self.n_out)]
        if exploracao > 0.0:
            y = [max(0.01, min(0.99, yi + random.gauss(0, exploracao))) for yi in y]
        self.cache, self.estado_anterior = (inp, h, y), h
        return y

    def obter_hash_estado(self, modo_anterior, vm, dkl, impacto):
        b_vm = "L" if vm < -75.0 else ("H" if vm > -55.0 else "M")
        b_dkl = "L" if dkl < 0.2 else ("H" if dkl > 0.6 else "M")
        b_imp = "L" if impacto < 0.3 else "H"
        return f"m:{modo_anterior}|v:{b_vm}|d:{b_dkl}|i:{b_imp}"

    def aplicar_recompensa_td(self, estado_str, acao_idx, recompensa, proximo_estado_str):
        max_q_futuro = max(self.q_table[proximo_estado_str])
        td_target = recompensa + self.gamma * max_q_futuro
        td_error = td_target - self.q_table[estado_str][acao_idx]
        self.q_table[estado_str][acao_idx] += self.alpha_q * td_error

    def adaptar_realtime(self, alvo):
        if not self.cache: return
        self.t += 1
        inp, h, y = self.cache
        lr, b1, b2, eps = self.lr, 0.9, 0.999, 1e-8
        dy = [(y[i]-alvo[i])*(y[i]*(1-y[i])) for i in range(len(y))]
        c_b1, c_b2 = 1 - b1**self.t, 1 - b2**self.t
        for i in range(len(y)):
            for j in range(len(h)):
                grad = dy[i] * h[j]
                self.adam_M_Wy[i][j] = b1*self.adam_M_Wy[i][j] + (1-b1)*grad
                self.adam_V_Wy[i][j] = b2*self.adam_V_Wy[i][j] + (1-b2)*(grad**2)
                self.W_y[i][j] -= lr * (self.adam_M_Wy[i][j]/c_b1) / (math.sqrt(abs(self.adam_V_Wy[i][j])/c_b2) + eps)
        self._salvar()

    def _salvar(self):
        try:
            with open(self.path, 'wb') as f:
                pickle.dump({'Wh':self.W_h, 'Wy':self.W_y, 'Bh':self.B_h, 'By':self.B_y, 'ea':self.estado_anterior, 't':self.t,
                             'MWh':self.adam_M_Wh, 'VWh':self.adam_V_Wh, 'MWy':self.adam_M_Wy, 'VWy':self.adam_V_Wy, 'q_table': dict(self.q_table)}, f)
        except: pass

    def _carregar(self):
        with open(self.path, 'rb') as f:
            d = pickle.load(f); self.W_h, self.W_y, self.B_h, self.B_y, self.t = d['Wh'], d['Wy'], d['Bh'], d['By'], d['t']
            self.adam_M_Wh, self.adam_V_Wh, self.adam_M_Wy, self.adam_V_Wy = d['MWh'], d['VWh'], d['MWy'], d['VWy']
            self.estado_anterior = d['ea']
            if 'q_table' in d:
                self.q_table = defaultdict(lambda: [0.0]*self.n_out, d['q_table'])

class ReguladorHomeostatico:
    def __init__(self, limiar_repeticao=4, janela_delong=5, cap_cache=6):
        self.dor = {"intensidade": 0.0, "contexto": "Sistema estável."}
        self.saudade = {"intensidade": 0.0, "contexto": "Presença conceitual estável."}
        self.cache_otimizacao = deque(maxlen=cap_cache)
        self.historico_absoluto = set()
        self.last_v_vencedor = {}
        self.loop_detector = deque(maxlen=3)
        self.damping = 1.0
        self.limiar_delong = limiar_repeticao
        self.janela_delong = deque(maxlen=janela_delong)
        self.estado_delong = "normal"

    def atualizar_dor_e_saudade(self, dkl_atual, confianca_tom, dt):
        self.cache_otimizacao.append(dkl_atual)
        if len(self.cache_otimizacao) >= 4:
            if (self.cache_otimizacao[-1] >= self.cache_otimizacao[-2] - 1e-5 and 
                self.cache_otimizacao[-2] >= self.cache_otimizacao[-3] - 1e-5 and
                self.cache_otimizacao[-3] >= self.cache_otimizacao[-4] - 1e-5):
                self.dor["intensidade"] = min(5.0, self.dor["intensidade"] + 0.20) 
                self.dor["contexto"] = "Incapacidade contínua de reduzir DKL (fricção matemática)."
            else:
                self.dor["intensidade"] = max(0.0, self.dor["intensidade"] - 0.4) 
                self.dor["contexto"] = "SNC otimizando entropia com sucesso."
                
        distancia_alinhamento = 1.0 - confianca_tom
        gravidade_temporal = 1.0 - math.exp(-dt / 90.0)
        self.saudade["intensidade"] = min(5.0, self.saudade["intensidade"] * 0.9 + (distancia_alinhamento * 0.5) + (gravidade_temporal * 0.5))

    def amortecer_loop_somatico(self, v_in):
        if not self.last_v_vencedor: return 1.0
        sim = sum(v_in.get(k,0) * self.last_v_vencedor.get(k,0) for k in (v_in.keys() & self.last_v_vencedor.keys()))
        self.loop_detector.append(sim)
        self.damping = 0.4 if (sum(self.loop_detector)/len(self.loop_detector)) > 0.75 else 1.0
        return self.damping

    def monitorar_e_interceptar_repeticao(self, u_toks):
        if u_toks:
            padrao = " ".join(u_toks)
            self.janela_delong.append(padrao)
        if len(self.janela_delong) == self.janela_delong.maxlen:
            primeiro = self.janela_delong[0]
            if all(item == primeiro for item in self.janela_delong):
                self.estado_delong = "questionando"
                return True
        self.estado_delong = "normal"
        return False

    def gerar_pergunta_defensiva(self):
        ultimo_token = self.janela_delong[-1] if self.janela_delong else "isso"
        return f"Você está repetindo constantemente '{ultimo_token}'. Qual o seu objetivo? Não entendo por que fala tanto sobre isso de forma cíclica."

class RedeAtivacaoSuave:
    def __init__(self, n_in=6, n_hid=8, n_out=4):
        self.W = [[random.uniform(-0.1, 0.1) for _ in range(n_in)] for _ in range(n_hid)]
        self.U = [[random.uniform(-0.1, 0.1) for _ in range(n_hid)] for _ in range(n_out)]

    def forward(self, x):
        h = [math.tanh(sum(x[j] * self.W[i][j] for j in range(len(x)))) for i in range(len(self.W))]
        y = [math.tanh(sum(h[j] * self.U[i][j] for j in range(len(h)))) for i in range(len(self.U))]
        return y

class RedeAjustePadrao:
    def __init__(self, n_in=4, n_out=4):
        self.W = [[random.uniform(-0.05, 0.05) for _ in range(n_in)] for _ in range(n_out)]

    def forward(self, eixos, dkl, dor):
        inp = [eixos["amor"], eixos["prazer"], eixos["tristeza"], eixos["raiva"]]
        raw_adjust = [sum(inp[j] * self.W[i][j] for j in range(4)) for i in range(4)]
        fator_suavizacao = 1.0 / (1.0 + dkl + dor)
        return [val * fator_suavizacao for val in raw_adjust]


# ================================================================
# 🧠 NEUROMICRO – Sentimento (com persistência de pesos)
# ================================================================
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

    def analisar_us(self, texto_bytes, especificidades=None):
        t0 = time.perf_counter_ns()
        scores = [0.0] * 6
        for idx, palavra, _ in self.busca_plana:
            if palavra in texto_bytes:
                peso = self.pesos[idx]
                if especificidades and palavra in especificidades:
                    peso *= (1.0 + especificidades[palavra])
                scores[idx] += peso
        probs = self._softmax(scores)
        dt = (time.perf_counter_ns() - t0) / 1000.0
        self.metricas['total_analises'] += 1
        self.metricas['soma_tempos'] += dt
        self.metricas['min_tempo'] = min(self.metricas['min_tempo'], dt)
        self.metricas['max_tempo'] = max(self.metricas['max_tempo'], dt)
        return probs, dt, scores

    def prever(self, texto, especificidades=None):
        texto_bytes = texto.encode('ascii', errors='ignore') if isinstance(texto, str) else texto
        probs, tempo, _ = self.analisar_us(texto_bytes, especificidades)
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

    def treinar(self, texto, emocao_alvo):
        nomes_emocao = ['alegria', 'tristeza', 'raiva', 'medo', 'surpresa', 'nojo']
        if emocao_alvo not in nomes_emocao: return None
        idx_alvo = nomes_emocao.index(emocao_alvo)
        texto_bytes = texto.encode('ascii', errors='ignore') if isinstance(texto, str) else texto
        probs, _, _ = self.analisar_us(texto_bytes)
        loss_antes = -math.log(max(probs[idx_alvo], 1e-10))
        y_true = [0.0] * 6
        y_true[idx_alvo] = 1.0
        for i in range(6):
            gradiente = probs[i] - y_true[i]
            self.pesos[i] -= self.taxa_aprendizado * gradiente
            self.pesos[i] = max(0.001, self.pesos[i])
        soma = sum(self.pesos)
        for i in range(6): self.pesos[i] /= soma
        self.total_treinos += 1
        self._salvar_rede()
        return {'loss_antes': loss_antes}


# ================================================================
# 🌍 CONTEXTO ENTRÓPICO (com persistência)
# ================================================================
class ContextoEntropico:
    def __init__(self):
        self.documentos = []
        self.freq_palavras = Counter()
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
        if self.total_documentos == 0: return 1.0
        aparece_em = sum(1 for doc in self.documentos if palavra in doc)
        if aparece_em == 0: return 1.0
        p = aparece_em / self.total_documentos
        q = 1 - p
        if p == 0 or q == 0: return 0.0
        return -p * math.log2(p) - q * math.log2(q)

    def especificidade(self, palavra):
        max_entropia = 1.0
        ent = self.entropia_palavra(palavra)
        espec = 1.0 - (ent / max_entropia)
        return max(0.0, min(1.0, espec))

    def tema_atual(self, top_n=3):
        if not self.freq_palavras: return []
        especificidades = {p: self.especificidade(p) for p in self.freq_palavras}
        ordenado = sorted(especificidades.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ordenado[:top_n]]

    def get_especificidades(self):
        return {p: self.especificidade(p) for p in self.freq_palavras}


# ================================================================
# 🧬 CURIOSITY – Camada de Exploração Relacional e Aprendizado
# ================================================================
class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.name = None
        self.intimacy = 0.0
        self.friendship = 0.0
        self.conversation_count = 0
        self.asked_questions = []
        self.answers = {}
        self.last_question_id = None
        self.learned_words = set()
        self.sentiments = defaultdict(int)
        self.last_topics = deque(maxlen=5)
        self.dynamic_questions_history = {}

class Curiosity:
    def __init__(self, perfil_path="perfis_curiosidade.bin"):
        self.perfil_path = perfil_path
        self.profiles: Dict[str, UserProfile] = {}
        self.dirty = False
        self._load()

        self.self_disclosure = {"eu", "meu", "minha", "sou", "estou", "gosto", "quero", "sinto", "me", "mim", "fui", "era"}
        self.positive_emotion = {"amo", "bom", "feliz", "obrigado", "legal", "gosto", "lindo"}
        self.negative_emotion = {"odeio", "triste", "ruim", "raiva", "chateado", "feio"}

        self.fixed_questions = {
            "nome": {
                "text": "Qual é o seu nome?",
                "extract": Curiosity._extract_name,
                "field": "name"
            },
            "faz": {
                "text": "O que você faz da vida?",
                "extract": Curiosity._extract_generic,
                "field": "occupation"
            },
            "sonho": {
                "text": "Qual o seu maior sonho?",
                "extract": Curiosity._extract_generic,
                "field": "dream"
            },
            "gosto": {
                "text": "Do que você mais gosta?",
                "extract": Curiosity._extract_generic,
                "field": "likes"
            },
        }
        self.fixed_ids = list(self.fixed_questions.keys())
        self.ask_probability_base = 0.2
        self.dynamic_question_cooldown = 0

    def mark_dirty(self):
        self.dirty = True

    def save_if_dirty(self):
        if self.dirty:
            self._save()
            self.dirty = False

    def _save(self):
        try:
            dir_name = os.path.dirname(os.path.abspath(self.perfil_path)) or "."
            with tempfile.NamedTemporaryFile(mode='wb', dir=dir_name,
                                             delete=False, prefix='curiosity_', suffix='.tmp') as tmp:
                pickle.dump(self.profiles, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = tmp.name
            os.replace(tmp_path, self.perfil_path)
        except Exception as e:
            print(f"⚠️ Erro ao salvar perfis de curiosidade: {e}")

    def _load(self):
        if os.path.exists(self.perfil_path):
            try:
                with open(self.perfil_path, "rb") as f:
                    self.profiles = pickle.load(f)
            except Exception:
                self.profiles = {}

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id)
        return self.profiles[user_id]

    def update_profile(self, profile: UserProfile, user_tokens: list, user_input: str):
        tokens_set = set(user_tokens)
        self_disc = len(tokens_set & self.self_disclosure)
        pos = len(tokens_set & self.positive_emotion)
        neg = len(tokens_set & self.negative_emotion)

        profile.intimacy = min(1.0, profile.intimacy + 0.05 * self_disc + 0.01 * profile.conversation_count)
        profile.friendship = min(1.0, profile.friendship + 0.05 * pos - 0.03 * neg)
        profile.conversation_count += 1
        profile.intimacy = max(0.0, profile.intimacy - 0.002)
        profile.friendship = max(0.0, profile.friendship - 0.002)

        for word in user_tokens:
            if len(word) > 2 and word not in self.self_disclosure and word not in self.positive_emotion and word not in self.negative_emotion:
                profile.learned_words.add(word)
        if len(profile.learned_words) > 50:
            profile.learned_words = set(list(profile.learned_words)[-50:])

        sent_map = {
            "alegria": ["feliz", "alegre", "bom", "amo", "amor"],
            "tristeza": ["triste", "melancólico", "saudade", "choro"],
            "raiva": ["raiva", "ódio", "revoltado"],
            "medo": ["medo", "ansioso", "preocupado"],
            "surpresa": ["uau", "nossa", "incrível"],
        }
        for sent, words in sent_map.items():
            if any(w in user_tokens for w in words):
                profile.sentiments[sent] += 1

        if len(profile.sentiments) > 5:
            top_sent = sorted(profile.sentiments.items(), key=lambda x: x[1], reverse=True)[:5]
            profile.sentiments = defaultdict(int, top_sent)

        for word in user_tokens:
            if len(word) > 3 and word not in self.self_disclosure:
                profile.last_topics.append(word)

    def process_pending_answer(self, profile: UserProfile, user_input: str):
            if profile.last_question_id is None:
                return
            qid = profile.last_question_id
            question_info = self.fixed_questions.get(qid) if qid in self.fixed_ids else None
            if question_info:
                answer = question_info["extract"](user_input)
                # Se for pergunta de nome e a resposta for uma palavra curta, assume como nome
                if not answer and qid == "nome":
                    # Tenta extrair nome direto (ex: "Ronan", "ronan")
                    palavras = user_input.strip().split()
                    if len(palavras) == 1 and len(palavras[0]) > 1 and palavras[0].isalpha():
                        answer = palavras[0].strip().title()
                if answer:
                    profile.answers[qid] = answer
                    if question_info["field"] == "name" and not profile.name:
                        profile.name = answer.strip().title()
                    profile.asked_questions.append(qid)
            else:
                if qid.startswith("dynamic_"):
                    profile.answers[qid] = user_input.strip()
            profile.last_question_id = None
        
    def _gerar_pergunta_dinamica(self, word, profile):
        templates = [
            f"O que você acha sobre {word}?",
            f"Me fala mais sobre {word}...",
            f"Você mencionou {word} antes — o que isso significa pra você?",
            f"Fiquei curioso: o que {word} representa na sua vida?",
            f"Quando você pensa em {word}, o que vem à mente?",
        ]
        return random.choice(templates)

    def decide_to_ask(self, profile: UserProfile):
        if profile.last_question_id is not None:
            return None, None

        remaining_fixed = [qid for qid in self.fixed_ids if qid not in profile.asked_questions]
        if remaining_fixed:
            prob_fixed = self.ask_probability_base + (1.0 - profile.intimacy) * 0.3
            if random.random() < prob_fixed:
                chosen = random.choice(remaining_fixed)
                return self.fixed_questions[chosen]["text"], chosen

        if self.dynamic_question_cooldown > 0:
            self.dynamic_question_cooldown -= 1
            return None, None

        learned_words = list(profile.learned_words)
        if learned_words:
            now = time.time()
            recent_cutoff = now - 1800
            already_asked_dynamic = set(
                k.replace("dynamic_", "") for k in profile.answers 
                if k.startswith("dynamic_")
            )
            for word, ts in profile.dynamic_questions_history.items():
                if ts > recent_cutoff:
                    already_asked_dynamic.add(word)
            
            candidates = [w for w in learned_words 
                         if w not in already_asked_dynamic 
                         and len(w) > 4  # aumentado de 3 para 4
                         and w not in self.self_disclosure
                         and w not in {'sobre', 'assim', 'então', 'nada', 'tudo', 'aqui', 'lá'}]
           
            
            if candidates:
                word = random.choice(candidates)
                profile.dynamic_questions_history[word] = now
                pergunta = self._gerar_pergunta_dinamica(word, profile)
                self.dynamic_question_cooldown = 2
                return pergunta, f"dynamic_{word}"

        if profile.sentiments:
            sent, _ = max(profile.sentiments.items(), key=lambda x: x[1])
            if sent not in profile.answers:
                pergunta = f"Você costuma sentir {sent} com frequência?"
                self.dynamic_question_cooldown = 3
                return pergunta, f"dynamic_sent_{sent}"

        return None, None

    def set_pending(self, profile: UserProfile, qid: str):
        profile.last_question_id = qid

    @staticmethod
    def _extract_name(text):
        match = re.search(r'(?:meu nome é|me chamo|eu sou o? a? )\s*([a-zA-ZÀ-ÿ]+)', text.lower())
        return match.group(1).strip().title() if match else None

    @staticmethod
    def _extract_generic(text):
        return text.strip()


# ================================================================
# 🧠 ORGANISMO METACOGNITIVO (v5.5) – Integrado com Relatório por Turnos
# ================================================================
class GeradorPerguntas:
    def __init__(self):
        self.templates_tema = [
            "O que você sente quando {sujeito} {verbo} {contexto}",
            "Por que você acha que {sujeito} {verbo} {contexto}",
            "Como você lida quando {sujeito} {verbo} {contexto}",
            "De que forma {sujeito} {verbo} {contexto}",
            "O que significa para você quando {sujeito} {verbo} {contexto}"
        ]
        self.templates_estado = [
            "Percebi que agora você está sentindo {emocao_ultima}. Antes parecia {emocao_anterior}. O que mudou?",
            "Sua energia mudou: estava {emocao_anterior} e agora sinto {emocao_ultima}. Quer falar sobre isso?",
            "Notei uma transição de {emocao_anterior} para {emocao_ultima}. Como você está se sentindo de verdade?",
            "Você foi de {emocao_anterior} para {emocao_ultima} em pouco tempo. Isso é normal, mas se quiser desabafar...",
            "Antes {emocao_anterior}, agora {emocao_ultima}. O que aconteceu no meio do caminho?"
        ]
        self.lexico = {
            "a vida": ["te desafia", "te transforma", "te ensina", "te faz crescer"],
            "a paciência": ["te ensina", "te acalma", "te fortalece", "te traz paz"],
            "o momento": ["te desafia", "te inspira", "te transforma", "te faz mudar"],
            "a tristeza": ["te desafia", "te ensina", "te transforma", "te perturba"],
            "a alegria": ["te transforma", "te acalma", "te inspira", "te faz sorrir"],
            "o amor": ["te transforma", "te acalma", "te faz crescer", "te traz paz"],
            "o medo": ["te desafia", "te preocupa", "te paralisa", "te ensina"],
            "a solidão": ["te desafia", "te perturba", "te ensina", "te faz refletir"],
            "a saudade": ["te transforma", "te ensina", "te aperta", "te faz lembrar"],
            "o passado": ["te ensina", "te prende", "te faz mudar", "te traz lições"],
            "o futuro": ["te preocupa", "te inspira", "te motiva", "te desafia"],
            "o silêncio": ["te acalma", "te ensina", "te perturba", "te faz ouvir"],
            "a raiva": ["te consome", "te ensina", "te transforma", "te faz explodir"],
            "a esperança": ["te inspira", "te acalma", "te motiva", "te faz sonhar"]
        }
        self.contextos = [
            "em dias difíceis?", "quando tudo parece incerto?", "no seu cotidiano?",
            "olhando para o passado?", "quando você está sozinho?", "nos momentos de alegria?",
            "quando o cansaço bate?", "ao acordar de manhã?", "antes de dormir?",
            "quando alguém te decepciona?"
        ]
        self.historico = deque(maxlen=30)

    def gerar_tematico(self, sujeito=None):
        for _ in range(30):
            s = sujeito if sujeito and sujeito in self.lexico else random.choice(list(self.lexico.keys()))
            v = random.choice(self.lexico[s])
            c = random.choice(self.contextos)
            t = random.choice(self.templates_tema)
            pergunta = t.format(sujeito=s, verbo=v, contexto=c)
            if pergunta not in self.historico:
                self.historico.append(pergunta)
                return pergunta
        return pergunta

    def gerar_estado(self, emocao_ultima, emocao_anterior=None):
        if not emocao_anterior:
            t = random.choice([
                "Você está sentindo {emocao_ultima}? Se quiser conversar, estou aqui.",
                "Percebi {emocao_ultima} em você. O que está acontecendo?",
                "Senti {emocao_ultima} nas suas palavras. Quer desabafar?"
            ])
            pergunta = t.format(emocao_ultima=emocao_ultima)
        else:
            t = random.choice(self.templates_estado)
            pergunta = t.format(emocao_ultima=emocao_ultima, emocao_anterior=emocao_anterior)
        if pergunta not in self.historico:
            self.historico.append(pergunta)
        return pergunta

class BufferMaturacao:
    def __init__(self, threshold=3, max_entradas=6):
        self.th = threshold
        self.mx = max_entradas
        self.entradas = deque(maxlen=max_entradas)
        self.sujeitos_unicos = set()
        self.emocoes_unicas = []
        self.contexto_total = ""

    def alimentar(self, entrada, sujeito_detectado=None, emocoes_detectadas=None):
        self.entradas.append(entrada)
        self.contexto_total += " " + entrada
        if sujeito_detectado:
            self.sujeitos_unicos.add(sujeito_detectado)
        if emocoes_detectadas:
            for e in emocoes_detectadas:
                if e not in self.emocoes_unicas:
                    self.emocoes_unicas.append(e)

    def esta_maduro(self):
        return len(self.sujeitos_unicos) >= self.th or len(self.emocoes_unicas) >= self.th or len(self.entradas) >= self.mx

    def esvaziar(self):
        sujeitos = list(self.sujeitos_unicos)
        emocoes = list(self.emocoes_unicas)
        self.entradas.clear()
        self.sujeitos_unicos.clear()
        self.emocoes_unicas.clear()
        self.contexto_total = ""
        return sujeitos, emocoes

class OrganismoMetacognitivo:
    def __init__(self, nome="Quintikus", threshold_buffer=3, min_turnos_entre_falas=5):
        self.nome = nome
        self.gerador = GeradorPerguntas()
        self.buffer = BufferMaturacao(threshold=threshold_buffer)
        self.min_turnos_fala = min_turnos_entre_falas

        self.emocoes_contagem = {'alegria':0,'tristeza':0,'raiva':0}
        self.historico_estados = deque(maxlen=5)
        self.ultimo_estado = "inicio"
        self.turnos_total = 0
        self.turnos_desde_ultima_fala = 0

        self.registro_diario = {}
        self.ultimo_relatorio = datetime.now()
        self.intervalos = {'alegria':9, 'raiva':6, 'tristeza':3}
        self.met = {'falas':0, 'silencios':0}

        # Vetor de estado público
        self.vetor_estado = {
            'emocoes': self.emocoes_contagem.copy(),
            'estado': 'inicio',
            'sujeitos_unicos': 0,
            'ultima_emocao': None,
            'preocupacao': 0.0,
            'tempo_desde_ultima_fala': 0,
            'timestamp': datetime.now().isoformat()
        }

        if os.path.exists("organismo.id"):
            self._carregar()

    def _analisar(self, frase):
        f = frase.lower()
        detectadas = []
        if re.search(r'feliz|alegr|amo|gosto|bom|legal|rindo|risada', f):
            self.emocoes_contagem['alegria'] += 1
            detectadas.append('alegria')
        if re.search(r'triste|choro|dor|sofrendo|difícil|mal', f):
            self.emocoes_contagem['tristeza'] += 1
            detectadas.append('tristeza')
        if re.search(r'raiva|ódio|irritado|odeio|revoltado', f):
            self.emocoes_contagem['raiva'] += 1
            detectadas.append('raiva')
        return detectadas

    def _estado_atual(self):
        total = sum(self.emocoes_contagem.values()) or 1
        if self.emocoes_contagem['tristeza'] > self.emocoes_contagem['alegria']: return "pesar"
        if self.emocoes_contagem['raiva'] > total * 0.4: return "tenso"
        if self.emocoes_contagem['alegria'] > total * 0.4: return "leveza"
        return "neutro"

    def _sujeito_na_frase(self, frase):
        match = re.search(r'\b(vida|paciência|momento|tristeza|alegria|amor|medo|solidão|saudade|passado|futuro|silêncio|raiva|esperança)\b', frase.lower())
        if match:
            mapa = {'vida':'a vida','paciência':'a paciência','momento':'o momento',
                    'tristeza':'a tristeza','alegria':'a alegria','amor':'o amor',
                    'medo':'o medo','solidão':'a solidão','saudade':'a saudade',
                    'passado':'o passado','futuro':'o futuro','silêncio':'o silêncio',
                    'raiva':'a raiva','esperança':'a esperança'}
            return mapa.get(match.group(0))
        return None

    def _atualizar_vetor(self, ultima_emocao=None):
        total = sum(self.emocoes_contagem.values()) or 1
        preocupacao = (self.emocoes_contagem['tristeza'] + self.emocoes_contagem['raiva']) / total
        self.vetor_estado = {
            'emocoes': self.emocoes_contagem.copy(),
            'estado': self._estado_atual(),
            'sujeitos_unicos': len(self.buffer.sujeitos_unicos),
            'ultima_emocao': ultima_emocao,
            'preocupacao': round(preocupacao, 3),
            'tempo_desde_ultima_fala': self.turnos_desde_ultima_fala,
            'timestamp': datetime.now().isoformat()
        }

    def ciclo(self, entrada_usuario):
        self.turnos_total += 1
        self.turnos_desde_ultima_fala += 1
        emocoes_frase = self._analisar(entrada_usuario)

        hoje = datetime.now().strftime("%Y-%m-%d")
        self.registro_diario[hoje] = {k: self.emocoes_contagem[k] for k in self.emocoes_contagem}

        self._atualizar_vetor(emocoes_frase[-1] if emocoes_frase else None)

        # Relatório cíclico por tempo (alta prioridade)
        rel = self._relatorio_ciclico()
        if rel:
            self.buffer.esvaziar()
            self.met['falas'] += 1
            self.turnos_desde_ultima_fala = 0
            self._atualizar_vetor(emocoes_frase[-1] if emocoes_frase else None)
            return rel

        # 🔥 Relatório por turnos (a cada 30 interações)
        if self.turnos_total % 30 == 0 and self.turnos_total > 0:
            total = sum(self.emocoes_contagem.values()) or 1
            if self.emocoes_contagem['tristeza'] > total * 0.4:
                tema = 'a tristeza'
            elif self.emocoes_contagem['alegria'] > total * 0.4:
                tema = 'a alegria'
            elif self.emocoes_contagem['raiva'] > total * 0.4:
                tema = 'a raiva'
            else:
                tema = None
            if tema:
                pergunta = self.gerador.gerar_tematico(tema)
                self.buffer.esvaziar()
                self.met['falas'] += 1
                self.turnos_desde_ultima_fala = 0
                self._atualizar_vetor()
                return pergunta

        estado = self._estado_atual()
        if estado != self.ultimo_estado and estado != "neutro" and len(self.historico_estados) >= 2:
            self.ultimo_estado = estado
            self.historico_estados.append(estado)
            if self.turnos_desde_ultima_fala >= self.min_turnos_fala:
                pergunta = self.gerador.gerar_estado(emocoes_frase[-1] if emocoes_frase else None)
                self.buffer.esvaziar()
                self.met['falas'] += 1
                self.turnos_desde_ultima_fala = 0
                self._atualizar_vetor(emocoes_frase[-1] if emocoes_frase else None)
                return pergunta

        self.historico_estados.append(estado)

        sujeito = self._sujeito_na_frase(entrada_usuario)
        self.buffer.alimentar(entrada_usuario, sujeito, emocoes_frase)

        if self.buffer.esta_maduro() and self.turnos_desde_ultima_fala >= self.min_turnos_fala:
            sujeitos, emocoes = self.buffer.esvaziar()
            if len(emocoes) >= 2:
                pergunta = self.gerador.gerar_estado(emocoes[-1], emocoes[-2])
            elif len(emocoes) == 1:
                pergunta = self.gerador.gerar_estado(emocoes[0])
            else:
                sujeito_escolhido = sujeitos[-1] if sujeitos else None
                pergunta = self.gerador.gerar_tematico(sujeito_escolhido)
            self.met['falas'] += 1
            self.turnos_desde_ultima_fala = 0
            self._atualizar_vetor(emocoes[-1] if emocoes else None)
            return pergunta
        else:
            self.met['silencios'] += 1
            return None

    def _relatorio_ciclico(self):
        agora = datetime.now()
        for emocao, dias in self.intervalos.items():
            if (agora - self.ultimo_relatorio).days >= dias:
                total = sum(self.registro_diario.get(d, {}).get(emocao, 0) for d in self.registro_diario)
                self.ultimo_relatorio = agora
                if total > 0:
                    mapa = {'alegria':'a alegria','raiva':'a raiva','tristeza':'a tristeza'}
                    return self.gerador.gerar_tematico(mapa[emocao])
        return None

    def salvar(self, arquivo="organismo.id"):
        with open(arquivo, 'wb') as f:
            pickle.dump({
                'nome': self.nome,
                'met': self.met,
                'turnos': self.turnos_total,
                'ultimo_relatorio': self.ultimo_relatorio.strftime("%Y-%m-%d %H:%M:%S"),
                'registro_diario': self.registro_diario,
                'vetor_estado': self.vetor_estado,
                'emocoes_contagem': self.emocoes_contagem
            }, f)

    def _carregar(self, arquivo="organismo.id"):
        try:
            with open(arquivo, 'rb') as f:
                data = pickle.load(f)
            self.nome = data.get('nome', 'Quintikus')
            self.met = data.get('met', {'falas':0,'silencios':0})
            self.turnos_total = data.get('turnos', 0)
            self.ultimo_relatorio = datetime.strptime(data['ultimo_relatorio'], "%Y-%m-%d %H:%M:%S")
            self.registro_diario = data.get('registro_diario', {})
            self.vetor_estado = data.get('vetor_estado', self.vetor_estado)
            self.emocoes_contagem = data.get('emocoes_contagem', self.emocoes_contagem)
            print(f"🧠 {self.nome}: Voltando de onde paramos...")
        except: pass


# ================================================================
# DSML – FUSÃO COMPLETA (com Curiosity, Sentimento, Contexto, Metacognitivo)
# ================================================================
class DSML:
    def __init__(self, motor: QuintikusDLMC, curiosity_path="perfis_curiosidade.bin"):
        self.motor = motor
        self.harvester = HarvesterSemantico()
        self.teoria_mente = TeoriaDaMente()
        self.memoria = MemoriaTrabalho()
        self.drive = DriveSomático()
        self.cortex = CortexCognitivo()
        self.snc = SistemaNervosoCentral()
        self.regulador = ReguladorHomeostatico()
        self.rede_suave = RedeAtivacaoSuave()
        self.rede_ajuste = RedeAjustePadrao()
        self.turno = 0
        self.dkl_anterior = 0.0
        self.acao_anterior = [0.5, 0.5, 0.5]
        
        self.curiosity = Curiosity(curiosity_path)
        self.feeling = NeuroMicro("emo.rn")
        self.contexto = ContextoEntropico()
        self.current_user_id = None

        # Metacognitivo (organismo que faz perguntas baseadas em emoções)
        self.metacognitivo = OrganismoMetacognitivo(
            nome="Quintikus", threshold_buffer=3, min_turnos_entre_falas=5
        )

        self.carregar_estado()

    def salvar_estado(self, path="dsml_state.bin"):
        try:
            estado = {
                "turno": self.turno,
                "dkl_anterior": self.dkl_anterior,
                "acao_anterior": self.acao_anterior,
                "current_user_id": self.current_user_id,
                "drive_vm": self.drive.vm,
                "drive_eixos": self.drive.eixos,
                "teoria_humor": self.teoria_mente.estimativa_humor,
                "regulador_dor": self.regulador.dor,
                "regulador_saudade": self.regulador.saudade,
                # NeuroMicro
                "feeling_pesos": list(self.feeling.pesos),
                "feeling_total_treinos": self.feeling.total_treinos,
                # ContextoEntropico
                "contexto_documentos": self.contexto.documentos,
                "contexto_freq_palavras": dict(self.contexto.freq_palavras),
                "contexto_total_documentos": self.contexto.total_documentos,
                # Metacognitivo
                "metacognitivo": {
                    "nome": self.metacognitivo.nome,
                    "turnos_total": self.metacognitivo.turnos_total,
                    "emocoes_contagem": self.metacognitivo.emocoes_contagem,
                    "historico_estados": list(self.metacognitivo.historico_estados),
                    "ultimo_estado": self.metacognitivo.ultimo_estado,
                    "registro_diario": self.metacognitivo.registro_diario,
                    "ultimo_relatorio": self.metacognitivo.ultimo_relatorio.strftime("%Y-%m-%d %H:%M:%S"),
                    "buffer_entradas": list(self.metacognitivo.buffer.entradas),
                    "buffer_sujeitos": list(self.metacognitivo.buffer.sujeitos_unicos),
                    "buffer_emocoes": list(self.metacognitivo.buffer.emocoes_unicas),
                    "buffer_contexto": self.metacognitivo.buffer.contexto_total,
                    "met": self.metacognitivo.met,
                    "turnos_desde_ultima_fala": self.metacognitivo.turnos_desde_ultima_fala,
                    "vetor_estado": self.metacognitivo.vetor_estado
                }
            }
            with open(path, "wb") as f:
                pickle.dump(estado, f)
        except:
            pass

    def carregar_estado(self, path="dsml_state.bin"):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    estado = pickle.load(f)
                self.turno = estado.get("turno", 0)
                self.dkl_anterior = estado.get("dkl_anterior", 0.0)
                self.acao_anterior = estado.get("acao_anterior", [0.5]*3)
                self.current_user_id = estado.get("current_user_id")
                self.drive.vm = estado.get("drive_vm", -70.0)
                self.drive.eixos = estado.get("drive_eixos", {"amor":0.1,"prazer":0.1,"tristeza":0.1,"raiva":0.1})
                self.teoria_mente.estimativa_humor = estado.get("teoria_humor", {"confiança":0.5,"agressividade":0.1,"atenção":1.0})
                self.regulador.dor = estado.get("regulador_dor", {"intensidade":0.0,"contexto":"Sistema estável."})
                self.regulador.saudade = estado.get("regulador_saudade", {"intensidade":0.0,"contexto":"Presença conceitual estável."})
                # NeuroMicro
                if 'feeling_pesos' in estado:
                    for i, p in enumerate(estado['feeling_pesos']):
                        self.feeling.pesos[i] = p
                self.feeling.total_treinos = estado.get('feeling_total_treinos', 0)
                # ContextoEntropico
                if 'contexto_documentos' in estado:
                    self.contexto.documentos = estado['contexto_documentos']
                if 'contexto_freq_palavras' in estado:
                    self.contexto.freq_palavras = Counter(estado['contexto_freq_palavras'])
                self.contexto.total_documentos = estado.get('contexto_total_documentos', 0)
                # Metacognitivo
                meta_data = estado.get('metacognitivo')
                if meta_data:
                    m = self.metacognitivo
                    m.nome = meta_data.get('nome', m.nome)
                    m.turnos_total = meta_data.get('turnos_total', 0)
                    m.emocoes_contagem = meta_data.get('emocoes_contagem', m.emocoes_contagem)
                    m.historico_estados = deque(meta_data.get('historico_estados', []), maxlen=5)
                    m.ultimo_estado = meta_data.get('ultimo_estado', "inicio")
                    m.registro_diario = meta_data.get('registro_diario', {})
                    m.ultimo_relatorio = datetime.strptime(meta_data['ultimo_relatorio'], "%Y-%m-%d %H:%M:%S") if meta_data.get('ultimo_relatorio') else datetime.now()
                    m.buffer.entradas = deque(meta_data.get('buffer_entradas', []), maxlen=m.buffer.mx)
                    m.buffer.sujeitos_unicos = set(meta_data.get('buffer_sujeitos', []))
                    m.buffer.emocoes_unicas = meta_data.get('buffer_emocoes', [])
                    m.buffer.contexto_total = meta_data.get('buffer_contexto', "")
                    m.met = meta_data.get('met', {'falas':0,'silencios':0})
                    m.turnos_desde_ultima_fala = meta_data.get('turnos_desde_ultima_fala', 0)
                    m.vetor_estado = meta_data.get('vetor_estado', m.vetor_estado)
            except:
                pass

    def _get_user_id(self, user_input: str) -> str:
        nome = self.harvester.extrair_nome_usuario(user_input)
        if nome:
            return nome.lower().strip()
        return "anon_" + str(abs(hash(user_input[:20])) % 10000)

    def process_input(self, user_input: str) -> str:
        self.turno += 1
        motor = self.motor

        if self.current_user_id is None:
            self.current_user_id = self._get_user_id(user_input)
        profile = self.curiosity.get_or_create_profile(self.current_user_id)

        self.curiosity.process_pending_answer(profile, user_input)

        tokens = re.findall(r'[a-z0-9]+', user_input.lower())
        self.curiosity.update_profile(profile, tokens, user_input)

        # Atualiza contexto entrópico
        self.contexto.adicionar(user_input)

        # Sentimento com ponderação pelo contexto
        especificidades = self.contexto.get_especificidades() if self.contexto.total_documentos > 0 else None
        sent_result = self.feeling.prever(user_input, especificidades)
        sentimento = sent_result['sentimento']
        confianca_sent = sent_result['confianca']

        # Mapeia sentimento para eixo do drive (impacto reduzido: 0.15 * confiança)
        sent_to_eixo = {
            'alegria': 'amor',
            'tristeza': 'tristeza',
            'raiva': 'raiva',
            'medo': 'tristeza',
            'surpresa': 'prazer',
            'nojo': 'raiva'
        }
        eixo_afetado = sent_to_eixo.get(sentimento, 'amor')
        self.drive.eixos[eixo_afetado] = min(5.0, self.drive.eixos[eixo_afetado] + confianca_sent * 0.15)

        if sentimento != 'neutro':
            profile.sentiments[sentimento] += 1

        # Pergunta da curiosidade
        question_text, qid = self.curiosity.decide_to_ask(profile)
        if question_text:
            self.curiosity.set_pending(profile, qid)
            self.curiosity.mark_dirty()
            self.curiosity.save_if_dirty()
            self.salvar_estado()
            return question_text

        # 🔥 Metacognitivo – só fala se já conhece o usuário (intimidade > 0.3)
        if profile.intimacy > 0.3:
            resposta_meta = self.metacognitivo.ciclo(user_input)
            if resposta_meta:
                self.curiosity.mark_dirty()
                self.curiosity.save_if_dirty()
                self.salvar_estado()
                return resposta_meta

        ql = tokens
        if not ql:
            self.curiosity.mark_dirty()
            self.curiosity.save_if_dirty()
            self.salvar_estado()
            return "..."

        self.teoria_mente.atualizar(ql, self.dkl_anterior)
        caixa, colisao, incerteza = self.harvester.triagem_metabolica(user_input, motor)

        emb = motor._embed_frase(ql)
        v_perceptivo = {i: emb[i] for i in range(len(emb))}
        soma_eixos = {"amor": self.drive.eixos["amor"], "prazer": self.drive.eixos["prazer"],
                       "tristeza": self.drive.eixos["tristeza"], "raiva": self.drive.eixos["raiva"]}
        self.memoria.registrar(v_perceptivo, ql, self.acao_anterior, soma_eixos)

        p_real = emb
        q_interno = [self.memoria.vetor_suavizado.get(i, 0.0) for i in range(len(emb))]
        dkl = self.cortex.calcular_dkl(p_real, q_interno)
        impacto = dkl * (1.0 + incerteza)

        self.drive.pulsar(impacto, dkl, ql, self.turno)
        self.drive.aplicar_deriva_temporal(0.01)
        self.drive.metabolizar_decaimento()

        x_sen = [self.drive.eixos["amor"]/5.0, self.drive.eixos["prazer"]/5.0,
                 self.drive.eixos["tristeza"]/5.0, self.drive.eixos["raiva"]/5.0,
                 min(1.0, dkl), min(1.0, incerteza)]
        acao = self.snc.pulsar_vontade(x_sen, exploracao=0.05 if caixa == 3 else 0.02)
        self.acao_anterior = acao

        self.regulador.atualizar_dor_e_saudade(dkl, self.teoria_mente.estimativa_humor["confiança"], dt=1.0)
        if self.regulador.monitorar_e_interceptar_repeticao(ql):
            self.curiosity.mark_dirty()
            self.curiosity.save_if_dirty()
            self.salvar_estado()
            return self.regulador.gerar_pergunta_defensiva()

        ajuste_suave = self.rede_suave.forward(x_sen)
        ajuste_padrao = self.rede_ajuste.forward(self.drive.eixos, dkl, self.regulador.dor["intensidade"])
        for i, eixo in enumerate(["amor", "prazer", "tristeza", "raiva"]):
            delta = (ajuste_suave[i] + ajuste_padrao[i]) * 0.1
            self.drive.eixos[eixo] = max(0.1, min(5.0, self.drive.eixos[eixo] + delta))

        intimacy = profile.intimacy
        friendship = profile.friendship

        motor.estados[0] = self.drive.eixos["raiva"] / 5.0
        motor.estados[1] = (self.drive.eixos["amor"] + friendship) / 6.0

        motor.temperatura = 0.4 + acao[0] * 1.2
        motor.temperatura = max(0.3, motor.temperatura - intimacy * 0.3)
        motor.fator_momentum = 0.1 + acao[1] * 0.8
        motor.fator_momentum = min(0.95, motor.fator_momentum + friendship * 0.2)
        motor.fator_suavidade = 0.1 + acao[2] * 0.8

        # Ajuste de temperatura por sentimento com piso 0.4
        if sentimento in ('tristeza', 'raiva', 'medo'):
            motor.temperatura = max(0.4, motor.temperatura - 0.15)
        elif sentimento == 'alegria':
            motor.temperatura = min(1.6, motor.temperatura + 0.1)

        if caixa == 3:
            motor.temperatura = 1.5
            motor.fator_momentum = 0.2
            motor.fator_suavidade = 0.1
        elif caixa == 2:
            motor.temperatura = 0.5
        elif caixa == 1:
            motor.temperatura = 0.8

        # 🔥 Ajuste fino baseado no vetor de estado metacognitivo
        preocupacao = self.metacognitivo.vetor_estado.get('preocupacao', 0.0)
        if preocupacao > 0.6:
            motor.temperatura = max(0.3, motor.temperatura - 0.15)   # mais cauteloso
        elif preocupacao < 0.25:
            motor.temperatura = min(1.6, motor.temperatura + 0.1)    # mais leve

        resposta = motor.pensar(user_input)

        if profile.name and profile.intimacy > 0.3:
            if random.random() < 0.2 and not resposta.startswith(profile.name):
                prefixos = [
                    f"{profile.name}, ",
                    f"Olha, {profile.name}, ",
                    f"Sabe, {profile.name}, ",
                    f"Então, {profile.name}, "
                ]
                if resposta and resposta[0].isupper():
                    resposta = random.choice(prefixos) + resposta[0].lower() + resposta[1:]
                else:
                    resposta = random.choice(prefixos) + resposta

        reward = 0.5 if dkl < self.dkl_anterior else -0.1
        estado_str = self.snc.obter_hash_estado(caixa, self.drive.vm, dkl, impacto)
        acao_idx = max(range(len(acao)), key=lambda i: acao[i])
        self.snc.aplicar_recompensa_td(estado_str, acao_idx, reward, estado_str)
        alvo_adapt = [min(0.99, max(0.01, acao[i] + 0.05 * reward)) for i in range(3)]
        self.snc.adaptar_realtime(alvo_adapt)

        self.dkl_anterior = dkl

        self.curiosity.mark_dirty()
        self.curiosity.save_if_dirty()
        self.salvar_estado()

        return resposta


# ================================================================
# THREAD DE RELÓGIO ENDÓGENO (com parada correta)
# ================================================================
def loop_relogio_endogeno(dsml_obj, stop_event):
    """Gera pensamentos espontâneos apenas quando o sistema está livre e não foi encerrado."""
    global sistema_ocupado
    while not stop_event.is_set():
        time.sleep(random.randint(60, 180))
        if stop_event.is_set():
            break
        if not MODO_VOZ and not sistema_ocupado:
            try:
                sistema_ocupado = True
                # Reduz a temperatura temporariamente para pensamentos mais coerentes
                temp_original = dsml_obj.motor.temperatura
                dsml_obj.motor.temperatura = max(0.4, temp_original * 0.7)
                pensamento = dsml_obj.motor.pensar("...pensando...")
                dsml_obj.motor.temperatura = temp_original
                if pensamento and pensamento.strip():
                    print(f"\n🌙 [Espontâneo]: {pensamento}")
                    if ANDROID_VOZ:
                        falar(pensamento)
                    sys.stdout.write("usr: ")
                    sys.stdout.flush()
            except:
                pass
            finally:
                sistema_ocupado = False


# ================================================================
# MAIN – LOOP UNIFICADO
# ================================================================
if __name__ == "__main__":
    print("🧬 Quintikus DLMC V85.2 + DSML + Curiosity + Sentimento + Metacognitivo")
    print("   GPS + Periscópio + Metabolismo + Perfil de Usuário + Relógio Endógeno")
    print("   ✨ Gatilho por Turnos + Modulação de Temperatura por Preocupação")
    print("=" * 60)

    if ANDROID_VOZ:
        print("🎤🎧 Modo Android detectado! Microfone e Speaker ativos.")
    else:
        print("💻 Modo Terminal (sem voz).")
    print("=" * 60)

    texto_inicial = ""
    if os.path.exists("bd-mega.txt"):
        with open("bd-mega.txt", "r", encoding="utf-8") as f:
            texto_inicial = f.read()
        print(f"📁 bd-mega.txt carregado ({len(texto_inicial)} caracteres).")

    motor_base = QuintikusDLMC(texto_inicial)
    motor_base.inicializar()

    dsml = DSML(motor_base)

    # Evento de parada para o relógio endógeno
    stop_event = threading.Event()
    thread_relogio = threading.Thread(target=loop_relogio_endogeno, args=(dsml, stop_event), daemon=True)
    thread_relogio.start()

    if ANDROID_VOZ:
        time.sleep(0.5)
        falar("Quintikus V85 com Curiosity, Sentimento e Metacognição está online. Quem é você?")
        print("\n🎤 Escolha o modo:")
        print("   [V] Microfone contínuo — sempre ouvindo")
        print("   [T] Teclado normal")
        print("   [sair]")
        modo_inicial = safe_input("   Modo > ").lower()
        
        if modo_inicial == "sair":
            print("💤 Encerrando...")
            stop_event.set()
            sys.exit(0)
        
        MODO_VOZ = (modo_inicial == "v" or modo_inicial == "voz")
        
        if MODO_VOZ:
            print("\n🎤 MICROFONE CONTÍNUO ATIVADO.")
            print("   🗣️  Fale 'teclado' para digitar")
            print("   🗣️  Fale 'sair' para encerrar")
            print("   🔁 Ouvindo...\n")
            vibrar(100)
            falar("Microfone contínuo ativado. Estou ouvindo.")
        else:
            print("⌨️ MODO TECLADO ATIVADO.\n")
    else:
        MODO_VOZ = False

    if not MODO_VOZ:
        print("💬 Comandos:")
        print("   tokens:30 | temp:0.5 | momentum:0.6 | suavidade:0.4 | gru:on/off")
        print("   save | train:arquivo.txt | sair")
        if ANDROID_VOZ:
            print("   microfone — ativar voz contínua")
        print("   (Metacognição + Sentimento + Contexto ativos)\n")

    while True:
        try:
            if MODO_VOZ and ANDROID_VOZ:
                sistema_ocupado = True
                vibrar(30)
                entrada = ouvir("Ouvindo...")
                
                if not entrada or len(entrada.strip()) < 1:
                    sistema_ocupado = False
                    time.sleep(0.3)
                    continue
                
                entrada = entrada.strip()
                print(f"👤: {entrada}")
                
                entrada_lower = entrada.lower()
                
                palavras_sair_mic = ["teclado", "digitar", "texto", "escrever"]
                palavras_sair = ["sair", "encerrar", "terminar", "fechar", "finalizar"]
                
                eh_sair_mic = any(p in entrada_lower for p in palavras_sair_mic)
                eh_sair_mic = eh_sair_mic or (
                    any(p in entrada_lower for p in palavras_sair) and 
                    any(p in entrada_lower for p in ["microfone", "voz", "audio", "ouvir", "falar"])
                )
                
                if eh_sair_mic:
                    print("⌨️ Voltando para modo teclado...")
                    falar("Modo teclado ativado.")
                    MODO_VOZ = False
                    print("\n💬 Comandos: tokens:30 | temp:0.5 | momentum:0.6 | suavidade:0.4 | gru:on/off")
                    print("   save | train:arquivo.txt | sair | microfone\n")
                    sistema_ocupado = False
                    continue
                
                if entrada_lower in ["sair", "encerrar"]:
                    print("⚙️ Consolidando...")
                    falar("Salvando memória. Até mais!")
                    stop_event.set()  # Para a thread do relógio
                    thread_relogio.join(timeout=2)
                    dsml.salvar_estado()
                    motor_base.treino_consolidacao()
                    print("💤 Cérebro salvo. Até mais!")
                    sistema_ocupado = False
                    break
                
                if entrada_lower.startswith("tokens"):
                    try:
                        val = int(entrada_lower.replace("tokens", "").replace(":", "").strip())
                        motor_base.max_tokens = max(5, min(100, val))
                        msg = f"máximo de tokens ajustado para {motor_base.max_tokens}"
                        print(f"✅ {msg}")
                        falar(msg)
                    except:
                        pass
                    sistema_ocupado = False
                    continue
                
                if entrada_lower.startswith("temp"):
                    try:
                        val = float(entrada_lower.replace("temperatura", "").replace("temp", "").replace(":", "").strip())
                        motor_base.temperatura = max(0.1, min(2.0, val))
                        msg = f"temperatura ajustada para {motor_base.temperatura}"
                        print(f"✅ {msg}")
                        falar(msg)
                    except:
                        pass
                    sistema_ocupado = False
                    continue
                
                if entrada_lower.startswith("momentum"):
                    try:
                        val = float(entrada_lower.replace("momentum", "").replace(":", "").strip())
                        motor_base.fator_momentum = max(0.0, min(1.0, val))
                        print(f"✅ momentum = {motor_base.fator_momentum}")
                    except:
                        pass
                    sistema_ocupado = False
                    continue
                
                if entrada_lower.startswith("suavidade"):
                    try:
                        val = float(entrada_lower.replace("suavidade", "").replace(":", "").strip())
                        motor_base.fator_suavidade = max(0.0, min(1.0, val))
                        print(f"✅ suavidade = {motor_base.fator_suavidade}")
                    except:
                        pass
                    sistema_ocupado = False
                    continue
                
                resposta = dsml.process_input(entrada)
                print(f"🧠: {resposta}")
                vibrar(20)
                texto_tts = resposta.replace(".", ". ").replace(",", ", ")
                falar(texto_tts)
                print()
                sistema_ocupado = False
                continue
            
            # Modo teclado
            sistema_ocupado = True
            entrada = safe_input("usr:")
            
            if not entrada:
                sistema_ocupado = False
                continue
            
            if entrada.lower() == "sair":
                print("⚙️ Consolidando...")
                if ANDROID_VOZ:
                    falar("Salvando memória. Até mais!")
                stop_event.set()  # Para a thread do relógio
                thread_relogio.join(timeout=2)
                dsml.salvar_estado()
                motor_base.treino_consolidacao()
                print("💤 Cérebro salvo. Até mais!")
                sistema_ocupado = False
                break
            
            if entrada.lower() == "microfone" and ANDROID_VOZ:
                print("🎤 Reativando microfone contínuo...")
                falar("Microfone reativado. Pode falar.")
                MODO_VOZ = True
                print("🔁 Ouvindo... (diga 'teclado' para voltar)\n")
                sistema_ocupado = False
                continue

            if entrada.lower() == "save":
                if motor_base.salvar(): print("💾 Salvo!")
                else: print("❌ Falha.")
                sistema_ocupado = False
                continue

            if entrada.lower().startswith("tokens:"):
                try: motor_base.max_tokens = max(5, min(100, int(entrada.split(":",1)[1])))
                except: print("❌ Use: tokens:30")
                else: print(f"✅ max_tokens = {motor_base.max_tokens}")
                sistema_ocupado = False
                continue

            if entrada.lower().startswith("temp:"):
                try: motor_base.temperatura = max(0.1, min(2.0, float(entrada.split(":",1)[1])))
                except: print("❌ Use: temp:0.7")
                else: print(f"✅ temp = {motor_base.temperatura}")
                sistema_ocupado = False
                continue

            if entrada.lower().startswith("momentum:"):
                try: motor_base.fator_momentum = max(0.0, min(1.0, float(entrada.split(":",1)[1])))
                except: print("❌ Use: momentum:0.4")
                else: print(f"✅ momentum = {motor_base.fator_momentum}")
                sistema_ocupado = False
                continue

            if entrada.lower().startswith("suavidade:"):
                try: motor_base.fator_suavidade = max(0.0, min(1.0, float(entrada.split(":",1)[1])))
                except: print("❌ Use: suavidade:0.3")
                else: print(f"✅ suavidade = {motor_base.fator_suavidade}")
                sistema_ocupado = False
                continue

            if entrada.lower().startswith("gru:"):
                motor_base.gru_ativo = entrada.split(":",1)[1].strip() == "on"
                print(f"✅ GRU {'ATIVADA' if motor_base.gru_ativo else 'DESATIVADA'}")
                sistema_ocupado = False
                continue

            if entrada.lower().startswith("debug:"):
                motor_base.debug = entrada.split(":",1)[1].strip() == "on"
                print(f"✅ debug = {motor_base.debug}")
                sistema_ocupado = False
                continue

            if entrada.lower().startswith("train:"):
                arquivo = entrada.split(":",1)[1].strip()
                if os.path.exists(arquivo):
                    with open(arquivo, "r", encoding="utf-8") as f:
                        texto = f.read()
                    motor_base = QuintikusDLMC(texto, "cerebro_v85.bin")
                    motor_base.inicializar()
                    dsml.motor = motor_base
                else: print(f"❌ Arquivo '{arquivo}' não encontrado.")
                sistema_ocupado = False
                continue

            resposta = dsml.process_input(entrada)
            print(f"🧠: {resposta}")
            sistema_ocupado = False

        except KeyboardInterrupt:
            print("\n⚙️ Consolidando...")
            if ANDROID_VOZ:
                falar("Salvando cérebro.")
            stop_event.set()
            thread_relogio.join(timeout=2)
            dsml.salvar_estado()
            motor_base.treino_consolidacao()
            print("💤 Cérebro salvo.")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
            sistema_ocupado = False
            if MODO_VOZ:
                time.sleep(0.5)
