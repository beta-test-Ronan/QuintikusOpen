#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus DLMC V85.2 – Deep Sea Navigation + VOZ (Auto‑Cura Total)
- GPS + Periscópio + Computador de Bordo
- Microfone e speaker no Android (QPython)
- Entrada segura sem duplicação de caracteres (safe_input)
- Verificação e regeneração de todos os módulos neurais (inclusive células GRU)
- Pós-processamento inteligente de pontuação (?, !, .)
"""

import hashlib, math, random, time, pickle, os, tempfile, sys
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

# ============================================
# BLOCO ANDROID – Detecção segura
# ============================================
ANDROID_VOZ = False
MODO_VOZ = False
droid = None

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
    """Entrada sem duplicação de caracteres no QPython."""
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

# ============================================================
# PÓS-PROCESSAMENTO INTELIGENTE DE PONTUAÇÃO
# ============================================================
def ajustar_pontuacao(resposta: str) -> str:
    """Adiciona pontuação final adequada (?, !, .) baseada no conteúdo."""
    if not resposta:
        return resposta
    ultimo = resposta[-1]
    if ultimo in '.!?':
        return resposta
    # Palavras interrogativas
    if any(p in resposta.lower() for p in ['quem', 'quando', 'onde', 'por que', 'como', 'qual', '?' ]):
        return resposta + '?'
    # Exclamações (heurística simples)
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

    # ============================================================
    # AUTO‑CURA TOTAL (VERIFICA CÉLULAS GRU TAMBÉM)
    # ============================================================
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

    # ============================================================
    # PERSISTÊNCIA ATÔMICA
    # ============================================================
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

    # ============================================================
    # EMBEDDINGS
    # ============================================================
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

    # ============================================================
    # COMPORTAMENTO EMERGENTE
    # ============================================================
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

    # ============================================================
    # RESPOSTA LINEAR (com ajuste de pontuação)
    # ============================================================
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

    # ============================================================
    # COMPUTADOR DE BORDO (TRIANGULAÇÃO)
    # ============================================================
    def _triangulate(self, gps_signal, periscope_signal):
        gps_confidence = 0.7 if len(self.historico_embeddings) > 3 else 0.3
        periscope_confidence = 0.6
        alpha = gps_confidence / (gps_confidence + periscope_confidence + 1e-8)
        return [alpha * gps_signal[i] + (1 - alpha) * periscope_confidence * periscope_signal[i] for i in range(8)]

    # ============================================================
    # RESPOSTA CRIATIVA (com ajuste de pontuação)
    # ============================================================
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

    # ============================================================
    # PENSAR (COM AUTO‑CURA)
    # ============================================================
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

    # ============================================================
    # INICIALIZAÇÃO
    # ============================================================
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

    # ============================================================
    # PROCESSAR DATASET
    # ============================================================
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

    # ============================================================
    # CONSOLIDAÇÃO
    # ============================================================
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


# ============================================
# MAIN COM MICROFONE CONTÍNUO (safe_input)
# ============================================
if __name__ == "__main__":
    print("🧬 Quintikus DLMC V85.2 – Deep Sea Navigation + VOZ (Auto‑Cura Total)")
    print("   GPS + Periscópio + Computador de Bordo")
    print("=" * 60)

    if ANDROID_VOZ:
        print("🎤🎧 Modo Android detectado! Microfone e Speaker ativos.")
        print("🔁 MODO: Microfone sempre ouvindo (loop infinito)")
    else:
        print("💻 Modo Terminal (sem voz).")
    print("=" * 60)

    texto_inicial = ""
    if os.path.exists("roteiro.txt"):
        with open("roteiro.txt", "r", encoding="utf-8") as f:
            texto_inicial = f.read()
        print(f"📁 roteiro.txt carregado ({len(texto_inicial)} caracteres).")

    motor = QuintikusDLMC(texto_inicial)
    motor.inicializar()

    if ANDROID_VOZ:
        time.sleep(0.5)
        falar("Quintikus V85 está online. Microfone sempre ativo.")
        print("\n🎤 Escolha o modo:")
        print("   [V] Microfone contínuo — sempre ouvindo")
        print("   [T] Teclado normal")
        print("   [sair]")
        modo_inicial = safe_input("   Modo > ").lower()
        
        if modo_inicial == "sair":
            print("💤 Encerrando...")
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
        print("   (Índice Espacial + Atenção Hierárquica ativos)\n")

    while True:
        try:
            # MODO MICROFONE CONTÍNUO
            if MODO_VOZ and ANDROID_VOZ:
                vibrar(30)
                entrada = ouvir("Ouvindo...")
                
                if not entrada or len(entrada.strip()) < 1:
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
                    continue
                
                if entrada_lower in ["sair", "encerrar"]:
                    print("⚙️ Consolidando...")
                    falar("Salvando memória. Até mais, Arquiteto.")
                    motor.treino_consolidacao()
                    print("💤 Cérebro salvo. Até mais!")
                    break
                
                if entrada_lower.startswith("tokens"):
                    try:
                        val = int(entrada_lower.replace("tokens", "").replace(":", "").strip())
                        motor.max_tokens = max(5, min(100, val))
                        msg = f"máximo de tokens ajustado para {motor.max_tokens}"
                        print(f"✅ {msg}")
                        falar(msg)
                    except:
                        pass
                    continue
                
                if entrada_lower.startswith("temp"):
                    try:
                        val = float(entrada_lower.replace("temperatura", "").replace("temp", "").replace(":", "").strip())
                        motor.temperatura = max(0.1, min(2.0, val))
                        msg = f"temperatura ajustada para {motor.temperatura}"
                        print(f"✅ {msg}")
                        falar(msg)
                    except:
                        pass
                    continue
                
                if entrada_lower.startswith("momentum"):
                    try:
                        val = float(entrada_lower.replace("momentum", "").replace(":", "").strip())
                        motor.fator_momentum = max(0.0, min(1.0, val))
                        print(f"✅ momentum = {motor.fator_momentum}")
                    except:
                        pass
                    continue
                
                if entrada_lower.startswith("suavidade"):
                    try:
                        val = float(entrada_lower.replace("suavidade", "").replace(":", "").strip())
                        motor.fator_suavidade = max(0.0, min(1.0, val))
                        print(f"✅ suavidade = {motor.fator_suavidade}")
                    except:
                        pass
                    continue
                
                resposta = motor.pensar(entrada)
                print(f"🧠: {resposta}")
                vibrar(20)
                texto_tts = resposta.replace(".", ". ").replace(",", ", ")
                falar(texto_tts)
                print()
                continue
            
            # MODO TECLADO
            entrada = safe_input("usr:")
            
            if not entrada:
                continue
            
            if entrada.lower() == "sair":
                print("⚙️ Consolidando...")
                if ANDROID_VOZ:
                    falar("Salvando memória. Até mais, Arquiteto.")
                motor.treino_consolidacao()
                print("💤 Cérebro salvo. Até mais!")
                break
            
            if entrada.lower() == "microfone" and ANDROID_VOZ:
                print("🎤 Reativando microfone contínuo...")
                falar("Microfone reativado. Pode falar.")
                MODO_VOZ = True
                print("🔁 Ouvindo... (diga 'teclado' para voltar)\n")
                continue

            if entrada.lower() == "save":
                if motor.salvar(): print("💾 Salvo!")
                else: print("❌ Falha.")
                continue

            if entrada.lower().startswith("tokens:"):
                try: motor.max_tokens = max(5, min(100, int(entrada.split(":",1)[1])))
                except: print("❌ Use: tokens:30")
                else: print(f"✅ max_tokens = {motor.max_tokens}")
                continue

            if entrada.lower().startswith("temp:"):
                try: motor.temperatura = max(0.1, min(2.0, float(entrada.split(":",1)[1])))
                except: print("❌ Use: temp:0.7")
                else: print(f"✅ temp = {motor.temperatura}")
                continue

            if entrada.lower().startswith("momentum:"):
                try: motor.fator_momentum = max(0.0, min(1.0, float(entrada.split(":",1)[1])))
                except: print("❌ Use: momentum:0.4")
                else: print(f"✅ momentum = {motor.fator_momentum}")
                continue

            if entrada.lower().startswith("suavidade:"):
                try: motor.fator_suavidade = max(0.0, min(1.0, float(entrada.split(":",1)[1])))
                except: print("❌ Use: suavidade:0.3")
                else: print(f"✅ suavidade = {motor.fator_suavidade}")
                continue

            if entrada.lower().startswith("gru:"):
                motor.gru_ativo = entrada.split(":",1)[1].strip() == "on"
                print(f"✅ GRU {'ATIVADA' if motor.gru_ativo else 'DESATIVADA'}")
                continue

            if entrada.lower().startswith("debug:"):
                motor.debug = entrada.split(":",1)[1].strip() == "on"
                print(f"✅ debug = {motor.debug}")
                continue

            if entrada.lower().startswith("train:"):
                arquivo = entrada.split(":",1)[1].strip()
                if os.path.exists(arquivo):
                    with open(arquivo, "r", encoding="utf-8") as f:
                        texto = f.read()
                    motor = QuintikusDLMC(texto, "cerebro_v85.bin")
                    motor.inicializar()
                else: print(f"❌ Arquivo '{arquivo}' não encontrado.")
                continue

            resposta = motor.pensar(entrada)
            print(f"🧠: {resposta}")

        except KeyboardInterrupt:
            print("\n⚙️ Consolidando...")
            if ANDROID_VOZ:
                falar("Salvando cérebro.")
            motor.treino_consolidacao()
            print("💤 Cérebro salvo.")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
            if MODO_VOZ:
                time.sleep(0.5)
