#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus AGI Unificado: Visão (Laminy/RADA) + Córtex Neural (TGNC/SNC)
Fusão monolítica sem dependências externas (exceto bibliotecas padrão e numpy).
"""

import os
import math
import random
import re
import pickle
import hashlib
import json
import time
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from PIL import Image
    TEM_PIL = True
except ImportError:
    TEM_PIL = False

os.environ["OMP_NUM_THREADS"] = "1"

# ==============================================================================
# 👁️ BLOCO 1: VISÃO - PARÂMETROS E DNA ENTÓPICO
# ==============================================================================
VETOR_DIM = 32
DNA_DIM = 64
RADA_RAIO = 16

class DNAEntropico:
    def __init__(self, entropia: float, geometria: float, momentos: List[float], 
                 rada: List[float], linearidade: float,
                 entropia_dentro: float = 0.0, entropia_fora: float = 0.0):
        self.entropia = entropia
        self.geometria = geometria
        self.momentos = momentos 
        self.rada = rada          
        self.linearidade = linearidade
        self.entropia_dentro = entropia_dentro
        self.entropia_fora = entropia_fora

class GeometriaTriangular:
    @staticmethod
    def encontrar_pontos_entropia(pixels: List[int], largura: int, altura: int, n_pontos: int = 3) -> List[Tuple[int, int]]:
        if len(pixels) < n_pontos: return [(0, 0)] * n_pontos
        bloco_w, bloco_h = max(1, largura // 8), max(1, altura // 8)
        pontos = []
        for y in range(0, altura - bloco_h + 1, bloco_h):
            for x in range(0, largura - bloco_w + 1, bloco_w):
                bloco = [pixels[(y + j) * largura + (x + i)] for j in range(bloco_h) for i in range(bloco_w) if (y + j) * largura + (x + i) < len(pixels)]
                if bloco:
                    hist = [0] * 256
                    for p in bloco: hist[p] += 1
                    ent, total = 0.0, len(bloco)
                    for count in hist:
                        if count > 0:
                            p = count / total
                            ent -= p * math.log2(p)
                    pontos.append((ent, x + bloco_w//2, y + bloco_h//2))
        pontos.sort(reverse=True, key=lambda p: p[0])
        return [(p[1], p[2]) for p in pontos[:n_pontos]]
    
    @staticmethod
    def area_triangulo(p1, p2, p3):
        return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)
    
    @staticmethod
    def perimetro_triangulo(p1, p2, p3):
        dist = lambda a, b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return dist(p1, p2) + dist(p2, p3) + dist(p3, p1)
    
    @staticmethod
    def angulos_triangulo(p1, p2, p3):
        comp = lambda a, b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        a, b, c = comp(p2, p3), comp(p1, p3), comp(p1, p2)
        if a == 0 or b == 0 or c == 0: return [0, 0, 0]
        ang_a = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c))))
        ang_b = math.acos(max(-1, min(1, (a**2 + c**2 - b**2) / (2*a*c))))
        return [ang_a, ang_b, math.pi - ang_a - ang_b]
    
    @staticmethod
    def momentos_geometricos(pontos: List[Tuple[int, int]]) -> List[float]:
        if len(pontos) < 3: return [0.0, 0.0, 0.0, 0.0]
        p1, p2, p3 = pontos[0], pontos[1], pontos[2]
        area = GeometriaTriangular.area_triangulo(p1, p2, p3)
        perimetro = GeometriaTriangular.perimetro_triangulo(p1, p2, p3)
        angulos = GeometriaTriangular.angulos_triangulo(p1, p2, p3)
        return [area / (1024 * 768), perimetro / max(perimetro, 1), angulos[0], angulos[1]]

class RADAMilitar:
    def __init__(self, n_angulos: int = 12):
        self.n_angulos = n_angulos
        self.angulos = [i * 360 / n_angulos for i in range(n_angulos)]
    
    def calcular(self, pixels: List[int], largura: int, altura: int) -> List[float]:
        acumuladores = [0.0] * self.n_angulos
        step = max(1, min(largura, altura) // 20)
        for y in range(1, altura-1, step):
            for x in range(1, largura-1, step):
                idx = y * largura + x
                gx = (pixels[min(idx+1, len(pixels)-1)] - pixels[max(idx-1, 0)]) / 2.0
                gy = (pixels[min(idx+largura, len(pixels)-1)] - pixels[max(idx-largura, 0)]) / 2.0
                if gx == 0 and gy == 0: continue
                angulo = math.degrees(math.atan2(gy, gx)) % 360
                mag = math.sqrt(gx*gx + gy*gy)
                idx_ang = int((angulo / (360 / self.n_angulos)) % self.n_angulos)
                acumuladores[idx_ang] += mag
        
        max_val = max(acumuladores) if acumuladores else 0
        exp_vals = [math.exp(v - max_val) for v in acumuladores]
        soma = sum(exp_vals)
        return [e / soma for e in exp_vals] if soma > 0 else [1.0 / len(acumuladores)] * len(acumuladores)

class ConversorUniversal:
    @staticmethod
    def _carregar_com_pil(caminho: str, redimensionar: bool = True):
        img = Image.open(caminho).convert('L')
        largura, altura = img.size
        if redimensionar and (largura > VETOR_DIM or altura > VETOR_DIM):
            img.thumbnail((VETOR_DIM, VETOR_DIM), Image.Resampling.LANCZOS)
            largura, altura = img.size
        pixels = list(img.get_flattened_data()) if hasattr(img, 'get_flattened_data') else list(img.getdata())
        return pixels, largura, altura
    
    @staticmethod
    def _ler_ppm(caminho: str):
        with open(caminho, 'rb') as f:
            f.readline(); f.readline(); dimensoes = f.readline().split()
            largura, altura = int(dimensoes[0]), int(dimensoes[1])
            maxval = int(f.readline())
            dados = f.read()
            r_iter = iter(dados)
            pixels = [int(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in zip(r_iter, r_iter, r_iter)]
            return pixels, largura, altura

    def converter_imagem(self, caminho: str) -> Dict[str, Any]:
        if TEM_PIL and not caminho.lower().endswith('.ppm'):
            pixels, largura, altura = self._carregar_com_pil(caminho, True)
        else:
            pixels, largura, altura = self._ler_ppm(caminho)
            
        def entropia(p_list):
            hist = [0]*256
            for p in p_list: hist[p] += 1
            return -sum((c/len(p_list)) * math.log2(c/len(p_list)) for c in hist if c > 0)

        pixels_redim = [pixels[int(y * (altura/VETOR_DIM)) * largura + int(x * (largura/VETOR_DIM))] for y in range(VETOR_DIM) for x in range(VETOR_DIM)]
        pixels_dentro = [pixels_redim[y*VETOR_DIM+x] for y in range(8, 24) for x in range(8, 24)]
        pixels_fora = [pixels_redim[y*VETOR_DIM+x] for y in range(VETOR_DIM) for x in range(VETOR_DIM) if not (8<=x<24 and 8<=y<24)]

        pontos = GeometriaTriangular.encontrar_pontos_entropia(pixels, largura, altura, 3)
        return {
            'entropia': entropia(pixels),
            'geometria': min(largura/altura, altura/largura) if altura>0 and largura>0 else 0,
            'momentos': [sum(pixels)/len(pixels)/255, 0, 0, 0],
            'momentos_tri': GeometriaTriangular.momentos_geometricos(pontos),
            'rada': RADAMilitar().calcular(pixels, largura, altura),
            'linearidade': 0.5,
            'entropia_dentro': entropia(pixels_dentro) if pixels_dentro else 0,
            'entropia_fora': entropia(pixels_fora) if pixels_fora else 0
        }

class Laminy:
    @staticmethod
    def comparar(dna1: DNAEntropico, dna2: DNAEntropico) -> float:
        col = lambda a, b: -math.log2(1.0 - max(0.0, min(1.0, abs(a - b))) + 1e-9)
        soma = (3.0 * col(dna1.entropia_dentro/8, dna2.entropia_dentro/8) +
                3.0 * col(dna1.entropia_fora/8, dna2.entropia_fora/8) +
                col(dna1.entropia/8, dna2.entropia/8) +
                sum(col(r1, r2) for r1, r2 in zip(dna1.rada, dna2.rada))/12)
        fator = 1.0 + abs(dna1.geometria - dna2.geometria)
        return math.exp(-soma / (10.0 * fator))

# ==============================================================================
# 🧠 BLOCO 2: CÓRTEX NEURAL E SENTIMENTOS (TGNC v14)
# ==============================================================================
def l2_normalize(x):
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8) if norm > 0 else x

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

class TGNC_NeuralCortex_v14:
    def __init__(self, dim=128, rank=64):
        self.dim, self.rank = dim, rank
        self.lr, self.decay = 0.5, 0.999
        self.solo_path = "tgnc_cortex_v14.json"
        self.vocab, self.concepts, self.actions = {}, {}, {}
        self.W1 = np.random.randn(dim, rank) * 0.01 
        self.W2 = np.random.randn(rank, dim) * 0.01
        self.W_conf = np.random.randn(dim + dim, 1) * 0.01
        self.W_context, self.W_action = np.eye(dim), np.eye(dim)
        self.ruido = ["o", "a", "os", "as", "de", "da", "do", "um", "uma", "esta", "com", "que", "para", "em", "no", "na", "e"]
        self.carregar_cortex()

    def _get_vector(self, word, space="vocab"):
        target = {"vocab": self.vocab, "concepts": self.concepts, "actions": self.actions}[space]
        if word not in target: target[word] = l2_normalize(np.random.uniform(-1, 1, self.dim))
        return target[word]

    # FIX DO BUGS OPTIONAL
    def processar_sequencia(self, texto: str) -> Tuple[Optional[np.ndarray], float]:
        if not texto or not texto.strip(): return None, 0.0
        palavras = [w for w in texto.lower().replace(".", " ").replace(",", "").split() if w not in self.ruido]
        if not palavras: return l2_normalize(np.mean([self._get_vector(p) for p in texto.lower().split()], axis=0)), 0.15
        
        h_states, prev_v = [], np.zeros(self.dim)
        for p in palavras:
            v_k = self._get_vector(p)
            h_k = l2_normalize(v_k + (prev_v @ self.W1) @ self.W2)
            h_states.append(h_k)
            prev_v = h_k
        context_vector = l2_normalize(np.mean(h_states, axis=0))
        conf_score = float(sigmoid(np.dot(np.concatenate([context_vector, prev_v]), self.W_conf.flatten())))
        return context_vector, conf_score

    def ensinar(self, texto_longo, nexo, acao):
        x, _ = self.processar_sequencia(texto_longo)
        if x is None: return
        y_t, z_t = self._get_vector(nexo, "concepts"), self._get_vector(acao, "actions")
        for _ in range(60): 
            self.W_context += self.lr * np.outer(x, (y_t - x @ self.W_context))
            self.W_action += self.lr * np.outer(y_t, (z_t - y_t @ self.W_action))

    def analisar(self, texto):
        x, conf_n = self.processar_sequencia(texto)
        if x is None: return "SILÊNCIO", "NENHUMA", 0.0, "NEUTRO"
        v_n, v_a = l2_normalize(x @ self.W_context), l2_normalize((x @ self.W_context) @ self.W_action)
        nexo_f, acao_f = self._buscar_proximo(v_n, "concepts"), self._buscar_proximo(v_a, "actions")
        sim_geo = float(np.dot(v_n, self.concepts.get(nexo_f, np.zeros(self.dim))))
        return nexo_f, acao_f, float((sim_geo * 0.7) + (conf_n * 0.3)), "ESTÁVEL"

    def _buscar_proximo(self, vetor, space):
        target = {"concepts": self.concepts, "actions": self.actions}[space]
        if not target: return "Indefinido"
        melhor, sim = "Indefinido", -1.0
        for k, v in target.items():
            if (s := np.dot(vetor, v)) > sim: sim, melhor = s, k
        return melhor

    def carregar_cortex(self):
        if os.path.exists(self.solo_path):
            try:
                with open(self.solo_path, "r") as f:
                    for k, v in json.load(f).items():
                        if isinstance(v, dict): setattr(self, k, {ik: np.array(iv) for ik, iv in v.items()})
                        elif isinstance(v, list) and k not in ["ruido"]: setattr(self, k, np.array(v))
                        else: setattr(self, k, v)
            except: pass
            
    def salvar_cortex(self):
        def conv(obj):
            if isinstance(obj, dict): return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        with open(self.solo_path, "w") as f: json.dump(conv(self.__dict__), f)

class WorldModel:
    def __init__(self):
        self.rules = {}
    def simular(self, acao, props):
        rule = self.rules.get(acao.upper(), {"risco": 0.5})
        if props.get("fragil") and rule["risco"] > 0.4: return "PERIGO", rule["risco"]
        return "OK", 0.0

class FeelingModule:
    def __init__(self):
        self.rules = []
    def avaliar(self, nexo, conf, current):
        for r in self.rules:
            if r.get("perceber", "") in nexo.upper() and r.get("v1", 0) <= conf <= r.get("v2", 1): return r.get("mood", current)
        return current

# ==============================================================================
# 🧩 BLOCO 3: ORQUESTRADOR UNIFICADO (VISÃO + MENTE)
# ==============================================================================
class Quintikus_AGI_Unificado:
    def __init__(self):
        # Módulo de Visão
        self.conversor = ConversorUniversal()
        self.memoria_visual = defaultdict(list)
        self.arquivo_visao = "classificador_dna.bin"
        self.carregar_visao()
        
        # Módulo Cognitivo
        self.tgnc = TGNC_NeuralCortex_v14()
        self.world = WorldModel()
        self.snc = FeelingModule()
        self.memoria_semantica = {}

    # --- FUNÇÕES DE VISÃO ---
    def treinar_imagem(self, caminho: str, tag: str):
        metricas = self.conversor.converter_imagem(caminho)
        dna = DNAEntropico(
            metricas['entropia'], metricas['geometria'],
            metricas['momentos'][:2] + metricas['momentos_tri'][:2],
            metricas['rada'], metricas['linearidade'],
            metricas['entropia_dentro'], metricas['entropia_fora']
        )
        self.memoria_visual[tag].append(dna)
        self.salvar_visao()
        return metricas

    def prever_imagem(self, caminho: str):
        if not self.memoria_visual: return "Nenhum dado", 0.0, 0.0
        metricas = self.conversor.converter_imagem(caminho)
        dna_entrada = DNAEntropico(
            metricas['entropia'], metricas['geometria'],
            metricas['momentos'][:2] + metricas['momentos_tri'][:2],
            metricas['rada'], metricas['linearidade'],
            metricas['entropia_dentro'], metricas['entropia_fora']
        )
        scores = []
        for tag, exemplos in self.memoria_visual.items():
            if exemplos:
                sim_media = sum(Laminy.comparar(dna_entrada, ex) for ex in exemplos) / len(exemplos)
                scores.append((tag, sim_media))
        scores.sort(key=lambda x: x[1], reverse=True)
        tag_melhor, melhor_score = scores[0]
        return tag_melhor, melhor_score, melhor_score

    def carregar_visao(self):
        if os.path.exists(self.arquivo_visao):
            try:
                with open(self.arquivo_visao, 'rb') as f:
                    dados = pickle.load(f)
                    for tag, exemplos in dados.get('classes', {}).items():
                        for ex in exemplos:
                            if len(ex) == 7: self.memoria_visual[tag].append(DNAEntropico(*ex))
            except: pass

    def salvar_visao(self):
        dados = {'classes': {tag: [(d.entropia, d.geometria, d.momentos, d.rada, d.linearidade, d.entropia_dentro, d.entropia_fora) for d in exemplos] for tag, exemplos in self.memoria_visual.items()}}
        with open(self.arquivo_visao, 'wb') as f: pickle.dump(dados, f)

    # --- FUNÇÕES COGNITIVAS ---
    def aprender_texto(self, partes):
        # Formato: aprender|quadro|MARCENARIA|BATER|fragil:False
        txt, nexo, acao = partes[1], partes[2], partes[3]
        self.tgnc.ensinar(txt, nexo, acao)
        self.tgnc.salvar_cortex()
        self.memoria_semantica[partes[1].lower()] = {"fragil": "fragil:True" in "|".join(partes)}
        return f"🌟 Organismo evoluiu com: '{txt}'"

    def pensar(self, entrada):
        nexo, acao, conf, mood_n = self.tgnc.analisar(entrada)
        mood = self.snc.avaliar(nexo, conf, mood_n)
        props = self.memoria_semantica.get(entrada.split()[0].lower() if entrada else "", {"fragil": False})
        info, risco = self.world.simular(acao, props)
        return f"🧠 Nexo: {nexo} | Ação: {acao} | Conf: {conf:.2f} | Humor: {mood}\n🎯 Decisão: {'INTERROMPER' if risco > 0.6 else 'EXECUTAR'} ({info})"


# ==============================================================================
# 🎮 INTERFACE DE LINHA DE COMANDO UNIFICADA
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 QUINTIKUS AGI UNIFICADO (Visão + Córtex)")
    print("=" * 60)
    print("Comandos suportados:")
    print("  VISÃO : treinar <img_path> <tag>  | prever <img_path>")
    print("  CÓRTEX: aprender|texto|NEXO|ACAO  | <qualquer outra frase para pensar>")
    print("  GERAL : sair")
    
    agi = Quintikus_AGI_Unificado()
    
    while True:
        cmd = input("\nAGI > ").strip()
        if not cmd: continue
        if cmd.lower() == 'sair':
            agi.tgnc.salvar_cortex()
            print("💾 Dados salvos. Encerrando.")
            break
            
        partes = cmd.split()
        
        try:
            # Roteamento Visão
            if partes[0] == 'treinar' and len(partes) >= 3:
                agi.treinar_imagem(partes[1], " ".join(partes[2:]))
                print(f"✅ Visão: Imagem {partes[1]} treinada.")
            elif partes[0] == 'prever' and len(partes) >= 2:
                tag, score, _ = agi.prever_imagem(partes[1])
                print(f"🎯 Visão: Identificado como '{tag}' (Similaridade: {score:.3f})")
            
            # Roteamento Córtex (One-Shot Learning)
            elif cmd.startswith('aprender|'):
                print(agi.aprender_texto(cmd.split('|')))
                
            # Processamento Neural de Texto (Pensar)
            else:
                print(agi.pensar(cmd))
        except Exception as e:
            print(f"❌ Erro na execução: {e}")
