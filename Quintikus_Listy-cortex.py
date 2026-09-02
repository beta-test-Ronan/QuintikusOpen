#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus_Listy-Cortex: Cérebro Entrópico Geométrico Multimodal (Refatorado)
Unificação de Percepção Visual com Processamento Neural/Emocional

Autor: Arquiteto Ronan & Soldado DeepN1
Versão: 2.1 - Otimizada (NumPy Vectorized)
"""

import os
import math
import pickle
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np

try:
    from PIL import Image
    TEM_PIL = True
except ImportError:
    TEM_PIL = False

os.environ["OMP_NUM_THREADS"] = "1"

# ==================================================================
# 🔬 CONSTANTES E PARÂMETROS DO SISTEMA
# ==================================================================
ENTROPIA_BINS = 256
VETOR_DIM = 32
DNA_DIM = 64
RADA_RAIO = 16

# ==================================================================
# 🧬 PARTE 1: DNA ENTÓPICO & VISÃO (Otimizada)
# ==================================================================
class DNAEntropico:
    __slots__ = ['entropia', 'geometria', 'momentos', 'rada', 'linearidade', 'entropia_dentro', 'entropia_fora']
    
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
    
    def to_binario(self) -> str:
        partes = []
        partes.append(f"{min(255, int((self.entropia / 8.0) * 255)):08b}")
        partes.append(f"{min(255, int(self.geometria * 255)):08b}")
        for m in self.momentos[:4]:
            partes.append(f"{min(255, int(abs(m) * 255)):08b}")
        
        rada_bits = "".join(f"{min(15, int(abs(r) * 15)):04b}" for r in self.rada[:12])
        partes.append(rada_bits)
        partes.append(f"{min(255, int(abs(self.linearidade) * 255)):08b}")
        return "".join(partes)


class GeometriaTriangular:
    @staticmethod
    def calcular_entropia_bloco(bloco: np.ndarray) -> float:
        if bloco.size == 0: return 0.0
        _, counts = np.unique(bloco, return_counts=True)
        prob = counts / counts.sum()
        return -np.sum(prob * np.log2(prob))

    @staticmethod
    def encontrar_pontos_entropia(img_2d: np.ndarray, n_pontos: int = 3) -> List[Tuple[int, int]]:
        altura, largura = img_2d.shape
        if img_2d.size < n_pontos:
            return [(0, 0)] * n_pontos
            
        bloco_w, bloco_h = max(1, largura // 8), max(1, altura // 8)
        pontos = []
        
        # OTIMIZAÇÃO: Varredura baseada em fatiamento de matrizes
        for y in range(0, altura - bloco_h + 1, bloco_h):
            for x in range(0, largura - bloco_w + 1, bloco_w):
                bloco = img_2d[y:y+bloco_h, x:x+bloco_w]
                ent = GeometriaTriangular.calcular_entropia_bloco(bloco)
                pontos.append((ent, x + bloco_w//2, y + bloco_h//2))
                
        pontos.sort(reverse=True, key=lambda p: p[0])
        return [(p[1], p[2]) for p in pontos[:n_pontos]]
    
    @staticmethod
    def area_triangulo(p1, p2, p3) -> float:
        return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)
    
    @staticmethod
    def momentos_geometricos(pontos: List[Tuple[int, int]], largura: int, altura: int) -> List[float]:
        if len(pontos) < 3: return [0.0, 0.0, 0.0, 0.0]
        p1, p2, p3 = pontos[0], pontos[1], pontos[2]
        area = GeometriaTriangular.area_triangulo(p1, p2, p3)
        perim = math.hypot(p1[0]-p2[0], p1[1]-p2[1]) + math.hypot(p2[0]-p3[0], p2[1]-p3[1]) + math.hypot(p3[0]-p1[0], p3[1]-p1[1])
        area_total = max(1, largura * altura)
        return [area / area_total, perim / max(perim, 1), 0.0, 0.0]


class RADAMilitar:
    def __init__(self, n_angulos: int = 12):
        self.n_angulos = n_angulos
    
    def calcular(self, img_2d: np.ndarray) -> List[float]:
        # OTIMIZAÇÃO: Cálculo de gradientes e magnitudes vetorizado (NumPy)
        gy, gx = np.gradient(img_2d.astype(float))
        magnitudes = np.hypot(gx, gy)
        angulos = np.degrees(np.arctan2(gy, gx)) % 360
        
        # Filtrar onde magnitude > 0 para evitar ruído zerado
        mask = magnitudes > 0
        magnitudes_validas = magnitudes[mask]
        angulos_validos = angulos[mask]
        
        acumuladores = np.zeros(self.n_angulos)
        if magnitudes_validas.size > 0:
            bins = np.linspace(0, 360, self.n_angulos + 1)
            indices = np.digitize(angulos_validos, bins) - 1
            indices[indices == self.n_angulos] = 0 # Ajustar borda 360 -> 0
            for i in range(self.n_angulos):
                acumuladores[i] = np.sum(magnitudes_validas[indices == i])
                
        # Softmax manual com estabilidade numérica
        max_v = np.max(acumuladores) if acumuladores.size > 0 else 0
        exp_v = np.exp(acumuladores - max_v)
        soma = np.sum(exp_v)
        return (exp_v / (soma + 1e-9)).tolist()


class ConversorUniversal:
    @staticmethod
    def _carregar_imagem(caminho: str) -> np.ndarray:
        if TEM_PIL:
            img = Image.open(caminho).convert('L')
            img.thumbnail((VETOR_DIM, VETOR_DIM), Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.uint8)
        raise RuntimeError("PIL é necessário para carregar a imagem neste formato.")

    def converter_imagem(self, caminho: str) -> Dict[str, Any]:
        img_2d = self._carregar_imagem(caminho)
        altura, largura = img_2d.shape
        
        # OTIMIZAÇÃO: Partição espacial com máscaras
        mask_dentro = np.zeros_like(img_2d, dtype=bool)
        if largura > 8 and altura > 8:
            mask_dentro[altura//4 : 3*altura//4, largura//4 : 3*largura//4] = True
            
        pixels_dentro = img_2d[mask_dentro]
        pixels_fora = img_2d[~mask_dentro]
        pixels_1d = img_2d.flatten()
        
        pontos_ent = GeometriaTriangular.encontrar_pontos_entropia(img_2d)
        
        return {
            'vetor': (pixels_1d[:VETOR_DIM*VETOR_DIM] / 255.0).tolist(),
            'entropia': GeometriaTriangular.calcular_entropia_bloco(pixels_1d),
            'geometria': min(largura/max(altura,1), altura/max(largura,1)),
            'momentos': [np.mean(pixels_1d)/255.0, 0.1, 0, 0],
            'momentos_tri': GeometriaTriangular.momentos_geometricos(pontos_ent, largura, altura),
            'rada': RADAMilitar().calcular(img_2d),
            'linearidade': 0.5,
            'entropia_dentro': GeometriaTriangular.calcular_entropia_bloco(pixels_dentro),
            'entropia_fora': GeometriaTriangular.calcular_entropia_bloco(pixels_fora)
        }


class LaminyPredictor:
    @staticmethod
    def cologaritmo_dist(a: float, b: float) -> float:
        return -math.log2(1.0 - max(0.0, min(1.0, abs(a - b))) + 1e-9)

    @classmethod
    def comparar(cls, d1: DNAEntropico, d2: DNAEntropico) -> float:
        soma = (3.0 * cls.cologaritmo_dist(d1.entropia_dentro/8, d2.entropia_dentro/8) +
                3.0 * cls.cologaritmo_dist(d1.entropia_fora/8, d2.entropia_fora/8) +
                cls.cologaritmo_dist(d1.entropia/8, d2.entropia/8))
        return math.exp(-soma / 10.0)

# ==================================================================
# 🧠 PARTE 2: CÓRTEX NEURAL E EMOÇÃO
# ==================================================================
def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / norm if norm > 1e-8 else x

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

class TGNC_NeuralCortex_v14:
    def __init__(self, dim=128, rank=64):
        self.dim, self.rank = dim, rank
        self.lr, self.decay = 0.5, 0.999
        self.solo_path = "quintikus_cortex_memory.json"
        self.vocab, self.concepts, self.actions = {}, {}, {}
        
        self.W1 = np.random.randn(dim, rank) * 0.01 
        self.W2 = np.random.randn(rank, dim) * 0.01
        self.W_conf = np.random.randn(dim * 2, 1) * 0.01
        self.W_context = np.eye(dim)
        self.W_action = np.eye(dim)
        self.ruido = {"o", "a", "os", "as", "de", "da", "do", "um", "uma", "esta", "com", "que", "para", "em"}
        self.carregar_cortex()

    def _get_vector(self, word: str, space: str = "vocab") -> np.ndarray:
        target = {"vocab": self.vocab, "concepts": self.concepts, "actions": self.actions}[space]
        if word not in target: 
            target[word] = l2_normalize(np.random.uniform(-1, 1, self.dim))
        return target[word]

    def processar_sequencia(self, texto: str) -> Tuple[Optional[np.ndarray], float]:
        if not texto or not texto.strip(): return None, 0.0
        palavras = [w for w in texto.lower().replace(".", " ").split() if w not in self.ruido]
        
        if not palavras: 
            media_vetor = np.mean([self._get_vector(p) for p in texto.lower().split()], axis=0)
            return l2_normalize(media_vetor), 0.15
            
        h_states, prev_v = [], np.zeros(self.dim)
        for p in palavras:
            v_k = self._get_vector(p)
            h_k = l2_normalize(v_k + (prev_v @ self.W1) @ self.W2)
            h_states.append(h_k)
            prev_v = h_k
            
        context_vector = l2_normalize(np.mean(h_states, axis=0))
        conf_score = float(sigmoid(np.dot(np.concatenate([context_vector, prev_v]), self.W_conf.flatten())))
        return context_vector, conf_score

    def ensinar(self, texto_longo: str, nexo: str, acao: str):
        self.W_context *= self.decay
        self.W_action *= self.decay
        x, _ = self.processar_sequencia(texto_longo)
        if x is None: return
        
        y_t = self._get_vector(nexo, "concepts")
        z_t = self._get_vector(acao, "actions")
        
        for _ in range(40): 
            self.W_context += self.lr * np.outer(x, (y_t - x @ self.W_context))
            self.W_action += self.lr * np.outer(y_t, (z_t - y_t @ self.W_action))

    def analisar(self, texto: str) -> Tuple[str, str, float, str]:
        x, conf_n = self.processar_sequencia(texto)
        if x is None: return "SILÊNCIO", "NENHUMA", 0.0, "NEUTRO"
        
        v_n = l2_normalize(x @ self.W_context)
        v_a = l2_normalize(v_n @ self.W_action)
        
        nexo_f = self._buscar_proximo(v_n, "concepts")
        acao_f = self._buscar_proximo(v_a, "actions")
        sim_geo = float(np.dot(v_n, self.concepts.get(nexo_f, np.zeros(self.dim))))
        
        return nexo_f, acao_f, float((sim_geo * 0.7) + (conf_n * 0.3)), "ESTÁVEL"

    def _buscar_proximo(self, vetor: np.ndarray, space: str) -> str:
        target = {"concepts": self.concepts, "actions": self.actions}[space]
        if not target: return "Indefinido"
        return max(target.keys(), key=lambda k: np.dot(vetor, target[k]))

    def salvar_cortex(self):
        def conv(obj):
            if isinstance(obj, dict): return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, set): return list(obj)
            return obj
        with open(self.solo_path, "w") as f: 
            json.dump(conv(self.__dict__), f)

    def carregar_cortex(self):
        if not os.path.exists(self.solo_path): return
        try:
            with open(self.solo_path, "r") as f:
                d = json.load(f)
                for k, v in d.items():
                    if isinstance(v, dict): 
                        setattr(self, k, {ik: np.array(iv) for ik, iv in v.items()})
                    elif isinstance(v, list) and k not in ["ruido"]: 
                        setattr(self, k, np.array(v))
                    elif k == "ruido":
                        self.ruido = set(v)
                    else: 
                        setattr(self, k, v)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Aviso: Não foi possível carregar o córtex ({e})")

class FeelingModule:
    def __init__(self):
        self.path = "quintikus_feeling_rules.json"
        self.rules = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f: self.rules = json.load(f)
            except Exception:
                pass

    def avaliar(self, nexo: str, conf: float, current: str) -> str:
        for r in self.rules:
            if r.get("perceber", "") in nexo.upper() and r.get("v1", 0) <= conf <= r.get("v2", 1): 
                return r.get("mood", current)
        return current

class WorldModel:
    def __init__(self):
        self.path = "quintikus_world_rules.json"
        self.rules = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f: self.rules = json.load(f)
            except Exception:
                pass

    def simular(self, acao: str, props: dict) -> Tuple[str, float]:
        rule = self.rules.get(acao.upper(), {"risco": 0.5})
        if props.get("fragil") and rule.get("risco", 0.5) > 0.4: 
            return "PERIGO: Risco de quebra", rule.get("risco", 0.5)
        return "OK", 0.0

# ==================================================================
# 🎛️ ORQUESTRADOR GERAL: QUINTIKUS_LISTY-CORTEX
# ==================================================================
class QuintikusListyCortex:
    def __init__(self):
        self.conversor = ConversorUniversal()
        self.tgnc = TGNC_NeuralCortex_v14()
        self.world = WorldModel()
        self.snc = FeelingModule()
        self.visual_memory: Dict[str, List[DNAEntropico]] = defaultdict(list)
        self.memory_path = "quintikus_visual_memory.pkl"
        self.carregar_memoria_visual()

    def registrar_visao(self, caminho_imagem: str, tag: str):
        metricas = self.conversor.converter_imagem(caminho_imagem)
        dna = DNAEntropico(
            entropia=metricas['entropia'], 
            geometria=metricas['geometria'],
            momentos=metricas['momentos'] + metricas['momentos_tri'][:2],
            rada=metricas['rada'], 
            linearidade=metricas['linearidade'],
            entropia_dentro=metricas['entropia_dentro'], 
            entropia_fora=metricas['entropia_fora']
        )
        self.visual_memory[tag].append(dna)
        self.salvar_memoria_visual()
        print(f"👁️ [VISÃO GRAVADA]: Padrão salvo para a tag '{tag}' com sucesso.")

    def reconhecer_visao(self, caminho_imagem: str) -> Tuple[str, float]:
        if not self.visual_memory: return "Nenhum padrão gravado", 0.0
        
        metricas = self.conversor.converter_imagem(caminho_imagem)
        dna_entrada = DNAEntropico(
            entropia=metricas['entropia'], 
            geometria=metricas['geometria'],
            momentos=metricas['momentos'] + metricas['momentos_tri'][:2],
            rada=metricas['rada'], 
            linearidade=metricas['linearidade'],
            entropia_dentro=metricas['entropia_dentro'], 
            entropia_fora=metricas['entropia_fora']
        )
        
        melhor_tag, maior_sim = "Desconhecido", -1.0
        for tag, exemplos in self.visual_memory.items():
            sim_media = sum(LaminyPredictor.comparar(dna_entrada, ex) for ex in exemplos) / len(exemplos)
            if sim_media > maior_sim:
                maior_sim = sim_media
                melhor_tag = tag
        return melhor_tag, float(maior_sim)

    def pensar_texto(self, texto: str) -> str:
        nexo, acao, conf, mood_n = self.tgnc.analisar(texto)
        mood = self.snc.avaliar(nexo, conf, mood_n)
        info, risco = self.world.simular(acao, {"fragil": "fragil" in texto.lower()})
        decisao = "INTERROMPER" if risco > 0.6 else "EXECUTAR"
        return f"  [NEXO]: {nexo} | [AÇÃO]: {acao} | [HUMOR]: {mood} | [CONFIANÇA]: {conf:.2f}\n  [DECISÃO]: {decisao} ({info})"

    def salvar_memoria_visual(self):
        dados = {tag: [(d.entropia, d.geometria, d.momentos, d.rada, d.linearidade, d.entropia_dentro, d.entropia_fora) 
                       for d in ex] for tag, ex in self.visual_memory.items()}
        with open(self.memory_path, "wb") as f: 
            pickle.dump(dados, f)

    def carregar_memoria_visual(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "rb") as f:
                    dados = pickle.load(f)
                    for tag, ex_list in dados.items():
                        for ex in ex_list:
                            self.visual_memory[tag].append(DNAEntropico(*ex))
            except Exception as e:
                print(f"Aviso: Não foi possível carregar memória visual ({e})")

# ==================================================================
# 🚀 EXECUÇÃO DO CÉREBRO UNIFICADO
# ==================================================================
if __name__ == "__main__":
    cortex = QuintikusListyCortex()
    print("=" * 65)
    print("🧠 QUINTIKUS_LISTY-CORTEX (Cérebro Entrópico Geométrico Multimodal) ATIVO")
    print("=" * 65)
    print("Comandos disponíveis:")
    print("  ver <caminho_imagem> <tag>     → Grava/associa padrão visual de uma pessoa ou imagem")
    print("  reconhecer <caminho_imagem>    → Analisa a imagem e identifica o padrão gravado")
    print("  <texto de comando/pensamento>  → Processa linguagem, nexo e tomada de decisão neural")
    print("  sair                           → Encerra")
    print("-" * 65)

    while True:
        try:
            cmd = input("\nCortex > ").strip()
            if not cmd: continue
            if cmd.lower() == 'sair': 
                cortex.tgnc.salvar_cortex() # Garante o salvamento no final
                break
            
            partes = cmd.split(maxsplit=2)
            if partes[0].lower() == 'ver' and len(partes) >= 3:
                cortex.registrar_visao(partes[1], partes[2])
            elif partes[0].lower() == 'reconhecer' and len(partes) >= 2:
                tag, sim = cortex.reconhecer_visao(partes[1])
                print(f"🎯 [RECONHECIMENTO VISUAL]: Identificado como '{tag}' (Similaridade: {sim:.3f})")
            else:
                print(cortex.pensar_texto(cmd))
        except Exception as e:
            print(f"❌ Erro de Execução: {e}")