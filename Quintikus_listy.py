#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus Listy Perceptivo com RADA Militar
Sistema de classificação entrópica com geometria triangular, 
DNA entrópico linear e predição de objetos em imagens.

Autor: Arquiteto Ronan & Soldado DeepN1

"""

import os
import math
import random
import re
import pickle
import hashlib
import json
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict, Any

# Tentar importar PIL para suporte a múltiplos formatos
try:
    from PIL import Image
    TEM_PIL = True
except ImportError:
    TEM_PIL = False
    print("⚠️ PIL não instalado. Use 'pip install Pillow' para suporte a JPG/PNG.")
    print("   Apenas imagens PPM serão suportadas.")

os.environ["OMP_NUM_THREADS"] = "1"

# ==================================================================
# 🔬 CONSTANTES E PARÂMETROS DO SISTEMA
# ==================================================================
ENTROPIA_BINS = 256
VETOR_DIM = 32  # 32x32 = 1024 dimensões
DNA_DIM = 64    # DNA entrópico tem 64 bits/características
RADA_RAIO = 16  # Raio para RADA (Razão de Amostragem por Densidade Angular)

# ==================================================================
# 🧬 CLASSE: DNA ENTÓPICO LINEAR
# ==================================================================
class DNAEntropico:
    """
    DNA entrópico: representa a assinatura única de uma imagem
    baseada em entropia, geometria e momentos estatísticos.
    """
    def __init__(self, entropia: float, geometria: float, momentos: List[float], 
                 rada: List[float], linearidade: float):
        self.entropia = entropia
        self.geometria = geometria
        self.momentos = momentos  # momentos de Hu simplificados
        self.rada = rada          # vetor RADA (12 ângulos)
        self.linearidade = linearidade
        self._hash = None
    
    def to_binario(self) -> str:
        """Converte DNA para string binária (DNA map bin)"""
        partes = []
        
        # Entropia normalizada (0-8) -> 8 bits
        ent_norm = min(255, int((self.entropia / 8.0) * 255))
        partes.append(f"{ent_norm:08b}")
        
        # Geometria normalizada (0-1) -> 8 bits
        geo_norm = min(255, int(self.geometria * 255))
        partes.append(f"{geo_norm:08b}")
        
        # Momentos (4 momentos, cada 8 bits)
        for m in self.momentos[:4]:
            m_norm = min(255, int(abs(m) * 255))
            partes.append(f"{m_norm:08b}")
        
        # RADA (12 ângulos, cada 4 bits -> 48 bits)
        rada_bits = ""
        for r in self.rada[:12]:
            r_norm = min(15, int(abs(r) * 15))
            rada_bits += f"{r_norm:04b}"
        partes.append(rada_bits)
        
        # Linearidade (8 bits)
        lin_norm = min(255, int(abs(self.linearidade) * 255))
        partes.append(f"{lin_norm:08b}")
        
        return "".join(partes)
    
    def to_hex(self) -> str:
        """Converte DNA para hexadecimal (para visualização)"""
        bin_str = self.to_binario()
        if len(bin_str) % 4 != 0:
            bin_str += "0" * (4 - len(bin_str) % 4)
        return hex(int(bin_str, 2))[2:].upper()
    
    def distancia(self, outro: 'DNAEntropico') -> float:
        """Distância de Hamming entre os DNAs binários"""
        bin1 = self.to_binario()
        bin2 = outro.to_binario()
        max_len = max(len(bin1), len(bin2))
        bin1 = bin1.ljust(max_len, '0')
        bin2 = bin2.ljust(max_len, '0')
        return sum(1 for a, b in zip(bin1, bin2) if a != b) / max_len

# ==================================================================
# 📐 GEOMETRIA TRIANGULAR
# ==================================================================
class GeometriaTriangular:
    """
    Calcula geometria triangular da imagem:
    - Pontos de alta entropia formam triângulos
    - Mede área, perímetro e ângulos dos triângulos formados
    """
    
    @staticmethod
    def encontrar_pontos_entropia(pixels: List[int], largura: int, altura: int, 
                                   n_pontos: int = 3) -> List[Tuple[int, int]]:
        """Encontra os N pontos de maior entropia local na imagem"""
        if len(pixels) < n_pontos:
            return [(0, 0)] * n_pontos
        
        bloco_w = max(1, largura // 8)
        bloco_h = max(1, altura // 8)
        
        pontos = []
        for y in range(0, altura - bloco_h + 1, bloco_h):
            for x in range(0, largura - bloco_w + 1, bloco_w):
                bloco = []
                for j in range(bloco_h):
                    for i in range(bloco_w):
                        idx = (y + j) * largura + (x + i)
                        if idx < len(pixels):
                            bloco.append(pixels[idx])
                if bloco:
                    hist = [0] * 256
                    for p in bloco:
                        hist[p] += 1
                    ent = 0.0
                    total = len(bloco)
                    for count in hist:
                        if count > 0:
                            p = count / total
                            ent -= p * math.log2(p)
                    pontos.append((ent, x + bloco_w//2, y + bloco_h//2))
        
        pontos.sort(reverse=True, key=lambda p: p[0])
        return [(p[1], p[2]) for p in pontos[:n_pontos]]
    
    @staticmethod
    def area_triangulo(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
        """Área do triângulo (fórmula de Shoelace)"""
        return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)
    
    @staticmethod
    def perimetro_triangulo(p1, p2, p3) -> float:
        """Perímetro do triângulo"""
        def dist(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return dist(p1, p2) + dist(p2, p3) + dist(p3, p1)
    
    @staticmethod
    def angulos_triangulo(p1, p2, p3) -> List[float]:
        """Ângulos do triângulo em radianos"""
        def comprimento(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
        a = comprimento(p2, p3)
        b = comprimento(p1, p3)
        c = comprimento(p1, p2)
        
        if a == 0 or b == 0 or c == 0:
            return [0, 0, 0]
        
        ang_a = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c))))
        ang_b = math.acos(max(-1, min(1, (a**2 + c**2 - b**2) / (2*a*c))))
        ang_c = math.pi - ang_a - ang_b
        return [ang_a, ang_b, ang_c]
    
    @staticmethod
    def momentos_geometricos(pontos: List[Tuple[int, int]]) -> List[float]:
        """Calcula momentos geométricos dos pontos de entropia"""
        if len(pontos) < 3:
            return [0.0, 0.0, 0.0, 0.0]
        
        p1, p2, p3 = pontos[0], pontos[1], pontos[2]
        
        area = GeometriaTriangular.area_triangulo(p1, p2, p3)
        perimetro = GeometriaTriangular.perimetro_triangulo(p1, p2, p3)
        angulos = GeometriaTriangular.angulos_triangulo(p1, p2, p3)
        
        max_area = 1024 * 768
        return [area / max_area, perimetro / max(perimetro, 1), angulos[0], angulos[1]]

# ==================================================================
# 🎯 RADA MILITAR (Razão de Amostragem por Densidade Angular)
# ==================================================================
class RADAMilitar:
    """
    RADA: Razão de Amostragem por Densidade Angular
    - Calcula densidade de gradientes em direções específicas (0°, 30°, 60°, ...)
    - Útil para encontrar direção do objeto e bordas
    - Aplica softmax para normalizar as respostas
    """
    
    def __init__(self, n_angulos: int = 12, raio: int = RADA_RAIO):
        self.n_angulos = n_angulos
        self.raio = raio
        self.angulos = [i * 360 / n_angulos for i in range(n_angulos)]
    
    def _gradiente(self, pixels: List[int], largura: int, altura: int, x: int, y: int) -> Tuple[float, float]:
        """Calcula gradiente (Gx, Gy) em um ponto"""
        if x <= 0 or x >= largura-1 or y <= 0 or y >= altura-1:
            return 0.0, 0.0
        
        idx = y * largura + x
        gx = (pixels[min(idx+1, len(pixels)-1)] - pixels[max(idx-1, 0)]) / 2.0
        gy = (pixels[min(idx+largura, len(pixels)-1)] - pixels[max(idx-largura, 0)]) / 2.0
        return gx, gy
    
    def _magnitude_angular(self, gx: float, gy: float) -> Dict[int, float]:
        """Distribui magnitude do gradiente entre ângulos adjacentes (bilinear)"""
        if gx == 0 and gy == 0:
            return {}
        
        angulo = math.degrees(math.atan2(gy, gx)) % 360
        mag = math.sqrt(gx*gx + gy*gy)
        
        idx = int((angulo / (360 / self.n_angulos)) % self.n_angulos)
        idx_next = (idx + 1) % self.n_angulos
        
        ang_centro = self.angulos[idx]
        ang_centro_next = self.angulos[idx_next] if idx_next > 0 else 360
        
        dist = abs(angulo - ang_centro)
        dist_total = abs(ang_centro_next - ang_centro) or 1
        
        peso1 = 1 - (dist / dist_total)
        peso2 = dist / dist_total
        
        return {idx: mag * peso1, idx_next: mag * peso2}
    
    def calcular(self, pixels: List[int], largura: int, altura: int) -> List[float]:
        """Calcula vetor RADA completo para a imagem"""
        acumuladores = [0.0] * self.n_angulos
        step = max(1, min(largura, altura) // 20)
        
        for y in range(1, altura-1, step):
            for x in range(1, largura-1, step):
                gx, gy = self._gradiente(pixels, largura, altura, x, y)
                contrib = self._magnitude_angular(gx, gy)
                for idx, mag in contrib.items():
                    acumuladores[idx] += mag
        
        return self._softmax(acumuladores)
    
    def _softmax(self, vetor: List[float]) -> List[float]:
        """Softmax: normaliza vetor para distribuição de probabilidade"""
        max_val = max(vetor) if vetor else 0
        exp_vals = [math.exp(v - max_val) for v in vetor]
        soma = sum(exp_vals)
        if soma == 0:
            return [1.0 / len(vetor)] * len(vetor)
        return [e / soma for e in exp_vals]
    
    def direcao_principal(self, rada: List[float]) -> float:
        """Retorna a direção principal (ângulo de maior densidade)"""
        if not rada:
            return 0.0
        idx_max = rada.index(max(rada))
        return self.angulos[idx_max]

# ==================================================================
# 🔢 CONVERSOR UNIVERSAL COM PIL
# ==================================================================
class ConversorUniversal:
    """Converte qualquer entrada (texto, número, imagem) em vetor + métricas"""
    
    @staticmethod
    def _rgb_para_cinza(rgb: List[int]) -> int:
        """Converte RGB para cinza usando fórmula ITU-R 601-2"""
        return int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    
    @staticmethod
    def _ler_ppm(caminho: str) -> Tuple[List[int], int, int]:
        """Leitor de PPM puro (fallback quando PIL não disponível)"""
        with open(caminho, 'rb') as f:
            linha = f.readline().decode('ascii').strip()
            if not linha.startswith('P'):
                raise ValueError("Formato PPM inválido")
            formato = linha
            
            linha = f.readline().decode('ascii').strip()
            while linha.startswith('#'):
                linha = f.readline().decode('ascii').strip()
            
            dimensoes = linha.split()
            if len(dimensoes) < 2:
                linha = f.readline().decode('ascii').strip()
                dimensoes = linha.split()
            
            largura, altura = int(dimensoes[0]), int(dimensoes[1])
            
            linha = f.readline().decode('ascii').strip()
            while linha.startswith('#'):
                linha = f.readline().decode('ascii').strip()
            maxval = int(linha)
            
            if formato == 'P3':
                dados = []
                for linha in f:
                    if linha.startswith(b'#'):
                        continue
                    dados.extend([int(x) for x in linha.split()])
                
                if len(dados) != largura * altura * 3:
                    raise ValueError("Número de pixels inconsistente")
                
                pixels = []
                for i in range(0, len(dados), 3):
                    cinza = ConversorUniversal._rgb_para_cinza(dados[i:i+3])
                    if maxval != 255:
                        cinza = int(cinza * 255 / maxval)
                    pixels.append(cinza)
            elif formato == 'P6':
                dados = f.read()
                if len(dados) != largura * altura * 3:
                    raise ValueError("Tamanho de dados binários incorreto")
                pixels = []
                for i in range(0, len(dados), 3):
                    r, g, b = dados[i], dados[i+1], dados[i+2]
                    cinza = ConversorUniversal._rgb_para_cinza([r, g, b])
                    if maxval != 255:
                        cinza = int(cinza * 255 / maxval)
                    pixels.append(cinza)
            else:
                raise ValueError(f"Formato não suportado: {formato}")
            
            return pixels, largura, altura
    
    @staticmethod
    def _carregar_com_pil(caminho: str) -> Tuple[List[int], int, int]:
        """Carrega imagem usando PIL (suporta JPG, PNG, etc.)"""
        if not TEM_PIL:
            raise RuntimeError("PIL não disponível")
        img = Image.open(caminho).convert('L')
        largura, altura = img.size
        if largura > VETOR_DIM or altura > VETOR_DIM:
            img.thumbnail((VETOR_DIM, VETOR_DIM), Image.Resampling.LANCZOS)
            largura, altura = img.size
        
        # SUPORTE MODERNO DO PILLOW: Remove o Warning do getdata()
        if hasattr(img, 'get_flattened_data'):
            pixels = list(img.get_flattened_data())
        else:
            pixels = list(img.getdata())
            
        return pixels, largura, altura
    
    @staticmethod
    def _carregar_imagem(caminho: str) -> Tuple[List[int], int, int]:
        """Carrega imagem (usa PIL se disponível, senão PPM)"""
        if TEM_PIL and not caminho.lower().endswith('.ppm'):
            return ConversorUniversal._carregar_com_pil(caminho)
        else:
            return ConversorUniversal._ler_ppm(caminho)
    
    @staticmethod
    def _redimensionar_pixels(pixels: List[int], largura_orig: int, altura_orig: int,
                               nova_largura: int, nova_altura: int) -> List[int]:
        """Redimensiona usando vizinho mais próximo"""
        novo = [0] * (nova_largura * nova_altura)
        fator_larg = largura_orig / nova_largura
        fator_alt = altura_orig / nova_altura
        
        for y in range(nova_altura):
            for x in range(nova_largura):
                src_x = int(x * fator_larg)
                src_y = int(y * fator_alt)
                idx_src = src_y * largura_orig + src_x
                idx_dst = y * nova_largura + x
                if idx_src < len(pixels):
                    novo[idx_dst] = pixels[idx_src]
        return novo
    
    @staticmethod
    def _histograma(pixels: List[int]) -> List[int]:
        hist = [0] * 256
        for p in pixels:
            hist[p] += 1
        return hist
    
    @staticmethod
    def _entropia(hist: List[int], total: int) -> float:
        ent = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                ent -= p * math.log2(p)
        return ent
    
    @staticmethod
    def _momentos_estatisticos(pixels: List[int]) -> List[float]:
        """Calcula momentos estatísticos (média, variância, assimetria, curtose)"""
        if not pixels:
            return [0, 0, 0, 0]
        n = len(pixels)
        media = sum(pixels) / n
        var = sum((p - media) ** 2 for p in pixels) / n
        skew = sum((p - media) ** 3 for p in pixels) / (n * (var ** 1.5 + 1e-9))
        kurt = sum((p - media) ** 4 for p in pixels) / (n * (var ** 2 + 1e-9)) - 3
        return [media / 255, var / (255*255), max(-1, min(1, skew/10)), max(-1, min(1, kurt/10))]
    
    @staticmethod
    def texto_para_vetor(texto: str, dim=8) -> List[float]:
        seed = int(hashlib.sha256(texto.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.gauss(0, 0.5) for _ in range(dim)]
    
    @staticmethod
    def texto_para_geometria(texto: str) -> float:
        palavras = texto.split()
        if not palavras:
            return 0.0
        comp = min(len(texto) / 1000, 1.0)
        div = len(set(palavras)) / len(palavras)
        return (comp + div) / 2
    
    @staticmethod
    def numero_para_vetor(num: float, dim=8) -> List[float]:
        return [math.sin(num * (i+1)) * math.exp(-abs(num)/10) for i in range(dim)]
    
    @staticmethod
    def numero_para_geometria(num: float) -> float:
        return math.tanh(abs(num)/100)
    
    def converter_imagem(self, caminho: str) -> Dict[str, Any]:
        """Converte imagem e extrai todas as métricas"""
        pixels, largura, altura = self._carregar_imagem(caminho)
        
        pixels_redim = self._redimensionar_pixels(pixels, largura, altura, VETOR_DIM, VETOR_DIM)
        vetor = [p / 255.0 for p in pixels_redim]
        
        hist = self._histograma(pixels)
        entropia = self._entropia(hist, len(pixels))
        
        geometria = min(largura/altura, altura/largura)
        
        momentos = self._momentos_estatisticos(pixels)
        
        pontos = GeometriaTriangular.encontrar_pontos_entropia(pixels, largura, altura, 3)
        momentos_tri = GeometriaTriangular.momentos_geometricos(pontos)
        
        rada = RADAMilitar().calcular(pixels, largura, altura)
        
        linearidade = self._calcular_linearidade_entropica(pixels, largura, altura)
        
        return {
            'vetor': vetor,
            'entropia': entropia,
            'geometria': geometria,
            'momentos': momentos,
            'momentos_tri': momentos_tri,
            'rada': rada,
            'linearidade': linearidade,
            'dimensoes': (largura, altura)
        }
    
    def _calcular_linearidade_entropica(self, pixels: List[int], largura: int, altura: int) -> float:
        """Calcula linearidade da entropia ao longo das linhas da imagem"""
        bloco_h = max(1, altura // 8)
        entropias = []
        
        for y in range(0, altura - bloco_h + 1, bloco_h):
            bloco = []
            for j in range(bloco_h):
                for x in range(largura):
                    idx = (y + j) * largura + x
                    if idx < len(pixels):
                        bloco.append(pixels[idx])
            if bloco:
                hist = [0] * 256
                for p in bloco:
                    hist[p] += 1
                ent = 0.0
                total = len(bloco)
                for count in hist:
                    if count > 0:
                        p = count / total
                        ent -= p * math.log2(p)
                entropias.append(ent)
        
        if len(entropias) < 2:
            return 0.0
        
        n = len(entropias)
        x = list(range(n))
        sx = sum(x)
        sy = sum(entropias)
        sxx = sum(xi*xi for xi in x)
        syy = sum(yi*yi for yi in entropias)
        sxy = sum(xi*yi for xi, yi in zip(x, entropias))
        
        denom = (n*sxx - sx*sx)*(n*syy - sy*sy)
        if denom == 0:
            return 0.0
        return abs((n*sxy - sx*sy) / math.sqrt(denom))

# ==================================================================
# 🧬 CLASSIFICADOR COM DNA ENTÓPICO
# ==================================================================
class ClassificadorDNA:
    """Classificador que usa DNA entrópico para identificar objetos"""
    
    def __init__(self):
        self.classes: Dict[str, List[DNAEntropico]] = defaultdict(list)
        self.historico: List[Tuple[str, float]] = []  # (tag, similaridade)
    
    def criar_dna(self, metricas: Dict[str, Any]) -> DNAEntropico:
        """Cria DNA entrópico a partir das métricas da imagem"""
        momentos = metricas.get('momentos', [0,0,0,0])[:2]
        momentos += metricas.get('momentos_tri', [0,0,0,0])[:2]
        
        return DNAEntropico(
            entropia=metricas['entropia'],
            geometria=metricas['geometria'],
            momentos=momentos,
            rada=metricas.get('rada', [0]*12),
            linearidade=metricas.get('linearidade', 0)
        )
    
    def adicionar(self, tag: str, metricas: Dict[str, Any]):
        """Adiciona uma imagem como exemplo de uma classe"""
        dna = self.criar_dna(metricas)
        self.classes[tag].append(dna)
    
    def prever(self, metricas: Dict[str, Any]) -> Tuple[str, float, float]:
        """
        Prediz a classe da imagem usando RADA + DNA.
        Retorna (tag, similaridade_max, confianca_softmax)
        """
        dna_entrada = self.criar_dna(metricas)
        
        scores = []
        for tag, exemplos in self.classes.items():
            if not exemplos:
                continue
            dist_media = sum(dna_entrada.distancia(ex) for ex in exemplos) / len(exemplos)
            similaridade = 1.0 - dist_media
            scores.append((tag, similaridade))
        
        if not scores:
            return "desconhecido", 0.0, 0.0
        
        scores.sort(key=lambda x: x[1], reverse=True)
        tag_melhor, melhor_score = scores[0]
        
        exp_scores = [math.exp(s*5) for _, s in scores]
        total = sum(exp_scores)
        confianca = exp_scores[0] / total if total > 0 else 0
        
        self.historico.append((tag_melhor, melhor_score))
        if len(self.historico) > 100:
            self.historico.pop(0)
        
        return tag_melhor, melhor_score, confianca
    
    def rada_direcao(self, metricas: Dict[str, Any]) -> float:
        """Retorna a direção principal RADA da imagem"""
        rada = metricas.get('rada', [0]*12)
        if not rada:
            return 0.0
        angulos = [i * 30 for i in range(12)]
        idx_max = rada.index(max(rada))
        return angulos[idx_max]
    
    def salvar(self, path: str):
        """Salva o classificador em disco"""
        dados = {
            'classes': {tag: [(dna.entropia, dna.geometria, dna.momentos, dna.rada, dna.linearidade) 
                              for dna in exemplos] 
                        for tag, exemplos in self.classes.items()},
            'historico': self.historico
        }
        with open(path, 'wb') as f:
            pickle.dump(dados, f)
    
    def carregar(self, path: str):
        """Carrega o classificador do disco"""
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    dados = pickle.load(f)
                self.classes.clear()
                for tag, exemplos in dados.get('classes', {}).items():
                    for (ent, geo, mom, rada, lin) in exemplos:
                        dna = DNAEntropico(ent, geo, mom, rada, lin)
                        self.classes[tag].append(dna)
                self.historico = dados.get('historico', [])
            except Exception as e:
                print(f"Erro ao carregar: {e}")

# ==================================================================
# 👁️ VISUALIZADOR DE OBJETOS (pinta área detectada)
# ==================================================================
class VisualizadorObjetos:
    """Detecta e marca área do objeto na imagem usando RADA + geometria"""
    
    def __init__(self, classificador: ClassificadorDNA):
        self.classificador = classificador
    
    def detectar_regiao(self, pixels: List[int], largura: int, height: int, 
                         direcao_rada: float) -> Tuple[int, int, int, int]:
        """
        Detecta a região do objeto baseado na direção RADA e entropia.
        Retorna (x, y, largura, altura) da bounding box.
        """
        bloco_w = max(1, largura // 10)
        bloco_h = max(1, height // 10)
        
        mapa_entropia = []
        for y in range(0, height - bloco_h + 1, bloco_h):
            linha = []
            for x in range(0, largura - bloco_w + 1, bloco_w):
                bloco = []
                for j in range(bloco_h):
                    for i in range(bloco_w):
                        idx = (y + j) * largura + (x + i)
                        if idx < len(pixels):
                            bloco.append(pixels[idx])
                if bloco:
                    hist = [0] * 256
                    for p in bloco:
                        hist[p] += 1
                    ent = 0.0
                    total = len(bloco)
                    for count in hist:
                        if count > 0:
                            p = count / total
                            ent -= p * math.log2(p)
                    linha.append(ent)
                else:
                    linha.append(0)
            mapa_entropia.append(linha)
        
        melhor_ent = -1
        melhor_pos = (0, 0)
        for y in range(len(mapa_entropia)):
            for x in range(len(mapa_entropia[y])):
                if mapa_entropia[y][x] > melhor_ent:
                    melhor_ent = mapa_entropia[y][x]
                    melhor_pos = (x * bloco_w, y * bloco_h)
        
        rad = math.radians(direcao_rada)
        dx = int(math.cos(rad) * bloco_w * 2)
        dy = int(math.sin(rad) * bloco_h * 2)
        
        x = max(0, melhor_pos[0] - abs(dx)//2)
        y = max(0, melhor_pos[1] - abs(dy)//2)
        w = min(largura - x, bloco_w * 3 + abs(dx))
        h = min(height - y, bloco_h * 3 + abs(dy))
        
        return (x, y, w, h)
    
    def marcar_regiao(self, pixels: List[int], largura: int, altura: int, 
                      bbox: Tuple[int, int, int, int], cor: int = 255) -> List[int]:
        """Marca a região detectada com a cor especificada (branco = 255)"""
        x, y, w, h = bbox
        novos_pixels = pixels.copy()
        
        for i in range(w):
            if y < altura and x+i < largura:
                idx = y * largura + (x + i)
                if idx < len(novos_pixels):
                    novos_pixels[idx] = cor
            if y+h-1 < altura and x+i < largura:
                idx = (y+h-1) * largura + (x + i)
                if idx < len(novos_pixels):
                    novos_pixels[idx] = cor
        
        for j in range(h):
            if y+j < altura and x < largura:
                idx = (y+j) * largura + x
                if idx < len(novos_pixels):
                    novos_pixels[idx] = cor
            if y+j < altura and x+w-1 < largura:
                idx = (y+j) * largura + (x + w - 1)
                if idx < len(novos_pixels):
                    novos_pixels[idx] = cor
        
        return novos_pixels
    
    def prever_e_marcar(self, metricas: Dict[str, Any], caminho_original: str) -> Tuple[str, float, List[int]]:
        """Prediz a classe e retorna os pixels marcados para a imagem original"""
        tag, score, conf = self.classificador.prever(metricas)
        direcao = self.classificador.rada_direcao(metricas)
        
        conversor = ConversorUniversal()
        pixels, largura, altura = conversor._carregar_imagem(caminho_original)
        
        bbox = self.detectar_regiao(pixels, largura, altura, direcao)
        pixels_marcados = self.marcar_regiao(pixels, largura, altura, bbox, cor=255)
        
        return tag, conf, pixels_marcados

# ==================================================================
# 📸 SISTEMA PRINCIPAL QUINTIKUS LISTY
# ==================================================================
class QuintikusListy:
    def __init__(self, arquivo_classes="classificador_dna.bin"):
        self.conversor = ConversorUniversal()
        self.classificador = ClassificadorDNA()
        self.visualizador = VisualizadorObjetos(self.classificador)
        self.arquivo = arquivo_classes
        self.historico_entradas = []
        self.carregar()
    
    def treinar_imagem(self, caminho: str, tag: str) -> Dict[str, Any]:
        """Treina o classificador com uma imagem"""
        metricas = self.conversor.converter_imagem(caminho)
        self.classificador.adicionar(tag, metricas)
        self.salvar()
        return metricas
    
    def prever_imagem(self, caminho: str) -> Tuple[str, float, float]:
        """Prediz a classe de uma imagem"""
        metricas = self.conversor.converter_imagem(caminho)
        tag, score, conf = self.classificador.prever(metricas)
        self.historico_entradas.append((caminho, tag, conf))
        if len(self.historico_entradas) > 50:
            self.historico_entradas.pop(0)
        return tag, score, conf
    
    def prever_e_marcar(self, caminho: str) -> Tuple[str, float, str]:
        """Prediz, marca a região e retorna a imagem salva marcada"""
        metricas = self.conversor.converter_imagem(caminho)
        tag, conf, pixels_marcados = self.visualizador.prever_e_marcar(metricas, caminho)
        
        saida = caminho.replace('.', '_marcado.')
        pixels, largura, altura = self.conversor._carregar_imagem(caminho)
        self._salvar_imagem_original(saida, pixels_marcados, largura, altura, caminho)
        
        return tag, conf, saida
    
    def _salvar_imagem_original(self, caminho: str, pixels: List[int], 
                                largura: int, altura: int, original: str):
        """Salva a imagem marcada no mesmo formato da original (suporta JPG/PNG via PIL)"""
        if TEM_PIL and not original.lower().endswith('.ppm'):
            from PIL import Image
            img_bytes = bytes(pixels)
            img = Image.frombytes('L', (largura, altura), img_bytes)
            img.save(caminho)
        else:
            self._salvar_ppm(caminho, pixels, largura, altura)
            
    def _salvar_ppm(self, caminho: str, pixels: List[int], largura: int, altura: int):
        """Salva pixels no formato PPM (P3)"""
        with open(caminho, 'w') as f:
            f.write("P3\n")
            f.write(f"{largura} {altura}\n")
            f.write("255\n")
            for y in range(altura):
                linha = []
                for x in range(largura):
                    p = pixels[y * largura + x]
                    linha.append(f"{p} {p} {p}")
                f.write(" ".join(linha) + "\n")
    
    def treinar_lote(self, pasta: str, tag: str):
        """Treina com todas as imagens de uma pasta"""
        import glob
        padroes = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        arquivos = []
        for padrao in padroes:
            arquivos.extend(glob.glob(os.path.join(pasta, padrao)))
        
        for arq in arquivos:
            try:
                self.treinar_imagem(arq, tag)
                print(f"✅ Treinado: {arq} -> {tag}")
            except Exception as e:
                print(f"❌ Erro em {arq}: {e}")
    
    def estatisticas(self) -> Dict:
        """Retorna estatísticas do classificador"""
        stats = {
            'total_classes': len(self.classificador.classes),
            'exemplos_por_classe': {tag: len(ex) for tag, ex in self.classificador.classes.items()},
            'historico_predicoes': self.historico_entradas[-10:],
            'ultimas_classificacoes': self.classificador.historico[-10:]
        }
        return stats
    
    def salvar(self):
        self.classificador.salvar(self.arquivo)
    
    def carregar(self):
        self.classificador.carregar(self.arquivo)

# ==================================================================
# 🎮 INTERFACE DE LINHA DE COMANDO
# ==================================================================
def main():
    print("=" * 60)
    print("QUINTIKUS LISTY PERCEPTIVO com RADA Militar")
    print("Classificação por DNA Entrópico + Geometria Triangular")
    print("=" * 60)
    print()
    
    sistema = QuintikusListy()
    
    if TEM_PIL:
        print("✅ PIL disponível: suporte a JPG/PNG")
    else:
        print("⚠️ PIL não disponível: apenas PPM")
    
    print(f"📊 Dimensão do vetor: {VETOR_DIM}x{VETOR_DIM} pixels")
    print(f"🧬 DNA entrópico: {DNA_DIM} bits")
    print(f"🎯 RADA: {RADA_RAIO} ângulos")
    print()
    
    while True:
        print("\n" + "─" * 40)
        print("COMANDOS:")
        print("  treinar <imagem> <tag>    → treina com uma imagem")
        print("  treinar-pasta <pasta> <tag> → treina todas as imagens da pasta")
        print("  prever <imagem>           → prediz a classe da imagem")
        print("  marcar <imagem>           → prediz e marca região do objeto")
        print("  stats                     → mostra estatísticas")
        print("  tags                      → lista classes treinadas")
        print("  sair                      → encerra")
        print("─" * 40)
        
        cmd = input("\n> ").strip()
        if not cmd:
            continue
        
        if cmd == 'sair':
            sistema.salvar()
            print("💾 Dados salvos. Até logo!")
            break
        
        partes = cmd.split()
        
        if partes[0] == 'treinar' and len(partes) >= 3:
            caminho = partes[1]
            tag = " ".join(partes[2:])
            try:
                metricas = sistema.treinar_imagem(caminho, tag)
                print(f"✅ Treinado: {caminho} -> {tag}")
                print(f"   Entropia: {metricas['entropia']:.3f}")
                print(f"   Geometria: {metricas['geometria']:.3f}")
                print(f"   Linearidade: {metricas['linearidade']:.3f}")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif partes[0] == 'treinar-pasta' and len(partes) >= 3:
            pasta = partes[1]
            tag = " ".join(partes[2:])
            if os.path.isdir(pasta):
                sistema.treinar_lote(pasta, tag)
            else:
                print(f"❌ Pasta não encontrada: {pasta}")
        
        elif partes[0] == 'prever' and len(partes) >= 2:
            caminho = partes[1]
            try:
                tag, score, conf = sistema.prever_imagem(caminho)
                print(f"🎯 Predição: {tag}")
                print(f"   Similaridade: {score:.3f}")
                print(f"   Confiança: {conf:.1%}")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif partes[0] == 'marcar' and len(partes) >= 2:
            caminho = partes[1]
            try:
                tag, conf, saida = sistema.prever_e_marcar(caminho)
                print(f"🎯 Predição: {tag} (confiança: {conf:.1%})")
                print(f"📸 Imagem marcada salva em: {saida}")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif partes[0] == 'stats':
            stats = sistema.estatisticas()
            print(f"📊 Classes: {stats['total_classes']}")
            for tag, count in stats['exemplos_por_classe'].items():
                print(f"   {tag}: {count} exemplos")
            print(f"📈 Últimas predições: {stats['ultimas_classificacoes']}")
        
        elif partes[0] == 'tags':
            if sistema.classificador.classes:
                print("🏷️ Classes treinadas:")
                for tag, exemplos in sistema.classificador.classes.items():
                    print(f"   {tag}: {len(exemplos)} exemplos")
            else:
                print("Nenhuma classe treinada ainda.")
        
        else:
            print("❌ Comando não reconhecido")

if __name__ == "__main__":
    main()
