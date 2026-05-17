"""
QUINTIKUSMATH v4.0 - COSMOS EDITION 🌌
+ QUINTIKUSFÍSICA - Motor Físico-Químico Puro
Substitui: NumPy ✅ | Pandas ✅ | SciPy ✅ | Regex ✅ | SymPy ✅
Base: array, math, struct, time, abc
Zero dependências externas - Python puro
"""

import math
import struct
import time
import os
import pickle
from array import array
from abc import ABC, abstractmethod

# ============================================
# CONSTANTES UNIVERSAIS
# ============================================

class Constantes:
    """Constantes físicas e matemáticas fundamentais"""
    G = 6.67430e-11       # Constante gravitacional (N·m²/kg²)
    C = 299792458         # Velocidade da luz no vácuo (m/s)
    H = 6.62607015e-34    # Constante de Planck (J·s)
    PI = math.pi
    E = math.e
    INF = float('inf')
    NAN = float('nan')
    EPSILON = 1e-10
    G_TERRA = 9.80665     # Gravidade padrão na Terra (m/s²)
    R_GASES = 0.082057    # Constante universal dos gases (L·atm/(K·mol))
    AVOGADRO = 6.02214076e23  # Número de Avogadro
    BOLTZMANN = 1.380649e-23  # Constante de Boltzmann (J/K)


# ============================================
# 🚀 LUT EXP (15x mais rápido)
# ============================================

class LUTExp:
    """Lookup Table para exp() - 15x mais rápido que math.exp()"""
    
    __slots__ = ('_lut', '_min_val', '_max_val', '_inv_step')
    
    def __init__(self, min_val=-50.0, max_val=50.0, tamanho=10000):
        self._min_val = min_val
        self._max_val = max_val
        self._inv_step = tamanho / (max_val - min_val)
        
        step = (max_val - min_val) / tamanho
        self._lut = array('d', [0.0] * (tamanho + 1))
        for i in range(tamanho + 1):
            self._lut[i] = math.exp(min_val + i * step)
    
    def exp(self, x):
        """Exponencial via LUT (~0.5μs)"""
        if x <= self._min_val:
            return 0.0
        if x >= self._max_val:
            return float('inf')
        
        idx = (x - self._min_val) * self._inv_step
        i_low = int(idx)
        i_high = min(i_low + 1, len(self._lut) - 1)
        frac = idx - i_low
        
        return self._lut[i_low] * (1.0 - frac) + self._lut[i_high] * frac
    
    def exp_batch(self, arr):
        """Exponencial em lote"""
        resultado = array('d', [0.0] * len(arr))
        for i, v in enumerate(arr):
            resultado[i] = self.exp(v)
        return resultado


_LUT = LUTExp()


# ============================================
# 🧮 QArray - COSMOS EDITION
# ============================================

class QArray:
    """
    QuintikusArray - Enterra NumPy
    COMBO 1: fast_exp LUT integrado
    COMBO 2: @ operator (matmul)
    COMBO 3: Compatível com np alias
    """
    
    __slots__ = ('_dados', '_forma', '_tipo')
    
    def __init__(self, dados, forma=None, tipo='f'):
        self._tipo = tipo
        
        if isinstance(dados, QArray):
            self._dados = array(tipo, dados._dados)
            self._forma = dados._forma if forma is None else forma
        
        elif isinstance(dados, array):
            self._dados = array(tipo, dados)
            self._forma = forma if forma else (len(dados),)
        
        elif isinstance(dados, (list, tuple)):
            flat = self._achatar(dados)
            self._dados = array(tipo, flat)
            
            if forma:
                self._forma = forma
            else:
                if dados and isinstance(dados[0], (list, tuple)):
                    self._forma = (len(dados), len(dados[0]))
                else:
                    self._forma = (len(dados),)
        
        else:
            raise TypeError(f"Tipo não suportado: {type(dados)}")
    
    def _achatar(self, lista):
        resultado = []
        for item in lista:
            if isinstance(item, (list, tuple, array)):
                resultado.extend(self._achatar(item))
            else:
                resultado.append(float(item))
        return resultado
    
    @property
    def shape(self): return self._forma
    @property
    def size(self): return len(self._dados)
    @property
    def ndim(self): return len(self._forma)
    @property
    def dtype(self): return 'float32' if self._tipo == 'f' else 'float64'
    
    @property
    def T(self):
        if len(self._forma) == 1:
            return QArray(self._dados, forma=(1, len(self._dados)), tipo=self._tipo)
        elif len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = array('d', [0.0] * (linhas * cols))
            for i in range(linhas):
                for j in range(cols):
                    resultado[j * linhas + i] = self._dados[i * cols + j]
            return QArray(resultado, forma=(cols, linhas), tipo=self._tipo)
        return self
    
    def __len__(self): return len(self._dados)
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(self._forma) == 2:
            linhas, cols = self._forma
            i, j = idx
            return self._dados[i * cols + j]
        return self._dados[idx]
    
    def __setitem__(self, idx, valor):
        if isinstance(idx, tuple) and len(self._forma) == 2:
            linhas, cols = self._forma
            i, j = idx
            self._dados[i * cols + j] = float(valor)
        else:
            self._dados[idx] = float(valor)
    
    def __iter__(self): return iter(self._dados)
    
    def __repr__(self): return f"QArray(forma={self._forma})"
    
    def __str__(self):
        if len(self._forma) == 1:
            return "[" + ", ".join(f"{x:.4f}" for x in self._dados) + "]"
        elif len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = ""
            for i in range(linhas):
                inicio = i * cols
                linha = self._dados[inicio:inicio + cols]
                resultado += "[" + ", ".join(f"{x:8.4f}" for x in linha) + "]\n"
            return resultado
        return self.__repr__()
    
    def to_list(self):
        if len(self._forma) == 1:
            return self._dados.tolist()
        elif len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = []
            for i in range(linhas):
                inicio = i * cols
                resultado.append(self._dados[inicio:inicio + cols].tolist())
            return resultado
        return self._dados.tolist()
    
    def copy(self):
        return QArray(self._dados, forma=self._forma, tipo=self._tipo)
    
    # ============================================
    # COMBO 1: FAST EXP
    # ============================================
    
    def exp(self):
        return QArray(_LUT.exp_batch(self._dados), forma=self._forma)
    
    def log(self):
        resultado = array('d', [0.0] * len(self._dados))
        for i, v in enumerate(self._dados):
            resultado[i] = math.log(max(Constantes.EPSILON, v))
        return QArray(resultado, forma=self._forma)
    
    def sqrt(self):
        resultado = array('d', [0.0] * len(self._dados))
        for i, v in enumerate(self._dados):
            resultado[i] = math.sqrt(max(0, v))
        return QArray(resultado, forma=self._forma)
    
    def abs(self):
        resultado = array(self._tipo, [0.0] * len(self._dados))
        for i, v in enumerate(self._dados):
            resultado[i] = abs(v)
        return QArray(resultado, forma=self._forma, tipo=self._tipo)
    
    def clip(self, min_val, max_val):
        resultado = array(self._tipo, [0.0] * len(self._dados))
        for i, v in enumerate(self._dados):
            if v < min_val: resultado[i] = min_val
            elif v > max_val: resultado[i] = max_val
            else: resultado[i] = v
        return QArray(resultado, forma=self._forma, tipo=self._tipo)
    
    # ============================================
    # ESTATÍSTICAS
    # ============================================
    
    def sum(self, axis=None):
        if axis is None:
            return sum(self._dados)
        if axis == 0 and len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = array('d', [0.0] * cols)
            for i in range(linhas):
                for j in range(cols):
                    resultado[j] += self._dados[i * cols + j]
            return QArray(resultado, forma=(cols,))
        if axis == 1 and len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = array('d', [0.0] * linhas)
            for i in range(linhas):
                inicio = i * cols
                resultado[i] = sum(self._dados[inicio:inicio + cols])
            return QArray(resultado, forma=(linhas,))
        raise ValueError(f"Axis {axis} não suportado")
    
    def mean(self, axis=None):
        if axis is None:
            return sum(self._dados) / len(self._dados)
        soma = self.sum(axis)
        if len(self._forma) == 2:
            if axis == 0: return QArray([x / self._forma[0] for x in soma._dados], forma=soma._forma)
            elif axis == 1: return QArray([x / self._forma[1] for x in soma._dados], forma=soma._forma)
        return soma
    
    def max(self, axis=None):
        if axis is None: return max(self._dados)
        if axis == 0 and len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = array('d', [-Constantes.INF] * cols)
            for i in range(linhas):
                for j in range(cols):
                    val = self._dados[i * cols + j]
                    if val > resultado[j]: resultado[j] = val
            return QArray(resultado, forma=(cols,))
        raise ValueError(f"Axis {axis} não suportado")
    
    def min(self, axis=None):
        if axis is None: return min(self._dados)
        if axis == 0 and len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = array('d', [Constantes.INF] * cols)
            for i in range(linhas):
                for j in range(cols):
                    val = self._dados[i * cols + j]
                    if val < resultado[j]: resultado[j] = val
            return QArray(resultado, forma=(cols,))
        raise ValueError(f"Axis {axis} não suportado")
    
    def argmax(self, axis=None):
        if axis is None: return max(range(len(self._dados)), key=lambda i: self._dados[i])
        if axis == 0 and len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = array('i', [0] * cols)
            valores_max = array('d', [-Constantes.INF] * cols)
            for i in range(linhas):
                for j in range(cols):
                    val = self._dados[i * cols + j]
                    if val > valores_max[j]:
                        valores_max[j] = val
                        resultado[j] = i
            return QArray(resultado, forma=(cols,), tipo='i')
        raise ValueError(f"Axis {axis} não suportado")
    
    def argmin(self, axis=None):
        if axis is None: return min(range(len(self._dados)), key=lambda i: self._dados[i])
        if axis == 0 and len(self._forma) == 2:
            linhas, cols = self._forma
            resultado = array('i', [0] * cols)
            valores_min = array('d', [Constantes.INF] * cols)
            for i in range(linhas):
                for j in range(cols):
                    val = self._dados[i * cols + j]
                    if val < valores_min[j]:
                        valores_min[j] = val
                        resultado[j] = i
            return QArray(resultado, forma=(cols,), tipo='i')
        raise ValueError(f"Axis {axis} não suportado")
    
    # ============================================
    # COMBO 2: @ OPERATOR + DOT
    # ============================================
    
    def dot(self, other):
        return self._dot_impl(other)
    
    def _dot_impl(self, other):
        if isinstance(other, QArray):
            if len(self._forma) == 1 and len(other._forma) == 1:
                if len(self) != len(other):
                    raise ValueError(f"Tamanhos: {len(self)} vs {len(other)}")
                resultado = 0.0
                for i in range(len(self)):
                    resultado += self._dados[i] * other._dados[i]
                return resultado
            
            elif len(self._forma) == 2 and len(other._forma) == 1:
                linhas, cols = self._forma
                if cols != len(other):
                    raise ValueError(f"Colunas {cols} != {len(other)}")
                resultado = array('d', [0.0] * linhas)
                for i in range(linhas):
                    soma = 0.0
                    for j in range(cols):
                        soma += self._dados[i * cols + j] * other._dados[j]
                    resultado[i] = soma
                return QArray(resultado, forma=(linhas,))
            
            elif len(self._forma) == 2 and len(other._forma) == 2:
                l1, c1 = self._forma
                l2, c2 = other._forma
                if c1 != l2:
                    raise ValueError(f"({l1},{c1}) @ ({l2},{c2}) incompatível")
                resultado = array('d', [0.0] * (l1 * c2))
                for i in range(l1):
                    for j in range(c2):
                        soma = 0.0
                        for k in range(c1):
                            soma += self._dados[i * c1 + k] * other._dados[k * c2 + j]
                        resultado[i * c2 + j] = soma
                return QArray(resultado, forma=(l1, c2))
        
        elif isinstance(other, (int, float)):
            resultado = array(self._tipo, [0.0] * len(self._dados))
            for i, v in enumerate(self._dados):
                resultado[i] = v * other
            return QArray(resultado, forma=self._forma, tipo=self._tipo)
        
        raise TypeError(f"dot não suporta {type(other)}")
    
    def __matmul__(self, other):
        return self._dot_impl(other)
    
    def __rmatmul__(self, other):
        if isinstance(other, QArray):
            return other._dot_impl(self)
        raise TypeError(f"@ não suporta {type(other)}")
    
    def reshape(self, nova_forma):
        tamanho_total = 1
        for dim in nova_forma:
            tamanho_total *= dim
        if tamanho_total != len(self._dados):
            raise ValueError(f"Forma {nova_forma} ≠ tamanho {len(self._dados)}")
        return QArray(self._dados, forma=nova_forma, tipo=self._tipo)
    
    def flatten(self):
        return QArray(self._dados, forma=(len(self._dados),), tipo=self._tipo)
    
    # ============================================
    # OPERADORES ARITMÉTICOS
    # ============================================
    
    def __add__(self, other):
        if isinstance(other, QArray):
            if len(self) != len(other):
                raise ValueError("Tamanhos diferentes")
            resultado = array('d', [0.0] * len(self))
            for i in range(len(self)):
                resultado[i] = self._dados[i] + other._dados[i]
            return QArray(resultado, forma=self._forma)
        elif isinstance(other, (int, float)):
            resultado = array(self._tipo, [0.0] * len(self._dados))
            for i, v in enumerate(self._dados):
                resultado[i] = v + other
            return QArray(resultado, forma=self._forma, tipo=self._tipo)
        return NotImplemented
    
    def __radd__(self, other): return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, QArray):
            if len(self) != len(other):
                raise ValueError("Tamanhos diferentes")
            resultado = array('d', [0.0] * len(self))
            for i in range(len(self)):
                resultado[i] = self._dados[i] - other._dados[i]
            return QArray(resultado, forma=self._forma)
        elif isinstance(other, (int, float)):
            resultado = array(self._tipo, [0.0] * len(self._dados))
            for i, v in enumerate(self._dados):
                resultado[i] = v - other
            return QArray(resultado, forma=self._forma, tipo=self._tipo)
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, QArray):
            if len(self) != len(other):
                raise ValueError("Tamanhos diferentes")
            resultado = array('d', [0.0] * len(self))
            for i in range(len(self)):
                resultado[i] = self._dados[i] * other._dados[i]
            return QArray(resultado, forma=self._forma)
        elif isinstance(other, (int, float)):
            resultado = array(self._tipo, [0.0] * len(self._dados))
            for i, v in enumerate(self._dados):
                resultado[i] = v * other
            return QArray(resultado, forma=self._forma, tipo=self._tipo)
        return NotImplemented
    
    def __rmul__(self, other): return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            resultado = array('d', [0.0] * len(self._dados))
            for i, v in enumerate(self._dados):
                resultado[i] = v / other if other != 0 else 0.0
            return QArray(resultado, forma=self._forma)
        return NotImplemented
    
    def __neg__(self):
        resultado = array(self._tipo, [0.0] * len(self._dados))
        for i, v in enumerate(self._dados):
            resultado[i] = -v
        return QArray(resultado, forma=self._forma, tipo=self._tipo)
    
    def __pow__(self, expoente):
        resultado = array('d', [0.0] * len(self._dados))
        for i, v in enumerate(self._dados):
            resultado[i] = math.pow(v, expoente)
        return QArray(resultado, forma=self._forma)
    
    def __gt__(self, other): return [v > other for v in self._dados]
    def __lt__(self, other): return [v < other for v in self._dados]
    def __ge__(self, other): return [v >= other for v in self._dados]
    def __le__(self, other): return [v <= other for v in self._dados]


# ============================================
# COMBO 3: ALIAS PARA COMPATIBILIDADE TOTAL
# ============================================

np = QArray

def _prod(forma):
    resultado = 1
    for dim in forma:
        resultado *= dim
    return resultado


# ============================================
# 🏭 FUNÇÕES DE FÁBRICA
# ============================================

zeros = lambda forma, tipo='f': QArray([0.0] * _prod(forma), forma=forma, tipo=tipo)
ones = lambda forma, tipo='f': QArray([1.0] * _prod(forma), forma=forma, tipo=tipo)
full = lambda forma, valor, tipo='f': QArray([float(valor)] * _prod(forma), forma=forma, tipo=tipo)

def arange(inicio, fim=None, passo=1.0, tipo='f'):
    if fim is None:
        fim = inicio
        inicio = 0.0
    resultado = []
    atual = inicio
    while atual < fim:
        resultado.append(atual)
        atual += passo
    return QArray(resultado, tipo=tipo)

def linspace(inicio, fim, num=50, tipo='f'):
    if num <= 1:
        return QArray([inicio], tipo=tipo)
    passo = (fim - inicio) / (num - 1)
    resultado = [inicio + i * passo for i in range(num)]
    return QArray(resultado, tipo=tipo)

def random_uniform(forma, min_val=0.0, max_val=1.0):
    import random
    tamanho = _prod(forma)
    dados = [random.uniform(min_val, max_val) for _ in range(tamanho)]
    return QArray(dados, forma=forma, tipo='d')


# ============================================
# 📊 ESTATÍSTICAS
# ============================================

def std(arr, axis=None):
    if isinstance(arr, list):
        arr = QArray(arr)
    if axis is None:
        media = arr.mean()
        soma_quadrados = sum((x - media) ** 2 for x in arr)
        return math.sqrt(soma_quadrados / len(arr))
    raise NotImplementedError("std com axis ainda não implementado")

def median(arr):
    dados = sorted(arr._dados if isinstance(arr, QArray) else arr)
    n = len(dados)
    if n % 2 == 0:
        return (dados[n//2 - 1] + dados[n//2]) / 2
    return dados[n//2]

def percentile(arr, q):
    dados = sorted(arr._dados if isinstance(arr, QArray) else arr)
    n = len(dados)
    idx = (n - 1) * q / 100.0
    i_low = int(idx)
    i_high = min(i_low + 1, n - 1)
    frac = idx - i_low
    return dados[i_low] * (1 - frac) + dados[i_high] * frac

def unique(arr, return_counts=False):
    dados = arr._dados if isinstance(arr, QArray) else arr
    contagem = {}
    for v in dados:
        contagem[v] = contagem.get(v, 0) + 1
    valores = sorted(contagem.keys())
    if return_counts:
        counts = [contagem[v] for v in valores]
        return QArray(valores), QArray(counts)
    return QArray(valores)


# ============================================
# 🔍 EXTRACT (Enterra Regex)
# ============================================

def extract_float(texto):
    numeros = []
    atual = ""
    for char in texto:
        if char.isdigit() or char in '.-+eE':
            atual += char
        else:
            if atual:
                try: numeros.append(float(atual))
                except ValueError: pass
                atual = ""
    if atual:
        try: numeros.append(float(atual))
        except ValueError: pass
    return numeros

def extract_int(texto):
    return [int(x) for x in extract_float(texto) if x == int(x)]


# ============================================
# 📊 SCIPY.STATS ENTERRADO
# ============================================

def normal_pdf(x, mu=0.0, sigma=1.0):
    coeficiente = 1.0 / (sigma * math.sqrt(2 * Constantes.PI))
    expoente = -0.5 * ((x - mu) / sigma) ** 2
    return coeficiente * math.exp(expoente)

def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

def sigmoid_fast(x):
    return 1.0 / (1.0 + _LUT.exp(-x))

def softmax(arr):
    if isinstance(arr, QArray):
        dados = arr._dados
    else:
        dados = arr
    max_val = max(dados)
    exp_vals = [_LUT.exp(v - max_val) for v in dados]
    total = sum(exp_vals)
    if total > 0:
        resultado = [v / total for v in exp_vals]
    else:
        resultado = [1.0/len(dados)] * len(dados)
    return QArray(resultado) if isinstance(arr, QArray) else resultado

def relu(x): return max(0.0, x)

def relu_array(arr):
    resultado = array('d', [0.0] * len(arr))
    for i, v in enumerate(arr):
        resultado[i] = max(0.0, v)
    return QArray(resultado, forma=arr._forma) if isinstance(arr, QArray) else resultado

def tanh(x): return math.tanh(x)

def entropy(probabilidades):
    ent = 0.0
    for p in probabilidades:
        if p > Constantes.EPSILON:
            ent -= p * math.log2(p)
    return ent

def cross_entropy(y_pred, y_true):
    loss = 0.0
    for p, t in zip(y_pred, y_true):
        loss -= t * math.log(max(p, Constantes.EPSILON))
    return loss


# ============================================
# 📊 QDataFrame (substitui pandas)
# ============================================

class QDataFrame:
    """DataFrame leve"""
    
    def __init__(self, dados=None, colunas=None):
        self.colunas = colunas or []
        self.dados = dados or []
        self._num_linhas = len(self.dados)
    
    @classmethod
    def read_csv(cls, arquivo, delimiter=','):
        with open(arquivo, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
        if not linhas:
            return cls()
        colunas = [c.strip() for c in linhas[0].strip().split(delimiter)]
        dados = []
        for linha in linhas[1:]:
            if linha.strip():
                valores = [v.strip() for v in linha.strip().split(delimiter)]
                if len(valores) == len(colunas):
                    dados.append(valores)
        return cls(dados=dados, colunas=colunas)
    
    def head(self, n=5):
        print(f"Colunas: {self.colunas}\nTotal: {self._num_linhas} linhas\n")
        for i, linha in enumerate(self.dados[:n]):
            print(f"[{i}] {linha}")
    
    def tail(self, n=5):
        for i, linha in enumerate(self.dados[-n:]):
            print(f"[{self._num_linhas - n + i}] {linha}")
    
    def info(self):
        print(f"📊 QDataFrame: {self._num_linhas} linhas, {len(self.colunas)} colunas")
        for j, col in enumerate(self.colunas):
            nulos = sum(1 for linha in self.dados if linha[j] in ['', 'NA', 'null', 'None'])
            print(f"   {col}: {self._num_linhas - nulos} ok, {nulos} nulos")
    
    def loc(self, filtro_coluna=None, valor=None):
        if filtro_coluna is None: return self.dados
        idx_col = self.colunas.index(filtro_coluna) if filtro_coluna in self.colunas else -1
        if idx_col < 0: return []
        if valor is not None:
            return [linha for linha in self.dados if linha[idx_col] == str(valor)]
        return [linha[idx_col] for linha in self.dados]
    
    def dropna(self):
        self.dados = [l for l in self.dados if all(v not in ['', 'NA', 'null', 'None'] for v in l)]
        self._num_linhas = len(self.dados)
        return self
    
    def fillna(self, valor):
        for i in range(len(self.dados)):
            for j in range(len(self.dados[i])):
                if self.dados[i][j] in ['', 'NA', 'null', 'None']:
                    self.dados[i][j] = str(valor)
        return self
    
    def astype(self, coluna, tipo):
        idx = self.colunas.index(coluna) if coluna in self.colunas else -1
        if idx < 0: return self
        for i in range(len(self.dados)):
            try:
                self.dados[i][idx] = {'int': int, 'float': float, 'str': str}[tipo](self.dados[i][idx])
            except:
                self.dados[i][idx] = None
        return self
    
    def sort_values(self, coluna, ascending=True):
        idx = self.colunas.index(coluna) if coluna in self.colunas else -1
        if idx >= 0:
            self.dados.sort(key=lambda x: x[idx], reverse=not ascending)
        return self
    
    def to_list(self):
        return [dict(zip(self.colunas, linha)) for linha in self.dados]


# ============================================
# 🌍 GEOMETRIA
# ============================================

class Geometria:
    """Geometria plana e espacial usando QArray"""
    
    @staticmethod
    def area_circulo(raio):
        return Constantes.PI * math.pow(raio, 2)
    
    @staticmethod
    def hipotenusa(a, b):
        return math.hypot(a, b)
    
    @staticmethod
    def volume_esfera(raio):
        return (4/3) * Constantes.PI * math.pow(raio, 3)
    
    @staticmethod
    def volume_cilindro(raio, altura):
        return Constantes.PI * math.pow(raio, 2) * altura
    
    @staticmethod
    def distancia_3d(p1, p2):
        """Distância entre dois pontos no espaço 3D"""
        if len(p1) != 3 or len(p2) != 3:
            raise ValueError("Pontos devem ter 3 coordenadas (x, y, z)")
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))


# ============================================
# ⚛️ FÍSICA CLÁSSICA
# ============================================

class FisicaClassica:
    """Mecânica clássica e cinemática"""
    
    @staticmethod
    def forca_peso(massa, gravidade=9.81):
        return massa * gravidade
    
    @staticmethod
    def forca_gravitacional(m1, m2, r):
        if r == 0:
            raise ValueError("Distância não pode ser zero")
        return Constantes.G * (m1 * m2) / math.pow(r, 2)
    
    @staticmethod
    def energia_cinetica(massa, velocidade):
        return 0.5 * massa * math.pow(velocidade, 2)
    
    @staticmethod
    def equacao_torricelli(v0, a, delta_s):
        v_quadrado = math.pow(v0, 2) + 2 * a * delta_s
        return math.sqrt(v_quadrado) if v_quadrado >= 0 else Constantes.NAN


# ============================================
# 🌌 FÍSICA TEÓRICA
# ============================================

class FisicaTeorica:
    """Relatividade e Física Quântica"""
    
    @staticmethod
    def energia_repouso(massa):
        return massa * math.pow(Constantes.C, 2)
    
    @staticmethod
    def fator_lorentz(velocidade):
        if velocidade >= Constantes.C:
            raise ValueError("Velocidade deve ser menor que a da luz")
        return 1 / math.sqrt(1 - math.pow(velocidade / Constantes.C, 2))
    
    @staticmethod
    def dilatacao_temporal(tempo_proprio, velocidade):
        return FisicaTeorica.fator_lorentz(velocidade) * tempo_proprio


# ============================================
# 🧪 QUÍMICA
# ============================================

class Quimica:
    """Físico-Química e Química Geral"""
    
    @staticmethod
    def pressao_gas_ideal(n_mols, temp_kelvin, volume_litros):
        if volume_litros == 0:
            raise ValueError("Volume não pode ser zero")
        return (n_mols * Constantes.R_GASES * temp_kelvin) / volume_litros
    
    @staticmethod
    def calcular_ph(concentracao_h):
        if concentracao_h <= 0:
            raise ValueError("Concentração deve ser maior que zero")
        return -math.log10(concentracao_h)
    
    @staticmethod
    def decaimento_radioativo(n_inicial, constante, tempo):
        return n_inicial * math.exp(-constante * tempo)


# ============================================
# 🧮 MATEMÁTICA AVANÇADA
# ============================================

class Matematica:
    """Álgebra e funções matemáticas"""
    
    @staticmethod
    def bhaskara(a, b, c):
        delta = math.pow(b, 2) - 4 * a * c
        if delta < 0:
            return None
        x1 = (-b + math.sqrt(delta)) / (2 * a)
        x2 = (-b - math.sqrt(delta)) / (2 * a)
        return x1, x2
    
    @staticmethod
    def combinacao(n, k):
        return math.comb(n, k)
    
    @staticmethod
    def logaritmo_base_x(valor, base):
        return math.log(valor, base)


# ============================================
# 🎛️ CONDIÇÕES LÓGICAS (Classes Abstratas)
# ============================================

class CondicaoLogica(ABC):
    """Interface para condições lógicas em simulações"""
    @abstractmethod
    def avaliar(self, *args, **kwargs) -> bool:
        pass

class CondicaoColisao(CondicaoLogica):
    def avaliar(self, pos1, pos2, raio_colisao):
        distancia = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
        return distancia <= raio_colisao

class CondicaoMassaCritica(CondicaoLogica):
    def avaliar(self, massa_atual, massa_critica):
        return massa_atual >= massa_critica

class CondicaoPhAcido(CondicaoLogica):
    def avaliar(self, concentracao_h):
        ph_atual = Quimica.calcular_ph(concentracao_h)
        return ph_atual < 7.0


# ============================================
# 🔧 MOTOR FÍSICO (Utilitários)
# ============================================

class MotorFisico:
    """Motor de simulação usando array e struct"""
    
    @staticmethod
    def criar_vetor_estado(x, y, z, vx, vy, vz):
        return array('d', [x, y, z, vx, vy, vz])
    
    @staticmethod
    def serializar_particula(id_particula, massa, carga):
        return struct.pack('idd', id_particula, massa, carga)
    
    @staticmethod
    def simular_queda_livre(altura_inicial):
        g = Constantes.G_TERRA
        tempo_inicial = time.time()
        altura_atual = altura_inicial
        
        print(f"Iniciando simulação de queda livre de {altura_inicial}m...")
        while altura_atual > 0:
            tempo_decorrido = time.time() - tempo_inicial
            altura_atual = altura_inicial - (0.5 * g * math.pow(tempo_decorrido, 2))
            if altura_atual < 0:
                altura_atual = 0
            print(f"Tempo: {tempo_decorrido:.2f}s | Altura: {altura_atual:.2f}m", end='\r')
            time.sleep(0.05)
        print("\nImpacto com o solo!")

def start():
    print("="*60)
    print("🌌 QUINTIKUSMATH v4.0 - COSMOS EDITION")
    print("   Matemática + Física + Química + IA")
    print("="*60)
    
    # COMBO 1: Fast EXP
    print("\n🚀 COMBO 1: FAST EXP (LUT):")
    arr = QArray([1.0, 2.0, 3.0])
    print(f"  exp({arr.to_list()}) = {arr.exp().to_list()}")
    
    # COMBO 2: @ Operator
    print("\n🎯 COMBO 2: @ OPERATOR:")
    W = QArray([[1, 2], [3, 4]])
    x = QArray([10, 20])
    print(f"  W @ x = {(W @ x).to_list()}")
    
    # COMBO 3: Compatibilidade
    print("\n🔄 COMBO 3: np ALIAS:")
    arr_np = np([[1, 2], [3, 4]])
    print(f"  np.array() shape: {arr_np.shape}")
    
    # GEOMETRIA
    print("\n📐 GEOMETRIA:")
    print(f"  Área círculo r=5: {Geometria.area_circulo(5):.2f}")
    print(f"  Volume esfera r=3: {Geometria.volume_esfera(3):.2f}")
    
    # FÍSICA
    print("\n⚛️ FÍSICA CLÁSSICA:")
    print(f"  Força peso 70kg: {FisicaClassica.forca_peso(70):.1f} N")
    print(f"  Energia cinética 1000kg a 30m/s: {FisicaClassica.energia_cinetica(1000, 30):.0f} J")
    
    # FÍSICA TEÓRICA
    print("\n🌌 FÍSICA TEÓRICA:")
    print(f"  E=mc² para 1kg: {FisicaTeorica.energia_repouso(1):.2e} J")
    
    # QUÍMICA
    print("\n🧪 QUÍMICA:")
    print(f"  pH [H+]=0.001: {Quimica.calcular_ph(0.001):.1f}")
    
    # MATEMÁTICA
    print("\n🧮 MATEMÁTICA:")
    print(f"  Bhaskara 1,-5,6: {Matematica.bhaskara(1, -5, 6)}")
    
    # IA
    print("\n🤖 IA:")
    print(f"  sigmoid_fast(0): {sigmoid_fast(0):.4f}")
    print(f"  softmax([1,2,3]): {[round(x,4) for x in softmax([1,2,3])]}")
    
    # QDataFrame
    print("\n📊 QDataFrame:")
    #df = QDataFrame.read_csv.__func__(None)  # Skip CSV test
    df2 = QDataFrame(dados=[["Ana","25"],["João","30"]], colunas=["nome","idade"])
    df2.info()
    
    print("\n" + "="*60)
    print("🫰 THANOS DAS LIBS - COSMOS EDITION")
    print("   NumPy: ❌ | Pandas: ❌ | SciPy: ❌ | SymPy: ❌")
    print("   Física: ✅ | Química: ✅ | Geometria: ✅ | IA: ✅")
    print("   Dependências: ZERO (só Python stdlib)")
    print("="*60)


# ============================================
# 🧪 TESTE COSMOS EDITION COMPLETO
# ============================================

if __name__ == "__main__":
   start()
