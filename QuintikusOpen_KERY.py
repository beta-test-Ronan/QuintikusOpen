import math
import time
import random
import struct
from array import array

# ============================================
# 📐 QMATH - Núcleo de Álgebra Linear
# ============================================
class QMath:
    @staticmethod
    def dot_product(vec_a, weights, offset, size):
        """Cálculo de Produto Escalar Otimizado para Memória Contígua"""
        soma = 0.0
        for i in range(size):
            soma += vec_a[i] * weights[offset + i]
        return soma

    @staticmethod
    def fast_softmax(buffer):
        mx = max(buffer)
        s_exp = 0.0
        for i in range(len(buffer)):
            x = buffer[i] - mx
            val = 1.0 + x + (x*x) * 0.5 # Aproximação Taylor
            buffer[i] = max(1e-6, val) 
            s_exp += buffer[i]
        for i in range(len(buffer)):
            buffer[i] /= (s_exp + 1e-10)

# ============================================
# 🧱 DLINEAR - Tensor de Alta Dimensão (20k)
# ============================================
class Dlinear:
    """
    Camada de Álgebra Linear Pesada.
    Suporta de 100 a 20.000 dimensões.
    Conecta Tensores Brutos aos Neurônios de Decisão.
    """
    def __init__(self, units, input_dim, activation='relu'):
        self.units = units
        self.input_dim = input_dim
        self.activation = activation
        
        print(f"📦 Alocando Dlinear: {input_dim} -> {units} ({input_dim * units} conexões)")
        
        # Inicialização Xavier em Array Linear (Memória Contígua)
        limit = math.sqrt(6.0 / (input_dim + units))
        # O peso é um Tensor de Rank-2 (Matriz) achatado
        self.w = array('f', [random.uniform(-limit, limit) for _ in range(input_dim * units)])
        self.b = array('f', [0.0] * units)
        self.a = array('f', [0.0] * units)

    def forward(self, x):
        """Operação de Tensor: Y = XW + B"""
        # Se a entrada não bater, o Cosmo deve ter sido chamado antes
        for j in range(self.units):
            # Calcula o neurônio J usando um deslocamento no Tensor de Pesos
            offset = j * self.input_dim
            soma = self.b[j] + QMath.dot_product(x, self.w, offset, self.input_dim)
            
            # Funções de Ativação de Precisão
            if self.activation == 'relu':
                self.a[j] = soma if soma > 0 else 0.01 * soma
            else:
                self.a[j] = soma
                
        if self.activation == 'softmax':
            QMath.fast_softmax(self.a)
        return self.a

# ============================================
# 🔭 COSMO & UNIVERCY - O Suporte Vital
# ============================================
class Cosmo:
    @staticmethod
    def ajustar(data, target_dim):
        in_dim = len(data)
        if in_dim == target_dim: return data
        res = array('f', [0.0] * target_dim)
        janela = in_dim / target_dim
        for i in range(target_dim):
            inicio, fim = int(i * janela), int((i + 1) * janela)
            trecho = data[inicio:fim]
            res[i] = sum(trecho)/len(trecho) if trecho else 0.0
        return res

class Univercy:
    @staticmethod
    def to_array(data):
        if isinstance(data, str): return array('f', [float(ord(c))/255.0 for c in data])
        if isinstance(data, (list, tuple, array)):
            mx = max(data) if data and max(data) > 0 else 1.0
            return array('f', [float(x)/mx for x in data])
        return array('f', [float(data)])

# ============================================
# 🧠 SEQUENCIAL VM - ENGINE FINAL
# ============================================
class SequentialVM:
    def __init__(self):
        self.layers = []

    def add(self, layer): self.layers.append(layer)

    def predict(self, raw_data):
        data = Univercy.to_array(raw_data)
        
        for i, l in enumerate(self.layers):
            # Ajuste Cosmico Automático entre dimensões (até 20k)
            target_in = l.input_dim if hasattr(l, 'input_dim') else l.units
            data = Cosmo.ajustar(data, target_in)
            data = l.forward(data)
        
        return data

# ============================================
# 🚀 TESTE DE ESTRESSE: 20.000 DIMENSÕES
# ============================================
if __name__ == "__main__":
    print("--- 🌌 QUINTIKUS-VM 9.0: DLINEAR ACTIVATED ---")
    
    # Criando um monstro de processamento no J2
    # Entrada de 20.000 dimensões (ex: imagem HD ou áudio bruto)
    # Reduzindo para 100 neurônios de características
    # Terminando em 2 classes de decisão
    
    vm = SequentialVM()
    vm.add(Dlinear(units=100, input_dim=20000, activation='relu'))
    vm.add(Dlinear(units=2, input_dim=100, activation='softmax'))
    
    # Simulação de Dado Gigante (20.000 pontos)
    print("\n📡 Gerando sinal de 20.000 dimensões...")
    dado_gigante = [random.random() for _ in range(20000)]
    
    print("🧠 Processando Álgebra Linear Pesada...")
    t_inicio = time.perf_counter()
    resultado = vm.predict(dado_gigante)
    t_fim = time.perf_counter()
    
    # Resultados
    ms = (t_fim - t_inicio) * 1000
    print(f"\n✅ Concluído em: {ms:.2f} ms")
    print(f"🎯 Decisão: {resultado.index(max(resultado))}")
    print(f"📊 Sinergia de Saída: {[round(v, 4) for v in resultado]}")
    
    # Memória total ocupada pelos pesos
    params = (20000 * 100) + 100 + (100 * 2) + 2
    print(f"\n📊 Total de Parâmetros: {params:,}")
    print(f"💾 RAM Estimada de Pesos: {params * 4 / 1024 / 1024:.2f} MB")
