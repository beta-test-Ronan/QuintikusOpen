import math
import time
import random
import struct
import os
from array import array

# ============================================
# 📐 QMATH - Kernel de Álgebra Linear INT8
# ============================================
class QMath:
    @staticmethod
    def dot_product_int8(vec_x, weights_int8, offset, size, scale):
        soma = 0.0
        for i in range(size):
            # Operação direto no registrador: Float32 * (Int8 * Scale)
            soma += vec_x[i] * (weights_int8[offset + i] * scale)
        return soma

    @staticmethod
    def fast_softmax(buffer):
        mx = max(buffer)
        s_exp = 0.0
        for i in range(len(buffer)):
            x = buffer[i] - mx
            val = 1.0 + x + (x*x) * 0.5 # Taylor
            buffer[i] = max(1e-6, val)
            s_exp += buffer[i]
        for i in range(len(buffer)):
            buffer[i] /= (s_exp + 1e-10)

# ============================================
# 🧱 DLINEAR-INT8 - O Motor de 20M de Pesos
# ============================================
class DlinearINT8:
    def __init__(self, units, input_dim, activation='relu', scale=0.01):
        self.units = units
        self.input_dim = input_dim
        self.activation = activation
        self.scale = scale
        
        # Alocação de memória contígua
        self.w_int8 = array('b', [0] * (input_dim * units))
        self.b = array('f', [0.0] * units)
        self.a = array('f', [0.0] * units)

    def inicializar_aleatorio(self):
        print(f"🎲 Inicializando {self.units * self.input_dim:,} pesos...")
        for i in range(len(self.w_int8)):
            self.w_int8[i] = random.randint(-64, 64)

    def forward(self, x):
        for j in range(self.units):
            offset = j * self.input_dim
            soma = self.b[j] + QMath.dot_product_int8(x, self.w_int8, offset, self.input_dim, self.scale)
            
            if self.activation == 'relu':
                self.a[j] = soma if soma > 0 else 0.01 * soma
            else:
                self.a[j] = soma
            
            # CPU Throttle (Respiro para o Android)
            if j % 50 == 0: time.sleep(0.0001)
                
        if self.activation == 'softmax':
            QMath.fast_softmax(self.a)
        return self.a

# ============================================
# 🧠 SEQUENCIAL VM - O Cérebro com Save/Load
# ============================================
class SequentialVM:
    def __init__(self):
        self.layers = []

    def add(self, layer): self.layers.append(layer)

    def predict(self, raw_data):
        # Conversão e Ajuste Cosmo Automático
        data = array('f', raw_data) if isinstance(raw_data, (list, array)) else array('f', [float(x) for x in raw_data])
        
        for l in self.layers:
            # Downsampling/Padding Automático via Cosmo Simplificado
            if len(data) != l.input_dim:
                data = self._cosmo_ajustar(data, l.input_dim)
            data = l.forward(data)
        return data

    def _cosmo_ajustar(self, data, target_dim):
        res = array('f', [0.0] * target_dim)
        janela = len(data) / target_dim
        for i in range(target_dim):
            inicio, fim = int(i * janela), int((i + 1) * janela)
            chunk = data[inicio:fim]
            res[i] = sum(chunk)/len(chunk) if chunk else 0.0
        return res

    # ----------------------------------------
    # 💾 SISTEMA DE PERSISTÊNCIA BINÁRIA (.QVM)
    # ----------------------------------------
    def save(self, filepath):
        print(f"💾 Salvando cérebro em {filepath}...")
        with open(filepath, 'wb') as f:
            # Header: [Magic(4b), NumLayers(I)]
            f.write(struct.pack('4sI', b'QK10', len(self.layers)))
            for l in self.layers:
                # Metadata da Camada: [Units, InDim, Ativ(0=rel, 1=soft), Scale]
                ativ_code = 1 if l.activation == 'softmax' else 0
                f.write(struct.pack('IIIf', l.units, l.input_dim, ativ_code, l.scale))
                # Dump binário direto dos pesos (INT8) e bias (Float32)
                l.w_int8.tofile(f)
                l.b.tofile(f)
        print(f"✅ Arquivo gerado: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

    @staticmethod
    def load(filepath):
        print(f"📦 Carregando cérebro binário {filepath}...")
        vm = SequentialVM()
        with open(filepath, 'rb') as f:
            magic, num_layers = struct.unpack('4sI', f.read(8))
            if magic != b'QK10': raise ValueError("Arquivo Inválido!")
            
            for _ in range(num_layers):
                units, indim, ativ_code, scale = struct.unpack('IIIf', f.read(16))
                ativ = 'softmax' if ativ_code == 1 else 'relu'
                
                # Reconstrói a camada e lê os dados direto para a memória
                l = DlinearINT8(units, indim, ativ, scale)
                l.w_int8.fromfile(f, units * indim)
                l.b.fromfile(f, units)
                vm.add(l)
        return vm

# ============================================
# 🚀 EXECUÇÃO FINAL: PERSISTÊNCIA DE 20M PESOS
# ============================================
if __name__ == "__main__":
    ARQUIVO = "brain_j2_20m.qvm"

   
    
    # 1. CRIAR E SALVAR (Simulando o fim de um treino)
    print("--- 🌌 ETAPA 1: CRIAÇÃO E SERIALIZAÇÃO ---")
    modelo_original = SequentialVM()
    # 200.000 entradas -> 100 neurônios = 20.000.000 conexões
    camada_monstro = DlinearINT8(units=100, input_dim=200000, activation='relu')
    camada_monstro.inicializar_aleatorio() # Simula pesos aprendidos
    modelo_original.add(camada_monstro)
    modelo_original.add(DlinearINT8(units=2, input_dim=100, activation='softmax'))
    
    modelo_original.save(ARQUIVO)
    
    # Limpa a memória RAM para provar o carregamento
    del modelo_original
    time.sleep(1)
    print("\n--- 📥 ETAPA 2: CARREGAMENTO E PREDICT ---")

    # 2. CARREGAR BINÁRIO
    start_load = time.perf_counter()
    novo_modelo = SequentialVM.load(ARQUIVO)
    end_load = time.perf_counter()
    print(f"⏱️ Tempo de carregamento: {end_load - start_load:.4f} s")
    
    # 3. PREDICT REAL
    dado_teste = array('f', [random.random() for _ in range(200000)])
    print(f"\n🧠 Fazendo predição com 20 milhões de parâmetros...")
    start_p = time.perf_counter()
    res = novo_modelo.predict(dado_teste)
    end_p = time.perf_counter()
    
    # 4. RESULTADO
    print(f"\n✅ Concluído em: {end_p - start_p:.4f} s")
    print(f"🎯 Decisão Final: {res.index(max(res))}")
    print(f"📊 Sinergia: {[round(v, 4) for v in res]}")
