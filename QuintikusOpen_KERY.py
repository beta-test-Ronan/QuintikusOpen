import math
import time
import random
import struct
from array import array

# ============================================
# ⚙️ QVM 8.1 - NÚCLEO DE OPERAÇÕES
# ============================================
class QVM:
    @staticmethod
    def fast_softmax(buffer):
        """Softmax com Clamp de segurança para hardware antigo"""
        mx = max(buffer)
        s_exp = 0.0
        for i in range(len(buffer)):
            x = buffer[i] - mx
            # Taylor estável: impede valores negativos
            val = 1.0 + x + (x*x) * 0.5
            buffer[i] = max(1e-6, val) 
            s_exp += buffer[i]
        for i in range(len(buffer)):
            buffer[i] /= (s_exp + 1e-10)

# ============================================
# 🌌 UNIVERCY - O TRADUTOR UNIVERSAL
# ============================================
class Univercy:
    @staticmethod
    def to_array(data):
        """Converte qualquer dado para o formato de hardware array('f')"""
        if isinstance(data, str):
            # Normalização ASCII para 0.0 - 1.0
            return array('f', [float(ord(c))/255.0 for c in data])
        if isinstance(data, (list, tuple, array)):
            mx = max(data) if data and max(data) > 0 else 1.0
            return array('f', [float(x)/mx for x in data])
        return array('f', [float(data)])

# ============================================
# 🔭 COSMO - O AJUSTADOR GEOMÉTRICO (POOLING)
# ============================================
class Cosmo:
    """Ajusta dimensões usando Mean Pooling para não destruir informação"""
    @staticmethod
    def ajustar(data, target_dim):
        in_dim = len(data)
        if in_dim == target_dim: return data, 1.0
        
        res = array('f', [0.0] * target_dim)
        # Score de Integridade: mede a distorção do dado
        integridade = min(in_dim, target_dim) / max(in_dim, target_dim)
        
        if in_dim > target_dim:
            # MEAN POOLING: Amalgama vizinhos para preservar o padrão
            janela = in_dim / target_dim
            for i in range(target_dim):
                inicio = int(i * janela)
                fim = int((i + 1) * janela)
                trecho = data[inicio:fim]
                if trecho:
                    res[i] = sum(trecho) / len(trecho)
        else:
            # EXPANSÃO (Zero Padding lateral)
            for i in range(min(in_dim, target_dim)):
                res[i] = data[i]
                
        return res, integridade

# ============================================
# 🧠 BIOLAYER - MEMÓRIA TEMPORAL (DECAY)
# ============================================
class BioLayer:
    """Simula neurônios com rastro de memória (Resiliência Temporal)"""
    def __init__(self, units, decay=0.8):
        self.units = units
        self.decay = decay 
        self.v = array('f', [-70.0] * units)
        self.memoria = array('f', [0.0] * units)

    def forward(self, current_in):
        for i in range(self.units):
            # Integração LIF simplificada
            self.v[i] += (current_in[i] * 0.5) - (self.v[i] + 70.0) * 0.1
            # Decaimento do rastro (Contexto)
            self.memoria[i] *= self.decay
            if self.v[i] >= -50.0:
                self.v[i] = -75.0
                self.memoria[i] = 1.0 # Disparo (Spike)
        return self.memoria

# ============================================
# 🧱 DENSE-VM - NEURÔNIOS DE PRECISÃO
# ============================================
class DenseVM:
    def __init__(self, units, activation='relu', input_dim=None):
        self.units = units
        self.activation = activation
        self.input_dim = input_dim
        self.w = None
        self.b = None
        self.a = array('f', [0.0] * units)
        if input_dim: self._build(input_dim)

    def _build(self, dim):
        self.input_dim = dim
        limit = math.sqrt(6.0 / (dim + self.units))
        self.w = array('f', [random.uniform(-limit, limit) for _ in range(dim * self.units)])
        self.b = array('f', [0.0] * self.units)

    def forward(self, x):
        if self.w is None: self._build(len(x))
        for j in range(self.units):
            base, soma = j * self.input_dim, self.b[j]
            for k in range(self.input_dim):
                soma += x[k] * self.w[base + k]
            
            if self.activation == 'relu': self.a[j] = soma if soma > 0 else 0.01 * soma
            else: self.a[j] = soma
            
        if self.activation == 'softmax': QVM.fast_softmax(self.a)
        return self.a

# ============================================
# 🧠 SEQUENCIAL VM - ENGINE DE RESILIÊNCIA
# ============================================
class SequentialVM:
    def __init__(self):
        self.layers = []
        self.confianca_total = 1.0

    def add(self, layer): self.layers.append(layer)

    def predict(self, raw_data):
        # 1. Tradução Universal
        data = Univercy.to_array(raw_data)
        
        # 2. Ajuste Inicial (Entrada -> Primeira Camada)
        target_in = self.layers[0].units if hasattr(self.layers[0], 'units') else self.layers[0].input_dim
        data, integridade = Cosmo.ajustar(data, target_in)
        self.confianca_total = integridade
        
        # 3. Fluxo de Camadas
        for i, l in enumerate(self.layers):
            data = l.forward(data)
            # Ajuste entre camadas (Saída -> Próxima Entrada)
            if i + 1 < len(self.layers):
                nxt = self.layers[i+1]
                target_nxt = nxt.input_dim if (hasattr(nxt, 'input_dim') and nxt.input_dim) else nxt.units
                data, _ = Cosmo.ajustar(data, target_nxt)
        
        return data

# ============================================
# 🚀 EXECUÇÃO: O SALVADOR DE MÁQUINAS
# ============================================
if __name__ == "__main__":
    print("🧱 QUINTIKUS-VM 8.1: Resiliência Total Ativa\n")
    
    # Criando modelo resiliente
    vm = SequentialVM()
    vm.add(BioLayer(8, decay=0.85))     # Camada com Memória Curta
    vm.add(DenseVM(2, activation='softmax')) # Decisão Final
    
    # Testando com dados de tamanhos variados e "sujos"
    entradas = [
        "SINAL_OK",                # Curto
        "ERRO_CRITICO_NO_SISTEMA",  # Longo (O Cosmo fará Pooling)
        [0.1, 0.9, 0.4]            # Lista de sensores
    ]
    
    for d in entradas:
        start = time.perf_counter()
        res = vm.predict(d)
        end = time.perf_counter()
        
        conf = vm.confianca_total * 100
        decisao = res.index(max(res))
        
        print(f"📥 Input: {str(d):<25}")
        print(f"📐 Integridade: {conf:>5.1f}% | Latência: {(end-start)*1000:.3f}ms")
        print(f"🧠 Decisão: Classe {decisao} | Sinergia: {[round(v,3) for v in res]}")
        print("-" * 60)

    print("\n✅ Processamento concluído com Sinergia Cosmica.")
