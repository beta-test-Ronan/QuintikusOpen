import math
import time
import random
import pickle
from array import array

# ============================================
# ⚙️ MOTOR DE BAIXO NÍVEL (LINEAR TENSOR)
# ============================================
class QuintikusEngine:
    @staticmethod
    def leaky_relu(x):
        return x if x > 0 else 0.01 * x

    @staticmethod
    def d_leaky_relu(x):
        return 1.0 if x > 0 else 0.01

    @staticmethod
    def fast_softmax(logits_buffer):
        """Softmax otimizado para evitar estouro de float64"""
        max_val = max(logits_buffer)
        exp_sum = 0.0
        for i in range(len(logits_buffer)):
            logits_buffer[i] = math.exp(logits_buffer[i] - max_val)
            exp_sum += logits_buffer[i]
        for i in range(len(logits_buffer)):
            logits_buffer[i] /= exp_sum

# ============================================
# 🧱 DENSE LINEAR (MEMÓRIA CONTÍGUA)
# ============================================
class DenseLinear:
    def __init__(self, units, input_dim, activation='lrelu'):
        self.units = units
        self.input_dim = input_dim
        self.activation = activation
        
        # Pesos e Bias em Array Linear (Float32)
        limit = math.sqrt(2.0 / input_dim)
        self.w = array('f', [random.gauss(0, limit) for _ in range(input_dim * units)])
        self.b = array('f', [0.0] * units)
        
        # Buffer de Momentum (SGD+M)
        self.v_w = array('f', [0.0] * (input_dim * units))
        self.v_b = array('f', [0.0] * units)
        
        # Buffers Reutilizáveis (Zero Alocação no Loop)
        self.z = array('f', [0.0] * units)
        self.a = array('f', [0.0] * units)
        self.grad_in = array('f', [0.0] * input_dim)
        self.last_input = None # Referência, não cópia

    def forward(self, x_input):
        self.last_input = x_input
        # XW + B Linear
        for j in range(self.units):
            soma = self.b[j]
            for k in range(self.input_dim):
                # Indexação manual: k * units + j
                soma += x_input[k] * self.w[k * self.units + j]
            self.z[j] = soma
            
            if self.activation == 'lrelu':
                self.a[j] = QuintikusEngine.leaky_relu(soma)
            elif self.activation == 'softmax':
                # Softmax é aplicado no final do modelo
                self.a[j] = soma
        
        if self.activation == 'softmax':
            QuintikusEngine.fast_softmax(self.a)
        return self.a

    def backward(self, grad_out, lr, momentum=0.9):
        # grad_out é o erro vindo da camada seguinte
        # 1. Calcular dZ (in-place)
        for j in range(self.units):
            deriv = QuintikusEngine.d_leaky_relu(self.z[j]) if self.activation == 'lrelu' else 1.0
            dz = grad_out[j] * deriv
            
            # 2. Atualizar Pesos com Momentum (SGD+M)
            # dW = input * dz
            for k in range(self.input_dim):
                idx = k * self.units + j
                gw = self.last_input[k] * dz
                # Velocidade = m*v - lr*gw
                self.v_w[idx] = momentum * self.v_w[idx] - lr * gw
                self.w[idx] += self.v_w[idx]
            
            # 3. Atualizar Bias
            self.v_b[j] = momentum * self.v_b[j] - lr * dz
            self.b[j] += self.v_b[j]
            
            # 4. Propagar erro para camada anterior (grad_in)
            # dX = dZ * W
            for k in range(self.input_dim):
                if j == 0: self.grad_in[k] = 0.0 # Reset no primeiro neurônio
                self.grad_in[k] += dz * self.w[k * self.units + j]
                
        return self.grad_in

# ============================================
# 🧠 MODELO ULTRA-LEVE (MODO INFERÊNCIA/TREINO)
# ============================================
class QuintikusKery:
    def __init__(self, lr=0.01, momentum=0.9):
        self.layers = []
        self.lr = lr
        self.momentum = momentum

    def add(self, layer):
        self.layers.append(layer)

    def fit_online(self, x_single, y_single):
        """Treino Online: Batch Size = 1 (Ouro para J2)"""
        # Forward
        out = x_single
        for layer in self.layers:
            out = layer.forward(out)
        
        # Loss Grad (pred - real)
        error = array('f', [p - r for p, r in zip(out, y_single)])
        
        # Backward
        grad = error
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.lr, self.momentum)

    def predict(self, x_single):
        """Modo Inferência: Sem cálculos de gradiente"""
        out = x_single
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def summary(self):
        total = sum(len(l.w) + len(l.b) for l in self.layers)
        print(f"🧱 Quintikus Low-Level | Params: {total} | RAM: {total*4/1024:.2f}KB")

# ============================================
# 🚀 TESTE DE PERFORMANCE REAL (J2 STYLE)
# ============================================
if __name__ == "__main__":
    # Configuração de 1 Camada Oculta (Input 784 -> 8 -> Output 2)
    # Simulando um fragmento de imagem 28x28
    print("🔥 Iniciando Redução de Sobrecarga...")
    
    model = QuintikusKery(lr=0.01)
    model.add(DenseLinear(units=8, input_dim=784, activation='lrelu'))
    model.add(DenseLinear(units=2, input_dim=8, activation='softmax'))
    
    model.summary()
    
    # Simulação de Treino Online (Stream)
    input_fake = array('f', [random.random() for _ in range(784)])
    target_fake = array('f', [1.0, 0.0])
    
    start = time.time()
    for _ in range(100):
        model.fit_online(input_fake, target_fake)
    end = time.time()
    
    print(f"✅ 100 Passos de Treino Online em: {end-start:.4f}s")
    
    # Inferência Ultra Rápida
    pred = model.predict(input_fake)
    print(f"🎯 Predição Final: {list(pred)}")
