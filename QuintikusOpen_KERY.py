import math
import time
import random
import pickle
from array import array

# ============================================
# 📐 QMATH - Inteligência Matemática Purificada
# ============================================
class QMath:
    @staticmethod
    def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(min(x, 50), -50)))
    
    @staticmethod
    def d_sigmoid(x):
        s = QMath.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def relu(x): return max(0.0, x)

    @staticmethod
    def d_relu(x): return 1.0 if x > 0 else 0.0

    @staticmethod
    def softmax(arr):
        max_val = max(arr)
        exp_vals = [math.exp(v - max_val) for v in arr]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]

    @staticmethod
    def cce_loss(pred, real):
        """Categorical Cross-Entropy: A perda dos profissionais"""
        # Evita log(0) com um epsilon pequeno
        return -sum(r * math.log(max(p, 1e-15)) for p, r in zip(pred, real))

    @staticmethod
    def iguame(pred, real):
        """Métrica Quintikus de Precisão"""
        a, b = array('f', pred), array('f', real)
        dist = math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))
        return (1.0 / (1.0 + dist)) * 100

# ============================================
# 🧱 DENSE - Camada Profissional com Adam
# ============================================
class Dense:
    def __init__(self, unidades, ativacao='relu'):
        self.unidades = unidades
        self.ativacao = ativacao
        self.pesos, self.bias = None, None
        self.m_w, self.v_w, self.m_b, self.v_b = None, None, None, None
        self.t = 0

    def _inicializar(self, dim):
        # He Initialization (Ideal para ReLU)
        f = math.sqrt(2.0 / dim)
        self.pesos = [[random.gauss(0, f) for _ in range(self.unidades)] for _ in range(dim)]
        self.bias = [0.0] * self.unidades
        self.m_w = [[0.0] * self.unidades for _ in range(dim)]
        self.v_w = [[0.0] * self.unidades for _ in range(dim)]
        self.m_b, self.v_b = [0.0]*self.unidades, [0.0]*self.unidades

    def forward(self, X):
        if self.pesos is None: self._inicializar(len(X[0]))
        self.entrada = X
        # Z = X * W + B
        self.z = [[sum(X[i][k] * self.pesos[k][j] for k in range(len(X[0]))) + self.bias[j] 
                   for j in range(self.unidades)] for i in range(len(X))]
        
        if self.ativacao == 'relu': return [[QMath.relu(v) for v in l] for l in self.z]
        if self.ativacao == 'sigmoid': return [[QMath.sigmoid(v) for v in l] for l in self.z]
        if self.ativacao == 'softmax': return [QMath.softmax(l) for l in self.z]
        return self.z

    def backward(self, grad_saida, lr):
        batch_size, dim_in = len(self.entrada), len(self.entrada[0])
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        
        # O SEGREDO: Se a ativação for Softmax e a perda for Cross-Entropy, 
        # o gradiente local dz é simplificado. Se for ReLU/Sigmoid, aplicamos a derivada.
        dz = []
        for i in range(batch_size):
            linha = []
            for j in range(self.unidades):
                if self.ativacao == 'softmax':
                    # Atalho matemático: dL/dz para Softmax+CCE é pred - real
                    # O grad_saida já traz essa diferença.
                    deriv = 1.0 
                elif self.ativacao == 'relu':
                    deriv = QMath.d_relu(self.z[i][j])
                elif self.ativacao == 'sigmoid':
                    deriv = QMath.d_sigmoid(self.z[i][j])
                else:
                    deriv = 1.0
                linha.append(grad_saida[i][j] * deriv)
            dz.append(linha)

        # Cálculo do Gradiente para a camada anterior (Isolado)
        grad_X = [[sum(dz[i][j] * self.pesos[k][j] for j in range(self.unidades)) 
                   for k in range(dim_in)] for i in range(batch_size)]

        # Update Adam (Otimização de Segunda Ordem)
        for j in range(self.unidades):
            gb = sum(dz[i][j] for i in range(batch_size)) / batch_size
            self.m_b[j] = b1 * self.m_b[j] + (1-b1) * gb
            self.v_b[j] = b2 * self.v_b[j] + (1-b2) * (gb**2)
            self.bias[j] -= lr * (self.m_b[j]/(1-b1**self.t)) / (math.sqrt(self.v_b[j]/(1-b2**self.t)) + eps)
            for k in range(dim_in):
                gw = sum(self.entrada[i][k] * dz[i][j] for i in range(batch_size)) / batch_size
                self.m_w[k][j] = b1 * self.m_w[k][j] + (1-b1) * gw
                self.v_w[k][j] = b2 * self.v_w[k][j] + (1-b2) * (gw**2)
                self.pesos[k][j] -= lr * (self.m_w[k][j]/(1-b1**self.t)) / (math.sqrt(self.v_w[k][j]/(1-b2**self.t)) + eps)
        
        return grad_X

# ============================================
# 🧠 SEQUENCIAL - O Cérebro LEGO Estilo Keras
# ============================================
class Sequencial:
    def __init__(self, camadas=None):
        self.camadas = camadas or []
        self.lr = 0.001
        self.melhor_loss = float('inf')

    def treinar(self, X, y, epocas=100, monitorar=True):
        print(f"🧱 Treinando 'QuintikusKery' em {len(X)} amostras...")
        for ep in range(epocas):
            # Forward Pass
            pred = X
            for c in self.camadas: pred = c.forward(pred)
            
            # Cálculo do Gradiente Inicial (dL/dz da última camada)
            # Para Softmax + Cross-Entropy, o gradiente inicial é (pred - real)
            grad = [[pi - ti for pi, ti in zip(p, t)] for p, t in zip(pred, y)]
            
            # Cálculo da Perda (CCE para classificação)
            loss = sum(QMath.cce_loss(p, t) for p, t in zip(pred, y)) / len(X)
            
            # Backward Pass (Retropropagação)
            for c in reversed(self.camadas):
                grad = c.backward(grad, self.lr)
            
            if monitorar and loss < self.melhor_loss:
                self.melhor_loss = loss
                self.salvar("melhor_modelo.qkr", verbose=False)

            if ep % 50 == 0 or ep == epocas-1:
                print(f"🚀 Ep {ep:3d} | Loss: {loss:.6f}")

    def prever(self, X):
        for c in self.camadas: X = c.forward(X)
        return X

    def resumo(self):
        total = sum((len(c.pesos) * c.unidades) + c.unidades for c in self.camadas if c.pesos)
        print(f"\n🧱 [RESUMO LEGO] Params: {total:,} | RAM: {total*4/1024:.2f} KB\n")

    def salvar(self, arq, verbose=True):
        with open(arq, 'wb') as f: pickle.dump(self.camadas, f)
        if verbose: print(f"✅ Modelo Salvo: {arq}")

    @staticmethod
    def carregar(arq):
        with open(arq, 'rb') as f: camadas = pickle.load(f)
        return Sequencial(camadas)

# ============================================
# 📝 TEXTY & IMAGIX (Inalterados e Eficientes)
# ============================================
class Texty:
    def __init__(self): self.vocab = {}
    def fit(self, frases):
        palavras = set(" ".join(frases).lower().split())
        self.vocab = {p: i for i, p in enumerate(sorted(list(palavras)))}
    def vetorizar(self, frase):
        v = [0.0] * len(self.vocab)
        for p in frase.lower().split():
            if p in self.vocab: v[self.vocab[p]] = 1.0
        return array('f', v).tolist()

# ============================================
# 🚀 EXEMPLO DE USO (EXTREMAMENTE KERAS)
# ============================================
if __name__ == "__main__":
    # Dados de Exemplo
    textos = ["muito bom", "excelente", "muito ruim", "pessimo"]
    labels = [[1, 0], [1, 0], [0, 1], [0, 1]] # [Pos, Neg]
    
    tx = Texty(); tx.fit(textos)
    X = [tx.vetorizar(f) for f in textos]

    # Arquitetura LEGO pura
    modelo = Sequencial([
        Dense(8, ativacao='relu'),
        Dense(2, ativacao='softmax')
    ])

    modelo.lr = 0.01
    modelo.treinar(X, labels, epocas=200)
    
    # Teste
    teste = "muito bom"
    pred = modelo.prever([tx.vetorizar(teste)])[0]
    print(f"\nFrase: '{teste}' | Predição: {pred}")
    print(f"Sinergia Quintikus: {QMath.iguame(pred, [1,0]):.2f}%")
