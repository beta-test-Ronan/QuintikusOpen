import math
import time
import random
import pickle
from array import array

# ============================================
# 📐 QMATH - Inteligência Matemática
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
    def iguame(pred, real):
        """Métrica Quintikus: 100% = Identidade Absoluta"""
        a, b = array('f', pred), array('f', real)
        dist = math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))
        return (1.0 / (1.0 + dist)) * 100

# ============================================
# 📝 TEXTY & 🖼️ IMAGIX - Tradutores de Hardware
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

class Imagix:
    @staticmethod
    def normalizar(pixels):
        m = max(pixels) if pixels else 1
        return [x / m for x in pixels]
    
    @staticmethod
    def ascii_art(pixels, largura):
        chars = " .:-=+*#%@"
        for i, p in enumerate(pixels):
            idx = int(p * (len(chars)-1))
            print(chars[idx], end=" " if (i + 1) % largura != 0 else "\n")

# ============================================
# 🧱 DENSE - Gradiente Isolado + Adam
# ============================================
class Dense:
    def __init__(self, unidades, ativacao='relu'):
        self.unidades = unidades
        self.ativacao = ativacao
        self.pesos, self.bias = None, None
        self.m_w, self.v_w, self.m_b, self.v_b = None, None, None, None
        self.t = 0

    def _inicializar(self, dim):
        f = math.sqrt(2.0 / dim)
        self.pesos = [[random.gauss(0, f) for _ in range(self.unidades)] for _ in range(dim)]
        self.bias = [0.0] * self.unidades
        self.m_w = [[0.0] * self.unidades for _ in range(dim)]
        self.v_w = [[0.0] * self.unidades for _ in range(dim)]
        self.m_b, self.v_b = [0.0]*self.unidades, [0.0]*self.unidades

    def forward(self, X):
        if self.pesos is None: self._inicializar(len(X[0]))
        self.entrada = X
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
        
        dz = [[grad_saida[i][j] * (QMath.d_relu(self.z[i][j]) if self.ativacao == 'relu' else (QMath.d_sigmoid(self.z[i][j]) if self.ativacao == 'sigmoid' else 1.0))
               for j in range(self.unidades)] for i in range(batch_size)]

        # ISOLAMENTO: Calculamos o erro da camada anterior ANTES de mudar os pesos
        grad_X = [[sum(dz[i][j] * self.pesos[k][j] for j in range(self.unidades)) 
                   for k in range(dim_in)] for i in range(batch_size)]

        # ATUALIZAÇÃO ADAM
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
# 🧠 SEQUENCIAL - O Cérebro LEGO
# ============================================
class Sequencial:
    def __init__(self, camadas=None):
        self.camadas = camadas or []
        self.lr = 0.001
        self.melhor_loss = float('inf')

    def treinar(self, X, y, epocas=100, monitorar=True):
        print(f"🚀 Sinergia em {len(X)} amostras...")
        for ep in range(epocas):
            pred = X
            for c in self.camadas: pred = c.forward(pred)
            grad = [[pi - ti for pi, ti in zip(p, t)] for p, t in zip(pred, y)]
            loss = sum(sum((pi - ti)**2 for pi, ti in zip(p, t)) for p, t in zip(pred, y)) / len(X)
            for c in reversed(self.camadas): grad = c.backward(grad, self.lr)
            if monitorar and loss < self.melhor_loss:
                self.melhor_loss = loss
                self.salvar("melhor_modelo.qkr", verbose=False)
            if ep % 20 == 0 or ep == epocas-1:
                prog = int(20 * ep / (epocas-1)) if epocas > 1 else 20
                print(f"⚡ Ep {ep:3d} |{'█'*prog:20}| Loss: {loss:.6f}")

    def prever(self, X):
        for c in self.camadas: X = c.forward(X)
        return X

    def salvar(self, arq, verbose=True):
        with open(arq, 'wb') as f: pickle.dump(self.camadas, f)
        if verbose: print(f"✅ Salvo: {arq}")

    def resumo(self):
        total = sum((len(c.pesos) * c.unidades) + c.unidades for c in self.camadas if c.pesos)
        print("\n" + "🧱" * 15)
        print(f"RESUMO TÉCNICO: {total:,} Params")
        print(f"RAM Estimada: {total*4/1024:.2f} KB")
        print("🧱" * 15 + "\n")

# ============================================
# 🏗️ QUINTIKUS-CLI - Gerador por Sinergia
# ============================================
class QuintikusCLI:
    def montar(self):
        print("\n" + "🧱"*20)
        print("   QUINTIKUSKERY CLI BUILDER")
        print("🧱"*20)
        tipo = input("\n[1] Visão (Imagem) [2] Texto: ")
        dim = int(input("Dimensão de Entrada: "))
        n = int(input("Quantos blocos ocultos? "))
        camadas = []
        for i in range(n):
            u = int(input(f" Neurônios B{i+1}: "))
            a = input(" Ativação (relu/sigmoid): ").lower()
            camadas.append(Dense(u, ativacao=a))
        out = int(input("Classes de Saída: "))
        camadas.append(Dense(out, ativacao='softmax'))
        
        modelo = Sequencial(camadas)
        print("\n✨ SINERGIA MONTADA ✨")
        modelo.resumo()
        return modelo

# ============================================
# 🚀 EXECUÇÃO FINAL
# ============================================
if __name__ == "__main__":
    # ========================================================
    # EXEMPLO: CLASSIFICADOR DE SENTIMENTOS (ESTILO KERAS)
    # ========================================================

    # 1. PREPARAR OS DADOS (Simulação de Textos)
    # "bom", "amei" -> Positivo [1, 0]
    # "ruim", "odio" -> Negativo [0, 1]
    textos = ["muito bom", "eu amei", "muito ruim", "que odio"]
    labels = [[1, 0], [1, 0], [0, 1], [0, 1]]

    # Tradutor Texty para transformar palavras em números
    processador = Texty()
    processador.fit(textos)
    X = [processador.vetorizar(f) for f in textos]

    # --------------------------------------------------------
    # 2. DEFINIR A ARQUITETURA (Igual ao Keras)
    # --------------------------------------------------------
    modelo = Sequencial([
        Dense(12, ativacao='relu'),    # Camada Oculta 1
        Dense(8, ativacao='relu'),     # Camada Oculta 2
        Dense(2, ativacao='softmax')   # Camada de Saída (Positivo/Negativo)
    ])

    # 3. CONFIGURAR O APRENDIZADO (Igual ao compile)
    modelo.lr = 0.01

    # --------------------------------------------------------
    # 4. TREINAR O MODELO (Igual ao model.fit)
    # --------------------------------------------------------
    modelo.treinar(X, labels, epocas=300)

    # 5. MOSTRAR RESUMO TÉCNICO
    modelo.resumo()

    # --------------------------------------------------------
    # 6. FAZER PREDIÇÕES (Igual ao model.predict)
    # --------------------------------------------------------
    nova_frase = "achei muito bom"
    vetor_teste = processador.vetorizar(nova_frase)

    # Previsão retorna as probabilidades
    probabilidades = modelo.prever([vetor_teste])[0]

    # Pega o índice da maior probabilidade (argmax)
    classe = probabilidades.index(max(probabilidades))
    resultado = "Positivo" if classe == 0 else "Negativo"

    print(f"Frase: '{nova_frase}'")
    print(f"Resultado: {resultado}")
    print(f"Confiança: {QMath.iguame(probabilidades, [1,0] if classe==0 else [0,1]):.2f}%")

    # --------------------------------------------------------
    # 7. SALVAR O CÉREBRO (Igual ao model.save)
    # --------------------------------------------------------
    modelo.salvar("classificador_sentimentos.qkr")

    # Para carregar depois em outro arquivo:
    # novo_cerebro = Sequencial.carregar("classificador_sentimentos.qkr")
