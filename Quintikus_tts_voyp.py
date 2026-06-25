#!/usr/bin/env python3
import math
import struct
import subprocess
import random

SAMPLE_RATE = 16000

# ═══════════════════════════════════════════════════════════════
# ESTRUTURA FÍSICA (SUAS CLASSES CALIBRADAS)
# ═══════════════════════════════════════════════════════════════

class WaveOscillator:
    def __init__(self, f0=130.0):
        self.f0 = f0
        self.phase = 0.0

    def tick(self, f0):
        self.phase += f0 / SAMPLE_RATE
        if self.phase >= 1.0: self.phase -= 1.0
        return 2.0 * self.phase - 1.0 # Sawtooth pura

class BiquadFilter:
    def __init__(self, freq, bw=80.0):
        self.x1 = self.x2 = self.z1 = self.z2 = 0.0
        self.update(freq, bw)

    def update(self, freq, bw):
        freq = max(100, min(freq, SAMPLE_RATE // 2 - 100))
        omega = 2 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) * math.sinh(math.log(2) / 2 * bw / freq * omega / math.sin(omega))
        b0, b2 = alpha, -alpha
        a0, a1, a2 = 1 + alpha, -2 * math.cos(omega), 1 - alpha
        self.b = (b0/a0, 0, b2/a0)
        self.a = (1.0, a1/a0, a2/a0)

    def process(self, x):
        y = (self.b[0]*x + self.b[1]*self.x1 + self.b[2]*self.x2 - self.a[1]*self.z1 - self.a[2]*self.z2)
        self.x2, self.x1 = self.x1, x
        self.z2, self.z1 = self.z1, y
        return y

# ═══════════════════════════════════════════════════════════════
# BANCOS DE DADOS GEOMÉTRICOS (PT-BR / EN)
# ═══════════════════════════════════════════════════════════════

GEOMETRIAS_PTBR = {
    'a': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'e': [[500, 55, 1.0], [1800, 85, 0.7], [2500, 120, 0.4]],
    'i': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],
    'o': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'u': [[320, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],
    's': [[4000, 500, 0.4], [6000, 800, 0.6], [7500, 1000, 0.3]],
    'p': [[160, 20, 1.2], [450, 80, 0.2], [900, 150, 0.1]],
    't': [[180, 25, 1.1], [1500, 80, 0.3], [2500, 150, 0.1]],
    'r': [[450, 40, 0.8], [1500, 80, 0.4], [2500, 150, 0.1]],
    'm': [[250, 30, 1.0], [900, 60, 0.4], [2000, 120, 0.2]],
    ' ': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
}

GEOMETRIAS_EN = {
    'a': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'e': [[500, 55, 1.0], [1800, 85, 0.7], [2500, 120, 0.4]],
    'i': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],
    'o': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'u': [[320, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],
    'h': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]], 
    's': [[4000, 500, 0.4], [6000, 800, 0.6], [7500, 1000, 0.3]],
    ' ': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
}

# Identificação de Consoantes Surdas (Apenas Sopro/Ruído)
SURDAS = {'s', 'p', 't', 'k', 'f', 'x', 'h'}

# ═══════════════════════════════════════════════════════════════
# MOTOR DE FLUXO GLOBAL
# ═══════════════════════════════════════════════════════════════

def arquinet_talk(texto, lang='pt'):
    banco = GEOMETRIAS_PTBR if lang == 'pt' else GEOMETRIAS_EN
    print(f"🌐 Arquinet [{lang.upper()}]: «{texto}»")
    
    osc = WaveOscillator(f0=125.0)
    filtros = [BiquadFilter(500), BiquadFilter(1500), BiquadFilter(2500)]
    f_atual = [500.0, 1500.0, 2500.0]

    cmd = ['aplay', '-t', 'raw', '-f', 'S16_LE', '-c', '1', '-r', str(SAMPLE_RATE)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for char in texto.lower():
        if char not in banco: char = ' '
        geo_alvo = banco[char]
        
        # Consoantes são mais curtas que vogais
        dur_sec = 0.08 if char in SURDAS or char in 'bcdfgj' else 0.14
        dur_amostras = int(SAMPLE_RATE * dur_sec)
        
        chunk = []
        for i in range(dur_amostras):
            t = i / dur_amostras
            # Micro-delta para naturalidade
            f0_dinamico = 125.0 * (1.0 - 0.05 * t)
            
            # Fonte: Ruído para surdas, Oscilador para o resto
            if char in SURDAS:
                fonte = random.uniform(-1.0, 1.0)
            elif char == ' ':
                fonte = 0.0
            else:
                fonte = osc.tick(f0_dinamico)

            # Processamento Harmônico Linear
            saida_filtros = 0.0
            for j in range(3):
                # Transição linear dos formantes
                f_atual[j] += (geo_alvo[j][0] - f_atual[j]) * 0.02
                filtros[j].update(f_atual[j], bw=geo_alvo[j][1])
                saida_filtros += filtros[j].process(fonte) * geo_alvo[j][2]

            # Estruturação Vocal: Envelope Silábico + Saturação Tanh
            envelope = math.sin(math.pi * t)
            sample = math.tanh(saida_filtros * envelope * 1.5)
            chunk.append(int(sample * 30000))
        
        proc.stdin.write(struct.pack(f'<{len(chunk)}h', *chunk))

    proc.stdin.close()
    proc.wait()

# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Teste em Português
    arquinet_talk("ola eu sou o robo do arquinet", lang='pt')
    
    # Teste em Inglês
    arquinet_talk("i am a robot", lang='en')
