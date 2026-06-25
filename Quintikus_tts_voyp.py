#!/usr/bin/env python3
import math
import struct
import subprocess
import random

SAMPLE_RATE = 16000

# ═══════════════════════════════════════════════════════════════
# ESTRUTURA FÍSICA (OSCILADOR + FILTRO + SATURAÇÃO)
# ═══════════════════════════════════════════════════════════════

class WaveOscillator:
    def __init__(self):
        self.phase = 0.0
    def tick(self, f0):
        self.phase += f0 / SAMPLE_RATE
        if self.phase >= 1.0: self.phase -= 1.0
        return 2.0 * self.phase - 1.0 # Dente de Serra (Rico em harmônicos)

class BiquadFilter:
    def __init__(self, freq, bw=80.0):
        self.x1 = self.x2 = self.z1 = self.z2 = 0.0
        self.update(freq, bw)
    def update(self, freq, bw):
        freq = max(80, min(freq, SAMPLE_RATE // 2 - 100))
        omega = 2 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) * math.sinh(math.log(2) / 2 * bw / freq * omega / math.sin(omega))
        b0, b2 = alpha, -alpha
        a0, a1, a2 = 1 + alpha, -2 * math.cos(omega), 1 - alpha
        self.b, self.a = (b0/a0, 0, b2/a0), (1.0, a1/a0, a2/a0)
    def process(self, x):
        y = (self.b[0]*x + self.b[1]*self.x1 + self.b[2]*self.x2 - self.a[1]*self.z1 - self.a[2]*self.z2)
        self.x2, self.x1, self.z2, self.z1 = self.x1, x, self.z1, y
        return y

# ═══════════════════════════════════════════════════════════════
# DICIONÁRIO INTEGRAL PT-BR (Sem cortes)
# ═══════════════════════════════════════════════════════════════
GEOMETRIAS_PTBR = {
    'a': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'e': [[500, 55, 1.0], [1800, 85, 0.7], [2500, 120, 0.4]],
    'i': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],
    'o': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'u': [[320, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],
    'á': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'à': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'â': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'ã': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'é': [[500, 55, 1.0], [1800, 85, 0.7], [2500, 120, 0.4]],
    'ê': [[500, 55, 1.0], [1800, 85, 0.7], [2500, 120, 0.4]],
    'í': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],
    'ó': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'ô': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'õ': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'ú': [[320, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],
    'p': [[160, 20, 1.2], [450, 80, 0.2], [900, 150, 0.1]],
    'b': [[180, 20, 1.2], [500, 80, 0.2], [1000, 150, 0.1]],
    't': [[180, 25, 1.1], [1500, 80, 0.3], [2500, 150, 0.1]],
    'd': [[190, 25, 1.1], [1500, 80, 0.3], [2500, 150, 0.1]],
    'k': [[170, 25, 1.1], [1200, 80, 0.3], [2000, 150, 0.1]],
    'g': [[180, 25, 1.1], [1200, 80, 0.3], [2000, 150, 0.1]],
    'f': [[2000, 300, 0.5], [3500, 500, 0.6], [5500, 800, 0.4]],
    'v': [[200, 40, 0.9], [1500, 80, 0.4], [2500, 150, 0.2]],
    's': [[4000, 500, 0.4], [6000, 800, 0.6], [7500, 1000, 0.3]],
    'z': [[300, 50, 0.8], [2500, 100, 0.5], [4000, 200, 0.3]],
    'm': [[250, 30, 1.0], [900, 60, 0.4], [2000, 120, 0.2]],
    'n': [[250, 35, 1.0], [1500, 70, 0.4], [2500, 130, 0.2]],
    'l': [[350, 45, 0.9], [1500, 75, 0.4], [2500, 150, 0.2]],
    'r': [[450, 40, 0.8], [1500, 80, 0.4], [2500, 150, 0.1]],
    'x': [[3000, 400, 0.4], [5000, 600, 0.5], [6500, 800, 0.3]],
    'j': [[300, 50, 0.8], [2000, 90, 0.5], [3000, 150, 0.3]],
    ' ': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
    ',': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
    '.': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
}

# ═══════════════════════════════════════════════════════════════
# DICIONÁRIO INTEGRAL EN (Sem cortes)
# ═══════════════════════════════════════════════════════════════
GEOMETRIAS_EN = {
    'a': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'e': [[500, 55, 1.0], [1800, 85, 0.7], [2500, 120, 0.4]],
    'i': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],
    'o': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'u': [[320, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],
    'â': [[600, 60, 1.0], [1200, 80, 0.7], [2300, 130, 0.4]],   # cup /ʌ/
    'î': [[400, 50, 1.0], [2000, 80, 0.7], [2800, 130, 0.4]],   # sit /ɪ/
    'ô': [[500, 55, 1.0], [850, 75, 0.7], [2400, 130, 0.3]],    # caught /ɔ/
    'ê': [[660, 60, 1.0], [1700, 80, 0.8], [2500, 130, 0.4]],   # cat /æ/
    'û': [[350, 45, 1.0], [1000, 70, 0.6], [2000, 120, 0.3]],   # foot /ʊ/
    'ë': [[500, 50, 0.5], [1500, 70, 0.4], [2500, 100, 0.2]],   # schwa
    'p': [[160, 20, 1.2], [450, 80, 0.2], [900, 150, 0.1]],
    'b': [[180, 20, 1.2], [500, 80, 0.2], [1000, 150, 0.1]],
    't': [[180, 25, 1.1], [1500, 80, 0.3], [2500, 150, 0.1]],
    'd': [[190, 25, 1.1], [1500, 80, 0.3], [2500, 150, 0.1]],
    'k': [[170, 25, 1.1], [1200, 80, 0.3], [2000, 150, 0.1]],
    'g': [[180, 25, 1.1], [1200, 80, 0.3], [2000, 150, 0.1]],
    'f': [[2000, 300, 0.5], [3500, 500, 0.6], [5500, 800, 0.4]],
    'v': [[200, 40, 0.9], [1500, 80, 0.4], [2500, 150, 0.2]],
    's': [[4000, 500, 0.4], [6000, 800, 0.6], [7500, 1000, 0.3]],
    'z': [[300, 50, 0.8], [2500, 100, 0.5], [4000, 200, 0.3]],
    'm': [[250, 30, 1.0], [900, 60, 0.4], [2000, 120, 0.2]],
    'n': [[250, 35, 1.0], [1500, 70, 0.4], [2500, 130, 0.2]],
    'l': [[350, 45, 0.9], [1500, 75, 0.4], [2500, 150, 0.2]],
    'r': [[450, 40, 0.8], [1500, 80, 0.4], [2500, 150, 0.1]],
    'h': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],               # h (sopro)
    'w': [[300, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],    # w
    'y': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],   # y
    'x': [[3000, 400, 0.4], [5000, 600, 0.5], [6500, 800, 0.3]], # sh
    'j': [[300, 50, 0.8], [2000, 90, 0.5], [3000, 150, 0.3]],   # zh
    'c': [[200, 25, 1.0], [3000, 400, 0.4], [5000, 600, 0.5]],  # ch
    ' ': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
}

# Identificação Global de Consoantes Surdas (Sopro)
SURDAS = {'p', 't', 'k', 'f', 's', 'x', 'c', 'h'}

# ═══════════════════════════════════════════════════════════════
# MOTOR DE SÍNTESE LINEAR UNIFICADO
# ═══════════════════════════════════════════════════════════════

def arquinet_stream(texto, idioma='pt'):
    banco = GEOMETRIAS_PTBR if idioma == 'pt' else GEOMETRIAS_EN
    print(f"🚀 Quintikus V10.0 [{idioma.upper()}] - Fluxo Linear Ativado")
    
    osc = WaveOscillator()
    filtros = [BiquadFilter(500), BiquadFilter(1500), BiquadFilter(2500)]
    f_atual = [500.0, 1500.0, 2500.0]

    # Driver de Áudio (ALSA)
    cmd = ['aplay', '-t', 'raw', '-f', 'S16_LE', '-c', '1', '-r', str(SAMPLE_RATE)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Parser EN para Digraphs
    if idioma == 'en':
        texto = texto.lower().replace("sh", "x").replace("ch", "c").replace("th", "t")

    for char in texto.lower():
        if char not in banco: continue
        geo_alvo = banco[char]
        
        # Duração Base e Inflexão
        dur_sec = 0.11 if idioma == 'en' else 0.14
        if char in ' ,.': dur_sec = 0.2
        dur_amostras = int(SAMPLE_RATE * dur_sec)

        chunk = []
        for i in range(dur_amostras):
            t = i / dur_amostras
            f0_delta = 125.0 * (1.0 - 0.06 * t) # Hertz Precisos

            # Fonte de Excitação (Voz ou Sopro)
            if char in SURDAS:
                fonte = random.uniform(-1.0, 1.0)
            elif char in ' ,.':
                fonte = 0.0
            else:
                fonte = osc.tick(f0_delta)

            # Processamento de Campo Harmônico
            saida = 0.0
            for j in range(3):
                # Deslize Linear Sem Gargalo (Inércia)
                f_atual[j] += (geo_alvo[j][0] - f_atual[j]) * 0.02
                filtros[j].update(f_atual[j], bw=geo_alvo[j][1])
                saida += filtros[j].process(fonte) * geo_alvo[j][2]

            # Saturação Vocal (Estruturação) + Cadência
            envelope = math.sin(math.pi * t)
            sample = math.tanh(saida * envelope * 1.5)
            chunk.append(int(sample * 30000))

        proc.stdin.write(struct.pack(f'<{len(chunk)}h', *chunk))

    proc.stdin.close()
    proc.wait()

if __name__ == '__main__':
    # Exemplo em Português (Com acentos e pontuação completa)
    arquinet_stream("Olá, eu sou o robô do Arquinét. O rato roeu a roupa do rei de Roma.", idioma='pt')
    
    # Exemplo em Inglês (Fluidez rítmica)
    arquinet_stream("I am a robot. This is a linear semantic test.", idioma='en')
