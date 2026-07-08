#!/usr/bin/env python3
import math
import struct
import random
import pyaudio # Troca de subprocess para pyaudio para rodar no Windows

SAMPLE_RATE = 16000 

# ═══════════════════════════════════════════════════════════════
# FÍSICA VGS (VIBRAÇÃO GEOMÉTRICA SONORA)
# ═══════════════════════════════════════════════════════════════

class VGSPulse:
    """ Fonte de Ação e Reação: Simula o choque do ar nas pregas vocais """
    def __init__(self):
        self.phase = 0.0
    def tick(self, f0):
        self.phase += f0 / SAMPLE_RATE
        if self.phase >= 1.0: self.phase -= 1.0
        # Geometria de Colisão: Subida (Ação) e Queda Súbita (Reação)
        return math.sin(math.pi * (self.phase / 0.8)) if self.phase < 0.8 else 0.0

class VGSChamber:
    """ Câmaras de Ressonância Geométrica """
    def __init__(self, freq, bw=80.0):
        self.x1 = self.x2 = self.z1 = self.z2 = 0.0
        self.update(freq, bw)
    def update(self, freq, bw):
        freq = max(80, min(freq, SAMPLE_RATE // 2 - 100))
        omega = 2 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) * math.sinh(math.log(2) / 2 * bw / freq * omega / math.sin(omega))
        a0 = 1 + alpha
        self.b = (alpha / a0, 0, -alpha / a0)
        self.a = (1.0, -2 * math.cos(omega) / a0, (1 - alpha) / a0)
    def process(self, x):
        y = (self.b[0]*x + self.b[2]*self.x2 - self.a[1]*self.z1 - self.a[2]*self.z2)
        self.x2, self.x1, self.z2, self.z1 = self.x1, x, self.z1, y
        return y

# ═══════════════════════════════════════════════════════════════
# DICIONÁRIO INTEGRAL PT-BR (CONSOLIDADO)
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
    'd': [[190, 25, 1.1], [1500, 80, 0.5], [2500, 150, 0.1]],
    'k': [[170, 25, 1.1], [1200, 80, 0.3], [2000, 150, 0.1]],
    'g': [[180, 25, 1.1], [1200, 80, 0.3], [2000, 150, 0.1]],
    'f': [[2000, 300, 0.5], [3500, 500, 0.6], [5500, 800, 0.4]],
    'v': [[200, 40, 0.9], [1500, 80, 0.4], [2500, 150, 0.2]],
    's': [[4000, 500, 0.4], [6000, 800, 0.6], [7500, 1000, 0.3]],
    'z': [[300, 50, 0.8], [2500, 100, 0.5], [4000, 200, 0.3]],
    'm': [[250, 30, 1.0], [900, 60, 0.4], [2000, 120, 0.2]],
    'n': [[250, 35, 1.0], [1500, 70, 0.4], [2500, 130, 0.2]],
    'l': [[350, 45, 0.9], [1500, 75, 0.4], [2500, 150, 0.2]],
    'r': [[400, 40, 0.5], [1200, 80, 0.2], [2000, 150, 0.1]],
    'x': [[3000, 400, 0.4], [5000, 600, 0.5], [6500, 800, 0.3]],
    'j': [[300, 50, 0.8], [2000, 90, 0.5], [3000, 150, 0.3]],
    ' ': [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]],
    ',': [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]],
    '.': [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]],
}

SURDAS = {'p', 't', 'k', 'f', 's', 'x', 'c', 'h'}

# ═══════════════════════════════════════════════════════════════
# MOTOR DE SÍNTESE VGS (VERSÃO WINDOWS)
# ═══════════════════════════════════════════════════════════════

def arquinet_vgs_stream(texto):
    print(f"🚀 Quintikus TTS VGS [Windows Ready] - Texto: {texto}")
    
    # Inicializa PyAudio em vez de subprocess
    p_audio = pyaudio.PyAudio()
    stream = p_audio.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=SAMPLE_RATE,
                          output=True)

    pulse = VGSPulse()
    chambers = [VGSChamber(500, 80) for _ in range(3)]
    picos_atuais = [500.0, 1500.0, 2500.0]

    for char in texto.lower():
        if char not in GEOMETRIAS_PTBR: continue
        geo_alvo = GEOMETRIAS_PTBR[char]
        
        dur_sec = 0.14 if char in 'aeiouáàâãéêíóôõú' else 0.10
        if char in ' ,.': dur_sec = 0.2
        num_amostras = int(SAMPLE_RATE * dur_sec)

        chunk = []
        for i in range(num_amostras):
            t = i / num_amostras
            f0 = 125.0 * (1.0 - 0.06 * t)

            if char in SURDAS:
                fonte = random.uniform(-1.0, 1.0) * 0.3
            elif char in ' ,.':
                fonte = 0.0
            else:
                fonte = pulse.tick(f0)

            saida = 0.0
            for j in range(3):
                picos_atuais[j] += (geo_alvo[j][0] - picos_atuais[j]) * 0.1
                chambers[j].update(picos_atuais[j], bw=geo_alvo[j][1])
                saida += chambers[j].process(fonte) * geo_alvo[j][2]

            env = math.sin(math.pi * t)
            sample = math.tanh(saida * env * 2.0)
            chunk.append(int(sample * 30000))

        # Escreve o áudio diretamente na placa de som
        stream.write(struct.pack(f'<{len(chunk)}h', *chunk))

    # Fecha o áudio corretamente
    stream.stop_stream()
    stream.close()
    p_audio.terminate()

if __name__ == '__main__':
    frase = "não foi eu, isso é mentira."
    arquinet_vgs_stream(frase)
