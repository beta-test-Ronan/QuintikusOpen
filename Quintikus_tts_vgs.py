#!/usr/bin/env python3
import math
import struct
import random
import pyaudio

# Configurações Globais
SAMPLE_RATE = 22050
VGS_HEADER = "Quintikus TTS VGS v1.2 - Vibração Geométrica Sonora"

# ═══════════════════════════════════════════════════════════════
# FÍSICA VGS (FONTES E CÂMARAS)
# ═══════════════════════════════════════════════════════════════

class VGSPulse:
    """ Fonte de Ação e Reação (Glotal) """
    def __init__(self):
        self.phase = 0.0
    def tick(self, f0):
        self.phase += f0 / SAMPLE_RATE
        if self.phase >= 1.0: self.phase -= 1.0
        # Geometria de colisão: Subida gradual, queda súbita
        return math.sin(math.pi * (self.phase / 0.75)) if self.phase < 0.75 else 0.0

class VGSChamber:
    """ Câmara de Colisão Sonora (Filtro de Ressonância) """
    def __init__(self, freq, bw):
        self.x1 = self.x2 = self.z1 = self.z2 = 0.0
        self.b = (0,0,0); self.a = (1,0,0)
        self.update(freq, bw)
    def update(self, freq, bw):
        freq = max(50, min(freq, SAMPLE_RATE // 2 - 500))
        bw = max(10, bw)
        omega = 2 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) * math.sinh(math.log(2)/2 * bw/freq * omega/math.sin(omega))
        a0 = 1 + alpha
        self.b = (alpha / a0, 0, -alpha / a0)
        self.a = (1.0, -2 * math.cos(omega) / a0, (1 - alpha) / a0)
    def process(self, x):
        y = (self.b[0]*x + self.b[1]*self.x1 + self.b[2]*self.x2 - self.a[1]*self.z1 - self.a[2]*self.z2)
        self.x2, self.x1, self.z2, self.z1 = self.x1, x, self.z1, y
        return y

# ═══════════════════════════════════════════════════════════════
# DICIONÁRIOS INTEGRAIS (VGS MAPPED)
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
    ' ': [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]],
    ',': [[0, 0, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
    '.': [[0, 0, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
}

GEOMETRIAS_EN = {
    'a': [[730, 60, 1.0], [1100, 80, 0.8], [2450, 120, 0.4]],
    'e': [[500, 55, 1.0], [1800, 85, 0.7], [2500, 120, 0.4]],
    'i': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],
    'o': [[450, 50, 1.0], [800, 70, 0.7], [2300, 140, 0.3]],
    'u': [[320, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],
    'â': [[600, 60, 1.0], [1200, 80, 0.7], [2300, 130, 0.4]],
    'î': [[400, 50, 1.0], [2000, 80, 0.7], [2800, 130, 0.4]],
    'ô': [[500, 55, 1.0], [850, 75, 0.7], [2400, 130, 0.3]],
    'ê': [[660, 60, 1.0], [1700, 80, 0.8], [2500, 130, 0.4]],
    'û': [[350, 45, 1.0], [1000, 70, 0.6], [2000, 120, 0.3]],
    'ë': [[500, 50, 0.5], [1500, 70, 0.4], [2500, 100, 0.2]],
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
    'h': [[0, 1, 0.0], [0, 1, 0.0], [0, 1, 0.0]],
    'w': [[300, 40, 1.0], [800, 65, 0.6], [2200, 120, 0.2]],
    'y': [[280, 40, 1.0], [2300, 90, 0.8], [3500, 150, 0.5]],
    'x': [[3000, 400, 0.4], [5000, 600, 0.5], [6500, 800, 0.3]],
    'j': [[300, 50, 0.8], [2000, 90, 0.5], [3000, 150, 0.3]],
    'c': [[200, 25, 1.0], [3000, 400, 0.4], [5000, 600, 0.5]],
    ' ': [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]],
}

# Identificação de Colisões Surdas (Sopro)
SURDAS = {'p', 't', 'k', 'f', 's', 'x', 'c', 'h'}

# ═══════════════════════════════════════════════════════════════
# MOTOR DE SÍNTESE VGS (VIBRAÇÃO GEOMÉTRICA SONORA)
# ═══════════════════════════════════════════════════════════════

def quintikus_vgs_engine(texto, idioma='pt'):
    banco = GEOMETRIAS_PTBR if idioma == 'pt' else GEOMETRIAS_EN
    print(f"🎙️ {VGS_HEADER} | Idioma: {idioma.upper()}")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)

    pulse = VGSPulse()
    chambers = [VGSChamber(500, 80) for _ in range(3)]
    picos_atuais = [500.0, 1500.0, 2500.0]

    # Pre-processamento para Inglês (Digraphs)
    if idioma == 'en':
        texto = texto.lower().replace("sh", "x").replace("ch", "c").replace("th", "t")

    for char in texto.lower():
        if char not in banco: continue
        alvo = banco[char]
        
        # Duração baseada na colisão (Vogais são mais longas, Consoantes mais curtas)
        dur_sec = 0.14 if char in 'aeiouáàâãéêíóôõú' else 0.10
        if char in ' ,.': dur_sec = 0.25
        num_amostras = int(SAMPLE_RATE * dur_sec)

        chunk = []
        for i in range(num_amostras):
            t = i / num_amostras
            f0_base = 120.0 * (1.0 - 0.05 * t) # Queda natural de pitch

            # Definição da Fonte de Excitação
            if char in SURDAS:
                fonte = random.uniform(-1.0, 1.0) * 0.25 # Colisão de Ruído
            elif char in ' ,.':
                fonte = 0.0
            else:
                fonte = pulse.tick(f0_base) # Choque Glotal

            # Processamento nas Câmaras de Geometria
            saida = 0.0
            for j in range(3):
                # Deslize de Trajetória (Inércia da boca)
                picos_atuais[j] += (alvo[j][0] - picos_atuais[j]) * 0.1
                chambers[j].update(picos_atuais[j], alvo[j][1])
                saida += chambers[j].process(fonte) * alvo[j][2]

            # Envelope de Colisão Vocal e Saturação Física
            env = math.sin(math.pi * t)
            # Saturação math.tanh simula a compressão do ar no trato vocal
            final = math.tanh(saida * env * 2.0)
            chunk.append(int(final * 30000))

        stream.write(struct.pack(f'<{len(chunk)}h', *chunk))

    # Finalização
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    # Teste em Português com acentos completos
    texto_pt = "Olá! A vibração geométrica sonora agora entende acentuação como á, ê, õ e ú."
    quintikus_vgs_engine(texto_pt, idioma='pt')
    
    # Teste em Inglês com fonemas especiais
    texto_en = "I am a robot. This is the sitting and caught test with schwa sound."
    quintikus_vgs_engine(texto_en, idioma='en')
