#!/usr/bin/env python3
import math
import struct
import random
import pyaudio

SAMPLE_RATE = 22050

# ═══════════════════════════════════════════════════════════════
# CAMADA 0: FONTE DE EXCITAÇÃO (CINEMÁTICA STICK-SLIP)
# ═══════════════════════════════════════════════════════════════
class VGS_Source:
    def __init__(self):
        self.tension = 0.0

    def generate(self, f0, e_restituicao, tipo):
        if tipo == 'fricativa':
            return random.uniform(-1.0, 1.0) * 0.3
        
        # Acúmulo de energia linear
        self.tension += f0 / SAMPLE_RATE
        if self.tension > 0.75:
            # Colisão: Transferência de momento linear
            impacto = self.tension * (1.1 - e_restituicao)
            self.tension = 0.0
            return impacto
        return 0.0

# ═══════════════════════════════════════════════════════════════
# CAMADA 1: GEOMETRIA DE RESSONÂNCIA (BIQUAD IIR)
# ═══════════════════════════════════════════════════════════════
class VGS_Chamber:
    def __init__(self, freq, bw, amp=1.0):
        self.x1 = self.x2 = self.z1 = self.z2 = 0.0
        self.amp = amp
        self.update(freq, bw)

    def update(self, freq, bw):
        freq = max(50, min(freq, SAMPLE_RATE // 2 - 500))
        omega = 2 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) * math.sinh(math.log(2)/2 * bw/freq * omega/math.sin(omega))
        a0 = 1 + alpha
        self.b = (alpha/a0, 0, -alpha/a0)
        self.a = (1.0, -2*math.cos(omega)/a0, (1-alpha)/a0)

    def process(self, x):
        y = (self.b[0]*x + self.b[2]*self.x2 - self.a[1]*self.z1 - self.a[2]*self.z2)
        self.x2, self.x1, self.z2, self.z1 = self.x1, x, self.z1, y
        return y * self.amp

# ═══════════════════════════════════════════════════════════════
# DICIONÁRIO GEOMÉTRICO (MAPEAMENTO DE TODAS AS LETRAS)
# ═══════════════════════════════════════════════════════════════
# Estrutura: [F1, F2, F3, F4, F5] + Coeficiente de Restituição (e)
GEOMETRIAS = {
    'a': [[800,60,1.0], [1200,80,0.8], [2800,120,0.4], [3800,200,0.2], [4800,250,0.1], 0.85],
    'e': [[550,55,1.0], [2100,85,0.7], [3000,120,0.4], [4000,200,0.2], [5000,250,0.1], 0.85],
    'i': [[300,40,1.0], [2500,90,0.8], [3800,150,0.5], [4500,200,0.2], [5500,250,0.1], 0.85],
    'o': [[500,50,1.0], [900,70,0.7], [2600,140,0.3], [3800,200,0.2], [4800,250,0.1], 0.85],
    'u': [[350,40,1.0], [900,65,0.6], [2500,120,0.2], [3800,200,0.2], [4800,250,0.1], 0.85],
    'p_': [[180,40,1.2], [600,100,0.2], [1800,200,0.1], [2800,300,0.1], [3800,400,0.1], 0.15],
    's_': [[5000,500,0.05], [7000,800,0.4], [9000,1000,0.5], [11000,1200,0.3], [13000,1500,0.2], 0.35],
    'm_': [[300,40,1.0], [1100,80,0.3], [2200,150,0.1], [3200,200,0.1], [4200,250,0.1], 0.70],
    'r_': [[500,80,0.8], [1700,100,0.4], [2800,200,0.2], [3800,200,0.1], [4800,200,0.1], 0.75],
    'sil': [[0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], 0.0]
}

def traduzir_texto(texto):
    res = []
    for c in texto.lower():
        if c in 'aeiouáàâãéêíóôõú':
            v = c
            if v in 'áàâã': v = 'a'
            elif v in 'éê': v = 'e'
            elif v in 'í': v = 'i'
            elif v in 'óôõ': v = 'o'
            elif v in 'ú': v = 'u'
            res.append((v, 'vogal'))
        elif c in 'pbtdkgq': res.append(('p_', 'plosiva'))
        elif c in 'szvjxcç': res.append(('s_', 'fricativa'))
        elif c in 'mn': res.append(('m_', 'nasal'))
        elif c in 'rl': res.append(('r_', 'liquida'))
        elif c in ' ,.?!': res.append(('sil', 'pausa'))
    return res

# ═══════════════════════════════════════════════════════════════
# MOTOR DE PROCESSAMENTO LINEAR EM CASCATA
# ═══════════════════════════════════════════════════════════════
def processar_audio(texto):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)

    fonte = VGS_Source()
    # Camada de Corpo (Fixa) + Camada de Boca (Dinâmica)
    body = [VGS_Chamber(160, 40), VGS_Chamber(400, 80)]
    mouth = [VGS_Chamber(500, 80) for _ in range(5)]
    picos_atuais = [f[0] for f in GEOMETRIAS['sil'][:5]]

    frase = traduzir_texto(texto)

    for som, tipo in frase:
        dados = GEOMETRIAS.get(som, GEOMETRIAS['sil'])
        alvos, e_restituicao = dados[:5], dados[5]
        
        num_samples = int(SAMPLE_RATE * (0.15 if tipo == 'vogal' else 0.09))
        if tipo == 'pausa': num_samples = int(SAMPLE_RATE * 0.08)

        for i in range(num_samples):
            t = i / num_samples
            f0 = 180.0 * (1.0 - 0.02 * t) # Frequência estável

            # 1. Excitação
            x = fonte.generate(f0, e_restituicao, tipo)

            # 2. Cascata Linear de Corpo
            x_body = body[0].process(x) + body[1].process(x)

            # 3. Cascata Linear de Boca com Inércia
            saida = 0.0
            for j in range(5):
                picos_atuais[j] += (alvos[j][0] - picos_atuais[j]) * 0.15
                mouth[j].update(picos_atuais[j], alvos[j][1])
                saida += mouth[j].process(x_body) * alvos[j][2]

            # 4. Envelope e Saturação Brutalista (100.0)
            env = 1.0 if tipo == 'plosiva' and t > 0.7 else (0.0 if tipo == 'plosiva' else math.sin(math.pi * t))
            if tipo == 'plosiva' and t > 0.7: env = math.exp(-15*(t-0.7))
            
            # A teoria do ganho 100: maximiza a densidade harmônica
            final = math.tanh(saida * env * 100.0)
            stream.write(struct.pack('<h', int(final * 30000)))

    stream.stop_stream(); stream.close(); p.terminate()

if __name__ == "__main__":
    processar_audio("ronan paulo gosta de cinema.")
