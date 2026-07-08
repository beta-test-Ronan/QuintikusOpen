#!/usr/bin/env python3
import math
import struct
import random
import pyaudio

SAMPLE_RATE = 22050

# ═══════════════════════════════════════════════════════════════
# FÍSICA CINEMÁTICA (POWER VERSION)
# ═══════════════════════════════════════════════════════════════

class VGS_KineticSource:
    def __init__(self):
        self.tension = 0.0

    def generate(self, f0, e_restituicao, tipo):
        if tipo == 'fricativa':
            return random.uniform(-1.0, 1.0) * 0.4 # Aumentado o volume do sopro
        
        incremento = f0 / SAMPLE_RATE
        self.tension += incremento
        
        if self.tension > 0.75:
            # Impacto com energia total
            impacto = self.tension * (1.1 - e_restituicao) 
            self.tension = 0.0
            return impacto
        return 0.0

class VGS_Chamber:
    def __init__(self, freq, bw):
        self.x1 = self.x2 = self.z1 = self.z2 = 0.0
        self.update(freq, bw)
        
    def update(self, freq, bw):
        freq = max(60, min(freq, SAMPLE_RATE // 2 - 500))
        omega = 2 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) * math.sinh(math.log(2)/2 * bw/freq * omega/math.sin(omega))
        a0 = 1 + alpha
        self.b = (alpha/a0, 0, -alpha/a0)
        self.a = (1.0, -2*math.cos(omega)/a0, (1-alpha)/a0)

    def process(self, x):
        y = (self.b[0]*x + self.b[2]*self.x2 - self.a[1]*self.z1 - self.a[2]*self.z2)
        self.x2, self.x1, self.z2, self.z1 = self.x1, x, self.z1, y
        return y

# ═══════════════════════════════════════════════════════════════
# DICIONÁRIO OMNI (F1-F5)
# ═══════════════════════════════════════════════════════════════
FONEMAS = {
    'a': [[730,60,1.0], [1100,80,0.8], [2450,120,0.4], [3500,200,0.2], [4500,250,0.1], 0.85],
    'e': [[500,55,1.0], [1800,85,0.7], [2500,120,0.4], [3500,200,0.2], [4500,250,0.1], 0.85],
    'i': [[280,40,1.0], [2300,90,0.8], [3500,150,0.5], [4200,200,0.2], [5000,250,0.1], 0.85],
    'o': [[450,50,1.0], [800,70,0.7], [2300,140,0.3], [3500,200,0.2], [4500,250,0.1], 0.85],
    'u': [[320,40,1.0], [800,65,0.6], [2200,120,0.2], [3500,200,0.2], [4500,250,0.1], 0.85],
    'p_': [[160,30,1.2], [500,100,0.2], [1500,200,0.1], [2500,300,0.1], [3500,400,0.1], 0.12],
    's_': [[4000,400,0.1], [6000,600,0.5], [8000,800,0.5], [10000,1000,0.3], [12000,1200,0.2], 0.35],
    'm_': [[250,40,1.0], [1000,80,0.3], [2000,150,0.1], [3000,200,0.1], [4000,250,0.1], 0.70],
    'r_': [[450,80,0.8], [1500,100,0.4], [2500,200,0.2], [3500,200,0.1], [4500,200,0.1], 0.75],
    'sil': [[0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], 0.0]
}

def tradutor_universal(texto):
    texto = texto.lower()
    res = []
    i = 0
    while i < len(texto):
        c = texto[i]
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
        i += 1
    return res

# ═══════════════════════════════════════════════════════════════
# MOTOR HIGH-PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def quintikus_vgs_high_perf(texto):
    print(f"🔊 Quintikus VGS High-Perf | Texto: {texto}")
    p_audio = pyaudio.PyAudio()
    stream = p_audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)

    source = VGS_KineticSource()
    chambers = [VGS_Chamber(500, 80) for _ in range(5)]
    picos_atuais = [500, 1500, 2500, 3500, 4500]

    fonemas = tradutor_universal(texto)

    for idx, (som_nome, tipo) in enumerate(fonemas):
        dados = FONEMAS.get(som_nome, FONEMAS['sil'])
        geo_alvo = dados[0:5]
        e_restituicao = dados[5]
        
        dur_sec = 0.16 if tipo == 'vogal' else 0.10
        if tipo == 'pausa': dur_sec = 0.07
        num_samples = int(SAMPLE_RATE * dur_sec)
        
        chunk = []
        for i in range(num_samples):
            t = i / num_samples
            f0 = 100.0 * (1.0 - 0.07 * t)

            x = source.generate(f0, e_restituicao, tipo)

            saida = 0.0
            for j in range(5):
                picos_atuais[j] += (geo_alvo[j][0] - picos_atuais[j]) * 0.12
                chambers[j].update(picos_atuais[j], geo_alvo[j][1])
                saida += chambers[j].process(x) * geo_alvo[j][2]

            # Envelope Anti-Impacto com Volume Alto
            if tipo == 'plosiva':
                if t < 0.7: env = 0.0
                elif t < 0.70: env = (t - 0.6) / 0.05
                else: env = math.exp(-10 * (t - 0.65))
            else:
                env = math.sin(math.pi * t)
            
            # GANHO POTENCIALIZADO (8.0) para volume máximo sem rachar
            final = math.tanh(saida * env * 15.0) 
            chunk.append(int(final * 32000))

        stream.write(struct.pack(f'<{len(chunk)}h', *chunk))

    stream.stop_stream(); stream.close(); p_audio.terminate()

if __name__ == "__main__":
    falar = "paulo gosta de cinema. o som e incrivel."
    quintikus_vgs_high_perf(falar)
