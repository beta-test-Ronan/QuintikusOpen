#!/usr/bin/env python3
import math
import struct
import random
import pyaudio

SAMPLE_RATE = 22050 

# ═══════════════════════════════════════════════════════════════
# FÍSICA VGS: CÂMARAS E FONTES DINÂMICAS
# ═══════════════════════════════════════════════════════════════

class VGSPulse:
    def __init__(self):
        self.phase = 0.0
    def tick(self, f0):
        self.phase += f0 / SAMPLE_RATE
        if self.phase >= 1.0: self.phase -= 1.0
        # Pulso Glotal: Ação e Reação
        return math.sin(math.pi * (self.phase / 0.75)) if self.phase < 0.75 else 0.0

class VGSChamber:
    def __init__(self, freq, bw):
        self.x1 = self.x2 = self.z1 = self.z2 = 0.0
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
        return y

# ═══════════════════════════════════════════════════════════════
# DICIONÁRIO DE GEOMETRIAS (AJUSTADO PARA ARTICULAÇÃO)
# ═══════════════════════════════════════════════════════════════
FONEMAS = {
    'a': [[730,60,1.0], [1100,80,0.8], [2450,120,0.4], [3500,200,0.2], [4500,250,0.1]],
    'e': [[500,55,1.0], [1800,85,0.7], [2500,120,0.4], [3500,200,0.2], [4500,250,0.1]],
    'i': [[280,40,1.0], [2300,90,0.8], [3500,150,0.5], [4200,200,0.2], [5000,250,0.1]],
    'o': [[450,50,1.0], [800,70,0.7], [2300,140,0.3], [3500,200,0.2], [4500,250,0.1]],
    'u': [[320,40,1.0], [800,65,0.6], [2200,120,0.2], [3500,200,0.2], [4500,250,0.1]],
    's_': [[4000,500,0.1], [6000,800,0.5], [8000,1000,0.4], [10000,1200,0.3], [12000,1500,0.2]],
    'p_': [[150,100,1.0], [500,200,0.3], [1500,500,0.1], [3000,500,0.1], [4000,500,0.1]],
    't_': [[200,100,1.0], [1500,200,0.3], [2500,500,0.1], [3500,500,0.1], [4500,500,0.1]],
    'k_': [[250,100,1.0], [1200,200,0.3], [2000,500,0.1], [3000,500,0.1], [4000,500,0.1]],
    'r_': [[450,100,0.8], [1500,200,0.4], [2500,200,0.2], [3500,200,0.1], [4500,200,0.1]],
    'm_': [[250,50,1.0], [1000,100,0.4], [2500,200,0.2], [3500,200,0.1], [4500,200,0.1]],
    'sil': [[0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]]
}

# ═══════════════════════════════════════════════════════════════
# CONVERSOR FONÉTICO MELHORADO
# ═══════════════════════════════════════════════════════════════
def traduzir_fonemas(texto):
    texto = texto.lower()
    res = []
    i = 0
    while i < len(texto):
        c = texto[i]
        if c in 'aeiouáéíóúâêôãõà':
            v = c
            if v in 'áàâã': v = 'a'
            elif v in 'éê': v = 'e'
            elif v in 'í': v = 'i'
            elif v in 'óôõ': v = 'o'
            elif v in 'ú': v = 'u'
            res.append((v, 'vogal'))
        elif c in 'pb': res.append(('p_', 'plosiva'))
        elif c in 'td': res.append(('t_', 'plosiva'))
        elif c in 'kgq': res.append(('k_', 'plosiva'))
        elif c in 'szjxcv': res.append(('s_', 'fricativa'))
        elif c in 'rl': res.append(('r_', 'liquida'))
        elif c in 'mn': res.append(('m_', 'nasal'))
        elif c in ' ,.?!': res.append(('sil', 'pausa'))
        i += 1
    return res

# ═══════════════════════════════════════════════════════════════
# MOTOR DE ARTICULAÇÃO VGS
# ═══════════════════════════════════════════════════════════════

def quintikus_vgs_articulador(texto):
    print(f"🎙️ Quintikus VGS Articulador | Texto: {texto}")
    p_audio = pyaudio.PyAudio()
    stream = p_audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)

    pulse = VGSPulse()
    chambers = [VGSChamber(500, 80) for _ in range(5)]
    picos_atuais = [500, 1500, 2500, 3500, 4500]

    fonemas = traduzir_fonemas(texto)

    for f_nome, tipo in fonemas:
        alvo = FONEMAS.get(f_nome, FONEMAS['sil'])
        
        # AJUSTE DE DURAÇÃO (O segredo da clareza)
        if tipo == 'plosiva': dur_sec = 0.07   # Estalo rápido
        elif tipo == 'nasal': dur_sec = 0.10   # Médio
        elif tipo == 'pausa': dur_sec = 0.15   # Pausa
        else: dur_sec = 0.14                  # Vogal longa
        
        num_amostras = int(SAMPLE_RATE * dur_sec)
        chunk = []

        for i in range(num_amostras):
            t = i / num_amostras
            
            # Fonte de excitação baseada no tipo
            if tipo == 'fricativa':
                fonte = random.uniform(-1.0, 1.0) * 0.2
            elif tipo == 'plosiva':
                # EXPLOSÃO: Silêncio no início e estouro de ruído no final
                fonte = random.uniform(-1.0, 1.0) * 0.5 if t > 0.6 else 0.0
            elif tipo == 'pausa':
                fonte = 0.0
            else:
                # Vogais, Nasais e Líquidas usam o pulso glotal
                fonte = pulse.tick(125.0)

            # Escultura de picos
            saida = 0.0
            for j in range(5):
                # Velocidade de transição (Coarticulação)
                # Consoantes mudam a boca mais rápido que vogais
                vel = 0.25 if tipo != 'vogal' else 0.12
                picos_atuais[j] += (alvo[j][0] - picos_atuais[j]) * vel
                chambers[j].update(picos_atuais[j], alvo[j][1])
                saida += chambers[j].process(fonte) * alvo[j][2]

            # ENVELOPE DINÂMICO
            if tipo == 'plosiva':
                # Envelope de "estalo" (Decaimento rápido)
                env = math.exp(-10 * t) if t > 0.6 else 0.0
            else:
                # Envelope suave para vogais
                env = math.sin(math.pi * t)
            
            final = math.tanh(saida * env * 2.5)
            chunk.append(int(final * 30000))

        stream.write(struct.pack(f'<{len(chunk)}h', *chunk))

    stream.stop_stream(); stream.close(); p_audio.terminate()

if __name__ == "__main__":
    # Teste agora esta frase: as consoantes P, T e R devem aparecer.
    falar = "o rato roeu a roupa do rei de roma"
    quintikus_vgs_articulador(falar)
