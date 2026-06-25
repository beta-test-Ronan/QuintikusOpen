#!/usr/bin/env python3
import math
import struct
import subprocess
import random

SAMPLE_RATE = 16000

# ═══════════════════════════════════════════════════════════════
# AS SUAS CLASSES (INTEGRADAS)
# ═══════════════════════════════════════════════════════════════

class WaveOscillator:
    def __init__(self, f0=130.0, wave_type='saw'):
        self.f0 = f0
        self.wave_type = wave_type
        self.phase = 0.0
        self.update_inc()

    def update_inc(self):
        self.phase_inc = self.f0 / SAMPLE_RATE

    def set_f0(self, f0):
        self.f0 = f0
        self.update_inc()

    def tick(self):
        if self.wave_type == 'saw':
            sample = 2.0 * self.phase - 1.0
        elif self.wave_type == 'square':
            sample = 1.0 if self.phase < 0.5 else -1.0
        else:
            sample = 2.0 * self.phase - 1.0
        self.phase += self.phase_inc
        if self.phase >= 1.0: self.phase -= 1.0
        return sample

class BiquadBandpass:
    def __init__(self, freq, bw=80.0):
        self.x1 = self.x2 = 0.0
        self.z1 = self.z2 = 0.0
        self.update_coeffs(freq, bw)

    def update_coeffs(self, freq, bw):
        # Proteção para frequências fora do limite de Nyquist
        freq = max(100, min(freq, SAMPLE_RATE // 2 - 100))
        omega = 2 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) * math.sinh(math.log(2) / 2 * bw / freq * omega / math.sin(omega))
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * math.cos(omega)
        a2 = 1 - alpha
        self.b = (b0/a0, b1/a0, b2/a0)
        self.a = (1.0, a1/a0, a2/a0)

    def process(self, sample):
        y = (self.b[0] * sample + self.b[1] * self.x1 + self.b[2] * self.x2
             - self.a[1] * self.z1 - self.a[2] * self.z2)
        self.x2, self.x1 = self.x1, sample
        self.z2, self.z1 = self.z1, y
        return y

def adsr_envelope(total_samples, attack, decay, sustain_level, release):
    env = [0.0] * total_samples
    for i in range(total_samples):
        if i < attack:
            env[i] = i / attack if attack > 0 else 1.0
        elif i < attack + decay:
            progress = (i - attack) / decay if decay > 0 else 0.0
            env[i] = 1.0 - progress * (1.0 - sustain_level)
        elif i < total_samples - release:
            env[i] = sustain_level
        else:
            release_start = total_samples - release
            progress = (i - release_start) / release if release > 0 else 0.0
            env[i] = sustain_level * (1.0 - progress)
    return env

# ═══════════════════════════════════════════════════════════════
# O DICIONÁRIO DE FONEMAS (SEU BANCO)
# ═══════════════════════════════════════════════════════════════

VOGAIS = {
    'a': (700, 1150, 2500), 'e': (500, 1800, 2500), 'i': (270, 2300, 3000),
    'o': (450, 850, 2300),  'u': (300, 850, 2250),
}

CONSOANTES = {
    'p': ('surda', 0.05, 800), 't': ('surda', 0.05, 2000), 'k': ('surda', 0.05, 1200),
    's': ('surda', 0.10, 4000), 'b': ('sonora', 0.06, 800), 'r': ('sonora', 0.08, 1500),
    'm': ('sonora', 0.08, 400), ' ': ('pausa', 0.1, 0),
}

# ═══════════════════════════════════════════════════════════════
# MOTOR DE SÍNTESE LINEAR UNIFICADO
# ═══════════════════════════════════════════════════════════════

def sintetizar_arquinet(texto):
    print(f"📡 QUINTIKUS v6.0 Ativado: «{texto}»")
    osc = WaveOscillator(f0=120.0, wave_type='saw')
    
    # Três filtros biquad para os formantes F1, F2, F3
    filtros = [BiquadBandpass(500), BiquadBandpass(1500), BiquadBandpass(2500)]
    
    cmd = ['aplay', '-t', 'raw', '-f', 'S16_LE', '-c', '1', '-r', str(SAMPLE_RATE)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Estado dos formantes para transição linear
    f_atual = [500.0, 1500.0, 2500.0]

    for char in texto.lower():
        if char in VOGAIS:
            f_alvo = VOGAIS[char]
            tipo = 'vogal'
            dur_sec = 0.12
        elif char in CONSOANTES:
            tipo_c, dur_sec, freq_c = CONSOANTES[char]
            f_alvo = (freq_c, freq_c * 1.5, freq_c * 2) if tipo_c != 'pausa' else f_atual
            tipo = tipo_c
        else: continue

        dur_amostras = int(SAMPLE_RATE * dur_sec)
        
        # Envelope ADSR para estruturação da sílaba
        if tipo == 'pausa':
            env = [0.0] * dur_amostras
        else:
            a, d, s, r = (200, 400, 0.7, 500) if tipo == 'vogal' else (100, 200, 0.5, 300)
            env = adsr_envelope(dur_amostras, a, d, s, r)

        chunk = []
        for i in range(dur_amostras):
            t = i / dur_amostras
            # Inflexão de pitch (Micro-delta linear)
            osc.set_f0(120.0 * (1.0 - 0.05 * t))
            
            # Fonte: Saw para voz, Ruído para consoantes surdas
            if tipo == 'surda':
                fonte = random.uniform(-1, 1)
            else:
                fonte = osc.tick()

            # Transição Linear dos Filtros Biquad
            res = 0.0
            for j in range(3):
                # Desliza a frequência do filtro em direção ao alvo
                f_atual[j] += (f_alvo[j] - f_atual[j]) * 0.01 
                filtros[j].update_coeffs(f_atual[j], bw=80)
                res += filtros[j].process(fonte)
            
            # Saturação e Estruturação Vocal (Linear -> Tanh)
            sample = math.tanh(res * env[i] * 0.5)
            chunk.append(int(sample * 30000))
        
        proc.stdin.write(struct.pack(f'<{len(chunk)}h', *chunk))

    proc.stdin.close()
    proc.wait()

if __name__ == '__main__':
    sintetizar_arquinet("oi,eu sou robo")
