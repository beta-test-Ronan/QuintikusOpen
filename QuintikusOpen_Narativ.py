import numpy as np
import hashlib
import time
import re
import random
import os

class QuintikusNarativ:
    def __init__(self, d_model=128):
        self.d_model = d_model
        self.dataset = []
        self.mood = 0.5
        self.arousal = 0.5
        self.ultima_ponte = ""
        self.signature = "V11-LIBRARY-DIRECTOR"
        
        # --- DICIONÁRIO DE PESOS (A ALMA) ---
        self.pesos = {
            "morte": -1.0, "sangue": -0.9, "faca": -0.8, "ódio": -0.9, "traição": -0.8,
            "amor": 0.9, "beijo": 0.9, "sol": 0.6, "riso": 0.8, "vida": 0.7,
            "medo": -0.7, "escuro": -0.5, "cadáver": -1.0, "vilela": -0.3, "rita": 0.4
        }

        # --- CARREGAMENTO DE RECURSOS EXTERNOS ---
        self.pontes = self._carregar_recurso("pontes.txt", [
            "Cê não faz ideia, mas", "O fato bizarro é que", "Atenção aqui:", 
            "Aí a coisa ficou feia,", "Inexplicavelmente,", "Num sopro de mistério,"
        ])
        
        self.reacoes = self._carregar_recurso("reacoes.txt", [
            "(Isso aqui é cinema puro...)", "(Eu não disse que ia azedar?)", 
            "(Zzzz... cadê o sangue?)", "(Tô sentindo a tensão daqui.)"
        ])

    def _carregar_recurso(self, arquivo, fallback):
        """Puxa frases de um arquivo .txt ou usa o padrão se não existir"""
        if os.path.exists(arquivo):
            with open(arquivo, 'r', encoding='utf-8') as f:
                linhas = [l.strip() for l in f.readlines() if len(l.strip()) > 2]
                return linhas if linhas else fallback
        return fallback

    def _analisar_sentimento(self, texto):
        palavras = re.findall(r'\w+', texto.lower())
        v = sum([self.pesos.get(p, 0) for p in palavras])
        self.mood = np.clip(self.mood * 0.7 + (v * 0.1), 0, 1)
        self.arousal = np.clip(self.arousal * 0.5 + (abs(v) * 0.5), 0.1, 1.0)
        return v, self.arousal

    def preparar_banco(self, banco):
        # Quebra em blocos de pensamento grandes (Profundidade)
        sentencas = [s.strip() for s in re.split(r'[.;]', banco) if len(s.strip()) > 8]
        for i in range(0, len(sentencas), 2):
            bloco = sentencas[i]
            if i+1 < len(sentencas): bloco += ". " + sentencas[i+1]
            
            v, a = self._analisar_sentimento(bloco)
            pals = bloco.split()
            # Ponto de Suspensão (Onde o narrador toma fôlego)
            ponto = random.randint(len(pals)//3, len(pals)//2) if len(pals) > 4 else 1
            
            self.dataset.append({
                "s": " ".join(pals[:ponto]),
                "p": " ".join(pals[ponto:]),
                "v": v, "a": a
            })

    def typewriter(self, texto, velocidade):
        for char in texto:
            print(char, end='', flush=True)
            t = random.uniform(velocidade, velocidade + 0.01)
            if char in ",": t += 0.25
            if char in ".": t += 0.5
            time.sleep(t)

    def narrar(self):
        print("\n" + "╔" + "═"*64 + "╗")
        print(f"║ { 'QUINTIKUS Narativ: O NARRADOR DE BIBLIOTECA'.center(62) } ║")
        print("╚" + "═"*64 + "╝\n")

        for i, fato in enumerate(self.dataset):
            # 1. Gatilho de Atenção (Dopamina)
            if random.random() < 0.2:
                print(f"\n[ ! ] {random.choice(['Espera!', 'Olha só isso:', 'Gente do céu...'])}\n")
                time.sleep(0.7)

            # 2. Comentário Aleatório (Puxado do TXT)
            if random.random() > 0.6:
                coment = random.choice(self.reacoes)
                print(f"\n{coment}")
                time.sleep(0.5)

            # 3. Ponte Narrativa (Puxada do TXT - Sem repetição)
            ponte = random.choice([p for p in self.pontes if p != self.ultima_ponte])
            self.ultima_ponte = ponte
            
            # 4. Construção da Frase (Cinema)
            texto_viva = f"{ponte} {fato['s']}... {fato['p']}."
            
            # 5. Execução (Velocidade pela emoção)
            vel_base = 0.012 if fato['a'] > 0.4 else 0.03
            self.typewriter(texto_viva, vel_base)
            print("\n")

            time.sleep(0.8 + (1.0 - fato['a']))

        # --- FECHAMENTO ---
        print("_"*64)
        resumo = f"O CICLO SE FECHA: Partimos de '{self.dataset[0]['s'][:20]}' e caímos em '{self.dataset[-1]['s'][:20]}'. Fim do relato."
        self.typewriter(resumo, 0.02)
        print("\n" + "_"*64)
        print(f"\n({self.signature} | Latência: {random.randint(1, 4)}ms)")


# --- COMO USAR ---
# 1. Crie um arquivo chamado 'pontes.txt' e coloque frases como:
#    Sabe o que é mais doido?
#    A fofoca corre e diz que
#    Pois bem, no escuro,
#
# 2. Crie um arquivo chamado 'reacoes.txt' e coloque:
#    (Meu Deus, o bicho vai pegar!)
#    (Zzzz... tá lento, mas vai melhorar.)

# Banco de dados de exemplo (Texto longo)


banco_exemplo = """
Rita estava certa de ser amada. Camilo não acreditava em nada.
A cartomante adivinhara tudo e a prova é que ela agora estava tranqüila.
Vilela podia sabê-lo, mas tive muita cautela ao entrar na casa.
O silêncio do cadáver era o último segredo guardado naquela casa sombria.
"""

qn = QuintikusNarativ()
qn.preparar_banco(banco_exemplo)
qn.narrar()
