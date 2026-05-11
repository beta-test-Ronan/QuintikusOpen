import numpy as np
import hashlib
import time
import re
import random
import os
from collections import Counter

class LNM_Quintikus:
    def __init__(self, d_model=128):
        self.d_model = d_model
        self.dataset = []
        self.mood = 0.5
        self.arousal = 0.5
        self.ultima_voz = ""
        self.signature = "LNM-VEGAS-V19"
        
        self.tags = ["odio", "medo", "morte", "amor", "riso", "sangue", "duvida"]
        self.cronometro_emocional = Counter()

        # --- SEU DICIONÁRIO MASSIVO DE PESOS ---
        self.pesos = {
            "morte": -1.0, "cadáver": -1.0, "assassinato": -1.0, "massacre": -1.0,
            "ódio": -0.9, "raiva": -0.8, "vingança": -0.9, "faca": -0.8,
            "medo": -0.7, "pavor": -0.9, "terror": -0.9, "horror": -0.9,
            "sangue": -0.7, "traição": -0.8, "falsidade": -0.7,
            "amor": 0.9, "beijo": 0.9, "sol": 0.7, "riso": 0.8, "vida": 0.6,
            "paz": 0.9, "esperança": 0.7, "duvida": -0.1, "silêncio": -0.1
        }

        # --- MAPEAMENTO DE DNA EMOCIONAL ---
        self.dna_tags = {
            "morte": "morte", "cadáver": "morte", "assassinato": "morte", "massacre": "morte",
            "ódio": "odio", "raiva": "odio", "vingança": "odio", "fúria": "odio",
            "medo": "medo", "pavor": "medo", "terror": "medo", "horror": "medo",
            "amor": "amor", "beijo": "amor", "afeto": "amor", "querido": "amor",
            "riso": "riso", "gargalhada": "riso", "sorriso": "riso",
            "sangue": "sangue", "faca": "sangue", "ferida": "sangue",
            "duvida": "duvida", "talvez": "duvida", "mistério": "duvida"
        }

        # --- RECURSOS EXTERNOS ---
        self.pontes = self._carregar_recurso("pontes.txt", ["O fato bizarro é que", "Eu podia jurar que", "Gente, escuta essa:"])
        self.reacoes = self._carregar_recurso("reacoes.txt", ["(Cinema puro...)", "(O bicho vai pegar!)", "(Arrepiei aqui.)"])
        self.final_txt = self._carregar_recurso("final.txt", "[duvida] O mistério sobreviveu ao tempo.")

    def _carregar_recurso(self, arquivo, fallback):
        if os.path.exists(arquivo):
            with open(arquivo, 'r', encoding='utf-8') as f:
                return f.read() if "final.txt" in arquivo else [l.strip() for l in f.readlines() if len(l.strip()) > 2]
        return fallback

    def _analisar_e_contar(self, texto):
        palavras = re.findall(r'\w+', texto.lower())
        v_total, encontrou = 0, False
        for p in palavras:
            v_total += self.pesos.get(p, 0.0)
            if p in self.dna_tags:
                self.cronometro_emocional[self.dna_tags[p]] += 1
                encontrou = True
        if not encontrou: self.cronometro_emocional["duvida"] += 0.5
        arousal = np.clip(abs(v_total) * 0.8, 0.1, 1.0)
        return v_total, arousal

    def preparar_banco(self, banco):
        sentencas = [s.strip() for s in re.split(r'[.;]', banco) if len(s.strip()) > 5]
        for i in range(0, len(sentencas), 2):
            bloco = sentencas[i]
            if i+1 < len(sentencas): bloco += ". " + sentencas[i+1]
            v, a = self._analisar_e_contar(bloco)
            pals = bloco.split()
            ponto = random.randint(len(pals)//3, len(pals)//2) if len(pals) > 3 else 1
            self.dataset.append({"s": " ".join(pals[:ponto]), "p": " ".join(pals[ponto:]), "v": v, "a": a})

    def typewriter(self, texto, velocidade_base):
        for char in texto:
            print(char, end='', flush=True)
            t = random.uniform(velocidade_base, velocidade_base + 0.01)
            if char in ",": t += 0.2
            if char in ".": t += 0.4
            time.sleep(t)

    def narrar(self):
        print("\n" + "═"*25 + "\n Quintikus[LNM-NARATIV] \n " + "═"*25)
        for fato in self.dataset:
            if random.random() < 0.15: print(f"\n[ ! ] {random.choice(['Espera!', 'Gente do céu...'])}\n")
            if random.random() > 0.8: print(f"{random.choice(self.reacoes)}")
            
            ponte = random.choice([p for p in self.pontes if p != self.ultima_voz])
            self.ultima_voz = ponte
            texto_viva = f"{ponte} {fato['s']}... {fato['p']}."
            
            vel = 0.012 if fato['a'] > 0.5 else 0.028
            self.typewriter(texto_viva, vel)
            print("\n")
            time.sleep(0.6)
        self._concluir_vegas()

    def _concluir_vegas(self):
        print("\n" + "="*60 + "\n [ PENSAMENTO FINAL PROPORCIONAL ] \n" + "="*60 + "\n")
        total_votos = sum(self.cronometro_emocional.values()) or 1
        tamanho_final_alvo = max(4, int(len(self.dataset) * 1.2))
        monologo, usadas = [], set()

        # Seleção Proporcional Rigorosa
        for tag_nome in self.tags:
            qtd = round((self.cronometro_emocional[tag_nome] / total_votos) * tamanho_final_alvo)
            pool = re.findall(r'\[{}\]\s*(.*?)(?=\[|$)'.format(tag_nome), self.final_txt, re.S)
            if pool:
                random.shuffle(pool)
                adc = 0
                for f in pool:
                    f = f.strip()
                    if f not in usadas and adc < qtd:
                        monologo.append(f); usadas.add(f); adc += 1

        random.shuffle(monologo)
        self.typewriter(" ".join(monologo) if monologo else "O fim chegou.", 0.035)
        print("\n\n" + "_"*64 + f"\n({self.signature} | DNA: {dict(self.cronometro_emocional)})")

# --- START ---
qn = LNM_Quintikus()
# Exemplo de banco para testar a proporção
banco = """
Era uma manhã de sol radiante, daquelas que fazem a vida parecer um presente. O riso das crianças brincando na praça ecoava como música, e o beijo de Clara e Pedro no coreto selava um amor que todos julgavam eterno. A felicidade pairava no ar, misturada ao perfume das flores do jardim de Dona Marlene. Os velhos sorriam, os cachorros latiam de alegria, e até o padre José, do alto da igreja, abençoava a união com uma gargalhada gostosa. A esperança era tanta que ninguém desconfiava da sombra que se aproximava.

Mas a inveja, essa serpente silenciosa, já havia feito ninho no coração de Vilela. Ele observava o casal com um ciúme doentio, e seus olhos faiscavam de ódio a cada abraço alheio. A falsidade vestia-se de amizade, e a traição foi arquitetada numa noite sem lua, regada a vinho barato e promessas de vingança. Ninguém na vila sabia, mas a faca já estava sendo afiada.

Naquela sexta-feira, o céu escureceu de repente. O vento trouxe um cheiro de chuva e de desgraça. Clara encontrou Pedro caído na sala, o sangue manchando o tapete persa. Um cadáver. A cena era de um pavor indescritível; as paredes tinham respingos vermelhos, e uma arma branca — um punhal de cabo de marfim — jazia ao lado do corpo. O grito de Clara congelou a cidade. O medo tomou conta de cada casa, e as pessoas trancaram as portas com desespero. A tristeza foi um manto pesado, e o choro correu solto. Até os sinos da igreja pareciam soluçar.

A investigação caiu nas mãos do delegado Romão, mas as provas eram confusas. As pistas apontavam para todos os lados, e a dúvida se instalou como uma névoa espessa. Quem matou Pedro? Teria sido Vilela, o amigo de infância? Ou seria uma vingança de algum desafeto secreto? A cada dia, uma nova suspeita, uma nova carta anônima, um novo rumor. A cidade dividiu-se entre a certeza e a incerteza, e o medo da verdade era tão grande quanto o medo da mentira. O velho padre dizia: "A verdade virá à luz", mas as sombras pareciam mais fortes.

A viúva Clara, consumida pela solidão e pela desconfiança, começou a ver o fantasma do noivo em cada canto escuro. As noites eram de terror: vultos, sussurros, portas rangendo. O sobrado dos Pereira virou uma assombração, e o pavor aumentava a cada hora. Mas a curiosidade de um jornalista forasteiro trouxe uma reviravolta: ele descobriu bilhetes antigos, manchados de ódio, escondidos no baú de Vilela. A caligrafia era do próprio Vilela, confessando um amor doentio por Clara e jurando destruir Pedro.

A revelação foi um choque sísmico. O júri se reuniu em praça pública, e a máscara caiu. Vilela, encurralado, confessou o assassinato com uma frieza que arrepiou até o carrasco mais sádico. A raiva da multidão era uma fúria coletiva; queriam linchamento. Mas a lei prevaleceu, e a cela fria foi seu novo lar. A cidade respirou, mas a dor ainda latejava.

Foi então que, do fundo da escuridão, brotou uma inesperada esperança. Uma carta de Vilela, já na prisão, pedia perdão. Não um perdão qualquer — uma súplica encharcada de lágrimas e arrependimento verdadeiro. E o mais surpreendente: Clara, depois de meses de luto e reflexão, decidiu perdoá-lo. O perdão foi um ato de coragem que ninguém esperava. A cena no presídio foi de uma ternura dilacerante: ela abraçou o assassino do próprio noivo, dizendo que o ódio não traria Pedro de volta.

A partir desse instante, a cidade começou a curar-se. As flores voltaram a brotar no jardim, e o riso, tímido no início, retomou seu lugar. As crianças fizeram uma peça de teatro sobre o "poder do amor", e a plateia riu e chorou. A páscoa foi celebrada com abraços longos, e o beijo da paz na missa teve gosto de reconciliação. A esperança renasceu, e até a figueira centenária deu frutos mais doces.

No fim, a história de Clara e Pedro (e Vilela) virou lenda: um conto sobre a fragilidade da vida, a força do perdão e a certeza de que, mesmo após o sangue e o pavor, o amor pode ser reconstruído. A vida seguiu, com suas ironias e surpresas, mas agora com uma sabedoria nova. O último suspiro foi uma gargalhada geral, daquelas que limpam a alma.

A dúvida, essa velha companheira, ainda sussurra em noites de insônia: "Será que foi justo?" Mas a cidade aprendeu a conviver com a pergunta sem resposta, pois a única certeza é que o sol sempre volta depois da tempestade.
"""
qn.preparar_banco(banco)
qn.narrar()
