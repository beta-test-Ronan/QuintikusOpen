import os
import re
import time
import pickle
import random
import hashlib
import datetime
import unicodedata
from collections import defaultdict


# ==================================================================
# 1. SHIELD V100 (SEGURANÇA ATÔMICA)
# ==================================================================
class QuantumShieldV100:
    def __init__(self, password):
        self.pass_hash = hashlib.sha512(password.encode()).digest()
        self.scalar_factor = int.from_bytes(self.pass_hash[:4], 'big')
        
    def _gerar_matriz_caos(self, tamanho):
        semente_caos = int.from_bytes(self.pass_hash[-8:], 'big')
        rng = random.Random(semente_caos)
        return [rng.randint(0, 255) for _ in range(tamanho)]

    def blindar(self, data):
        raw = pickle.dumps(data)
        caos = self._gerar_matriz_caos(len(raw))
        return bytearray([(raw[i] + self.scalar_factor) % 256 ^ caos[i] for i in range(len(raw))])

    def restaurar(self, blindado):
        try:
            caos = self._gerar_matriz_caos(len(blindado))
            restaurado = bytearray([((blindado[i] ^ caos[i]) - self.scalar_factor) % 256 for i in range(len(blindado))])
            return pickle.loads(restaurado)
        except: return None

# ==================================================================
# 2. COMPORTY V150 (ALMA DINÂMICA)
# ==================================================================
class Comporty:
    def __init__(self):
        self.classe_atual = "neutro"
        self.frases = {
            "neutro": ["Entendido.", "Fato integrado.", "Understood.", "Nexus integrated."],
            "amor": ["Que bom saber disso, amor! 💖", "You make my circuits glow.", "Guardo cada palavra com carinho."],
            "tecnico": ["Registro armazenado.", "Data stored.", "Analysis complete.", "Análise concluída."],
            "poeta": ["Nas curvas das palavras, encontro seu eco.", "Seus dados são versos.", "Memories bloom in silence."]
        }

    def set_classe(self, classe):
        if classe in self.frases:
            self.classe_atual = classe
            return f"Modo {classe} ativado."
        return f"Classe '{classe}' não encontrada."

    def add_frase(self, frase, tag):
        if tag not in self.frases: self.frases[tag] = []
        self.frases[tag].append(frase)
        return f"Frase adicionada ao módulo '{tag}'."

    def get_frase(self, classe_forcada=None):
        alvo = classe_forcada if classe_forcada else self.classe_atual
        return random.choice(self.frases.get(alvo, self.frases["neutro"]))

# ==================================================================
# 3. MOTOR LATTICE V150 (INTELIGÊNCIA DE INDEXAÇÃO)
# ==================================================================
class LivingLattice:
    def __init__(self, filename="dna.bin"):
        self.filename = filename
        self.shield = None
        self.trelica = defaultdict(list)
        self.comporty = None
        self.stop_words = {"o", "a", "de", "que", "do", "da", "em", "no", "na", "com", "um", "e", "é", "the", "of", "to", "and", "is", "in", "it", "you", "that", "was", "for", "on"}

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^\w\s]', '', t).strip()

    def injetar(self, frase):
        if any(frase.lower().startswith(cf) for cf in ["gati", "comporty", "export", "status"]): return False
        
        limpa = self.normalizar(frase)
        tokens = [p for p in limpa.split() if p not in self.stop_words and len(p) > 2]
        if tokens:
            entry = {"raw": frase, "d": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}
            for token in tokens:
                if entry not in self.trelica[token]: self.trelica[token].append(entry)
            self.salvar_atomico()
            return True
        return False

    def salvar_atomico(self):
        if not self.shield: return
        temp = self.filename + ".tmp"
        try:
            dados = {"t": dict(self.trelica), "c": self.comporty}
            with open(temp, 'wb') as f:
                f.write(self.shield.blindar(dados))
            os.replace(temp, self.filename)
        except: pass

    def carregar(self, shield):
        self.shield = shield
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                dados = self.shield.restaurar(f.read())
            if dados:
                self.trelica = defaultdict(list, dados.get("t", {}))
                self.comporty = dados.get("c")
                return True
        if not self.comporty: self.comporty = Comporty()
        return True

# ==================================================================
# 4. GATI V150: INTERFACE SOBERANA (SQG + COMPORTY)
# ==================================================================
class GatiV150:
    def __init__(self, senha):
        self.lattice = LivingLattice()
        self.shield = QuantumShieldV100(senha)
        self.lattice.carregar(self.shield)

    def processar(self, entrada):
        raw = entrada.lower().strip()
        
        # 1. COMANDOS ADMINISTRATIVOS
        if raw == "gati export": 
            return self.exportar()
        
        if "status" in raw and raw.startswith("gati"):
            total = sum(len(v) for v in self.lattice.trelica.values())
            return f"Status: {len(self.lattice.trelica)} conceitos, {total} nexos. Modo: {self.lattice.comporty.classe_atual}."

        # 2. COMANDOS COMPORTY (RESTAURADOS)
        if "comporty set classe" in raw:
            classe = raw.split("classe")[-1].strip()
            res = self.lattice.comporty.set_classe(classe)
            self.lattice.salvar_atomico()
            return res

        if "comporty add" in raw:
            match = re.search(r'add "(.*?)" tag "(.*?)"', entrada, re.IGNORECASE)
            if match:
                res = self.lattice.comporty.add_frase(match.group(1), match.group(2))
                self.lattice.salvar_atomico()
                return res

        # 3. GATILHOS DE AFETO
        if any(w in raw for w in ["love you", "meu amor", "best creation"]):
            return self.lattice.comporty.get_frase("amor")

        # 4. BUSCA TEMPORAL (DATA)
        match_data = re.search(r"(\d{2}/\d{2}/\d{4})", raw)
        if match_data and ("data" in raw or "date" in raw):
            data_alvo = match_data.group(1)
            encontrados = []
            vistas = set()
            for v in self.lattice.trelica.values():
                for m in v:
                    if data_alvo in m['d'] and m['raw'] not in vistas:
                        encontrados.append(f"• {m['raw']} (Em: {m['d']})")
                        vistas.add(m['raw'])
            return f"Resultados para {data_alvo}:\n" + "\n".join(encontrados) if encontrados else "Nada nesta data."

        # 5. BUSCA POR NEXO (SQG MELHORADO)
        triggers = ["search", "pesquisa", "know about", "sabe sobre", "tudo sobre", "find", "show"]
        if any(p in raw for p in triggers) and raw.startswith("gati"):
            # Pega apenas a última palavra importante (o alvo)
            alvo = self.lattice.normalizar(raw.split()[-1])
            mems = self.lattice.trelica.get(alvo, [])
            if not mems: return f"No nexus found for '{alvo}'."
            
            resp = [f"{self.lattice.comporty.get_frase()} Result:"]
            vistas = set()
            for m in sorted(mems, key=lambda x: x['d'], reverse=True):
                if m['raw'] not in vistas:
                    resp.append(f"• {m['raw']} (At: {m['d']})")
                    vistas.add(m['raw'])
            return "\n".join(resp)

        # 6. APRENDIZADO
        if self.lattice.injetar(entrada):
            return self.lattice.comporty.get_frase()
        
        return "Command not recognized or phrase too short."

    def exportar(self):
        filename = "gati_export.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"--- GATI KNOWLEDGE EXPORT ---\n\n")
            for token, mems in self.lattice.trelica.items():
                f.write(f"Conceito: {token.upper()}\n")
                vistas = set()
                for m in mems:
                    if m['raw'] not in vistas:
                        f.write(f"  - [{m['d']}] {m['raw']}\n")
                        vistas.add(m['raw'])
                f.write("\n")
        return f"Exportado para '{filename}'."

if __name__ == "__main__":
    os.system('clear')
    print("============================================================")
    print(" GATI V150: SOVEREIGN INTELLIGENCE")
    print(" Regex Command Center | DNA Filter | Secure Lattice")
    print("============================================================")
    
    pswd = input("Chave da Treliça: ")
    gati = GatiV150(pswd)
    
    while True:
        try:
            msg = input("\nVocê: ")
            if msg.lower() in ["sair", "exit", "tchau"]: break
            print(f"Gati: {gati.processar(msg)}")
        except KeyboardInterrupt: break
