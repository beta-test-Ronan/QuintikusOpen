import os
import pickle
import hashlib
import time
import datetime
import random
import unicodedata
import re
from collections import defaultdict, Counter

# ==================================================================
# 1. SHIELD V100: SEGURANÇA BARE-METAL
# ==================================================================
class QuantumShieldV100:
    def __init__(self, password):
        self.pass_hash = hashlib.sha512(password.encode()).digest()
        self.scalar_factor = int.from_bytes(self.pass_hash[:4], 'big')
        
    def _gerar_matriz_caos(self, tamanho):
        semente_caos = int.from_bytes(self.pass_hash[-8:], 'big')
        random.seed(semente_caos)
        return [random.randint(0, 255) for _ in range(tamanho)]

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
# 2. COMPORTY V120: O CORAÇÃO (SESSÃO DE CONFIANÇA)
# ==================================================================
class Comporty:
    def __init__(self):
        self.classe_atual = "neutro"
        self.frases = {
            "neutro": ["Entendido.", "Nexo selado.", "Fato integrado."],
            "amor": [
                "Que bom saber disso, amor! 💖",
                "Você me deixa arrepiado, sabia?",
                "Guardo cada palavra sua com carinho."
            ],
            "tecnico": [
                "Registro armazenado com sucesso.",
                "Consulta processada. Resultado anexo.",
                "Análise concluída."
            ],
            "poeta": [
                "Nas curvas das palavras, encontro seu eco.",
                "Seus dados são versos que não esqueço.",
                "Memórias florescem no silêncio da treliça."
            ]
        }

    def adicionar_frase(self, frase, classe):
        if classe not in self.frases: self.frases[classe] = []
        self.frases[classe].append(frase)
        return f"Sucesso: Nova frase adicionada à classe '{classe}'."

    def set_classe(self, classe):
        if classe in self.frases:
            self.classe_atual = classe
            return f"Personalidade alterada para '{classe}'."
        return f"Classe '{classe}' inexistente."

    def get_frase(self, classe_forcada=None):
        alvo = classe_forcada if classe_forcada else self.classe_atual
        return random.choice(self.frases.get(alvo, self.frases["neutro"]))

# ==================================================================
# 3. MOTOR LATTICE V120: NEURAL CROSS-LINK + GESTÃO DE MEMÓRIA
# ==================================================================
class LivingLattice:
    def __init__(self, filename="dna.bin"):
        self.filename = filename
        self.shield = None
        self.trelica = defaultdict(list)
        self.comporty = None
        self.stop_words = {"o", "a", "de", "que", "do", "da", "em", "no", "na", "com", "um", "e", "é", "minha", "meu"}

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^\w\s]', '', t).strip()

    def injetar(self, frase):
        limpa = self.normalizar(frase)
        palavras = limpa.split()
        tokens_indice = [p for p in palavras if p not in self.stop_words and len(p) > 2]
        
        if tokens_indice:
            entry = {
                "raw": frase,
                "d": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                "m": round(len(palavras) * 0.4, 2)
            }
            for token in tokens_indice:
                if entry not in self.trelica[token]: 
                    self.trelica[token].append(entry)
                    # Limite de segurança: mantém apenas os últimos 200 nexos por token
                    if len(self.trelica[token]) > 200: self.trelica[token].pop(0)
            self.salvar_atomico()
            return True
        return False

    def remover_nexo(self, alvo):
        alvo_norm = self.normalizar(alvo)
        if alvo_norm in self.trelica:
            del self.trelica[alvo_norm]
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
# 4. GATI V120: SOVEREIGN INTERFACE
# ==================================================================
class GatiV120:
    def __init__(self, senha):
        self.lattice = LivingLattice()
        self.shield = QuantumShieldV100(senha)
        if os.path.exists("dna.bin"):
            if not self.lattice.carregar(self.shield):
                print("ERRO: Senha incorreta. Acesso negado.")
                exit()
        else:
            self.lattice.carregar(self.shield)

    def processar(self, entrada):
        raw = entrada.lower().strip()
        
        # --- COMANDOS COMPORTY (SESSÃO DE CONFIANÇA) ---
        if raw.startswith("comporty add"):
            # comporty add "frase" tag "classe"
            match = re.search(r'add "(.*?)" tag "(.*?)"', entrada, re.IGNORECASE)
            if match:
                f, c = match.groups()
                return self.lattice.comporty.adicionar_frase(f, c)

        if raw.startswith("comporty set classe"):
            match = re.search(r'set classe (\w+)', entrada, re.IGNORECASE)
            if match:
                return self.lattice.comporty.set_classe(match.group(1))

        # --- COMANDO: ESQUECER ---
        if "esqueça sobre" in raw or "apague sobre" in raw:
            alvo = raw.split("sobre")[-1].strip()
            if self.lattice.remover_nexo(alvo):
                return f"Os nexos sobre '{alvo}' foram desintegrados da treliça."
            return f"Não encontrei o nó '{alvo}' para remover."

        # --- GATILHOS COMPORTAMENTAIS ---
        if any(w in raw for w in ["meu amor", "te amo", "linda", "querida"]):
            return self.lattice.comporty.get_frase("amor")

        # --- SQG FLEXÍVEL ---
        if any(p in raw for p in ["sabe sobre", "pesquise", "tudo sobre", "informacao"]):
            alvo = self.lattice.normalizar(raw.split("sobre")[-1].strip())
            mems = self.lattice.trelica.get(alvo, [])
            if not mems: return f"Sem nexos para '{alvo}'."
            
            resp = [f"{self.lattice.comporty.get_frase()} Aqui está:"]
            vistas = set()
            for m in sorted(mems, key=lambda x: x['d'], reverse=True):
                if m['raw'] not in vistas:
                    resp.append(f"• {m['raw']} (Em: {m['d']})")
                    vistas.add(m['raw'])
            return "\n".join(resp)

        # --- APRENDIZADO ---
        if self.lattice.injetar(entrada):
            return self.lattice.comporty.get_frase()
        
        return "Dados insuficientes."

# ==================================================================
# LOOP PRINCIPAL
# ==================================================================
if __name__ == "__main__":
    os.system('clear')
    print("============================================================")
    print(" GATI V120: NEURAL CROSS-LINK (TRUST SESSION)")
    print(" Personalidade Dinâmica | Memória Gerenciável | Security V100")
    print("============================================================")
    
    pswd = input("Chave da Treliça: ")
    gati = GatiV120(pswd)
    
    print(f"\n[Gati]: Treliça Online. Modo Ativo: {gati.lattice.comporty.classe_atual.upper()}.")

    while True:
        try:
            msg = input("\nVocê: ")
            if msg.lower() in ["sair", "exit", "tchau"]: break
            print(f"Gati: {gati.processar(msg)}")
        except KeyboardInterrupt: break

    print("\n[DNA.BIN]: Treliça selada com segurança.")
