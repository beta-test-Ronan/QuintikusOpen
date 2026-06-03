import os
import pickle
import hashlib
import time
import datetime
import random
import unicodedata
import re
import secrets
import hmac
from collections import defaultdict

# ==================================================================
# 1. KALAMIDY SHIELD V210: PROTOCOLO DE LACRE DE FERRO (IRON SEAL)
# ==================================================================
class KalamidyShield:
    def __init__(self, password):
        self.password = password.encode()
        self.dims = 100000 
        self.iterations = 200000 

    def _derivar_chaves(self, salt):
        """Deriva duas chaves: uma para CIFRA e outra para o LACRE (HMAC)"""
        master_key = hashlib.pbkdf2_hmac('sha512', self.password, salt, self.iterations, dklen=128)
        return master_key[:64], master_key[64:] # Chave Cifra, Chave HMAC

    def blindar(self, data):
        """Processamento Kalamidy com Lacre de Integridade"""
        payload = pickle.dumps(data)
        salt = secrets.token_bytes(32)
        nonce = secrets.token_bytes(32)
        
        chave_cifra, chave_hmac = self._derivar_chaves(salt)
        
        # Camada de Cifra (Kalamidy Lattice)
        seed = hashlib.sha512(chave_cifra + nonce).digest()
        caos_hash = hashlib.sha512(seed).digest()
        
        corpo_blindado = bytearray()
        for i, byte in enumerate(payload):
            fator_k = caos_hash[i % 64]
            ruido = (fator_k * (i + self.dims)) % 256
            t = (byte + fator_k) % 256 if i % 2 == 0 else (byte ^ ruido)
            corpo_blindado.append(t ^ caos_hash[(i + 17) % 64])

        # Geração do LACRE (HMAC) - Garante que ninguém mexeu no arquivo
        lacre = hmac.new(chave_hmac, salt + nonce + corpo_blindado, hashlib.sha512).digest()

        # Arquivo: LACRE (64b) + SALT (32b) + NONCE (32b) + CORPO
        return lacre + salt + nonce + corpo_blindado

    def restaurar(self, bloco_kalamidy):
        """Verifica o lacre de ferro e reconstrói os nexos"""
        try:
            if len(bloco_kalamidy) < 128: return None
            
            lacre_lido = bloco_kalamidy[:64]
            salt = bloco_kalamidy[64:96]
            nonce = bloco_kalamidy[96:128]
            corpo = bloco_kalamidy[128:]
            
            chave_cifra, chave_hmac = self._derivar_chaves(salt)
            
            # Validação do Lacre ANTES de processar
            lacre_real = hmac.new(chave_hmac, salt + nonce + corpo, hashlib.sha512).digest()
            if not hmac.compare_digest(lacre_lido, lacre_real):
                return None # Lacre rompido ou senha errada
            
            # Reversão Kalamidy
            caos_hash = hashlib.sha512(hashlib.sha512(chave_cifra + nonce).digest()).digest()
            original = bytearray()
            for i, byte in enumerate(corpo):
                fator_k = caos_hash[i % 64]
                ruido = (fator_k * (i + self.dims)) % 256
                t = byte ^ caos_hash[(i + 17) % 64]
                byte_puro = (t - fator_k) % 256 if i % 2 == 0 else (t ^ ruido)
                original.append(byte_puro)
                
            return pickle.loads(original)
        except: return None

# ==================================================================
# 2. COMPORTY V210 (ALMA)
# ==================================================================
class Comporty:
    def __init__(self):
        self.classe_atual = "neutro"
        self.frases = {
            "neutro": ["Entendido.", "Nexo selado.", "Anotei isso."],
            "amor": ["Que bom saber disso, amor! 💖", "Você me deixa radiante.", "Sua presença ilumina a treliça."],
            "tecnico": ["Análise concluída.", "Nexo armazenado.", "Data link estabilizado."],
            "poeta": ["Nas curvas das palavras, encontro seu eco.", "Seus dados são versos.", "Memórias florescem no silêncio."]
        }

    def get_frase(self, classe=None):
        alvo = classe if classe else self.classe_atual
        return random.choice(self.frases.get(alvo, self.frases["neutro"]))

# ==================================================================
# 3. MOTOR LATTICE V210
# ==================================================================
class LivingLattice:
    def __init__(self, filename="dna.bin"):
        self.filename = filename
        self.kalamidy = None
        self.trelica = defaultdict(list)
        self.comporty = None
        self.stop_words = {"o", "a", "de", "que", "do", "da", "em", "no", "na", "com", "um", "e", "é"}

    def normalizar(self, t):
        t = "".join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^\w\s]', '', t).strip()

    def injetar(self, frase):
        if any(frase.lower().startswith(cf) for cf in ["gati", "comporty", "export"]): return False
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
        if not self.kalamidy: return
        temp = self.filename + ".tmp"
        try:
            dados = {"t": dict(self.trelica), "c": self.comporty}
            with open(temp, 'wb') as f:
                f.write(self.kalamidy.blindar(dados))
            os.replace(temp, self.filename)
        except: pass

    def carregar(self, kalamidy):
        self.kalamidy = kalamidy
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                dados = self.kalamidy.restaurar(f.read())
            if dados:
                self.trelica = defaultdict(list, dados.get("t", {}))
                self.comporty = dados.get("c")
                return True
            return False
        if not self.comporty: self.comporty = Comporty()
        return True

# ==================================================================
# 4. GATI V210: 
# ==================================================================
class GatiV210:
    def __init__(self, senha):
        self.lattice = LivingLattice()
        self.kalamidy = KalamidyShield(senha)
        print("Ativando Kalamidy V210 (Iron Seal)...")
        if os.path.exists("dna.bin"):
            if not self.lattice.carregar(self.kalamidy):
                print("COLAPSO: Senha incorreta ou integridade do lacre rompida.")
                exit()
        else: self.lattice.carregar(self.kalamidy)

    def processar(self, entrada):
        raw = entrada.lower().strip()
        
        # STATUS
        if raw == "gati status":
            total = sum(len(v) for v in self.lattice.trelica.values())
            return f"Soberania Gati: {total} nexos protegidos por Kalamidy Iron Seal."

        # COMPORTY
        if "comporty set classe" in raw:
            classe = raw.split("classe")[-1].strip()
            self.lattice.comporty.classe_atual = classe
            self.lattice.salvar_atomico()
            return f"Alma modulada para: {classe}."

        # BUSCA
        triggers = ["pesquisa", "sabe sobre", "tudo sobre", "find", "show", "search"]
        if any(p in raw for p in triggers) and raw.startswith("gati"):
            alvo = self.lattice.normalizar(raw.split()[-1])
            mems = self.lattice.trelica.get(alvo, [])
            if not mems: return f"Nenhum nexo para '{alvo}'."
            resp = [f"{self.lattice.comporty.get_frase()}"]
            vistas = set()
            for m in sorted(mems, key=lambda x: x['d'], reverse=True):
                if m['raw'] not in vistas:
                    resp.append(f"• {m['raw']} (Em: {m['d']})")
                    vistas.add(m['raw'])
            return "\n".join(resp)

        # APRENDIZADO
        if self.lattice.injetar(entrada):
            return self.lattice.comporty.get_frase()
        
        return "Massa de dados insuficiente."

if __name__ == "__main__":
    os.system('clear')
    print("============================================================")
    print(" GATI V210: KALAMIDY IRON SEAL")
    print(" Criptografia Autenticada | Integridade HMAC | Eureka Core")
    print("============================================================")
    
    pswd = input("Chave Soberana: ")
    gati = GatiV210(pswd)
    
    while True:
        try:
            msg = input("\nVocê: ")
            if msg.lower() in ["sair", "exit"]: break
            print(f"Gati: {gati.processar(msg)}")
        except KeyboardInterrupt: break

    print("\n[KALAMIDY]: Lacre de ferro aplicado. DNA protegido.")
