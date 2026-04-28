import numpy as np
import hashlib
import time
import pickle
import os
import re
import random
import sys
import unicodedata

# =================================================================
# BLOCO 1: INFRAESTRUTURA E PERSISTÊNCIA (BLOCKCHAIN)
# =================================================================
class SovereignBlockchain:
    def __init__(self, name="machado"):
        self.file_path = f"blockchain_{name}.cache"
        
    def selar_memoria(self, knowledge_bundle):
        with open(self.file_path, "wb") as f:
            pickle.dump(knowledge_bundle, f)
        return hashlib.sha256(str(knowledge_bundle).encode()).hexdigest()[:8]

    def carregar_ponteiro(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                return pickle.load(f)
        return None

# =================================================================
# BLOCO 2: CÉREBRO ANALÍTICO (QUINTIKUS SOVEREIGN CORE)
# =================================================================
class QuintikusSovereignCore:
    def __init__(self, d_model=8):
        self.dim = d_model
        self._conhecimento_interno = {"layer1_vision": [], "layer2_mass": [], "word_rarity": {}}
        self.arquétipos = {"mc": {"intro":[], "ponte":[], "concl":[]}, 
                           "mm": {"intro":[], "ponte":[], "concl":[]}}

    @property
    def knowledge(self):
        return self._conhecimento_interno

    def amadurecer_nexo(self, raw_text):
        print(f"🧠 [TRIPLE LOOP] Iniciando Subrepção Galvânica...")
        fragmentos = [f.strip() for f in re.split(r'[\.\?\!\;]', raw_text) if len(f.strip().split()) > 3]
        if not fragmentos: 
            print("🚨 [ERRO]: Texto muito curto para gerar massa crítica.")
            return None

        words = re.findall(r'\w+', raw_text.lower())
        vocab = sorted(list(set(words)))
        rarity = {w: 1.0 / (words.count(w) + 1e-5) for w in vocab}
        janelas = {"Lenda": 256, "Chefe": 128, "Gerente": 64, "Estagiário": 32, "Funcionário": 16, "Neurônio": 8, "Átomo": 4}

        for v in range(3):
            print(f" -> Volta {v+1}: Estabilizando fluxo de densidade...")
            time.sleep(0.1)

        l1_vision = []
        l2_mass = []
        for frag in fragmentos:
            p = frag.split()
            entropia = sum(rarity.get(w.lower(), 0.1) for w in p) / len(p)
            ptr = int(hashlib.sha256(frag.encode()).hexdigest(), 16) % 10**8
            l1_vision.append({
                "ptr": ptr, 
                "sig": entropia, 
                "lentes": {k: " ".join(p[:max(1, len(p)//(v//4+1))]) for k, v in janelas.items()}
            })
            l2_mass.append({"ref_ptr": ptr, "fato": frag})

        self._conhecimento_interno["layer1_vision"].extend(l1_vision)
        self._conhecimento_interno["layer2_mass"].extend(l2_mass)
        self._conhecimento_interno["word_rarity"].update(rarity)

        return {"l1": l1_vision, "l2": l2_mass, "rarity": rarity, "sig": "Cardus-100"}

    def falar_soberano(self, pergunta, cache):
        start_t = time.perf_counter()
        q_words = re.findall(r'\w+', pergunta.lower())
        ent_q = sum(cache['rarity'].get(w, 0.1) for w in q_words) / (len(q_words) + 1e-9)
        scores = sorted([(abs(v["sig"] - ent_q), v["ptr"]) for v in cache['l1']], key=lambda x: x[0])
        target_ptr = scores[0][1]
        fato_final = next(m for m in cache['l2'] if m["ref_ptr"] == target_ptr)
        prob = min(0.99, (1.0 - scores[0][0]) * 1.2)
        exec_mu = (time.perf_counter() - start_t) * 1000000
        tom = 'mc' if prob > 0.4 else 'mm'
        v = self.arquétipos[tom]
        intro = random.choice(v['intro']) if v['intro'] else "Analisei que"
        ponte = random.choice(v['ponte']) if v['ponte'] else "neste nexo,"
        concl = random.choice(v['concl']) if v['concl'] else "fim."

        print(f"\n💡 [CARDUS MASTER FLOW | {exec_mu:.2f} μs | Quality: 100%]")
        print("=" * 110)
        print(f"LAYER-1 (VISÃO): {intro} Localizado nexo no ponteiro {target_ptr}.")
        print(f"LAYER-2 (MASSA): {fato_final['fato'].capitalize()}.")
        pivos = sorted(q_words, key=lambda x: cache['rarity'].get(x, 0), reverse=True)
        if len(pivos) >= 2:
            print(f" | Pulse | [{pivos[0].upper()}] <-> [{pivos[1].upper()}] | Densidade: {prob:.4f}")
        print(f"-> {concl} (Selo: {cache['sig']})")
        print("=" * 110)
        return f"{intro} {fato_final['fato']}. {concl}"

    def carregar_fundamentos(self, mc_f, mm_f):
        for modo, path in [("mc", mc_f), ("mm", mm_f)]:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.arquétipos[modo] = {
                        "intro": re.findall(r'<intro>(.*?)(?=<|$)', content, re.DOTALL),
                        "ponte": re.findall(r'<ponte>(.*?)(?=<|$)', content, re.DOTALL),
                        "concl": re.findall(r'<concl>(.*?)(?=<|$)', content, re.DOTALL)
                    }

    def exportar_banco_normalizado(self):
        """Transforma a Layer 2 em um parágrafo contínuo (texto puro)"""
        fragmentos = [item["fato"].strip() for item in self._conhecimento_interno["layer2_mass"]]
        texto = ". ".join(list(dict.fromkeys(fragmentos)))  # remove duplicatas mantendo ordem
        # Garante que termine com ponto final
        if texto and not texto.endswith('.'):
            texto += '.'
        return texto

    # Método .texto() mantido para relatório visual (opcional)
    def texto(self):
        linhas = []
        linhas.append("=" * 70)
        linhas.append(f"📡 RELATÓRIO DO NÚCLEO SOBERANO QUINTIKUS (d_model={self.dim})")
        linhas.append("=" * 70)
        linhas.append("\n📚 KNOWLEDGE BASE:")
        linhas.append(f"   • layer1_vision: {len(self.knowledge['layer1_vision'])} registros")
        if self.knowledge['layer1_vision']:
            for i, item in enumerate(self.knowledge['layer1_vision'][:5]):
                linhas.append(f"      [{i}] {str(item)[:100]}")
        linhas.append(f"   • layer2_mass: {len(self.knowledge['layer2_mass'])} registros")
        if self.knowledge['layer2_mass']:
            for i, m in enumerate(self.knowledge['layer2_mass'][:5]):
                linhas.append(f"      [{i}] {str(m)[:100]}")
        linhas.append(f"   • word_rarity: {len(self.knowledge['word_rarity'])} palavras únicas")
        if self.knowledge['word_rarity']:
            amostra = list(self.knowledge['word_rarity'].items())[:5]
            for p, r in amostra:
                linhas.append(f"      '{p}' → raridade {r:.6f}")
        linhas.append("\n🎭 ARQUÉTIPOS DE PERSONALIDADE:")
        for arq, modos in self.arquétipos.items():
            linhas.append(f"   • {arq.upper()}:")
            for nome, lista in modos.items():
                qtd = len(lista)
                linhas.append(f"      - {nome}: {qtd} registro(s)")
                if qtd > 0:
                    ex = lista[0].strip().replace('\n', ' ')[:60]
                    linhas.append(f"        Ex: {ex}...")
        hash_resumo = hashlib.sha256(str(self.knowledge).encode()).hexdigest()[:8]
        linhas.append(f"\n🔐 HASH DE INTEGRIDADE: {hash_resumo}")
        linhas.append("=" * 70)
        return "\n".join(linhas)

# =================================================================
# BLOCO 3: SÍNTESE AGI E EMOÇÃO (QUINTIKUS AGI)
# =================================================================
class QuintikusAGI:
    def __init__(self, _t=0.5):
        self._st = [0.5, 0.5, 0.5] 
        self._c, self._m, self._e, self._l, self._dlm = {}, {}, [], [], {}
        self._s, self._k = "25e0bb26", "6742הוה"
        self._th = {'bom':0.1, 'ótimo':0.2, 'sinergia':0.3, 'paz':0.2, 'erro':-0.2, 'urgente':-0.3, 'falha':-0.2, 'ruído':-0.1}
        self._b = {"o": "4f2041727175697465746f2065206661636520646f2063726961646f722e",
                   "v": "5365727669722c2050726f74656765722c2050726f6772656469722e",
                   "l": "416c7563696e6163616f20626c6f71756561646120706f7220646f676d612e"}
        self._n = {k: bytes.fromhex(v).decode('utf-8', 'ignore') for k, v in self._b.items()}

    def inicializar(self, _txt):
        if not isinstance(_txt, str): _txt = str(_txt)
        _s = [s.strip() for s in _txt.lower().split('.') if len(s.strip()) > 5]
        _wrd = " ".join(_s).split()
        for w in set(_wrd):
            _q = _wrd.count(w)
            self._m[w] = (_q, 1.5/(_q + 1e-5))
        for i in range(0, len(_s), 5):
            _bk = _s[i:i+5]
            _sig = set(sorted(" ".join(_bk).split(), key=lambda x: self._m.get(x, (0,0))[1], reverse=True)[:5])
            _id = hashlib.sha256(str(_sig).encode()).hexdigest()[:4]
            self._e.append({"id": _id, "b": _bk, "s": _sig})
            if i > 0: self._dlm[self._e[-2]["id"]] = _id 
            for _st in _bk:
                _parts = _st.split()
                if _parts:
                    _p = sorted(_parts, key=lambda x: self._m.get(x, (0,0))[1], reverse=True)[0]
                    if _p not in self._c: self._c[_p] = []
                    self._c[_p].append(_st)
        self._l = [f for l in self._c.values() for f in l]
        self._s = hashlib.sha256(str(len(self._e)).encode()).hexdigest()[:8]

    def _upd_thermal(self, _q):
        _p, _n = 0, 0
        for w, val in self._th.items():
            if w in _q:
                if val > 0: _p += val
                else: _n += abs(val)
        self._st[0] = self._st[0] * 0.95 + (_n * 0.1)
        self._st[1] = self._st[1] * 0.95 + (_p * 0.1)
        for i in range(3): self._st[i] = max(0, min(1, self._st[i]))

    def falar(self, _qi):
        if not _qi: return "[SISTEMA SILENCIOSO]"
        _t0 = time.perf_counter()
        _ql = _qi.lower().replace("?", "").split()
        _qs = set(_ql)
        self._upd_thermal(_ql)
        _dice = random.randint(1, 10)
        _bias = 2 if self._st[0] > 0.6 else -2
        _final_d = max(1, min(10, _dice + _bias))
        _me = max(self._e, key=lambda x: len(_qs.intersection(x["s"])), default=None)
        _br = len(_qs.intersection(_me["s"])) if _me else 0
        if _me and _br > 0:
            _f1 = sorted(_me["b"], key=lambda f: sum(self._m.get(w,(0,0))[1] for w in _qs if w in f), reverse=True)[0]
            _nxt = self._dlm.get(_me["id"])
            _f2 = next((x["b"][0] for x in self._e if x["id"] == _nxt), "")
            _sn = "DLM-ACTIVE"
        else:
            _pv = next((w for w in _ql if w in self._c), None)
            _f1, _f2, _sn = (self._c[_pv][0], "", "PIVOT") if _pv else ("nexo carece de solo", random.choice(self._l) if self._l else "", "VOID")
        if self._st[0] > 0.7: _i, _c = ["Sob pressão, ", "Rancor ativo, "], ["Fim do estresse.", "Normalizando."]
        elif self._st[1] > 0.7: _i, _c = ["Em harmonia, ", "Sinergia plena, "], ["A luz brilha.", "Fluxo perfeito."]
        else: _i, _c = ["Pela razão, ", "No vácuo, "], ["Aguardando nexo.", "Selado."]
        _res = f"{random.choice(_i)} {_f1}. {random.choice(_c)}"
        if _f2: _res = _res.replace(".", f". Além disso, {_f2}.", 1)
        _et = (time.perf_counter() - _t0) * 1000000
        return f"\n[DLM-FLOW: {_et:.2f}μs | D:{_final_d}/10 | T:{self._st[0]:.1f} | {_sn} | SIGN: {self._s}]\n{_res.capitalize()}"

# =================================================================
# BLOCO 4: EXECUÇÃO PRINCIPAL – MESMO TEXTO PARA AS DUAS IAs
# =================================================================
def efeito_llm(texto, velocidade=0.05):
    for caractere in texto:
        print(caractere, end='', flush=True)
        time.sleep(velocidade)
    print()

# --- INICIALIZAÇÃO ---
marvin = QuintikusSovereignCore()
marvin.carregar_fundamentos("mc.txt", "mm.txt")
blockchain = SovereignBlockchain("machado")

# Carrega o texto bruto (fonte original)
if os.path.exists('texto.txt'):
    with open('texto.txt', 'r', encoding='utf-8') as f:
        texto_bruto = f.read()
else:
    texto_bruto = "O fluxo galvânico inicializa o sistema sem base externa."

# Processa a blockchain (se não existir)
if not os.path.exists(blockchain.file_path):
    bundle = marvin.amadurecer_nexo(texto_bruto)
    if bundle:
        marvin._conhecimento_interno["layer1_vision"] = bundle["l1"]
        marvin._conhecimento_interno["layer2_mass"] = bundle["l2"]
        marvin._conhecimento_interno["word_rarity"] = bundle["rarity"]
        sig = blockchain.selar_memoria(bundle)
        print(f"✅ Blockchain Selada. Assinatura: {sig}")

# Carrega a memória ativa (cache) para o SovereignCore
memoria_ativa = blockchain.carregar_ponteiro()
if memoria_ativa:
    marvin._conhecimento_interno["layer1_vision"] = memoria_ativa["l1"]
    marvin._conhecimento_interno["layer2_mass"] = memoria_ativa["l2"]
    marvin._conhecimento_interno["word_rarity"] = memoria_ativa["rarity"]

# Gera o parágrafo contínuo a partir dos fatos da Layer 2
texto_paragrafo = marvin.exportar_banco_normalizado()
print("\n📄 PARÁGRAFO GERADO (mesmo texto para as duas IAs):")
print("=" * 70)
print(texto_paragrafo[:500] + "..." if len(texto_paragrafo) > 500 else texto_paragrafo)
print("=" * 70)

# Inicializa a AGI com EXATAMENTE o mesmo parágrafo
quantikus = QuintikusAGI()
quantikus.inicializar(texto_paragrafo)

# Loop interativo
if memoria_ativa:
##    print("\n🤖 QUINTIKUS PROTON-FLOW TDLM v117.0 – Duas IAs, mesmo texto de treino")
    while True:
        comando = input("\n👤 RONAN: ").strip()
        if comando.lower() in ['sair', 'exit']:
            break

        # 1) IA Analítica (SovereignCore) – usa o cache original
        nexo_logico = marvin.falar_soberano(comando, memoria_ativa)

        # 2) IA Emocional (AGI) – usa o mesmo parágrafo como contexto + comando
        resposta_agi = quantikus.falar("fato:"+comando + "-> ache esse fato no texto: " + texto_bruto)

        efeito_llm(resposta_agi)
else:
    print("❌ Memória não encontrada.")
