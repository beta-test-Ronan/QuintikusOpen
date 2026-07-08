import numpy as np
import json
import os
import time

# ==============================================================================
# BLOCO 1: UTILITÁRIOS E FEELING-CORTEX (MOTOR v14)
# ==============================================================================

def l2_normalize(x):
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8) if norm > 0 else x

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

class TGNC_NeuralCortex_v14:
    def __init__(self, dim=128, rank=64):
        self.dim, self.rank = dim, rank
        self.lr, self.decay = 0.5, 0.999
        self.solo_path = "tgnc_cortex_v14.json"
        self.vocab, self.concepts, self.actions = {}, {}, {}
        self.experiencias = [] 
        self.W1 = np.random.randn(dim, rank) * 0.01 
        self.W2 = np.random.randn(rank, dim) * 0.01
        self.W_conf = np.random.randn(dim + dim, 1) * 0.01
        self.W_context, self.W_action = np.eye(dim), np.eye(dim)
        self.ruido = ["o", "a", "os", "as", "de", "da", "do", "um", "uma", "esta", "com", "que", "para", "em", "no", "na", "e"]
        self.carregar_cortex()

    def _get_vector(self, word, space="vocab"):
        target = {"vocab": self.vocab, "concepts": self.concepts, "actions": self.actions}[space]
        if word not in target: target[word] = l2_normalize(np.random.uniform(-1, 1, self.dim))
        return target[word]

    def processar_sequencia(self, texto):
        if not texto or not texto.strip(): return None, 0.0
        palavras = [w for w in texto.lower().replace(".", " . ").replace(",", "").split() if w not in self.ruido]
        if not palavras: return l2_normalize(np.mean([self._get_vector(p) for p in texto.lower().split()], axis=0)), 0.15
        h_states, prev_v = [], np.zeros(self.dim)
        for p in palavras:
            v_k = self._get_vector(p)
            h_k = l2_normalize(v_k + (prev_v @ self.W1) @ self.W2)
            h_states.append(h_k)
            prev_v = h_k
        context_vector = l2_normalize(np.mean(h_states, axis=0))
        conf_score = float(sigmoid(np.dot(np.concatenate([context_vector, prev_v]), self.W_conf.flatten())))
        return context_vector, conf_score

    def ensinar(self, texto_longo, nexo, acao, registrar=True):
        self.W_context *= self.decay; self.W_action *= self.decay
        x, _ = self.processar_sequencia(texto_longo)
        if x is None: return
        y_t, z_t = self._get_vector(nexo, "concepts"), self._get_vector(acao, "actions")
        for _ in range(60): 
            self.W_context += self.lr * np.outer(x, (y_t - x @ self.W_context))
            self.W_action += self.lr * np.outer(y_t, (z_t - y_t @ self.W_action))
        if registrar: self.experiencias.append({"t": texto_longo, "n": nexo, "a": acao})

    def analisar(self, texto):
        x, conf_n = self.processar_sequencia(texto)
        if x is None: return "SILÊNCIO", "NENHUMA", 0.0, "NEUTRO"
        v_n, v_a = l2_normalize(x @ self.W_context), l2_normalize((x @ self.W_context) @ self.W_action)
        nexo_f, acao_f = self._buscar_proximo(v_n, "concepts"), self._buscar_proximo(v_a, "actions")
        sim_geo = float(np.dot(v_n, self.concepts.get(nexo_f, np.zeros(self.dim))))
        conf_final = (sim_geo * 0.7) + (conf_n * 0.3)
        return nexo_f, acao_f, float(conf_final), "ESTÁVEL"

    def _buscar_proximo(self, vetor, space):
        target = {"concepts": self.concepts, "actions": self.actions}[space]
        if not target: return "Indefinido"
        melhor, sim = "Indefinido", -1.0
        for k, v in target.items():
            s = np.dot(vetor, v)
            if s > sim: sim, melhor = s, k
        return melhor

    def salvar_cortex(self):
        def conv(obj):
            if isinstance(obj, dict): return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        with open(self.solo_path, "w") as f: json.dump(conv(self.__dict__), f)

    def carregar_cortex(self):
        if os.path.exists(self.solo_path):
            try:
                with open(self.solo_path, "r") as f:
                    d = json.load(f)
                    for k, v in d.items():
                        if isinstance(v, dict): setattr(self, k, {ik: np.array(iv) for ik, iv in v.items()})
                        elif isinstance(v, list) and k not in ["ruido", "experiencias"]: setattr(self, k, np.array(v))
                        else: setattr(self, k, v)
            except: pass

# ==============================================================================
# BLOCO 2: MÓDULOS DE AGI (SNC, MUNDO E SEMÂNTICA)
# ==============================================================================

class FeelingModule:
    def __init__(self):
        self.path = "feeling_rules.json"
        self.rules = []
        self.carregar()

    def adicionar(self, perceber, v1, v2, mood):
        self.rules = [r for r in self.rules if not (r["perceber"] == perceber.upper() and r["v1"] == v1)]
        self.rules.append({"perceber": perceber.upper(), "v1": float(v1), "v2": float(v2), "mood": mood.upper()})
        self.salvar()

    def avaliar(self, nexo, conf, current):
        for r in self.rules:
            if r["perceber"] in nexo.upper() and r["v1"] <= conf <= r["v2"]: return r["mood"]
        return current

    def salvar(self):
        with open(self.path, "w") as f: json.dump(self.rules, f)

    def carregar(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f: self.rules = json.load(f)

class WorldModel:
    def __init__(self):
        self.path = "world_rules.json"
        self.rules = {}
        self.carregar()

    def simular(self, acao, props):
        rule = self.rules.get(acao.upper(), {"risco": 0.5})
        if props.get("fragil") and rule["risco"] > 0.4: return "PERIGO: Risco de quebra", rule["risco"]
        return "OK", 0.0

    def salvar(self):
        with open(self.path, "w") as f: json.dump(self.rules, f)

    def carregar(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f: self.rules = json.load(f)

class SemanticMemory:
    def __init__(self):
        self.path = "memoria_objetos.json"
        self.entities = {}
        self.carregar()

    def carregar(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f: self.entities = json.load(f)

# ==============================================================================
# BLOCO 3: CONSCIOUS CORE (O ORQUESTRADOR DE ONE-SHOT LEARNING)
# ==============================================================================

class ConsciousCore:
    def __init__(self):
        self.tgnc = TGNC_NeuralCortex_v14()
        self.world = WorldModel()
        self.memory = SemanticMemory()
        self.snc = FeelingModule()

    def pensar(self, entrada):
        print(f"\n--- 🧠 CICLO COGNITIVO ---")
        nexo, acao, conf, mood_n = self.tgnc.analisar(entrada)
        mood = self.snc.avaliar(nexo, conf, mood_n)
        
        objeto_alvo = self._extrair_objeto(entrada)
        props = self.memory.entities.get(objeto_alvo, {"fragil": False})
        
        info, risco = self.world.simular(acao, props)
        print(f"  [ESTADO]: Nexo: {nexo} | Ação: {acao} | Humor: {mood} | Risco: {risco}")
        
        if conf < 0.4: return f"❓ [CURIOSIDADE]: Pouca confiança ({conf:.2f})."
        
        decisao = "INTERROMPER" if risco > 0.6 else "EXECUTAR"
        return f"🤖 STATUS: {mood}\n🎯 DECISÃO: {decisao} ({info})"

    def aprender_unificado(self, partes):
        # aprender|quadro|MARCENARIA|BATER|fragil:False|massa:1.5|risco:0.3|sentimento:SEGURO|0.6|0.9
        try:
            txt, nexo, acao = partes[1], partes[2], partes[3]
            
            # 1. Comportamento (Neural)
            self.tgnc.ensinar(txt, nexo, acao)
            self.tgnc.salvar_cortex()
            
            # 2. Objeto e Mundo (Simbólico)
            if partes[1].lower() not in self.memory.entities: self.memory.entities[partes[1].lower()] = {}
            
            for p in partes[4:]:
                if ":" in p:
                    k, v = p.split(":")
                    if k == "risco":
                        self.world.rules[acao.upper()] = {"risco": float(v)}
                    elif k == "sentimento":
                        # sentimento:NOME está na mesma 'parte', os limites vêm depois
                        mood_name = v
                        v_min = partes[partes.index(p)+1]
                        v_max = partes[partes.index(p)+2]
                        self.snc.adicionar(nexo, v_min, v_max, mood_name)
                    else:
                        # Propriedades genéricas (fragil, massa, etc)
                        val = (v.lower() == "true") if v.lower() in ["true", "false"] else v
                        self.memory.entities[partes[1].lower()][k] = val

            with open(self.memory.path, "w") as f: json.dump(self.memory.entities, f)
            self.world.salvar()
            print(f"🌟 [ONE-SHOT]: Organismo evoluiu com '{txt}'.")
        except Exception as e:
            print(f"❌ Erro no aprendizado: {e}")

    def _extrair_objeto(self, texto):
        for k in self.memory.entities.keys():
            if k in texto.lower(): return k
        return "desconhecido"

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    agi = ConsciousCore()
    print("\n--- 🧠 AGI CORE v120 ATIVA ---")
    print("Use o comando One-Shot:")
    print("aprender|quadro|MARCENARIA|BATER|fragil:False|massa:1.5|risco:0.3|sentimento:SEGURO|0.6|0.9")
    
    while True:
        try:
            cmd = input("\nLog > ").strip()
            if cmd.lower() == 'sair': break
            partes = cmd.split("|")
            
            if partes[0] == "aprender":
                agi.aprender_unificado(partes)
            elif cmd.lower() == 'consolidar':
                agi.tgnc.consolidar_mente()
            else:
                print(agi.pensar(cmd))
        except Exception as e:
            print(f"Erro: {e}")
