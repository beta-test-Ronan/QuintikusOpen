import numpy as np
import json
import os
import time

# ==============================================================================
# BLOCO 1: UTILITÁRIOS E FEELING-CORTEX (O SEU MOTOR v14)
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
        mood = "ALERTA" if any(c in texto.lower() for c in ["perigo", "quebrar", "bater", "erro"]) else "ESTÁVEL"
        return nexo_f, acao_f, float(conf_final), mood

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
# BLOCO 2: MÓDULOS DE AGI PERSISTENTES (MUNDO, MEMÓRIA E SENTIMENTOS)
# ==============================================================================

class FeelingModule:
    """ SNC: Sistema Nervoso Central para regras de sentimentos """
    def __init__(self):
        self.path = "feeling_rules.json"
        self.rules = [] # List of dicts: {"perceber": nexo/txt, "v1": float, "v2": float, "mood": str}
        self.carregar()

    def adicionar_regra(self, perceber, v1, v2, mood):
        nova_regra = {"perceber": perceber.upper(), "v1": float(v1), "v2": float(v2), "mood": mood.upper()}
        self.rules.append(nova_regra)
        self.salvar()
        print(f"❤️ [SNC]: Sentimento '{mood}' mapeado para '{perceber}'.")

    def avaliar(self, nexo, conf, current_mood):
        for r in self.rules:
            if r["perceber"] in nexo.upper():
                if r["v1"] <= conf <= r["v2"]:
                    return r["mood"]
        return current_mood

    def salvar(self):
        with open(self.path, "w") as f: json.dump(self.rules, f)

    def carregar(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f: self.rules = json.load(f)

class WorldModel:
    def __init__(self):
        self.path = "world_rules.json"
        self.rules = {"BATER": {"risco": 0.8, "consequencia": "deforma/quebra"}}
        self.carregar()

    def simular(self, acao, objeto_props):
        rule = self.rules.get(acao.upper(), {"risco": 0.5})
        if objeto_props.get("fragil") and rule["risco"] > 0.3:
            return f"PERIGO: {rule.get('consequencia', 'Dano iminente')}", rule["risco"]
        return "SEGURO", 0.0

    def salvar(self):
        with open(self.path, "w") as f: json.dump(self.rules, f)

    def carregar(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f: self.rules = json.load(f)

class SemanticMemory:
    def __init__(self):
        self.path = "memoria_objetos.json"
        self.entities = {"copo": {"fragil": True}}
        self.carregar()

    def salvar(self):
        with open(self.path, "w") as f: json.dump(self.entities, f)

    def carregar(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f: self.entities = json.load(f)

# ==============================================================================
# BLOCO 3: CONSCIOUS CORE (O MAESTRO)
# ==============================================================================

class ConsciousCore:
    def __init__(self):
        self.tgnc = TGNC_NeuralCortex_v14()
        self.world = WorldModel()
        self.memory = SemanticMemory()
        self.snc = FeelingModule()
        self.objetivos = ["PRESERVAR_OBJETOS"]

    def pensar(self, entrada):
        print(f"\n--- 🧠 CICLO COGNITIVO ATIVO ---")
        nexo, acao, conf, mood_neural = self.tgnc.analisar(entrada)
        
        # O SNC avalia se deve sobrepor o sentimento neural baseado nas regras
        mood_final = self.snc.avaliar(nexo, conf, mood_neural)
        
        objeto_alvo = self._extrair_entidade(entrada)
        props = self.memory.entities.get(objeto_alvo, {"fragil": False})

        print(f"  [ESTADO]: Confiança: {conf:.2f} | Sentimento: {mood_final}")
        
        if conf < 0.4 or nexo == "Indefinido":
            return "❓ [CURIOSIDADE]: Explique melhor ou use 'ensinar'."

        previsao, risco = self.world.simular(acao, props)
        
        if risco > 0.6:
            decisao = f"INTERROMPER: Risco para {objeto_alvo}."
        else:
            decisao = f"EXECUTAR: {acao} (Nexo: {nexo})."

        self.tgnc.salvar_cortex()
        return f"\n🤖 MOOD: {mood_final}\n🎯 DECISÃO: {decisao}"

    def _extrair_entidade(self, texto):
        for k in self.memory.entities.keys():
            if k in texto.lower(): return k
        return "desconhecido"

# ==============================================================================
# EXECUÇÃO DA AGI v110
# ==============================================================================

if __name__ == "__main__":
    agi = ConsciousCore()
    print("\n--- 🧠 AGI CONSCIOUS CORE v110 ---")
    print("Comandos:\n- ensinar|frase|nexo|acao\n- objeto|nome|fragil(true/false)\n- regra|acao|risco|desc\n- sentimento|perceber|val_min|val_max|nome")
    
    while True:
        try:
            user_input = input("\nLog > ").strip()
            if user_input.lower() == 'sair': break
            
            partes = user_input.split("|")
            
            if partes[0] == "ensinar" and len(partes) == 4:
                agi.tgnc.ensinar(partes[1], partes[2], partes[3])
                agi.tgnc.salvar_cortex()
                print("🎓 Neural aprendido.")
            elif partes[0] == "objeto" and len(partes) == 3:
                agi.memory.entities[partes[1].lower()] = {"fragil": (partes[2].lower() == "true")}
                agi.memory.salvar()
                print("📦 Objeto salvo.")
            elif partes[0] == "regra" and len(partes) == 4:
                agi.world.rules[partes[1].upper()] = {"risco": float(partes[2]), "consequencia": partes[3]}
                agi.world.salvar()
                print("⚖️ Regra salva.")
            elif partes[0] == "sentimento" and len(partes) == 5:
                # perceber > valor x > valor y = sentimento
                agi.snc.adicionar_regra(partes[1], partes[2], partes[3], partes[4])
            else:
                print(agi.pensar(user_input))

        except Exception as e:
            print(f"Erro: {e}")
