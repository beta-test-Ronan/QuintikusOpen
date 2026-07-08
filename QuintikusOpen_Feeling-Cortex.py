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
        self.dim = dim
        self.rank = rank 
        self.lr = 0.5 
        self.decay = 0.999
        self.solo_path = "tgnc_cortex_v14.json"
        
        self.vocab, self.concepts, self.actions = {}, {}, {}
        self.experiencias = [] 
        
        self.W1 = np.random.randn(dim, rank) * 0.01 
        self.W2 = np.random.randn(rank, dim) * 0.01
        self.W_conf = np.random.randn(dim + dim, 1) * 0.01
        self.W_context = np.eye(dim) 
        self.W_action = np.eye(dim)  
        
        self.ruido = ["o", "a", "os", "as", "de", "da", "do", "um", "uma", "esta", "com", "que", "para", "em", "no", "na", "e"]
        self.carregar_cortex()

    def _get_vector(self, word, space="vocab"):
        target = {"vocab": self.vocab, "concepts": self.concepts, "actions": self.actions}[space]
        if word not in target:
            target[word] = l2_normalize(np.random.uniform(-1, 1, self.dim))
        return target[word]

    def processar_sequencia(self, texto):
        if not texto or not texto.strip(): return None, 0.0
        palavras = [w for w in texto.lower().replace(".", " . ").replace(",", "").split() if w not in self.ruido]
        if not palavras:
            h_pivot = [self._get_vector(p, "vocab") for p in texto.lower().split()]
            return l2_normalize(np.mean(h_pivot, axis=0)), 0.15
        h_states, prev_v = [], np.zeros(self.dim)
        for p in palavras:
            v_k = self._get_vector(p)
            bias = (prev_v @ self.W1) @ self.W2
            h_k = l2_normalize(v_k + bias)
            h_states.append(h_k)
            prev_v = h_k
        context_vector = l2_normalize(np.mean(h_states, axis=0))
        concat_conf = np.concatenate([context_vector, prev_v])
        conf_score = float(sigmoid(np.dot(concat_conf, self.W_conf.flatten())))
        return context_vector, conf_score

    def ensinar(self, texto_longo, nexo, acao, registrar=True):
        self.W_context *= self.decay
        self.W_action *= self.decay
        x, _ = self.processar_sequencia(texto_longo)
        if x is None: return
        y_target = self._get_vector(nexo, "concepts")
        z_target = self._get_vector(acao, "actions")
        for _ in range(60): 
            self.W_context += self.lr * np.outer(x, (y_target - x @ self.W_context))
            self.W_action += self.lr * np.outer(y_target, (z_target - y_target @ self.W_action))
        if registrar:
            self.experiencias.append({"t": texto_longo, "n": nexo, "a": acao})
            if len(self.experiencias) > 20: self.experiencias.pop(0)

    def consolidar_mente(self):
        if not self.experiencias: return
        print("🌙 [SONHO]: Consolidando sinapses...")
        for exp in self.experiencias:
            self.ensinar(exp['t'], exp['n'], exp['a'], registrar=False)

    def analisar(self, texto):
        x, conf_neural = self.processar_sequencia(texto)
        if x is None: return "SILÊNCIO", "NENHUMA", 0.0, "NEUTRO"
        v_nexo = l2_normalize(x @ self.W_context)
        v_acao = l2_normalize(v_nexo @ self.W_action)
        nexo_f = self._buscar_proximo(v_nexo, "concepts")
        acao_f = self._buscar_proximo(v_acao, "actions")
        sim_geo = float(np.dot(v_nexo, self.concepts.get(nexo_f, np.zeros(self.dim))))
        conf_final = (sim_geo * 0.7) + (conf_neural * 0.3)
        valencia = "ESTÁVEL"
        if conf_final < 0.3: valencia = "CONFUSO"
        elif any(c in texto.lower() for c in ["perigo", "erro", "ataque", "quebrar"]): valencia = "ALERTA"
        elif conf_final > 0.8: valencia = "SEGURO"
        return nexo_f, acao_f, float(conf_final), valencia

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
# BLOCO 2: MÓDULOS DE AGI (MODELO DE MUNDO, MEMÓRIA E PLANEJAMENTO)
# ==============================================================================

class WorldModel:
    """ Req 1: Entende causa e efeito física/lógica """
    def __init__(self):
        self.rules = {
            "BATER": {"impacto": "alto", "risco": 0.8, "consequencia": "deforma/quebra"},
            "OLHAR": {"impacto": "nulo", "risco": 0.0, "consequencia": "informação"},
            "EMPURRAR": {"impacto": "medio", "risco": 0.4, "consequencia": "deslocamento"}
        }

    def simular(self, acao, objeto_props):
        rule = self.rules.get(acao.upper(), {"risco": 0.5})
        if objeto_props.get("fragil") and rule["risco"] > 0.3:
            return "PERIGO: Objeto pode ser destruído", rule["risco"]
        return "SEGURO: Ação permitida", 0.0

class SemanticMemory:
    """ Req 5: Raciocínio Simbólico """
    def __init__(self):
        self.entities = {
            "copo": {"fragil": True, "massa": 0.2, "material": "vidro"},
            "martelo": {"fragil": False, "massa": 2.0, "material": "aço"},
            "prego": {"fragil": False, "massa": 0.01, "material": "ferro"}
        }

class ConsciousCore:
    """ O Maestro da AGI: Une Quintikus, TGNC e Metacognição """
    def __init__(self):
        self.tgnc = TGNC_NeuralCortex_v14()
        self.world = WorldModel()
        self.memory = SemanticMemory()
        self.objetivos = ["PRESERVAR_OBJETOS", "APRENDER_CONCEITOS"] # Req 9: Motor de Objetivos
        self.episodios = [] # Req 3: Memória Episódica Real

    def pensar(self, entrada):
        print(f"\n--- 🧠 CICLO COGNITIVO ATIVO ---")
        
        # 1. PERCEPÇÃO (Feeling-Cortex)
        # O TGNC agora faz o papel de Percepção + Entendimento Profundo
        nexo, acao, conf, mood = self.tgnc.analisar(entrada)
        
        # 2. IDENTIFICAÇÃO SEMÂNTICA (Req 5)
        objeto_alvo = self._extrair_entidade(entrada)
        props = self.memory.entities.get(objeto_alvo, {"fragil": False})

        # 3. METACOGNIÇÃO (Req 8: Autoconsciência operacional)
        print(f"  [METACONTROL]: Confiança Neural: {conf:.2f} | Humor: {mood}")
        
        # Req 6: Curiosidade Ativa
        if conf < 0.4 or nexo == "Indefinido":
            return self._curiosidade_ativa(entrada)

        # 4. SIMULAÇÃO CAUSAL (Req 1: Modelo de Mundo)
        previsao, risco = self.world.simular(acao, props)
        print(f"  [MUNDO]: Simulando '{acao}' em '{objeto_alvo}' -> {previsao}")

        # 5. TOMADA DE DECISÃO (Req 2: Planejamento / Req 9: Objetivos)
        if risco > 0.6 and "PRESERVAR_OBJETOS" in self.objetivos:
            decisao = f"INTERROMPER: Ação '{acao}' é muito arriscada para um objeto frágil."
        else:
            decisao = f"EXECUTAR: Realizando '{acao}' via nexo '{nexo}'."

        # 6. REGISTRO EPISÓDICO (Req 3 e 10: Aprendizado por Experiência)
        self._registrar_episodio(entrada, nexo, acao, decisao)
        
        return f"\n🤖 STATUS: {mood}\n🎯 DECISÃO: {decisao}\n💡 RAZÃO: {previsao}"

    def _extrair_entidade(self, texto):
        for k in self.memory.entities.keys():
            if k in texto.lower(): return k
        return "desconhecido"

    def _curiosidade_ativa(self, entrada):
        return "❓ [CURIOSIDADE]: Me falta nexo causal para entender isso. Pode me ensinar com 'ensinar'?"

    def _registrar_episodio(self, entrada, nexo, acao, decisao):
        evento = {"t": time.time(), "in": entrada, "nx": nexo, "ac": acao, "dec": decisao}
        self.episodios.append(evento)
        # Se houve sucesso, o TGNC salva no buffer de "sonho" (Req 10)
        self.tgnc.salvar_cortex()

    def ensinar_novo(self, texto, nexo, acao):
        print(f"🎓 [APRENDIZADO]: Mapeando '{texto}' -> '{nexo}'")
        self.tgnc.ensinar(texto, nexo, acao)
        self.tgnc.salvar_cortex()

# ==============================================================================
# EXECUÇÃO DA AGI UNIFICADA
# ==============================================================================

if __name__ == "__main__":
    agi = ConsciousCore()

    # Simulando um estado de aprendizado inicial
    agi.ensinar_novo("preciso pregar esse quadro", "CONSTRUCAO", "BATER")
    agi.ensinar_novo("o copo esta sujo", "LIMPEZA", "OLHAR")

    print("\n--- AGI CONSCIOUS CORE v100 ATIVA ---")
    
    while True:
        try:
            user_input = input("\nInteração > ").strip()
            if user_input.lower() == 'sair': break
            if user_input.lower() == 'consolidar':
                agi.tgnc.consolidar_mente()
                continue
                
            # Req 10: Feedback em tempo real
            if user_input.lower().startswith("ensinar"):
                _, t, n, a = user_input.split("|")
                agi.ensinar_novo(t, n, a)
                continue

            resposta = agi.pensar(user_input)
            print(resposta)

        except Exception as e:
            print(f"Erro no processamento: {e}")
