import numpy as np
import json
import os

# ======================== MOTOR v13.0 (TEMPORAL + PROTOCOLO PIVOT) ========================

def l2_normalize(x):
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8) if norm > 0 else x

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

class TGNC_NeuralCortex_v13:
    def __init__(self, dim=128, rank=64):
        self.dim = dim
        self.rank = rank 
        self.lr = 0.5 
        self.decay = 0.998  # Taxa de esquecimento (Decaimento Temporal)
        self.solo_path = "tgnc_cortex_v13.json"
        
        self.vocab, self.concepts, self.actions = {}, {}, {}
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

    def _protocolo_pivot(self, texto_original):
        """
        Trata entradas que contém apenas ruído. 
        Em vez de retornar vazio, o sistema foca na estrutura dos conectivos.
        """
        palavras_pivot = texto_original.lower().split()
        if not palavras_pivot: return None, 0.0
        
        # Cria um vetor sintático baseado apenas nos conectivos
        h_pivot = [self._get_vector(p, "vocab") for p in palavras_pivot]
        v_pivot = l2_normalize(np.mean(h_pivot, axis=0))
        return v_pivot, 0.15 # Confiança baixa por ser apenas ruído

    def processar_sequencia(self, texto):
        palavras = [w for w in texto.lower().replace(".", " . ").replace(",", "").split() if w not in self.ruido]
        
        # Ativação do PIVOT se não houver palavras de conteúdo
        if not palavras:
            return self._protocolo_pivot(texto)

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

    def ensinar(self, texto_longo, nexo, acao):
        # 1. Aplicar Decaimento Temporal antes de aprender o novo
        # Isso faz com que o córtex "esqueça" ou abra espaço para novas conexões
        self.W_context *= self.decay
        self.W_action *= self.decay
        self.W1 *= self.decay
        self.W2 *= self.decay

        x, _ = self.processar_sequencia(texto_longo)
        if x is None: return
        
        y_target = self._get_vector(nexo, "concepts")
        z_target = self._get_vector(acao, "actions")
        
        for _ in range(80): 
            self.W_context += self.lr * np.outer(x, (y_target - x @ self.W_context))
            self.W_action += self.lr * np.outer(y_target, (z_target - y_target @ self.W_action))
            self.W1 += self.lr * 0.02 * np.outer(x, (y_target @ self.W2.T)) 
            
        self.W_context = np.clip(self.W_context, -1, 1)
        self.W_action = np.clip(self.W_action, -1, 1)

    def analisar(self, texto):
        x, conf_neural = self.processar_sequencia(texto)
        if x is None: return "Silêncio", "Nenhuma", 0.0
        
        v_nexo = l2_normalize(x @ self.W_context)
        v_acao = l2_normalize(v_nexo @ self.W_action)
        
        nexo_f = self._buscar_proximo(v_nexo, "concepts")
        acao_f = self._buscar_proximo(v_acao, "actions")
        
        sim_geo = float(np.dot(v_nexo, self.concepts.get(nexo_f, np.zeros(self.dim))))
        conf_final = (sim_geo * 0.7) + (conf_neural * 0.3)
        
        # Se for um caso de Pivot (só ruído), forçamos o nexo para 'vazio_sintatico'
        if conf_neural == 0.15:
            return "RUÍDO_PURO", "SOLICITAR_CONTEÚDO", 0.15

        return nexo_f, acao_f, float(conf_final)

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
                        elif isinstance(v, list) and k != "ruido": setattr(self, k, np.array(v))
                        else: setattr(self, k, v)
                print(f"🧠 Córtex v13 carregado. (Taxa de Decaimento: {self.decay})")
            except: pass

# ======================== EXECUÇÃO ========================
if __name__ == "__main__":
    tgnc = TGNC_NeuralCortex_v13()

    if not tgnc.concepts:
        print("--- 📚 CONSOLIDANDO BASE INICIAL ---")
        tgnc.ensinar("vazando duto principal", "ruptura_infraestrutura", "fechar_valvulas")
        tgnc.ensinar("acesso negado firewall", "ataque_brute_force", "bloquear_ip")
        tgnc.salvar_cortex()

    print("\n--- 🧠 ANALISADOR v13 ATIVO (Temporal + Pivot) ---")
    
    while True:
        entrada = input("\nLog > ")
        if entrada.lower() == 'sair': break
        
        if entrada.lower() == 'ensinar':
            txt = input("Frase: ")
            nx = input("Nexo: ")
            ac = input("Ação: ")
            tgnc.ensinar(txt, nx, ac)
            tgnc.salvar_cortex()
            continue

        nexo, acao, conf = tgnc.analisar(entrada)
        
        if nexo == "RUÍDO_PURO":
            print("⚠️ PIVOT: A entrada contém apenas conectivos/ruído.")
            print(f"🤖 Sugestão: {acao}")
        else:
            print(f"🤖 Entendimento: {nexo.upper()} | Confiança: {conf:.2f}")
            print(f"🚀 Ação: {acao.upper()}")

        if 0 < conf < 0.45:
            feedback = input("🤖 Baixa confiança. Corrigir? (s/n): ")
            if feedback.lower() == 's':
                correto = input("Nexo correto: ")
                acao_correta = input("Ação correta: ")
                tgnc.ensinar(entrada, correto, acao_correta)
                tgnc.salvar_cortex()
