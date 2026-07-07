import numpy as np
import json
import os

# ======================== MOTOR v12.7 (ESTÁVEL + APRENDIZADO ATIVO) ========================
def l2_normalize(x):
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8) if norm > 0 else x

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

class TGNC_NeuralCortex_v12:
    def __init__(self, dim=128, rank=64):
        self.dim = dim
        self.rank = rank 
        self.lr = 0.5 
        self.solo_path = "tgnc_cortex_v12.json"
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

    def processar_sequencia(self, texto):
        palavras = [w for w in texto.lower().replace(".", " . ").replace(",", "").split() if w not in self.ruido]
        if not palavras: return None, 0.0
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
        x, _ = self.processar_sequencia(texto_longo)
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
        if x is None: return "Vazio", "Nenhuma", 0.0
        v_nexo = l2_normalize(x @ self.W_context)
        v_acao = l2_normalize(v_nexo @ self.W_action)
        nexo_f = self._buscar_proximo(v_nexo, "concepts")
        acao_f = self._buscar_proximo(v_acao, "actions")
        sim_geo = float(np.dot(v_nexo, self.concepts[nexo_f]))
        conf_final = (sim_geo * 0.7) + (conf_neural * 0.3)
        return nexo_f, acao_f, float(conf_final)

    def _buscar_proximo(self, vetor, space):
        target = {"concepts": self.concepts, "actions": self.actions}[space]
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
                print("🧠 Córtex carregado.")
            except: pass

# ======================== INTERFACE DE APRENDIZADO ATIVO ========================
if __name__ == "__main__":
    tgnc = TGNC_NeuralCortex_v12()

    # Treino Inicial de Base (Só se o solo estiver vazio)
    if not tgnc.concepts:
        print("--- 📚 CONSOLIDANDO BASE DE CONHECIMENTO ---")
        tgnc.ensinar("vazando duto principal", "ruptura_infraestrutura", "fechar_valvulas")
        tgnc.ensinar("acesso negado firewall", "ataque_brute_force", "bloquear_ip")
        tgnc.ensinar("sistema lento travando", "sobrecarga_processamento", "limpar_cache")
        tgnc.salvar_cortex()

    print("\n--- 🧠 ANALISADOR INTERATIVO ATIVO ---")
    print("Digite 'sair' para encerrar ou 'ensinar' para treinar um novo nexo.")

    while True:
        entrada = input("\nLog > ")
        if entrada.lower() == 'sair': break
        
        if entrada.lower() == 'ensinar':
            txt = input("Frase do Fato: ")
            nx = input("Nexo (Conclusão): ")
            ac = input("Ação Sugerida: ")
            tgnc.ensinar(txt, nx, ac)
            tgnc.salvar_cortex()
            print("✨ Conhecimento integrado ao Solo.")
            continue

        nexo, acao, conf = tgnc.analisar(entrada)
        print(f"🤖 Entendimento: {nexo.upper()} | Confiança: {conf:.2f}")
        print(f"🚀 Ação: {acao.upper()}")

        # MODO CORREÇÃO: Se o sistema estiver em dúvida ou errar
        if conf < 0.45:
            feedback = input("🤖 Acertei o diagnóstico? (s/n): ")
            if feedback.lower() == 'n':
                correto = input("🤖 Qual seria o Nexo correto? ")
                acao_correta = input("🤖 Qual a Ação para esse nexo? ")
                tgnc.ensinar(entrada, correto, acao_correta)
                tgnc.salvar_cortex()
                print("🧠 Obrigado. Minhas sinapses foram ajustadas.")
