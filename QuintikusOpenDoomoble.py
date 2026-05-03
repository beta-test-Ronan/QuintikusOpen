#!/usr/bin/env python3
"""
Quintikus Doomoble v3.1 – ESPECIALISTA AUTÔNOMO
Motor de regras portátil (d=64, seq=32) com ativação configurável,
tradutor de intenções, memória viva, salvamento completo do especialista.
"""
import os, json, numpy as np, time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ════════════════════════════ NÚCLEO DO MODELO ════════════════════════════
class QuintikusDoomoble:
    def __init__(self, d_model=64, seq_len=32, ativacao='relu'):
        self.d, self.seq = d_model, seq_len
        self.ativacao = ativacao.lower()
        self.scale = np.float32(d_model ** -0.5)
        self.mask = (np.tril(np.ones((seq_len, seq_len), dtype=np.float32)) - 1) * 1e9
        self.weights, self.m, self.v = {}, {}, {}
        self.step = 0
        self.vocab = None

    def _ativar(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15))) if self.ativacao == 'sigmoid' else np.maximum(0, x)

    def save(self, nome="doomoble_brain.npz"):
        np.savez_compressed(nome, w=self.weights, v=self.vocab,
                            d=self.d, s=self.seq, step=self.step,
                            m=self.m, vv=self.v, ativacao=self.ativacao)
        print(f"💾 Modelo salvo em {nome}")

    def load(self, nome="doomoble_brain.npz"):
        if os.path.exists(nome):
            data = np.load(nome, allow_pickle=True)
            self.weights = data['w'].item()
            self.vocab = data['v'].item()
            self.ivocab = {i: c for c, i in self.vocab.items()}
            self.d = data['d'].item(); self.seq = data['s'].item()
            self.step = data['step'].item()
            if 'm' in data: self.m = data['m'].item()
            if 'vv' in data: self.v = data['vv'].item()
            if 'ativacao' in data: self.ativacao = data['ativacao'].item()
            self.scale = np.float32(self.d ** -0.5)
            self.mask = (np.tril(np.ones((self.seq, self.seq), dtype=np.float32)) - 1) * 1e9
            print(f"📥 Modelo carregado! Step {self.step} | Ativação: {self.ativacao}")
            return True
        return False

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    def _think(self, x):
        t = x.shape[1]
        h = self.weights['emb'][x] + self.weights['pos'][:t]
        qkv = h @ self.weights['qkv']
        q, k, v = np.split(qkv, 3, axis=-1)
        scores = np.einsum('btd,bsd->bts', q, k) * self.scale + self.mask[:t, :t]
        probs = self.softmax(scores)
        out = probs @ v
        rms = (np.sum(out**2, axis=-1, keepdims=True) / self.d + 1e-6) ** -0.5
        return out * rms, probs, q, k, v, h, rms

    def treinar(self, texto, épocas=500, lr=0.005, target_loss=None):
        chars = sorted(list(set(texto)))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.ivocab = {i: c for c, i in self.vocab.items()}
        vs = len(chars)

        f = lambda i, o: (np.random.randn(i, o) * np.sqrt(2 / i)).astype(np.float32)
        self.weights = {'emb': f(vs, self.d)*0.1, 'pos': f(self.seq, self.d)*0.1,
                        'qkv': f(self.d, self.d*3), 'head': f(self.d, vs)}
        for k, w in self.weights.items():
            if k not in self.m: self.m[k] = np.zeros_like(w)
            if k not in self.v: self.v[k] = np.zeros_like(w)

        data = np.array([self.vocab[c] for c in texto if c in self.vocab], dtype=np.int32)
        b1, b2, eps = 0.9, 0.999, 1e-8
        print(f"💾 Quintikus Doomoble | {self.d}D | {épocas} épocas | {vs} letras")
        if target_loss: print(f"🎯 Stoploss ativado: {target_loss:.4f}")

        for e in range(épocas + 1):
            t0 = time.perf_counter()
            ix = np.random.randint(0, len(data) - self.seq - 1, (8,))
            xb = np.stack([data[i:i + self.seq] for i in ix])
            yb = np.stack([data[i + 1:i + self.seq + 1] for i in ix])

            norm_out, probs, q, k, v, hin, rms = self._think(xb)
            logits = norm_out @ self.weights['head']
            p = self.softmax(logits)

            g_logits = p.copy()
            g_logits[np.arange(8)[:, None], np.arange(self.seq), yb] -= 1.0
            g_logits /= (8 * self.seq)

            g_head = norm_out.transpose(0, 2, 1).reshape(-1, self.d).T @ g_logits.reshape(-1, vs)
            dnorm = g_logits @ self.weights['head'].T
            mean_term = np.sum(dnorm * norm_out, axis=-1, keepdims=True) / self.d
            dattn = (dnorm - norm_out * mean_term) * rms

            dV = np.einsum('bts,btd->bsd', probs, dattn)
            dprobs = np.einsum('btd,bsd->bts', dattn, v)
            dscores = (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True)) * probs
            dQ = np.einsum('bts,bsd->btd', dscores, k) * self.scale
            dK = np.einsum('bst,btd->bsd', dscores, q) * self.scale
            dQKV = np.concatenate([dQ, dK, dV], axis=-1)
            g_qkv = np.einsum('btd,btf->df', hin, dQKV)
            dh = dQKV @ self.weights['qkv'].T
            g_emb = np.zeros_like(self.weights['emb'])
            np.add.at(g_emb, xb, dh)
            g_pos = np.sum(dh, axis=0)

            self.step += 1
            for key, grad in [('head', g_head), ('qkv', g_qkv), ('emb', g_emb), ('pos', g_pos)]:
                np.clip(grad, -1.0, 1.0, out=grad)
                self.m[key] = b1 * self.m[key] + (1 - b1) * grad
                self.v[key] = b2 * self.v[key] + (1 - b2) * (grad * grad)
                m_hat = self.m[key] / (1 - b1 ** self.step)
                v_hat = self.v[key] / (1 - b2 ** self.step)
                self.weights[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if e % 50 == 0:
                loss = -np.mean(np.log(p[np.arange(8)[:, None], np.arange(self.seq), yb] + 1e-9))
                print(f"E{e:04d} | loss {loss:.4f}")
                if target_loss and loss < target_loss:
                    print(f"🏁 Stoploss atingido ({loss:.4f} < {target_loss:.4f}). Treino encerrado.")
                    break
            time.sleep(max(0.1, (time.perf_counter() - t0) * 2.0))

    # ─── GERAÇÃO ───
    def gerar_com_alma_util(self, frase, tamanho=80, temp_base=0.7, entropia_alvo=1.0):
        idx = [self.vocab.get(c, 0) for c in frase]
        gerado = frase; current_temp = temp_base
        for _ in range(tamanho):
            out = self._think(np.array([idx[-self.seq:]]))[0]
            logits = (out @ self.weights['head'])[0, -1, :] / current_temp
            probs = self.softmax(logits)
            entropia = -np.sum(probs * np.log(probs + 1e-9))
            if entropia < entropia_alvo * 0.8: current_temp *= 1.05
            elif entropia > entropia_alvo * 1.2: current_temp *= 0.95
            current_temp = np.clip(current_temp, 0.1, 2.0)
            nxt = np.random.choice(len(self.vocab), p=probs)
            char = self.ivocab[nxt]; gerado += char; idx.append(nxt)
            if char == '\n' or (char in '.!?' and len(gerado) > 30): break
        return gerado.split('\n')[0].strip().rstrip('.')

    # Outros modos de geração (greedy, polido, regras) podem ser mantidos da versão anterior,
    # mas não são essenciais para o assistente. Vamos manter apenas o essencial para enxugar.
    def gerar_greedy(self, frase, tamanho=200):
        idx = [self.vocab.get(c, 0) for c in frase]; print(frase, end="")
        for _ in range(tamanho):
            out = self._think(np.array([idx[-self.seq:]]))[0]
            nxt = int(np.argmax((out @ self.weights['head'])[0, -1, :]))
            print(self.ivocab[nxt], end="", flush=True); idx.append(nxt)
        print()


# ════════════════ FUNÇÕES AUXILIARES ════════════════
def corrigir_ortografia(texto, vocabulario):
    palavras = texto.split()
    corrigidas = []
    for p in palavras:
        if p in vocabulario:
            corrigidas.append(p)
        else:
            for v in vocabulario:
                if len(p) == len(v) and sum(c1 != c2 for c1, c2 in zip(p, v)) <= 1:
                    corrigidas.append(v); break
            else:
                corrigidas.append(p)
    return ' '.join(corrigidas)

def traduzir_intencao(frase, mapa):
    frase = frase.lower().strip()
    for chave in sorted(mapa.keys(), key=len, reverse=True):
        if chave in frase:
            return mapa[chave]
    return None


# ════════════════ CÉREBRO CONSCIENTE (ESPECIALISTA) ════════════════
class CerebroConsciente:
    def __init__(self, modelo, vocabulario=None, tradutor=None):
        self.modelo = modelo
        self.memoria = []
        self.vocabulario = vocabulario or set()
        self.tradutor = tradutor or {}

    def adicionar_corpus(self, texto, comentario='#'):
        for linha in texto.strip().split('\n'):
            linha = linha.strip()
            if linha and not linha.startswith(comentario) and linha not in self.memoria:
                self.memoria.append(linha)
        print(f"📚 Corpus adicionado: {len(self.memoria)} frases na memória.")

    def aprender_frase(self, frase):
        frase = frase.strip()
        if frase and frase not in self.memoria:
            self.memoria.append(frase)
            print(f"🧠 Memorizado: '{frase}'")

    def responder(self, pergunta, tamanho=60):
        prompt_trad = traduzir_intencao(pergunta, self.tradutor)

        # Busca exata
        for frase in self.memoria:
            if pergunta.strip() in frase: return frase
        if prompt_trad:
            for frase in self.memoria:
                if prompt_trad in frase: return frase

        # Busca com correção ortográfica
        pergunta_corr = corrigir_ortografia(pergunta, self.vocabulario)
        if pergunta_corr != pergunta:
            for frase in self.memoria:
                if pergunta_corr.strip() in frase: return frase
        if prompt_trad:
            trad_corr = corrigir_ortografia(prompt_trad, self.vocabulario)
            for frase in self.memoria:
                if trad_corr in frase: return frase

        # Geração com alma controlada
        prompt_final = prompt_trad if prompt_trad else pergunta
        resposta = self.modelo.gerar_com_alma_util(prompt_final, tamanho=tamanho, entropia_alvo=0.4)
        resposta = corrigir_ortografia(resposta, self.vocabulario)
        return resposta if len(resposta) > 5 else "Ainda não aprendi sobre isso. Pode me ensinar?"

    # ─── Persistência do especialista completo ───
    def salvar_especialista(self, prefixo="especialista"):
        """Salva modelo, memória e tradutor em arquivos separados."""
        self.modelo.save(f"{prefixo}_modelo.npz")
        with open(f"{prefixo}_memoria.json", "w", encoding="utf-8") as f:
            json.dump({"memoria": self.memoria, "tradutor": self.tradutor,
                       "vocabulario": list(self.vocabulario)}, f, ensure_ascii=False, indent=2)
        print(f"📦 Especialista completo salvo como '{prefixo}_*'.")

    @classmethod
    def carregar_especialista(cls, prefixo="especialista", d_model=64, seq_len=32):
        """Carrega o especialista completo (modelo + memória + tradutor)."""
        modelo = QuintikusDoomoble(d_model, seq_len)
        if not modelo.load(f"{prefixo}_modelo.npz"):
            print("❌ Modelo não encontrado.")
            return None
        try:
            with open(f"{prefixo}_memoria.json", "r", encoding="utf-8") as f:
                dados = json.load(f)
            memoria = dados.get("memoria", [])
            tradutor = dados.get("tradutor", {})
            vocabulario = set(dados.get("vocabulario", []))
        except FileNotFoundError:
            print("⚠️ Arquivo de memória não encontrado. Iniciando vazio.")
            memoria, tradutor, vocabulario = [], {}, set()
        especialista = cls(modelo, vocabulario, tradutor)
        especialista.memoria = memoria
        print(f"📥 Especialista carregado: {len(memoria)} frases, {len(tradutor)} regras de tradução.")
        return especialista


# ════════════════════════ EXECUÇÃO ════════════════════════
if __name__ == "__main__":
    # ─── CONFIGURAÇÃO DO ESPECIALISTA ───
    CORPUS = """
    se > bateria = 10 >> comprar bateria nova
    se > luz = 0 >> acabou a luz
    se > caixa = 10x10 >> caixa pequena
    se > caixa = 100x100 >> caixa grande
    """ * 30

    TRADUTOR = {
        'bateria fraca': 'bateria = 10',
        'bateria baixa': 'bateria = 10',
        'sem luz': 'luz = 0',
        'luz apagou': 'luz = 0',
        'caixa enorme': 'caixa = 100x100',
        'caixa grande': 'caixa = 100x100',
        'caixa pequena': 'caixa = 10x10',
    }

    VOCABULARIO = {'bateria', 'comprar', 'nova', 'caixa', 'grande', 'pequena', 'luz', 'acabou'}

    # ─── INICIALIZAÇÃO ───
    nome_especialista = "meu_assistente"
    cerebro = CerebroConsciente.carregar_especialista(nome_especialista)

    if cerebro is None:
        print("🧠 Criando novo especialista...")
        maquina = QuintikusDoomoble(d_model=64, seq_len=32, ativacao='relu')
        maquina.treinar(CORPUS, épocas=2000, lr=0.003, target_loss=0.2)
        cerebro = CerebroConsciente(maquina, vocabulario=VOCABULARIO, tradutor=TRADUTOR)
        cerebro.adicionar_corpus(CORPUS)
        cerebro.salvar_especialista(nome_especialista)

    # ─── LOOP INTERATIVO ───
    print("\n🤖 Assistente pronto. Comandos:")
    print("   - Diga 'bateria fraca', 'sem luz' ou uma regra como 'se >...'")
    print("   - Ensine: 'ensinar: pergunta >> resposta'")
    print("   - Digite 'salvar' para gravar o especialista")
    print("   - Digite 'sair' para encerrar\n")

    while True:
        prompt = input("> ").strip()
        if not prompt: continue
        if prompt.lower() == 'sair': break
        if prompt.lower() == 'salvar':
            cerebro.salvar_especialista(nome_especialista)
            continue
        if prompt.startswith("ensinar:"):
            conteudo = prompt.split("ensinar:", 1)[1].strip()
            if ">>" in conteudo:
                partes = conteudo.split(">>", 1)
                nova_frase = f"{partes[0].strip()} >> {partes[1].strip()}"
                cerebro.aprender_frase(nova_frase)
                print(f"✅ Aprendido: {nova_frase}")
            else:
                print("ℹ️ Formato: ensinar: pergunta >> resposta")
            continue

        resposta = cerebro.responder(prompt)
        print("IA:", resposta)
