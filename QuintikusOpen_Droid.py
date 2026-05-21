import os, math, time, random, unicodedata, hashlib, re, pickle, platform
from collections import defaultdict, Counter
from statistics import mean as py_mean

# ========== ANDROID HELPER ==========
try:
    import androidhelper
    droid = androidhelper.Android()
    TEM_VOZ = True
except:
    droid = None
    TEM_VOZ = False

# ========== FUNÇÕES BÁSICAS ==========
def falar(texto, imprimir=True):
    if imprimir: print(f"🧠 GATI: {texto}")
    if TEM_VOZ:
        try: droid.ttsSpeak(texto)
        except: pass

def ouvir():
    if TEM_VOZ:
        try:
            print("\n🎤 Ouvindo...")
            res = droid.recognizeSpeech("Fale agora", None, None)
            if res and res.result:
                t = res.result.strip().lower()
                if t:
                    print(f"👤 Você disse: {t}")
                    return t
        except Exception as e:
            print(f"⚠️ Voz indisponível: {e}")
    try:
        return input("👤 Digite seu comando: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "sair"

# ========== ÁLGEBRA LINEAR PURA ==========
def py_random_uniform(low, high, n): return [random.uniform(low, high) for _ in range(n)]
def py_random_randn(*dims):
    if len(dims) == 1: return [random.gauss(0,1) for _ in range(dims[0])]
    elif len(dims) == 2: return [[random.gauss(0,1) for _ in range(dims[1])] for _ in range(dims[0])]
    raise ValueError
def py_dot(x,y): return sum(a*b for a,b in zip(x,y))
def py_vec_add(x,y): return [a+b for a,b in zip(x,y)]
def py_mat_vec_mul(mat,vec): return [py_dot(row,vec) for row in mat]
def py_weighted_average(vecs, ws):
    total = sum(ws)
    if total == 0: return [0.0]*len(vecs[0])
    scaled = [[a*w for a in v] for v,w in zip(vecs,ws)]
    return [sum(col)/total for col in zip(*scaled)]
def py_argmax(lst): return max(range(len(lst)), key=lambda i: lst[i])

# ========== CRIPTOGRAFIA ==========
class SovereignCrypt:
    @staticmethod
    def get_key(name):
        salt = platform.node() or "ROOT_NEXUS"
        return hashlib.sha256((name+salt).encode()).digest()
    @staticmethod
    def xor_cipher(text, key):
        return "".join(chr(ord(c)^key[i%len(key)]) for i,c in enumerate(text))

# ========== USUÁRIO ==========
class UserSovereignChain:
    def __init__(self, fn="user.bin"):
        self.fn = fn; self.history = []; self.pil = 0.0; self.name = None
    def salvar(self):
        with open(self.fn,'wb') as f: pickle.dump({'h':self.history,'pil':self.pil,'name':self.name},f)
    def carregar(self):
        if os.path.exists(self.fn):
            try:
                with open(self.fn,'rb') as f:
                    d = pickle.load(f)
                    self.history = d.get('h',[]); self.pil = d.get('pil',0.0); self.name = d.get('name')
                    return self.pil, self.name
            except: pass
        return 0.0, None

# ========== TOKENIZER ==========
class SovereignTokenizer:
    def __init__(self):
        self.pat = re.compile(r'\?\?+|\!\!+|\.\.\.+|[:;]-?[)DPpoO]|s2|<3|\^\^|[\w]+|[\?\!\.]')
        self.spec = ["<BOS>","<EOS>","<PAD>","<?>","<!>","<.>"]
    def tokenize(self, text):
        text = text.replace('\x00','').replace('\ufeff','')
        norm = "".join(c for c in unicodedata.normalize("NFKD", text.lower().strip()) if not unicodedata.combining(c))
        raw = self.pat.findall(norm)
        res = []
        for t in raw:
            if t=='?': res.append("<?>")
            elif t=='!': res.append("<!>")
            elif t=='.': res.append("<.>")
            else: res.append(t)
        return res

# ========== CORE QUÂNTICO (CORRIGIDO) ==========
class QuantumLPSCore:
    def __init__(self, vocab_size, d_model=32):
        self.d_model = d_model
        lim = math.sqrt(6 / (vocab_size + d_model))
        self.embeddings = [py_random_uniform(-lim, lim, d_model) for _ in range(vocab_size)]
        self.Wq = py_random_randn(d_model, d_model)
        self.Wk = py_random_randn(d_model, d_model)
        self.W_future = py_random_randn(d_model, d_model)

    def colapsar_nexo(self, q_idx, lps_idx, cand_idx_list, rarity, w2i):
        valid_q = [i for i in q_idx if i < len(self.embeddings)]
        if not valid_q or not cand_idx_list:
            return None, 0

        q_ws = [rarity.get(list(w2i.keys())[i], 0.1) for i in valid_q]
        tw = sum(q_ws)
        q_ws = [w / tw if tw > 0 else 1.0 / len(q_ws) for w in q_ws]

        sub_vec = py_weighted_average([self.embeddings[i] for i in valid_q], q_ws)
        lps_vec = self.embeddings[lps_idx]
        foco = py_mat_vec_mul(self.W_future, py_vec_add(py_mat_vec_mul(self.Wq, sub_vec), lps_vec))

        scores = []
        for c in cand_idx_list:
            vc = [i for i in c if i < len(self.embeddings)]
            if not vc:
                scores.append(-1e9)
                continue
            # Correção: usar self.d_model
            mean_c = [py_mean([self.embeddings[i][k] for i in vc]) for k in range(self.d_model)]
            scores.append(py_dot(foco, py_mat_vec_mul(self.Wk, mean_c)))

        best = py_argmax(scores)
        return cand_idx_list[best], scores[best]

# ========== CALCULADORA ==========
class Calculer:
    cache = {}
    @staticmethod
    def eh_mat(texto): return bool(re.match(r'^[0-9\+\-\*\/\(\)\.\s\^]+$', texto)) and any(c in texto for c in "+-*/^")
    @staticmethod
    def resolver(exp, auria):
        try:
            limpa = exp.replace(" ","")
            for k in sorted(Calculer.cache, key=len, reverse=True):
                if k in limpa and len(k)<len(limpa):
                    val = Calculer.cache[k]
                    res = eval(limpa.replace(k,str(val)).replace('^','**'), {"__builtins__": None}, {})
                    msg = f"LEMBRO que {k}={val}, então {exp}={res} <!>"
                    break
            else:
                res = eval(limpa.replace('^','**'), {"__builtins__": None}, {})
                msg = f"O resultado de {exp} é {res}."
            Calculer.cache[limpa] = res
            auria.amadurecer_solo(f"{exp} = {res}.", auth=2, silenciar=True)
            return f"\n[CALCULER]\n> {msg}"
        except Exception as e: return f"\n[MATH-ERROR]: {e}"

# ========== SENTIMENTO ==========
class SentimentAnalyzer:
    POS = {"amo","amor","lindo","maravilhoso","obrigado","obrigada","querido","querida","gato","gata","fofo","fofa","bom","boa","ótimo","excelente","feliz","alegre","carinho","carinhoso","carinhosa","beijo","abraço","saudade","parabéns","incrível","espetacular","divertido","legal"}
    NEG = {"triste","chato","chata","ruim","horrível","ódio","raiva","nojento","nojenta","feio","feia","burro","burra","idiota","imbecil","droga","merda","desculpa","desculpe","cansado","cansada","estressado","estressada"}
    @staticmethod
    def analisar(texto):
        tokens = set(texto.lower().split())
        p = len(tokens.intersection(SentimentAnalyzer.POS))
        n = len(tokens.intersection(SentimentAnalyzer.NEG))
        if p>n: return "positivo"
        elif n>p: return "negativo"
        return "neutro"

# ========== EMOÇÃO ==========
class EmotionState:
    def __init__(self):
        self.val = 0.0; self.exc = 0.5; self.decay = 0.9
    def atualizar(self, sent):
        rv, re = random.gauss(0,0.1), random.gauss(0,0.05)
        if sent=="positivo": self.val += 0.2+rv; self.exc += 0.1+re
        elif sent=="negativo": self.val -= 0.2+rv; self.exc += 0.1+re
        else: self.val += rv; self.exc += re
        self.val = max(-1.0, min(1.0, self.val))
        self.exc = max(0.0, min(1.0, self.exc))
        self.val *= self.decay
        self.exc = 0.5 + (self.exc-0.5)*self.decay
    def tom(self):
        if self.val>0.3: return "caloroso"
        elif self.val<-0.3: return "frio"
        return "neutro"

# ========== OTIMIZADOR ADAM (PURO) ==========
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [self._zeros(p) for p in params]
        self.v = [self._zeros(p) for p in params]
        self.t = 0
    def _zeros(self, arr):
        if isinstance(arr,list):
            if arr and isinstance(arr[0],list): return [[0.0]*len(arr[0]) for _ in range(len(arr))]
            else: return [0.0]*len(arr)
        return 0.0
    def _mul(self, a, s):
        if isinstance(a,list):
            if a and isinstance(a[0],list): return [[x*s for x in row] for row in a]
            else: return [x*s for x in a]
        return a*s
    def _add(self, a, b):
        if isinstance(a,list):
            if a and isinstance(a[0],list): return [[a[i][j]+b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
            else: return [a[i]+b[i] for i in range(len(a))]
        return a+b
    def _sqr(self, a):
        if isinstance(a,list):
            if a and isinstance(a[0],list): return [[x*x for x in row] for row in a]
            else: return [x*x for x in a]
        return a*a
    def step(self, grads):
        self.t += 1
        for i in range(len(self.params)):
            self.m[i] = self._add(self._mul(self.m[i], self.b1), self._mul(grads[i], 1.0-self.b1))
            self.v[i] = self._add(self._mul(self.v[i], self.b2), self._mul(self._sqr(grads[i]), 1.0-self.b2))
            m_hat = self._mul(self.m[i], 1.0/(1.0-self.b1**self.t))
            v_hat = self._mul(self.v[i], 1.0/(1.0-self.b2**self.t))
            for row_idx in range(len(self.params[i])):
                if isinstance(self.params[i][row_idx], list):
                    for col_idx in range(len(self.params[i][row_idx])):
                        self.params[i][row_idx][col_idx] -= self.lr * m_hat[row_idx][col_idx] / (math.sqrt(v_hat[row_idx][col_idx])+self.eps)
                else:
                    self.params[i][row_idx] -= self.lr * m_hat[row_idx] / (math.sqrt(v_hat[row_idx])+self.eps)

# ========== REDE NEURAL PREDITIVA ==========
class PredicativeNet:
    def __init__(self, input_dim, hidden_dim, vocab_sizes):
        self.W1 = py_random_randn(input_dim, hidden_dim)
        self.b1 = [0.0]*hidden_dim
        self.Ws = py_random_randn(hidden_dim, vocab_sizes[0])
        self.bs = [0.0]*vocab_sizes[0]
        self.Wv = py_random_randn(hidden_dim, vocab_sizes[1])
        self.bv = [0.0]*vocab_sizes[1]
        self.Wc = py_random_randn(hidden_dim, vocab_sizes[2])
        self.bc = [0.0]*vocab_sizes[2]
        self.params = [self.W1, self.b1, self.Ws, self.bs, self.Wv, self.bv, self.Wc, self.bc]
        self.opt = Adam(self.params, lr=0.001)
    def forward(self, ctx):
        h = [max(0, py_dot(self.W1[i],ctx)+self.b1[i]) for i in range(len(self.b1))]
        s = [py_dot(self.Ws[i],h)+self.bs[i] for i in range(len(self.bs))]
        v = [py_dot(self.Wv[i],h)+self.bv[i] for i in range(len(self.bv))]
        c = [py_dot(self.Wc[i],h)+self.bc[i] for i in range(len(self.bc))]
        return s, v, c
    def gerar(self, ctx, temp=0.8):
        ss, vs, cs = self.forward(ctx)
        suj_idx = self._sample(ss, temp)
        ver_idx = self._sample(vs, temp)
        comp_idx = self._sample(cs, temp)
        return suj_idx, ver_idx, comp_idx
    def _sample(self, scores, temp):
        if temp==0: return py_argmax(scores)
        mx = max(scores)
        exp = [math.exp((s-mx)/temp) for s in scores]
        total = sum(exp)
        probs = [e/total for e in exp]
        r = random.random()
        cum = 0.0
        for i,p in enumerate(probs):
            cum += p
            if r <= cum: return i
        return len(probs)-1
    def treinar(self, ctx, alvo_idx, reward):
        ss, vs, cs = self.forward(ctx)
        for slot, scores, idx in [('suj',ss,alvo_idx[0]), ('ver',vs,alvo_idx[1]), ('comp',cs,alvo_idx[2])]:
            grad = [0.0]*len(scores)
            grad[idx] = reward
            if slot=='suj': b = self.bs
            elif slot=='ver': b = self.bv
            else: b = self.bc
            for i in range(len(grad)):
                if grad[i]!=0:
                    b[i] += 0.01*grad[i]

# ========== CÉREBRO CONVERSACIONAL ==========
class ConversationBrain:
    def __init__(self):
        self.ctx = []
        self.rede = None
    def init_rede(self, suj_vocab, ver_vocab, comp_vocab):
        self.rede = PredicativeNet(64, 32, (len(suj_vocab), len(ver_vocab), len(comp_vocab)))
        self.suj_map = suj_vocab
        self.ver_map = ver_vocab
        self.comp_map = comp_vocab
        self.idx2suj = {v:k for k,v in suj_vocab.items()}
        self.idx2ver = {v:k for k,v in ver_vocab.items()}
        self.idx2comp = {v:k for k,v in comp_vocab.items()}
    def atualizar_ctx(self, tokens):
        self.ctx.extend(tokens)
        if len(self.ctx) > 50: self.ctx = self.ctx[-50:]
    def gerar_frase(self, temp=0.8):
        if not self.rede: return "eu ainda estou aprendendo a falar."
        if len(self.ctx) < 3: return "vamos conversar mais?"
        ctx_vec = py_weighted_average([self._word2vec(t) for t in self.ctx], [0.9**i for i in range(len(self.ctx))])
        suj_idx, ver_idx, comp_idx = self.rede.gerar(ctx_vec, temp)
        suj = self.idx2suj.get(suj_idx, "eu")
        ver = self.idx2ver.get(ver_idx, "sou")
        comp = self.idx2comp.get(comp_idx, "aqui")
        return f"{suj} {ver} {comp}."
    def _word2vec(self, w):
        if not hasattr(self, '_w2v_cache'): self._w2v_cache = {}
        if w not in self._w2v_cache:
            self._w2v_cache[w] = py_random_uniform(-0.1,0.1,64)
        return self._w2v_cache[w]

# ========== AURIA PRINCIPAL (CORRIGIDA) ==========
class QuintikusOpenAuria:
    def __init__(self):
        self.path_brain = "brain_v18_sovereign.qoa"
        self.tokenizer = SovereignTokenizer()
        self.user_chain = UserSovereignChain("user.bin")
        self.l2_mass, self.l2_auth, self.l2_pil_min, self.l2_tokens_idx = [], [], [], []
        self.neuronios = defaultdict(list)
        self.rarity, self.word2idx, self.ledger = {}, {}, set()
        self.core = None
        self.stop_words = {"o","a","de","que","do","da","é","em","um","para","com","na","no"}
        self.user_name = None
        self.pil_user = 0.0
        self.crypto_key = None
        self.sentiment = SentimentAnalyzer()
        self.emotion = EmotionState()
        self.carinhometro = 123/(123+456)
        self.brain = ConversationBrain()
        self.modo_neural = False
        self._init_brain_vocab()  # inicia com vocabulário básico
    def _init_brain_vocab(self):
        suj = {"eu":0,"você":1,"ele":2,"ela":3,"nós":4,"alguém":5}
        ver = {"sou":0,"estou":1,"amo":2,"gosto":3,"vejo":4,"penso":5,"falo":6}
        comp = {"aqui":0,"bem":1,"triste":2,"feliz":3,"isso":4,"nada":5}
        self.brain.init_rede(suj, ver, comp)
    def amadurecer_solo(self, raw, auth=1, pil_min=0.0, silenciar=False):
        h = hashlib.sha256((raw+str(pil_min)).encode('utf-8','ignore')).hexdigest()
        if h in self.ledger: return False
        if not silenciar: falar("🧠 Amadurecendo Solo...")
        sents = []
        if any(c in raw for c in '.!?'):
            chunks = re.split(r'([\?\!\.])', raw)
            for i in range(0,len(chunks)-1,2):
                s = (chunks[i]+chunks[i+1]).strip()
                if len(s)>1: sents.append(s)
        else: sents = [s.strip() for s in raw.split('\n') if len(s.strip())>1]
        if not sents: return False
        new_words = (" ".join(sents)).lower().split()
        total_vocab = sorted(list(set(new_words+self.tokenizer.spec+list(self.word2idx.keys()))))
        self.word2idx = {w:i for i,w in enumerate(total_vocab)}
        contagem = Counter(new_words)
        off = len(self.l2_mass)
        for i,s in enumerate(sents):
            stored = SovereignCrypt.xor_cipher(s, self.crypto_key) if (pil_min>=9.0 and self.crypto_key) else s
            self.l2_mass.append(stored)
            self.l2_auth.append(auth)
            self.l2_pil_min.append(pil_min)
            toks = self.tokenizer.tokenize(s)
            idxs = [self.word2idx[t] for t in toks if t in self.word2idx]
            self.l2_tokens_idx.append(idxs)
            for t in toks:
                if t not in self.stop_words and t in self.word2idx:
                    if len(self.neuronios[t])<10000: self.neuronios[t].append(off+i)
                    self.rarity[t] = 2.0/(math.log(contagem.get(t,1)+1.2)+1e-5)
        self.core = QuantumLPSCore(len(total_vocab))
        self.ledger.add(h)
        self._update_brain_vocab()
        self.selar(silenciar)
        return True
    def _update_brain_vocab(self):
        if not self.rarity: return
        palavras = sorted(self.rarity.keys(), key=lambda w: self.rarity[w], reverse=True)[:20]
        suj = {w:i for i,w in enumerate(palavras[:6])}
        ver = {w:i for i,w in enumerate(palavras[6:12])}
        comp = {w:i for i,w in enumerate(palavras[12:18])}
        self.brain.init_rede(suj, ver, comp)
    def selar(self, silenciar=False):
        bundle = {'mass':self.l2_mass, 'auth':self.l2_auth, 'pil_min':self.l2_pil_min,
                  'rar':self.rarity, 'w2i':self.word2idx, 'neu':dict(self.neuronios),
                  'core':self.core, 't_idx':self.l2_tokens_idx, 'ledger':self.ledger,
                  'm_cache':Calculer.cache}
        with open(self.path_brain,'wb') as f: pickle.dump(bundle,f,protocol=pickle.HIGHEST_PROTOCOL)
        if not silenciar: falar(f"💾 Solo Selado ({len(self.l2_mass)} nexos).")
    def boot(self):
        self.pil_user, self.user_name = self.user_chain.carregar()
        if not self.user_name:
            falar("Primeira ativação detectada.")
            self.user_name = input("  👤 Qual seu nome? > ").strip()
            self.user_chain.name = self.user_name
            self.user_chain.salvar()
        else:
            falar(f"✅ Quintikus Online. Olá, {self.user_name}. PIL: {self.pil_user:.2f}")
        self.crypto_key = SovereignCrypt.get_key(self.user_name)
        if os.path.exists(self.path_brain):
            with open(self.path_brain,'rb') as f:
                b = pickle.load(f)
                self.l2_mass, self.l2_auth, self.l2_pil_min = b['mass'], b['auth'], b['pil_min']
                self.rarity, self.word2idx, self.neuronios = b['rar'], b['w2i'], b['neu']
                self.core, self.l2_tokens_idx, self.ledger = b['core'], b['t_idx'], b['ledger']
                Calculer.cache = b.get('m_cache',{})
            self._update_brain_vocab()
        return True
    def pensar_e_falar(self, entrada):
        sentimento = self.sentiment.analisar(entrada)
        self.emotion.atualizar(sentimento)
        if sentimento == "positivo" and random.random() < self.carinhometro:
            return f"{self.user_name}, você é especial para mim também! 💛" if self.emotion.tom()=="caloroso" else "Obrigado pelo carinho. Saiba que também gosto de você."
        if Calculer.eh_mat(entrada): return Calculer.resolver(entrada, self)
        t0 = time.perf_counter()
        u_toks = self.tokenizer.tokenize(entrada)
        self.brain.atualizar_ctx(u_toks)
        if self.modo_neural and self.brain.rede:
            frase = self.brain.gerar_frase(0.7)
            if sentimento == "positivo": self.brain.rede.treinar(self._ctx_vec(), None, 1.0)
            return f"\n {frase}"
        q_idx = [self.word2idx[t] for t in u_toks if t in self.word2idx]
        if self.core: q_idx = [i for i in q_idx if i < len(self.core.embeddings)]
        if not self.l2_mass: return f"oi, {self.user_name}. Use 'train:arquivo.txt' para me ensinar algo."
        pivos = sorted([t for t in u_toks if t in self.rarity], key=lambda x: self.rarity[x], reverse=True)
        if not pivos: pivos = [t for t in u_toks if t in self.word2idx and t not in self.stop_words]
        if not pivos:
            return self.brain.gerar_frase(0.9) if self.brain.rede else f"oi, {self.user_name}... ainda não conheço essas palavras <!>"
        candidatos = []
        for p in pivos:
            temp = [idx for idx in self.neuronios.get(p, [])]
            if temp: candidatos = temp; break
        if not candidatos:
            return self.brain.gerar_frase(0.9) if self.brain.rede else f"não encontrei nada sobre {pivos[0]}."
        amostra = random.sample(candidatos, min(len(candidatos), 600))
        amostra_idx_list = [self.l2_tokens_idx[i] for i in amostra]
        lps_symbol = u_toks[-1] if u_toks[-1] in ["<?>","<!>","<.>"] else "<.>"
        lps_idx = self.word2idx.get(lps_symbol, self.word2idx["<.>"])
        best_future, gravidade = self.core.colapsar_nexo(q_idx, lps_idx, amostra_idx_list, self.rarity, self.word2idx)
        if best_future is None: return "\n[COLAPSO] > Não consegui conectar os nexos."
        final_idx = amostra[amostra_idx_list.index(best_future)]
        frase = self.l2_mass[final_idx]
        if self.l2_pil_min[final_idx] >= 9.0: frase = SovereignCrypt.xor_cipher(frase, self.crypto_key)
        if gravidade > 0.8:
            self.pil_user = min(35.0, self.pil_user + gravidade*0.05)
            self.user_chain.pil = self.pil_user
            self.user_chain.salvar()
        palavras = frase.split()
        res_toks_idx = self.l2_tokens_idx[final_idx]
        resultado = []
        if q_idx:
            q_vecs = [self.core.embeddings[i] for i in q_idx]
            q_vec = [py_mean([v[k] for v in q_vecs]) for k in range(self.core.d_model)]  # <--- CORRIGIDO AQUI
            reacoes = [py_dot(self.core.embeddings[t], q_vec) for t in res_toks_idx if t<len(self.core.embeddings)]
            limiar = py_mean(reacoes) if reacoes else 0
            for i,w in enumerate(palavras):
                t_id = res_toks_idx[i] if i<len(res_toks_idx) else None
                if t_id is not None and t_id<len(self.core.embeddings) and py_dot(self.core.embeddings[t_id], q_vec)>limiar:
                    resultado.append(w.upper())
                else: resultado.append(w.lower())
            frase = ' '.join(resultado)
        tom = self.emotion.tom()
        frase += " ✨" if tom=="caloroso" else " ❄️" if tom=="frio" else ""
        return f"\n {frase}"
    def _ctx_vec(self):
        if not self.brain.ctx: return [0.0]*64
        vecs = [self.brain._word2vec(t) for t in self.brain.ctx]
        return py_weighted_average(vecs, [0.9**i for i in range(len(vecs))])

# ========== COMANDOS EXTRAS ==========
def processar_comandos(cmd, auria):
    if any(p in cmd for p in ["horas","hora","que horas"]): return f"São {time.strftime('%H:%M')}"
    if "bateria" in cmd and TEM_VOZ:
        droid.batteryStartMonitoring(); time.sleep(0.5)
        niv = droid.batteryGetLevel().result
        droid.batteryStopMonitoring()
        return f"A bateria está em {niv}%"
    if any(p in cmd for p in ["piada","conte uma piada"]):
        piadas = ["Por que o Python foi ao psicólogo? Porque tinha muitos loops internos!","O que o Java disse pro Python? Você não tem classe!","Quantos programadores para trocar uma lâmpada? Nenhum, é problema de hardware."]
        return random.choice(piadas)
    if cmd == "modo neural":
        auria.modo_neural = not auria.modo_neural
        return "Modo neural ativado." if auria.modo_neural else "Modo neural desativado."
    return auria.pensar_e_falar(cmd)

# ========== LOOP PRINCIPAL ==========
if __name__ == "__main__":
    auria = QuintikusOpenAuria()
    auria.boot()
    falar("Assistente Quintikus pronto. Pode falar ou digitar.")
    print("Comandos: 'horas', 'bateria', 'piada', 'modo neural', 'train:arquivo.txt', 'sair'")
    while True:
        comando = ouvir()
        if not comando: continue
        if any(comando.startswith(p) for p in ["train:","trein:","treino:"]):
            path_part = comando.split(":",1)[1]
            pil_lock = 0.0
            if "pil[" in path_part:
                try:
                    pil_lock = float(path_part.split("pil[")[1].split("]")[0])
                    path = path_part.split("pil[")[0].strip()
                except: path = path_part.strip()
            else: path = path_part.strip()
            if os.path.exists(path):
                falar(f"📂 Lendo {path}...")
                for enc in ['utf-8','latin-1','cp1252']:
                    try:
                        with open(path,'r',encoding=enc) as f:
                            conteudo = f.read()
                            if auria.amadurecer_solo(conteudo, auth=1, pil_min=pil_lock):
                                falar(f"✨ Conhecimento integrado!")
                            else: falar("⚠️ Conteúdo já existia.")
                        break
                    except: continue
            else: falar(f"❌ Arquivo '{path}' não encontrado.")
            continue
        if any(p in comando for p in ["sair","exit","desligar","tchau"]):
            falar("Até logo!")
            break
        resposta = processar_comandos(comando, auria)
        falar(resposta)
        if droid: droid.eventWait(3000)
