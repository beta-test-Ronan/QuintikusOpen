import numpy as np, re, pickle, os, hashlib, time, sys, unicodedata
from collections import defaultdict, Counter

# ========== NORMALIZAÇÃO ROBUSTA ==========
def normalizar_texto(texto):
    if isinstance(texto, bytes):
        for enc in ['utf-8', 'latin-1', 'windows-1252', 'utf-16']:
            try: texto = texto.decode(enc); break
            except UnicodeError: continue
        else: texto = texto.decode('utf-8', errors='ignore')
    texto = unicodedata.normalize('NFC', texto)
    texto = re.sub(r'[^\x20-\x7EáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇàèìòùäëïöüÿÀÈÌÒÙÄËÏÖÜŸ\n\t.,!?;:()\-]', ' ', texto)
    return texto

# ========== GEOMETRIA ==========
class DiscoPoincare:
    def __init__(self, d=64): self.d = d
    def dist(self, x, y):
        nx = np.clip(np.linalg.norm(x), 0, 0.95)
        ny = np.clip(np.linalg.norm(y), 0, 0.95)
        xp = x/(np.linalg.norm(x)+1e-8)*nx
        yp = y/(np.linalg.norm(y)+1e-8)*ny
        d2 = np.sum((xp-yp)**2)
        val = 1 + 2*d2/((1-nx**2)*(1-ny**2)+1e-12)
        return np.arccosh(np.clip(val, 1.0, 1e6))

class TokenQuantico:
    def __init__(self, d=64, n=4): self.d, self.n, self.e = d, n, {}
    def ini(self, t, v):
        if t not in self.e: self.e[t]=[v.copy()]
    def add(self, t, ctx, w=0.3):
        if t not in self.e: return
        nv = self.e[t][0]*(1-w)+ctx*w; nv /= max(np.linalg.norm(nv),1e-8)
        self.e[t].append(nv)
        if len(self.e[t])>self.n: self.e[t].pop(1)
    def col(self, t, qv, disc):
        if t not in self.e: return None
        melhores = [(1.0/(1.0+float(disc.dist(qv,s))), s) for s in self.e[t]]
        if not melhores: return None
        melhores.sort(reverse=True)
        probs = np.array([m[0] for m in melhores])**2
        return melhores[np.random.choice(len(melhores), p=probs/probs.sum())][1] if probs.sum()>0 else melhores[0][1]

class MPS:
    def __init__(self, d=64, b=8):
        self.A = np.random.randn(d, b) * np.sqrt(2./d)
        self.B = np.random.randn(b, b) * np.sqrt(2./b)
        self.C = np.random.randn(b, d) * np.sqrt(2./b)
    def trans(self, v):
        o = v @ self.A @ self.B @ self.C
        n = np.linalg.norm(o)
        if n == 0: return np.random.randn(len(v))*0.1
        return o/n*0.90 if n > 0.90 else o

class Planner:
    def __init__(self, d=64, a=0.03, lim=0.01):
        self.d, self.a, self.lim, self.r, self.on = d, a, lim, np.zeros(d), False
    def load(self, v):
        n = np.linalg.norm(v)
        self.r = v/n*0.5 if n > 0 else np.random.randn(self.d)*0.1
        self.on = True
    def fuse(self, m): return (0.7*m + 0.3*self.r) if self.on else m
    def eat(self, e):
        if self.on: self.r -= self.a * e
    def exausto(self): return self.on and np.linalg.norm(self.r) < self.lim

# ========== BLOCKCHAIN ==========
class Blockchain:
    def __init__(self, f="bc.pkl"):
        self.f = f; self.chain = []
        if os.path.exists(f):
            with open(f,'rb') as h: self.chain = pickle.load(h)
    def add(self, frases):
        prev = self.chain[-1]['h'] if self.chain else '0'*8
        h = hashlib.sha256((prev+'|'.join(frases)).encode()).hexdigest()[:8]
        self.chain.append({'h':h, 'd':frases})
        with open(self.f,'wb') as f: pickle.dump(self.chain, f)
        return h
    def all(self): return [f for b in self.chain for f in b['d']]

# ========== TGP-2.5 QUÂNTICO ULTRA ==========
class TGP25_QuantumUltra:
    def __init__(self, dim=64, arq="tgp25q_ultra.pkl"):
        self.dim, self.arq = dim, arq
        self.disco = DiscoPoincare(dim)
        self.quantico = TokenQuantico(dim)
        self.mps = MPS(dim)
        self.tv = {}
        self.bi = defaultdict(lambda: defaultdict(int))
        self.tri = defaultdict(lambda: defaultdict(int))
        self.pref = defaultdict(list)
        self.tf = '<END>'
        self.clusters, self.vocab_busca = [], {}
        self.emaranhados, self.mi = {}, {}
        self.planner = Planner(dim)
        self._load()

    def _reg(self, t):
        if t not in self.tv:
            v = np.random.randn(self.dim)*0.2
            self.tv[t] = v
            self.quantico.ini(t, v)

    def tok(self, txt):
        if not txt: return []
        txt = unicodedata.normalize('NFD', txt.lower())
        txt = ''.join(c for c in txt if unicodedata.category(c)!='Mn')
        return [t for t in re.findall(r'[a-z0-9]+|[.,!?;:]+|\s+', txt) if t.strip()]

    def _vet(self, txt):
        toks = self.tok(txt)
        if not toks: return np.random.randn(self.dim)*0.1
        vs = []
        for w in toks:
            if w not in self.vocab_busca:
                self.vocab_busca[w] = np.random.randn(self.dim)*0.2
            vs.append(self.vocab_busca[w])
        m = np.mean(vs, axis=0)
        return m/(np.linalg.norm(m)+1e-8) if np.linalg.norm(m)>0 else m

    def indexar(self, docs):
        if not docs: return
        vs = [self._vet(d) for d in docs]
        n_cl = min(10, max(1, len(docs)//20))
        cs = [np.random.randn(self.dim)*0.2 for _ in range(n_cl)]
        for _ in range(5):
            buckets = [[] for _ in range(n_cl)]
            for i,v in enumerate(vs):
                buckets[np.argmin([self.disco.dist(v,c) for c in cs])].append(i)
            for j in range(n_cl):
                if buckets[j]: cs[j] = np.mean([vs[i] for i in buckets[j]], axis=0)
        self.clusters = [{'c':cs[j], 'docs':[(docs[i],vs[i]) for i in buckets[j]]} for j in range(n_cl) if buckets[j]]

    def buscar(self, q, k=5):
        if not self.clusters: return []
        qv = self._vet(q)
        for tok in self.tok(q):
            for (a,b),_ in self.emaranhados.items():
                if a==tok:
                    for cl in self.clusters:
                        if any(b in d[0].lower() for d in cl['docs']):
                            return [d[0] for d in sorted(cl['docs'], key=lambda x:self.disco.dist(qv,x[1]))[:k]]
        dists = [self.disco.dist(qv,cl['c']) for cl in self.clusters]
        alvo = self.clusters[np.argmin(dists)]
        return [d for d,_ in sorted(alvo['docs'], key=lambda x:self.disco.dist(qv,x[1]))[:k]]

    def devorar(self, texto, verbose=True):
        tokens = self.tok(texto)
        if len(tokens) < 3: return
        for t in tokens: self._reg(t)
        for i in range(len(tokens)-1): self.bi[tokens[i]][tokens[i+1]] += 1
        for i in range(len(tokens)-2):
            self.tri[(tokens[i],tokens[i+1])][tokens[i+2]] += 1
            self.pref[(tokens[i],tokens[i+1])].append(tokens[i+2])
        frases = [f+'.' for f in re.split(r'[\.\?\!]', texto) if len(f.split())>3]
        self.indexar(frases)
        co = Counter()
        for i in range(len(tokens)):
            for j in range(i+1, min(i+20, len(tokens))):
                if tokens[i]!=tokens[j]: co[(tokens[i],tokens[j])] += 1
        total = sum(co.values()); unigramas = Counter(tokens)
        for (a,b),c in co.most_common(1000):
            p_ab=c/total; p_a=unigramas[a]/len(tokens); p_b=unigramas[b]/len(tokens)
            if p_a>0 and p_b>0:
                mi=p_ab*np.log2(p_ab/(p_a*p_b)+1e-12)
                if mi>0.01 and a in self.tv and b in self.tv:
                    self.emaranhados[(a,b)]=np.outer(self.tv[b],self.tv[a]); self.mi[(a,b)]=mi
        self._save()
        if verbose: print(f"✅ {len(tokens)} tokens | {len(self.emaranhados)} pares emaranhados.")

    def _save(self):
        with open(self.arq, 'wb') as f:
            pickle.dump({'tv':self.tv,'qe':self.quantico.e,'bi':dict(self.bi),'tri':dict(self.tri),
                        'pref':dict(self.pref),'clusters':self.clusters,'vb':self.vocab_busca,
                        'em':self.emaranhados,'mi':self.mi}, f)

    def _load(self):
        if os.path.exists(self.arq):
            try:
                with open(self.arq, 'rb') as f:
                    d = pickle.load(f)
                self.tv = d.get('tv',{}); self.quantico.e = d.get('qe',{})
                for k,v in d.get('bi',{}).items(): self.bi[k].update(v)
                for k,v in d.get('tri',{}).items(): self.tri[k].update(v)
                for k,v in d.get('pref',{}).items(): self.pref[k]=v
                self.clusters = d.get('clusters',[]); self.vocab_busca = d.get('vb',{})
                self.emaranhados = d.get('em',{}); self.mi = d.get('mi',{})
            except: print("⚠️ Erro ao carregar modelo. Iniciando novo.")
        if not self.tv: self._reg(self.tf)

    def gerar(self, prompt, contexto="", max_t=80):
        if contexto: self.devorar(contexto, verbose=False)
        toks_in = self.tok(prompt)
        if not toks_in:
            if not self.tv: return
            toks_in = [np.random.choice(list(self.tv.keys()))]
        conhecidos = [self.tv[t] for t in toks_in if t in self.tv]
        att = np.mean(conhecidos, axis=0) if conhecidos else np.random.randn(self.dim)*0.1
        self.planner.load(att)
        ctx, freq, inercia = toks_in.copy(), Counter(), 1.0
        for _ in range(max_t):
            if self.planner.exausto():
                if ctx[-1] not in '.!?': yield '.'
                break
            tok = None
            if len(ctx) >= 2:
                chave = tuple(ctx[-2:])
                if chave in self.pref and np.random.random() > 0.4:
                    sugestoes = self.pref[chave]
                    if sugestoes:
                        tok = Counter(sugestoes).most_common(1)[0][0]
            if not tok:
                v_mps = self.mps.trans(self.tv.get(ctx[-1], att))
                alvo = self.planner.fuse(v_mps)
                ult = ctx[-1] if ctx else None
                if ult and ult in [p[0] for p in self.emaranhados]:
                    for (a,b),M in self.emaranhados.items():
                        if a==ult and b in ctx:
                            novo = np.dot(M, alvo)
                            n2 = np.linalg.norm(novo)
                            if n2 > 0.95: novo = novo/n2*0.95
                            alvo = 0.7*alvo + 0.3*novo
                            break
                tok = self._decodificar(ctx, alvo, freq, inercia)
            if not tok or self.planner.exausto():
                if ctx[-1] not in '.!?': yield '.'
                break
            if tok in self.tv: self.planner.eat(self.tv[tok])
            freq[tok] += 1; ctx.append(tok); yield tok
        self._save()

    def _decodificar(self, ctx, alvo, freq, inercia):
        cands = []
        evitar = ctx[-2:]
        for t in self.tv:
            if t in evitar or t == self.tf: continue
            dist = self.disco.dist(alvo, self.tv.get(t, alvo))
            score_geo = 1.0/(1.0+dist)
            score_mem = 0.0
            if len(ctx) >= 2:
                chave = tuple(ctx[-2:])
                if chave in self.tri and t in self.tri[chave]:
                    score_mem = self.tri[chave][t] * 2.0
            punicao = (freq.get(t,0)**1.5) * (3.0 / max(0.1, inercia))
            cands.append((t, (score_geo + score_mem) - punicao))
        if not cands: return None
        cands.sort(key=lambda x: x[1], reverse=True)
        top = cands[:5]
        scores = np.array([s[1] for s in top])
        probs = np.exp(scores - np.max(scores))
        probs /= probs.sum()
        idx = np.random.choice(len(top), p=probs)
        return top[idx][0]


# ========== INTERFACE ==========
if __name__ == "__main__":
    modelo = TGP25_QuantumUltra(dim=256)
    bc = Blockchain()
    frases = bc.all()
    if frases:
        print(f"📦 Blockchain: {len(bc.chain)} blocos, {len(frases)} frases.")
        modelo.devorar(" ".join(frases))
    else:
        print("📦 Blockchain vazia. Use train:arquivo.txt para iniciar.")

    print("\n🧠 TGP‑2.5 Quântico ULTRA pronto. Comandos: train:arquivo.txt | sair")
    while True:
        try:
            e = input("\n🙋 ").strip()
            if not e: continue
            if e.lower() == 'sair': break
            if e.startswith("train:"):
                arq = e.split(":",1)[1].strip()
                if os.path.exists(arq):
                    with open(arq, 'rb') as f: raw = f.read()
                    txt = normalizar_texto(raw)
                    fr = [f+'.' for f in re.split(r'[\.\?\!]', txt) if len(f.split())>2]
                    if fr:
                        bc.add(fr)
                        modelo.devorar(" ".join(bc.all()))
                        print(f"✅ {len(fr)} frases adicionadas à blockchain.")
                    else: print("❌ Nenhuma frase válida.")
                else: print("❌ Arquivo não encontrado.")
                continue
            
            t0 = time.time()
            docs = modelo.buscar(e, k=5)
            t1 = time.time()
            ctx = " ".join(docs)
            if docs: print(f"📚 {len(docs)} docs em {(t1-t0)*1000:.0f}ms")
            
            print("🧠 ", end="")
            for tok in modelo.gerar(e, contexto=ctx, max_t=60):
                print(tok, end=' ', flush=True)
            print()
        except KeyboardInterrupt: break
        except Exception as ex: print(f"\n⚠️ Erro: {ex}")
