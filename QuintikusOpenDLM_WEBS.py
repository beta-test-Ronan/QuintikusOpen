#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLM_webs v8.5 — Regex catalog validated + evidence map consensus.
Patches: NOME_PROPRIO case-sensitive, min_fontes=1 (determinístico),
         lookup prefere wikipedia, role busca ampliada.
"""

import sys, re, json, math, time, hashlib, os, sqlite3, heapq, warnings, argparse, random, pickle, unicodedata, difflib, threading
from collections import defaultdict
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, wait

try:
    import requests; warnings.filterwarnings("ignore")
    try: requests.packages.urllib3.disable_warnings()
    except Exception: pass
except ImportError: requests = None
try:
    from bs4 import BeautifulSoup; HAS_BS4 = True
except ImportError: HAS_BS4 = False
try:
    import trafilatura; HAS_TRAFILATURA = True
except ImportError: HAS_TRAFILATURA = False
try:
    from ddgs import DDGS; HAS_DDGS = True
except ImportError: HAS_DDGS = False
try:
    from googlesearch import search as google_search; HAS_GOOGLESEARCH = True
except ImportError: HAS_GOOGLESEARCH = False

STATE_DIR = os.path.join(os.path.expanduser("~"), ".dlm_webs")
DB_FILE = "dlm.db"
MODEL_FILE = "modelo_gns_adaptive.pkl"
MIN_SOURCES = 2
SEAL_THRESHOLD = 0.5
FETCH_TIMEOUT = 8
WEB_CACHE_TTL = 300
GLOBAL_RATE = 0.5  # v8.5: aumentado para evitar 429 do Wiki

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]
def get_ua(): return random.choice(USER_AGENTS)

def safe_print(*a, **k):
    try: print(*a, **k)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(" ".join(str(x) for x in a).encode(enc, "replace").decode(enc, "replace"), **k)

def _ensure_dir(d):
    if not os.path.isdir(d): os.makedirs(d, exist_ok=True)


# ======================================================================
# REGEX CATALOG (v8.5 — validado do catálogo)
# ======================================================================
# Nome próprio: força case-sensitive no grupo. Limita a 4 palavras compostas.
# Uso: re.search(r"..." + NOME_PROPRIO_PATTERN, para) SEM re.IGNORECASE global.
NOME_PROPRIO_PATTERN = (
    r"((?-i:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zá-ú]+"
    r"(?:\s+(?:de|da|do|dos|das|e|[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zá-ú]+)){0,3}))"
)

# Datas: grupos nomeados para parse limpo
REGEX_DATAS = {
    "extenso_pt": re.compile(
        r"\b(?P<dia>\d{1,2})\s+de\s+"
        r"(?P<mes>janeiro|fevereiro|março|abril|maio|junho|julho|agosto|"
        r"setembro|outubro|novembro|dezembro)\s+de\s+(?P<ano>\d{4})\b",
        re.IGNORECASE
    ),
    "br_numerica": re.compile(
        r"\b(?P<dia>[0-3]?\d)[\/\-\.](?P<mes>[0-1]?\d)[\/\-\.](?P<ano>\d{4})\b"
    ),
    "intervalo": re.compile(
        r"\([^)]*?(?P<ini>\d{4})\s*(?:—|–|-)\s*[^)]*?(?P<fim>\d{4})[^)]*\)"
    ),
    "ano_isolado": re.compile(r"(?<!\d)(?P<ano>1[5-9]\d{2}|20[0-2]\d)(?!\d)"),
}

# Quantidades: mesmo padrão do código atual mas compilado
REGEX_QUANTIDADE = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(milh[õo]es|mil|bilh[õo]es)\s+(?:de\s+)?(?:habitantes|pessoas|brasileiros)',
    re.IGNORECASE
)
REGEX_QUANTIDADE_POP = re.compile(
    r'popula[çc][ãa]o[^.]{0,80}?(\d+(?:[.,]\d+)?)\s*(milh[õo]es|mil|bilh[õo]es)',
    re.IGNORECASE
)
REGEX_QUANTIDADE_RAW = re.compile(
    r'(\d{1,3}(?:[.,]\d{3})+|\d{5,})\s+(?:habitantes|pessoas)',
    re.IGNORECASE
)

# Filtro anti-snippet-de-notícia ("19 de jul. de 2026 · Tenha acesso...")
REGEX_SNIPPET_NEWS = re.compile(r"\d{1,2}\s+de\s+\w+\s+de\s+20\d{2}\s*·")

MESES_PT = {"janeiro","fevereiro","março","abril","maio","junho","julho",
            "agosto","setembro","outubro","novembro","dezembro"}


# ======================================================================
# CACHE + RATE + TIMEOUT
# ======================================================================
class TTLCache:
    def __init__(self, ttl=WEB_CACHE_TTL, max_size=500):
        self.ttl, self.max_size, self.data = ttl, max_size, {}
        self.lock = threading.Lock()
    def get(self, key):
        with self.lock:
            e = self.data.get(key)
            if e and time.time() - e[1] < self.ttl: return e[0]
            if e: del self.data[key]
            return None
    def set(self, key, val):
        with self.lock:
            if len(self.data) >= self.max_size:
                oldest = min(self.data.items(), key=lambda kv: kv[1][1])
                del self.data[oldest[0]]
            self.data[key] = (val, time.time())

class GlobalRateLimiter:
    def __init__(self, min_delay=GLOBAL_RATE):
        self.min_delay, self.last = min_delay, 0.0
        self.lock = threading.Lock()
    def wait(self):
        with self.lock:
            d = self.min_delay - (time.time() - self.last)
            if d > 0: time.sleep(d)
            self.last = time.time()

_RATE = GlobalRateLimiter()

def _run_with_timeout(fn, *args, timeout=FETCH_TIMEOUT, **kwargs):
    result = [None, None]; done = [False]
    def worker():
        try:
            result[0] = fn(*args, **kwargs); done[0] = True
        except Exception as e:
            result[1] = e; done[0] = True
    t = threading.Thread(target=worker, daemon=True)
    t.start(); t.join(timeout=timeout)
    if not done[0]: return None
    if result[1] is not None: return None
    return result[0]


# ======================================================================
# NORMALIZAÇÃO
# ======================================================================
STOPWORDS = {"a","o","as","os","um","uma","de","do","da","dos","das","em","no","na","nos","nas",
             "para","com","por","que","e","ou","mas","se","já","não"}

def normalizar_entidade(text):
    if not text: return ""
    t = unicodedata.normalize('NFD', text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != 'Mn')
    t = re.sub(r"^(?:o|a|os|as|um|uma)\s+", "", t)
    t = re.sub(r"^(?:ordene|liste|enumere|cite|relacione|quem|quando|onde|aonde|quantos?|qual|quais)\s+", "", t)
    t = re.sub(r"\b(?:foi|foram|é|era|são|aconteceu|ocorreu|sucedeu|fundad[oa]|criad[oa]|inaugurad[oa]|proclamad[oa]|abolida|abolido)\b", " ", t)
    t = re.sub(r"\s+(?:em|no|na|nos|nas)\s+[a-z][\w\s]*$", "", t)
    for w in STOPWORDS: t = re.sub(rf"\b{w}\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ======================================================================
# EVIDENCE MAP
# ======================================================================
def split_paragrafos(text):
    if not text: return []
    paragrafos = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 60]
    if paragrafos: return paragrafos
    sents = re.split(r'(?<=[.!?])\s+', text)
    buf, out = "", []
    for s in sents:
        buf += s + " "
        if len(buf) > 200:
            out.append(buf.strip()); buf = ""
    if buf.strip(): out.append(buf.strip())
    return out or [text]

class EvidenceMap:
    def __init__(self):
        self.evidence = defaultdict(list)
    def add(self, candidato, fonte, paragrafo):
        if not candidato: return
        norm = normalizar_entidade(candidato)
        if not norm or len(norm.split()) < 1: return
        self.evidence[norm].append({
            "raw": candidato, "fonte": fonte, "paragrafo": paragrafo[:200],
        })
    def ranking(self):
        out = []
        for cand_norm, evs in self.evidence.items():
            fontes = set(e["fonte"] for e in evs)
            out.append((len(fontes), len(evs), cand_norm, evs))
        out.sort(reverse=True)
        return out
    def consensus(self, min_fontes=2, allow_wiki_solo=True):
        ranking = self.ranking()
        if not ranking: return None, None, []
        n_fontes, n_ocorr, cand_norm, evs = ranking[0]
        has_wiki = any(e["fonte"] == "wikipedia" for e in evs)
        if n_fontes >= min_fontes or (allow_wiki_solo and has_wiki):
            return evs[0]["raw"], evs[0]["paragrafo"], ranking
        return None, None, ranking
    def is_empty(self):
        return not self.evidence


# ======================================================================
# TOPOLOGIA GEOMÉTRICA (intacta)
# ======================================================================
class SymbolicNetGeometric:
    def __init__(self, db):
        self.db = db
        self.nodes = set()
        self.edges = defaultdict(float)
        self._load()
    def _load(self):
        for r in self.db.execute("SELECT node FROM net_nodes"): self.nodes.add(r["node"])
        for r in self.db.execute("SELECT a, b, weight FROM net_edges"): self.edges[(r["a"], r["b"])] = r["weight"]
    def reinforce(self, a, b, amount=0.1):
        if not a or not b: return
        a, b = str(a), str(b)
        if a == b: return
        key = tuple(sorted([a, b]))
        self.edges[key] = self.edges.get(key, 0.0) + amount
        self.nodes.update(key)
        self.db.execute("INSERT OR REPLACE INTO net_edges (a, b, weight) VALUES (?, ?, ?)", (key[0], key[1], self.edges[key]))
        for n in key: self.db.execute("INSERT OR IGNORE INTO net_nodes (node) VALUES (?)", (n,))
    def decay(self, f=0.995):
        for k in list(self.edges):
            self.edges[k] *= f
            if self.edges[k] < 1e-3:
                del self.edges[k]
                self.db.execute("DELETE FROM net_edges WHERE a=? AND b=?", (k[0], k[1]))
            else:
                self.db.execute("UPDATE net_edges SET weight=? WHERE a=? AND b=?", (self.edges[k], k[0], k[1]))
    def topo_distance(self, a, b):
        if a == b: return 0.0
        if a not in self.nodes or b not in self.nodes: return float("inf")
        adj = defaultdict(list)
        for (x, y), w in self.edges.items():
            c = -math.log(max(w, 1e-9))
            adj[x].append((y, c)); adj[y].append((x, c))
        dist, pq = {a: 0.0}, [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b: return d
            if d > dist.get(u, float("inf")): continue
            for v, c in adj[u]:
                nd = d + c
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd; heapq.heappush(pq, (nd, v))
        return float("inf")
    def nearest(self, entity, k=3, threshold=5.0):
        if entity not in self.nodes: return []
        cands = []
        for n in self.nodes:
            if n == entity: continue
            d = self.topo_distance(entity, n)
            if d < threshold: cands.append((d, n))
        cands.sort()
        return cands[:k]
    def fuzzy_match(self, entity, threshold_topo=3.0, threshold_str=0.85):
        if entity in self.nodes:
            for d, n in self.nearest(entity, k=3, threshold=threshold_topo):
                return n, f"topo({d:.2f})"
        best = None
        for n in self.nodes:
            if n == entity: continue
            sim = difflib.SequenceMatcher(None, entity, n).ratio()
            if sim >= threshold_str and (best is None or sim > best[0]):
                best = (sim, n)
        return (best[1], f"str({best[0]:.2f})") if best else (None, None)
    def _best_year_for(self, text, entity_keywords):
        candidates = []
        kws = [w.lower() for w in entity_keywords.split() if len(w) > 3]
        for m in REGEX_DATAS["ano_isolado"].finditer(text):
            y = int(m.group("ano"))
            before = text[max(0, m.start()-120):m.start()].lower()
            after = text[m.end():m.end()+30].lower()
            score = 0
            if re.search(r'(?:foi|foram|é|era|ocorreu|aconteceu|sucedeu|culminou|resultou)\s+(?:\w+\s+){0,3}(?:em|no|na|com)\s*$', before):
                score += 30
            if re.search(r'(?:fundad|criad|inaugurad|proclamad|abolida|abolido|assinad|decretad|concluíd)\w*\s+(?:em|no|na|com)\s*$', before):
                score += 25
            if re.search(r'(?:de|em)\s*$', before): score += 10
            if re.search(r'(?:desde|a partir de|iniciado em)\s*$', before): score += 5
            if re.match(r'\s*(?:-|–|a|até)\s*\d{4}', after): score -= 10
            for kw in kws:
                if kw in before: score += 3
            candidates.append((score, -m.start(), y))
        if not candidates: return None
        candidates.sort(reverse=True)
        return candidates[0][2]
    def spectral_embedding(self, dim=2, iters=40):
        nodes = sorted(self.nodes); N = len(nodes)
        if N < 3: return {}
        idx = {n: i for i, n in enumerate(nodes)}
        A = [[0.0]*N for _ in range(N)]
        for (x, y), w in self.edges.items():
            i, j = idx[x], idx[y]; A[i][j] = A[j][i] = w
        def mv(v): return [sum(A[i][j]*v[j] for j in range(N)) for i in range(N)]
        def norm(v):
            n = math.sqrt(sum(x*x for x in v)) or 1.0
            return [x/n for x in v]
        def orth(v, basis):
            for b in basis:
                dot = sum(x*y for x, y in zip(v, b))
                v = [x - dot*y for x, y in zip(v, b)]
            return norm(v)
        basis, v = [], norm([1.0]*N)
        for _ in range(iters): v = norm(mv(v))
        basis.append(v)
        for _ in range(1, dim):
            v = norm([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
            for _ in range(iters): v = orth(mv(v), basis)
            basis.append(v)
        return {n: tuple(basis[d][idx[n]] for d in range(dim)) for n in nodes}
    def stats(self):
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "density": len(self.edges) / max(1, len(self.nodes) * (len(self.nodes) - 1) / 2)}


# ======================================================================
# GRAMÁTICA + GNS (intactas)
# ======================================================================
class GramaticaGenerativa:
    PADROES_ABSTRATOS = [
        ("AGENTE","CRIACAO","CRIADOR_DE"),("AGENTE","FUNDACAO","CRIADOR_DE"),
        ("AGENTE","INVENCAO","CRIADOR_DE"),("AGENTE","DESENVOLVIMENTO","CRIADOR_DE"),
        ("TEMPO","MORTE","DATA_MORTE"),("TEMPO","NASCIMENTO","DATA_NASCIMENTO"),
        ("TEMPO","EVENTO","DATA_EVENTO"),("TEMPO","PASSADO","DATA_EVENTO"),
        ("LUGAR","LOCALIZACAO","LOCALIZACAO"),("QUANTIDADE",None,"QUANTIDADE"),
        ("LISTAR",None,"LISTA"),("ORDENAR",None,"ORDENAR"),("ESPECIFICAR",None,"ESPECIFICAR"),
        ("AGENTE","PASSADO","LOOKUP_PESSOA"),("AGENTE","PRESENTE","LOOKUP_PESSOA"),
        ("PROCESSO",None,"PROCESSO"),("CAUSA",None,"CAUSA"),("SAUDACAO",None,"SAUDACAO"),
    ]
    @classmethod
    def gerar_regex(cls, wh, verbo, alvo):
        wh_map = {"AGENTE":r"quem","TEMPO":r"(?:quando|em\s+que\s+ano|em\s+que\s+data)",
                  "LUGAR":r"(?:aonde|onde|em\s+que\s+lugar)","QUANTIDADE":r"(?:quantos?|quantas?)",
                  "LISTAR":r"(?:quais|liste|enumere)","ORDENAR":r"(?:ordene|ordene\s+cronologicamente)",
                  "ESPECIFICAR":r"(?:qual|quais)","PROCESSO":r"como","CAUSA":r"por\s+que",
                  "SAUDACAO":r"(?:oi|olá|bom\s+dia|boa\s+tarde|boa\s+noite)"}
        wh_pat = wh_map.get(wh)
        if not wh_pat: return None
        verbo_map = {"CRIACAO":r"(?:criou|fez|desenvolveu|inventou)","FUNDACAO":r"(?:fundou|é\s+o\s+fundador)",
                     "INVENCAO":r"(?:inventou|criou)","DESENVOLVIMENTO":r"(?:desenvolveu|criou)",
                     "MORTE":r"(?:morreu|faleceu)","NASCIMENTO":r"nasceu",
                     "EVENTO":r"(?:aconteceu|ocorreu|sucedeu|foi)","PASSADO":r"foi",
                     "PRESENTE":r"é","LOCALIZACAO":r"(?:fica|está|localizado)"}
        if verbo:
            vpat = verbo_map.get(verbo, r"\w+")
            return rf"^{wh_pat}\s+{vpat}\s+(?:o\s+|a\s+|os\s+|as\s+)?(?P<entity>.+?)(?:\?|$)"
        return rf"^{wh_pat}\s+(?:o\s+|a\s+|os\s+|as\s+)?(?P<entity>.+?)(?:\?|$)"
    @classmethod
    def gerar_todos(cls):
        r = {}
        for wh, v, a in cls.PADROES_ABSTRATOS:
            p = cls.gerar_regex(wh, v, a)
            if p: r.setdefault(a, []).append(p)
        return r

class GNSAdaptive:
    def __init__(self, vocab_size=500, hidden1=32, hidden2=16, lr=0.05):
        self.vocab_size, self.hidden1, self.hidden2, self.lr = vocab_size, hidden1, hidden2, lr
        self.vocab, self.idf, self.labels = {}, {}, []
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [], [], [], [], [], []
        self.is_trained = False
        self.exemplos_usuario = []
    def _tokenize(self, text): return re.findall(r'\w+', text.lower())
    def _build_tfidf(self, texts):
        counts = defaultdict(int)
        for t in texts:
            for w in self._tokenize(t): counts[w] += 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:self.vocab_size]
        self.vocab = {w: i for i, (w, _) in enumerate(top)}
        N = len(texts)
        for w in self.vocab:
            df = sum(1 for t in texts if w in self._tokenize(t))
            self.idf[w] = math.log((1 + N) / (1 + df)) + 1
    def _vectorize(self, text):
        vec = [0.0] * len(self.vocab)
        tokens = self._tokenize(text)
        if not tokens: return vec
        counts = defaultdict(int)
        for w in tokens:
            if w in self.vocab: counts[w] += 1
        for w, c in counts.items(): vec[self.vocab[w]] = (c / len(tokens)) * self.idf[w]
        n = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x/n for x in vec]
    def _init_weights(self):
        V, C, H1, H2 = len(self.vocab), len(self.labels), self.hidden1, self.hidden2
        self.W1 = [[random.uniform(-0.1,0.1)/(1.0+abs(i-j)/10.0) for j in range(V)] for i in range(H1)]
        self.b1 = [0.0]*H1
        self.W2 = [[random.uniform(-0.1,0.1) for j in range(H1)] for i in range(H2)]
        self.b2 = [0.0]*H2
        self.W3 = [[random.uniform(-0.1,0.1) for j in range(H2)] for i in range(C)]
        self.b3 = [0.0]*C
    def train(self, texts, labels, epochs=50):
        if not texts: return
        self.labels = sorted(set(labels))
        self._build_tfidf(texts)
        self._init_weights()
        X = [self._vectorize(t) for t in texts]
        Y = []
        for l in labels:
            y = [0.0]*len(self.labels); y[self.labels.index(l)] = 1.0; Y.append(y)
        V, C, H1, H2 = len(self.vocab), len(self.labels), self.hidden1, self.hidden2
        for _ in range(epochs):
            for x, y in zip(X, Y):
                h1 = [math.tanh(sum(self.W1[i][j]*x[j] for j in range(V))+self.b1[i]) for i in range(H1)]
                h2 = [math.tanh(sum(self.W2[i][j]*h1[j] for j in range(H1))+self.b2[i]) for i in range(H2)]
                out = [sum(self.W3[i][j]*h2[j] for j in range(H2))+self.b3[i] for i in range(C)]
                m = max(out); exp = [math.exp(o-m) for o in out]; s = sum(exp)
                probs = [e/s for e in exp]
                d_out = [probs[i]-y[i] for i in range(C)]
                d_h2 = [sum(d_out[j]*self.W3[j][i] for j in range(C))*(1-h2[i]**2) for i in range(H2)]
                d_h1 = [sum(d_h2[j]*self.W2[j][i] for j in range(H2))*(1-h1[i]**2) for i in range(H1)]
                for i in range(C):
                    for j in range(H2): self.W3[i][j] -= self.lr*d_out[i]*h2[j]
                    self.b3[i] -= self.lr*d_out[i]
                for i in range(H2):
                    for j in range(H1): self.W2[i][j] -= self.lr*d_h2[i]*h1[j]
                    self.b2[i] -= self.lr*d_h2[i]
                for i in range(H1):
                    for j in range(V): self.W1[i][j] -= self.lr*d_h1[i]*x[j]
                    self.b1[i] -= self.lr*d_h1[i]
        self.is_trained = True
    def predict(self, text):
        if not self.is_trained: return {}
        x = self._vectorize(text)
        V, C, H1, H2 = len(self.vocab), len(self.labels), self.hidden1, self.hidden2
        if len(x) != len(self.W1[0]):
            x = x[:len(self.W1[0])] + [0.0]*max(0, len(self.W1[0])-len(x))
        h1 = [math.tanh(sum(self.W1[i][j]*x[j] for j in range(len(self.W1[0])))+self.b1[i]) for i in range(H1)]
        h2 = [math.tanh(sum(self.W2[i][j]*h1[j] for j in range(H1))+self.b2[i]) for i in range(H2)]
        out = [sum(self.W3[i][j]*h2[j] for j in range(H2))+self.b3[i] for i in range(C)]
        m = max(out); exp = [math.exp(o-m) for o in out]; s = sum(exp)
        return {self.labels[i]: exp[i]/s for i in range(C)}
    def learn_online(self, question, correct_intent):
        self.exemplos_usuario.append((question, correct_intent))
        if len(self.exemplos_usuario) >= 10 and len(self.exemplos_usuario) % 10 == 0:
            self.train([q for q,_ in self.exemplos_usuario], [l for _,l in self.exemplos_usuario], epochs=20)
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "idf": self.idf, "labels": self.labels,
                         "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                         "W3": self.W3, "b3": self.b3, "exemplos": self.exemplos_usuario}, f)
    def load(self, path):
        if not os.path.isfile(path): return False
        try:
            with open(path, "rb") as f: d = pickle.load(f)
            self.vocab, self.idf, self.labels = d["vocab"], d["idf"], d["labels"]
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"]
            self.exemplos_usuario = d.get("exemplos", [])
            if self.W1: self.hidden1 = len(self.W1)
            if self.W2: self.hidden2 = len(self.W2)
            self.is_trained = True
            return True
        except Exception: return False

def gerar_dataset_dinamico():
    texts, labels = [], []
    regexes = GramaticaGenerativa.gerar_todos()
    phs = ["algo","isso","aquilo","o objeto","a coisa","o fato","o evento","a pessoa","o lugar"]
    for alvo, pats in regexes.items():
        for pat in pats:
            tpl = re.sub(r"\(\?P<entity>\.\+\?\)", "{entity}", pat)
            tpl = re.sub(r"\(\?:[^)]+\)", lambda m: m.group(0).split("|")[0].strip("(?:").strip(")"), tpl)
            tpl = tpl.replace(r"^\s*","").replace(r"(?:\?|$)","")
            if "{entity}" in tpl:
                for ph in phs: texts.append(tpl.format(entity=ph)); labels.append(alvo)
            else: texts.append(tpl); labels.append(alvo)
    return texts, labels


# ======================================================================
# BLOCKCHAIN (intacta)
# ======================================================================
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS chain0 (idx INTEGER PRIMARY KEY, timestamp REAL, entity TEXT, attribute TEXT, normalized TEXT, value_type TEXT, consensus REAL, sources TEXT, prev_hash TEXT, hash TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chain0_lookup ON chain0(entity, attribute);
CREATE TABLE IF NOT EXISTS chain1 (entity TEXT, attribute TEXT, normalized TEXT, value_type TEXT, sources TEXT, has_prior INTEGER, first_seen REAL, last_seen REAL, PRIMARY KEY (entity, attribute, normalized));
CREATE TABLE IF NOT EXISTS chain1_totals (entity TEXT, attribute TEXT, total INTEGER, PRIMARY KEY (entity, attribute));
CREATE TABLE IF NOT EXISTS net_nodes (node TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS net_edges (a TEXT NOT NULL, b TEXT NOT NULL, weight REAL NOT NULL, PRIMARY KEY (a, b));
"""

class DB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
    def execute(self, sql, params=()): return self.conn.execute(sql, params)
    def commit(self):
        if self.conn.in_transaction: self.conn.commit()

class Chain0:
    def __init__(self, db): self.db = db
    def get(self, e, a):
        r = self.db.execute("SELECT * FROM chain0 WHERE entity=? AND attribute=?", (e, a)).fetchone()
        return dict(r) if r else None
    def seal(self, cand, consensus):
        prev = self.get(cand.entity, cand.attribute)
        if prev and prev["normalized"] == cand.normalized: return prev
        n = self.db.execute("SELECT COALESCE(MAX(idx), -1) + 1 AS n FROM chain0").fetchone()["n"]
        ph = self.db.execute("SELECT hash FROM chain0 ORDER BY idx DESC LIMIT 1").fetchone()
        prev_hash = ph["hash"] if ph else "genesis"
        ts = time.time()
        h = hashlib.sha256(f"{n}|{ts}|{cand.entity}|{cand.attribute}|{cand.normalized}|{prev_hash}".encode()).hexdigest()[:16]
        self.db.execute("INSERT INTO chain0 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (n, ts, cand.entity, cand.attribute, cand.normalized, cand.value_type,
                         round(consensus, 3), json.dumps(cand.sources), prev_hash, h))
        return self.get(cand.entity, cand.attribute)

class Chain1:
    def __init__(self, db): self.db = db
    def submit(self, entity, attribute, raw, vtype, source):
        norm = self._norm(raw, vtype)
        if not norm: return None
        row = self.db.execute("SELECT * FROM chain1 WHERE entity=? AND attribute=? AND normalized=?", (entity, attribute, norm)).fetchone()
        sources = set(json.loads(row["sources"])) if row else set()
        is_new = source not in sources
        sources.add(source)
        ts = time.time()
        if row:
            self.db.execute("UPDATE chain1 SET sources=?, last_seen=? WHERE entity=? AND attribute=? AND normalized=?",
                           (json.dumps(sorted(sources)), ts, entity, attribute, norm))
        else:
            self.db.execute("INSERT INTO chain1 VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                           (entity, attribute, norm, vtype, json.dumps(sorted(sources)), ts, ts))
        if is_new:
            self.db.execute("INSERT INTO chain1_totals VALUES (?, ?, 1) ON CONFLICT(entity, attribute) DO UPDATE SET total = total + 1", (entity, attribute))
        return {"entity": entity, "attribute": attribute, "normalized": norm, "sources": sorted(sources), "value_type": vtype}
    def _norm(self, raw, vtype):
        if vtype == "year":
            m = re.search(r"(\d{4})", str(raw)); return str(int(m.group(1))) if m else None
        if vtype == "number":
            try: return f"{float(str(raw).replace('.','').replace(',','.')):.4g}"
            except: return None
        return str(raw).lower().strip() or None
    def ready(self, entity, attribute):
        total = self.db.execute("SELECT total FROM chain1_totals WHERE entity=? AND attribute=?", (entity, attribute)).fetchone()
        if not total or total["total"] < MIN_SOURCES: return []
        out = []
        for r in self.db.execute("SELECT * FROM chain1 WHERE entity=? AND attribute=?", (entity, attribute)).fetchall():
            srcs = json.loads(r["sources"])
            if len(srcs) < MIN_SOURCES: continue
            cons = len(srcs) / total["total"]
            if cons >= SEAL_THRESHOLD: out.append((r, cons))
        return out
    def best(self, entity, attribute):
        rows = self.db.execute("SELECT * FROM chain1 WHERE entity=? AND attribute=?", (entity, attribute)).fetchall()
        if not rows: return None
        return max(rows, key=lambda r: len(json.loads(r["sources"])))["normalized"]


# ======================================================================
# WEB MINER (intacto, retry 429 melhorado)
# ======================================================================
class WebMiner:
    def __init__(self, verbose=False):
        self.session = requests.Session() if requests else None
        self.verbose = verbose
        self.cache = TTLCache(ttl=WEB_CACHE_TTL, max_size=500)

    def _fetch_wiki(self, query):
        if not self.session: return None
        for attempt in range(3):
            try:
                _RATE.wait()
                r = self.session.get("https://pt.wikipedia.org/w/api.php",
                                     params={"action":"query","generator":"search","gsrsearch":query,
                                             "gsrlimit":1,"prop":"extracts","explaintext":1,"exintro":1,"format":"json","utf8":1},
                                     headers={"User-Agent":"DLM_webs/8.5 (research; contact@example.com)"},
                                     timeout=FETCH_TIMEOUT, verify=False)
                if r.status_code == 429:
                    backoff = 2 ** attempt
                    if self.verbose: safe_print(f"    [429 wiki] esperando {backoff}s")
                    time.sleep(backoff); continue
                pages = r.json().get("query",{}).get("pages",{})
                for _, p in pages.items():
                    if "extract" in p: return (p["extract"], "wikipedia")
                return None
            except Exception: pass
        return None

    def _fetch_ddg(self, query):
        return _run_with_timeout(self._ddg_impl, query, timeout=FETCH_TIMEOUT)
    def _ddg_impl(self, query):
        if HAS_DDGS:
            try:
                _RATE.wait()
                with DDGS() as ddgs:
                    res = list(ddgs.text(query, max_results=3, region="br-pt"))
                    if res: return (res[0].get("body",""), "duckduckgo")
            except Exception: pass
        return None

    def _fetch_google(self, query):
        return _run_with_timeout(self._google_impl, query, timeout=FETCH_TIMEOUT)
    def _google_impl(self, query):
        if HAS_GOOGLESEARCH:
            try:
                _RATE.wait()
                urls = list(google_search(query, num_results=1, lang="pt", region="br"))
                if urls:
                    _RATE.wait()
                    r = requests.get(urls[0], headers={"User-Agent":get_ua()}, timeout=FETCH_TIMEOUT, verify=False)
                    if r.status_code == 200 and HAS_BS4:
                        if HAS_TRAFILATURA:
                            t = trafilatura.extract(r.text)
                            if t: return (t[:800], "google")
                        soup = BeautifulSoup(r.text, "html.parser")
                        for s in soup(["script","style"]): s.decompose()
                        return (soup.get_text(" ", strip=True)[:800], "google")
            except Exception: pass
        return None

    def _fetch_bing(self, query):
        if not HAS_BS4: return None
        try:
            _RATE.wait()
            r = requests.get("https://www.bing.com/search", params={"q":query,"setlang":"pt-BR"},
                             headers={"User-Agent":get_ua(),"Accept-Language":"pt-BR,pt;q=0.9"},
                             timeout=FETCH_TIMEOUT, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                p = soup.select_one("li.b_algo p")
                if p: return (p.get_text(strip=True), "bing")
        except Exception: pass
        return None

    def _fetch_brave(self, query):
        if not HAS_BS4: return None
        try:
            _RATE.wait()
            r = requests.get("https://search.brave.com/search", params={"q":query,"source":"web"},
                             headers={"User-Agent":get_ua(),"Accept-Language":"pt-BR,pt;q=0.9"},
                             timeout=FETCH_TIMEOUT, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                snip = soup.select_one(".snippet-description, .snippet")
                if snip: return (snip.get_text(strip=True), "brave")
        except Exception: pass
        return None

    def _fetch_mojeek(self, query):
        if not HAS_BS4: return None
        try:
            _RATE.wait()
            r = requests.get("https://www.mojeek.com/search", params={"q":query},
                             headers={"User-Agent":get_ua(),"Accept-Language":"pt-BR,pt;q=0.9"},
                             timeout=FETCH_TIMEOUT, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                p = soup.select_one(".s")
                if p: return (p.get_text(strip=True), "mojeek")
        except Exception: pass
        return None

    def mine_all(self, query):
        cache_key = f"all:{query}"
        cached = self.cache.get(cache_key)
        if cached is not None: return cached
        tasks = {"wikipedia": self._fetch_wiki,
                 "duckduckgo": self._fetch_ddg,
                 "google": self._fetch_google,
                 "bing": self._fetch_bing,
                 "brave": self._fetch_brave,
                 "mojeek": self._fetch_mojeek}
        results = []
        ex = ThreadPoolExecutor(max_workers=6)
        futs = {ex.submit(fn, query): n for n, fn in tasks.items()}
        try:
            done, not_done = wait(futs, timeout=FETCH_TIMEOUT + 2)
            for f in done:
                try:
                    r = f.result(timeout=0)
                    if r and len(r[0]) > 50:
                        results.append(r)
                        if self.verbose: safe_print(f"    [{r[1]}] {len(r[0])} chars")
                except Exception: pass
            for f in not_done: f.cancel()
        finally:
            ex.shutdown(wait=False)
        results.sort(key=lambda x: 0 if x[1] == "wikipedia" else 1)
        self.cache.set(cache_key, results)
        return results


# ======================================================================
# DLM_webs v8.5
# ======================================================================
class DLM_webs:
    def __init__(self, verbose=False, state_dir=STATE_DIR):
        _ensure_dir(state_dir)
        self.db = DB(os.path.join(state_dir, DB_FILE))
        self.chain0, self.chain1 = Chain0(self.db), Chain1(self.db)
        self.net = SymbolicNetGeometric(self.db)
        self.miner = WebMiner(verbose=verbose)
        self.verbose = verbose
        self.sign = hashlib.sha256(b"dlm_webs_v8_5").hexdigest()[:8]

        self.gns = GNSAdaptive()
        model_path = os.path.join(state_dir, MODEL_FILE)
        if not self.gns.load(model_path):
            texts, labels = gerar_dataset_dinamico()
            self.gns.train(texts, labels, epochs=50)
            self.gns.save(model_path)
        self.regex_dinamicos = GramaticaGenerativa.gerar_todos()

    def _classify_adaptive(self, q):
        ql_clean = q.lower().strip().rstrip("?.")
        ql_clean = re.sub(r"^(?:o|a|os|as|um|uma)\s+", "", ql_clean, flags=re.IGNORECASE)
        ql_norm = re.sub(r"^fundad[oa]\b", "fundador", ql_clean)
        ql_norm = re.sub(r"^criad[oa]\b", "criador", ql_norm)

        m = re.match(
            r"^(presidente|criador|fundador|dono|inventor|desenvolvedor|ceo|"
            r"vice[- ]presidente|diretor)\s+(?:global\s+|atual\s+|mundial\s+|nacional\s+)?"
            r"(?:d[ao]s?|de)\s+(?P<entity>.+)$",
            ql_norm, re.IGNORECASE)
        if m:
            entity = m.group("entity").strip()
            entity = re.sub(r"^(?:empresa|companhia|firma|marca|grupo)\s+", "", entity, flags=re.IGNORECASE)
            if self.verbose: safe_print(f"  [atomic] ROLE_OF role={m.group(1).lower()}")
            return "ROLE_OF", {"role": m.group(1).lower(), "entity": entity}

        ql = q.lower()
        if re.search(r"\bquant[oa]s?\b", ql):
            if self.verbose: safe_print("  [atomic] QUANTIDADE")
            return "QUANTIDADE", {"entity": q}
        if self.gns.is_trained:
            probs = self.gns.predict(q)
            if probs:
                cat, conf = max(probs.items(), key=lambda x: x[1])
                if conf >= 0.40:
                    if self.verbose: safe_print(f"  [atomic] GNS={cat} conf={conf:.2f}")
                    return cat, self._extract_entity(q, cat)
        for cat, pats in self.regex_dinamicos.items():
            for pat in pats:
                m = re.search(pat, q, re.IGNORECASE)
                if m:
                    if self.verbose: safe_print(f"  [atomic] regex={cat}")
                    return cat, {"entity": m.groupdict().get("entity", q)}
        return "LOOKUP", {"entity": q}

    def _extract_entity(self, q, cat):
        for pat in self.regex_dinamicos.get(cat, []):
            m = re.search(pat, q, re.IGNORECASE)
            if m and "entity" in m.groupdict():
                return {"entity": m.group("entity")}
        return {"entity": q}

    def _get_or_seal(self, entity_text, attribute="date_year", vtype="year"):
        norm = normalizar_entidade(entity_text)
        if not norm: return None
        sealed = self.chain0.get(norm, attribute)
        if sealed:
            self.net.reinforce(norm, attribute, 0.3)
            return sealed["normalized"]
        similar, _ = self.net.fuzzy_match(norm, threshold_topo=3.0)
        if similar:
            sealed = self.chain0.get(similar, attribute)
            if sealed:
                self.net.reinforce(norm, similar, 0.5)
                return sealed["normalized"]
        sources = self.miner.mine_all(norm)
        for text, src in sources:
            if vtype == "year":
                y = self.net._best_year_for(text, norm)
                if y:
                    self.chain1.submit(norm, attribute, y, vtype, src)
                    self.net.reinforce(norm, attribute, 0.1)
                    ready = self.chain1.ready(norm, attribute)
                    if ready:
                        r, cons = ready[0]
                        cand = type("C", (), {"entity": norm, "attribute": attribute,
                                              "normalized": r["normalized"], "value_type": vtype,
                                              "sources": r["sources"]})()
                        self.chain0.seal(cand, cons)
                        self.net.reinforce(norm, attribute, 0.8)
                        return r["normalized"]
        return self.chain1.best(norm, attribute)

    # ================================================================
    # HANDLERS com Evidence Map + patches v8.5
    # ================================================================
    def _handle_creator(self, q, params):
        entity = params.get("entity", "").strip()
        entity = re.sub(r"^(?:a|o|as|os|um|uma|da|do|das|dos|de)\s+", "", entity, flags=re.IGNORECASE)
        norm = normalizar_entidade(entity)
        self.net.reinforce("CRIADOR_DE", norm, 0.1)
        sources = self.miner.mine_all(entity)
        if self.verbose: safe_print(f"  [creator] {entity} ({len(sources)} fontes)")

        emap = EvidenceMap()
        for text, src in sources:
            for para in split_paragrafos(text):
                # PATCH v8.5: verbos sem IGNORECASE, nome com NOME_PROPRIO_PATTERN (case-sensitive)
                for padrao in [
                    r"(?i:gerenciad[ao]|criad[ao]|desenvolvidad[ao]|fundad[ao]|produzid[ao]|publicad[ao])"
                    r"\s+(?i:pel[ao])\s+" + NOME_PROPRIO_PATTERN,
                    r"(?i:é\s+um[a]?)\s+(?:\w+\s+){0,4}(?i:d[ao])\s+" + NOME_PROPRIO_PATTERN,
                ]:
                    m = re.search(padrao, para)
                    if m:
                        cand = m.group(1).strip()
                        # limpeza pós: remove "de/e" no final
                        cand = re.sub(r"\s+(?:de|da|do|dos|das|e)$", "", cand)
                        cn = normalizar_entidade(cand)
                        if len(cn.split()) < 2: continue
                        if cn == norm: continue
                        emap.add(cand, src, para)
                        break

        vencedor, para_v, ranking = emap.consensus(min_fontes=2, allow_wiki_solo=True)
        if self.verbose and ranking:
            safe_print(f"  [evidence-map] {len(ranking)} candidatos:")
            for nf, ne, cn, evs in ranking[:5]:
                fontes = ", ".join(sorted(set(e["fonte"] for e in evs)))
                safe_print(f"    {cn}: {nf} fontes ({fontes})")

        if vencedor:
            self.net.reinforce(norm, normalizar_entidade(vencedor), 0.6)
            self.net.reinforce("CRIADOR_DE", norm, 0.2)
            if self.verbose:
                safe_print(f"  [consenso] {vencedor}")
                safe_print(f"  [parágrafo] {para_v[:150]}")
            return vencedor
        return None

    def _handle_role(self, q, params):
        role = params.get("role", "").strip()
        entity = params.get("entity", "").strip()
        if not role or not entity: return None
        norm = normalizar_entidade(entity)
        self.net.reinforce("ROLE_OF", norm, 0.1)
        self.net.reinforce(role, norm, 0.2)

        sources = self.miner.mine_all(entity)
        # PATCH v8.5: buscas ampliadas para presidente/CEO
        if role in ("presidente", "ceo", "diretor", "vice-presidente"):
            sources.extend(self.miner.mine_all(f"presidente atual {entity}"))
            sources.extend(self.miner.mine_all(f"quem é o {role} da {entity}"))
            sources.extend(self.miner.mine_all(f"{role} {entity} 2024 2025"))
            sources.extend(self.miner.mine_all(f"{entity} CEO presidente história"))

        if self.verbose: safe_print(f"  [role] {role} de {entity} ({len(sources)} fontes)")

        role_en = {"presidente": "president", "criador": "creator",
                   "fundador": "founder", "dono": "owner",
                   "inventor": "inventor", "desenvolvedor": "developer",
                   "ceo": "CEO", "vice-presidente": "vice president",
                   "diretor": "director"}.get(role, role)

        emap = EvidenceMap()
        for text, src in sources:
            for para in split_paragrafos(text):
                cands = []
                # P1: SÓ para fundador/criador/inventor/desenvolvedor (evita Fusajiro como presidente)
                if role in ("fundador", "criador", "inventor", "desenvolvedor"):
                    m = re.search(
                        r"(?i:fundad[ao]|criad[ao]|estabelecid[ao]|iniciad[ao])"
                        r"[^.]{0,150}?(?i:pel[ao]|por)\s+(?:[a-zá-ú]+\s+){0,3}" + NOME_PROPRIO_PATTERN,
                        para)
                    if m: cands.append(m.group(1))
                # P2: "X, o role"
                m = re.search(
                    r"(" + NOME_PROPRIO_PATTERN[1:-1] + r")\s*,\s*"
                    r"(?i:o\s+|a\s+)?(?:atual\s+)?" + re.escape(role) + r"\b",
                    para)
                if m: cands.append(m.group(1))
                # P3: "X é o role"
                m = re.search(
                    r"(" + NOME_PROPRIO_PATTERN[1:-1] + r")\s+"
                    r"(?i:é|foi)\s+(?i:o\s+|a\s+)?(?:atual\s+)?" + re.escape(role) + r"\b",
                    para)
                if m: cands.append(m.group(1))
                # P4: "role da Y é X"
                m = re.search(
                    re.escape(role) + r"\s+d[ao][^.]{0,80}?(?i:é|foi)\s+(?i:o\s+|a\s+)?"
                    + NOME_PROPRIO_PATTERN,
                    para)
                if m: cands.append(m.group(1))
                # P5: EN
                m = re.search(
                    r"(" + NOME_PROPRIO_PATTERN[1:-1] + r")\s+(?i:is)\s+"
                    r"(?i:the\s+)?(?:current\s+)?" + re.escape(role_en),
                    para)
                if m: cands.append(m.group(1))
                # P6: EN
                m = re.search(
                    re.escape(role_en) + r"\s+of\s+[A-ZÁ-Ú][a-z-ú]+,?\s+" + NOME_PROPRIO_PATTERN,
                    para)
                if m: cands.append(m.group(1))

                for c in cands:
                    cn = normalizar_entidade(c)
                    if len(cn.split()) < 2: continue
                    if cn == norm: continue
                    if cn in {"presidente", "fundador", "criador"}: continue
                    emap.add(c, src, para)

        vencedor, para_v, ranking = emap.consensus(min_fontes=2, allow_wiki_solo=True)
        if self.verbose and ranking:
            safe_print(f"  [evidence-map] {len(ranking)} candidatos:")
            for nf, ne, cn, evs in ranking[:5]:
                fontes = ", ".join(sorted(set(e["fonte"] for e in evs)))
                safe_print(f"    {cn}: {nf} fontes ({fontes})")

        if vencedor:
            self.net.reinforce(norm, "ROLE_OF", 0.5)
            self.net.reinforce(norm, role, 0.5)
            self.net.reinforce(norm, normalizar_entidade(vencedor), 0.6)
            if self.verbose:
                safe_print(f"  [consenso] {vencedor}")
                safe_print(f"  [parágrafo] {para_v[:150]}")
            return vencedor

        if self.verbose: safe_print(f"  [sem-consenso]")
        return None

    def _handle_quantity(self, q, params):
        entity = params.get("entity", "").strip()
        entity = re.sub(r"^(?:quantos?|quantas?)\s+", "", entity, flags=re.IGNORECASE)
        entity = re.sub(r"\s+tem\s+.*$", "", entity, flags=re.IGNORECASE)
        entity = re.sub(r"^(?:a|o|as|os|um|uma|da|do|das|dos|de)\s+", "", entity, flags=re.IGNORECASE)
        norm = normalizar_entidade(entity)
        self.net.reinforce("QUANTIDADE", norm, 0.1)
        sources = self.miner.mine_all(entity)
        if self.verbose: safe_print(f"  [quantity] {entity} ({len(sources)} fontes)")

        emap = EvidenceMap()
        for text, src in sources:
            for para in split_paragrafos(text):
                # PATCH v8.5: usa regex compilados do catálogo
                m = REGEX_QUANTIDADE.search(para)
                if m:
                    emap.add(f"{m.group(1)} {m.group(2)} de habitantes", src, para)
                    continue
                m = REGEX_QUANTIDADE_POP.search(para)
                if m:
                    emap.add(f"{m.group(1)} {m.group(2)}", src, para)
                    continue
                m = REGEX_QUANTIDADE_RAW.search(para)
                if m:
                    emap.add(f"{m.group(1)} habitantes", src, para)

        # PATCH v8.5: quantidade é determinística → 1 fonte confiável basta
        vencedor, para_v, ranking = emap.consensus(min_fontes=1, allow_wiki_solo=False)
        if self.verbose and ranking:
            safe_print(f"  [evidence-map] {len(ranking)} candidatos:")
            for nf, ne, cn, evs in ranking[:3]:
                fontes = ", ".join(sorted(set(e["fonte"] for e in evs)))
                safe_print(f"    {cn}: {nf} fontes ({fontes})")

        if vencedor:
            self.net.reinforce(norm, "QUANTIDADE", 0.3)
            if self.verbose:
                safe_print(f"  [consenso] {vencedor}")
                safe_print(f"  [parágrafo] {para_v[:150]}")
            return vencedor
        return None

    def _handle_date_of(self, q, params):
        entity = params.get("entity", "").strip()
        ql = q.lower()
        is_death = "morreu" in ql or "faleceu" in ql
        if not is_death:
            v = self._get_or_seal(entity, "date_year", "year")
            if v: return v
        sources = self.miner.mine_all(entity)
        norm = normalizar_entidade(entity)
        if self.verbose: safe_print(f"  [date] {entity} ({len(sources)} fontes)")

        emap = EvidenceMap()
        for text, src in sources:
            for para in split_paragrafos(text):
                if is_death:
                    m = re.search(r"(?:morreu|faleceu)\s+em\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})", para, re.IGNORECASE)
                    if m:
                        emap.add(m.group(1), src, para); continue
                    # PATCH v8.5: usa REGEX_DATAS compilados
                    m = REGEX_DATAS["intervalo"].search(para)
                    if m:
                        # extrai data completa do lado direito do intervalo (morte)
                        parte_dir = para[m.start():m.end()].split("—")[-1] if "—" in para[m.start():m.end()] else para[m.start():m.end()]
                        full = REGEX_DATAS["extenso_pt"].search(parte_dir)
                        if full:
                            emap.add(f"{full.group('dia')} de {full.group('mes')} de {full.group('ano')}", src, para)
                            continue
                        full = REGEX_DATAS["br_numerica"].search(parte_dir)
                        if full:
                            emap.add(f"{full.group('dia')}/{full.group('mes')}/{full.group('ano')}", src, para)

        if is_death and not emap.is_empty():
            # PATCH v8.5: data também é determinística → min_fontes=1
            vencedor, para_v, ranking = emap.consensus(min_fontes=1, allow_wiki_solo=False)
            if self.verbose and ranking:
                safe_print(f"  [evidence-map] {len(ranking)} candidatos")
                for nf, ne, cn, evs in ranking[:3]:
                    fontes = ", ".join(sorted(set(e["fonte"] for e in evs)))
                    safe_print(f"    {cn}: {nf} fontes ({fontes})")
            if vencedor:
                self.net.reinforce(norm, "death_date", 0.5)
                if self.verbose:
                    safe_print(f"  [consenso] {vencedor}")
                    safe_print(f"  [parágrafo] {para_v[:150]}")
                return vencedor
        return None

    def _handle_lookup(self, q, params):
        entity = params.get("entity", "").strip()
        entity = re.sub(r"^(?:a|o|as|os|um|uma|da|do|das|dos|de)\s+", "", entity, flags=re.IGNORECASE)
        norm = normalizar_entidade(entity)
        self.net.reinforce("LOOKUP", norm, 0.1)
        sources = self.miner.mine_all(entity)

        # PATCH v8.5: preferência explícita por Wikipedia
        wiki_source = next((s for s in sources if s[1] == "wikipedia"), None)
        if wiki_source and len(wiki_source[0]) > 200:
            self.net.reinforce(norm, "LOOKUP", 0.3)
            paras = split_paragrafos(wiki_source[0])
            if paras:
                return paras[0][:400]

        # Fallback: qualquer fonte com texto substancial (>300 chars) e sem padrão de snippet
        for text, src in sources:
            if len(text) < 300: continue
            if REGEX_SNIPPET_NEWS.search(text[:200]): continue  # filtra "19 de jul. de 2026 · Tenha acesso..."
            paras = split_paragrafos(text)
            if paras:
                self.net.reinforce(norm, "LOOKUP", 0.2)
                return paras[0][:400]

        # Último recurso: snippet curto (marca fraco)
        if sources:
            return sources[0][0][:400]
        return None

    def _handle_where(self, q, params):
        return self._handle_lookup(q, params)

    def falar(self, question):
        t0 = time.perf_counter()
        intent, params = self._classify_adaptive(question)
        self.net.decay(0.995)
        answer, mode = None, "?"
        try:
            if intent == "SAUDACAO": answer, mode = "Olá! Sou DLM_webs geométrico.", "SAUDACAO"
            elif intent in ("DATA_MORTE", "DATA_EVENTO", "DATA_NASCIMENTO"):
                answer = self._handle_date_of(question, params); mode = intent
            elif intent == "CRIADOR_DE": answer = self._handle_creator(question, params); mode = "CRIADOR"
            elif intent == "ROLE_OF": answer = self._handle_role(question, params); mode = "ROLE_OF"
            elif intent == "QUANTIDADE": answer = self._handle_quantity(question, params); mode = "QUANTIDADE"
            elif intent == "LOCALIZACAO": answer = self._handle_where(question, params); mode = "LOCAL"
            elif intent in ("LOOKUP_PESSOA", "ESPECIFICAR", "LOOKUP"): answer = self._handle_lookup(question, params); mode = "LOOKUP"
            else: answer = self._handle_lookup(question, params); mode = "FALLBACK"
            if not answer: answer, mode = "Sem dados suficientes.", "SEM-FONTE"
            if answer and mode != "SEM-FONTE":
                self.gns.learn_online(question, intent)
                ent = normalizar_entidade(params.get("entity", ""))
                if ent: self.net.reinforce(intent, ent, 0.05)
        except Exception as e:
            answer, mode = f"Erro: {e}", "ERRO"
        self.db.commit()
        self.gns.save(os.path.join(STATE_DIR, MODEL_FILE))
        elapsed = (time.perf_counter() - t0) * 1000
        ns = self.net.stats()
        return (f"[DLM_webs v8.5 | {elapsed:.0f}ms | {mode} | "
                f"G:{ns['nodes']}n/{ns['edges']}e | SIGN:{self.sign}]\n{answer}")

    def cmd_mapa(self):
        emb = self.net.spectral_embedding(dim=2)
        if not emb:
            return f"Grafo com {len(self.net.nodes)} nós. Faça mais perguntas."
        xs = [c[0] for c in emb.values()]; ys = [c[1] for c in emb.values()]
        minx, maxx = min(xs), max(xs) or 1.0
        miny, maxy = min(ys), max(ys) or 1.0
        W, H = 60, 20
        legend = {}
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        for i, n in enumerate(sorted(emb.keys())):
            if i < len(chars): legend[chars[i]] = n
        grid = [[' '] * W for _ in range(H)]
        for c, n in legend.items():
            x, y = emb[n]
            gx = int((x - minx) / (maxx - minx + 1e-9) * (W - 1))
            gy = int((y - miny) / (maxy - miny + 1e-9) * (H - 1))
            gx, gy = max(0, min(W - 1, gx)), max(0, min(H - 1, gy))
            grid[gy][gx] = c
        lines = ["=== MAPA GEOMÉTRICO ==="]
        for row in grid: lines.append(''.join(row))
        lines.append(f"\nNós: {len(emb)} | Arestas: {len(self.net.edges)} | Dens: {self.net.stats()['density']:.3f}")
        lines.append("\n=== Legenda ===")
        for c, n in legend.items(): lines.append(f"  {c}: {n}")
        return "\n".join(lines)

    def cmd_vizinhos(self, entity):
        norm = normalizar_entidade(entity)
        if norm not in self.net.nodes:
            return f"'{norm}' não está no grafo."
        viz = self.net.nearest(norm, k=10, threshold=10.0)
        if not viz: return f"'{norm}' sem vizinhos topológicos."
        lines = [f"=== Vizinhos de '{norm}' ==="]
        for d, v in viz: lines.append(f"  {v:<30} d={d:.3f}")
        return "\n".join(lines)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8")
        except Exception: pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--mapa", action="store_true")
    parser.add_argument("--vizinhos", type=str, metavar="ENTITY")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.reset:
        import shutil
        if os.path.isdir(STATE_DIR): shutil.rmtree(STATE_DIR)
        safe_print("[reset] Estado apagado.")
    dlm = DLM_webs(verbose=args.verbose)
    if args.query: print(dlm.falar(args.query))
    elif args.mapa: print(dlm.cmd_mapa())
    elif args.vizinhos: print(dlm.cmd_vizinhos(args.vizinhos))
    else:
        perguntas = [
            "quem criou a steam",
            "quando morreu silvio santos",
            "aonde fica belford roxo",
            "O Brasil tem quantos habitantes",
            "quem é o presidente lula",
            "O fundador da Nintendo",
            "O presidente da empresa Nintendo",
            "O presidente global da empresa Nintendo",
        ]
        for q in perguntas:
            print(f"\n>>> {q}")
            print(dlm.falar(q))
