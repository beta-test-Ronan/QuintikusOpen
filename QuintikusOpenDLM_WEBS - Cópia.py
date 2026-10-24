#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLM_webs v7.2 — GNS Adaptativa + Multi-Search (6 provedores) + Blockchain.
Correções: Prioridade de SORT por vírgulas e reescrita semântica para QUANTIDADE.
"""

import sys, re, json, math, time, hashlib, os, sqlite3, warnings, argparse, random, pickle, unicodedata, difflib
from collections import defaultdict
from urllib.parse import quote, parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    try: requests.packages.urllib3.disable_warnings()
    except Exception: pass
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    from googlesearch import search as google_search
    HAS_GOOGLESEARCH = True
except ImportError:
    HAS_GOOGLESEARCH = False

STATE_DIR = os.path.join(os.path.expanduser("~"), ".dlm_webs")
DB_FILE = "dlm.db"
MODEL_FILE = "modelo_gns_adaptive.pkl"
MIN_SOURCES = 2
SEAL_THRESHOLD = 0.5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def get_ua(): return random.choice(USER_AGENTS)

def safe_print(*a, **k):
    try: print(*a, **k)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(" ".join(str(x) for x in a).encode(enc, "replace").decode(enc, "replace"), **k)

def _ensure_dir(d):
    if not os.path.isdir(d): os.makedirs(d, exist_ok=True)

def remove_accents(text):
    return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

# ======================================================================
# PRIMITIVOS E GRAMÁTICA
# ======================================================================
STOPWORDS = {"a", "o", "as", "os", "um", "uma", "uns", "umas",
             "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
             "para", "com", "por", "que", "e", "ou", "mas", "se", "já", "não"}

class GramaticaGenerativa:
    PADROES_ABSTRATOS = [
        ("AGENTE", "CRIACAO", "CRIADOR_DE"),
        ("AGENTE", "FUNDACAO", "CRIADOR_DE"),
        ("AGENTE", "INVENCAO", "CRIADOR_DE"),
        ("AGENTE", "DESENVOLVIMENTO", "CRIADOR_DE"),
        ("TEMPO", "MORTE", "DATA_MORTE"),
        ("TEMPO", "NASCIMENTO", "DATA_NASCIMENTO"),
        ("TEMPO", "EVENTO", "DATA_EVENTO"),
        ("TEMPO", "PASSADO", "DATA_EVENTO"),
        ("LUGAR", "LOCALIZACAO", "LOCALIZACAO"),
        ("QUANTIDADE", None, "QUANTIDADE"),
        ("LISTAR", None, "LISTA"),
        ("ORDENAR", None, "ORDENAR"),
        ("ESPECIFICAR", None, "ESPECIFICAR"),
        ("AGENTE", "PASSADO", "LOOKUP_PESSOA"),
        ("AGENTE", "PRESENTE", "LOOKUP_PESSOA"),
        ("PROCESSO", None, "PROCESSO"),
        ("CAUSA", None, "CAUSA"),
        ("SAUDACAO", None, "SAUDACAO"),
    ]
    
    @classmethod
    def gerar_regex(cls, wh, verbo, alvo):
        wh_map = {
            "AGENTE": r"quem", "TEMPO": r"(?:quando|em\s+que\s+ano|em\s+que\s+data)",
            "LUGAR": r"(?:aonde|onde|em\s+que\s+lugar)", "QUANTIDADE": r"(?:quantos?|quantas?)",
            "LISTAR": r"(?:quais|liste|enumere)", "ORDENAR": r"(?:ordene|ordene\s+cronologicamente)",
            "ESPECIFICAR": r"(?:qual|quais)", "PROCESSO": r"como", "CAUSA": r"por\s+que",
            "SAUDACAO": r"(?:oi|olá|bom\s+dia|boa\s+tarde|boa\s+noite)"
        }
        wh_pat = wh_map.get(wh)
        if not wh_pat: return None
        
        verbo_map = {
            "CRIACAO": r"(?:criou|fez|desenvolveu|inventou)", "FUNDACAO": r"(?:fundou|é\s+o\s+fundador)",
            "INVENCAO": r"(?:inventou|criou)", "DESENVOLVIMENTO": r"(?:desenvolveu|criou)",
            "MORTE": r"(?:morreu|faleceu)", "NASCIMENTO": r"nasceu",
            "EVENTO": r"(?:aconteceu|ocorreu|sucedeu|foi)", "PASSADO": r"foi",
            "PRESENTE": r"é", "LOCALIZACAO": r"(?:fica|está|localizado)"
        }
        
        if verbo:
            verbo_pat = verbo_map.get(verbo, r"\w+")
            return rf"^{wh_pat}\s+{verbo_pat}\s+(?:o\s+|a\s+|os\s+|as\s+)?(?P<entity>.+?)(?:\?|$)"
        else:
            return rf"^{wh_pat}\s+(?:o\s+|a\s+|os\s+|as\s+)?(?P<entity>.+?)(?:\?|$)"
    
    @classmethod
    def gerar_todos(cls):
        regexes = {}
        for wh, verbo, alvo in cls.PADROES_ABSTRATOS:
            pat = cls.gerar_regex(wh, verbo, alvo)
            if pat: regexes.setdefault(alvo, []).append(pat)
        return regexes

# ======================================================================
# REDE GNS
# ======================================================================
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
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x/norm for x in vec]

    def _init_weights(self):
        V, C, H1, H2 = len(self.vocab), len(self.labels), self.hidden1, self.hidden2
        self.W1 = [[random.uniform(-0.1, 0.1) / (1.0 + abs(i-j)/10.0) for j in range(V)] for i in range(H1)]
        self.b1 = [0.0] * H1
        self.W2 = [[random.uniform(-0.1, 0.1) for j in range(H1)] for i in range(H2)]
        self.b2 = [0.0] * H2
        self.W3 = [[random.uniform(-0.1, 0.1) for j in range(H2)] for i in range(C)]
        self.b3 = [0.0] * C

    def train(self, texts, labels, epochs=50):
        if not texts: return
        self.labels = sorted(list(set(labels)))
        self._build_tfidf(texts)
        self._init_weights()
        X = [self._vectorize(t) for t in texts]
        Y = []
        for l in labels:
            y = [0.0] * len(self.labels); y[self.labels.index(l)] = 1.0; Y.append(y)
        V, C, H1, H2 = len(self.vocab), len(self.labels), self.hidden1, self.hidden2
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                h1 = [math.tanh(sum(self.W1[i][j]*x[j] for j in range(V)) + self.b1[i]) for i in range(H1)]
                h2 = [math.tanh(sum(self.W2[i][j]*h1[j] for j in range(H1)) + self.b2[i]) for i in range(H2)]
                out = [sum(self.W3[i][j]*h2[j] for j in range(H2)) + self.b3[i] for i in range(C)]
                m = max(out); exp = [math.exp(o - m) for o in out]; s = sum(exp)
                probs = [e/s for e in exp]
                d_out = [probs[i] - y[i] for i in range(C)]
                d_h2 = [sum(d_out[j] * self.W3[j][i] for j in range(C)) * (1 - h2[i]**2) for i in range(H2)]
                d_h1 = [sum(d_h2[j] * self.W2[j][i] for j in range(H2)) * (1 - h1[i]**2) for i in range(H1)]
                for i in range(C):
                    for j in range(H2): self.W3[i][j] -= self.lr * d_out[i] * h2[j]
                    self.b3[i] -= self.lr * d_out[i]
                for i in range(H2):
                    for j in range(H1): self.W2[i][j] -= self.lr * d_h2[i] * h1[j]
                    self.b2[i] -= self.lr * d_h2[i]
                for i in range(H1):
                    for j in range(V): self.W1[i][j] -= self.lr * d_h1[i] * x[j]
                    self.b1[i] -= self.lr * d_h1[i]
        self.is_trained = True

    def predict(self, text):
        if not self.is_trained: return {}
        x = self._vectorize(text)
        V, C, H1, H2 = len(self.vocab), len(self.labels), self.hidden1, self.hidden2
        if len(x) != len(self.W1[0]): x = x[:len(self.W1[0])] + [0.0] * max(0, len(self.W1[0]) - len(x))
        h1 = [math.tanh(sum(self.W1[i][j]*x[j] for j in range(len(self.W1[0]))) + self.b1[i]) for i in range(H1)]
        h2 = [math.tanh(sum(self.W2[i][j]*h1[j] for j in range(H1)) + self.b2[i]) for i in range(H2)]
        out = [sum(self.W3[i][j]*h2[j] for j in range(H2)) + self.b3[i] for i in range(C)]
        m = max(out); exp = [math.exp(o - m) for o in out]; s = sum(exp)
        return {self.labels[i]: exp[i]/s for i in range(C)}

    def learn_online(self, question, correct_intent):
        self.exemplos_usuario.append((question, correct_intent))
        if len(self.exemplos_usuario) >= 10 and len(self.exemplos_usuario) % 10 == 0:
            texts = [q for q, _ in self.exemplos_usuario]
            labels = [l for _, l in self.exemplos_usuario]
            self.train(texts, labels, epochs=20)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "idf": self.idf, "labels": self.labels,
                         "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                         "W3": self.W3, "b3": self.b3,
                         "exemplos": self.exemplos_usuario}, f)

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
    placeholders = ["algo", "isso", "aquilo", "o objeto", "a coisa", "o fato", "o evento", "a pessoa", "o lugar"]
    for alvo, patterns in regexes.items():
        for pat in patterns:
            template = re.sub(r"\(\?P<entity>\.\+\?\)", "{entity}", pat)
            template = re.sub(r"\(\?:[^)]+\)", lambda m: m.group(0).split("|")[0].strip("(?:").strip(")"), template)
            template = template.replace(r"^\s*", "").replace(r"(?:\?|$)", "")
            if "{entity}" in template:
                for ph in placeholders:
                    texts.append(template.format(entity=ph)); labels.append(alvo)
            else:
                texts.append(template); labels.append(alvo)
    return texts, labels

def normalizar_entidade(text):
    if not text: return ""
    t = unicodedata.normalize('NFD', text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != 'Mn')
    t = re.sub(r"^(?:ordene|liste|enumere|cite|relacione|quem|quando|onde|aonde|quantos?|qual|quais)\s+", "", t)
    t = re.sub(r"\b(?:foi|foram|é|era|são|aconteceu|ocorreu|sucedeu|fundad[oa]|criad[oa]|inaugurad[oa]|proclamad[oa]|abolida|abolido)\b", " ", t)
    t = re.sub(r"\s+(?:em|no|na|nos|nas)\s+[a-z][\w\s]*$", "", t)
    for w in STOPWORDS: t = re.sub(rf"\b{w}\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def encontrar_similar(entidade, db, threshold=0.85):
    ent_norm = normalizar_entidade(entidade)
    if not ent_norm: return None
    for r in db.execute("SELECT entity FROM chain0").fetchall():
        db_ent = r["entity"]
        if difflib.SequenceMatcher(None, ent_norm, db_ent).ratio() >= threshold:
            return db_ent
    return None

# ======================================================================
# BLOCKCHAIN
# ======================================================================
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS chain0 (idx INTEGER PRIMARY KEY, timestamp REAL, entity TEXT, attribute TEXT, normalized TEXT, value_type TEXT, consensus REAL, sources TEXT, prev_hash TEXT, hash TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chain0_lookup ON chain0(entity, attribute);
CREATE TABLE IF NOT EXISTS chain1 (entity TEXT, attribute TEXT, normalized TEXT, value_type TEXT, sources TEXT, has_prior INTEGER, first_seen REAL, last_seen REAL, PRIMARY KEY (entity, attribute, normalized));
CREATE TABLE IF NOT EXISTS chain1_totals (entity TEXT, attribute TEXT, total INTEGER, PRIMARY KEY (entity, attribute));
"""

class DB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, isolation_level=None)
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
                        (n, ts, cand.entity, cand.attribute, cand.normalized, cand.value_type, round(consensus, 3), json.dumps(cand.sources), prev_hash, h))
        return self.get(cand.entity, cand.attribute)

class Chain1:
    def __init__(self, db): self.db = db
    def submit(self, entity, attribute, raw, vtype, source):
        norm = self._normalize(raw, vtype)
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
        return {"entity": entity, "attribute": attribute, "normalized": norm, "sources": sorted(sources)}
    def _normalize(self, raw, vtype):
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
        best = max(rows, key=lambda r: len(json.loads(r["sources"])))
        return best["normalized"]

# ======================================================================
# MULTI-PROVIDER WEB MINER
# ======================================================================
class WebMiner:
    def __init__(self):
        self.session = requests.Session() if requests else None

    def _fetch_wiki(self, query):
        if not self.session: return None
        try:
            r = self.session.get("https://pt.wikipedia.org/w/api.php", 
                                 params={"action":"opensearch", "search":query, "limit":1, "format":"json"},
                                 headers={"User-Agent": "DLM_webs/7.2 (research bot; contact@example.com)"},
                                 timeout=5, verify=False)
            title = r.json()[1][0] if r.status_code == 200 and r.json()[1] else query
        except Exception: title = query
        
        try:
            r = self.session.get("https://pt.wikipedia.org/w/api.php",
                                 params={"action":"query", "generator":"search", "gsrsearch":title,
                                         "gsrlimit":1, "prop":"extracts", "explaintext":1, "exintro":1,
                                         "format":"json", "utf8":1},
                                 headers={"User-Agent": "DLM_webs/7.2 (research bot; contact@example.com)"},
                                 timeout=8, verify=False)
            if r.status_code == 429: return None
            pages = r.json().get("query", {}).get("pages", {})
            for _, p in pages.items():
                if "extract" in p: return (p["extract"], "wikipedia")
        except Exception: pass
        return None

    def _fetch_ddg(self, query):
        if HAS_DDGS:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3, region="br-pt"))
                    if results: return (results[0].get("body", ""), "duckduckgo")
            except Exception: pass
        if not HAS_BS4: return None
        try:
            r = requests.post("https://lite.duckduckgo.com/lite/", data={"q": query},
                              headers={"User-Agent": get_ua()}, timeout=10, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                snippet = soup.select_one(".result-snippet")
                if snippet: return (snippet.get_text(strip=True), "ddg_lite")
        except Exception: pass
        return None

    def _fetch_google(self, query):
        if HAS_GOOGLESEARCH:
            try:
                urls = list(google_search(query, num_results=1, lang="pt", region="br"))
                if urls:
                    r = requests.get(urls[0], headers={"User-Agent": get_ua()}, timeout=8, verify=False)
                    if r.status_code == 200:
                        if HAS_TRAFILATURA:
                            text = trafilatura.extract(r.text)
                            if text: return (text[:800], "google")
                        if HAS_BS4:
                            soup = BeautifulSoup(r.text, "html.parser")
                            for s in soup(["script","style"]): s.decompose()
                            return (soup.get_text(" ", strip=True)[:800], "google")
            except Exception: pass
        return None

    def _fetch_bing(self, query):
        if not HAS_BS4: return None
        try:
            r = requests.get("https://www.bing.com/search", params={"q": query, "setlang": "pt-BR"},
                             headers={"User-Agent": get_ua(), "Accept-Language": "pt-BR,pt;q=0.9"},
                             timeout=10, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                p = soup.select_one("li.b_algo p")
                if p: return (p.get_text(strip=True), "bing")
        except Exception: pass
        return None

    def _fetch_brave(self, query):
        if not HAS_BS4: return None
        try:
            r = requests.get("https://search.brave.com/search", params={"q": query, "source": "web"},
                             headers={"User-Agent": get_ua(), "Accept-Language": "pt-BR,pt;q=0.9"},
                             timeout=10, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                snippet = soup.select_one(".snippet-description, .snippet")
                if snippet: return (snippet.get_text(strip=True), "brave")
        except Exception: pass
        return None

    def _fetch_mojeek(self, query):
        if not HAS_BS4: return None
        try:
            r = requests.get("https://www.mojeek.com/search", params={"q": query},
                             headers={"User-Agent": get_ua(), "Accept-Language": "pt-BR,pt;q=0.9"},
                             timeout=10, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                p = soup.select_one(".s")
                if p: return (p.get_text(strip=True), "mojeek")
        except Exception: pass
        return None

    def mine_all(self, query):
        tasks = {
            "wikipedia": lambda: self._fetch_wiki(query),
            "duckduckgo": lambda: self._fetch_ddg(query),
            "google": lambda: self._fetch_google(query),
            "bing": lambda: self._fetch_bing(query),
            "brave": lambda: self._fetch_brave(query),
            "mojeek": lambda: self._fetch_mojeek(query),
        }
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(future_to_name):
                try:
                    res = future.result(timeout=12)
                    if res and len(res[0]) > 50: results.append(res)
                except Exception: pass
        results.sort(key=lambda x: 0 if x[1] == "wikipedia" else 1)
        return results

# ======================================================================
# DLM_webs CORE
# ======================================================================
class DLM_webs:
    def __init__(self, verbose=False, state_dir=STATE_DIR):
        os.makedirs(state_dir, exist_ok=True)
        self.db = DB(os.path.join(state_dir, DB_FILE))
        self.chain0, self.chain1 = Chain0(self.db), Chain1(self.db)
        self.miner = WebMiner()
        self.verbose = verbose
        self.sign = hashlib.sha256(b"dlm_webs_v7_2").hexdigest()[:8]
        
        self.gns = GNSAdaptive()
        model_path = os.path.join(state_dir, MODEL_FILE)
        if not self.gns.load(model_path):
            texts, labels = gerar_dataset_dinamico()
            self.gns.train(texts, labels, epochs=50)
            self.gns.save(model_path)
        
        self.regex_dinamicos = GramaticaGenerativa.gerar_todos()

    def _classify_adaptive(self, q):
        ql = q.lower()
        # FIX 1: Se tiver 2 ou mais vírgulas e não for pergunta, FORÇA ORDENAR antes da GNS
        if ql.count(",") >= 2 and "?" not in ql:
            if self.verbose: safe_print("  [force] ORDENAR (vírgulas)")
            return "ORDENAR", {"entity": q}

        # 1. GNS
        if self.gns.is_trained:
            probs = self.gns.predict(q)
            if probs:
                cat, conf = max(probs.items(), key=lambda x: x[1])
                if conf >= 0.40:
                    if self.verbose: safe_print(f"  [gns] {cat} conf={conf:.2f}")
                    return cat, self._extract_entity(q, cat)
        
        # 2. Regex
        for cat, patterns in self.regex_dinamicos.items():
            for pat in patterns:
                m = re.search(pat, q, re.IGNORECASE)
                if m:
                    if self.verbose: safe_print(f"  [regex] {cat}")
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
        if sealed: return sealed["normalized"]
        
        similar = encontrar_similar(norm, self.db, threshold=0.85)
        if similar:
            sealed = self.chain0.get(similar, attribute)
            if sealed:
                if self.verbose: safe_print(f"  [fuzzy] {norm} ~ {similar}")
                return sealed["normalized"]
        
        sources = self.miner.mine_all(norm)
        if self.verbose: safe_print(f"  [mine] {len(sources)} fontes para '{norm}'")
        
        for text, src in sources:
            if vtype == "year":
                for m in re.finditer(r'(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)', text):
                    self.chain1.submit(norm, attribute, m.group(1), vtype, src)
                    ready = self.chain1.ready(norm, attribute)
                    if ready:
                        r, cons = ready[0]
                        cand = type("C", (), {"entity": norm, "attribute": attribute, "normalized": r["normalized"], "value_type": vtype, "sources": r["sources"]})()
                        self.chain0.seal(cand, cons)
                        if self.verbose: safe_print(f"  [chain0] SELADO: {norm} = {r['normalized']}")
                        return r["normalized"]
        
        return self.chain1.best(norm, attribute)

    def _handle_sort(self, q):
        body = q.split(":", 1)[-1].strip()
        parts = re.split(r"[,;]|\s+e\s+|\s+ou\s+", body)
        ents = [normalizar_entidade(p) for p in parts if p and len(p.strip()) > 2]
        
        dated = []
        for e in ents:
            val = self._get_or_seal(e, "date_year", "year")
            if val: dated.append((int(val), e))
        
        if len(dated) < 2: return None
        dated.sort()
        return " → ".join(f"{e} ({v})" for v, e in dated)

    def _handle_lookup(self, q, params):
        entity = params.get("entity", "").strip()
        entity = re.sub(r"^(?:a|o|as|os|um|uma|da|do|das|dos|de)\s+", "", entity, flags=re.IGNORECASE)
        sources = self.miner.mine_all(entity)
        if sources:
            text = sources[0][0]
            sents = re.split(r"(?<=[.!?])\s+", text)
            return " ".join(sents[:2])
        return None

    def _handle_date_of(self, q, params):
        entity = params.get("entity", "").strip()
        ql = q.lower()
        is_death = "morreu" in ql or "faleceu" in ql
        
        if not is_death:
            val = self._get_or_seal(entity, "date_year", "year")
            if val: return val
        
        sources = self.miner.mine_all(entity)
        for text, _ in sources:
            if is_death:
                m = re.search(r"(?:morreu|faleceu)\s+em\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})", text, re.IGNORECASE)
                if m: return m.group(1)
                m = re.search(r'\(([^)]*?\d{4})\s*(?:—|–|-)\s*([^)]*?\d{4})[^)]*\)', text)
                if m:
                    dates = re.findall(r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})', m.group(2))
                    if dates: return f"{dates[0][0]} de {dates[0][1]} de {dates[0][2]}"
        return None

    def _handle_creator(self, q, params):
        entity = params.get("entity", "").strip()
        entity = re.sub(r"^(?:a|o|as|os|um|uma|da|do|das|dos|de)\s+", "", entity, flags=re.IGNORECASE)
        sources = self.miner.mine_all(entity)
        for text, _ in sources:
            m = re.search(r"(?:gerenciad[ao]|criad[ao]|desenvolvidad[ao]|fundad[ao])\s+pel[ao]\s+([A-ZÁ-Ú][a-zá-ú]+(?:\s+[A-ZÁ-Ú][a-zá-ú]+)*)", text)
            if m: return m.group(1)
            m = re.search(r"é\s+um[a]?\s+(?:\w+\s+){0,4}d[ao]\s+([A-ZÁ-Ú][a-zá-ú]+(?:\s+[A-ZÁ-Ú][a-zá-ú]+)*)", text)
            if m: return m.group(1)
        return None

    def _handle_quantity(self, q, params):
        entity = params.get("entity", "").strip()
        entity = re.sub(r"^(?:quantos?|quantas?)\s+", "", entity, flags=re.IGNORECASE)
        entity = re.sub(r"\s+tem\s+.*$", "", entity, flags=re.IGNORECASE)
        entity = re.sub(r"^(?:a|o|as|os|um|uma|da|do|das|dos|de)\s+", "", entity, flags=re.IGNORECASE)
        
        # FIX 2: Reescrita semântica para "habitantes" -> "população"
        if "habitantes" in entity.lower():
            entity = re.sub(r"habitantes", "população", entity, flags=re.IGNORECASE).strip()
        
        if not entity: entity = params.get("entity", "")
        
        sources = self.miner.mine_all(entity)
        for text, _ in sources:
            # Procura um número com contexto
            m = re.search(r'(\d+(?:[.,]\d+)?)\s*(milh[õo]es|mil|bilh[õo]es)?\s*(?:de\s+)?(?:habitantes|pessoas|km²|metros)?', text, re.IGNORECASE)
            if m: return f"{m.group(1)} {m.group(2) or ''} — {text[:150]}..."
            m = re.search(r'\b(\d{1,3}(?:\.\d{3})+|\d{4,})\b', text)
            if m: return f"{m.group(1)} — {text[:150]}..."
        return None

    def _handle_where(self, q, params):
        return self._handle_lookup(q, params)

    def falar(self, question):
        t0 = time.perf_counter()
        intent, params = self._classify_adaptive(question)
        
        answer, mode = None, "?"
        try:
            if intent == "SAUDACAO": answer, mode = "Olá! Sou DLM_webs adaptativo.", "SAUDACAO"
            elif intent == "ORDENAR": answer = self._handle_sort(question); mode = "ORDENAR"
            elif intent == "DATA_MORTE": answer = self._handle_date_of(question, params); mode = "DATA-MORTE"
            elif intent == "DATA_EVENTO": answer = self._handle_date_of(question, params); mode = "DATA-EVENTO"
            elif intent == "CRIADOR_DE": answer = self._handle_creator(question, params); mode = "CRIADOR"
            elif intent == "QUANTIDADE": answer = self._handle_quantity(question, params); mode = "QUANTIDADE"
            elif intent == "LOCALIZACAO": answer = self._handle_where(question, params); mode = "LOCAL"
            elif intent in ("LOOKUP_PESSOA", "ESPECIFICAR", "LOOKUP"): answer = self._handle_lookup(question, params); mode = "LOOKUP"
            else: answer = self._handle_lookup(question, params); mode = "FALLBACK"
            
            if not answer: answer, mode = "Sem dados suficientes.", "SEM-FONTE"
            
            if answer and mode != "SEM-FONTE":
                self.gns.learn_online(question, intent)
        except Exception as e:
            answer, mode = f"Erro: {e}", "ERRO"
        
        self.db.commit()
        self.gns.save(os.path.join(STATE_DIR, MODEL_FILE))
        
        elapsed = (time.perf_counter() - t0) * 1000
        return f"[DLM_webs v7.2 | {elapsed:.0f}ms | {mode} | SIGN:{self.sign}]\n{answer}"

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8")
        except Exception: pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.reset:
        import shutil
        if os.path.isdir(STATE_DIR): shutil.rmtree(STATE_DIR)
        safe_print("[reset] Estado apagado.")
    
    dlm = DLM_webs(verbose=args.verbose)
    
    if args.query:
        print(dlm.falar(args.query))
    else:
        perguntas = [
            "quem criou a steam",
            "quando morreu silvio santos",
            "aonde fica belford roxo",
            "O Brasil tem quantos habitantes",
            "quem é o presidente lula",
            "independencia do brasil, abolicao da escravatura, proclamacao da republica",
        ]
        for q in perguntas:
            print(f"\n>>> {q}")
            print(dlm.falar(q))
