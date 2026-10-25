#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus DSpark 3.2 — Emergent Hebbian Reasoning
==================================================
Correções acumuladas (v3.0 → v3.1 → v3.2):
  v3.1: cosseno, L2, Spark normalizado, anti-repetição, loop-breaker
  v3.2: bigrama normalizado (peso 3.0), binding local, ativação com decaimento
        exponencial, exploração só no início, parser train com espaço E dois pontos
"""

import numpy as np
import math, random, time, os, pickle, tempfile, sys
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

# ==========================================
# ANDROID VOZ
# ==========================================
ANDROID_VOZ = False
MODO_VOZ = False
droid = None
try:
    import androidhelper; droid = androidhelper.Android(); ANDROID_VOZ = True
    print("✅ QPython detectado.")
except Exception:
    try:
        from sl4a import Android; droid = Android(); ANDROID_VOZ = True
        print("✅ SL4A detectado.")
    except Exception:
        print("ℹ️  Terminal (sem voz).")

def safe_input(p=""):
    if p: sys.stdout.write(p); sys.stdout.flush()
    return sys.stdin.readline().strip()

def falar(txt):
    if not ANDROID_VOZ or droid is None: return False
    try: droid.ttsSpeak(txt); droid.eventWait(3000); time.sleep(0.4); return True
    except Exception: return False

def ouvir(p="Ouvindo..."):
    if not ANDROID_VOZ or droid is None: return ""
    try:
        r = droid.recognizeSpeech(p)
        return r.result.strip() if r and r.result else ""
    except Exception: return ""

def vibrar(ms=80):
    if not ANDROID_VOZ or droid is None: return
    try: droid.vibrate(ms)
    except Exception: pass

# ==========================================
# UTILITÁRIOS
# ==========================================
def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def ajustar_pontuacao(s):
    if not s: return s
    if s[-1] in '.!?': return s
    if any(p in s.lower() for p in ['quem','quando','onde','por que','como','qual']):
        return s + '?'
    if any(p in s for p in ['!','nossa','caramba']):
        return s + '!'
    return s + '.'

# ==========================================================
# HEBBIAN GRAPH (v3.2)
# ==========================================================
class HebbianGraph:
    def __init__(self, n_concepts=48):
        self.tok_to_id: Dict[str, int] = {}
        self.id_to_tok: Dict[int, str] = {}
        self.n_concepts = n_concepts
        self.c_vec: Dict[int, np.ndarray] = {}
        self.CC = np.zeros((n_concepts, n_concepts))
        self.TT: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        self.act_to_id = {a: i for i, a in enumerate(
            ['statement','question','request','ack','greeting','farewell','exclamation'])}
        self.id_to_act = {i: a for a, i in self.act_to_id.items()}
        self.AC: Dict[int, np.ndarray] = {}

        self.pragma_to_id = {p: i for i, p in enumerate(
            ['formal','informal','technical','emotional'])}
        self.PC: Dict[int, np.ndarray] = {}

        # Plasticidade
        self.eta_tt = 0.02
        self.eta_c = 0.015
        self.eta_cc = 0.01
        self.eta_ac = 0.015
        self.eta_pc = 0.015
        self.lambda_decay = 0.001
        self.w_max = 5.0
        self.context_window = 4
        self.tau = 2.0

        self.current_act = None
        self.current_pragma = None
        self.session_updates = 0

    def ensure_token(self, tok: str) -> int:
        if tok in self.tok_to_id: return self.tok_to_id[tok]
        tid = len(self.tok_to_id)
        self.tok_to_id[tok] = tid
        self.id_to_tok[tid] = tok
        v = np.random.dirichlet(np.ones(self.n_concepts) * 0.3)
        n = float(np.linalg.norm(v))
        self.c_vec[tid] = v / n if n > 1e-8 else v
        return tid

    # ── PATCH v3.2: peso exponencial (último token domina) ──
    def compute_concept_activation(self, tids: List[int]) -> np.ndarray:
        if not tids:
            return np.ones(self.n_concepts) / self.n_concepts
        a = np.zeros(self.n_concepts)
        ws = [math.exp(-(len(tids) - 1 - i) * 0.35) for i in range(len(tids))]
        tw = sum(ws) + 1e-8
        for t, w in zip(tids, ws):
            if t in self.c_vec:
                a += (w / tw) * self.c_vec[t]
        a = a + 0.15 * (self.CC @ a)
        a = np.clip(a, 0, None)
        s = a.sum()
        return a / s if s > 0 else a

    # ── PATCH v3.2: bigrama normalizado com peso 3.0 ──
    def predict_scores(self, context_tids: List[int], a_C: np.ndarray,
                       banned: Optional[set] = None) -> Dict[int, float]:
        raw: Dict[int, float] = {}
        banned = banned or set()
        last6 = set(context_tids[-6:]) if context_tids else set()
        last15 = set(context_tids[-15:]) if context_tids else set()

        # bigrama normalizado (0-1)
        bigram = {}
        if context_tids:
            last = context_tids[-1]
            row = self.TT.get(last, {})
            total = sum(row.values()) + 1e-8
            for nxt, w in row.items():
                if nxt in banned or nxt in last6:
                    continue
                bigram[nxt] = w / total

        # conceito + bigrama
        for tid, cv in self.c_vec.items():
            if tid in banned or tid in last6:
                continue
            c = cosine(a_C, cv)
            base = c
            if tid in bigram:
                base += 3.0 * bigram[tid]      # bigrama domina
            if tid in last15:
                base *= 0.15
            if base > 0.15:
                raw[tid] = base

        if not raw and banned:
            return self.predict_scores(context_tids, a_C, None)
        return raw

    def sample(self, scores: Dict[int, float], temperature=0.85, top_k=8) -> Optional[int]:
        if not scores: return None
        items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        vals = np.array([v for _, v in items], dtype=float)
        vals = vals - vals.min() + 1e-8
        vals = vals ** (1.0 / max(temperature, 0.1))
        probs = vals / vals.sum()
        return items[int(np.random.choice(len(items), p=probs))][0]

    def penalize_bigram(self, a: int, b: int, factor: float = 0.5):
        if a in self.TT and b in self.TT[a]:
            self.TT[a][b] *= factor
            if self.TT[a][b] < 1e-3:
                del self.TT[a][b]

    # ── PATCH v3.2: binding LOCAL com vizinhos (não média global) ──
    def hebbian_update(self, fired_tids: List[int], reward: float = 0.0):
        if not fired_tids: return
        M = 1.0 + math.tanh(reward)
        self.session_updates += 1
        n = len(fired_tids)

        # T→T (STDP)
        eta_eff = self.eta_tt * M
        for i in range(n):
            a = fired_tids[i]
            for j in range(i + 1, min(i + 1 + self.context_window, n)):
                b = fired_tids[j]
                d = eta_eff * math.exp(-(j - i) / self.tau)
                self.TT[a][b] = min(self.w_max, self.TT[a][b] + d)
                self.TT[b][a] = max(0.0, self.TT[b][a] - d * 0.3)
            for dst in list(self.TT[a].keys()):
                self.TT[a][dst] *= (1 - self.lambda_decay)

        # T↔C: binding LOCAL com vizinhos
        if n >= 2 and reward > 0.02:
            eta_c = self.eta_c * M * min(1.0, max(0.0, reward))
            for i, t in enumerate(fired_tids):
                if t not in self.c_vec:
                    continue
                nbrs = []
                if i > 0: nbrs.append(fired_tids[i - 1])
                if i < n - 1: nbrs.append(fired_tids[i + 1])
                if not nbrs:
                    continue
                for nb in nbrs:
                    if nb in self.c_vec:
                        new = (1 - eta_c) * self.c_vec[t] + eta_c * self.c_vec[nb]
                        new = np.clip(new, 0, None)
                        norm = float(np.linalg.norm(new))
                        self.c_vec[t] = new / norm if norm > 1e-8 else new

        # C→C
        if n >= 2:
            vecs = [self.c_vec[t] for t in fired_tids if t in self.c_vec]
            eta_cc = self.eta_cc * M
            for i in range(len(vecs)):
                for j in range(i + 1, min(i + 1 + self.context_window, len(vecs))):
                    delta = eta_cc * math.exp(-(j - i) / self.tau) * np.outer(vecs[i], vecs[j])
                    self.CC += delta
            self.CC *= (1 - self.lambda_decay)
            np.clip(self.CC, 0, self.w_max, out=self.CC)

        # A→C
        if self.current_act is not None:
            prior = self.AC.get(self.current_act, np.ones(self.n_concepts) / self.n_concepts)
            c = self.compute_concept_activation(fired_tids)
            new = (1 - self.eta_ac * M) * prior + self.eta_ac * M * c
            s = new.sum()
            self.AC[self.current_act] = new / s if s > 0 else new

        # P→C
        if self.current_pragma is not None:
            prior = self.PC.get(self.current_pragma, np.ones(self.n_concepts) / self.n_concepts)
            c = self.compute_concept_activation(fired_tids)
            new = (1 - self.eta_pc * M) * prior + self.eta_pc * M * c
            s = new.sum()
            self.PC[self.current_pragma] = new / s if s > 0 else new

    def topology_stats(self) -> Dict:
        return {
            'tokens': len(self.tok_to_id),
            'concepts': self.n_concepts,
            'edges_TT': sum(len(v) for v in self.TT.values()),
            'edges_CC_active': int((self.CC > 0.01).sum()),
            'updates': self.session_updates,
        }

# ==========================================================
# INTRINSIC REWARD
# ==========================================================
class IntrinsicReward:
    def __init__(self):
        self.surprise_ema = 2.0
        self.coherence_ema = 0.5
        self.competence = defaultdict(lambda: 0.5)
        self.reward_history = deque(maxlen=50)
        self.session_competence_log = []

    def observe_input(self, tids, graph):
        if len(tids) < 2: return 0.0
        seq_concept = graph.compute_concept_activation(tids)
        cs = []
        for t in tids:
            if t in graph.c_vec:
                cs.append(cosine(seq_concept, graph.c_vec[t]))
        if not cs: return 0.0
        coherence = float(np.mean(cs))
        delta = coherence - self.coherence_ema
        self.coherence_ema = 0.9 * self.coherence_ema + 0.1 * coherence
        for c in range(len(seq_concept)):
            if seq_concept[c] > 0.1:
                lr = 0.05 * seq_concept[c]
                self.competence[c] = (1 - lr) * self.competence[c] + lr * coherence
        return float(np.clip(delta * 3.0, -1.5, 1.5))

    def compute_step_reward(self, coherence, surprise_norm, a_C):
        r = coherence - 0.25 * surprise_norm
        self.reward_history.append(r)
        for c in range(len(a_C)):
            if a_C[c] > 0.1:
                lr = 0.05 * a_C[c]
                self.competence[c] = (1 - lr) * self.competence[c] + lr * max(0.0, min(1.0, coherence))
        return float(r)

    def mean_competence(self):
        return float(np.mean(list(self.competence.values()))) if self.competence else 0.5

    def log_session(self):
        self.session_competence_log.append(self.mean_competence())
        if len(self.session_competence_log) > 20:
            self.session_competence_log.pop(0)

# ==========================================================
# DIALOGUE / PRAGMATICS / CURRICULUM
# ==========================================================
class DialogueTracker:
    def __init__(self):
        self.recent_acts = deque(maxlen=8)
        self.topic_vector = None
        self.turn_count = 0

    def classify_act(self, tokens, text):
        if not tokens: return 'statement'
        if text.strip().endswith('?') or any(w in tokens for w in
                ['quem','que','onde','quando','como','qual','porque']):
            return 'question'
        if any(w in tokens for w in ['favor','poderia','gostaria','quero','ajuda']):
            return 'request'
        if any(w in tokens for w in ['oi','olá','hello','eae','salve','bom','boa']):
            return 'greeting'
        if any(w in tokens for w in ['tchau','até','adeus','bye','falou']):
            return 'farewell'
        if any(w in tokens for w in ['ok','sim','não','certo','entendi','beleza']):
            return 'ack'
        if text.strip().endswith('!'):
            return 'exclamation'
        return 'statement'

    def observe_user(self, tokens, tids, graph):
        self.turn_count += 1
        act = self.classify_act(tokens, ' '.join(tokens))
        self.recent_acts.append(act)
        tv = graph.compute_concept_activation(tids)
        self.topic_vector = tv if self.topic_vector is None else 0.65*self.topic_vector + 0.35*tv

    def predict_act(self, graph):
        mapping = {'greeting':'greeting','question':'statement','request':'ack',
                   'farewell':'farewell','ack':'statement','exclamation':'statement',
                   'statement':'statement'}
        last = self.recent_acts[-1] if self.recent_acts else 'statement'
        return graph.act_to_id.get(mapping.get(last, 'statement'), 0)

class PragmaticModel:
    def __init__(self):
        self.formality = 0.5

    def update_from_user(self, tokens):
        formal = ['você','senhor','senhora','favor','poderia','gostaria','atenciosamente']
        informal = ['tu','cara','mano','oi','legal','né','vlw','fala']
        delta = 0.0
        for t in tokens:
            if any(m in t for m in formal): delta += 0.08
            if any(m in t for m in informal): delta -= 0.08
        self.formality = float(np.clip(0.9*self.formality + 0.1*(0.5+delta), 0.0, 1.0))

    def modulate(self, a_C, graph):
        key = 0 if self.formality > 0.6 else (1 if self.formality < 0.4 else None)
        if key is not None and key in graph.PC:
            a_C = a_C * (1.0 + 0.3 * graph.PC[key])
            s = a_C.sum()
            if s > 0: a_C = a_C / s
        return a_C

    def current_pragma_id(self):
        if self.formality > 0.6: return 0
        if self.formality < 0.4: return 1
        return None

class CurriculumController:
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic

    def lr_modifier(self, a_C):
        mod = np.ones(len(a_C))
        for c in range(len(a_C)):
            comp = self.intrinsic.competence[c]
            if comp < 0.4: mod[c] = 2.0
            elif comp > 0.7: mod[c] = 0.7
        return mod

    def curriculum_report(self):
        comps = list(self.intrinsic.competence.values())
        if not comps: return {'mean': 0.5, 'low': 0, 'high': 0, 'total': 0}
        return {
            'mean': float(np.mean(comps)),
            'low': int(sum(1 for c in comps if c < 0.4)),
            'high': int(sum(1 for c in comps if c > 0.7)),
            'total': len(comps),
        }

# ==========================================================
# DSPARK REASONER (v3.2)
# ==========================================================
class DSparkReasoner:
    def __init__(self, brain):
        self.brain = brain
        self.draft_k = 8
        self.coherence_w = 1.0
        self.surprise_w = 0.25
        self.step_rewards = []
        self.temperature = 0.75

    def _detect_period(self, seq: List[int]) -> Optional[int]:
        n = len(seq)
        if n < 6: return None
        for P in range(2, min(5, n // 2) + 1):
            if seq[-2*P:-P] == seq[-P:]:
                return P
        return None

    def _break_loop(self, context_tids, a_C, g):
        P = self._detect_period(context_tids)
        if P is None:
            return context_tids, set()
        loop = context_tids[-P:]
        for i in range(len(loop) - 1):
            g.penalize_bigram(loop[i], loop[i + 1], factor=0.25)
        g.penalize_bigram(loop[-1], loop[0], factor=0.25)
        banned = {loop[0]}
        new_ctx = list(context_tids[:-P])
        if len(new_ctx) < 3:
            new_ctx = list(context_tids[-3:])
        return new_ctx, banned

    def think(self, user_text: str, max_tokens=40) -> List[int]:
        brain = self.brain
        g = brain.graph

        # 1) parse + aprendizagem
        toks = brain.tokenize(user_text)
        tids = [g.ensure_token(t) for t in toks]
        if len(tids) >= 2:
            r_in = brain.intrinsic.observe_input(tids, g)
            g.current_act = None
            g.current_pragma = None
            g.hebbian_update(tids, reward=r_in)

        # 2) diálogo + pragmática
        brain.dialogue.observe_user(toks, tids, g)
        brain.pragmatics.update_from_user(toks)

        # 3) content selection
        a_C = g.compute_concept_activation(tids)
        a_C = brain.pragmatics.modulate(a_C, g)

        # 4) act priming
        g.current_act = brain.dialogue.predict_act(g)
        if g.current_act in g.AC:
            a_C = a_C * (1.0 + 0.25 * g.AC[g.current_act])
            s = a_C.sum()
            if s > 0: a_C = a_C / s
        g.current_pragma = brain.pragmatics.current_pragma_id()

        # 5) geração dual
        out_tids: List[int] = []
        context = list(tids)
        self.step_rewards = []
        generated_ngrams = set()

        for step in range(max_tokens):
            banned = set()
            if step > 0 and step % 3 == 0:
                context, banned = self._break_loop(context, a_C, g)
                if context:
                    a_C = g.compute_concept_activation(context[-(g.context_window + 1):])
                    a_C = brain.pragmatics.modulate(a_C, g)

            scores = g.predict_scores(context, a_C, banned)
            if not scores:
                break

            items = sorted(scores.items(), key=lambda x: -x[1])[:self.draft_k]
            if not items:
                break

            vals = np.array([s for _, s in items], dtype=float)
            vals = vals - vals.min() + 1e-8
            cand_probs = vals / vals.sum()

            best = None
            best_score = -1e9
            log_K = math.log(self.draft_k + 1)
            for (tid, raw_s), p in zip(items, cand_probs):
                cv = g.c_vec.get(tid)
                coherence = float(cosine(a_C, cv)) if cv is not None else 0.0
                surprise = -math.log(max(p, 1e-8))
                surprise_norm = surprise / log_K
                # raw já contém bigrama; score prioriza o raw
                score = raw_s + 0.3 * coherence - self.surprise_w * surprise_norm

                if len(context) >= 3 and tuple(context[-3:] + [tid]) in generated_ngrams:
                    score -= 0.6
                if tid in context[-8:]:
                    score -= 0.4

                if score > best_score:
                    best_score = score
                    best = (tid, p, coherence, surprise_norm)

            if best is None:
                break
            tid, p, coherence, surprise = best

            # exploração SÓ nos primeiros tokens
            if step <= 1 and random.random() < 0.15 and len(items) >= 2:
                alt = items[1][0] if items[0][0] == tid else items[0][0]
                if alt not in context[-6:]:
                    tid = alt
                    cv = g.c_vec.get(tid)
                    coherence = float(cosine(a_C, cv)) if cv is not None else 0.0
                    surprise = 0.5

            out_tids.append(tid)
            generated_ngrams.add(tuple(context[-3:] + [tid]))

            reward = brain.intrinsic.compute_step_reward(coherence, surprise, a_C)
            self.step_rewards.append(reward)
            window = context[-(g.context_window + 1):] + [tid]
            g.hebbian_update(window, reward=reward)

            context.append(tid)
            a_C = g.compute_concept_activation(context[-(g.context_window + 1):])
            a_C = brain.pragmatics.modulate(a_C, g)

            tok_str = g.id_to_tok[tid]
            if tok_str in {'.', '?', '!'} and step >= 2:
                break

        return out_tids

# ==========================================================
# QUINTIKUS BRAIN
# ==========================================================
class QuintikusBrain:
    def __init__(self, file="cerebro_dspark3.bin"):
        self.graph = HebbianGraph(n_concepts=48)
        self.intrinsic = IntrinsicReward()
        self.curriculum = CurriculumController(self.intrinsic)
        self.dialogue = DialogueTracker()
        self.pragmatics = PragmaticModel()
        self.reasoner = DSparkReasoner(self)
        self.file = file
        self.max_tokens = 40
        self.debug = False
        self.session_log = []

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        for p in '.,!?;:':
            text = text.replace(p, f' {p} ')
        return [t for t in text.split() if t]

    def learn_text(self, text: str, epochs: int = 1):
        for _ in range(epochs):
            raw = text.replace('!', '.').replace('?', '.').replace(';', '.')
            sentences = [s.strip() for s in raw.split('.') if s.strip()]
            for sent in sentences:
                toks = self.tokenize(sent)
                if len(toks) < 2: continue
                toks = toks + ['.']   # períodos preservados
                tids = [self.graph.ensure_token(t) for t in toks]
                r = self.intrinsic.observe_input(tids, self.graph)
                self.graph.hebbian_update(tids, reward=r)
                for i in range(len(tids) - 3):
                    self.graph.hebbian_update(tids[i:i+4], reward=r * 0.5)

    def think(self, user_text: str) -> str:
        if not self.graph.tok_to_id:
            return "Preciso de mais dados. Use train:arquivo.txt"
        out_tids = self.reasoner.think(user_text, max_tokens=self.max_tokens)
        if not out_tids:
            return "Não consegui gerar resposta."
        txt = ' '.join(self.graph.id_to_tok[t] for t in out_tids)
        txt = (txt.replace(' ,', ',').replace(' .', '.')
                  .replace(' !', '!').replace(' ?', '?')
                  .replace(' ;', ';').replace(' :', ':'))
        txt = ' '.join(txt.split())
        if txt and txt[0].islower():
            txt = txt[0].upper() + txt[1:]
        return ajustar_pontuacao(txt)

    # ---------- persistência ----------
    def save(self) -> bool:
        try:
            data = {
                'tok_to_id': self.graph.tok_to_id,
                'id_to_tok': self.graph.id_to_tok,
                'c_vec': {k: v for k, v in self.graph.c_vec.items()},
                'TT': {k: dict(v) for k, v in self.graph.TT.items()},
                'CC': self.graph.CC,
                'AC': dict(self.graph.AC),
                'PC': dict(self.graph.PC),
                'n_concepts': self.graph.n_concepts,
                'competence': dict(self.intrinsic.competence),
                'coherence_ema': self.intrinsic.coherence_ema,
                'surprise_ema': self.intrinsic.surprise_ema,
                'recent_acts': list(self.dialogue.recent_acts),
                'topic_vector': self.dialogue.topic_vector,
                'turn_count': self.dialogue.turn_count,
                'formality': self.pragmatics.formality,
                'session_log': self.session_log,
                'max_tokens': self.max_tokens,
            }
            d = os.path.dirname(os.path.abspath(self.file)) or '.'
            with tempfile.NamedTemporaryFile(mode='wb', dir=d, delete=False, suffix='.tmp') as tmp:
                pickle.dump(data, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.flush(); os.fsync(tmp.fileno())
                path = tmp.name
            os.replace(path, self.file)
            return True
        except Exception as e:
            print(f"⚠️ Save: {e}")
            return False

    def load(self) -> bool:
        if not os.path.exists(self.file): return False
        try:
            with open(self.file, 'rb') as f:
                data = pickle.load(f)
            g = self.graph
            g.tok_to_id = data['tok_to_id']
            g.id_to_tok = data['id_to_tok']
            g.c_vec = {k: np.array(v) for k, v in data['c_vec'].items()}
            g.TT = defaultdict(lambda: defaultdict(float))
            for k, v in data['TT'].items():
                g.TT[k] = defaultdict(float, v)
            g.CC = data['CC']
            g.AC = {k: np.array(v) for k, v in data['AC'].items()}
            g.PC = {k: np.array(v) for k, v in data['PC'].items()}
            g.n_concepts = data['n_concepts']
            self.intrinsic.competence = defaultdict(lambda: 0.5, data['competence'])
            self.intrinsic.coherence_ema = data['coherence_ema']
            self.intrinsic.surprise_ema = data['surprise_ema']
            self.dialogue.recent_acts = deque(data['recent_acts'], maxlen=8)
            self.dialogue.topic_vector = data['topic_vector']
            self.dialogue.turn_count = data['turn_count']
            self.pragmatics.formality = data['formality']
            self.session_log = data['session_log']
            self.max_tokens = data['max_tokens']
            return True
        except Exception as e:
            print(f"⚠️ Load: {e}")
            return False

    # ---------- relatórios ----------
    def report(self):
        topo = self.graph.topology_stats()
        cur = self.curriculum.curriculum_report()
        mean_comp = self.intrinsic.mean_competence()
        ema_coh = self.intrinsic.coherence_ema
        if len(self.session_log) >= 2:
            trend = self.session_log[-1] - self.session_log[-2]
            trend_str = f"{'↑' if trend > 0.005 else ('↓' if trend < -0.005 else '→')} {trend:+.3f}"
        else:
            trend_str = "(sem baseline)"
        print("\n" + "═" * 50)
        print("📊 RELATÓRIO DE EMERGÊNCIA — Quintikus DSpark 3.2")
        print("═" * 50)
        print(f"  Topologia:")
        print(f"    Tokens: {topo['tokens']:>5}  Conceitos: {topo['concepts']}")
        print(f"    Arestas T→T: {topo['edges_TT']:>5}  C→C ativas: {topo['edges_CC_active']}")
        print(f"    Updates Hebbianos (sessão): {topo['updates']}")
        print(f"  Currículo:")
        print(f"    Competência média: {cur['mean']:.3f}  (low={cur['low']} high={cur['high']})")
        print(f"  Intrínseco:")
        print(f"    Coerência EMA: {ema_coh:.3f}  →  tendência: {trend_str}")
        print(f"  Diálogo:")
        print(f"    Turnos: {self.dialogue.turn_count}  Ato recente: "
              f"{list(self.dialogue.recent_acts)[-1] if self.dialogue.recent_acts else '—'}")
        print(f"    Formalidade: {self.pragmatics.formality:.2f}")
        print("═" * 50)

    def end_session(self):
        self.intrinsic.log_session()
        self.save()

# ==========================================================
# DEMO
# ==========================================================
DEMO_CORPUS = """
o gato come peixe. o cão come osso. o gato dorme no sofá.
o cão late no quintal. o gato mia de noite. o cão corre no parque.
animais domésticos vivem com humanos. gatos e cães são animais.
a inteligência artificial aprende com dados. redes neurais processam padrões.
aprendizagem profunda usa muitas camadas. modelos aprendem representações.
o cérebro humano tem muitos neurônios. sinapses conectam neurônios.
plasticidade hebbiana fortalece conexões. neurônios que disparam juntos se conectam.
a curiosidade motiva a exploração. incerteza reduzida é recompensa intrínseca.
conhecimento vive nas conexões. o grafo codifica a experiência.
linguagem expressa o pensamento. composição organiza o conteúdo.
pragmática adapta ao interlocutor. o diálogo constrói significado.
um gato é um animal. um cão é um animal. peixe é um animal.
comida é necessária para viver. água é essencial para a vida.
o sol nasce no leste. a lua aparece à noite. estrelas brilham no céu.
"""

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("🧬 Quintikus DSpark 3.2 — Emergent Hebbian Reasoning")
    print("   Sem backprop · Grafo Hebbiano · Curiosidade · Dual-process")
    print("=" * 62)

    brain = QuintikusBrain()
    loaded = brain.load()
    if loaded:
        print(f"🧠 Cérebro carregado. Tokens={len(brain.graph.tok_to_id)} | "
              f"Arestas_TT={sum(len(v) for v in brain.graph.TT.values())}")
        print(f"   Competência média: {brain.intrinsic.mean_competence():.3f}")
    else:
        print("🆕 Cérebro novo. Treinando demo embutido...")
        brain.learn_text(DEMO_CORPUS, epochs=3)
        print(f"✅ Demo treinado. Tokens={len(brain.graph.tok_to_id)}")
        brain.save()

    if ANDROID_VOZ:
        time.sleep(0.4)
        falar("Quintikus DSpark três ponto dois está online.")
        print("\n🎤 [V] Microfone contínuo  [T] Teclado")
        modo = safe_input("   Modo > ").lower()
        MODO_VOZ = (modo == 'v')

    if not MODO_VOZ:
        print("\n💬 Comandos:")
        print("   stats              → relatório de emergência")
        print("   train:file         → aprender arquivo (ou 'train file')")
        print("   tokens:N           → tamanho máximo de resposta")
        print("   debug:on/off       → ver sinais internos")
        print("   save | sair")

    while True:
        try:
            if MODO_VOZ and ANDROID_VOZ:
                vibrar(30)
                entrada = ouvir("Ouvindo...")
                if not entrada or len(entrada.strip()) < 1:
                    time.sleep(0.3); continue
                print(f"👤: {entrada}")
                low = entrada.lower()
                if any(p in low for p in ["teclado","digitar"]):
                    falar("Modo teclado."); MODO_VOZ = False; continue
                if low in ["sair","encerrar"]:
                    falar("Salvando memória."); brain.end_session(); break
                if low.startswith("stats"):
                    brain.report(); falar("Relatório exibido na tela.")
                    continue
                resposta = brain.think(entrada)
                print(f"🧠: {resposta}")
                vibrar(20); falar(resposta)
                continue

            entrada = safe_input("usr:")
            if not entrada: continue
            low = entrada.lower()

            if low == "sair":
                print("⚙️ Encerrando sessão (emergência registrada)...")
                brain.end_session(); brain.report(); print("💤 Até mais!")
                break

            if low == "stats":
                brain.report(); continue

            if low == "save":
                print("💾 Salvo!" if brain.save() else "❌ Falha.")
                continue

            if low.startswith("tokens:"):
                try: brain.max_tokens = max(5, min(120, int(entrada.split(":",1)[1])))
                except Exception: print("❌ Use: tokens:40")
                else: print(f"✅ max_tokens = {brain.max_tokens}")
                continue

            if low.startswith("debug:"):
                brain.debug = entrada.split(":",1)[1].strip() == "on"
                print(f"✅ debug = {brain.debug}"); continue

            # ── PATCH v3.2: aceita "train:file", "train file", "treinar file" ──
            if low.startswith("train") or low.startswith("treinar"):
                parts = entrada.split(":", 1) if ":" in entrada else entrada.split(None, 1)
                if len(parts) >= 2:
                    arq = parts[1].strip()
                    if os.path.exists(arq):
                        with open(arq, "r", encoding="utf-8") as f:
                            txt = f.read()
                        t0 = time.time()
                        brain.learn_text(txt, epochs=2)
                        print(f"✅ Aprendido em {time.time()-t0:.2f}s. "
                              f"Tokens={len(brain.graph.tok_to_id)}")
                        continue
                    else:
                        print(f"❌ '{arq}' não encontrado.")
                        continue
                else:
                    print("❌ Use: train:arquivo.txt  ou  train arquivo.txt")
                    continue

            resposta = brain.think(entrada)
            print(f"🧠: {resposta}")
            if brain.debug:
                r = brain.reasoner
                if r.step_rewards:
                    print(f"   [DEBUG] reward_médio={np.mean(r.step_rewards):+.3f} "
                          f"coher_ema={brain.intrinsic.coherence_ema:.3f} "
                          f"comp={brain.intrinsic.mean_competence():.3f}")

        except KeyboardInterrupt:
            print("\n⚙️ Encerrando...")
            brain.end_session(); brain.report(); break
        except Exception as e:
            print(f"❌ Erro: {e}")
