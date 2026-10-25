#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus DSpark 3.0 — Emergent Hebbian Reasoning
=================================================
Princípios (Biologically-Inspired Emergent Reasoning):
  1. SEM gradiente, SEM backprop, SEM loss. Apenas plasticidade LOCAL.
  2. Regra Hebbiana modulada:  Δw_ij = η·pre_i·post_j·M − λ·w_ij
     M = neuromodulador intrínseco (curiosity/competência).
  3. Conhecimento vive na TOPOLOGIA do grafo (T→T, T↔C, C→C, A→C, P→C).
  4. Conceitos EMERGEM da co-ocorrência (sem rótulos).
  5. Generalização mediada por conceito (gato↔cão via C:animal).
  6. Raciocínio DUAL-PROCESS (DSpark):
       - Draft (Sistema 1): paralelo, associativo, dirigido por conceito.
       - Spark (Sistema 2): sequencial, coerente, guiado por bigrama.
  7. Recompensa INTRÍNSECA: coerência − surpresa (não é recompensa de output).
  8. Currículo: conceitos de baixa competência recebem mais atenção.
  9. Auto-avaliação por SESSÃO (emergência ao longo do tempo).
"""

import numpy as np
import math, random, time, os, pickle, json, tempfile, sys
from collections import defaultdict, deque, Counter
from typing import List, Dict, Tuple, Optional

# ==========================================
# ANDROID VOZ (preservado, compacto)
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
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a, b) / (na * nb))

def softmax(x, t=1.0):
    x = x / max(t, 1e-6)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def ajustar_pontuacao(s):
    if not s: return s
    if s[-1] in '.!?': return s
    if any(p in s.lower() for p in ['quem','quando','onde','por que','como','qual']):
        return s + '?'
    if any(p in s for p in ['!','nossa','caramba']):
        return s + '!'
    return s + '.'

# ==========================================================
# HEBBIAN GRAPH — núcleo de conhecimento (sem backprop)
# ==========================================================
class HebbianGraph:
    """
    Nós: T (token), C (conceito emergente), A (ato), P (pragma)
    Arestas:
      T→T  bigrama de superfície
      T↔C  binding token↔conceito (c_vec é o soft assignment)
      C→C  transição conceitual (= composição/sintaxe emergente)
      A→C  ato dialoga com conceitos (content selection)
      P→C  pragma modula conceitos
    Hebbian rule:  Δw_ij = η·a_i·a_j·M − λ·w_ij
    """

    def __init__(self, n_concepts=48):
        self.tok_to_id: Dict[str, int] = {}
        self.id_to_tok: Dict[int, str] = {}
        self.n_concepts = n_concepts

        self.c_vec: Dict[int, np.ndarray] = {}          # tid → soft concept vector
        self.CC = np.zeros((n_concepts, n_concepts))    # concept transitions
        self.TT: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        self.act_to_id = {a: i for i, a in enumerate(
            ['statement', 'question', 'request', 'ack', 'greeting', 'farewell', 'exclamation'])}
        self.id_to_act = {i: a for a, i in self.act_to_id.items()}
        self.AC: Dict[int, np.ndarray] = {}   # act_id → concept prior

        self.pragma_to_id = {p: i for i, p in enumerate(['formal', 'informal', 'technical', 'emotional'])}
        self.PC: Dict[int, np.ndarray] = {}   # pragma_id → concept prior

        # Plasticity params
        self.eta_tt = 0.03
        self.eta_c = 0.04
        self.eta_cc = 0.02
        self.eta_ac = 0.02
        self.eta_pc = 0.02
        self.lambda_decay = 0.0005
        self.w_max = 8.0
        self.context_window = 5
        self.tau = 2.0
        self.bigram_weight = 1.5

        # Contexto corrente (definido pelo reasoner antes do update)
        self.current_act: Optional[int] = None
        self.current_pragma: Optional[int] = None

        self.session_updates = 0

    # ---------- token management ----------
    def ensure_token(self, tok: str) -> int:
        if tok in self.tok_to_id: return self.tok_to_id[tok]
        tid = len(self.tok_to_id)
        self.tok_to_id[tok] = tid
        self.id_to_tok[tid] = tok
        # inicialização Hebbiana: vetor Dirichlet esparso (poucos conceitos ativos)
        cv = np.random.dirichlet(np.ones(self.n_concepts) * 0.3)
        self.c_vec[tid] = cv
        return tid

    # ---------- ativação conceitual ----------
    def compute_concept_activation(self, tids: List[int]) -> np.ndarray:
        if not tids:
            return np.ones(self.n_concepts) / self.n_concepts
        a = np.zeros(self.n_concepts)
        n = 0
        for t in tids:
            if t in self.c_vec:
                a += self.c_vec[t]; n += 1
        if n == 0:
            return np.ones(self.n_concepts) / self.n_concepts
        a /= n
        # espalhamento por transições conceituais (composição emergente)
        a = a + 0.35 * (self.CC @ a)
        a = np.clip(a, 0, None)
        s = a.sum()
        return a / s if s > 0 else a

    # ---------- predição ----------
    def predict_scores(self, context_tids: List[int], a_C: np.ndarray) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        # similaridade conceito↔token (generalização mediada)
        for tid, cv in self.c_vec.items():
            s = float(a_C @ cv)
            if s > 0.001:
                scores[tid] = s
        # bigrama de superfície
        if context_tids:
            last = context_tids[-1]
            for nxt, w in self.TT.get(last, {}).items():
                scores[nxt] = scores.get(nxt, 0.0) + self.bigram_weight * w
        # evita repetição imediata
        if context_tids:
            for t in context_tids[-2:]:
                scores.pop(t, None)
        return scores

    def sample(self, scores: Dict[int, float], temperature=0.85, top_k=8) -> Optional[int]:
        if not scores: return None
        items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        vals = np.array([v for _, v in items])
        vals = np.exp(vals / max(temperature, 0.1))
        probs = vals / vals.sum()
        idx = int(np.random.choice(len(items), p=probs))
        return items[idx][0]

    # ---------- REGRA HEBBIANA LOCAL ----------
    def hebbian_update(self, fired_tids: List[int], reward: float = 0.0):
        """
        Δw_ij = η·a_i·a_j·M − λ·w_ij
        M = 1 + tanh(reward) ∈ [0, 2]  — neuromodulador intrínseco
        """
        if len(fired_tids) < 1: return
        M = 1.0 + math.tanh(reward)
        self.session_updates += 1

        # ── 1. T→T (com decaimento posicional tipo STDP) ──
        eta_eff = self.eta_tt * M
        n = len(fired_tids)
        for i in range(n):
            a = fired_tids[i]
            for j in range(i + 1, min(i + 1 + self.context_window, n)):
                b = fired_tids[j]
                dt = j - i
                d = eta_eff * math.exp(-dt / self.tau)
                self.TT[a][b] = min(self.w_max, self.TT[a][b] + d)
                # anti-Hebbian simétrico (mais fraco)
                self.TT[b][a] = max(0.0, self.TT[b][a] - d * 0.25)
            # decay da linha
            for dst in list(self.TT[a].keys()):
                self.TT[a][dst] *= (1 - self.lambda_decay)

        # ── 2. T↔C: cada token se aproxima do conceito do contexto ──
        ctx_concept = self.compute_concept_activation(fired_tids[:-1]) if n > 1 \
            else self.compute_concept_activation(fired_tids)
        eta_c = self.eta_c * M
        for t in fired_tids:
            if t in self.c_vec:
                new = (1 - eta_c) * self.c_vec[t] + eta_c * ctx_concept
                new = np.clip(new, 0, None)
                s = new.sum()
                self.c_vec[t] = new / s if s > 0 else new

        # ── 3. C→C (composição emergente) ──
        if n >= 2:
            vecs = [self.c_vec[t] for t in fired_tids if t in self.c_vec]
            eta_cc = self.eta_cc * M
            for i in range(len(vecs)):
                for j in range(i + 1, min(i + 1 + self.context_window, len(vecs))):
                    delta = eta_cc * math.exp(-(j - i) / self.tau) * np.outer(vecs[i], vecs[j])
                    self.CC += delta
            self.CC *= (1 - self.lambda_decay)
            np.clip(self.CC, 0, self.w_max, out=self.CC)

        # ── 4. A→C (ato dialoga com conceito) ──
        if self.current_act is not None:
            prior = self.AC.get(self.current_act, np.ones(self.n_concepts) / self.n_concepts)
            new = (1 - self.eta_ac * M) * prior + self.eta_ac * M * ctx_concept
            s = new.sum()
            self.AC[self.current_act] = new / s if s > 0 else new

        # ── 5. P→C (pragma modula conceito) ──
        if self.current_pragma is not None:
            prior = self.PC.get(self.current_pragma, np.ones(self.n_concepts) / self.n_concepts)
            new = (1 - self.eta_pc * M) * prior + self.eta_pc * M * ctx_concept
            s = new.sum()
            self.PC[self.current_pragma] = new / s if s > 0 else new

    # ---------- relatório ----------
    def topology_stats(self) -> Dict:
        edges_tt = sum(len(v) for v in self.TT.values())
        edges_cc = int((self.CC > 0.01).sum())
        return {
            'tokens': len(self.tok_to_id),
            'concepts': self.n_concepts,
            'edges_TT': edges_tt,
            'edges_CC_active': edges_cc,
            'updates': self.session_updates,
        }

# ==========================================================
# INTRINSIC REWARD — curiosity (redução de incerteza)
# ==========================================================
class IntrinsicReward:
    """Recompensa = coerência (fit ao conceito) − surpresa (incerteza)."""
    def __init__(self):
        self.surprise_ema = 2.0
        self.coherence_ema = 0.5
        self.competence = defaultdict(lambda: 0.5)   # por conceito
        self.reward_history = deque(maxlen=50)
        self.session_competence_log = []

    def observe_input(self, tids, graph) -> float:
        """Aprender com input: reward = coerência da sequência."""
        if len(tids) < 2: return 0.0
        seq_concept = graph.compute_concept_activation(tids)
        total = 0.0; n = 0
        for t in tids:
            if t in graph.c_vec:
                total += float(seq_concept @ graph.c_vec[t]); n += 1
        if n == 0: return 0.0
        coherence = total / n
        delta = coherence - self.coherence_ema
        self.coherence_ema = 0.9 * self.coherence_ema + 0.1 * coherence
        # update competência por conceito
        for c in range(len(seq_concept)):
            if seq_concept[c] > 0.1:
                lr = 0.03 * seq_concept[c]
                self.competence[c] = (1 - lr) * self.competence[c] + lr * coherence
        return float(delta * 2.0)

    def compute_step_reward(self, coherence, surprise, a_C) -> float:
        """Recompensa passo-a-passo: coerência − λ·surpresa."""
        r = coherence - 0.35 * surprise
        self.reward_history.append(r)
        # atualiza competência dos conceitos ativos
        for c in range(len(a_C)):
            if a_C[c] > 0.1:
                lr = 0.05 * a_C[c]
                self.competence[c] = (1 - lr) * self.competence[c] + lr * max(0.0, min(1.0, coherence))
        return float(r)

    def mean_competence(self) -> float:
        if not self.competence: return 0.5
        return float(np.mean(list(self.competence.values())))

    def log_session(self):
        self.session_competence_log.append(self.mean_competence())
        if len(self.session_competence_log) > 20:
            self.session_competence_log.pop(0)

# ==========================================================
# DIALOGUE TRACKER — state + act classification
# ==========================================================
class DialogueTracker:
    def __init__(self):
        self.recent_acts = deque(maxlen=8)
        self.topic_vector: Optional[np.ndarray] = None
        self.turn_count = 0

    def classify_act(self, tokens: List[str], text: str) -> str:
        if not tokens: return 'statement'
        low = text.lower()
        if text.strip().endswith('?') or any(w in tokens for w in
                ['quem','que','onde','quando','como','qual','por que','porque','?']):
            return 'question'
        if any(w in tokens for w in ['por favor','poderia','gostaria','quero','ajuda','faz','faça']):
            return 'request'
        if any(w in tokens for w in ['oi','olá','hello','eae','e aí','salve','bom dia','boa tarde','boa noite']):
            return 'greeting'
        if any(w in tokens for w in ['tchau','até','adeus','bye','falou','adeus']):
            return 'farewell'
        if any(w in tokens for w in ['ok','sim','não','certo','entendi','beleza','tá','ta']):
            return 'ack'
        if text.strip().endswith('!'):
            return 'exclamation'
        return 'statement'

    def observe_user(self, tokens, tids, graph):
        self.turn_count += 1
        act = self.classify_act(tokens, ' '.join(tokens))
        self.recent_acts.append(act)
        tv = graph.compute_concept_activation(tids)
        self.topic_vector = tv if self.topic_vector is None else 0.65 * self.topic_vector + 0.35 * tv

    def predict_act(self, graph) -> int:
        mapping = {'greeting':'greeting','question':'statement','request':'ack',
                   'farewell':'farewell','ack':'statement','exclamation':'statement',
                   'statement':'statement'}
        last = self.recent_acts[-1] if self.recent_acts else 'statement'
        return graph.act_to_id.get(mapping.get(last, 'statement'), 0)

# ==========================================================
# PRAGMATIC MODEL — adapta ao interlocutor
# ==========================================================
class PragmaticModel:
    def __init__(self):
        self.formality = 0.5

    def update_from_user(self, tokens):
        formal = ['você','senhor','senhora','por favor','poderia','gostaria','atenciosamente']
        informal = ['tu','cara','mano','oi','legal','né','vlw','fala','po']
        delta = 0.0
        for t in tokens:
            if any(m in t for m in formal): delta += 0.08
            if any(m in t for m in informal): delta -= 0.08
        self.formality = float(np.clip(0.9 * self.formality + 0.1 * (0.5 + delta), 0.0, 1.0))

    def modulate(self, a_C, graph) -> np.ndarray:
        # pragma prior → concept
        key = 0 if self.formality > 0.6 else (1 if self.formality < 0.4 else None)
        if key is not None and key in graph.PC:
            prior = graph.PC[key]
            a_C = a_C * (1.0 + 0.3 * prior)
            s = a_C.sum()
            if s > 0: a_C = a_C / s
        return a_C

    def current_pragma_id(self):
        if self.formality > 0.6: return 0
        if self.formality < 0.4: return 1
        return None

# ==========================================================
# CURRICULUM — competência por conceito → atenção
# ==========================================================
class CurriculumController:
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic

    def lr_modifier(self, a_C) -> np.ndarray:
        mod = np.ones(len(a_C))
        for c in range(len(a_C)):
            comp = self.intrinsic.competence[c]
            if comp < 0.4: mod[c] = 2.0     # precisa aprender
            elif comp > 0.7: mod[c] = 0.7   # já domina
        return mod

    def curriculum_report(self) -> Dict:
        comps = list(self.intrinsic.competence.values())
        if not comps: return {'mean': 0.5, 'low': 0, 'high': 0}
        return {
            'mean': float(np.mean(comps)),
            'low': int(sum(1 for c in comps if c < 0.4)),
            'high': int(sum(1 for c in comps if c > 0.7)),
            'total': len(comps),
        }

# ==========================================================
# DSPARK REASONER — Draft (S1) + Spark (S2) + Hebbian
# ==========================================================
class DSparkReasoner:
    """
    Draft (Sistema 1): amostragem paralela dirigida por conceito.
    Spark (Sistema 2): avaliação sequencial coerência vs surpresa.
    Cada passo gera um HEBBIAN UPDATE modulado por recompensa intrínseca.
    """
    def __init__(self, brain):
        self.brain = brain
        self.draft_k = 8
        self.coherence_w = 1.0
        self.surprise_w = 0.35
        self.step_rewards = []

    def think(self, user_text: str, max_tokens=40) -> List[int]:
        brain = self.brain; g = brain.graph

        # ── 1. Parse + aprender com input ──
        toks = brain.tokenize(user_text)
        tids = [g.ensure_token(t) for t in toks]
        if len(tids) >= 2:
            r_in = brain.intrinsic.observe_input(tids, g)
            g.current_act = None; g.current_pragma = None
            g.hebbian_update(tids, reward=r_in)

        # ── 2. Estado de diálogo + pragmática ──
        brain.dialogue.observe_user(toks, tids, g)
        brain.pragmatics.update_from_user(toks)

        # ── 3. Content selection: ativação conceitual ──
        a_C = g.compute_concept_activation(tids)
        a_C = brain.pragmatics.modulate(a_C, g)

        # ── 4. Act priming (composição guiada por ato) ──
        g.current_act = brain.dialogue.predict_act(g)
        if g.current_act in g.AC:
            a_C = a_C * (1.0 + 0.25 * g.AC[g.current_act])
            s = a_C.sum()
            if s > 0: a_C = a_C / s
        g.current_pragma = brain.pragmatics.current_pragma_id()

        # ── 5. Loop Draft + Spark + Hebbian ──
        out_tids: List[int] = []
        context = list(tids)
        self.step_rewards = []

        for step in range(max_tokens):
            scores = g.predict_scores(context, a_C)
            if not scores:
                break

            # ── DRAFT ──
            items = sorted(scores.items(), key=lambda x: -x[1])[:self.draft_k]
            total = sum(s for _, s in items) + 1e-8
            candidates = [(tid, s / total) for tid, s in items]

            # ── SPARK ──
            best = None; best_score = -1e9
            for tid, p in candidates:
                cv = g.c_vec.get(tid)
                coherence = float(a_C @ cv) if cv is not None else 0.0
                surprise = -math.log(p + 1e-12)
                score = self.coherence_w * coherence - self.surprise_w * surprise
                if score > best_score:
                    best_score = score
                    best = (tid, p, coherence, surprise)
            if best is None: break
            tid, p, coherence, surprise = best
            out_tids.append(tid)

            # ── INTRINSIC REWARD ──
            reward = brain.intrinsic.compute_step_reward(coherence, surprise, a_C)
            self.step_rewards.append(reward)

            # ── HEBBIAN UPDATE (local, modulado por reward) ──
            window = context[-(g.context_window + 1):] + [tid]
            g.hebbian_update(window, reward=reward)

            # ── Update contexto/ativação ──
            context.append(tid)
            a_C = g.compute_concept_activation(context[-(g.context_window + 1):])
            a_C = brain.pragmatics.modulate(a_C, g)

            # parada em fim de sentença
            tok_str = g.id_to_tok[tid]
            if tok_str in {'.', '?', '!'} and step >= 3:
                break

        return out_tids

# ==========================================================
# QUINTIKUS BRAIN — orquestrador
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
            sentences = [s.strip() for s in
                         text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            for sent in sentences:
                toks = self.tokenize(sent)
                if len(toks) < 2: continue
                tids = [self.graph.ensure_token(t) for t in toks]
                r = self.intrinsic.observe_input(tids, self.graph)
                self.graph.hebbian_update(tids, reward=r)
                # sub-janelas (mais amostras para o grafo)
                for i in range(len(tids) - 3):
                    self.graph.hebbian_update(tids[i:i+4], reward=r * 0.5)

    def think(self, user_text: str) -> str:
        if not self.graph.tok_to_id:
            return "Preciso de mais dados. Use train:arquivo.txt"
        out_tids = self.reasoner.think(user_text, max_tokens=self.max_tokens)
        if not out_tids:
            return "Não consegui gerar resposta."
        txt = ' '.join(self.graph.id_to_tok[t] for t in out_tids)
        txt = txt.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        txt = ' '.join(txt.split())
        if txt and txt[0].islower(): txt = txt[0].upper() + txt[1:]
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
                tmp.flush(); os.fsync(tmp.fileno()); path = tmp.name
            os.replace(path, self.file)
            return True
        except Exception as e:
            print(f"⚠️ Save: {e}"); return False

    def load(self) -> bool:
        if not os.path.exists(self.file): return False
        try:
            with open(self.file, 'rb') as f: data = pickle.load(f)
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
            print(f"⚠️ Load: {e}"); return False

    # ---------- relatórios ----------
    def report(self):
        topo = self.graph.topology_stats()
        cur = self.curriculum.curriculum_report()
        mean_comp = self.intrinsic.mean_competence()
        ema_coh = self.intrinsic.coherence_ema
        # emergência: delta entre últimas sessões
        if len(self.session_log) >= 2:
            trend = self.session_log[-1] - self.session_log[-2]
            trend_str = f"{'↑' if trend > 0.005 else ('↓' if trend < -0.005 else '→')} {trend:+.3f}"
        else:
            trend_str = "(sem baseline)"
        print("\n" + "═" * 50)
        print("📊 RELATÓRIO DE EMERGÊNCIA — Quintikus DSpark 3.0")
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
# DEMO — dataset embutido (PT/EN) para auto-avaliação
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
    print("🧬 Quintikus DSpark 3.0 — Emergent Hebbian Reasoning")
    print("   Sem backprop · Grafo Hebbiano · Curiosidade intrínseca · Dual-process")
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
        falar("Quintikus DSpark três está online.")
        print("\n🎤 [V] Microfone contínuo  [T] Teclado")
        modo = safe_input("   Modo > ").lower()
        MODO_VOZ = (modo == 'v')

    if not MODO_VOZ:
        print("\n💬 Comandos:")
        print("   stats         → relatório de emergência")
        print("   train:file    → aprender arquivo")
        print("   tokens:N      → ajustar tamanho de resposta")
        print("   debug:on/off  → ver sinais internos")
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
                brain.end_session()
                brain.report()
                print("💤 Até mais!")
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

            if low.startswith("train:"):
                arq = entrada.split(":",1)[1].strip()
                if os.path.exists(arq):
                    with open(arq, "r", encoding="utf-8") as f: txt = f.read()
                    t0 = time.time()
                    brain.learn_text(txt, epochs=2)
                    print(f"✅ Aprendido em {time.time()-t0:.2f}s. "
                          f"Tokens={len(brain.graph.tok_to_id)}")
                else:
                    print(f"❌ '{arq}' não encontrado.")
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
            brain.end_session()
            brain.report()
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
