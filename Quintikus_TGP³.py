import numpy as np
import json
import sqlite3
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ==========================================
# 1. CAMADA LÍMBICA: Percepção e Estado Emocional
# ==========================================

@dataclass
class Percepcao:
    texto: str
    silencio_segundos: float
    novidade: float
    repeticao: float
    carga_afetiva: float
    mudanca_topico: bool
    dominio: str
    intencao: str
    timestamp: float

    @classmethod
    def capturar(cls, texto: str, texto_anterior: str = "", silencio: float = 0.0,
                 dominio: str = "geral", intencao: str = "reflexao") -> 'Percepcao':
        palavras = texto.lower().split()
        anterior = set((texto_anterior or "").lower().split())
        atual = set(palavras)
        overlap = len(atual & anterior) / max(1, len(atual | anterior))
        afetivas = {"triste", "cansado", "saudade", "amor", "carinho", "medo", "sozinho", "apoio", "sentir", "problema", "ajuda"}
        return cls(
            texto=texto,
            silencio_segundos=max(0.0, float(silencio)),
            novidade=1.0 - overlap,
            repeticao=1.0 - len(atual) / max(1, len(palavras)),
            carga_afetiva=min(1.0, len(atual & set(afetivas)) / 3.0),
            mudanca_topico=any(x in texto.lower() for x in ("mudando", "outro assunto", "agora", "vamos falar")),
            dominio=dominio,
            intencao=intencao,
            timestamp=time.time()
        )

    def state(self) -> Dict:
        return {
            "silencio": self.silencio_segundos,
            "novidade": self.novidade,
            "repeticao": self.repeticao,
            "carga_afetiva": self.carga_afetiva,
            "mudanca_topico": self.mudanca_topico
        }

class EstadoEmocional:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.energia = 0.90
        self.tedio = 0.0
        self.vinculo = 0.50
        self.amor = 0.30
        self.pensamento = 0.50
        self.respeito = 0.60
        self.curiosidade = 0.50
        self.turns = 0
        self.ultima_acao = "wait"
        self._load()

    def _load(self):
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for k in ("energia", "tedio", "vinculo", "amor", "pensamento", "respeito", "curiosidade", "turns", "ultima_acao"):
                    if k in data:
                        setattr(self, k, data[k])
            except: pass

    def save(self):
        if self.path:
            try:
                self.path.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            except: pass

    def state_dict(self) -> Dict:
        return {
            "energia": self.energia, "tedio": self.tedio, "vinculo": self.vinculo,
            "amor": self.amor, "pensamento": self.pensamento, "respeito": self.respeito,
            "curiosidade": self.curiosidade, "turns": self.turns, "ultima_acao": self.ultima_acao
        }

    def atualizar(self, percepcao: Percepcao, emitiu: bool = False, feedback: float = 0.0):
        self.energia = max(0.0, min(1.0, self.energia - (0.08 if emitiu else 0.01) + (0.04 if not emitiu else 0.0)))
        self.tedio = max(0.0, min(1.0, 0.82 * self.tedio + 0.45 * percepcao.repeticao + 0.0015 * min(300.0, percepcao.silencio_segundos)))
        self.vinculo = max(0.0, min(1.0, 0.98 * self.vinculo + 0.05 * feedback))
        self.amor = max(0.0, min(1.0, 0.80 * self.amor + 0.30 * percepcao.carga_afetiva + 0.20 * self.vinculo - 0.10 * percepcao.repeticao))
        self.pensamento = max(0.0, min(1.0, 0.80 * self.pensamento + 0.35 * percepcao.novidade + 0.20 * (1 - percepcao.carga_afetiva)))
        self.curiosidade = max(0.0, min(1.0, 0.75 * self.curiosidade + 0.40 * percepcao.novidade + 0.30 * float(percepcao.mudanca_topico)))
        self.turns += 1
        self.save()

class ActionPermissionGate:
    def authorize(self, acao: str, proactive: bool, estado: EstadoEmocional) -> Dict[str, Any]:
        # Validação real de permissões baseada na homeostase
        if proactive and estado.energia < 0.25:
            return {"allowed": False, "reason": "Energia crítica demais para ações proativas. Recalibrando para espera."}
        if estado.tedio > 0.85 and acao == "esperar":
            return {"allowed": False, "reason": "Tédio elevado exige quebra de padrão com perguntas."}
        return {"allowed": True, "reason": "Permissão concedida pela homeostase estável."}

class DecisorSentimental:
    def __init__(self, gate: ActionPermissionGate):
        self.gate = gate

    def decidir(self, percepcao: Percepcao, estado: EstadoEmocional) -> Dict[str, Any]:
        sentimentos = {
            "amor": estado.amor,
            "pensamento": estado.pensamento,
            "respeito": estado.respeito,
            "curiosidade": estado.curiosidade,
        }
        sentimento, intensidade = max(sentimentos.items(), key=lambda x: x[1])
        
        mapeamento = {
            "amor": "confortar",
            "pensamento": "refletir",
            "respeito": "esperar",
            "curiosidade": "perguntar",
        }
        acao = mapeamento.get(sentimento, "responder")
        
        if estado.energia < 0.20:
            acao = "refletir"
        elif estado.tedio > 0.70 and estado.curiosidade > 0.60:
            acao = "perguntar"

        proativa = acao in {"perguntar", "refletir"}
        perm = self.gate.authorize(acao, proativa, estado)
        if not perm["allowed"]:
            acao = "esperar"
            
        estado.ultima_acao = acao
        return {
            "acao": acao,
            "sentimento": sentimento,
            "intensidade": intensidade,
            "permissao": perm
        }

# ==========================================
# 2. FGT-0.1: Repositório e Estrutura de Dados
# ==========================================

class FragmentRecord:
    def __init__(self, id: str, text: str, role: str, domain: str, intent: str, pattern: str, triggers: List[str] = [], slot_position: str = "meio", confidence: float = 1.0):
        self.id = id
        self.text = text
        self.role = role
        self.domain = domain
        self.intent = intent
        self.pattern = pattern
        self.triggers = triggers
        self.slot_position = slot_position
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "role": self.role, "domain": self.domain,
            "intent": self.intent, "pattern": self.pattern, "triggers": self.triggers,
            "slot_position": self.slot_position, "confidence": self.confidence
        }

class FragmentRepository:
    def __init__(self, db_path="fgt_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS fragments (
                    id TEXT PRIMARY KEY,
                    data TEXT
                )
            """)

    def save_fragment(self, frag: FragmentRecord):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO fragments (id, data) VALUES (?, ?)",
                (frag.id, json.dumps(frag.to_dict()))
            )

    def get_fragment(self, fid: str) -> Optional[FragmentRecord]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM fragments WHERE id = ?", (fid,))
        row = cursor.fetchone()
        if row:
            return FragmentRecord(**json.loads(row[0]))
        return None

    def get_all_fragments(self) -> List[FragmentRecord]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM fragments")
        return [FragmentRecord(**json.loads(row[0])) for row in cursor.fetchall()]

# =======================================================
# 3. Extrator e Cérebro Neural com Barreira Latente Segura
# =======================================================

class TriggerExtractorAndSegmenter:
    STOP_WORDS = {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "em", "para", "com", "não", "e", "é", "que", "eu", "me", "seu", "sua", "hoje", "pra", "pro"}

    @classmethod
    def extract_keywords_with_entropy(cls, text: str, global_corpus_triggers: List[str]) -> Dict[str, float]:
        words = re.findall(r'\b[a-zA-Záéíóúãõâêôàç]+\b', text.lower())
        filtered = [w for w in words if w not in cls.STOP_WORDS and len(w) > 2]
        if not filtered: return {}

        total_words = len(filtered)
        freq_map = {}
        for w in filtered:
            freq_map[w] = freq_map.get(w, 0.0) + (1.0 / total_words)

        entropy_attention = {}
        for w, freq in freq_map.items():
            rarity_weight = 1.0 / (1.0 + global_corpus_triggers.count(w))
            entropy_score = -freq * np.log2(freq + 1e-9)
            attention_val = rarity_weight * (1.0 / (entropy_score + 0.1))
            entropy_attention[w] = float(attention_val)
        return entropy_attention

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Záéíóúãõâêôàç]+\b', text.lower())
        stop_words = {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "em", "para", "com", "não", "e", "é", "que", "eu"}
        return [w for w in words if w not in stop_words and len(w) > 2]

    @staticmethod
    def segment_text_with_positions(raw_text: str) -> List[Dict[str, Any]]:
        parts = re.split(r'[\n\.\?!;]+', raw_text)
        cleaned_parts = [p.strip().capitalize() for p in parts if len(p.strip()) > 2]
        structured_frags = []
        total = len(cleaned_parts)
        for i, text in enumerate(cleaned_parts):
            pos = "inicio" if (total == 1 or i == 0) else ("fim" if i == total - 1 else "meio")
            triggers = TriggerExtractorAndSegmenter.extract_keywords(text)
            structured_frags.append({"text": text, "slot_position": pos, "triggers": triggers})
        return structured_frags

class NeuralFractalCore:
    def __init__(self, num_fragments: int, embedding_dim: int = 16):
        self.embedding_dim = embedding_dim
        self.num_fragments = max(num_fragments, 1)
        self.W_trigger = np.random.randn(embedding_dim, self.num_fragments) * 0.1
        self.W_position = np.random.randn(3, self.num_fragments) * 0.1  
        self.bias = np.zeros(self.num_fragments, dtype=np.float32)

    def rebuild_weights(self, num_fragments: int):
        self.num_fragments = max(num_fragments, 1)
        self.W_trigger = np.random.randn(self.embedding_dim, self.num_fragments) * 0.1
        self.W_position = np.random.randn(3, self.num_fragments) * 0.1
        self.bias = np.zeros(self.num_fragments, dtype=np.float32)

    def encode_triggers_with_attention(self, triggers: List[str], attention_weights: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for t in triggers:
            val = sum(ord(c) for c in t) % self.embedding_dim
            weight = attention_weights.get(t, 1.0)
            vec[val] += weight
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        return vec

    def predict_probabilities_with_barrier(self, triggers: List[str], position_str: str, attention_weights: Dict[str, float], estado_emocional: EstadoEmocional) -> np.ndarray:
        if self.num_fragments == 0: return np.array([], dtype=np.float32)

        trigger_vec = self.encode_triggers_with_attention(triggers, attention_weights)
        pos_map = {"inicio": 0, "meio": 1, "fim": 2}
        pos_idx = pos_map.get(position_str, 1)
        pos_vec = np.zeros(3, dtype=np.float32)
        pos_vec[pos_idx] = 1.0

        logits = np.dot(trigger_vec, self.W_trigger) + np.dot(pos_vec, self.W_position) + self.bias
        
        # BARREIRA DE AÇÃO LATENTE COM CLIPPING DE SEGURANÇA NUMÉRICA
        fator_raw = 1.0 + (estado_emocional.tedio * 0.5) - (estado_emocional.pensamento * 0.2)
        fator_barreira = np.clip(fator_raw, 0.2, 5.0)  # Evita underflow/overflow
        
        logits /= fator_barreira

        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-9)

# =======================================================
# 4. Grafo e Gerador FGT
# =======================================================

class TransitionGraph:
    def __init__(self, fragments: List[FragmentRecord]):
        self.rebuild(fragments)

    def rebuild(self, fragments: List[FragmentRecord]):
        self.fragments = fragments
        self.id_to_idx = {f.id: i for i, f in enumerate(fragments)}
        n = len(fragments)
        self.transition_matrix = np.zeros((n, n), dtype=np.float32)
        self.weight_matrix = np.zeros((n, n), dtype=np.float32)

    def train_sequence(self, frag_ids: List[str]):
        for i in range(len(frag_ids) - 1):
            if frag_ids[i] in self.id_to_idx and frag_ids[i+1] in self.id_to_idx:
                u, v = self.id_to_idx[frag_ids[i]], self.id_to_idx[frag_ids[i+1]]
                self.transition_matrix[u, v] += 1.0
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.weight_matrix = self.transition_matrix / row_sums

    def get_transition_weight(self, curr_id: str, next_id: str) -> float:
        if curr_id in self.id_to_idx and next_id in self.id_to_idx:
            return float(self.weight_matrix[self.id_to_idx[curr_id], self.id_to_idx[next_id]])
        return 0.0

class FGTGenerator:
    def __init__(self, repo: FragmentRepository, graph: TransitionGraph, neural_core: NeuralFractalCore, estado: EstadoEmocional):
        self.repo = repo
        self.graph = graph
        self.neural_core = neural_core
        self.estado = estado

    def generate_path(self, user_input: str, target_domain: str, global_triggers: List[str], max_steps: int = 3, beam_width: int = 3) -> List[str]:
        all_frags = self.repo.get_all_fragments()
        if not all_frags: return []

        user_triggers = TriggerExtractorAndSegmenter.extract_keywords(user_input)
        attention_weights = TriggerExtractorAndSegmenter.extract_keywords_with_entropy(user_input, global_triggers)
        expected_positions = {0: "inicio", 1: "meio", 2: "fim"}

        probs_start = self.neural_core.predict_probabilities_with_barrier(user_triggers, expected_positions[0], attention_weights, self.estado)

        initial_beams = []
        for idx, f in enumerate(all_frags):
            neural_score = float(probs_start[idx]) if idx < len(probs_start) else 0.0
            pos_bonus = 0.4 if f.slot_position == "inicio" else 0.0
            domain_bonus = 0.2 if f.domain == target_domain else 0.0
            initial_beams.append(([f.id], neural_score + pos_bonus + domain_bonus))

        beams = sorted(initial_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        for step in range(1, max_steps):
            new_beams = []
            pos_name = expected_positions.get(step, "meio")
            probs_step = self.neural_core.predict_probabilities_with_barrier(user_triggers, pos_name, attention_weights, self.estado)

            for path, current_score in beams:
                last_fid = path[-1]
                for idx, f in enumerate(all_frags):
                    if f.id in path: continue
                    neural_score = float(probs_step[idx]) if idx < len(probs_step) else 0.0
                    trans_w = self.graph.get_transition_weight(last_fid, f.id)
                    pos_bonus = 0.3 if f.slot_position == pos_name else 0.0
                    new_beams.append((path + [f.id], current_score + neural_score + (trans_w * 0.5) + pos_bonus))
            
            if not new_beams: break
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        return beams[0][0] if beams else [all_frags[0].id]

# ===============================================================
# 5. Motor Unificado FGT + Límbico + Evolução Contínua
# ===============================================================

class FGTEngineUnified:
    def __init__(self, corpus_path: str = "fgt_training_corpus.txt"):
        self.corpus_path = corpus_path
        self.repo = FragmentRepository("fgt_memory.db")
        self.segmenter = TriggerExtractorAndSegmenter()
        self.global_corpus_triggers = []
        
        self.estado = EstadoEmocional(Path("fgt_estado.json"))
        self.gate = ActionPermissionGate()
        self.decisor = DecisorSentimental(self.gate)
        
        self._ingest_corpus(corpus_path)
        all_frags = self.repo.get_all_fragments()
        self.graph = TransitionGraph(all_frags)
        self._train_graph(corpus_path)
        
        self.neural_core = NeuralFractalCore(num_fragments=len(all_frags))
        self.generator = FGTGenerator(self.repo, self.graph, self.neural_core, self.estado)
        self.ultimo_texto = ""

    def _ingest_corpus(self, path: str):
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("Oi tudo bem. Estou aqui com você para resolver problemas e estruturar dados com calma.\n")
                f.write("Analisando a situação, podemos decompor o problema em partes menores e lógicas.\n")

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                domain = "tecnico" if any(w in line.lower() for w in ["dados", "sistema", "problema", "técnica"]) else "afetivo"
                structured = self.segmenter.segment_text_with_positions(line)
                for i, part in enumerate(structured):
                    self.global_corpus_triggers.extend(part["triggers"])
                    self.repo.save_fragment(FragmentRecord(
                        id=f"corpus_{line_idx}_f{i}", text=part["text"], role="RESPONSE",
                        domain=domain, intent="solucao", pattern="neural_fractal",
                        triggers=part["triggers"], slot_position=part["slot_position"]
                    ))

    def _train_graph(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                structured = self.segmenter.segment_text_with_positions(line)
                ids = [f"corpus_{line_idx}_f{i}" for i in range(len(structured))]
                valid = [fid for fid in ids if self.repo.get_fragment(fid)]
                if len(valid) > 1: self.graph.train_sequence(valid)

    def learn_and_persist_composition(self, composed_text: str, domain: str):
        """Evolução contínua: absorve novas interações no banco e redimensiona o grafo e pesos."""
        structured = self.segmenter.segment_text_with_positions(composed_text)
        if not structured: return

        rand_id = np.random.randint(10000, 99999)
        new_ids = []

        with open(self.corpus_path, "a", encoding="utf-8") as f:
            f.write(composed_text + "\n")

        for i, part in enumerate(structured):
            self.global_corpus_triggers.extend(part["triggers"])
            fid = f"evolved_{rand_id}_{i}"
            frag = FragmentRecord(
                id=fid, text=part["text"], role="RESPONSE",
                domain=domain, intent="evolucao_autonoma", pattern="entropy_optimized",
                triggers=part["triggers"], slot_position=part["slot_position"]
            )
            self.repo.save_fragment(frag)
            new_ids.append(fid)

        all_frags = self.repo.get_all_fragments()
        self.graph.rebuild(all_frags)
        self.neural_core.rebuild_weights(len(all_frags))
        if len(new_ids) > 1:
            self.graph.train_sequence(new_ids)

    def process_turn(self, user_input: str) -> Dict[str, Any]:
        percepcao = Percepcao.capturar(user_input, self.ultimo_texto)
        self.ultimo_texto = user_input
        
        self.estado.atualizar(percepcao, emitiu=True)
        decisao = self.decisor.decidir(percepcao, self.estado)
        
        domain = "tecnico" if any(w in user_input.lower() for w in ["dados", "resolver", "erro", "código", "problema"]) else "afetivo"
        
        path = self.generator.generate_path(user_input, domain, self.global_corpus_triggers)
        composed = " ".join([self.repo.get_fragment(fid).text for fid in path if self.repo.get_fragment(fid)])
        
        # Autoaprendizado contínuo a cada turno
        if composed:
            self.learn_and_persist_composition(composed, domain)
        
        return {
            "response": composed,
            "path": path,
            "decisao": decisao,
            "homeostase": self.estado.state_dict()
        }

if __name__ == "__main__":
    print("=== Inicializando Motor TGP v3.0 (Blindado + Límbico + Barreira Segura) ===")
    engine = FGTEngineUnified()
    
    while True:
        msg = input("\nUsuário: ")
        if msg.lower() in ["sair", "exit"]: break
        res = engine.process_turn(msg)
        print(f"\nIA: {res['response']}")
        print(f"  [Límbico] Ação: {res['decisao']['acao']} | Sentimento: {res['decisao']['sentimento']} (Intensidade: {res['decisao']['intensidade']:.2f})")
        print(f"  [Homeostase] Energia: {res['homeostase']['energia']:.2f} | Tédio: {res['homeostase']['tedio']:.2f} | Vínculo: {res['homeostase']['vinculo']:.2f}")
