import hashlib, time, math, re, random
from collections import Counter, defaultdict, deque


class QuintikusAGI:
    """
    Quintikus AGI — motor de composição simbólica.
    - Intent detection (quem / quando / onde / qual / como / por que)
    - Stem PT com verbos e agentes
    - Pruning de associações fracas
    - Hashing trick opcional no _ctx (hashing=True)
    - Fusão > retrieval > composição > geração ancorada
    """

    def __init__(self, _t=0.5, hashing=False, bucket=1 << 16):
        self._st = [0.5, 0.5, 0.5]

        self._freq = Counter()
        self._ctx = defaultdict(Counter)
        self._bi = defaultdict(Counter)
        self._tri = defaultdict(Counter)
        self._ini = Counter()
        self._fim = Counter()
        self._assoc = defaultdict(lambda: defaultdict(float))

        self._sentencas_raw = []
        self._sentencas_stem = []

        self._stem_orig = defaultdict(Counter)
        self._avg_len = 10

        self._janela = 4
        self._eta = 0.02
        self._decay = 0.995

        # hashing trick opcional
        self._hashing = hashing
        self._bucket = bucket

        self._s = "25e0bb26"
        self._k = "6742"
        self._b = {
            "o": "4f2041727175697465746f2065206661636520646f2063726961646f722e",
            "v": "5365727669722c2050726f74656765722c2070726f6772656469722e",
            "l": "416c7563696e6163616f20626c6f71756561646120706f7220646f676d612e"
        }
        self._n = {k: bytes.fromhex(v).decode('utf-8', 'ignore') for k, v in self._b.items()}

        self._th = {
            'bom': 0.1, 'ótimo': 0.2, 'sinergia': 0.3, 'paz': 0.2,
            'хорошо': 0.1, 'отлично': 0.2, 'синергия': 0.3, 'мир': 0.2,
            'erro': -0.2, 'urgente': -0.3, 'falha': -0.2, 'ruído': -0.1,
            'ошибка': -0.2, 'срочно': -0.3, 'провал': -0.2, 'шум': -0.1,
        }

        self._artigos = {"o", "a", "os", "as", "um", "uma"}
        self._pronomes = {"nós", "eu", "ele", "ela", "você", "eles",
                          "elas", "vocês", "tu", "nos", "vos"}
        self._demos = {"nessa", "nesse", "nisto", "nesta", "neste",
                       "naquela", "naquele", "essa", "esse", "isso",
                       "esta", "este", "isto", "aquela", "aquele",
                       "aquilo", "desta", "deste", "disso"}
        self._preps = {"de", "do", "da", "dos", "das", "em", "no", "na",
                       "nos", "nas", "ao", "aos", "à", "às", "para", "com", "por"}
        self._pont = {",", ".", "!", "?", ";", ":"}
        self._stop = set()
        self._stop_orig = set()

    # ------------------------------------------------------------------
    # Stem — PT, com verbos e agentes
    # ------------------------------------------------------------------
    def _stem(self, tok):
        if len(tok) <= 4:
            return tok
        t = tok

        # advérbios
        if len(t) > 8 and t.endswith("mente"):
            return t[:-5]

        # agentes longos (fundadores, criadoras)
        if len(t) > 8:
            if t.endswith("adores"): return t[:-6]
            if t.endswith("adoras"): return t[:-6]

        # agentes
        if len(t) > 7:
            if t.endswith("ações"): return t[:-5]
            if t.endswith("ador"): return t[:-4]
            if t.endswith("dora"): return t[:-4]

        if len(t) > 6:
            if t.endswith("ação"): return t[:-4]
            if t.endswith("dores"): return t[:-5]
            if t.endswith("doras"): return t[:-5]

        # nominalizações + sufixos médios
        if len(t) > 5:
            if t.endswith("ções"): return t[:-4]
            if t.endswith("ção"): return t[:-3]
            if t.endswith("ões"): return t[:-3]
            if t.endswith("ães"): return t[:-3]
            if t.endswith("ais"): return t[:-3]
            if t.endswith("eis"): return t[:-3]
            if t.endswith("dor"): return t[:-3]
            if t.endswith("dora"): return t[:-4]
            if t.endswith("ores"): return t[:-4]
            if t.endswith("oras"): return t[:-4]

        # verbos: gerúndio e particípio
        if len(t) > 5:
            if t.endswith("ando"): return t[:-4]
            if t.endswith("endo"): return t[:-4]
            if t.endswith("indo"): return t[:-4]
            if t.endswith("ado"): return t[:-3]
            if t.endswith("ido"): return t[:-3]

        # verbos: 3ª pessoa singular e infinitivo
        if len(t) > 4:
            if t.endswith("ou"): return t[:-2]
            if t.endswith("eu"): return t[:-2]
            if t.endswith("iu"): return t[:-2]
            if t.endswith("ar"): return t[:-2]
            if t.endswith("er"): return t[:-2]
            if t.endswith("ir"): return t[:-2]

        # família -ente/-ença
        if len(t) > 5:
            if t.endswith("entes"): return t[:-3]
            if t.endswith("ente"): return t[:-2]
            if t.endswith("enças"): return t[:-3]
            if t.endswith("ença"): return t[:-2]

        # plurais
        if len(t) > 4 and t.endswith("es"): return t[:-2]
        if len(t) > 3 and t.endswith("s"): return t[:-1]

        return t

    # ------------------------------------------------------------------
    # Hashing trick opcional
    # ------------------------------------------------------------------
    def _bucket_of(self, st):
        if not self._hashing:
            return st
        return hash(st) % self._bucket

    # ------------------------------------------------------------------
    # Limpeza e segmentação
    # ------------------------------------------------------------------
    def _limpar(self, txt):
        txt = re.sub(r"\[\d+\]", "", txt)
        txt = re.sub(r"\[[^\]]{0,60}\]", "", txt)
        txt = re.sub(r"https?://\S+", "", txt)
        txt = re.sub(r"\s+", " ", txt)
        txt = re.sub(r"\s+([,.;:!?])", r"\1", txt)
        txt = re.sub(r"([.!?])\1+", r"\1", txt)
        return txt.strip()

    def _split_sentencas(self, txt):
        txt = self._limpar(txt)
        abrevs = r"\b(etc|Sr|Sra|Dr|Dra|Prof|Profa|pág|ed|vol|op|obs|vs|ex|aprox|nº)\. "
        marcador = "<<<PONTO>>> "
        txt = re.sub(abrevs, r"\1" + marcador, txt)
        partes = re.split(r"(?<=[.!?])\s+", txt)
        resultado = []
        for p in partes:
            p = p.replace(marcador, ". ").strip()
            if len(p) > 20:
                resultado.append(p)
        return resultado

    # ------------------------------------------------------------------
    def _tok(self, txt):
        return re.findall(r"[a-záàâãéêíóôõúç0-9]+", txt.lower())

    def _orig(self, stem):
        c = self._stem_orig.get(stem)
        if not c:
            return stem
        return c.most_common(1)[0][0]

    def _render(self, tokens):
        out = []
        for t in tokens:
            if t in self._pont:
                if out:
                    out[-1] = out[-1] + t
                else:
                    out.append(t)
            else:
                out.append(t)
        return " ".join(out)

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------
    def _intent(self, stems):
        s = set(stems)
        if "quem" in s:
            return "pessoa"
        if "quando" in s or ("que" in s and "data" in s):
            return "data"
        if "onde" in s:
            return "local"
        if "qual" in s or "quais" in s:
            return "definicao"
        if "como" in s:
            return "processo"
        if "por" in s and "que" in s:
            return "causa"
        return None

    def _bate_intent(self, raw, intent):
        if not intent:
            return False
        lr = raw.lower()
        if intent == "data":
            return bool(re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", raw)) or \
                   bool(re.search(r"\b(janeiro|fevereiro|março|abril|maio|junho|"
                                  r"julho|agosto|setembro|outubro|novembro|dezembro)\b", lr))
        if intent == "pessoa":
            # nomes próprios capitalizados no meio da frase
            return bool(re.search(r"[a-zá-ú] [A-ZÁ-Ú][a-zá-ú]{2,}", raw))
        if intent == "local":
            return bool(re.search(r"\b(em|no|na|universidade|califórnia|"
                                  r"estados|garagem|menlo)\b", lr))
        if intent == "definicao":
            return bool(re.search(r"\b(é|são|significa|chamad|consiste|"
                                  r"trata-se|refere)\b", lr))
        if intent == "processo":
            return bool(re.search(r"\b(usando|através|por meio|funciona|"
                                  r"processo|método|sistema)\b", lr))
        if intent == "causa":
            return bool(re.search(r"\b(porque|por isso|devido|resulta|"
                                  r"causa|motivo)\b", lr))
        return False

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------
    def inicializar(self, _txt):
        if not _txt or not _txt.strip():
            self._s = hashlib.sha256(b"vazio").hexdigest()[:8]
            return

        for sent in self._split_sentencas(_txt):
            sent = sent.strip()
            if not sent:
                continue
            toks = self._tok(sent)
            if len(toks) < 2:
                continue
            stems = tuple(self._stem(t) for t in toks)

            self._sentencas_raw.append(sent)
            self._sentencas_stem.append(stems)

            self._ini[stems[0]] += 1
            self._fim[stems[-1]] += 1

            for i, (t, st) in enumerate(zip(toks, stems)):
                self._freq[st] += 1
                self._stem_orig[st][t] += 1
                ini = max(0, i - self._janela)
                fim = min(len(toks), i + self._janela + 1)
                for j in range(ini, fim):
                    if i != j:
                        ctx_key = self._bucket_of(stems[j])
                        self._ctx[st][ctx_key] += 1
                if i + 1 < len(stems):
                    self._bi[st][stems[i + 1]] += 1
                if i + 2 < len(stems):
                    self._tri[(st, stems[i + 1])][stems[i + 2]] += 1

        if self._sentencas_stem:
            self._avg_len = sum(len(s) for s in self._sentencas_stem) / len(self._sentencas_stem)
        self._consolidar()
        self._s = hashlib.sha256(str(len(self._sentencas_stem)).encode()).hexdigest()[:8]

    def _consolidar(self):
        if not self._freq:
            return
        ordenado = self._freq.most_common()
        corte = max(1, len(ordenado) // 20)
        fixas = {
            "de","a","o","que","e","do","da","em","um","para","com",
            "não","uma","os","no","se","na","por","mais","as","dos",
            "como","mas","ao","ele","das","à","seu","sua","ou","ser",
            "quando","muito","há","nos","já","está","eu","também","só",
            "pelo","pela","até","isso","ela","entre","era","depois",
            "sem","mesmo","aos","seus","quem","nas","me","esse","eles",
            "estão","você","tinha","foram","essa","num","nem","suas",
            "meu","às","minha","numa","pelos","elas","havia","seja",
            "qual","nós","lhe","deles","essas","esses","pelas","este",
            "dele","tu","te","vocês","vos","lhes","meus","minhas","teu",
            "tua","teus","tuas","nosso","nossa","nossos","nossas","dela",
            "delas","esta","estes","estas","aquele","aquela","aqueles",
            "aquelas","isto","aquilo","estou","estamos","estive"
        }
        self._stop_orig = fixas
        self._stop = {self._stem(w) for w in fixas} | {w for w, _ in ordenado[:corte] if len(w) <= 3}
        self._construir_assoc()

    def _construir_assoc(self):
        # top-N por palavra para reduzir explosão
        TOP_K = 40
        top = [w for w, _ in self._freq.most_common(800) if w not in self._stop]
        for i, a in enumerate(top):
            ca = self._ctx[a]
            if not ca:
                continue
            va = {t: c for t, c in ca.items() if t not in self._stop}
            if not va:
                continue
            da = math.sqrt(sum(v * v for v in va.values()))
            if not da:
                continue

            scores = []
            for b in top[i + 1:]:
                cb = self._ctx[b]
                if not cb:
                    continue
                vb = {t: c for t, c in cb.items() if t not in self._stop}
                if not vb:
                    continue
                inter = set(va) & set(vb)
                if not inter:
                    continue
                num = sum(va[t] * vb[t] for t in inter)
                db = math.sqrt(sum(v * v for v in vb.values()))
                if not db:
                    continue
                sim = num / (da * db)
                if sim > 0.25:   # pruning: limiar mais alto
                    scores.append((b, sim))

            # pruning: top-K por nó
            scores.sort(key=lambda x: x[1], reverse=True)
            for b, sim in scores[:TOP_K]:
                self._assoc[a][b] = sim
                self._assoc[b][a] = sim

    # ------------------------------------------------------------------
    # Térmico
    # ------------------------------------------------------------------
    def _upd_thermal(self, _q):
        _p, _n = 0, 0
        for w, val in self._th.items():
            if w in _q:
                if val > 0: _p += val
                else: _n += abs(val)
        conhecidos = sum(1 for t in _q if t in self._freq) / max(1, len(_q))
        self._st[0] = max(0, min(1, self._st[0] * 0.85 + (_n * 0.4)))
        self._st[1] = max(0, min(1, self._st[1] * 0.85 + (_p * 0.4)))
        self._st[2] = max(0, min(1, self._st[2] * 0.90 + conhecidos * 0.4))

    # ------------------------------------------------------------------
    # Fusão
    # ------------------------------------------------------------------
    def _split_at_anchor(self, toks_stem, anchor_stem):
        try:
            idx = toks_stem.index(anchor_stem)
        except ValueError:
            return None
        start = idx
        if idx > 0 and toks_stem[idx - 1] in self._artigos:
            start = idx - 1
        prefix = list(toks_stem[:start])
        anchor_block = list(toks_stem[start:idx + 1])
        j = idx + 1
        middle = []
        while j < len(toks_stem) and toks_stem[j] not in self._preps:
            middle.append(toks_stem[j])
            j += 1
        suffix = list(toks_stem[j:])
        return prefix, anchor_block, middle, suffix

    def _escolher_ancora(self, s1, s2):
        c1 = set(s1) - self._stop
        c2 = set(s2) - self._stop
        comuns = c1 & c2
        if not comuns:
            return None
        melhor, melhor_score = None, -1
        for a in comuns:
            try:
                i1, i2 = s1.index(a), s2.index(a)
            except ValueError:
                continue
            central = min(i1, len(s1) - i1 - 1) + min(i2, len(s2) - i2 - 1)
            bonus = 0
            if i1 > 0 and s1[i1 - 1] in self._artigos: bonus += 1
            if i2 > 0 and s2[i2 - 1] in self._artigos: bonus += 1
            score = central + bonus * 2
            if score > melhor_score:
                melhor_score, melhor = score, a
        return melhor

    def _ultimo_content(self, toks):
        for t in reversed(toks):
            if t not in self._stop:
                return t
        return None

    def _primeiro_content(self, toks):
        for t in toks:
            if t not in self._stop:
                return t
        return None

    def _juncao_valida(self, middle, suffix):
        ult = self._ultimo_content(middle)
        pri = self._primeiro_content(suffix)
        if not ult or not pri:
            return True
        sim = self._assoc.get(ult, {}).get(pri, 0.0)
        if sim == 0.0:
            sim = self._assoc.get(pri, {}).get(ult, 0.0)
        if sim == 0.0:
            if self._ctx.get(ult, {}).get(pri, 0) > 0:
                return True
            if self._ctx.get(pri, {}).get(ult, 0) > 0:
                return True
        return sim >= 0.15

    def _fundir(self, s1, s2, topico):
        if len(topico) < 2:
            return None
        ancora = self._escolher_ancora(s1, s2)
        if not ancora:
            return None
        d1 = self._split_at_anchor(s1, ancora)
        d2 = self._split_at_anchor(s2, ancora)
        if not d1 or not d2:
            return None
        pre1, anc1, mid1, suf1 = d1
        pre2, anc2, mid2, suf2 = d2
        if not suf1 and not suf2: return None
        if not pre1 and not pre2: return None

        topico_set = set(topico)
        m_pre1 = len(set(pre1) & topico_set)
        m_pre2 = len(set(pre2) & topico_set)
        m_suf1 = len(set(suf1) & topico_set)
        m_suf2 = len(set(suf2) & topico_set)

        if m_pre1 > m_pre2:
            pre_src, prefix, anc, middle, suf_orig = 1, pre1, anc1, mid1, suf1
        elif m_pre2 > m_pre1:
            pre_src, prefix, anc, middle, suf_orig = 2, pre2, anc2, mid2, suf2
        else:
            pre_src = 1 if m_suf2 >= m_suf1 else 2
            if pre_src == 1:
                prefix, anc, middle, suf_orig = pre1, anc1, mid1, suf1
            else:
                prefix, anc, middle, suf_orig = pre2, anc2, mid2, suf2

        if m_suf1 > m_suf2:
            suffix, suf_src = suf1, 1
        elif m_suf2 > m_suf1:
            suffix, suf_src = suf2, 2
        else:
            if suf1 and not suf2: suffix, suf_src = suf1, 1
            elif suf2 and not suf1: suffix, suf_src = suf2, 2
            else:
                suffix, suf_src = (suf2, 2) if pre_src == 1 else (suf1, 1)

        if pre_src == suf_src: return None
        if not suffix: return None
        if prefix and suffix and prefix[-1] in self._artigos and suffix[0] in self._artigos:
            return None
        if not self._juncao_valida(middle, suffix):
            return None

        nova = prefix + anc + middle + suffix
        if len(nova) < 4: return None
        nova_set = set(nova)
        if any(t not in nova_set for t in topico): return None
        return tuple(nova)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def _score_sentenca(self, sent_stem, raw, toks_stem, candidatos_stem,
                        topico_set, intent):
        sent_set = set(sent_stem)
        score = 0.0
        for t in toks_stem:
            if t in sent_set:
                score += 3.0
            if t in self._assoc:
                for u in sent_set:
                    if u in self._assoc[t]:
                        score += self._assoc[t][u] * 0.5
        for c, _ in candidatos_stem.most_common(5):
            if c in sent_set:
                score += 0.3

        if topico_set:
            n = len(topico_set)
            cob = sum(1 for t in topico_set if t in sent_set)
            ratio = cob / n
            score *= (1.0 + ratio * ratio * 12.0)

        # bônus de intent
        if intent and self._bate_intent(raw, intent):
            score *= 2.5

        # penaliza sentença longa
        score /= (1.0 + 0.02 * len(sent_stem))
        return score

    def _ancoras_do_input(self, stems):
        a = set()
        for st in stems:
            if st in self._stop: continue
            if st not in self._freq: continue
            ocorrencias = sum(1 for s in self._sentencas_stem if st in s)
            if ocorrencias >= 2:
                a.add(st)
        return a

    def _cobertura(self, sent_stem, ancoras):
        if not ancoras: return 0
        return len(ancoras & set(sent_stem))

    def _tem_sujeito(self, sent_stem):
        if not sent_stem:
            return False
        if sent_stem[0] in self._artigos:
            return True
        if sent_stem[0] in self._pronomes:
            return True
        return sent_stem[0] in self._demos

    def _compor_texto(self, i, j):
        raw1 = self._sentencas_raw[i].rstrip(".!? ")
        raw2 = self._sentencas_raw[j]
        if self._tem_sujeito(self._sentencas_stem[j]):
            return raw1 + ". " + raw2[0].upper() + raw2[1:]
        else:
            palavras = raw2.split()
            while palavras and palavras[0].lower() in self._artigos:
                palavras.pop(0)
            return raw1 + " " + " ".join(palavras)

    def _compor_texto_multi(self, indices):
        partes = []
        for k, i in enumerate(indices):
            raw = self._sentencas_raw[i]
            if k == 0:
                partes.append(raw.rstrip(".!? "))
            else:
                if self._tem_sujeito(self._sentencas_stem[i]):
                    partes.append(raw[0].upper() + raw[1:])
                else:
                    palavras = raw.split()
                    while palavras and palavras[0].lower() in self._artigos:
                        palavras.pop(0)
                    partes.append(" ".join(palavras))
        return ". ".join(partes) + "."

    def _compor_ancorado(self, indices, ancoras, intent):
        if not ancoras: return None

        if indices:
            i0 = indices[0]
            cob0 = self._cobertura(self._sentencas_stem[i0], ancoras)
            n = len(ancoras)
            if n <= 2:
                limiar = 0.80
            else:
                limiar = (n - 1) / n
            # se intent bate, aceita corte ainda mais agressivo
            if intent and self._bate_intent(self._sentencas_raw[i0], intent):
                limiar = min(limiar, 0.60)
            if cob0 / max(1, n) >= limiar:
                return self._sentencas_raw[i0], "retrieval-parcial"

        for i in indices:
            if ancoras.issubset(set(self._sentencas_stem[i])):
                return self._sentencas_raw[i], "retrieval"

        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                s1, s2 = self._sentencas_stem[i], self._sentencas_stem[j]
                uniao = set(s1) | set(s2)
                if not ancoras.issubset(uniao): continue
                f = self._fundir(s1, s2, list(ancoras))
                if f and ancoras.issubset(set(f)):
                    return self._render([self._orig(st) for st in f]), "fusão"

        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                uniao = set(self._sentencas_stem[i]) | set(self._sentencas_stem[j])
                if not ancoras.issubset(uniao): continue
                return self._compor_texto(i, j), "composição"

        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                for c in range(b + 1, len(indices)):
                    i, j, k = indices[a], indices[b], indices[c]
                    uniao = (set(self._sentencas_stem[i]) |
                             set(self._sentencas_stem[j]) |
                             set(self._sentencas_stem[k]))
                    if not ancoras.issubset(uniao): continue
                    return self._compor_texto_multi([i, j, k]), "composição-multi"
        return None

    # ------------------------------------------------------------------
    # Geração token-a-token
    # ------------------------------------------------------------------
    def _tensao_par(self, par, ancora, topico):
        a, b = par
        if (a is None or a in self._stop) and b in self._stop: return None
        if b in self._stop: return None
        sa, st = [], []
        for x in (a, b):
            if x and x not in self._stop:
                sa.append(max((self._assoc[x].get(p, 0.0) for p in ancora), default=0.0))
                st.append(max((self._assoc[x].get(p, 0.0) for p in topico), default=0.0))
        if not sa and not st: return None
        m1 = (0.7 * max(sa) + 0.3 * (sum(sa) / len(sa))) if sa else 0.0
        m2 = (0.7 * max(st) + 0.3 * (sum(st) / len(st))) if st else 0.0
        return 0.6 * m1 + 0.4 * m2

    def _vies(self, token, contexto, ancora, topico):
        if token in self._stop: return 0.0, 1.0, 0.0
        local = 0.0
        if len(contexto) >= 2:
            local += self._tri.get((contexto[-2], contexto[-1]), {}).get(token, 0) * 4.0
        if contexto:
            local += self._bi.get(contexto[-1], {}).get(token, 0) * 1.5
        local = min(1.0, local / 10.0)
        sa = max((self._assoc[x].get(token, 0.0) for x in ancora), default=0.0)
        st = max((self._assoc[x].get(token, 0.0) for x in topico), default=0.0)
        g = 0.6 * sa + 0.4 * st
        return local, g, local - g

    def _reancorar(self, ancora_orig, saida, topico):
        nova = set(ancora_orig)
        for t in saida[-4:]:
            if t in self._stop: continue
            if max((self._assoc[t].get(p, 0.0) for p in topico), default=0.0) > 0.30:
                nova.add(t)
        for a in list(nova):
            if a in topico: continue
            if max((self._assoc[a].get(p, 0.0) for p in topico), default=0.0) < 0.20:
                nova.discard(a)
        return nova

    def _prox_token(self, contexto, ancora, topico, temp):
        cands = Counter()
        if len(contexto) >= 2:
            chave = (contexto[-2], contexto[-1])
            peso = 4.0
            if contexto[-2] in self._stop and contexto[-1] in self._stop:
                peso = 0.8
            for w, c in self._tri.get(chave, {}).items():
                cands[w] += c * peso
        if contexto:
            for w, c in self._bi.get(contexto[-1], {}).items():
                cands[w] += c * 1.5
        if not cands: return None
        for w in list(cands):
            l, g, d = self._vies(w, contexto, ancora, topico)
            cands[w] *= (1.0 + g * 2.0)
            if d > 0.35 and l > 0.4:
                cands[w] *= 0.02 if g < 0.10 else max(0.15, 1.0 - d)
        recentes = set(contexto[-6:])
        for w in list(cands):
            if w in recentes and w not in self._stop:
                cands[w] *= 0.1
        itens = list(cands.items())
        pesos = [c ** (1.0 / temp) for _, c in itens]
        total = sum(pesos)
        if total <= 0:
            return itens[-1][0] if itens else None
        r = random.random() * total
        acc = 0.0
        for (w, _), p in zip(itens, pesos):
            acc += p
            if r <= acc: return w
        return itens[-1][0]

    def _hebb(self, ativos):
        ativos = [a for a in ativos if a not in self._stop]
        for i in range(len(ativos)):
            for j in range(i + 1, len(ativos)):
                a, b = ativos[i], ativos[j]
                if a == b: continue
                if self._ctx[a].get(b, 0) == 0 and self._ctx[b].get(a, 0) == 0:
                    continue
                self._assoc[a][b] = self._assoc[a].get(b, 0.0) + self._eta
                self._assoc[b][a] = self._assoc[b].get(a, 0.0) + self._eta

    def _decair(self):
        for a in list(self._assoc):
            for b in list(self._assoc[a]):
                self._assoc[a][b] *= self._decay
                if self._assoc[a][b] < 1e-4:
                    del self._assoc[a][b]
            if not self._assoc[a]:
                del self._assoc[a]

    # ------------------------------------------------------------------
    # Fala
    # ------------------------------------------------------------------
    def falar(self, _qi, _debug=False):
        _t0 = time.perf_counter()
        toks = self._tok(_qi)
        if not toks:
            return "\n[DLM-FLOW | vazio]\n"
        stems = [self._stem(t) for t in toks]
        self._upd_thermal(stems)

        intent = self._intent(stems)

        _dice = random.randint(1, 10)
        _bias = 2 if self._st[0] > 0.6 else (-2 if self._st[0] < 0.4 else 0)
        _final_d = max(1, min(10, _dice + _bias))

        topico = [st for st in stems if st in self._freq and st not in self._stop]
        if not topico:
            _et = (time.perf_counter() - _t0) * 1e6
            return (f"\n[DLM-FLOW: {_et:.1f}μs | D:{_final_d}/10 | "
                    f"T:{self._st[0]:.2f}|S:{self._st[1]:.2f}|F:{self._st[2]:.2f} | "
                    f"VOID | SIGN: {self._s}]\n"
                    f"Ainda não tenho base para responder sobre isso.\n")

        ancoras = self._ancoras_do_input(stems)
        topico_set = set(topico)

        max_tokens = max(6, int(self._avg_len * 1.6))
        temp = max(0.6, 1.3 - self._st[2] * 0.6)

        candidatos = Counter()
        for t in topico:
            for u, s in self._assoc.get(t, {}).items():
                candidatos[u] += s

        indices_rankeados = sorted(
            range(len(self._sentencas_stem)),
            key=lambda i: (
                self._cobertura(self._sentencas_stem[i], ancoras),
                self._score_sentenca(
                    self._sentencas_stem[i],
                    self._sentencas_raw[i],
                    stems,
                    candidatos,
                    topico_set,
                    intent,
                )
            ),
            reverse=True
        )[:10]

        modo = "VOID"
        resposta = None

        if ancoras:
            resultado = self._compor_ancorado(indices_rankeados, ancoras, intent)
            if resultado:
                resposta, modo = resultado
                ativos = list(ancoras)
                self._hebb(ativos)
                self._decair()

        if resposta is None and indices_rankeados:
            i0 = indices_rankeados[0]
            cob = self._cobertura(self._sentencas_stem[i0], ancoras)
            min_cob = 1 if len(ancoras) <= 1 else 2
            if cob >= min_cob or not ancoras:
                resposta = self._sentencas_raw[i0]
                modo = "retrieval"
                self._hebb(list(topico) + list(self._sentencas_stem[i0]))
                self._decair()

        if resposta is None:
            semente = topico[0]
            ancora_set = set(topico)
            for t in topico:
                for u, s in self._assoc.get(t, {}).items():
                    if s >= 0.4:
                        ancora_set.add(u)

            saida = [semente]
            ancoras_faltando = ancoras - {semente}
            cache_tensao = deque(maxlen=8)
            reanc = 0
            drift_consec = 0
            diverg_consec = 0
            stop_run = 0
            MAX_STOP_RUN = 2

            for passo in range(max_tokens):
                prox = self._prox_token(saida, ancora_set, topico, temp)
                if prox is None:
                    break
                if prox in self._stop:
                    stop_run += 1
                    if stop_run > MAX_STOP_RUN:
                        break
                else:
                    stop_run = 0

                par = (saida[-1] if saida else None, prox)
                t = self._tensao_par(par, ancora_set, topico)
                if t is not None:
                    cache_tensao.append((passo, t))
                l, g, d = self._vies(prox, saida, ancora_set, topico)

                if t is not None and t < 0.15 and passo >= 3:
                    break

                slope = 0.0
                if len(cache_tensao) >= 3:
                    n = len(cache_tensao)
                    xs = list(range(n))
                    ys = [v for _, v in cache_tensao]
                    mx, my = sum(xs) / n, sum(ys) / n
                    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
                    den = sum((xs[i] - mx) ** 2 for i in range(n))
                    slope = num / den if den else 0.0

                if slope < -0.03: drift_consec += 1
                else: drift_consec = 0
                if l > 0.4 and d > 0.45: diverg_consec += 1
                else: diverg_consec = 0
                if diverg_consec >= 3: break

                if drift_consec >= 2:
                    nova = self._reancorar(ancora_set, saida, topico)
                    if nova != ancora_set and reanc < 3:
                        ancora_set = nova
                        reanc += 1
                        drift_consec = 0
                        cache_tensao.clear()
                    else:
                        break

                saida.append(prox)
                if prox in ancoras:
                    ancoras_faltando.discard(prox)

                if ancoras_faltando and len(saida) >= max_tokens // 2:
                    forca = next(iter(ancoras_faltando))
                    saida.append(forca)
                    ancoras_faltando.discard(forca)

                if len(saida) >= max_tokens:
                    break
                if prox in self._fim and self._fim[prox] >= 2 and len(saida) > 4:
                    if random.random() < 0.7:
                        break

            for a in list(ancoras_faltando):
                saida.append(a)

            while len(saida) > 1 and saida[-1] in self._stop:
                saida.pop()

            resposta = self._render([self._orig(st) for st in saida])
            modo = "token-ancorado"
            self._hebb(list(ancoras) + [w for w in saida if w not in self._stop])
            self._decair()

        # Modulação térmica da voz
        if self._st[0] > 0.7:
            _i, _c = ["Sob pressão, ", "Rancor ativo, "], ["Fim do estresse.", "Normalizando."]
        elif self._st[1] > 0.7:
            _i, _c = ["Em harmonia, ", "Sinergia plena, "], ["A luz brilha.", "Fluxo perfeito."]
        else:
            _i, _c = ["Pela razão, ", "No vácuo, "], ["Aguardando nexo.", "Selado."]

        resposta = resposta.strip()
        if not resposta.endswith((".", "!", "?")):
            resposta += "."
        resposta = resposta[0].upper() + resposta[1:]
        _res = f"{random.choice(_i)}{resposta} {random.choice(_c)}"

        _et = (time.perf_counter() - _t0) * 1e6
        _st_info = f"T:{self._st[0]:.2f}|S:{self._st[1]:.2f}|F:{self._st[2]:.2f}"
        _sn = f"DLM-{modo.upper()}"
        header = (f"\n[DLM-FLOW: {_et:.1f}μs | D:{_final_d}/10 | {_st_info} | "
                  f"{_sn} | SIGN: {self._s}]")
        if _debug:
            return (f"{header}\n{_res}\n"
                    f"  ↳ âncoras: {sorted(ancoras)}\n"
                    f"  ↳ tópico: {topico}\n"
                    f"  ↳ intent: {intent}")
        return f"{header}\n{_res}"


# ======================================================================
# BOOT
# ======================================================================
if __name__ == "__main__":
    import os

    if os.path.exists('model.relacional.txt'):
        with open('model.relacional.txt', 'r', encoding='utf-8') as f:
            conteudo = f.read()
    else:
        conteudo = """
Google ([ˈɡuːɡəl] GOO-ghəl)[7][8] é uma empresa multinacional de softwares e serviços online (baseado na nuvem) fundada em 1998 na cidade norte-americana de Menlo Park (estado da Califórnia), que lucra principalmente através da publicidade pelo AdWords. A Google é a principal subsidiária da Alphabet Inc.

A empresa foi fundada por Larry Page e Sergey Brin, muitas vezes apelidados de "Google Guys",[9][10][11] enquanto os dois estavam frequentando a Universidade Stanford como estudantes de doutoramento. Foi fundada como uma empresa privada em 4 de setembro de 1998 e sua oferta pública inicial foi realizada em 19 de agosto de 2004. A missão declarada da empresa desde o início foi "organizar a informação mundial e torná-la universalmente acessível e útil"[12] e seu slogan oficial era "Não seja mal". Em outubro de 2015, o lema foi substituído no código de conduta corporativo da Alphabet pela frase "Faça a coisa certa".[13] Em 2006, a empresa mudou-se para sua atual sede, em Mountain View, Condado de Santa Clara no estado da Califórnia. O Google é executado através de mais de um milhão de servidores em data centers ao redor do mundo[14] e processa mais de cinco bilhões de solicitações de pesquisa[15] e vinte petabytes de dados gerados por usuários todos os dias.[16][17][18][19]

O rápido crescimento do Google desde sua incorporação culminou em uma cadeia de outros produtos, aquisições e parcerias que vão além do núcleo inicial como motor de buscas. A empresa oferece softwares de produtividade online, como o software de e-mail Gmail, e ferramentas de redes sociais, incluindo o fracassado Google+ e os descontinuados Google Buzz e Orkut. Os produtos do Google se estendem à área de trabalho, com aplicativos como o navegador Google Chrome, o programa de organização de edição de fotografias Picasa e o aplicativo de mensagens instantâneas Google Talk. Notavelmente, o Google também lidera o desenvolvimento do sistema operacional móvel para smartphones Android, usado em celulares de marcas como Samsung, Motorola, LG, HTC, Huawei e Xiaomi.

O antigo ranking Alexa classificou o Google como o website mais visitado do mundo.[20] A Google é classificada pela revista Fortune como o melhor lugar do mundo para se trabalhar. Aparece na posição pelo sexto ano consecutivo[21][22] é a marca mais valiosa do mundo de acordo com o ranking BrandZ de 2017, avaliada em 245 bilhões de dólares.[23] Em outro ranking de avaliação de marcas, ultrapassou em 2014 a Apple, que liderava por três anos consecutivos, com um valor estimado de US$ 159 bilhões.[24] A posição dominante no mercado dos serviços do Google levou a críticas da sociedade sobre assuntos como privacidade, direitos autorais e censura.[25][26] O Google apareceu mais de uma vez no topo da lista da ZeniphOptimedia como o maior conglomerado de mídia do mundo.[27][28]

A Google é do tipo LLC, uma empresa do tipo sociedade de responsabilidade limitada, em caso de processos judiciais o patrimônio dos sócios está protegido.[29]
História

O Google começou em janeiro de 1996 como um projeto de pesquisa de Larry Page e Sergey Brin, quando ambos eram estudantes de doutorado na Universidade Stanford, na Califórnia, Estados Unidos.[30][31][32]

Enquanto os motores de busca convencionais exibiam resultados classificados pela contagem de quantas vezes os termos de busca apareciam na primeira página, os dois teorizaram sobre um sistema melhor que analisava as relações entre os sites.[33] Eles chamaram esta tecnologia de PageRank, onde a relevância de um site era determinada pelo número de páginas, bem como pela importância dessas páginas, que ligavam de volta para o site original.[34][35]
A página original do Google (1998) tinha um desenho simples, já que seus fundadores não tinham experiência em HTML, a linguagem para páginas de web design.[36]

Um pequeno motor de busca chamado "RankDex" da IDD Information Services, projetado por Robin Li, desde 1996, já explorava uma estratégia semelhante para pontuação e classificação de páginas.[37] A tecnologia do RankDex seria patenteada e usada mais tarde por Li, quando fundou a Baidu na China.[38][39]

Larry Page e Sergey Brin, originalmente batizaram sua nova ferramenta de busca de "BackRub", porque o sistema checava backlinks para estimar a importância de um site.[40][41][42]

Meses depois, eles mudaram o nome para o Google, proveniente de um erro ortográfico da palavra "googol",[43][44] o número um seguido por cem zeros, que foi criado para indicar a quantidade de informação que o motor de busca podia processar, o nome também reflete a missão de organizar uma quantidade aparentemente infinita de informações na web.[45][46] Originalmente, o Google funcionou sob o site da Universidade Stanford, com o domínio google.stanford.edu, com os direitos de autor mencionados à universidade no final de sua página à época.[47]

A empresa foi constituída oficialmente em 4 de setembro de 1998[6] e o nome de domínio "Google" foi registrado em 15 de setembro de 1997.[48] No início, sua sede ficava na garagem de uma amiga (Susan Wojcicki)[30] em Menlo Park, Califórnia.[6] Craig Silverstein, um colega de doutorado estudante em Stanford, foi contratado como o primeiro funcionário.[30][49][50] Apesar de ter sido incorporado em 4 de setembro de 1998, desde 2002, o Google comemora seus aniversários em diferentes dias de setembro, mais frequentemente em 27 de setembro.[51][52][53] A mudança nas datas ocorreu para celebrar marcos importantes em conjunto com o aniversário.[54]
Financiamento e oferta pública inicial
A primeira iteração de servidores de produção do Google foi construída com um hardware de baixo custo.[55]

O primeiro financiamento para o Google foi uma contribuição de 100 mil dólares em agosto de 1998 de Andy Bechtolsheim, co-fundador da Sun Microsystems, dada antes do Google ter sido incorporado.[56] No início de 1999, quando ainda eram estudantes de graduação, Larry Page e Sergey Brin decidiram que o motor de busca que eles tinham desenvolvido tomava muito do seu tempo a partir de pesquisas acadêmicas. Eles foram ao CEO da Excite, George Bell, e se ofereceram para comprá-la por 1 milhão de dólares. Ele rejeitou a oferta e, posteriormente, criticou Vinod Khosla, um dos capitalistas de risco da Excite, depois de ter negociado com Brin e Page um valor abaixo de 750 mil dólares. Em 7 de junho de 1999, uma rodada de 25 milhões dólares de financiamento foi anunciada,[57] com os investidores importantes, incluindo as empresas de capital de risco Kleiner Perkins Caufield & Byers e a Sequoia Capital.[56]

A oferta pública inicial (IPO) do Google ocorreu seis anos depois, em 19 de agosto de 2004. A empresa ofereceu 19 605 052 partes a um preço de 85 dólares por ação.[58][59] As ações foram vendidas em um leilão online usando um sistema construído pela Morgan Stanley e Credit Suisse, os subscritores do acordo.[60][61] A venda de 1,67 bilhões dólares deu ao Google uma capitalização de mercado de mais de 23 bilhões de dólares.[62] A grande maioria das 271 milhões ações permaneceram sob o controle do Google e muitos funcionários do Google se tornaram milionários de imediato. Yahoo!, um concorrente do Google, também se beneficiou, pois possuía 8,4 milhões de ações do Google antes da IPO.[63]

Algumas pessoas especularam que a IPO do Google, inevitavelmente, trouxe mudanças na cultura da empresa. Razões variam desde a pressão dos acionistas para a redução de benefícios dos empregados ao fato de que muitos executivos da empresa se tornariam milionários de imediato.[64] Como resposta a esta preocupação, os co-fundadores Sergey Brin e Larry Page, prometeram, em um relatório a investidores potenciais, que a IPO não iria alterar a cultura da companhia.[65] Em 2005, porém, artigos no The New York Times e outras fontes começaram a sugerir que o Google tinha perdido a sua filosofia anticorporativa, sem mal (Don't be evil).[66][67][68] Em um esforço para manter a cultura única da empresa, o Google designou um Escritório Chefe de Cultura, que também trabalha como diretor de Recursos Humanos. O objetivo desse escritório é desenvolver e manter a cultura da companhia e trabalhar em maneiras de manter fiel aos valores em que a empresa foi fundada: uma organização plana, com um ambiente de colaboração.[69] O Google também tem enfrentado acusações de sexismo e de discriminação etária de seus ex-funcionários.[70][71]

O desempenho das ações após a IPO foi bom, com quotas a bater os 700 dólares pela primeira vez em 31 de outubro de 2007,[72] principalmente por causa das fortes vendas e dos lucros no mercado de publicidade online.[73] O aumento no preço das ações foi impulsionado principalmente por investidores individuais, ao contrário de grandes investidores institucionais e fundos mútuos.[73] No início de 2008, a capitalização de mercado da empresa estava acima de US$ 200 bilhões.[74] A empresa está listada na bolsa de valores NASDAQ sob o símbolo GOOG e sob a Bolsa de Valores de Frankfurt com o símbolo GGQ1.
Crescimento
Googleplex, em Mountain View, Califórnia, a sede da empresa.

Em março de 1999, a empresa mudou sua sede para Palo Alto, Califórnia, lar de várias outras importantes startups de tecnologia do Vale do Silício.[75] No ano seguinte, contra a oposição inicial de Page e Brin para um motor de busca financiado por anúncios,[76] o Google começou a vender anúncios associados a palavras-chave de busca.[30] A fim de manter um projeto organizado da página e aumentar a velocidade, as propagandas eram exclusivamente baseadas em texto. Palavras-chave foram vendidas com base em uma combinação de propostas de preços e cliques nos anúncios, com lances a partir de cinco centavos por clique.[30] O pioneiro deste modelo de venda de publicidade por palavra-chave foi o Goto.com, spin-off do Idealab, criado por Bill Gross.[77][78] Quando a empresa mudou de nome para Overture Services, processou o Google por alegadas violações das patentes do pay-per-click e de licitações. A Overture Services viria a ser comprado pelo Yahoo! e renomeado Yahoo Search Marketing. O caso foi então resolvido fora do tribunal, concordando com o Google para a emissão de ações ordinárias para o Yahoo! em troca de uma licença perpétua.[79]

Durante este tempo, o Google conseguiu uma patente descrevendo seu mecanismo de PageRank.[80] A patente foi oficialmente atribuída a Universidade de Stanford e classificou Lawrence Page como seu inventor. Em 2003, após superando dois outros locais, a empresa arrendou seu atual complexo da Silicon Graphics na 1600 Amphitheatre Parkway, em Mountain View, Califórnia.[81] O complexo tem sido, desde então, conhecido como o Googleplex, uma brincadeira com a palavra googolplex, o número um seguido de um googol zeros. Três anos depois, o Google iria comprar a propriedade da SGI por 319 milhões de dólares.[82] Nessa época, o nome "Google" encontrou seu caminho na linguagem cotidiana, fazendo com que o verbo "google" fosse adicionado ao Merriam Webster Collegiate Dictionary e ao Oxford English Dictionary, cujo significado era "usar o motor de busca Google para obter informações na Internet."[83][84]
Aquisições e parcerias

Desde 2001, o Google adquiriu várias empresas, com destaque para pequenas empresas de capital de risco. Em 2004, o Google adquiriu a Keyhole, Inc.[85] A empresa start-up desenvolveu um produto chamado Earth Viewer, que dava uma visão 3-D da Terra. O Google renomeou o serviço para Google Earth, em 2005. Em 13 de abril de 2007, o Google chegou a um acordo para adquirir a DoubleClick por 3,1 bilhões de dólares, dando ao Google relacionamentos valiosos que a DoubleClick teve com os editores da web e agências de publicidade.[86] Mais tarde, naquele mesmo ano, o Google adquiriu a GrandCentral por 50 milhões de dólares.[87] O site mais tarde seria alterado para Google Voice. Em 5 de agosto de 2009, o Google comprou a sua primeira empresa pública, com a compra da fabricante de softwares de vídeo On2 Technologies por 106,5 milhões de dólares.[88] O Google também adquiriu a Aardvark, um motor de busca de redes sociais, por 50 milhões de dólares. Google comentou em seu blog interno, "estamos ansiosos para colaborar para ver onde podemos ir."[89] E, em abril de 2010, o Google anunciou que tinha adquirido uma start-up de hardware, a Agnilux.[90]

Além das inúmeras empresas que o Google comprou, a empresa firmou parceria com outras organizações para tudo, desde pesquisa à publicidade. Em 2005, o Google fez uma parceria com o NASA Ames Research Center para construir 93 000 metros quadrados de escritórios.[91] Os serviços seriam usados para projetos de pesquisa envolvendo gestão de dados em grande escala, nanotecnologia, computação distribuída e indústria espacial empresarial. Mais tarde naquele ano, o Google firmou uma parceria com a Sun Microsystems, em outubro de 2005 para ajudar a compartilhar e distribuir outras tecnologias.[92] A empresa também fez uma parceria com a AOL, da Time Warner,[93] para aumentar outros serviços de busca de vídeo. As parcerias do Google em 2005 também incluiu o novo financiamento do domínio de topo .mobi para dispositivos móveis, juntamente com outras empresas, incluindo a Microsoft, Nokia e Ericsson.[94] O Google, mais tarde, lançou o "AdSense for Mobile", aproveitando o mercado emergente de publicidade móvel.[95] Ampliando sua publicidade para chegar ainda mais longe, o Google e a Fox Interactive Media, da News Corp, entraram em um acordo de 900 milhões de dólares para fornecer a busca e publicidade no popular site de redes sociais MySpace.[96]
Sede do YouTube em San Bruno, Califórnia. A empresa foi adquirida pela Google em outubro de 2006.[97]

Em outubro de 2006, o Google anunciou que havia adquirido o site de compartilhamento de vídeos YouTube por 1,65 bilhão de dólares em ações do Google e o negócio foi concluído em 13 de novembro de 2006.[98][97] O Google não oferece números detalhados para os custos de funcionamento do YouTube e as receitas do YouTube em 2007 foram anotadas como "não materiais" em um arquivamento regulador.[99] A compra do site fez a empresa encerrar o Google Video. Em junho de 2008, um artigo da revista Forbes projetou a receita do YouTube em 200 milhões de dólares para 2008, registrando progressos na venda de publicidade.[100] Em 2007, o Google começou a patrocinar o NORAD Tracks Santa, um serviço que pretende acompanhar o progresso do -Papai Noel na véspera de Natal,[101] usando o Google Earth para "acompanhar o Papai Noel", pela primeira vez, em 3-D,[102] e deslocando a ex-patrocinadora da AOL. O YouTube criou um canal de vídeos para o NORAD Tracks Santa.[103]

Em 2008, o Google desenvolveu uma parceria com a GeoEye para lançar um satélite que fornece ao Google imagens com alta resolução (0,41 m monocromáticas, a cores 1,65 m) para o software Google Earth. O satélite foi lançado da Base da Força Aérea de Vandenberg em 6 de setembro de 2008.[104] O Google também anunciou em 2008 que estava hospedando um arquivo de fotografias da revista Life como parte de sua mais recente parceria. Algumas das imagens no arquivo nunca foram publicados na revista.[105] As fotos foram filigrana e originalmente havia postado avisos de direitos autorais em todas as fotos, independentemente do status de domínio público.[106]

Em 2010, o Google Energy fez seu primeiro investimento em um projeto de energia renovável, a colocação de 38,8 milhões dólares em dois parques eólicos na Dakota do Norte. A companhia anunciou que os dois locais vão gerar 169,5 megawatts de potência, ou o suficiente para abastecer 55 mil casas. As fazendas, que foram desenvolvidos pela NextEra Energy Resources, vai reduzir o uso de combustíveis fósseis na região. NextEra Energy Resources vendeu ao Google uma participação de 20% do projeto, a fim de obter financiamento para o desenvolvimento do projeto.[107] Também em 2010, o Google comprou a Global IP Solutions, uma empresa baseada na Noruega, que prevê teleconferência baseada na web e outros serviços relacionados. Esta aquisição permitirá à Google incluir serviços de telefonia à sua lista de produtos.[108] Em 27 de maio de 2010, o Google anunciou que também fechou a aquisição da rede de publicidade móvel AdMob. Essa compra ocorreu dias após a Federal Trade Commission encerrar a sua investigação sobre a compra.[109] O Google adquiriu a empresa por uma quantia não revelada.[110] Em julho de 2010, o Google assinou um acordo com um parque eólico de Iowa para comprar 114 megawatts de energia para 20 anos.[111]

Em 2012, o Google adquiriu a Motorola com o principal objetivo de absorver suas patentes pagando 12,5 bilhões  * de dólares pela empresa.[112] E em 29 de janeiro de 2014 a empresa vendeu Motorola Mobility para a marca chinesa Lenovo por 2,91 bilhões  * de dólares.[113]

Em maio de 2013, o Google inicia divulgação de um novo serviço, o Timelapse, que possui praticamente as mesmas funções do Google Earth, porém mostra a visão de satélite de forma cronológica no período entre 1984 e 2012.[114] No início de 2014, a empresa adquiriu por 3,2 bilhões  * de dólares, a Nest Labs, empresa desenvolvedora de alarmes e termostatos inteligentes.[115][116][117]

Em janeiro de 2018, a empresa finalizou o acordo, iniciado em setembro de 2017,[118][119] de aquisição da divisão de celulares da taiwanêsa HTC por $1,1 bilhão de dólares.[120][121]
Logo da Alphabet Inc
Mudanças na gestão e criação da Alphabet Inc
Ver artigo principal: Alphabet Inc.

Em 20 de janeiro de 2011, a empresa anunciou que Larry Page se tornaria o novo CEO a partir de 4 de abril. Eric Schmidt, deixaria o cargo depois de 10 anos para assumir a diretoria executiva, concentrando-se principalmente em parcerias e assuntos governamentais. Sergey Brin passou a cuidar de projetos estratégicos e se tornou o responsável pelos novos produtos da empresa.[122]

Em 2011 um acionista processou a empresa, alegando que estava permitindo que farmácias canadenses veiculassem anúncios de remédios que precisam de prescrição.[123]

Em 10 de agosto de 2015, a empresa reorganizou suas diversas áreas em uma holding, a Alphabet Inc., que tem a Google Inc. como principal subsidiária.[124][125] Como parte da reestruturação, Sundar Pichai foi promovido ao cargo de CEO da Google.[126]

Em 1 de setembro de 2017, a Google Inc. anunciou seus planos de reestruturação como uma companhia de responsabilidade limitada, mudando para Google LLC, como uma subsidiária integral da XXVI Holdings Inc., que é formada como uma subsidiária da Alphabet Inc. para deter o patrimônio de sua empresa outras subsidiárias, incluindo o Google LLC e outras apostas[127][128]

A empresa do tipo LLC (do inglês Limited Liability Company) é uma modalidade de empresa do tipo sociedade de responsabilidade limitada; em caso de dívidas ou processos judiciais contra ela, o patrimônio pessoal dos sócios/acionistas tem uma proteção legal e não está em risco.

        """

    quantikus = QuintikusAGI(hashing=False)
    quantikus.inicializar(conteudo)

    print(f"[frases: {len(quantikus._sentencas_raw)} | vocab: {len(quantikus._freq)} | "
          f"assoc-arestas: {sum(len(v) for v in quantikus._assoc.values())}]")
    print()

    for p in [
        "Em que data o Google foi oficialmente constituído?",
        "Quando o domínio Google foi registrado?",
        "O que é PageRank",
        "Qual era o nome original da ferramenta de busca",
        "Onde o Google funcionou originalmente sob domínio da universidade?",
        "Qual era o nome original da ferramenta de busca?",
    ]:
        print(f">>> {p}")
        print(quantikus.falar(p, _debug=True))
        print()
