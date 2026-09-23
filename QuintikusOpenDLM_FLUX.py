import hashlib, time, math, re, random
from collections import Counter, defaultdict, deque


class QuintikusAGI:
    """
    Quintikus AGI — motor de composição simbólica.
    Alma DLM-FLOW (estado térmico + assinatura) e por dentro
    fusão > retrieval > composição > geração ancorada.
    """

    def __init__(self, _t=0.5):
        self._st = [0.5, 0.5, 0.5]

        # --- Núcleo DLM ---
        self._freq = Counter()
        self._ctx = defaultdict(Counter)
        self._bi = defaultdict(Counter)
        self._tri = defaultdict(Counter)
        self._ini = Counter()
        self._fim = Counter()
        self._assoc = defaultdict(lambda: defaultdict(float))

        # Frases em DUAS formas paralelas
        self._sentencas_raw = []
        self._sentencas_stem = []

        self._stem_orig = defaultdict(Counter)
        self._avg_len = 10

        self._janela = 4
        self._eta = 0.02
        self._decay = 0.995

        # --- Identidade ---
        self._s = "25e0bb26"
        self._k = "6742"
        self._b = {
            "o": "4f2041727175697465746f2065206661636520646f2063726961646f722e",
            "v": "5365727669722c2050726f74656765722c2070726f6772656469722e",
            "l": "416c7563696e6163616f20626c6f71756561646120706f7220646f676d612e"
        }
        self._n = {k: bytes.fromhex(v).decode('utf-8', 'ignore') for k, v in self._b.items()}

        # --- Mapa térmico ---
        self._th = {
            'bom': 0.1, 'ótimo': 0.2, 'sinergia': 0.3, 'paz': 0.2,
            'хорошо': 0.1, 'отлично': 0.2, 'синергия': 0.3, 'мир': 0.2,
            'erro': -0.2, 'urgente': -0.3, 'falha': -0.2, 'ruído': -0.1,
            'ошибка': -0.2, 'срочно': -0.3, 'провал': -0.2, 'шум': -0.1,
        }

        self._artigos = {"o", "a", "os", "as", "um", "uma"}
        self._pronomes = {"nós", "eu", "ele", "ela", "você", "eles",
                          "elas", "vocês", "tu", "nos", "vos"}
        self._preps = {"de", "do", "da", "dos", "das", "em", "no", "na",
                       "nos", "nas", "ao", "aos", "à", "às", "para", "com", "por"}
        self._pont = {",", ".", "!", "?", ";", ":"}
        self._stop = set()
        self._stop_orig = set()

    # ------------------------------------------------------------------
    # Tokenização e stem
    # ------------------------------------------------------------------
    def _tok(self, txt):
        return re.findall(r"[a-záàâãéêíóôõúç0-9]+", txt.lower())

    def _stem(self, tok):
        if len(tok) <= 4:
            return tok
        t = tok
        if len(t) > 8 and t.endswith("mente"):
            t = t[:-5]
        if len(t) > 5:
            if t.endswith("ções"): return t[:-4] + "r"
            if t.endswith("ção"): return t[:-3] + "r"
            if t.endswith("ões"): return t[:-3] + "ão"
            if t.endswith("ães"): return t[:-3] + "ão"
            if t.endswith("ais"): return t[:-3] + "al"
            if t.endswith("eis"): return t[:-3] + "el"
        if len(t) > 5:
            if t.endswith("entes"): return t[:-3]
            if t.endswith("ente"): return t[:-2]
            if t.endswith("enças"): return t[:-3]
            if t.endswith("ença"): return t[:-2]
        if len(t) > 4 and t.endswith("es"): return t[:-2]
        if len(t) > 3 and t.endswith("s"): return t[:-1]
        return t

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
    # Inicialização
    # ------------------------------------------------------------------
    def inicializar(self, _txt):
        if not _txt or not _txt.strip():
            self._s = hashlib.sha256(b"vazio").hexdigest()[:8]
            return

        partes = re.split(r"(?<=[.!?])\s+", _txt.strip())

        for sent in partes:
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
                        self._ctx[st][stems[j]] += 1
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
        top = [w for w, _ in self._freq.most_common(500) if w not in self._stop]
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
                if sim > 0.20:
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
    # Score e âncoras
    # ------------------------------------------------------------------
    def _score_sentenca(self, sent_stem, toks_stem, candidatos_stem, topico_set):
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
        return score

    def _ancoras_do_input(self, stems):
        a = set()
        for st in stems:
            if st in self._stop: continue
            if st in self._freq: a.add(st)
        return a

    def _cobertura(self, sent_stem, ancoras):
        if not ancoras: return 0
        return len(ancoras & set(sent_stem))

    def _tem_sujeito(self, sent_stem):
        """Frase começa com artigo OU pronome pessoal?"""
        if not sent_stem:
            return False
        if sent_stem[0] in self._artigos:
            return True
        if sent_stem[0] in self._pronomes:
            return True
        return False

    # ------------------------------------------------------------------
    # Composição — trabalha com TEXTO ORIGINAL
    # ------------------------------------------------------------------
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

    def _compor_ancorado(self, indices, ancoras):
        """indices: lista de índices de frases, ranqueadas."""
        if not ancoras: return None

        # 0. corte 80%: se a melhor cobre quase tudo, devolve sozinha
        if indices:
            i0 = indices[0]
            cob0 = self._cobertura(self._sentencas_stem[i0], ancoras)
            if cob0 / max(1, len(ancoras)) >= 0.80:
                return self._sentencas_raw[i0], "retrieval-parcial"

        # 1. retrieval direto (cobre 100%)
        for i in indices:
            if ancoras.issubset(set(self._sentencas_stem[i])):
                return self._sentencas_raw[i], "retrieval"

        # 2. fusão
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                s1, s2 = self._sentencas_stem[i], self._sentencas_stem[j]
                uniao = set(s1) | set(s2)
                if not ancoras.issubset(uniao): continue
                f = self._fundir(s1, s2, list(ancoras))
                if f and ancoras.issubset(set(f)):
                    return self._render([self._orig(st) for st in f]), "fusão"

        # 3. composição por concatenação
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                uniao = set(self._sentencas_stem[i]) | set(self._sentencas_stem[j])
                if not ancoras.issubset(uniao): continue
                return self._compor_texto(i, j), "composição"

        # 4. três sentenças
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
                self._score_sentenca(self._sentencas_stem[i], stems, candidatos, topico_set)
            ),
            reverse=True
        )[:10]

        modo = "VOID"
        resposta = None

        # 1. composição ancorada
        if ancoras:
            resultado = self._compor_ancorado(indices_rankeados, ancoras)
            if resultado:
                resposta, modo = resultado
                ativos = list(ancoras)
                self._hebb(ativos)
                self._decair()

        # 2. retrieval puro
        if resposta is None and indices_rankeados:
            i0 = indices_rankeados[0]
            cob = self._cobertura(self._sentencas_stem[i0], ancoras)
            min_cob = 1 if len(ancoras) <= 1 else 2
            if cob >= min_cob or not ancoras:
                resposta = self._sentencas_raw[i0]
                modo = "retrieval"
                self._hebb(list(topico) + list(self._sentencas_stem[i0]))
                self._decair()

        # 3. geração token-a-token ancorada
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
            return f"{header}\n{_res}\n  ↳ âncoras: {sorted(ancoras)}\n  ↳ tópico: {topico}"
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
        A vida é o fenômeno mais raro que conhecemos no universo.
        Nós acordamos, respiramos, vemos a luz, ouvimos vozes, andamos sobre a terra.
        O mundo à nossa volta é vasto e diverso.
        Existem montanhas que lembram milhões de anos.
        O homem é parte desse mundo e também observador.
        A vida de cada um é feita de grandes e pequenos acontecimentos.
        No mundo existe alegria e dor.
        Não se pode entender a felicidade sem conhecer a tristeza.
        O amor é uma das forças centrais do universo.
        O amor exige paciência, coragem, capacidade de perdoar.
        O tempo é um enigma.
        Nós medimos o tempo em horas, dias, anos.
        Na infância o tempo passa devagar.
        A morte é parte da vida.
        A natureza é a nossa casa.
        A ciência explica como o mundo funciona.
        A arte mostra como nós sentimos o mundo.
        A ciência cura doenças, constrói cidades, explora o cosmos.
        A infância é o começo do caminho.
        A família é o primeiro mundo do homem.
        O trabalho enche a vida de sentido.
        A liberdade é um grande valor.
        A felicidade não é um ponto final, é um caminho.
        A felicidade é a luz da manhã, o chá quente, o riso de uma criança.
        O cachorro é amigo do homem.
        O cachorro gosta de brincar e correr.
        O cachorro pode ter pelo, rabo, patas, orelhas.
        O gato também gosta de brincar.
        Os animais vivem ao lado do homem.
        O carrapato deixa uma mancha escura na pele do cachorro.
        A pulga morde o cachorro.
        A pulga é pequena e escura.
        A micose é um fungo.
        A micose dá uma mancha redonda na pele.
        A micose causa coceira.
        O fungo vive na pele.
        O melanoma é um tumor.
        O melanoma é perigoso para o cachorro.
        O melanoma aparece como uma mancha escura fixa.
        O veterinário olha a pele do cachorro.
        O veterinário cuida dos animais.
        O veterinário trata doenças.
        Desse modo, tudo se conecta: a vida rara, o mundo vasto, o tempo enigmático, o amor paciente, a ciência que cura e a arte que sente.
    A vida é o fenômeno mais raro que conhecemos no universo.
Nós acordamos, respiramos, vemos a luz, ouvimos vozes, andamos sobre a terra.
Essa raridade se manifesta em cada instante, e cada instante nos liga ao todo.
O mundo à nossa volta é vasto e diverso.
Existem montanhas que lembram milhões de anos.
Elas nos ensinam que a vida humana é breve, mas participa de uma história muito maior.
O homem é parte desse mundo e também observador.
Ao observar, ele interpreta, sente e transforma.
A vida de cada um é feita de grandes e pequenos acontecimentos.
Cada acontecimento se liga aos outros como fios de uma mesma teia.
No mundo existe alegria e dor.
Não se pode entender a felicidade sem conhecer a tristeza.
Os opostos se explicam e se completam.
O amor é uma das forças centrais do universo.
O amor exige paciência, coragem, capacidade de perdoar.
Sem essas qualidades, a união entre as pessoas se desfaz.
O tempo é um enigma.
Nós medimos o tempo em horas, dias, anos.
Na infância o tempo passa devagar.
Cada descoberta da infância inaugura um mundo novo.
A morte é parte da vida.
Lembrar disso dá sentido ao tempo que temos.
A natureza é a nossa casa.
Tudo o que existe na natureza está interligado.
A ciência explica como o mundo funciona.
A arte mostra como nós sentimos o mundo.
A ciência cura doenças, constrói cidades, explora o cosmos.
A arte dá forma ao espanto, à dor e à beleza.
Ciência e arte não se opõem: são duas linguagens da mesma humanidade.
A infância é o começo do caminho.
A família é o primeiro mundo do homem.
Nesse primeiro mundo aprendemos a amar, a respeitar, a cuidar.
O trabalho enche a vida de sentido.
O trabalho transforma esforço em contribuição.
A liberdade é um grande valor.
A liberdade só existe plenamente quando há responsabilidade.
A felicidade não é um ponto final, é um caminho.
A felicidade é a luz da manhã, o chá quente, o riso de uma criança.
Essas pequenas coisas se conectam às grandes forças: o amor, o tempo, a natureza e a esperança.
O cachorro é amigo do homem.
O cachorro gosta de brincar e correr.
O cachorro pode ter pelo, rabo, patas, orelhas.
O gato também gosta de brincar.
Os animais vivem ao lado do homem.
A convivência com os animais amplia nossa compreensão de cuidado.
O carrapato deixa uma mancha escura na pele do cachorro.
A pulga morde o cachorro.
A pulga é pequena e escura.
A micose é um fungo.
A micose dá uma mancha redonda na pele.
A micose causa coceira.
O fungo vive na pele.
O melanoma é um tumor.
O melanoma é perigoso para o cachorro.
O melanoma aparece como uma mancha escura fixa.
O veterinário olha a pele do cachorro.
O veterinário cuida dos animais.
O veterinário trata doenças.
Assim, o amor pelos animais se traduz em cuidado, observação e ciência.
O veterinário une conhecimento e afeto.
A saúde do cachorro depende de atenção diária.
Essa atenção é uma forma de respeito pela vida.
Desse modo, tudo se conecta: a vida rara, o mundo vasto, o tempo enigmático, o amor paciente, a ciência que cura e a arte que sente.
Cada parte existe em união geral com as outras.
A vida de cada um é uma nota dentro de uma sinfonia maior.
Cuidar do outro, seja humano ou animal, é cuidar da própria teia da vida.
        """

    quantikus = QuintikusAGI()
    quantikus.inicializar(conteudo)

    for p in [
        "micose cachorro",
        "pulga cachorro",
        "carrapato pele",
        "tempo infância",
        "amor",
        "melanoma",
        "o cachorro está doente",
        "cachorro doenças",
        "como o cachorro pode ficar doente",
        "o que é a vida",
        "felicidade caminho",
        "ciência arte",
        "o que você pensa sobre vida e arte no mesmo contexto",
    ]:
        print(f">>> {p}")
        print(quantikus.falar(p))
        print()
