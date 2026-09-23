import hashlib, math, re, random
from collections import Counter, defaultdict, deque


class DLM:
    def __init__(self, janela=4, eta=0.02, decay=0.995):
        self._janela = janela
        self._eta = eta
        self._decay = decay

        self._freq = Counter()
        self._ctx  = defaultdict(Counter)
        self._bi   = defaultdict(Counter)
        self._tri  = defaultdict(Counter)
        self._ini  = Counter()
        self._fim  = Counter()
        self._assoc = defaultdict(lambda: defaultdict(float))

        self._sentencas = []
        self._avg_len = 10
        self._st = [0.5, 0.5, 0.5]
        self._stop = set()
        self._artigos = {"o","a","os","as","um","uma"}
        self._preps = {"de","do","da","dos","das","em","no","na","nos","nas",
                       "ao","aos","à","às","para","com","por"}

    # ------------------------------------------------------------------
    def _tok(self, txt):
        return re.findall(r"[a-záàâãéêíóôõúç0-9]+", txt.lower())

    # ------------------------------------------------------------------
    # Aprendizado
    # ------------------------------------------------------------------
    def aprender(self, texto):
        lens = []
        for sent in re.split(r"[.!?]+", texto):
            toks = self._tok(sent)
            if len(toks) < 2:
                continue
            self._sentencas.append(tuple(toks))
            lens.append(len(toks))
            self._ini[toks[0]] += 1
            self._fim[toks[-1]] += 1
            for i, t in enumerate(toks):
                self._freq[t] += 1
                ini, fim = max(0, i - self._janela), min(len(toks), i + self._janela + 1)
                for j in range(ini, fim):
                    if i != j:
                        self._ctx[t][toks[j]] += 1
                if i + 1 < len(toks):
                    self._bi[t][toks[i + 1]] += 1
                if i + 2 < len(toks):
                    self._tri[(t, toks[i + 1])][toks[i + 2]] += 1
        if lens:
            self._avg_len = sum(lens) / len(lens)

    def consolidar(self):
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
        self._stop = fixas | {w for w, _ in ordenado[:corte] if len(w) <= 3}
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
    # Fusão
    # ------------------------------------------------------------------
    def _split_at_anchor(self, toks, anchor):
        try:
            idx = toks.index(anchor)
        except ValueError:
            return None
        start = idx
        if idx > 0 and toks[idx - 1] in self._artigos:
            start = idx - 1
        prefix = list(toks[:start])
        anchor_block = list(toks[start:idx + 1])
        j = idx + 1
        middle = []
        while j < len(toks) and toks[j] not in self._preps:
            middle.append(toks[j])
            j += 1
        suffix = list(toks[j:])
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
                i1 = s1.index(a)
                i2 = s2.index(a)
            except ValueError:
                continue
            central = min(i1, len(s1) - i1 - 1) + min(i2, len(s2) - i2 - 1)
            bonus = 0
            if i1 > 0 and s1[i1 - 1] in self._artigos:
                bonus += 1
            if i2 > 0 and s2[i2 - 1] in self._artigos:
                bonus += 1
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
        """Verifica se a junção middle+suffix tem base associativa."""
        ult_mid = self._ultimo_content(middle)
        pri_suf = self._primeiro_content(suffix)
        if not ult_mid or not pri_suf:
            return True  # sem conteúdo dos dois lados, não bloqueia
        sim = self._assoc.get(ult_mid, {}).get(pri_suf, 0.0)
        if sim == 0.0:
            sim = self._assoc.get(pri_suf, {}).get(ult_mid, 0.0)
        if sim == 0.0:
            if self._ctx.get(ult_mid, {}).get(pri_suf, 0) > 0:
                return True
            if self._ctx.get(pri_suf, {}).get(ult_mid, 0) > 0:
                return True
        return sim >= 0.15

    def _fundir(self, s1, s2, topico):
        # CORREÇÃO 1: fusão exige 2+ palavras de tópico
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

        if not suf1 and not suf2:
            return None
        if not pre1 and not pre2:
            return None

        topico_set = set(topico)
        m_pre1 = len(set(pre1) & topico_set)
        m_pre2 = len(set(pre2) & topico_set)
        m_suf1 = len(set(suf1) & topico_set)
        m_suf2 = len(set(suf2) & topico_set)

        if m_pre1 > m_pre2:
            pre_src, prefix, anc, middle = 1, pre1, anc1, mid1
            suf_orig = suf1
        elif m_pre2 > m_pre1:
            pre_src, prefix, anc, middle = 2, pre2, anc2, mid2
            suf_orig = suf2
        else:
            pre_src = 1 if m_suf2 >= m_suf1 else 2
            if pre_src == 1:
                prefix, anc, middle = pre1, anc1, mid1
                suf_orig = suf1
            else:
                prefix, anc, middle = pre2, anc2, mid2
                suf_orig = suf2

        if m_suf1 > m_suf2:
            suffix, suf_src = suf1, 1
        elif m_suf2 > m_suf1:
            suffix, suf_src = suf2, 2
        else:
            if suf1 and not suf2:
                suffix, suf_src = suf1, 1
            elif suf2 and not suf1:
                suffix, suf_src = suf2, 2
            else:
                suffix, suf_src = (suf2, 2) if pre_src == 1 else (suf1, 1)

        if pre_src == suf_src:
            return None
        if not suffix:
            return None
        if prefix and suffix and prefix[-1] in self._artigos and suffix[0] in self._artigos:
            return None

        # CORREÇÃO 2: junção tem que ter base associativa
        # só exige quando o sufixo novo é DIFERENTE do natural da frase do prefixo
        if suffix != suf_orig:
            if not self._juncao_valida(middle, suffix):
                return None
        else:
            if not self._juncao_valida(middle, suffix):
                return None

        nova = prefix + anc + middle + suffix
        if len(nova) < 4:
            return None
        nova_set = set(nova)
        if any(t not in nova_set for t in topico):
            return None
        return tuple(nova)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def _score_sentenca(self, sent, toks_input, candidatos):
        sent_set = set(sent)
        score = 0.0
        for t in toks_input:
            if t in sent_set:
                score += 3.0
            if t in self._assoc:
                for u in sent_set:
                    if u in self._assoc[t]:
                        score += self._assoc[t][u] * 0.5
        for c, _ in candidatos.most_common(5):
            if c in sent_set:
                score += 0.3
        return score

    # ------------------------------------------------------------------
    # Tensão, viés, geração token-a-token
    # ------------------------------------------------------------------
    def _tensao_par(self, par, ancora, topico):
        a, b = par
        if (a is None or a in self._stop) and b in self._stop:
            return None
        if b in self._stop:
            return None
        sims_anc, sims_top = [], []
        for x in (a, b):
            if x and x not in self._stop:
                sims_anc.append(max((self._assoc[x].get(p, 0.0) for p in ancora), default=0.0))
                sims_top.append(max((self._assoc[x].get(p, 0.0) for p in topico), default=0.0))
        if not sims_anc and not sims_top:
            return None
        s_anc = (0.7 * max(sims_anc) + 0.3 * (sum(sims_anc) / len(sims_anc))) if sims_anc else 0.0
        s_top = (0.7 * max(sims_top) + 0.3 * (sum(sims_top) / len(sims_top))) if sims_top else 0.0
        return 0.6 * s_anc + 0.4 * s_top

    def _vies(self, token, contexto, ancora, topico):
        if token in self._stop:
            return 0.0, 1.0, 0.0
        local = 0.0
        if len(contexto) >= 2:
            chave = (contexto[-2], contexto[-1])
            local += self._tri.get(chave, {}).get(token, 0) * 4.0
        if contexto:
            local += self._bi.get(contexto[-1], {}).get(token, 0) * 1.5
        local = min(1.0, local / 10.0)
        s_anc = max((self._assoc[x].get(token, 0.0) for x in ancora), default=0.0)
        s_top = max((self._assoc[x].get(token, 0.0) for x in topico), default=0.0)
        global_b = 0.6 * s_anc + 0.4 * s_top
        return local, global_b, local - global_b

    def _reancorar(self, ancora_orig, saida, topico):
        nova = set(ancora_orig)
        for t in saida[-4:]:
            if t in self._stop:
                continue
            if max((self._assoc[t].get(p, 0.0) for p in topico), default=0.0) > 0.30:
                nova.add(t)
        for a in list(nova):
            if a in topico:
                continue
            if max((self._assoc[a].get(p, 0.0) for p in topico), default=0.0) < 0.20:
                nova.discard(a)
        return nova

    def _prox_token(self, contexto, ancora, topico, temp):
        cands = Counter()
        if len(contexto) >= 2:
            chave = (contexto[-2], contexto[-1])
            peso_tri = 4.0
            if contexto[-2] in self._stop and contexto[-1] in self._stop:
                peso_tri = 0.8
            for w, c in self._tri.get(chave, {}).items():
                cands[w] += c * peso_tri
        if contexto:
            for w, c in self._bi.get(contexto[-1], {}).items():
                cands[w] += c * 1.5
        if not cands:
            return None

        for w in list(cands):
            l, g, d = self._vies(w, contexto, ancora, topico)
            cands[w] *= (1.0 + g * 2.0)
            if d > 0.35 and l > 0.4:
                if g < 0.10:
                    cands[w] *= 0.02
                else:
                    cands[w] *= max(0.15, 1.0 - d)

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
            if r <= acc:
                return w
        return itens[-1][0]

    def _upd_thermal(self, toks):
        dens_stop = sum(1 for t in toks if t in self._stop) / max(1, len(toks))
        variedade = len(set(toks)) / max(1, len(toks))
        conhecidos = sum(1 for t in toks if t in self._freq) / max(1, len(toks))
        self._st[0] = max(0, min(1, self._st[0] * 0.9 + dens_stop * 0.3))
        self._st[1] = max(0, min(1, self._st[1] * 0.9 + variedade * 0.3))
        self._st[2] = max(0, min(1, self._st[2] * 0.9 + conhecidos * 0.3))

    def _hebb(self, ativos):
        ativos = [a for a in ativos if a not in self._stop]
        for i in range(len(ativos)):
            for j in range(i + 1, len(ativos)):
                a, b = ativos[i], ativos[j]
                if a == b:
                    continue
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
    # Fala — FUSÃO > RETRIEVAL > TOKEN
    # ------------------------------------------------------------------
    def falar(self, pergunta, debug=False):
        toks = self._tok(pergunta)
        if not toks:
            return ""
        self._upd_thermal(toks)

        topico = [t for t in toks if t in self._freq and t not in self._stop]
        if not topico:
            return "Ainda não tenho base para responder sobre isso."

        max_tokens = max(6, int(self._avg_len * 1.4))
        temp = max(0.6, 1.3 - self._st[2] * 0.6)

        candidatos = Counter()
        for t in topico:
            for u, s in self._assoc.get(t, {}).items():
                candidatos[u] += s

        pontuadas = sorted(
            self._sentencas,
            key=lambda s: self._score_sentenca(s, toks, candidatos),
            reverse=True
        )[:3]

        # --- 1. TENTA FUSÃO ---
        fusao = None
        fusao_par = None
        if len(pontuadas) >= 2:
            pares = [(pontuadas[0], pontuadas[1])]
            if len(pontuadas) >= 3:
                pares.append((pontuadas[0], pontuadas[2]))
            for par in pares:
                f = self._fundir(par[0], par[1], topico)
                if f:
                    fusao = f
                    fusao_par = par
                    break

        if fusao:
            texto = " ".join(fusao)
            ativos = topico + [w for w in fusao if w not in self._stop]
            self._hebb(ativos)
            self._decair()
            texto = texto[0].upper() + texto[1:] + "."
            if debug:
                return f"{texto}\n[modo: fusão | tópico: {topico} | {' '.join(fusao_par[0])} + {' '.join(fusao_par[1])}]"
            return texto

        # --- 2. RETRIEVAL ---
        if pontuadas:
            melhor = pontuadas[0]
            score_melhor = self._score_sentenca(melhor, toks, candidatos)
            if score_melhor >= 3.0:
                texto = " ".join(melhor)
                ativos = topico + [w for w in melhor if w not in self._stop]
                self._hebb(ativos)
                self._decair()
                texto = texto[0].upper() + texto[1:] + "."
                if debug:
                    return f"{texto}\n[modo: retrieval | score: {score_melhor:.1f} | tópico: {topico}]"
                return texto

        # --- 3. FALLBACK: token-a-token ---
        semente = topico[0]
        ancora = set(topico)
        saida = [semente]
        cache_tensao = deque(maxlen=8)
        reancoragens = 0
        drift_consec = 0
        diverg_consec = 0
        stop_run = 0
        MAX_STOP_RUN = 2

        for passo in range(max_tokens):
            prox = self._prox_token(saida, ancora, topico, temp)
            if prox is None:
                break

            if prox in self._stop:
                stop_run += 1
                if stop_run > MAX_STOP_RUN:
                    break
            else:
                stop_run = 0

            par = (saida[-1] if saida else None, prox)
            t = self._tensao_par(par, ancora, topico)
            if t is not None:
                cache_tensao.append((passo, t))

            l, g, d = self._vies(prox, saida, ancora, topico)

            if t is not None and t < 0.15 and passo >= 3:
                break

            slope = 0.0
            if len(cache_tensao) >= 3:
                n = len(cache_tensao)
                xs = list(range(n))
                ys = [v for _, v in cache_tensao]
                mx, my = sum(xs)/n, sum(ys)/n
                num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
                den = sum((xs[i]-mx)**2 for i in range(n))
                slope = num/den if den else 0.0

            if slope < -0.03:
                drift_consec += 1
            else:
                drift_consec = 0

            if l > 0.4 and d > 0.45:
                diverg_consec += 1
            else:
                diverg_consec = 0

            if diverg_consec >= 3:
                break

            if drift_consec >= 2:
                nova = self._reancorar(ancora, saida, topico)
                if nova != ancora and reancoragens < 3:
                    ancora = nova
                    reancoragens += 1
                    drift_consec = 0
                    cache_tensao.clear()
                else:
                    break

            saida.append(prox)

            if len(saida) >= max_tokens:
                break

            if prox in self._fim and self._fim[prox] >= 2 and len(saida) > 4:
                if random.random() < 0.7:
                    break

        while len(saida) > 1 and saida[-1] in self._stop:
            saida.pop()

        texto = " ".join(saida)
        ativos = topico + [w for w in saida if w not in self._stop]
        self._hebb(ativos)
        self._decair()
        texto = texto[0].upper() + texto[1:] + "."
        if debug:
            return f"{texto}\n[modo: token | tópico: {topico}]"
        return texto


# ======================================================================
# CORPUS E DEMO
# ======================================================================
if __name__ == "__main__":
    texto = """
    A vida é o fenômeno mais raro que conhecemos no universo.
    Nós acordamos, respiramos, vemos a luz, ouvimos vozes, andamos sobre a terra.
    A própria existência é um milagre.

    O mundo à nossa volta é vasto e diverso.
    Existem montanhas que lembram milhões de anos.
    Existem mares onde se escondem segredos ainda não descobertos.
    Existem florestas onde cada árvore é testemunha do tempo.

    O homem é parte desse mundo e também observador.
    Nós nascemos frágeis, aprendemos a andar, a falar, a amar, a perdoar, a sonhar.
    A vida de cada um é feita de grandes e pequenos acontecimentos.

    No mundo existe alegria e dor.
    Elas caminham juntas, como dia e noite.
    Não se pode entender a felicidade sem conhecer a tristeza.
    Não se pode valorizar a saúde sem provar a doença.

    O amor é uma das forças centrais do universo.
    Ele se manifesta de muitas formas.
    O amor exige paciência, coragem, capacidade de perdoar.
    Sem amor, o homem é vazio como uma casa sem fogo.

    O tempo é um enigma.
    Nós medimos o tempo em horas, dias, anos, mas o sentimos de modos diferentes.
    Na infância o tempo passa devagar.
    Na juventude ele corre. Na maturidade ele é implacável.

    A morte é parte da vida.
    Ela assusta, mas também lembra o valor de cada instante.
    Nós não sabemos o que espera além do limiar.
    Nossos atos, palavras, amor e memória continuam vivos nos outros.

    A natureza é a nossa casa.
    Ela nos alimenta, cura, inspira.
    Muitas vezes esquecemos que ela não é infinita.
    As florestas são derrubadas, os rios são poluídos, o ar fica mais pesado.

    A ciência e a arte são as duas asas da humanidade.
    A ciência explica como o mundo funciona.
    A arte mostra como nós sentimos o mundo.
    A ciência cura doenças, constrói cidades, explora o cosmos.
    A arte consola, inspira, lembra da beleza.

    A infância é o começo do caminho.
    Na infância o mundo parece enorme, brilhante e cheio de maravilhas.
    Cada dia traz uma descoberta.

    A família é o primeiro mundo do homem.
    Na família aprendemos a amar, a perdoar, a suportar, a cuidar.

    O trabalho enche a vida de sentido.
    O homem foi criado para criar, para fazer, para servir.
    Alguns curam pessoas, alguns constroem casas, alguns ensinam crianças.

    A liberdade é um grande valor.
    A verdadeira liberdade é saber escolher o bem, assumir responsabilidade, respeitar os outros.

    A felicidade não é um ponto final, é um caminho.
    Ela não está em ter tudo, mas em valorizar o que se tem.
    A felicidade é a luz da manhã, o chá quente, o riso de uma criança, o abraço de alguém próximo.

    O cachorro é amigo do homem.
    O cachorro gosta de brincar e correr.
    O cachorro pode ter pelo, rabo, patas, orelhas.
    O gato também gosta de brincar. O gato ronrona e dorme.
    Os animais vivem ao lado do homem.

    O carrapato deixa uma mancha escura na pele do cachorro.
    A pulga morde o cachorro. A pulga é pequena e escura.
    A micose é um fungo. A micose dá uma mancha redonda na pele.
    A micose causa coceira. O fungo vive na pele.
    O melanoma é um tumor. O melanoma é perigoso para o cachorro.
    O melanoma aparece como uma mancha escura fixa.
    O veterinário olha a pele do cachorro.
    O veterinário cuida dos animais. O veterinário trata doenças.
    """

    dlm = DLM()
    dlm.aprender(texto)
    dlm.consolidar()

    perguntas = [
        "micose cachorro",
        "pulga cachorro",
        "tempo infância",
        "amor",
        "melanoma",
        "felicidade caminho",
        "carrapato pele",
        "o que é a vida",
        "como o cachorro pode ficar doente",
        "micose pele",
        "mancha cachorro",
        "cachorro gato",
    ]

    for p in perguntas:
        print(f">>> {p}")
        print(dlm.falar(p))
        print()
