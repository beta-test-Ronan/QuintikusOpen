/* ============================================================================
   QUINTIKUS — Visão + Córtex (port fiel do Python para JS puro)
   Zero dependências. JSON como formato único de persistência.
   ============================================================================ */

/* ============================================================================
   BLOCO 1 — VISÃO: DNA ENTÓPICO
   ============================================================================ */

const VETOR_DIM = 32;
const DNA_DIM   = 64;
const RADA_RAIO = 16;

/* ---------------------------------------------------------------------------
   DNAEntropico — mesma assinatura do Python, com extras do JS
   --------------------------------------------------------------------------- */
class DNAEntropico {
  constructor(entropia, geometria, momentos, rada, linearidade,
              entropia_dentro = 0, entropia_fora = 0) {
    this.entropia = entropia;
    this.geometria = geometria;
    this.momentos = momentos;            // [area, perim, ang0, ang1, ...] (compatível)
    this.rada = rada;                    // vetor de 12 (padrão) ou 36
    this.linearidade = linearidade;
    this.entropia_dentro = entropia_dentro;
    this.entropia_fora = entropia_fora;

    // Campos extras que só existiam no JS (opcionais)
    this.momentos_hu = null;
    this.hash = null;
    this.vetor8D = null;
    this.size = null;
    this.timestamp = new Date().toISOString();
  }

  // Compatível com `pickle`-style reconstruction: passar um array
  static fromArray(arr) {
    const d = new DNAEntropico(...arr.slice(0, 7));
    return d;
  }

  toArray() {
    return [this.entropia, this.geometria, this.momentos, this.rada,
            this.linearidade, this.entropia_dentro, this.entropia_fora];
  }
}

/* ---------------------------------------------------------------------------
   GeometriaTriangular — encontrar_pontos_entropia, area, perimetro, angulos
   Port literal do Python
   --------------------------------------------------------------------------- */
class GeometriaTriangular {
  static encontrar_pontos_entropia(pixels, largura, altura, n_pontos = 3) {
    const bloco_w = Math.max(1, Math.floor(largura / 8));
    const bloco_h = Math.max(1, Math.floor(altura / 8));
    const pontos = [];
    for (let y = 0; y + bloco_h <= altura; y += bloco_h) {
      for (let x = 0; x + bloco_w <= largura; x += bloco_w) {
        const bloco = [];
        for (let j = 0; j < bloco_h; j++)
          for (let i = 0; i < bloco_w; i++) {
            const idx = (y + j) * largura + (x + i);
            if (idx < pixels.length) bloco.push(pixels[idx]);
          }
        if (bloco.length) {
          const hist = new Array(256).fill(0);
          for (const p of bloco) hist[p]++;
          let ent = 0;
          const total = bloco.length;
          for (const c of hist) if (c > 0) { const p = c / total; ent -= p * Math.log2(p); }
          pontos.push([ent, x + (bloco_w >> 1), y + (bloco_h >> 1)]);
        }
      }
    }
    pontos.sort((a, b) => b[0] - a[0]);
    return pontos.slice(0, n_pontos).map(p => [p[1], p[2]]);
  }

  static area_triangulo(p1, p2, p3) {
    return Math.abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2);
  }

  static perimetro_triangulo(p1, p2, p3) {
    const d = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1]);
    return d(p1, p2) + d(p2, p3) + d(p3, p1);
  }

  static angulos_triangulo(p1, p2, p3) {
    const c = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1]);
    const a = c(p2, p3), b = c(p1, p3), cc = c(p1, p2);
    if (a === 0 || b === 0 || cc === 0) return [0, 0, 0];
    const angA = Math.acos(Math.max(-1, Math.min(1, (b*b + cc*cc - a*a) / (2*b*cc))));
    const angB = Math.acos(Math.max(-1, Math.min(1, (a*a + cc*cc - b*b) / (2*a*cc))));
    return [angA, angB, Math.PI - angA - angB];
  }

  static momentos_geometricos(pontos) {
    if (pontos.length < 3) return [0, 0, 0, 0];
    const [p1, p2, p3] = pontos;
    const area = GeometriaTriangular.area_triangulo(p1, p2, p3);
    const perim = GeometriaTriangular.perimetro_triangulo(p1, p2, p3);
    const ang = GeometriaTriangular.angulos_triangulo(p1, p2, p3);
    return [area / (1024 * 768), perim / Math.max(perim, 1), ang[0], ang[1]];
  }
}

/* ---------------------------------------------------------------------------
   RADAMilitar — 12 ângulos com softmax exponencial (fiel ao Python)
   --------------------------------------------------------------------------- */
class RADAMilitar {
  constructor(n_angulos = 12) {
    this.n_angulos = n_angulos;
    this.angulos = Array.from({ length: n_angulos }, (_, i) => i * 360 / n_angulos);
  }

  calcular(pixels, largura, altura) {
    const acum = new Array(this.n_angulos).fill(0);
    const step = Math.max(1, Math.floor(Math.min(largura, altura) / 20));
    for (let y = 1; y < altura - 1; y += step) {
      for (let x = 1; x < largura - 1; x += step) {
        const idx = y * largura + x;
        const gx = (pixels[Math.min(idx + 1, pixels.length - 1)] -
                    pixels[Math.max(idx - 1, 0)]) / 2;
        const gy = (pixels[Math.min(idx + largura, pixels.length - 1)] -
                    pixels[Math.max(idx - largura, 0)]) / 2;
        if (gx === 0 && gy === 0) continue;
        const ang = ((Math.atan2(gy, gx) * 180 / Math.PI) % 360 + 360) % 360;
        const mag = Math.hypot(gx, gy);
        const iAng = Math.floor((ang / (360 / this.n_angulos)) % this.n_angulos);
        acum[iAng] += mag;
      }
    }
    const maxVal = acum.length ? Math.max(...acum) : 0;
    const expVals = acum.map(v => Math.exp(v - maxVal));
    const soma = expVals.reduce((s, v) => s + v, 0);
    return soma > 0 ? expVals.map(e => e / soma)
                    : new Array(this.n_angulos).fill(1 / this.n_angulos);
  }
}

/* ---------------------------------------------------------------------------
   ConversorUniversal — imagem → DNA (mesma pipeline do Python)
   Aceita: { pixels, largura, altura } vindo do Canvas/JS
   --------------------------------------------------------------------------- */
class ConversorUniversal {
  static entropia(pList) {
    const hist = new Array(256).fill(0);
    for (const p of pList) hist[p]++;
    let h = 0;
    for (const c of hist) if (c > 0) { const p = c / pList.length; h -= p * Math.log2(p); }
    return h;
  }

  converter(pixels, largura, altura) {
    // Amostra em grade VETOR_DIM×VETOR_DIM
    const redim = new Array(VETOR_DIM * VETOR_DIM);
    for (let y = 0; y < VETOR_DIM; y++)
      for (let x = 0; x < VETOR_DIM; x++)
        redim[y*VETOR_DIM + x] =
          pixels[Math.floor(y * (altura / VETOR_DIM)) * largura +
                 Math.floor(x * (largura / VETOR_DIM))];

    const dentro = [], fora = [];
    for (let y = 0; y < VETOR_DIM; y++)
      for (let x = 0; x < VETOR_DIM; x++) {
        const v = redim[y*VETOR_DIM + x];
        if (x >= 8 && x < 24 && y >= 8 && y < 24) dentro.push(v);
        else fora.push(v);
      }

    const pontos = GeometriaTriangular.encontrar_pontos_entropia(pixels, largura, altura, 3);

    return {
      entropia: ConversorUniversal.entropia(pixels),
      geometria: Math.min(largura/altura, altura/largura),
      momentos: [pixels.reduce((s,v)=>s+v,0)/pixels.length/255, 0, 0, 0],
      momentos_tri: GeometriaTriangular.momentos_geometricos(pontos),
      rada: new RADAMilitar(12).calcular(pixels, largura, altura),
      linearidade: 0.5,   // placeholder (JS original calculava por PCA)
      entropia_dentro: ConversorUniversal.entropia(dentro),
      entropia_fora: ConversorUniversal.entropia(fora),
      size: [largura, altura]
    };
  }

  // Helper: aceita ImageData direto do Canvas
  fromImageData(imgData) {
    const { width, height, data } = imgData;
    const px = new Array(width * height);
    for (let i = 0, p = 0; i < data.length; i += 4, p++)
      px[p] = Math.round(0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]);
    return this.converter(px, width, height);
  }
}

/* ---------------------------------------------------------------------------
   Laminy — divergência logarítmica do Python (mais estável que euclidiana)
   --------------------------------------------------------------------------- */
class Laminy {
  static comparar(d1, d2) {
    const col = (a, b) => -Math.log2(1 - Math.max(0, Math.min(1, Math.abs(a - b))) + 1e-9);
    let soma =
      3.0 * col(d1.entropia_dentro / 8, d2.entropia_dentro / 8) +
      3.0 * col(d1.entropia_fora   / 8, d2.entropia_fora   / 8) +
           col(d1.entropia          / 8, d2.entropia        / 8);

    const n = Math.min(d1.rada.length, d2.rada.length);
    for (let i = 0; i < n; i++) soma += col(d1.rada[i], d2.rada[i]) / 12;

    const fator = 1.0 + Math.abs(d1.geometria - d2.geometria);
    return Math.exp(-soma / (10.0 * fator));
  }
}

/* ============================================================================
   BLOCO 2 — CÓRTEX NEURAL (TGNC v14) — port fiel do Python
   ============================================================================ */

/* -------- helpers de álgebra linear -------- */
function l2norm(x) {
  let n = 0;
  for (const v of x) n += v * v;
  n = Math.sqrt(n);
  return n > 0 ? x.map(v => v / n) : x.slice();
}
function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x))));
}
function matVec(M, v) {                       // M: [m][n] , v: [n] → [m]
  return M.map(row => { let s = 0; for (let i = 0; i < v.length; i++) s += row[i]*v[i]; return s; });
}
function outer(a, b) {                        // [m][n]
  return a.map(x => b.map(y => x * y));
}
function matAdd(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}
function zeros(m, n) { return Array.from({length:m}, () => new Array(n).fill(0)); }
function randn(m, n, scale = 0.01) {
  return Array.from({length:m}, () => Array.from({length:n}, () => {
    // Box-Muller
    const u = Math.random() || 1e-9, v = Math.random();
    return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v) * scale;
  }));
}
function eye(n) {
  return Array.from({length:n}, (_, i) => Array.from({length:n}, (_, j) => i===j ? 1 : 0));
}

/* ---------------------------------------------------------------------------
   TGNC — mesmo estado, mesmas equações, mesmo comportamento do Python
   --------------------------------------------------------------------------- */
class TGNC_NeuralCortex {
  constructor(dim = 128, rank = 64) {
    this.dim = dim; this.rank = rank;
    this.lr = 0.5; this.decay = 0.999;
    this.solo_path = 'tgnc_cortex.json';
    this.vocab = {}; this.concepts = {}; this.actions = {};
    this.W1 = randn(dim, rank);
    this.W2 = randn(rank, dim);
    this.W_conf = randn(dim + dim, 1);
    this.W_context = eye(dim);
    this.W_action  = eye(dim);
    this.ruido = new Set(['o','a','os','as','de','da','do','um','uma','esta',
                          'com','que','para','em','no','na','e']);
  }

  _getVector(word, space = 'vocab') {
    const t = { vocab: this.vocab, concepts: this.concepts, actions: this.actions }[space];
    if (!t[word]) t[word] = l2norm(Array.from({length:this.dim}, () => Math.random()*2-1));
    return t[word];
  }

  processarSequencia(texto) {
    if (!texto || !texto.trim()) return [null, 0];
    const palavras = texto.toLowerCase().replace(/[.,]/g, ' ').split(/\s+/)
                          .filter(w => w && !this.ruido.has(w));
    if (!palavras.length) {
      const all = texto.toLowerCase().split(/\s+/).filter(Boolean).map(p => this._getVector(p));
      const mean = new Array(this.dim).fill(0);
      for (const v of all) for (let i = 0; i < this.dim; i++) mean[i] += v[i];
      return [l2norm(mean.map(v => v / Math.max(1, all.length))), 0.15];
    }

    const hStates = [];
    let prev = new Array(this.dim).fill(0);
    for (const p of palavras) {
      const v_k = this._getVector(p);
      const W1v = matVec(this.W1, prev);   // [rank]
      const W2v = matVec(this.W2, W1v);    // [dim]
      const h_k = l2norm(v_k.map((x, i) => x + W2v[i]));
      hStates.push(h_k);
      prev = h_k;
    }
    const mean = new Array(this.dim).fill(0);
    for (const h of hStates) for (let i = 0; i < this.dim; i++) mean[i] += h[i];
    const ctx = l2norm(mean.map(v => v / hStates.length));

    const cat = ctx.concat(prev);           // [2*dim]
    let confRaw = 0;
    for (let i = 0; i < this.W_conf.length; i++) confRaw += this.W_conf[i][0] * cat[i];
    const conf = sigmoid(confRaw);
    return [ctx, conf];
  }

  ensinar(textoLongo, nexo, acao) {
    const [x] = this.processarSequencia(textoLongo);
    if (!x) return;
    const y_t = this._getVector(nexo, 'concepts');
    const z_t = this._getVector(acao, 'actions');

    for (let it = 0; it < 60; it++) {
      // W_context += lr * outer(x, y_t - x @ W_context)
      const yPred = matVec(this.W_context, x);
      const errCtx = y_t.map((v, i) => v - yPred[i]);
      const deltaCtx = outer(x, errCtx).map(r => r.map(v => this.lr * v));
      this.W_context = matAdd(this.W_context, deltaCtx);

      // W_action += lr * outer(y_t, z_t - y_t @ W_action)
      const zPred = matVec(this.W_action, y_t);
      const errAct = z_t.map((v, i) => v - zPred[i]);
      const deltaAct = outer(y_t, errAct).map(r => r.map(v => this.lr * v));
      this.W_action = matAdd(this.W_action, deltaAct);
    }
  }

  analisar(texto) {
    const [x, conf_n] = this.processarSequencia(texto);
    if (!x) return ['SILÊNCIO', 'NENHUMA', 0, 'NEUTRO'];
    const v_n = l2norm(matVec(this.W_context, x));
    const v_a = l2norm(matVec(this.W_action, v_n));
    const nexo_f = this._buscarProximo(v_n, 'concepts');
    const acao_f = this._buscarProximo(v_a, 'actions');
    const sim_geo = dot(v_n, this.concepts[nexo_f] || new Array(this.dim).fill(0));
    return [nexo_f, acao_f, sim_geo * 0.7 + conf_n * 0.3, 'ESTÁVEL'];
  }

  _buscarProximo(vetor, space) {
    const t = { concepts: this.concepts, actions: this.actions }[space];
    let melhor = 'Indefinido', sim = -1;
    for (const [k, v] of Object.entries(t)) {
      const s = dot(vetor, v);
      if (s > sim) { sim = s; melhor = k; }
    }
    return melhor;
  }

  toJSON() {
    const enc = (m) => m.map(r => r.map(v => +v.toFixed(4)));
    const encDict = (d) => Object.fromEntries(
      Object.entries(d).map(([k, v]) => [k, Array.from(v).map(x => +x.toFixed(4))])
    );
    return {
      dim: this.dim, rank: this.rank, lr: this.lr, decay: this.decay,
      vocab: encDict(this.vocab), concepts: encDict(this.concepts), actions: encDict(this.actions),
      W1: enc(this.W1), W2: enc(this.W2), W_conf: enc(this.W_conf),
      W_context: enc(this.W_context), W_action: enc(this.W_action)
    };
  }

  fromJSON(j) {
    const dec = (m) => m.map(r => r.slice());
    const decDict = (d) => Object.fromEntries(
      Object.entries(d).map(([k, v]) => [k, v.slice()])
    );
    this.vocab = decDict(j.vocab); this.concepts = decDict(j.concepts); this.actions = decDict(j.actions);
    this.W1 = dec(j.W1); this.W2 = dec(j.W2); this.W_conf = dec(j.W_conf);
    this.W_context = dec(j.W_context); this.W_action = dec(j.W_action);
    this.dim = j.dim; this.rank = j.rank;
  }

  salvar() {
    try {
      if (typeof localStorage !== 'undefined')
        localStorage.setItem(this.solo_path, JSON.stringify(this.toJSON()));
    } catch (e) { console.warn('save fail', e); }
  }

  carregar() {
    try {
      if (typeof localStorage !== 'undefined') {
        const s = localStorage.getItem(this.solo_path);
        if (s) this.fromJSON(JSON.parse(s));
      }
    } catch (e) { console.warn('load fail', e); }
  }
}

/* ---------------------------------------------------------------------------
   WorldModel + FeelingModule — port direto
   --------------------------------------------------------------------------- */
class WorldModel {
  constructor() { this.rules = {}; }
  simular(acao, props) {
    const rule = this.rules[acao.toUpperCase()] || { risco: 0.5 };
    if (props.fragil && rule.risco > 0.4) return ['PERIGO', rule.risco];
    return ['OK', 0];
  }
}

class FeelingModule {
  constructor() { this.rules = []; }
  avaliar(nexo, conf, current) {
    for (const r of this.rules) {
      if (r.perceber && nexo.toUpperCase().includes(r.perceber.toUpperCase()) &&
          conf >= (r.v1 ?? 0) && conf <= (r.v2 ?? 1)) return r.mood ?? current;
    }
    return current;
  }
}

/* ============================================================================
   BLOCO 3 — ORQUESTRADOR UNIFICADO
   ============================================================================ */
class Quintikus {
  constructor() {
    this.conversor = new ConversorUniversal();
    this.memoriaVisual = {};          // { tag: [DNAEntropico, ...] }
    this.tgnc = new TGNC_NeuralCortex();
    this.world = new WorldModel();
    this.snc = new FeelingModule();
    this.memoriaSemantica = {};
    this.tgnc.carregar();
  }

  /* ---------- VISÃO ---------- */
  treinarImagem(imgData, tag) {
    const m = this.conversor.fromImageData(imgData);
    const dna = new DNAEntropico(
      m.entropia, m.geometria,
      m.momentos.slice(0, 2).concat(m.momentos_tri.slice(0, 2)),
      m.rada, m.linearidade, m.entropia_dentro, m.entropia_fora
    );
    dna.size = m.size;
    if (!this.memoriaVisual[tag]) this.memoriaVisual[tag] = [];
    this.memoriaVisual[tag].push(dna);
    this.salvarVisao();
    return m;
  }

  preverImagem(imgData) {
    if (!Object.keys(this.memoriaVisual).length) return ['Nenhum dado', 0];
    const m = this.conversor.fromImageData(imgData);
    const dnaEntrada = new DNAEntropico(
      m.entropia, m.geometria,
      m.momentos.slice(0, 2).concat(m.momentos_tri.slice(0, 2)),
      m.rada, m.linearidade, m.entropia_dentro, m.entropia_fora
    );

    const scores = [];
    for (const [tag, exemplos] of Object.entries(this.memoriaVisual)) {
      if (!exemplos.length) continue;
      const sim = exemplos.reduce((s, ex) => s + Laminy.comparar(dnaEntrada, ex), 0) / exemplos.length;
      scores.push([tag, sim]);
    }
    scores.sort((a, b) => b[1] - a[1]);
    if (!scores.length) return ['Nenhum dado', 0];
    return [scores[0][0], scores[0][1], scores[0][1]];
  }

  salvarVisao() {
    const dados = { classes: {} };
    for (const [tag, exemplos] of Object.entries(this.memoriaVisual))
      dados.classes[tag] = exemplos.map(d => d.toArray());
    try {
      if (typeof localStorage !== 'undefined')
        localStorage.setItem('quintikus_visual.json', JSON.stringify(dados));
    } catch (e) { console.warn(e); }
  }

  carregarVisao() {
    try {
      if (typeof localStorage !== 'undefined') {
        const s = localStorage.getItem('quintikus_visual.json');
        if (!s) return;
        const dados = JSON.parse(s);
        for (const [tag, arr] of Object.entries(dados.classes || {}))
          this.memoriaVisual[tag] = arr.map(a => DNAEntropico.fromArray(a));
      }
    } catch (e) { console.warn(e); }
  }

  /* ---------- COGNIÇÃO ---------- */
  aprenderTexto(texto, nexo, acao, fragil = false) {
    this.tgnc.ensinar(texto, nexo, acao);
    this.tgnc.salvar();
    this.memoriaSemantica[texto.toLowerCase()] = { fragil };
    return `🌟 Organismo evoluiu com: "${texto}"`;
  }

  pensar(entrada) {
    const [nexo, acao, conf, moodN] = this.tgnc.analisar(entrada);
    const mood = this.snc.avaliar(nexo, conf, moodN);
    const props = this.memoriaSemantica[(entrada.split(/\s+/)[0] || '').toLowerCase()]
                   || { fragil: false };
    const [info, risco] = this.world.simular(acao, props);
    return {
      nexo, acao, conf, mood,
      decisao: risco > 0.6 ? 'INTERROMPER' : 'EXECUTAR',
      info, risco
    };
  }
}

/* ============================================================================
   EXPORTAÇÃO
   ============================================================================ */
if (typeof window !== 'undefined') {
  window.Quintikus = Quintikus;
  window.DNAEntropico = DNAEntropico;
  window.GeometriaTriangular = GeometriaTriangular;
  window.RADAMilitar = RADAMilitar;
  window.ConversorUniversal = ConversorUniversal;
  window.Laminy = Laminy;
  window.TGNC_NeuralCortex = TGNC_NeuralCortex;
  window.WorldModel = WorldModel;
  window.FeelingModule = FeelingModule;
}
if (typeof module !== 'undefined') module.exports = {
  Quintikus, DNAEntropico, GeometriaTriangular, RADAMilitar,
  ConversorUniversal, Laminy, TGNC_NeuralCortex, WorldModel, FeelingModule
};