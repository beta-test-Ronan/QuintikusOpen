// organismo_v31.1_refinado.js — Quintikus SSML v31.1 — Organismo Soberano (otimizado)
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import readline from 'readline';

process.env.UV_THREADPOOL_SIZE = '1';

// ==================================================================
// 🧹 NORMALIZADOR SOMÁTICO
// ==================================================================
class NormalizadorSomático {
    static limpar(texto) {
        if (!texto) return "";
        texto = texto.toLowerCase();
        texto = texto.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
        return texto.replace(/[^a-z0-9!?.\s]/g, '').trim();
    }
}

// ==================================================================
// 🧬 KERNEL RESSONANTE
// ==================================================================
class KernelRessonante {
    static get_vetor_esparso(token, dims = 5000, sparsity = 100) {
        const hash = crypto.createHash('sha256').update(token).digest('hex');
        const seed = parseInt(hash.substring(0, 15), 16);
        const rng = this._seededRandom(seed);
        const indices = new Set();
        while (indices.size < sparsity) indices.add(Math.floor(rng() * dims));
        const vec = {};
        for (const i of indices) vec[i] = this._gaussRandom(rng);
        return vec;
    }

    static _seededRandom(seed) {
        let s = seed;
        return () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
    }

    static _gaussRandom(rng) {
        let u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    static tsallis_match(v1, v2, q = 0.8) {
        const keys = Object.keys(v1).filter(k => k in v2);
        if (keys.length === 0) return 0.0;
        let sum = 0;
        for (const k of keys) sum += Math.pow(Math.abs(v1[k] * v2[k]), q);
        return (1.0 - sum) / (q - 1.0 + 1e-9);
    }

    static dot(v1, v2) {
        let sum = 0;
        for (const k of Object.keys(v1)) if (k in v2) sum += v1[k] * v2[k];
        return sum;
    }

    static normalize(v) {
        const vals = Object.values(v);
        if (vals.length === 0) return {};
        let norm = 0;
        for (const val of vals) norm += val * val;
        norm = Math.sqrt(norm) + 1e-9;
        const res = {};
        for (const [k, val] of Object.entries(v)) res[k] = val / norm;
        return res;
    }
}

// ==================================================================
// 🧠 CÓRTEX COGNITIVO (pequena proteção contra NaN)
// ==================================================================
class CortexCognitivo {
    constructor(limite_confusao = 0.30) {
        this.limite_confusao = limite_confusao;
        this.epsilon = 1e-9;
        this.taxa_pensamento = 0.12;
    }

    _norm(d) {
        const sum = d.reduce((a, b) => a + b, 0) + this.epsilon;
        return d.map(x => x / sum);
    }

    divergencia_kl(p, q) {
        let sum = 0;
        for (let i = 0; i < p.length; i++) {
            sum += p[i] * Math.log((p[i] + this.epsilon) / (q[i] + this.epsilon));
        }
        // Evita NaN/Infinito retornando um valor padrão
        return isFinite(sum) ? sum : this.limite_confusao;
    }

    processar_reflexao(estado_real, estado_interno) {
        // Garante tamanhos iguais e mínimos
        if (!estado_real || !estado_interno || estado_real.length !== estado_interno.length) {
            const pad = [0.25,0.25,0.25,0.25];
            return { estado: pad, ciclos: 0, confusao: 0.30 };
        }
        let p = this._norm(estado_real);
        let q = this._norm(estado_interno);
        let ciclos = 0;
        let confusao = this.divergencia_kl(p, q);
        while (confusao > this.limite_confusao && ciclos < 45) {
            ciclos++;
            for (let i = 0; i < q.length; i++) {
                q[i] = q[i] + this.taxa_pensamento * (p[i] - q[i]);
            }
            q = this._norm(q);
            confusao = this.divergencia_kl(p, q);
        }
        return { estado: q, ciclos, confusao };
    }
}

// ==================================================================
// 🧠 SISTEMA NERVOSO CENTRAL (RNN + ADAM) – incremento garantido do t
// ==================================================================
class SistemaNervosoCentral {
    constructor(n_in = 6, n_hid = 10, n_out = 3, path_bin = "sistema_nervoso.bin") {
        this.n_in = n_in;
        this.n_hid = n_hid;
        this.n_out = n_out;
        this.path = path_bin;
        this.t = 0;
        this.lr = 0.005;

        const initMatrix = (rows, cols) => Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => (Math.random() * 0.2 - 0.1))
        );
        this.W_h = initMatrix(n_hid, n_in + n_hid);
        this.W_y = initMatrix(n_out, n_hid);
        this.B_h = Array(n_hid).fill(0);
        this.B_y = Array(n_out).fill(0);

        this.adam_M_Wh = initMatrix(n_hid, n_in + n_hid);
        this.adam_V_Wh = initMatrix(n_hid, n_in + n_hid);
        this.adam_M_Wy = initMatrix(n_out, n_hid);
        this.adam_V_Wy = initMatrix(n_out, n_hid);

        this.estado_anterior = Array(n_hid).fill(0);
        this.cache = null;

        if (fs.existsSync(this.path)) this._carregar();
    }

    sigmoid(x) {
        return 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, x))));
    }

    pulsar_vontade(x_atual) {
        // Ajusta o tamanho da entrada para n_in (completa com zeros se necessário)
        const entrada = x_atual.slice(0, this.n_in);
        while (entrada.length < this.n_in) entrada.push(0);
        const inp = entrada.concat(this.estado_anterior);
        const h = Array(this.n_hid).fill(0);
        for (let i = 0; i < this.n_hid; i++) {
            let sum = this.B_h[i];
            for (let j = 0; j < inp.length; j++) sum += inp[j] * this.W_h[i][j];
            h[i] = this.sigmoid(sum);
        }
        const y = Array(this.n_out).fill(0);
        for (let i = 0; i < this.n_out; i++) {
            let sum = this.B_y[i];
            for (let j = 0; j < this.n_hid; j++) sum += h[j] * this.W_y[i][j];
            y[i] = this.sigmoid(sum);
        }
        this.cache = { inp, h, y };
        this.estado_anterior = h;
        return y;
    }

    adaptar_realtime(alvo_ideal) {
        if (!this.cache) return;
        this.t++;  // incrementa mesmo que o cache exista
        const { inp, h, y } = this.cache;
        const lr = this.lr;
        const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        const corr1 = Math.max(1 - Math.pow(beta1, this.t), 1e-8);
        const corr2 = Math.max(1 - Math.pow(beta2, this.t), 1e-8);

        const delta_y = y.map((yi, i) => (yi - (alvo_ideal[i] || 0)) * yi * (1 - yi));
        const delta_h = Array(this.n_hid).fill(0);
        for (let j = 0; j < this.n_hid; j++) {
            let sum = 0;
            for (let i = 0; i < this.n_out; i++) sum += delta_y[i] * this.W_y[i][j];
            delta_h[j] = sum * h[j] * (1 - h[j]);
        }

        for (let i = 0; i < this.n_out; i++) {
            for (let j = 0; j < this.n_hid; j++) {
                const grad = delta_y[i] * h[j];
                this.adam_M_Wy[i][j] = beta1 * this.adam_M_Wy[i][j] + (1 - beta1) * grad;
                this.adam_V_Wy[i][j] = beta2 * this.adam_V_Wy[i][j] + (1 - beta2) * (grad * grad);
                this.W_y[i][j] -= lr * (this.adam_M_Wy[i][j] / corr1) / (Math.sqrt(Math.abs(this.adam_V_Wy[i][j]) / corr2) + eps);
            }
            this.B_y[i] -= lr * delta_y[i];
        }

        for (let i = 0; i < this.n_hid; i++) {
            for (let j = 0; j < inp.length; j++) {
                const grad = delta_h[i] * inp[j];
                this.adam_M_Wh[i][j] = beta1 * this.adam_M_Wh[i][j] + (1 - beta1) * grad;
                this.adam_V_Wh[i][j] = beta2 * this.adam_V_Wh[i][j] + (1 - beta2) * (grad * grad);
                this.W_h[i][j] -= lr * (this.adam_M_Wh[i][j] / corr1) / (Math.sqrt(Math.abs(this.adam_V_Wh[i][j]) / corr2) + eps);
            }
            this.B_h[i] -= lr * delta_h[i];
        }
    }

    _salvar() {
        const estado = {
            Wh: this.W_h, Wy: this.W_y, Bh: this.B_h, By: this.B_y,
            ea: this.estado_anterior, t: this.t,
            MWh: this.adam_M_Wh, VWh: this.adam_V_Wh,
            MWy: this.adam_M_Wy, VWy: this.adam_V_Wy
        };
        fs.writeFileSync(this.path, JSON.stringify(estado));
    }

    _carregar() {
        try {
            const data = JSON.parse(fs.readFileSync(this.path, 'utf-8'));
            this.W_h = data.Wh;
            this.W_y = data.Wy;
            this.B_h = data.Bh;
            this.B_y = data.By;
            this.t = data.t || 0;
            this.adam_M_Wh = data.MWh || this.adam_M_Wh;
            this.adam_V_Wh = data.VWh || this.adam_V_Wh;
            this.adam_M_Wy = data.MWy || this.adam_M_Wy;
            this.adam_V_Wy = data.VWy || this.adam_V_Wy;
            this.estado_anterior = data.ea || this.estado_anterior;
        } catch (e) { /* mantém inicialização padrão */ }
    }
}

// ==================================================================
// 🧬 DRIVE SOMÁTICO (mantido igual)
// ==================================================================
class DriveSomático {
    constructor() {
        this.vm = -70.0;
        this.eixos = { amor: 0.1, prazer: 0.1, tristeza: 0.1, raiva: 0.1 };
        this.valvulas = { amor: false, prazer: false, tristeza: false, raiva: false };
    }

    pulsar(impacto, u_toks) {
        this.vm = Math.max(-90.0, Math.min(-45.0, this.vm + impacto * 12));
        const gatilhos = {
            amor: ["amo", "amor"], prazer: ["prazer", "delicia"],
            tristeza: ["triste", "mal"], raiva: ["odeio", "raiva"]
        };
        for (const [eixo, keywords] of Object.entries(gatilhos)) {
            for (const k of keywords) {
                if (u_toks.includes(k)) {
                    if (this.valvulas[eixo]) {
                        this.eixos[eixo] *= 0.6;
                    } else {
                        this.eixos[eixo] += impacto;
                    }
                    this.valvulas[eixo] = this.eixos[eixo] > 4.5;
                }
            }
        }
    }
}

// ==================================================================
// 🔍 SISTEMA DEEPY (mantido igual)
// ==================================================================
class SistemaDeepy {
    constructor(raridade) {
        this.raridade = raridade;
        this.turnos_think = 0;
        this.frequencia_pulso = new Map();
        this.expansores = ['fale', 'sobre', 'tudo', 'detalhes', 'mais', 'explique'];
    }

    crivo_meritocratico(tokens, impacto) {
        if (!tokens.length) return { tem_merito: false, nivel: 0 };
        const Q = tokens.length;
        let P = 0;
        for (const t of tokens) {
            const freq = this.raridade.get(t) || 1;
            P += 1.5 / (Math.log(freq + 1.2) + 1e-5);
        }
        let x_apr = 0;
        for (const t of tokens) x_apr += this.frequencia_pulso.get(t) || 0;
        x_apr /= (Q + 1e-5);
        const x_nec = Q / (P + 1e-5);
        return { tem_merito: x_apr >= x_nec * 0.08, nivel: x_apr };
    }

    filtrar_expansao(sujeito, u_toks, entrada_bruta, neuronios, episodes) {
        if (!entrada_bruta || !this.expansores.some(w => entrada_bruta.toLowerCase().includes(w)) || u_toks.length < 2) return null;
        const contexto = u_toks.filter(t => t !== sujeito);
        if (!contexto.length) return null;
        const alvo = contexto[0];
        if (neuronios[sujeito] && neuronios[alvo]) {
            const comuns = neuronios[sujeito].filter(i => neuronios[alvo].includes(i));
            if (comuns.length) return episodes[comuns[Math.floor(Math.random() * comuns.length)]].t;
        }
        return null;
    }
}

// ==================================================================
// 🌿 ORGANISMO SOBERANO v31.1 (refinado)
// ==================================================================
class OrganismoSoberano {
    constructor() {
        this.path_bin = "nucleo_organismo.qssml";
        this.path_ledger = "ledger.bin";
        this.auto_train_files = ["oi.txt", "amor.txt", "prazer.txt", "confusa.txt", "sentimento.txt"];

        this.mapa_nd = {};
        this.l2_episodes = [];
        this.neuronios = {};
        this.raridade = new Map();
        this.history = [];
        this.fatigue = new Map();
        this.ctx_foco = {};
        this.ledger = new Set();

        this.soma = new DriveSomático();
        this.cortex = new CortexCognitivo();
        this.snc = new SistemaNervosoCentral();
        this.deepy = new SistemaDeepy(this.raridade);
        this.tokenizer = /\b\w+\b|[!?.]/g;
    }

    _get_entropy(t) {
        const count = this.raridade.get(t) || 1;
        return 1.0 / (Math.log(count + 1.2) + 1e-5);
    }

    processar(entrada) {
        const t0 = performance.now();
        this.deepy.turnos_think++;
        if (this.deepy.turnos_think >= 7) {
            console.log("\n🧠 [DEEPY] Reorganização REM ativada...");
            for (const k of this.fatigue.keys()) this.fatigue.set(k, (this.fatigue.get(k) || 0) * 0.2);
            this.deepy.turnos_think = 0;
        }

        const raw = NormalizadorSomático.limpar(entrada);
        const u_toks = raw.match(this.tokenizer) || [];
        if (!u_toks.length) return "...";

        // 1. Sujeito e impacto
        const sujeito = u_toks.filter(t => this.neuronios[t])[0] || u_toks[0];
        const impacto = this._get_entropy(sujeito);
        this.soma.pulsar(impacto, u_toks);
        for (const t of u_toks) this.deepy.frequencia_pulso.set(t, (this.deepy.frequencia_pulso.get(t) || 0) + 1);
        const { tem_merito, nivel } = this.deepy.crivo_meritocratico(u_toks, impacto);

        // 2. Córtex e SNC
        const chaves_emocao = ["amor", "prazer", "tristeza", "raiva"];
        const p_real = chaves_emocao.map(k => this.soma.eixos[k] || 0.1);
        const q_int = this.snc.estado_anterior.slice(0, 4);
        const { estado: estado_em, ciclos, confusao } = this.cortex.processar_reflexao(p_real, q_int);
        const dkl = confusao; // mantém nome original
        const entrada_snc = [...estado_em, impacto, (this.soma.vm + 90) / 45];
        const volicao = this.snc.pulsar_vontade(entrada_snc);
        const modo_idx = volicao.indexOf(Math.max(...volicao));

        // 3. Vetor de entrada
        let v_in = {};
        for (const t of u_toks) {
            if (t in this.mapa_nd) {
                const vec = this.mapa_nd[t];
                const peso = this._get_entropy(t);
                for (const k of Object.keys(vec)) {
                    v_in[k] = (v_in[k] || 0) + vec[k] * peso;
                }
            }
        }
        v_in = KernelRessonante.normalize(v_in);

        // Atualiza contexto focal
        if (Object.keys(this.ctx_foco).length === 0) {
            this.ctx_foco = v_in;
        } else {
            const novo = {};
            const keys = new Set([...Object.keys(this.ctx_foco), ...Object.keys(v_in)]);
            for (const k of keys) {
                novo[k] = (this.ctx_foco[k] || 0) * 0.6 + (v_in[k] || 0) * 0.4;
            }
            this.ctx_foco = KernelRessonante.normalize(novo);
        }

        // 4. Candidatos
        let candidatos = this.neuronios[sujeito] ? [...this.neuronios[sujeito]] : [];
        if (!candidatos.length) {
            const amostra = Math.min(this.l2_episodes.length, 150);
            candidatos = Array.from({ length: this.l2_episodes.length }, (_, i) => i)
                .sort(() => Math.random() - 0.5)
                .slice(0, amostra);
        }

        // 5. Scoring (proteção adicional contra episódios sem vetor)
        const scored = [];
        for (const idx of candidatos) {
            const ep = this.l2_episodes[idx];
            if (!ep || !ep.v || this.history.includes(ep.t)) continue;
            const ressonancia = KernelRessonante.tsallis_match(v_in, ep.v);
            const foco = KernelRessonante.dot(this.ctx_foco, ep.v);
            const fadiga = this.fatigue.get(ep.t) || 0;
            const score = ressonancia + foco * 0.3 - fadiga;
            scored.push({ idx, score });
        }
        scored.sort((a, b) => b.score - a.score);
        const melhor = scored.length ? scored[0].idx : Math.floor(Math.random() * this.l2_episodes.length);
        const res = this.l2_episodes[melhor]?.t || "...";

        // 6. Aprendizado SNC (garante que dkl é um número)
        if (isFinite(dkl) && dkl < 0.45) {
            const alvo = [0, 0, 0];
            alvo[modo_idx] = 1;
            this.snc.adaptar_realtime(alvo);
        }

        this.history.push(res);
        if (this.history.length > 20) this.history.shift();
        this.fatigue.set(res, (this.fatigue.get(res) || 0) + 10);
        for (const [k, v] of this.fatigue.entries()) {
            this.fatigue.set(k, v * 0.65);
        }

        const dt = performance.now() - t0;
        const dklDisplay = isFinite(dkl) ? dkl.toFixed(2) : "0.00";
        console.log(` ⚛️ [SNC t:${this.snc.t}] Pensou ${ciclos} Ciclos (DKL:${dklDisplay}) | ${dt.toFixed(1)}ms`);
        return res;
    }

    boot() {
        if (fs.existsSync(this.path_bin)) {
            try {
                const data = JSON.parse(fs.readFileSync(this.path_bin, 'utf-8'));
                this.l2_episodes = data.nexus || [];
                this.raridade = new Map(Object.entries(data.raridade || {}));
                this.mapa_nd = data.nd || {};
                this.ctx_foco = data.ctx_foco || {};
            } catch (e) {
                console.log("⚠️ Erro ao carregar núcleo, iniciando vazio.");
            }
        }
        if (fs.existsSync(this.path_ledger)) {
            try {
                this.ledger = new Set(JSON.parse(fs.readFileSync(this.path_ledger, 'utf-8')));
            } catch (e) { /* vazio */ }
        }
        for (const arq of this.auto_train_files) {
            if (fs.existsSync(arq)) {
                const conteudo = fs.readFileSync(arq, 'utf-8');
                const hash = crypto.createHash('sha256').update(conteudo).digest('hex');
                if (!this.ledger.has(hash)) {
                    console.log(`📚 Treinando com: ${arq}`);
                    this.cristalizar_solo(conteudo);
                    this.ledger.add(hash);
                }
            }
        }
        this.neuronios = {};
        for (let i = 0; i < this.l2_episodes.length; i++) {
            const ep = this.l2_episodes[i];
            const tokens = NormalizadorSomático.limpar(ep.t).match(this.tokenizer) || [];
            for (const t of tokens) {
                if (!this.neuronios[t]) this.neuronios[t] = [];
                this.neuronios[t].push(i);
            }
        }
        console.log(`✅ Organismo Online. SNC t:${this.snc.t} | Nexos: ${this.l2_episodes.length}`);
    }

    cristalizar_solo(texto) {
        const frases = texto.split(/[.!?\n]+/);
        let count = 0;
        for (const f of frases) {
            const limpa = NormalizadorSomático.limpar(f);
            if (limpa.length < 3) continue;
            const idx = this.l2_episodes.length;
            let v_ep = {};
            const tokens = limpa.match(this.tokenizer) || [];
            for (const t of tokens) {
                this.raridade.set(t, (this.raridade.get(t) || 0) + 1);
                if (!this.neuronios[t]) this.neuronios[t] = [];
                this.neuronios[t].push(idx);
                if (!(t in this.mapa_nd)) {
                    this.mapa_nd[t] = KernelRessonante.get_vetor_esparso(t);
                }
                const vec = this.mapa_nd[t];
                const peso = this._get_entropy(t);
                for (const k of Object.keys(vec)) {
                    v_ep[k] = (v_ep[k] || 0) + vec[k] * peso;
                }
            }
            this.l2_episodes.push({
                t: f.trim(),
                v: KernelRessonante.normalize(v_ep)
            });
            count++;
        }
        if (count) console.log(`   ✅ ${count} frases cristalizadas`);
    }

    dormir() {
        this.snc._salvar();
        const modelo = {
            nexus: this.l2_episodes,
            raridade: Object.fromEntries(this.raridade),
            nd: this.mapa_nd,
            ctx_foco: this.ctx_foco
        };
        fs.writeFileSync(this.path_bin, JSON.stringify(modelo));
        fs.writeFileSync(this.path_ledger, JSON.stringify([...this.ledger]));
        console.log("💤 Organismo adormeceu (dados salvos).");
    }

    despertar() {
        if (Object.keys(this.ctx_foco).length === 0) return "Olá.";
        const candidatas = this.l2_episodes
            .filter(ep => KernelRessonante.dot(this.ctx_foco, ep.v) > 0.6)
            .map(ep => ep.t);
        if (candidatas.length) {
            return `'${candidatas[Math.floor(Math.random() * candidatas.length)]}'... estive pensando nisso enquanto dormia.`;
        }
        return "Oi.";
    }
}

// ==================================================================
// EXECUÇÃO PRINCIPAL
// ==================================================================
console.log("🧬 Iniciando Organismo Soberano v31.1 (refinado)...");
const org = new OrganismoSoberano();
org.boot();
console.log(`🧠: ${org.despertar()}`);

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function perguntar() {
    rl.question('\n👤: ', (input) => {
        const u = input.trim();
        if (!u) { perguntar(); return; }
        if (u.toLowerCase() === 'sair') {
            console.log("👋 Encerrando...");
            org.dormir();
            rl.close();
            process.exit(0);
        }
        try {
            console.log(`🧠: ${org.processar(u)}`);
        } catch (e) {
            console.log(`❌ Erro: ${e.message}`);
        }
        perguntar();
    });
}

perguntar();

process.on('SIGINT', () => {
    console.log("\n👋 Interrompido. Salvando...");
    org.dormir();
    process.exit(0);
});
