const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const os = require('os');
const readline = require('readline/promises');

// =================================================================
// SEEDED RANDOM UTILITIES (substituto de random.Random)
// =================================================================
class SeededRandom {
    constructor(seed) {
        // Inicializa com hash numérico
        this.state = seed % 2147483647;
        if (this.state <= 0) this.state += 2147483646;
    }
    // Gerador congruencial linear (MINSTD)
    next() {
        this.state = (this.state * 16807) % 2147483647;
        return (this.state - 1) / 2147483646;
    }
    // Box-Muller para distribuição normal
    gauss(mu = 0, sigma = 1) {
        let u = 1 - this.next();
        let v = 1 - this.next();
        let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return z * sigma + mu;
    }
    // Amostra k elementos únicos de um array (população)
    sample(population, k) {
        let n = population.length;
        if (k > n) throw new Error('Sample larger than population');
        let result = population.slice(); // cópia
        // Embaralhamento parcial de Fisher-Yates
        for (let i = 0; i < k; i++) {
            let j = i + Math.floor(this.next() * (n - i));
            [result[i], result[j]] = [result[j], result[i]];
        }
        return result.slice(0, k);
    }
}

// =================================================================
// 1. KERNEL DE FÍSICA E MATEMÁTICA (SSML)
// =================================================================
class SSML_Kernel {
    static get_sparse_vec(token, dims = 5000, sparsity = 30) {
        const hash = crypto.createHash('sha256').update(token).digest('hex');
        // Usa os primeiros 8 caracteres do hash como seed (32 bits)
        const seed = parseInt(hash.substring(0, 8), 16);
        const rng = new SeededRandom(seed);
        const indices = rng.sample([...Array(dims).keys()], sparsity);
        const vec = {};
        for (let i of indices) {
            vec[i] = rng.gauss(0, 1);
        }
        return vec;
    }

    static dot(v1, v2) {
        if (!v1 || !v2) return 0.0;
        let sum = 0;
        for (let key in v1) {
            if (key in v2) sum += v1[key] * v2[key];
        }
        return sum;
    }

    static normalize(v) {
        let norm = 0;
        for (let key in v) norm += v[key] * v[key];
        norm = Math.sqrt(norm) + 1e-9;
        const res = {};
        for (let key in v) res[key] = v[key] / norm;
        return res;
    }

    static rashba_interaction(pathos_vec, momentum_vec, alpha = 0.2) {
        const p1 = pathos_vec[0] || 0.1;
        const m1 = momentum_vec[1] || 0.1;
        return alpha * (p1 * m1);
    }
}

// =================================================================
// 2. QUINTIKUS SSML - ARQUITETURA SOBERANA
// =================================================================
class QuintikusSSML {
    constructor() {
        this.dims = 5000;
        this.path_bin = "brain_sovereign.qssml.json";   // JSON para serialização
        this.path_ledger = "ledger.json";
        this.auto_train_files = ["oi.txt", "amor.txt", "conversa.txt", "confusa.txt"];

        this.mapa_nd = {};           // token -> sparse vector
        this.l2_episodes = [];       // { t, v, origin }
        this.neuronios = new Map();  // token -> array de índices
        this.raridade = {};          // token -> contagem
        this.ledger = new Set();

        this.psi_logos = {};
        this.psi_pathos = {};
        this.thermal_pressure = 0.5;

        this.tokenizer = /[\w]+|[?!.]/g;
        this.fatigue = {};           // índice -> valor
    }

    _atomic_save(data, filepath) {
        const dir = path.dirname(path.resolve(filepath));
        const tmpFile = path.join(dir, `tmp_qssml_${Date.now()}_${Math.random().toString(36).substring(2)}.json`);
        try {
            fs.writeFileSync(tmpFile, JSON.stringify(data, null, 2), 'utf8');
            fs.renameSync(tmpFile, filepath);
        } catch (e) {
            if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
            console.error(`⚠️ Erro ao salvar ${filepath}: ${e.message}`);
        }
    }

    salvar() {
        console.log("💾 Cristalizando Memória Binária Atômica...");
        const brain_data = {
            nexus: this.l2_episodes,
            raridade: this.raridade,
            nd: this.mapa_nd,
            logos: this.psi_logos,
            pathos: this.psi_pathos,
            thermal: this.thermal_pressure
        };
        this._atomic_save(brain_data, this.path_bin);
        this._atomic_save([...this.ledger], this.path_ledger);
    }

    boot() {
        if (fs.existsSync(this.path_bin)) {
            const d = JSON.parse(fs.readFileSync(this.path_bin, 'utf8'));
            this.l2_episodes = d.nexus;
            this.raridade = d.raridade;
            this.mapa_nd = d.nd;
            this.psi_logos = d.logos;
            this.psi_pathos = d.pathos;
            this.thermal_pressure = d.thermal;

            for (let i = 0; i < this.l2_episodes.length; i++) {
                const tokens = this.l2_episodes[i].t.toLowerCase().match(this.tokenizer) || [];
                for (let t of tokens) {
                    if (!this.neuronios.has(t)) this.neuronios.set(t, []);
                    this.neuronios.get(t).push(i);
                }
            }
            console.log(`✅ SSML Online. ${this.l2_episodes.length} nexos carregados.`);
        }

        if (fs.existsSync(this.path_ledger)) {
            this.ledger = new Set(JSON.parse(fs.readFileSync(this.path_ledger, 'utf8')));
        }

        for (let arq of this.auto_train_files) {
            if (fs.existsSync(arq)) {
                const conteudo = fs.readFileSync(arq, 'utf8');
                const h = crypto.createHash('sha256').update(conteudo).digest('hex');
                if (!this.ledger.has(h)) {
                    console.log(`🔄 Novo Solo Detectado: ${arq}. Cristalizando...`);
                    this.cristalizar_solo(conteudo);
                    this.ledger.add(h);
                    this.salvar();
                }
            }
        }
    }

    cristalizar_solo(texto, origin = "first_person") {
        const frases = texto.split(/[.!?\n]+/);
        for (let f of frases) {
            f = f.trim();
            if (f.length < 3) continue;

            const tokens = f.toLowerCase().match(this.tokenizer) || [];
            const idx = this.l2_episodes.length;
            let v_nexus = {};

            for (let t of tokens) {
                this.raridade[t] = (this.raridade[t] || 0) + 1;
                if (!this.neuronios.has(t)) this.neuronios.set(t, []);
                this.neuronios.get(t).push(idx);

                if (!(t in this.mapa_nd)) {
                    this.mapa_nd[t] = SSML_Kernel.get_sparse_vec(t);
                }

                const peso = 1.0 / (Math.log(this.raridade[t] + 1.2) + 1e-5);
                v_nexus = this._add_vectors(v_nexus, this.mapa_nd[t], 1.0, peso);
            }

            this.l2_episodes.push({
                t: f,
                v: SSML_Kernel.normalize(v_nexus),
                origin: origin
            });
        }
    }

    _add_vectors(v1, v2, w1, w2) {
        const res = {};
        for (let d in v1) res[d] = v1[d] * w1;
        for (let d in v2) {
            res[d] = (res[d] || 0) + v2[d] * w2;
        }
        return res;
    }

    processar(entrada) {
        const t0 = performance.now();
        const u_toks = entrada.toLowerCase().match(this.tokenizer) || [];
        if (u_toks.length === 0) return "...";

        // Dinâmica Térmica
        let p_inc = 0;
        let e_inc = 0;
        for (let x of u_toks) {
            if (["não", "por que", "falha", "erro", "confuso"].includes(x)) p_inc += 0.12;
            if (["amo", "lindo", "sorriso", "feliz", "jeito"].includes(x)) e_inc += 0.08;
        }
        this.thermal_pressure = Math.min(1.0, this.thermal_pressure * 0.85 + p_inc + e_inc);

        let v_in = {};
        for (let t of u_toks) {
            if (t in this.mapa_nd) {
                const w = 1.0 / (Math.log((this.raridade[t] || 1) + 1.2) + 1e-5);
                v_in = this._add_vectors(v_in, this.mapa_nd[t], 1.0, w);
            }
        }
        v_in = SSML_Kernel.normalize(v_in);

        // Pivô: token mais raro
        const pivo = u_toks.reduce((a, b) =>
            (this.raridade[a] ?? 9999) < (this.raridade[b] ?? 9999) ? a : b
        );
        let candidatos = this.neuronios.get(pivo) || [];

        if (candidatos.length === 0) {
            // fallback: amostra aleatória de todos os episódios
            const todos = [...Array(this.l2_episodes.length).keys()];
            candidatos = new SeededRandom(Date.now()).sample(todos, Math.min(todos.length, 100));
        }

        let melhor_nexo = -1;
        let max_vibration = -Infinity;

        // Amostra até 150 candidatos
        const shuffled = candidatos.slice().sort(() => 0.5 - Math.random());
        const amostra = shuffled.slice(0, Math.min(shuffled.length, 150));

        for (let idx of amostra) {
            const ep = this.l2_episodes[idx];
            const sim_l = SSML_Kernel.dot(v_in, ep.v);
            const sim_p = SSML_Kernel.dot(this.psi_pathos, ep.v);
            const tunneling = Math.exp(-(1.0 - sim_l) / (this.thermal_pressure + 1e-9));
            const vibration = (sim_l * 0.6) + (sim_p * 0.3) + tunneling - (this.fatigue[idx] || 0);

            if (vibration > max_vibration) {
                max_vibration = vibration;
                melhor_nexo = idx;
            }
        }

        // Gatilho de Subconsciente
        if (max_vibration < 0.15) {
            this.thermal_pressure = Math.min(1.0, this.thermal_pressure + 0.2);
            const sub_pivo = "confusa";
            const sub_candidatos = this.neuronios.get(sub_pivo) || [];
            if (sub_candidatos.length > 0) {
                melhor_nexo = sub_candidatos[Math.floor(Math.random() * sub_candidatos.length)];
                console.log(`🌀 [SUBCONSCIENTE ATIVADO] Vibe: ${max_vibration.toFixed(2)}`);
            }
        }

        if (melhor_nexo === -1) return "...";

        const target_v = this.l2_episodes[melhor_nexo].v;
        this.psi_pathos = this._add_vectors(this.psi_pathos, target_v, 0.97, 0.03);
        this.psi_pathos = SSML_Kernel.normalize(this.psi_pathos);
        this.fatigue[melhor_nexo] = (this.fatigue[melhor_nexo] || 0) + 5.0;
        for (let k in this.fatigue) this.fatigue[k] *= 0.7;

        const ms = (performance.now() - t0);
        console.log(` ⧉ [SSML] T:${this.thermal_pressure.toFixed(2)} | Vibe:${max_vibration.toFixed(2)} | ${ms.toFixed(1)}ms`);
        return this.l2_episodes[melhor_nexo].t;
    }
}

// =================================================================
// FUNÇÕES DE VOZ E TEXTO (simuladas)
// =================================================================
let tem_voz = false; 
function falar(texto, imprimir = true) {
    if (imprimir) console.log(`🧠 [SSML]: ${texto}`);
    // Não há TTS nativo, apenas console
}

async function ouvir(rl) {
    // Tentativa de input via terminal (síncrono assíncrono)
    console.log("👤 Digite seu comando: ");
    const resposta = await rl.question('');
    return resposta.trim().toLowerCase();
}

// =================================================================
// EXECUÇÃO PRINCIPAL
// =================================================================
(async function main() {
    const ssml = new QuintikusSSML();
    ssml.boot();

    if (ssml.l2_episodes.length === 0) {
        ssml.cristalizar_solo("Eu sou um nexo de lógica pura aguardando solo data.");
        ssml.salvar();
    }

    falar("SSML pronto. Pode falar ou digitar.");
    console.log("Comandos: 'treinar tudo', 'train:arquivo.txt', 'salvar', 'sair'");

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    while (true) {
        const comando = await ouvir(rl);
        if (!comando) continue;

        if (["sair", "exit", "desligar", "tchau"].some(p => comando.includes(p))) {
            falar("Até logo!");
            ssml.salvar();
            break;
        }

        if (comando.startsWith("train:")) {
            const filePath = comando.split(":")[1].trim();
            if (fs.existsSync(filePath)) {
                falar(`📂 Treinando ${filePath}...`);
                const conteudo = fs.readFileSync(filePath, 'utf8');
                ssml.cristalizar_solo(conteudo);
                ssml.salvar();
                falar(`✨ Treinado. Nexos: ${ssml.l2_episodes.length}`);
            } else {
                falar(`❌ Arquivo '${filePath}' não encontrado.`);
            }
            continue;
        }

        if (comando === 'salvar') {
            ssml.salvar();
            falar("💾 Cérebro salvo.");
            continue;
        }

        const resposta = ssml.processar(comando);
        falar(resposta);
    }

    rl.close();
})();
