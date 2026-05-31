// quintikus_v18.js - Execute com: node quintikus_v18.js
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import readline from 'readline';

// ==================================================================
// ❄️ ESCUDO TÉRMICO
// ==================================================================
process.env.UV_THREADPOOL_SIZE = '1';

// ==================================================================
// 🧹 MÓDULO DE NORMALIZAÇÃO (SEMÂNTICA LIMPA)
// ==================================================================
class TextNormalizer {
    static limpar(texto) {
        if (!texto) return "";
        texto = texto.toLowerCase();
        // Remove acentos (normalização NFD)
        texto = texto.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
        // Mantém apenas letras, números e pontuação básica
        texto = texto.replace(/[^a-z0-9!?.\s]/g, '');
        return texto.trim();
    }
}

class SSML_Kernel {
    static get_sparse_vec(token, dims = 5000, sparsity = 100) {
        const hash = crypto.createHash('sha256').update(token).digest('hex');
        const seed = parseInt(hash.substring(0, 15), 16);
        
        const rng = this._seededRandom(seed);
        const indices = new Set();
        const vec = {};
        
        while (indices.size < sparsity) {
            indices.add(Math.floor(rng() * dims));
        }
        
        for (const i of indices) {
            vec[i] = this._gaussRandom(rng);
        }
        
        return vec;
    }
    
    static _seededRandom(seed) {
        let s = seed;
        return function() {
            s = (s * 1664525 + 1013904223) % 4294967296;
            return s / 4294967296;
        };
    }
    
    static _gaussRandom(rng) {
        let u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
    
    static tsallis_match(v1, v2, q = 0.8) {
        const keys1 = Object.keys(v1);
        const keys2 = new Set(Object.keys(v2));
        const commonKeys = keys1.filter(k => keys2.has(k));
        
        if (commonKeys.length === 0) return 0.0;
        
        let sum_pq = 0;
        for (const k of commonKeys) {
            sum_pq += Math.pow(Math.abs(v1[k] * v2[k]), q);
        }
        
        return (1.0 - sum_pq) / (q - 1.0 + 1e-9);
    }
    
    static dot(v1, v2) {
        if (!v1 || !v2) return 0.0;
        const keys1 = Object.keys(v1);
        const keys2 = new Set(Object.keys(v2));
        const commonKeys = keys1.filter(k => keys2.has(k));
        
        let result = 0;
        for (const k of commonKeys) {
            result += v1[k] * v2[k];
        }
        return result;
    }
    
    static normalize(v) {
        let norm = 0;
        for (const val of Object.values(v)) {
            norm += val * val;
        }
        norm = Math.sqrt(norm);
        
        const result = {};
        for (const [d, val] of Object.entries(v)) {
            result[d] = val / (norm + 1e-9);
        }
        return result;
    }
}

// =================================================================
// 2. MOTOR BIO-LOGIC
// =================================================================
class BioLogicDrive {
    constructor() {
        this.vm = -70.0;
        this.vm_max = -45.0;
        this.vm_min = -90.0;
        this.wavefunction = [0.5, 0.5, 0.5];
    }
    
    pulsar(impacto) {
        this.vm = Math.max(this.vm_min, Math.min(this.vm_max, this.vm + impacto * 12));
        for (let i = 0; i < 3; i++) {
            this.wavefunction[i] = this.wavefunction[i] * 0.75 + (Math.abs(impacto) * 0.25);
        }
    }
    
    get_tau() {
        return ((this.vm - this.vm_min) / (this.vm_max - this.vm_min)) + 0.15;
    }
}

// =================================================================
// 3. QUINTIKUS SSML v18.0 - Ghost Subject
// =================================================================
class QuintikusSSML {
    constructor() {
        this.dims = 5000;
        this.path_bin = "brain_sovereign.qssml";
        this.path_ledger = "ledger.bin";
        this.auto_train_files = ["oi.txt", "amor.txt", "conversa.txt", "confusa.txt", "sentimento.txt"];
        
        this.mapa_nd = {};
        this.l2_episodes = [];
        this.neuronios = {};
        this.raridade = new Map();
        this.ledger = new Set();
        
        this.psi_pathos = {};
        this.tokenizer = /\b\w+\b|[!?.]/g;
        this.fatigue = new Map();
        this.history = [];
        this.turn_count = 0;
        
        // 🧠 CÓRTEX PRÉ-FRONTAL (MEMÓRIA DE TRABALHO)
        this.ctx_foco = {};
        this.ctx_sujeitos_ativos = {};
        this.ctx_inercia = 0.65;
        this.ctx_esquecimento = 0.75;
        this.bio = new BioLogicDrive();
    }
    
    _get_entropy(token) {
        const count = this.raridade.get(token) || 1;
        return 1.0 / (Math.log(count + 1.2) + 1e-5);
    }
    
    _resgatar_sujeito_oculto(u_toks) {
        /** Identifica se a frase é curta e resgata o sujeito do Córtex */
        const conhecidos = u_toks.filter(t => t in this.neuronios);
        
        // Se frase vaga ou sem sujeitos raros conhecidos
        if (conhecidos.length === 0 || u_toks.length <= 2) {
            const sujeitos_vivos = Object.entries(this.ctx_sujeitos_ativos)
                .sort((a, b) => b[1] - a[1]);
            
            if (sujeitos_vivos.length > 0) {
                console.log(`👻 [GHOST SUBJECT] Resgatando: '${sujeitos_vivos[0][0]}'`);
                return sujeitos_vivos[0][0];
            }
        }
        
        // Caso contrário, o novo sujeito mais raro assume
        if (conhecidos.length > 0) {
            return conhecidos.reduce((max, t) => 
                this._get_entropy(t) > this._get_entropy(max) ? t : max
            );
        }
        
        return u_toks[0] || "vazio";
    }
    
    _gerar_proatividade_triade(sujeito, u_toks, nexo_vencedor_idx) {
        const tau = this.bio.get_tau();
        const modo = tau > 0.8 ? 
            (Math.random() < 0.5 ? "CAOS" : "PREDICADO") : 
            (Math.random() < 0.5 ? "SUJEITO" : "PREDICADO");
        
        let hook = "";
        const conectores = ["Aliás,", "Sabe...", "Me veio na mente que", "Fico pensando que", "Mas olha,"];
        
        if (modo === "CAOS") {
            const raros = [];
            for (const [w, c] of this.raridade.entries()) {
                if (c > 1 && c < 15) raros.push(w);
            }
            
            if (raros.length > 0) {
                const pivo = raros[Math.floor(Math.random() * raros.length)];
                const candidatos = this.neuronios[pivo] || [0];
                const idx = candidatos[Math.floor(Math.random() * candidatos.length)];
                hook = `${conectores[Math.floor(Math.random() * conectores.length)]} você já parou pra pensar que ${this.l2_episodes[idx]['t'].toLowerCase()}?`;
            }
        } else if (modo === "SUJEITO") {
            const candidatos = (this.neuronios[sujeito] || []).filter(i => i !== nexo_vencedor_idx);
            
            if (candidatos.length > 0) {
                const idx = candidatos.reduce((max, curr) => 
                    SSML_Kernel.dot(this.psi_pathos, this.l2_episodes[curr]['v']) > 
                    SSML_Kernel.dot(this.psi_pathos, this.l2_episodes[max]['v']) ? curr : max
                );
                hook = `Sobre ${sujeito}, ${this.l2_episodes[idx]['t'].toLowerCase()}, não acha?`;
            } else {
                hook = `O que você realmente acredita sobre ${sujeito}?`;
            }
        } else if (modo === "PREDICADO") {
            const sorted_toks = [...u_toks].sort((a, b) => this._get_entropy(b) - this._get_entropy(a));
            const acao = sorted_toks.length > 1 ? sorted_toks[1] : "isso";
            hook = `E se ${acao} fosse o segredo para tudo?`;
        }
        
        return [modo, hook];
    }
    
    processar(entrada) {
        const t0 = performance.now();
        this.turn_count += 1;
        
        // 1. NORMALIZAÇÃO E LIMPEZA
        const entrada_limpa = TextNormalizer.limpar(entrada);
        const u_toks = entrada_limpa.match(this.tokenizer) || [];
        
        if (u_toks.length === 0) return "...";
        
        // 2. RESGATE DE SUJEITO (NORMAL OU OCULTO)
        const sujeito_atual = this._resgatar_sujeito_oculto(u_toks);
        this.bio.pulsar(this._get_entropy(sujeito_atual));
        
        for (const s of Object.keys(this.ctx_sujeitos_ativos)) {
            this.ctx_sujeitos_ativos[s] *= this.ctx_esquecimento;
        }
        this.ctx_sujeitos_ativos[sujeito_atual] = 1.0;
        
        // 3. VETORES
        let v_in = {};
        for (const t of u_toks) {
            if (t in this.mapa_nd) {
                v_in = this._add_vecs(v_in, this.mapa_nd[t], 1.0, this._get_entropy(t));
            }
        }
        
        if (Object.keys(v_in).length === 0) {
            return "Desculpe, não entendi. Pode reformular?";
        }
        
        v_in = SSML_Kernel.normalize(v_in);
        
        if (Object.keys(this.ctx_foco).length === 0) {
            this.ctx_foco = v_in;
        } else {
            this.ctx_foco = this._add_vecs(this.ctx_foco, v_in, this.ctx_inercia, 1.0 - this.ctx_inercia);
        }
        this.ctx_foco = SSML_Kernel.normalize(this.ctx_foco);
        
        // 4. BUSCA E SCORING
        let candidatos_idx = this.neuronios[sujeito_atual] || [];
        
        if (candidatos_idx.length === 0) {
            if (this.l2_episodes.length === 0) {
                return "Ainda estou aprendendo... Conte-me mais!";
            }
            const sampleSize = Math.min(this.l2_episodes.length, 150);
            candidatos_idx = this._randomSample(
                Array.from({length: this.l2_episodes.length}, (_, i) => i), 
                sampleSize
            );
        }
        
        const scored_data = [];
        const sampleSize = Math.min(candidatos_idx.length, 250);
        const sampledCandidates = this._randomSample(candidatos_idx, sampleSize);
        
        for (const idx of sampledCandidates) {
            const ep = this.l2_episodes[idx];
            if (this.history.includes(ep['t'])) continue;
            
            const s_q = SSML_Kernel.tsallis_match(v_in, ep['v']);
            const sim_f = SSML_Kernel.dot(this.ctx_foco, ep['v']);
            const sim_p = SSML_Kernel.dot(this.psi_pathos, ep['v']);
            
            const fatigue = this.fatigue.get(idx) || 0;
            // Balanço v18: Maior peso para o Foco Frontal
            const score = (s_q * 0.35) + (sim_f * 0.4) + (sim_p * 0.25) - fatigue;
            scored_data.push([idx, score]);
        }
        
        if (scored_data.length === 0) {
            return "Interessante... continue...";
        }
        
        // 5. COLAPSO QUANTUM
        scored_data.sort((a, b) => b[1] - a[1]);
        const top_k = scored_data.slice(0, 10);
        const tau = this.bio.get_tau();
        
        let melhor_idx;
        try {
            const max_s = Math.max(...top_k.map(x => x[1]));
            const exp_vals = top_k.map(x => 
                Math.exp(Math.max(-10, Math.min(10, (x[1] - max_s) / tau)))
            );
            const total = exp_vals.reduce((a, b) => a + b, 0);
            
            const r = Math.random();
            let cumulative = 0;
            melhor_idx = top_k[0][0];
            
            for (let i = 0; i < top_k.length; i++) {
                cumulative += exp_vals[i] / total;
                if (r <= cumulative) {
                    melhor_idx = top_k[i][0];
                    break;
                }
            }
        } catch (e) {
            melhor_idx = top_k[0][0];
        }
        
        // 6. RESPOSTA + PROATIVIDADE
        let res_base = this.l2_episodes[melhor_idx]['t'];
        
        if (this.turn_count % 3 === 0) {
            const [modo, hook] = this._gerar_proatividade_triade(sujeito_atual, u_toks, melhor_idx);
            if (hook) {
                res_base = `${res_base}. ${hook}`;
                console.log(`📡 [PROATIVIDADE: ${modo}]`);
            }
        }
        
        // 7. EVOLUÇÃO
        const v_vencedor = this.l2_episodes[melhor_idx]['v'];
        this.psi_pathos = SSML_Kernel.normalize(
            this._add_vecs(this.psi_pathos, v_vencedor, 0.94, 0.06)
        );
        
        this.history.push(this.l2_episodes[melhor_idx]['t']);
        if (this.history.length > 20) this.history.shift();
        
        this.fatigue.set(melhor_idx, (this.fatigue.get(melhor_idx) || 0) + 15.0);
        
        for (const [k, v] of this.fatigue.entries()) {
            this.fatigue.set(k, v * 0.6);
        }
        
        const dt = (performance.now() - t0);
        console.log(` ⚛️ [Vm: ${this.bio.vm.toFixed(1)}mV | Tau: ${this.bio.get_tau().toFixed(2)}] Subj: ${sujeito_atual} | ${dt.toFixed(1)}ms`);
        
        return res_base;
    }
    
    _add_vecs(v1, v2, w1, w2) {
        const res = {};
        for (const [d, v] of Object.entries(v1)) {
            res[d] = v * w1;
        }
        for (const [d, v] of Object.entries(v2)) {
            res[d] = (res[d] || 0) + (v * w2);
        }
        return res;
    }
    
    _randomSample(arr, size) {
        const shuffled = [...arr];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled.slice(0, size);
    }
    
    boot() {
        if (fs.existsSync(this.path_bin)) {
            try {
                const data = JSON.parse(fs.readFileSync(this.path_bin, 'utf-8'));
                this.l2_episodes = data.nexus || [];
                this.raridade = new Map(Object.entries(data.raridade || {}));
                this.mapa_nd = data.nd || {};
                this.psi_pathos = data.pathos || {};
                
                for (let i = 0; i < this.l2_episodes.length; i++) {
                    const ep = this.l2_episodes[i];
                    const textoLimpo = TextNormalizer.limpar(ep['t']);
                    const tokens = textoLimpo.match(this.tokenizer) || [];
                    for (const t of tokens) {
                        if (!this.neuronios[t]) this.neuronios[t] = [];
                        this.neuronios[t].push(i);
                    }
                }
                
                console.log(`✅ SSML v18.0 Ghost-Subject Online (${this.l2_episodes.length} nexos)`);
            } catch (e) {
                console.log("⚠️ Erro ao carregar brain, iniciando vazio:", e.message);
            }
        } else {
            console.log("🧠 Novo cérebro inicializado");
        }
        
        if (fs.existsSync(this.path_ledger)) {
            try {
                const data = JSON.parse(fs.readFileSync(this.path_ledger, 'utf-8'));
                this.ledger = new Set(data);
            } catch (e) {
                console.log("⚠️ Erro ao carregar ledger:", e.message);
            }
        }
        
        for (const arq of this.auto_train_files) {
            if (fs.existsSync(arq)) {
                try {
                    const c = fs.readFileSync(arq, 'utf-8');
                    const h = crypto.createHash('sha256').update(c).digest('hex');
                    
                    if (!this.ledger.has(h)) {
                        console.log(`📚 Treinando com: ${arq}`);
                        this.cristalizar_solo(c);
                        this.ledger.add(h);
                        this.salvar();
                    }
                } catch (e) {
                    console.log(`⚠️ Erro ao processar ${arq}:`, e.message);
                }
            }
        }
    }
    
    cristalizar_solo(texto) {
        const frases = texto.split(/[.!?\n]+/);
        let count = 0;
        
        for (const f of frases) {
            const f_limpa = TextNormalizer.limpar(f);
            if (f_limpa.length < 3) continue;
            
            const idx = this.l2_episodes.length;
            let v_ep = {};
            const tokens = f_limpa.match(this.tokenizer) || [];
            
            for (const t of tokens) {
                this.raridade.set(t, (this.raridade.get(t) || 0) + 1);
                
                if (!this.neuronios[t]) this.neuronios[t] = [];
                this.neuronios[t].push(idx);
                
                if (!(t in this.mapa_nd)) {
                    this.mapa_nd[t] = SSML_Kernel.get_sparse_vec(t);
                }
                
                v_ep = this._add_vecs(v_ep, this.mapa_nd[t], 1.0, this._get_entropy(t));
            }
            
            this.l2_episodes.push({
                't': f.trim(),
                'v': SSML_Kernel.normalize(v_ep)
            });
            count++;
        }
        
        if (count > 0) {
            console.log(`   ✅ ${count} frases cristalizadas`);
        }
    }
    
    salvar() {
        try {
            this._atomic_save({
                nexus: this.l2_episodes,
                raridade: Object.fromEntries(this.raridade),
                nd: this.mapa_nd,
                pathos: this.psi_pathos
            }, this.path_bin);
            
            this._atomic_save([...this.ledger], this.path_ledger);
            console.log("💾 Cérebro salvo com sucesso!");
        } catch (e) {
            console.log("❌ Erro ao salvar:", e.message);
        }
    }
    
    _atomic_save(data, filepath) {
        const dir = path.dirname(path.resolve(filepath));
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        
        const tmpPath = path.join(dir, `tmp_qssml_${Date.now()}_${Math.random().toString(36).substr(2)}`);
        
        try {
            const jsonStr = JSON.stringify(data);
            fs.writeFileSync(tmpPath, jsonStr);
            fs.renameSync(tmpPath, filepath);
        } catch (e) {
            if (fs.existsSync(tmpPath)) {
                fs.unlinkSync(tmpPath);
            }
            throw e;
        }
    }
}

// =================================================================
// EXECUÇÃO DIRETA
// =================================================================
console.log("🚀 Iniciando Quintikus SSML v18.0 - Ghost Subject...");
console.log("=".repeat(50));

const ssml = new QuintikusSSML();
ssml.boot();

console.log("=".repeat(50));
console.log("💬 Chat iniciado! Digite 'sair' para encerrar.");
console.log("👻 Ghost Subject ativo: frases curtas resgatam contexto anterior");
console.log("");

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function perguntar() {
    rl.question('👤: ', (input) => {
        const u = input.trim();
        
        if (!u) {
            perguntar();
            return;
        }
        
        if (u.toLowerCase() === 'sair' || u.toLowerCase() === 'exit') {
            console.log("\n👋 Salvando e saindo...");
            ssml.salvar();
            rl.close();
            process.exit(0);
            return;
        }
        
        try {
            const resposta = ssml.processar(u);
            console.log(`🧠 Dany: ${resposta}\n`);
        } catch (error) {
            console.log(`❌ Erro: ${error.message}\n`);
            console.error(error);
        }
        
        perguntar();
    });
}

perguntar();

// Capturar Ctrl+C
process.on('SIGINT', () => {
    console.log("\n\n👋 Interrompido! Salvando...");
    ssml.salvar();
    process.exit(0);
});
