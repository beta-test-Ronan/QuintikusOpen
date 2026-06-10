#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import sys
import hashlib
import numpy as np
import time
import random
import threading
from collections import Counter, deque, defaultdict

# --- ESCUDO TÉRMICO ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class Quintikus_GNDLM_V140:
    def __init__(self, raw_text, d_model=64, seq_len=16, num_clusters=8):
        self.d = d_model
        self.seq = seq_len
        self.k_clusters = num_clusters
        
        # Parâmetros de escala geométrica adaptativa
        self.scale = np.float32(1.0 / (d_model ** 0.5))
        self.gamma_base = np.float32(0.15)
        self.mask = (np.tril(np.ones((seq_len, seq_len), dtype=np.float32)) - 1) * 1e9
        
        # --- ATRIBUTOS DA MATRIX DLM (GRAFO DINÂMICO E RELACIONAL) ---
        self.matrix = {} 
        self.blocos = [] 
        self.blocos_xyz = {}  # Coordenadas 3D dinâmicas (GEOMETRIA APRENDIDA)
        self.relacoes = defaultdict(lambda: defaultdict(set))  # Estrutura: [entidade][tipo_relacao] = {atributos}
        self.estados = [0.5, 0.5]  # [Temperatura/Pressão, Sinergia/Harmonia]
        self.rastro = deque(maxlen=10)
        self.termometro = {'erro': -0.3, 'falha': -0.2, 'ruído': -0.1, 'bom': 0.2, 'sinergia': 0.3, 'paz': 0.2}
        
        # Tipos de relações mapeadas explicitamente
        self.tipos_relacao = {"é", "tem", "usa", "causa", "vive_em", "precisa"}
        
        # 1. PROCESSAMENTO DE TOKENS INICIAL
        self.tokens = raw_text.lower().replace(".", " . ").replace(",", " , ").split()
        
        # Construção do vocabulário dinâmico
        palavras_unicas = sorted(list(set(self.tokens)))
        self.vocab = {p: i for i, p in enumerate(palavras_unicas)}
        self.ivocab = {i: p for i, p in enumerate(palavras_unicas)}
        self.vs = len(self.vocab)
        
        # 2. CONSTRUÇÃO INICIAL DO GRAFO, RELAÇÕES E COORDENADAS 3D
        self.build_dlm_matrix()
        self.extrair_triplas_relacionais(self.tokens)
        
        # 3. INICIALIZAÇÃO NEURAL E CANAL DE ABSTRAÇÃO
        f = lambda i, o: (np.random.randn(i, o).astype(np.float32) * np.sqrt(2.0 / i))
        d_bottleneck = d_model // 2
        
        self.weights = {
            'emb': f(self.vs, self.d) * 0.1,
            'pos': f(self.seq, self.d) * 0.1,
            'xyz_proj': f(3, self.d) * 0.1,
            'qkv': f(self.d, self.d * 3),
            'ff_down': f(self.d, d_bottleneck),
            'ff_up': f(d_bottleneck, self.d * 4),
            'ff2': f(self.d * 4, self.d)
        }
        
        self.m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.step = 0
        
        self.short_term_memory = [] 
        self.running = True
        self.lock = threading.Lock()

    def _rms(self, x):
        s = (np.mean(x**2, axis=-1, keepdims=True) + 1e-6)**-0.5
        return x * s, s

    def _rms_backward(self, dy, x, s):
        y = x * s
        return s * (dy - y * np.mean(dy * y, axis=-1, keepdims=True))

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)

    def get_id_geometrico(self, tokens_bloco):
        raras = sorted(tokens_bloco, key=lambda x: self.matrix.get(x, {"m": 0})["m"], reverse=True)[:5]
        h = hashlib.sha256(" ".join(raras).encode()).hexdigest()
        x = ((int(h[:4], 16) % 200) - 100) / 100.0
        y = ((int(h[4:8], 16) % 200) - 100) / 100.0
        z = ((int(h[8:12], 16) % 200) - 100) / 100.0
        return h[:4], np.array([x, y, z], dtype=np.float32)

    def build_dlm_matrix(self):
        freq = Counter(self.tokens)
        for t in freq:
            self.matrix[t] = {"m": 1.5 / (freq[t] + 1e-5), "links": Counter()}

        for i in range(len(self.tokens) - 1):
            self.matrix[self.tokens[i]]["links"][self.tokens[i+1]] += 1

        tamanho_bloco = 12
        for i in range(0, len(self.tokens), tamanho_bloco):
            bloco_tokens = self.tokens[i:i+tamanho_bloco]
            if not bloco_tokens: 
                continue
            id_b, xyz = self.get_id_geometrico(bloco_tokens)
            self.blocos.append({"id": id_b, "xyz": xyz, "txt": bloco_tokens})
            self.blocos_xyz[id_b] = xyz.copy()

    def extrair_triplas_relacionais(self, tokens):
        """
        Mapeia triplas lógicas explícitas da estrutura: Entidade -> Relação -> Atributo
        """
        for i in range(len(tokens) - 2):
            sujeito = tokens[i]
            verbo = tokens[i+1]
            objeto = tokens[i+2]
            if verbo in self.tipos_relacao:
                self.relacoes[sujeito][verbo].add(objeto)

    def _atualizar_grafo_dinamico(self, novos_tokens):
        with self.lock:
            vocab_modificado = False
            for t in novos_tokens:
                if t not in self.vocab:
                    self.vocab[t] = self.vs
                    self.ivocab[self.vs] = t
                    self.vs += 1
                    vocab_modificado = True
                    
                    nova_linha = (np.random.randn(1, self.d).astype(np.float32) * np.sqrt(2.0 / self.d)) * 0.1
                    self.weights['emb'] = np.vstack([self.weights['emb'], nova_linha])
                    self.m['emb'] = np.vstack([self.m['emb'], np.zeros((1, self.d), dtype=np.float32)])
                    self.v['emb'] = np.vstack([self.v['emb'], np.zeros((1, self.d), dtype=np.float32)])
                
                if t not in self.matrix:
                    self.matrix[t] = {"m": 1.5, "links": Counter()}
            
            for i in range(len(novos_tokens) - 1):
                t_at = novos_tokens[i]
                t_px = novos_tokens[i+1]
                self.matrix[t_at]["links"][t_px] += 1
                total_f = sum(self.matrix[w]["links"][t_px] for w in self.matrix if t_px in self.matrix[w]["links"]) + 1
                self.matrix[t_px]["m"] = 1.5 / (total_f + 1e-5)
                
            self.extrair_triplas_relacionais(novos_tokens)
                
            if vocab_modificado and len(novos_tokens) >= 5:
                id_b, xyz = self.get_id_geometrico(novos_tokens)
                self.blocos.append({"id": id_b, "xyz": xyz, "txt": novos_tokens})
                self.blocos_xyz[id_b] = xyz.copy()

    def atualizar_termica(self, tokens_in):
        for t in tokens_in:
            if t in self.termometro:
                val = self.termometro[t]
                if val < 0: 
                    self.estados[0] = min(1.0, self.estados[0] + abs(val))
                else: 
                    self.estados[1] = min(1.0, self.estados[1] + val)
        self.estados[0] *= 0.95
        self.estados[1] *= 0.95

    def calcular_prior_relacional(self, xb):
        t = xb.shape[1]
        R = np.zeros((t, t), dtype=np.float32)
        for i in range(t):
            w_i = self.ivocab[xb[0, i]]
            links_i = self.matrix.get(w_i, {}).get("links", {})
            for j in range(t):
                if i == j: 
                    continue
                w_j = self.ivocab[xb[0, j]]
                
                eh_atributo_restrito = False
                for entidade, rels in self.relacoes.items():
                    if entidade != w_i:
                        for rel, objs in rels.items():
                            if w_j in objs:
                                eh_atributo_restrito = True
                                break
                
                tem_relacao_direta = False
                for rel, objs in self.relacoes[w_i].items():
                    if w_j in objs:
                        tem_relacao_direta = True
                        break
                
                if w_j in links_i:
                    R[i, j] = 1.5 if tem_relacao_direta else 0.5
                else:
                    caminho_abstrato = False
                    if "é" in self.relacoes[w_i]:
                        for classe_b in self.relacoes[w_i]["é"]:
                            for rel_b, objs_b in self.relacoes[classe_b].items():
                                if w_j in objs_b:
                                    caminho_abstrato = True
                                    break
                    
                    if caminho_abstrato:
                        R[i, j] = 1.5  
                    elif eh_atributo_restrito:
                        R[i, j] = -1e9  
                        
        return R

    def _think(self, x, xyz_offset, r_prior_full):
        t = x.shape[1]
        
        h = self.weights['emb'][x] + self.weights['pos'][:t]
        xyz_proj = xyz_offset @ self.weights['xyz_proj']  
        h += xyz_proj[:, None, :]  
        
        n1, s1 = self._rms(h)
        
        segment_size = t // self.k_clusters
        if segment_size < 1: 
            segment_size = 1
            
        micro_clusters = []
        for i in range(self.k_clusters):
            start = i * segment_size
            end = start + segment_size if i < self.k_clusters - 1 else t
            cluster_mean = np.mean(n1[:, start:end, :], axis=1, keepdims=True)
            micro_clusters.append(cluster_mean)
        
        h_clustered = np.concatenate(micro_clusters, axis=1)
        
        qkv = h_clustered @ self.weights['qkv']
        q, k, v = qkv[..., :self.d], qkv[..., self.d:self.d*2], qkv[..., self.d*2:]
        
        gamma_adaptativo = self.gamma_base * (1.0 + self.estados[1])
        
        q_sq = np.sum(q**2, axis=-1, keepdims=True)
        k_sq = np.sum(k**2, axis=-1, keepdims=True).transpose(0, 2, 1)
        qk = np.einsum('btd,bsd->bts', q, k, optimize=True)
        dist_geometrica = q_sq - 2.0 * qk + k_sq
        
        caos = self.estados[0]
        foco = 1.0 + self.estados[1]
        
        scores = ((self.scale * qk) - (gamma_adaptativo * dist_geometrica)) * foco
        if caos > 0.1:
            scores += np.random.normal(0, caos * 0.05, scores.shape)
            
        R_c = np.zeros((self.k_clusters, self.k_clusters), dtype=np.float32)
        for i in range(self.k_clusters):
            st_i = i * segment_size
            en_i = st_i + segment_size if i < self.k_clusters - 1 else t
            for j in range(self.k_clusters):
                st_j = j * segment_size
                en_j = st_j + segment_size if j < self.k_clusters - 1 else t
                R_c[i, j] = np.mean(r_prior_full[st_i:en_i, st_j:en_j])
                
        scores += 2.0 * R_c * foco
        
        mask_c = (np.tril(np.ones((self.k_clusters, self.k_clusters), dtype=np.float32)) - 1) * 1e9
        scores += mask_c
        
        probs = self.softmax(scores)
        attn_out = probs @ v
        
        attn_expanded = np.repeat(attn_out, segment_size, axis=1)[:, :t, :]
        if attn_expanded.shape[1] < t:
            pad_len = t - attn_expanded.shape[1]
            attn_expanded = np.concatenate([attn_expanded, attn_expanded[:, -pad_len:, :]], axis=1)
            
        h1 = h + attn_expanded
        n2, s2 = self._rms(h1)
        
        # Canal de Abstração Bottleneck
        ff_mid = np.maximum(0, n2 @ self.weights['ff_down'])  
        ff1 = np.maximum(0, ff_mid @ self.weights['ff_up'])    
        ffn_out = ff1 @ self.weights['ff2']
        
        h2 = h1 + ffn_out
        logits = h2 @ self.weights['emb'].T
        
        return logits, probs, q, k, v, h, n1, s1, attn_out, h1, n2, s2, ff_mid, ff1, h2, segment_size, gamma_adaptativo

    def atualizar_sinapses(self, seq_indices, bloco_id, start_idx=None, lr_custom=None):
        if len(seq_indices) < self.seq + 1:
            return
            
        if start_idx is None:
            max_start = len(seq_indices) - self.seq - 1
            start = np.random.randint(0, max_start + 1)
        else:
            start = start_idx
        
        xb = np.array([seq_indices[start : start + self.seq]])
        yb = np.array([seq_indices[start + 1 : start + self.seq + 1]])
        
        xyz_offset = self.blocos_xyz[bloco_id]
        xyz_batch = np.array([xyz_offset], dtype=np.float32)
        
        r_prior_full = self.calcular_prior_relacional(xb)
        
        with self.lock:
            logits, probs, q, k, v, h, n1, s1, attn_out, h1, n2, s2, ff_mid, ff1, h2, seg_sz, gamma_ad = self._think(xb, xyz_batch, r_prior_full)
            p = self.softmax(logits)
            
            g_logits = p.copy()
            g_logits[0, np.arange(self.seq), yb[0]] -= 1
            g_logits /= self.seq
            
            g_emb_out = g_logits.reshape(-1, self.vs).T @ h2.reshape(-1, self.d)
            g_h2 = g_logits @ self.weights['emb']
            
            g_ffn_out = g_h2
            g_ff2 = ff1.reshape(-1, self.d*4).T @ g_ffn_out.reshape(-1, self.d)
            g_ff1_post = g_ffn_out @ self.weights['ff2'].T
            g_ff1_pre = g_ff1_post * (ff1 > 0)
            
            g_ff_up = ff_mid.reshape(-1, self.d//2).T @ g_ff1_pre.reshape(-1, self.d*4)
            g_ff_mid_post = g_ff1_pre @ self.weights['ff_up'].T
            g_ff_mid_pre = g_ff_mid_post * (ff_mid > 0)
            
            g_ff_down = n2.reshape(-1, self.d).T @ g_ff_mid_pre.reshape(-1, self.d//2)
            g_n2 = g_ff_mid_pre @ self.weights['ff_down'].T
            
            g_h1 = g_h2 + self._rms_backward(g_n2, h1, s2)
            
            g_attn_expanded = g_h1
            g_attn_out = np.zeros_like(attn_out)
            for i in range(self.k_clusters):
                st_idx = i * seg_sz
                en_idx = st_idx + seg_sz if i < self.k_clusters - 1 else self.seq
                g_attn_out[:, i, :] = np.sum(g_attn_expanded[:, st_idx:en_idx, :], axis=1)
                
            g_v = np.einsum('bts,btd->bsd', probs, g_attn_out)
            g_probs = np.einsum('btd,bsd->bts', g_attn_out, v)
            g_scores = probs * (g_probs - np.sum(g_probs * probs, axis=-1, keepdims=True))
            
            sum_g_scores_t = np.sum(g_scores, axis=-1, keepdims=True)
            sum_g_scores_s = np.sum(g_scores, axis=1, keepdims=True).transpose(0, 2, 1)
            
            foco = 1.0 + self.estados[1]
            g_q = ((self.scale + 2.0 * gamma_ad) * (g_scores @ k) - 2.0 * gamma_ad * (sum_g_scores_t * q)) * foco
            g_k = ((self.scale + 2.0 * gamma_ad) * (g_scores.transpose(0, 2, 1) @ q) - 2.0 * gamma_ad * (sum_g_scores_s * k)) * foco
            
            g_qkv_condensed = np.concatenate([g_q, g_k, g_v], axis=-1)
            g_n1_condensed = g_qkv_condensed @ self.weights['qkv'].T
            
            g_n1 = np.zeros_like(n1)
            for i in range(self.k_clusters):
                st_idx = i * seg_sz
                en_idx = st_idx + seg_sz if i < self.k_clusters - 1 else self.seq
                g_n1[:, st_idx:en_idx, :] = g_n1_condensed[:, i:i+1, :] / (en_idx - st_idx)
                
            g_qkv = np.zeros_like(self.weights['qkv'])
            for i in range(self.k_clusters):
                st_idx = i * seg_sz
                en_idx = st_idx + seg_sz if i < self.k_clusters - 1 else self.seq
                g_qkv += np.einsum('btd,btf->df', np.mean(n1[:, st_idx:en_idx, :], axis=1, keepdims=True), g_qkv_condensed[:, i:i+1, :])
            
            g_h = g_h1 + self._rms_backward(g_n1, h, s1)
            g_xyz_proj = np.einsum('bd,btf->df', xyz_batch, g_h)
            
            g_xyz = np.sum(g_h, axis=1) @ self.weights['xyz_proj'].T
            
            g_emb_in = np.zeros_like(self.weights['emb'])
            np.add.at(g_emb_in, xb, g_h)
            g_pos = np.sum(g_h, axis=0)
            
            self.step += 1
            lr = lr_custom if lr_custom is not None else 0.001
            b1, b2, eps = 0.9, 0.999, 1e-8
            
            self.blocos_xyz[bloco_id] -= lr * g_xyz[0]
            
            grads = {
                'emb': g_emb_in + g_emb_out, 'pos': g_pos, 'xyz_proj': g_xyz_proj,
                'qkv': g_qkv, 'ff_down': g_ff_down, 'ff_up': g_ff_up, 'ff2': g_ff2
            }
            
            for key, grad in grads.items():
                grad = np.clip(grad, -1.0, 1.0)
                self.m[key] = b1 * self.m[key] + (1 - b1) * grad
                self.v[key] = b2 * self.v[key] + (1 - b2) * (grad**2)
                mh = self.m[key] / (1 - b1**self.step)
                vh = self.v[key] / (1 - b2**self.step)
                self.weights[key] -= lr * mh / (np.sqrt(vh) + eps)

    def save(self, filepath="brain_NDLM.npz"):
        with self.lock:
            relacoes_puras = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.relacoes.items()}
            matrix_pura = {k: {"m": v["m"], "links": dict(v["links"])} for k, v in self.matrix.items()}
            
            np.savez_compressed(
                filepath,
                weights=self.weights,
                m=self.m,
                v=self.v,
                vocab=self.vocab,
                ivocab=self.ivocab,
                vs=self.vs,
                blocos=self.blocos,
                blocos_xyz=self.blocos_xyz,
                relacoes=relacoes_puras,
                matrix=matrix_pura,
                tokens=self.tokens,
                step=self.step
            )
            print(f"💾 Estado cognitivo salvo com sucesso em: {filepath}")

    def load(self, filepath="brain_NDLM.npz"):
        if not os.path.exists(filepath):
            return False
        try:
            with self.lock:
                data = np.load(filepath, allow_pickle=True)
                self.weights = data['weights'].item()
                self.m = data['m'].item()
                self.v = data['v'].item()
                self.vocab = data['vocab'].item()
                self.ivocab = data['ivocab'].item()
                self.vs = int(data['vs'])
                self.blocos = list(data['blocos'])
                self.blocos_xyz = data['blocos_xyz'].item()
                
                relacoes_raw = data['relacoes'].item()
                self.relacoes = defaultdict(lambda: defaultdict(set))
                for k, v in relacoes_raw.items():
                    for kk, vv in v.items():
                        self.relacoes[k][kk] = set(vv)
                        
                matrix_raw = data['matrix'].item()
                self.matrix = {}
                for k, v in matrix_raw.items():
                    self.matrix[k] = {"m": v["m"], "links": Counter(v["links"])}
                    
                self.tokens = list(data['tokens'])
                self.step = int(data['step'])
                print(f"📥 Estado cognitivo carregado de: {filepath} ({self.vs} termos, Passo: {self.step})")
                return True
        except Exception as e:
            print(f"⚠️ Erro ao carregar arquivo compactado: {e}")
            return False

    def pre_treinar_base(self, epocas=140):
        print("⏳ Geometrizando o espaço latente: estruturando o manifold por micro-clusters...")
        num_janelas = len(self.tokens) - self.seq - 1
        indices_totais = [self.vocab[t] for t in self.tokens]
        
        t_inicio = time.perf_counter()
        for epoca in range(epocas):
            for start in range(num_janelas):
                fatia_tokens = self.tokens[start : start + self.seq]
                melhor_bloco = self.localizar_melhor_bloco(fatia_tokens)
                self.atualizar_sinapses(indices_totais, melhor_bloco["id"], start_idx=start, lr_custom=0.002)
                
            if epoca % 35 == 0 or epoca == epocas - 1:
                print(f"   Mapeamento NDLM: Época {epoca:03d}/{epocas:03d}")
        t_fim = time.perf_counter()
        print(f"✅ Manifold NDLM Consolidado em {t_fim - t_inicio:.2f}s!")
        self.save()

    def finetune(self, novo_texto, epocas=60):
        print(f"🔧 Iniciando ajuste fino (Finetuning)...")
        novos_tokens = novo_texto.lower().replace(".", " . ").replace(",", " , ").split()
        
        self._atualizar_grafo_dinamico(novos_tokens)
        self.tokens.extend(novos_tokens)
        
        num_janelas = len(novos_tokens) - self.seq - 1
        if num_janelas <= 0:
            print("⚠️ Novo texto curto demais para treinamento estruturado.")
            self.save()
            return
            
        indices_totais = [self.vocab[t] for t in novos_tokens]
        for e in range(epocas):
            for start in range(num_janelas):
                fatia_tokens = novos_tokens[start : start + self.seq]
                melhor_bloco = self.localizar_melhor_bloco(fatia_tokens)
                self.atualizar_sinapses(indices_totais, melhor_bloco["id"], start_idx=start, lr_custom=0.001)
                
        print(f"✅ Ajuste Fino Concluído! Novo vocabulário: {self.vs} termos.")
        self.save()

    def localizar_melhor_bloco(self, tokens_consulta):
        qs = set(tokens_consulta)
        return max(self.blocos, key=lambda b: len(qs.intersection(b["txt"])), default=self.blocos[0])

    def responder_em_tempo_real(self, prompt_usuario):
        ql = prompt_usuario.lower().split()
        
        self._atualizar_grafo_dinamico(ql)
        self.atualizar_termica(ql)
        
        melhor_bloco = self.localizar_melhor_bloco(ql)
        bloco_id = melhor_bloco["id"]
        xyz_offset = self.blocos_xyz[bloco_id]
        
        contexto_indices = [self.vocab[t] for t in ql]
        for idx in contexto_indices:
            self.short_term_memory.append(idx)
            
        while len(self.short_term_memory) > 128:
            self.short_term_memory.pop(0)

        # PLASTICIDADE COGNITIVA ONLINE IMEDIATA
        if len(self.short_term_memory) >= self.seq + 1:
            for _ in range(4):  
                self.atualizar_sinapses(self.short_term_memory, bloco_id, lr_custom=0.001)

        contexto_decode = contexto_indices[-self.seq:]
        if len(contexto_decode) < self.seq:
            contexto_decode = [0] * (self.seq - len(contexto_decode)) + contexto_decode
            
        xb_test = np.array([contexto_decode[-self.seq:]])
        r_prior_full = self.calcular_prior_relacional(xb_test)
        
        resposta_tokens = []
        resposta_indices = []
        
        temp_dinamica = 0.12 * (1.0 + self.estados[0] - self.estados[1])
        temp_dinamica = max(0.05, min(1.0, temp_dinamica))
        
        prefixo = "DLM-ACTIVE | "
        if self.estados[0] > 0.5: 
            prefixo = "[SOB PRESSÃO] "
        elif self.estados[1] > 0.5: 
            prefixo = "[SINERGIA] "
            
        # --- FILTRO CONTRA DILUIÇÃO DO TRUST GATE ---
        conhece_entidade = any(t in self.relacoes for t in ql)
        
        len_prompt = len(ql)
        if len_prompt > 1:
            sub_prior = r_prior_full[:len_prompt, :len_prompt]
            path_strength = np.max(sub_prior)  
        else:
            path_strength = 1.0  

        # O gate só bloqueia se o modelo não conhecer a entidade E não encontrar caminhos no prior
        if len_prompt > 1 and not conhece_entidade and path_strength < 0.15:
            print(f"CÉREBRO: {prefixo}Não tenho informações lógicas suficientes para responder com certeza.")
            return

        print(f"CÉREBRO: {prefixo}", end="", flush=True)
        
        for _ in range(35):
            input_arr = np.array([contexto_decode[-self.seq:]])
            with self.lock:
                logits, *_ = self._think(input_arr, np.array([xyz_offset], dtype=np.float32), r_prior_full)
            
            logits_finais = logits[0, -1, :].copy()
            for idx_passado in resposta_indices[-8:]:
                logits_finais[idx_passado] -= 3.0  
                
            # --- MÁSCARA DE LOGITS BASEADA EM REGRAS SIMBÓLICAS (GUIDED SYMBOLIC LOGIT MASKING) ---
            # Identifica e prioriza os caminhos factuais estritos presentes no Grafo Relacional
            if contexto_decode:
                ultimo_token = self.ivocab[contexto_decode[-1]]
                penultimo_token = self.ivocab[contexto_decode[-2]] if len(contexto_decode) > 1 else ""
                
                # Caso 1: O último token gerado é uma Entidade conhecida no Grafo Relacional
                # Estimula com prioridade absoluta os verbos de ligação registrados para ela (Ex: se ultimo é "maria", boost em "é")
                if ultimo_token in self.relacoes:
                    for rel in self.relacoes[ultimo_token].keys():
                        if rel in self.vocab:
                            logits_finais[self.vocab[rel]] += 150.0
                            
                # Caso 2: O penúltimo é Entidade e o último é um Verbo Relacional (Ex: "maria" e "é")
                # Estimula os atributos ou objetos válidos registrados na tripla (Ex: boost em "rosa")
                if penultimo_token in self.relacoes and ultimo_token in self.relacoes[penultimo_token]:
                    for obj in self.relacoes[penultimo_token][ultimo_token]:
                        if obj in self.vocab:
                            logits_finais[self.vocab[obj]] += 150.0

            sub_logits = logits_finais / (temp_dinamica + 1e-9)
            probs = self.softmax(sub_logits)
            
            if not resposta_indices and np.max(probs) < 0.012:
                print("Não tenho informações lógicas suficientes para responder com certeza.")
                return

            nxt = np.random.choice(self.vs, p=probs)
            word = self.ivocab[nxt]
            
            print(f"{word} ", end="", flush=True)
            resposta_tokens.append(word)
            resposta_indices.append(nxt)
            contexto_decode.append(nxt)
            
            xb_test = np.array([contexto_decode[-self.seq:]])
            r_prior_full = self.calcular_prior_relacional(xb_test)
            
            if word == "." or len(resposta_tokens) > 30:
                break
                
        self._atualizar_grafo_dinamico(resposta_tokens)
        print(f"\n[Status T:{self.estados[0]:.2f} | S:{self.estados[1]:.2f} | Dimensões do Vocabulário: {self.vs}]")


# --- DATASET INICIAL ---
banco_dlm = """
gato é animal .
animal precisa comer , quando gato fica com fome ele precisa de comer .
gato tem de comer comida de animal .
pedra é mineral .
mineral é sólido .
pedra tem massa .
amor é um sentimento humano forte .
sentimento traz paz , harmonia e sinergia entre as pessoas .
quando há erro ou ruído na comunicação , a sinergia cai e a pressão térmica sobe .
o cérebro tenta resolver a falha para recuperar a paz .
joão é verde .
maria é rosa .
"""

if __name__ == "__main__":
    motor = Quintikus_GNDLM_V140(banco_dlm, d_model=64, seq_len=16, num_clusters=8)
    
    if not motor.load("brain_NDLM.npz"):
        print("🆕 Nenhum cérebro salvo encontrado. Iniciando pré-treinamento da base de dados...")
        motor.pre_treinar_base(epocas=140)
    
    print("\n" + "="*60)
    print("QUINTIKUS GNDLM V140: COGNIÇÃO CONCEITUAL-RELACIONAL (D-LEARNING)")
    print("="*60)
    
    while True:
        try:
            print("\nMenu NDLM:")
            print("1. Conversar com o Cérebro")
            print("2. Alimentar novo conhecimento (Finetuning)")
            print("3. Sair")
            opcao = input("Escolha > ").strip()
            
            if opcao == "1":
                while True:
                    p = input("\nINPUT > ").strip()
                    if p.lower() in ['voltar', 'sair']:
                        break
                    motor.responder_em_tempo_real(p)
            elif opcao == "2":
                print("\nDigite ou cole o novo conhecimento estruturado abaixo (frases curtas pontuadas):")
                novo_conhecimento = input("Texto > ").strip()
                if novo_conhecimento:
                    motor.finetune(novo_conhecimento, epocas=60)
            elif opcao == "3" or opcao.lower() == "sair":
                motor.save()
                break
            else:
                print("Opção inválida.")
                
        except KeyboardInterrupt: 
            motor.save()
            break
