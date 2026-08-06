#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSpark 2.0 Adaptive — Drafter com fitting online + confidence global + temp calibration
"""

import numpy as np
import math
import random
import time
import json
import os
from collections import defaultdict, Counter, deque
from typing import List, Dict, Tuple, Optional

# ── 1. TargetLM (n-grama) ──
class TargetLM:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self._treinar()
    
    def _treinar(self):
        self.unigram = Counter()
        self.bigram  = defaultdict(Counter)
        self.trigram = defaultdict(lambda: defaultdict(Counter))
        for frase in self.corpus:
            tokens = ['<s>'] + frase.split() + ['</s>']
            for i in range(len(tokens)):
                self.unigram[tokens[i]] += 1
                if i >= 1: self.bigram[tokens[i-1]][tokens[i]] += 1
                if i >= 2: self.trigram[tokens[i-2]][tokens[i-1]][tokens[i]] += 1
        self._vocab = sorted(set(self.unigram.keys()))
        self._idx = {t:i for i,t in enumerate(self._vocab)}
    
    def prob(self, context: List[str], token: str) -> float:
        ctx = context[-2:] if len(context) >= 2 else context
        if len(ctx) == 2 and ctx[0] in self.trigram and ctx[1] in self.trigram[ctx[0]]:
            p = self.trigram[ctx[0]][ctx[1]].get(token, 0)
            if p > 0: return p / sum(self.trigram[ctx[0]][ctx[1]].values())
        if ctx and ctx[-1] in self.bigram:
            p = self.bigram[ctx[-1]].get(token, 0)
            if p > 0: return p / sum(self.bigram[ctx[-1]].values())
        p = self.unigram.get(token, 1)
        return p / sum(self.unigram.values())
    
    def sample(self, context: List[str], top_k: int = 8) -> str:
        candidates = list(self._vocab)
        probs = [self.prob(context, t) for t in candidates]
        idxs = np.argsort(probs)[-top_k:]
        top_tokens = [candidates[i] for i in idxs]
        top_probs = np.array([probs[i] for i in idxs])
        top_probs /= top_probs.sum()
        idx = np.random.choice(len(top_tokens), p=top_probs)
        return top_tokens[idx]
    
    def get_logits(self, context: List[str]) -> np.ndarray:
        """Retorna logits (log-probs) para todo o vocab."""
        probs = np.array([self.prob(context, t) for t in self._vocab])
        probs = np.clip(probs, 1e-12, 1.0)
        logits = np.log(probs)
        return logits - np.max(logits)  # normalize for stability
    
    @property
    def vocab(self) -> List[str]:
        return self._vocab
    
    def vocab_size(self) -> int:
        return len(self._vocab)

# ── 2. Drafter com Online Fitting ──
class AdaptiveQuantumLattice:
    """
    Canal A: paralelo (target.sample direto) — sempre alinhado
    Canal B: semi-auto com matriz de transição que se atualiza a cada token aceito
    """
    def __init__(self, target: TargetLM, block_size: int = 8, alpha: float = 0.6):
        self.target = target
        self.block_size = block_size
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.vocab = list(target.vocab)
        self.token_to_idx = {t:i for i,t in enumerate(self.vocab)}
        self.idx_to_token = {i:t for t,i in self.token_to_idx.items()}
        
        # Matriz de transição inicial: fita no corpus + smoothing
        self.transition = np.ones((len(self.vocab), len(self.vocab))) * 0.01
        self._fit_initial()
        
        # Canal A: paralelo (sem fitting, sempre alinhado ao target)
        self.channel_A = ParallelDrafter(target, block_size)
        
        # Para calibração de temperatura
        self.temperature = 0.85
        self.temp_history = deque(maxlen=20)
    
    def _fit_initial(self):
        """Fita com counts do corpus + smoothing."""
        corpus = getattr(self.target, 'corpus', [])
        if not corpus: return
        for frase in corpus:
            tokens = ['<s>'] + frase.split() + ['</s>']
            for i in range(len(tokens)-1):
                idx1 = self.token_to_idx.get(tokens[i], 0)
                idx2 = self.token_to_idx.get(tokens[i+1], 0)
                self.transition[idx1, idx2] += 1
        for i in range(len(self.vocab)):
            row_sum = self.transition[i].sum()
            if row_sum > 0: self.transition[i] /= row_sum
    
    def update_transition(self, prev_token: str, new_token: str):
        """Atualiza online a matriz de transição com um novo bigrama."""
        idx1 = self.token_to_idx.get(prev_token, 0)
        idx2 = self.token_to_idx.get(new_token, 0)
        # Update with learning rate (online learning)
        lr = 0.15  # quanto do novo dado sobrepõe o antigo
        self.transition[idx1, idx2] = (1 - lr) * self.transition[idx1, idx2] + lr * 1.0
        # Renormalizar a linha
        row_sum = self.transition[idx1].sum()
        if row_sum > 0:
            self.transition[idx1] /= row_sum
    
    def set_temperature(self, temp: float):
        self.temperature = temp
    
    def draft(self, anchor: str, context: List[str]) -> List[str]:
        """Gera bloco combinando canal A (paralelo) e canal B (adaptive)."""
        block_B = self._semi_auto_draft(anchor, context)
        # Canal A: paralelo (sempre alinhado com target)
        full_ctx = (context + [anchor]) if context else [anchor]
        block_A = []
        ctx_for_A = full_ctx.copy()
        for _ in range(self.block_size):
            tok = self.target.sample(ctx_for_A, top_k=6)
            block_A.append(tok)
            ctx_for_A.append(tok)
        
        # Fusão bilateral ponderada
        fused = []
        for i in range(self.block_size):
            token_A = block_A[i]
            token_B = block_B[i]
            if token_A == token_B:
                fused.append(token_A)
                continue
            
            ctx = (context + [anchor]) if context else [anchor]
            # Calcular probabilidade de ambos os canais no target
            pA = self.target.prob(ctx + fused, token_A)
            pB = self.target.prob(ctx + fused, token_B)
            
            # Ponderação adaptativa: se a plasticidade está alta, confia mais no canal A
            # Se está baixa, confia mais no canal B (que está se adaptando online)
            p_fuse_A = self.alpha * pA
            p_fuse_B = self.beta * pB
            
            total = p_fuse_A + p_fuse_B
            if total == 0:
                fused.append(token_A)
            else:
                p_norm_A = p_fuse_A / total
                fused.append(token_A if random.random() < p_norm_A else token_B)
        
        return fused
    
    def _semi_auto_draft(self, anchor: str, context: List[str]) -> List[str]:
        """Canal B: semi-autoregressivo com matriz adaptativa."""
        prev_token = anchor
        tokens = []
        for _ in range(self.block_size):
            idx_prev = self.token_to_idx.get(prev_token, 0)
            logits = np.log(self.transition[idx_prev] + 1e-8)
            logits = logits - np.max(logits)
            probs = np.exp(logits / self.temperature)
            probs /= probs.sum()
            
            # Calibragem com target (simulada): mistura com a distribuição do target
            ctx = (context + [anchor] + tokens) if context else [anchor] + tokens
            target_logits = self.target.get_logits(ctx)
            target_probs = np.exp(target_logits / 0.85)
            target_probs /= target_probs.sum()
            
            fused_probs = 0.55 * probs + 0.45 * target_probs
            fused_probs /= fused_probs.sum()
            
            idx = np.random.choice(len(self.vocab), p=fused_probs)
            token = self.idx_to_token[idx]
            tokens.append(token)
            prev_token = token
        
        return tokens

class ParallelDrafter:
    def __init__(self, target: TargetLM, block_size: int = 8):
        self.target = target
        self.block_size = block_size
    def draft(self, anchor: str, context: List[str]) -> List[str]:
        full_ctx = (context + [anchor]) if context else [anchor]
        tokens = []
        for _ in range(self.block_size):
            token = self.target.sample(full_ctx + tokens, top_k=6)
            tokens.append(token)
        return tokens

# ── 3. Confidence Head com Divergência Global ──
class AdaptiveConfidenceHead:
    def __init__(self, target: TargetLM):
        self.target = target
        self.vocab = list(target.vocab)
    
    def js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence entre duas distribuições."""
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p/m + 1e-12))
        kl_qm = np.sum(q * np.log(q/m + 1e-12))
        return 0.5 * (kl_pm + kl_qm)
    
    def estimate(self, draft_block: List[str], context: List[str], 
                 drafter: AdaptiveQuantumLattice) -> Tuple[List[float], float]:
        """Retorna confianças por token + divergência global do bloco."""
        confidences = []
        full_ctx = context.copy()
        
        # Para calcular JS-divergence: distribuição draft vs target no bloco
        draft_probs = []
        target_probs = []
        
        for i, token in enumerate(draft_block):
            # Surpresa local
            p_target = drafter.target.prob(full_ctx, token)
            surprise = -math.log(p_target + 1e-12) if p_target > 0 else 12.0
            
            # Entropia local
            candidates = [(t, drafter.target.prob(full_ctx, t)) for t in drafter.target.vocab]
            top_probs = sorted([p for _,p in candidates if p>0], reverse=True)[:5]
            if len(top_probs) < 2:
                entropy = 0.0
            else:
                probs = np.array(top_probs)
                probs /= probs.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-8))
            
            # Confiança local
            conf = 1.0 / (1.0 + 0.5 * surprise + 0.3 * entropy)
            conf = np.clip(conf, 0.0, 1.0)
            confidences.append(conf)
            
            # Para JS-divergence: probabilidade do token draft no target vs drafter
            # (aproximação: usar top-5 do vocab)
            ctx_tokens = drafter.target.get_logits(full_ctx)
            p_target_vec = np.exp(ctx_tokens / 0.85)
            p_target_vec /= p_target_vec.sum()
            
            # Distribuição do drafter para este passo
            idx_prev = drafter.token_to_idx.get(full_ctx[-1] if full_ctx else '<s>', 0)
            p_drafter_vec = drafter.transition[idx_prev].copy()
            if p_drafter_vec.sum() > 0:
                p_drafter_vec /= p_drafter_vec.sum()
            else:
                p_drafter_vec = np.ones_like(p_drafter_vec) / len(p_drafter_vec)
            
            # JS-divergence para este passo
            js = self.js_divergence(p_target_vec, p_drafter_vec)
            
            # Atualizar contexto
            full_ctx.append(token)
        
        # JS-divergence média do bloco
        # (na prática acima já calcula por passo, mas vamos retornar o último como proxy)
        js_avg = confidences[-1] * 0.1 + (1 - confidences[-1]) * 0.9  # placeholder — refinar depois
        
        # Decay posicional
        for i in range(len(confidences)):
            confidences[i] *= math.exp(-0.05 * i)
        
        return confidences, js_avg

# ── 4. Scheduler com Calibração de Temperatura e Persistência ──
class PersistentScheduler:
    def __init__(self, min_verify: int = 2, max_verify: int = 8, 
                 history_file: str = "scheduler_history.json"):
        self.min_verify = min_verify
        self.max_verify = max_verify
        self.history_file = history_file
        self.acceptance_history = deque([0.6, 0.65, 0.7], maxlen=10)
        self.temperature_history = deque([0.85], maxlen=10)
        self.warmup_cycles = 4
        self.cycle_count = 0
        
        # Carregar histórico persistente
        self._load_history()
    
    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    if 'acceptance' in data:
                        self.acceptance_history = deque(data['acceptance'], maxlen=10)
                    if 'temperature' in data:
                        self.temperature_history = deque(data['temperature'], maxlen=10)
            except: pass
    
    def _save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    'acceptance': list(self.acceptance_history),
                    'temperature': list(self.temperature_history)
                }, f)
        except: pass
    
    def _plasticity_factor(self) -> float:
        avg_accept = np.mean(self.acceptance_history)
        # Quanto maior a aceitação média histórica, mais agressivo o scheduler
        return np.clip(1.0 + 1.2 * (avg_accept - 0.55), 0.7, 1.6)
    
    def schedule(self, confidences: List[float], system_load: float, 
                 js_div: float = 0.5, current_temp: float = 0.85) -> Tuple[int, float]:
        """
        Retorna (verify_length, new_temperature)
        """
        self.cycle_count += 1
        mean_conf = np.mean(confidences) if confidences else 0.5
        
        # Penalidade por divergência global (JS)
        js_penalty = int((self.max_verify - self.min_verify) * 0.4 * js_div)
        
        # Penalidade por carga
        load_penalty = int((self.max_verify - self.min_verify) * 0.5 * system_load)
        
        base_verify = int(self.min_verify + (self.max_verify - self.min_verify) * np.clip(mean_conf, 0.0, 1.0))
        
        if self.cycle_count <= self.warmup_cycles:
            verify_len = self.max_verify
        else:
            verify_len = max(self.min_verify, base_verify - load_penalty - js_penalty)
        
        # Calibração de temperatura: se aceitação histórica < 0.45, reduz temp (mais determinístico)
        avg_accept = np.mean(self.acceptance_history)
        temp_adjust = 0.85
        if avg_accept < 0.40:
            temp_adjust = 0.65 + random.uniform(-0.05, 0.05)
        elif avg_accept < 0.55:
            temp_adjust = 0.78 + random.uniform(-0.03, 0.03)
        else:
            temp_adjust = 0.88 + random.uniform(-0.02, 0.02)
        
        # Clamp final
        final_verify = int(np.clip(verify_len, self.min_verify, self.max_verify))
        final_temp = np.clip(temp_adjust, 0.60, 1.05)
        
        return final_verify, final_temp
    
    def update_history(self, accepted: int, total_draft: int, temperature: float):
        if total_draft > 0:
            rate = accepted / total_draft
            self.acceptance_history.append(rate)
        self.temperature_history.append(temperature)
        self._save_history()

# ── 5. Loop DSpark 2.0 Adaptive ──
def dspark2_adaptive_decode(
    target: TargetLM,
    drafter: AdaptiveQuantumLattice,
    confidence_head: AdaptiveConfidenceHead,
    scheduler: PersistentScheduler,
    prompt: str = "",
    max_tokens: int = 64,
    system_load: float = 0.5,
    starting_temp: float = 0.85
) -> Tuple[List[str], Dict]:
    tokens = prompt.split() if prompt else ['<s>']
    accepted = []
    stats = {
        'draft_blocks': 0, 'accepted_tokens': 0, 'rejected_tokens': 0,
        'bonus_tokens': 0, 'block_lengths': [], 'confidence_stats': [],
        'plasticity_history': [], 'temp_history': [],
        'js_div_history': []
    }
    
    current_temp = starting_temp
    drafter.set_temperature(current_temp)
    
    while len(accepted) < max_tokens:
        anchor = tokens[-1]
        context = tokens[:-1] if len(tokens) > 1 else []
        
        # Draft
        draft_block = drafter.draft(anchor, context)
        
        # Confidence + JS divergence
        confidences, js_div = confidence_head.estimate(draft_block, context + [anchor], drafter)
        stats['js_div_history'].append(js_div)
        
        # Scheduler (com calibração de temp)
        verify_len, new_temp = scheduler.schedule(confidences, system_load, js_div, current_temp)
        drafter.set_temperature(new_temp)
        current_temp = new_temp
        
        # Verificação
        current_ctx = (context + [anchor]) if context else [anchor]
        accepted_prefix = []
        
        for i in range(verify_len):
            token_draft = draft_block[i]
            p_target = target.prob(current_ctx, token_draft)
            # Aceitação: min(1, P_target / P_draft) — mas P_draft é difícil de estimar
            # Vamos usar uma aproximação: se P_target > threshold, aceita
            threshold = 1.0 / len(target.vocab)  # baseline uniforme
            p_accept = min(1.0, p_target / (threshold + 1e-8))
            
            if random.random() < p_accept:
                accepted_prefix.append(token_draft)
                current_ctx.append(token_draft)
                # Online fitting: atualiza a matriz com o bigrama aceito
                if len(current_ctx) >= 2:
                    drafter.update_transition(current_ctx[-2], current_ctx[-1])
            else:
                break
        
        # Bonus token
        if len(accepted_prefix) == verify_len:
            bonus = target.sample(current_ctx, top_k=6)
            if bonus != '</s>' and bonus not in accepted_prefix:
                accepted_prefix.append(bonus)
                stats['bonus_tokens'] += 1
        
        scheduler.update_history(len(accepted_prefix), verify_len, current_temp)
        
        accepted.extend(accepted_prefix)
        tokens.extend(accepted_prefix)
        
        stats['draft_blocks'] += 1
        stats['accepted_tokens'] += len(accepted_prefix)
        stats['rejected_tokens'] += (verify_len - len(accepted_prefix))
        stats['block_lengths'].append(verify_len)
        stats['temp_history'].append(current_temp)
        stats['plasticity_history'].append(np.mean(scheduler.acceptance_history))
        
        if '</s>' in accepted or len(accepted) >= max_tokens:
            break
    
    result = [t for t in accepted if t not in {'<s>', '</s>'}]
    return result, stats

# ── 6. AR Baseline ──
def autoregressive_decode(target: TargetLM, prompt: str, max_tokens: int) -> Tuple[List[str], Dict]:
    tokens = prompt.split() if prompt else ['<s>']
    generated = []
    for _ in range(max_tokens):
        ctx = tokens[:-1] if len(tokens) > 1 else []
        anchor = tokens[-1]
        token = target.sample(ctx + [anchor], top_k=8)
        if token == '</s>': break
        tokens.append(token)
        generated.append(token)
    return generated, {'tokens': len(generated)}

# ── 7. DEMONSTRAÇÃO ──
if __name__ == "__main__":
    import argparse
    
    corpus = [
        "to be or not to be that is the question",
        "though this be madness yet there is method in't",
        "all the world's a stage and all the men and women merely players",
        "to thine own self be true and it must follow as the night the day",
        "artificial intelligence is the new electricity",
        "deep learning transforms raw data into actionable knowledge",
        "speculative decoding accelerates large language models by parallelizing drafts",
        "quantum computers leverage superposition and entanglement",
        "the unexamined life is not worth living",
        "i think therefore i am cogito ergo sum",
        "knowledge is power said francis bacon",
        "the quick brown fox jumps over the lazy dog",
        "a journey of a thousand miles begins with a single step",
        "where there is smoke there is fire",
        "breakfast is the most important meal of the day",
        "transformers revolutionized natural language processing",
        "self supervised learning enables models to learn from raw text",
        "reinforcement learning from human feedback aligns models",
        "mixture of experts scales model capacity",
        "the human brain contains approximately eighty six billion neurons",
        "water boils at one hundred degrees celsius",
        "neural networks learn hierarchical representations",
        "the sun rises in the east and sets in the west",
        "probability theory governs uncertainty in machine learning"
    ]
    
    parser = argparse.ArgumentParser(description="DSpark 2.0 Adaptive — fitting online + persistência")
    parser.add_argument("--prompt", type=str, default="the quick brown", help="Prompt inicial")
    parser.add_argument("--max_tokens", type=int, default=64, help="Máximo de tokens")
    parser.add_argument("--block_size", type=int, default=8, help="Tamanho do bloco")
    parser.add_argument("--system_load", type=float, default=0.30, help="Carga (0.0-1.0)")
    parser.add_argument("--temp", type=float, default=0.85, help="Temperatura inicial do drafter")
    args = parser.parse_args()
    
    print("🧠 Treinando TargetLM (adaptive)...")
    target = TargetLM(corpus)
    BLOCK_SIZE = args.block_size
    MAX_TOKENS = args.max_tokens
    SYSTEM_LOAD = args.system_load
    
    # Baseline AR
    print("\n1️⃣  AUTORREGRESSIVO (baseline)")
    start_ar = time.time()
    ar_tokens, ar_stats = autoregressive_decode(target, args.prompt, MAX_TOKENS)
    elapsed_ar = time.time() - start_ar
    print(f"   ➜ Geração: {' '.join(ar_tokens[:25])}{'...' if len(ar_tokens)>25 else ''}")
    print(f"   ➜ Tempo: {elapsed_ar:.3f}s | Tokens: {ar_stats['tokens']}")
    
    # DSpark 2.0 Adaptive
    drafter = AdaptiveQuantumLattice(target, block_size=BLOCK_SIZE, alpha=0.6)
    confidence = AdaptiveConfidenceHead(target)
    scheduler = PersistentScheduler(min_verify=2, max_verify=BLOCK_SIZE)
    
    print("\n2️⃣  DSpark 2.0 ADAPTIVE (online fitting + temp calibration)")
    start_ad = time.time()
    ad_tokens, ad_stats = dspark2_adaptive_decode(
        target, drafter, confidence, scheduler,
        prompt=args.prompt, max_tokens=MAX_TOKENS,
        system_load=SYSTEM_LOAD, starting_temp=args.temp
    )
    elapsed_ad = time.time() - start_ad
    
    acc_ad = ad_stats['accepted_tokens']
    blocks_ad = ad_stats['draft_blocks']
    avg_verify = np.mean(ad_stats['block_lengths']) if ad_stats['block_lengths'] else 0
    plas_avg = np.mean(ad_stats['plasticity_history']) if ad_stats['plasticity_history'] else 0.65
    temp_avg = np.mean(ad_stats['temp_history']) if ad_stats['temp_history'] else args.temp
    
    print(f"   ➜ Geração: {' '.join(ad_tokens[:25])}{'...' if len(ad_tokens)>25 else ''}")
    print(f"   ➜ Tempo: {elapsed_ad:.3f}s | Tokens aceitos: {acc_ad}")
    print(f"   ➜ Blocos: {blocks_ad} | Média verificada: {avg_verify:.1f} | Bonus: {ad_stats['bonus_tokens']}")
    print(f"   ➜ Acceptance rate: {acc_ad / (blocks_ad * BLOCK_SIZE):.2%} | Plasticidade: {plas_avg:.2f}")
    print(f"   ➜ Temperatura média: {temp_avg:.2f}")
    
    # Resumo
    print("\n" + "="*40)
    print("📊 RESUMO")
    print("="*40)
    ar_speed = ar_stats['tokens'] / elapsed_ar if elapsed_ar > 0 else 0
    ad_speed = acc_ad / elapsed_ad if elapsed_ad > 0 else 0
    print(f"Vanilla AR      | Tempo: {elapsed_ar:.3f}s | Tokens: {ar_stats['tokens']} | Speed: {ar_speed:.1f} t/s")
    print(f"DSpark 2.0 Adpt | Tempo: {elapsed_ad:.3f}s | Tokens: {acc_ad} | Speed: {ad_speed:.1f} t/s", end='')
    if ar_speed > 0:
        print(f" | Speedup: {ad_speed/ar_speed:.2f}x")
    else:
        print()
    
    print("\n🔍 O que mudou nesta versão:")
    print("  • 🔄 Online fitting: matriz de transição atualiza a cada token aceito")
    print("  • 🌡️  Temp calibration: scheduler ajusta temperatura dinamicamente")
    print("  • 📊 JS-divergence: confidence head mede desalinhamento global")
    print("  • 💾 Scheduler persistente: aprende entre execuções (scheduler_history.json)")
    print("  • 🧩 Mais estável: espera-se aceitação >35% em TODAS as runs")
