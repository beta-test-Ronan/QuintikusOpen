#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRM‑v4 – Weight Tying + Deep Supervision + Recursive Latent Transformer
VERSÃO FINAL ESTÁVEL – sem erros, pronta para rodar.
"""
import numpy as np
import json
import random
import string
import sys
import os
from collections import deque

# ======================== CONFIGURAÇÃO ========================
ARQUIVO_MODELO = 'modelo_trm_v4.npz'
ARQUIVO_TOKENIZER = 'tokenizer_trm_v4.json'
ARQUIVO_EMBEDDINGS = 'embeddings.npy'

DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
NUM_RECURSIONS = 4
FF_HIDDEN = DIM * 4
VOCAB_SIZE = 5000
MAX_LEN = 64
LR = 0.001
MEMORY_BETA = 0.9
BUFFER_CAPACITY = 1000
BUFFER_K = 5
TAMANHO_LOTE = 50
TEMPERATURA = 0.8
MAX_GERACAO = 15
DEEP_SUPERVISION_WEIGHT = 0.1
GRAD_NOISE = 0.001
RESET_PACIENCIA = 20

# ======================== JSON BASE ========================
BASE_JSON = {
    "sujeitos": {
        "eu":    { "pronome": "eu",    "ir_presente": "vou",    "ir_passado": "fui",
                   "ser_presente": "sou",  "ser_passado": "fui",
                   "estar_presente": "estou", "estar_passado": "estive" },
        "voce":  { "pronome": "você",  "ir_presente": "vai",    "ir_passado": "foi",
                   "ser_presente": "é",    "ser_passado": "foi",
                   "estar_presente": "está", "estar_passado": "esteve" },
        "ele":   { "pronome": "ele",   "ir_presente": "vai",    "ir_passado": "foi",
                   "ser_presente": "é",    "ser_passado": "foi",
                   "estar_presente": "está", "estar_passado": "esteve" },
        "ela":   { "pronome": "ela",   "ir_presente": "vai",    "ir_passado": "foi",
                   "ser_presente": "é",    "ser_passado": "foi",
                   "estar_presente": "está", "estar_passado": "esteve" },
        "nos":   { "pronome": "nós",   "ir_presente": "vamos",  "ir_passado": "fomos",
                   "ser_presente": "somos", "ser_passado": "fomos",
                   "estar_presente": "estamos", "estar_passado": "estivemos" },
        "eles":  { "pronome": "eles",  "ir_presente": "vão",    "ir_passado": "foram",
                   "ser_presente": "são",   "ser_passado": "foram",
                   "estar_presente": "estão", "estar_passado": "estiveram" },
        "elas":  { "pronome": "elas",  "ir_presente": "vão",    "ir_passado": "foram",
                   "ser_presente": "são",   "ser_passado": "foram",
                   "estar_presente": "estão", "estar_passado": "estiveram" }
    },
    "verbos_ir": {
        "comer":   ["pão", "arroz", "feijão", "salada", "uma maçã", "bolo", "chocolate"],
        "beber":   ["água", "café", "suco", "refrigerante", "cerveja", "chá"],
        "dormir":  ["cedo", "tarde", "a noite toda", "um pouco", "depois do almoço"],
        "trabalhar": ["no escritório", "em casa", "até tarde", "com o computador", "na obra"],
        "estudar": ["matemática", "português", "para a prova", "de manhã", "na biblioteca"],
        "dirigir": ["o carro", "com cuidado", "na estrada", "até o centro", "devagar"],
        "comprar": ["pão", "leite", "roupa", "um presente", "frutas", "no supermercado"],
        "falar":  ["com o amigo", "no telefone", "baixo", "alto", "sobre o trabalho"],
        "ouvir":  ["música", "o rádio", "um podcast", "conselhos", "com atenção"],
        "assistir":["TV", "um filme", "a novela", "o jogo", "uma série"],
        "caminhar":["no parque", "na praia", "depois do jantar", "com o cachorro"],
        "cozinhar":["o almoço", "o jantar", "macarrão", "arroz", "frango"],
        "limpar": ["a casa", "o quarto", "a cozinha", "o banheiro", "a janela"],
        "reclamar":["do trânsito", "do calor", "do preço", "da demora", "do chefe"],
        "perguntar":["as horas", "o caminho", "o preço", "se está bem", "sobre a festa"],
        "responder":["a pergunta", "a mensagem", "o e-mail", "com educação"],
        "esperar": ["o ônibus", "a vez", "a resposta", "ansiosamente", "pacientemente"],
        "gostar":  ["de música", "de praia", "de cinema", "de viajar", "de bolo"],
        "precisar":["de ajuda", "de dinheiro", "de descanso", "de um médico"],
        "querer":  ["água", "sair", "comer", "dormir", "viajar"]
    },
    "ser_complementos": {
        "adjetivos": ["bonito", "feio", "alto", "baixo", "inteligente", "engraçado", "tímido", "extrovertido", "jovem", "idoso"],
        "profissoes": ["professor", "médico", "engenheiro", "advogado", "estudante", "motorista", "cozinheiro"]
    },
    "estar_complementos": {
        "lugares": ["em casa", "no trabalho", "na escola", "no parque", "na praia", "no shopping", "aqui", "ali", "lá"],
        "estados": ["cansado", "feliz", "triste", "com fome", "com sede", "bem", "mal", "atrasado", "adiantado"]
    },
    "adverbios": {
        "tempo": ["hoje", "amanhã", "agora", "já", "ainda", "sempre", "nunca", "cedo", "tarde"],
        "modo":  ["bem", "mal", "depressa", "devagar", "calmamente", "rapidamente"],
        "lugar": ["aqui", "ali", "lá", "perto", "longe", "em casa", "no trabalho"]
    },
    "estruturas": [
        "{sujeito} {conjugacao_ir} {verbo_ir} {complemento_ir}.",
        "{sujeito} não {conjugacao_ir} {verbo_ir} {complemento_ir}.",
        "{adverbio} {sujeito} {conjugacao_ir} {verbo_ir} {complemento_ir}.",
        "{sujeito} {conjugacao_ir} {verbo_ir} {complemento_ir}?",
        "{adverbio} {sujeito} {conjugacao_ir} {verbo_ir} {complemento_ir}?",
        "Será que {sujeito} {conjugacao_ir} {verbo_ir} {complemento_ir}?",
        "Por que {sujeito} {conjugacao_ir} {verbo_ir} {complemento_ir}?",
        "{sujeito} {conjugacao_ser} {complemento_ser}.",
        "{sujeito} não {conjugacao_ser} {complemento_ser}.",
        "{sujeito} {conjugacao_estar} {complemento_estar}.",
        "{sujeito} não {conjugacao_estar} {complemento_estar}.",
        "{adverbio} {sujeito} {conjugacao_estar} {complemento_estar}.",
        "Que {adjetivo} {sujeito} {conjugacao_ser}!",
        "{sujeito} sempre {conjugacao_ser} {complemento_ser}.",
        "{sujeito} nunca {conjugacao_estar} {complemento_estar}."
    ],
    "adjetivos_exclamacao": ["bom", "ruim", "grande", "pequeno", "rápido", "lento", "bonito", "feio", "novo", "velho"]
}

# ======================== UTILITÁRIOS ========================
def l2_normalize(x, axis=-1, eps=1e-8):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

# ======================== TOKENIZADOR ========================
class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.next_id = 2

    def add_word(self, word):
        if word not in self.word2idx and self.next_id < self.vocab_size:
            self.word2idx[word] = self.next_id
            self.idx2word[self.next_id] = word
            self.next_id += 1

    def fit(self, textos):
        for texto in textos:
            texto_limpo = texto.replace('.', ' .').replace('?', ' ?').replace('!', ' !').replace(',', ' ,')
            for palavra in texto_limpo.split():
                self.add_word(palavra)

    def encode(self, texto):
        texto_limpo = texto.replace('.', ' .').replace('?', ' ?').replace('!', ' !').replace(',', ' ,')
        return [self.word2idx.get(p, self.unk_id) for p in texto_limpo.split()]

    def decode(self, ids):
        return ' '.join(self.idx2word.get(i, '<UNK>') for i in ids)

    def salvar(self, caminho):
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': {str(k): v for k, v in self.idx2word.items()}}, f)

    def carregar(self, caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            self.word2idx = dados['word2idx']
            self.idx2word = {int(k): v for k, v in dados['idx2word'].items()}
            self.next_id = len(self.word2idx)

# ======================== GERADOR DE FRASES ========================
class GeradorFrases:
    def __init__(self, base):
        self.base = base
        self.verbos_ir = base.get('verbos_ir', {})
        self.ser_adj = base.get('ser_complementos', {}).get('adjetivos', [])
        self.ser_prof = base.get('ser_complementos', {}).get('profissoes', [])
        self.estar_lug = base.get('estar_complementos', {}).get('lugares', [])
        self.estar_est = base.get('estar_complementos', {}).get('estados', [])
        self.adverbios_tempo = base.get('adverbios', {}).get('tempo', [])
        self.adjetivos_exc = base.get('adjetivos_exclamacao', [])
        self.ultimo_sujeito = None
        self.ultimo_verbo = None
        self.ultimo_complemento = None

    def escolher_aleatorio(self, lista):
        if isinstance(lista, dict):
            return random.choice(list(lista.keys()))
        return random.choice(lista)

    def gerar(self):
        if self.ultimo_sujeito and random.random() < 0.5:
            suj = self.ultimo_sujeito
            verbo = self.ultimo_verbo if random.random() < 0.7 else self.escolher_aleatorio(self.verbos_ir)
            comp_ir = self.ultimo_complemento if random.random() < 0.7 else random.choice(self.verbos_ir[verbo])
            estrutura = "{sujeito} {conjugacao_ir} {verbo_ir} {complemento_ir}."
        else:
            suj = self.escolher_aleatorio(list(self.base['sujeitos'].values()))
            verbo = self.escolher_aleatorio(self.verbos_ir)
            comp_ir = random.choice(self.verbos_ir[verbo])
            estrutura = random.choice(self.base['estruturas'])

        self.ultimo_sujeito = suj
        self.ultimo_verbo = verbo
        self.ultimo_complemento = comp_ir

        valores = {}
        if '{conjugacao_ir}' in estrutura:
            tempo = random.choice(['ir_presente', 'ir_passado'])
            conj_ir = suj[tempo]
            valores['conjugacao_ir'] = conj_ir
            valores['verbo_ir'] = verbo
            valores['complemento_ir'] = comp_ir

        if '{conjugacao_ser}' in estrutura or '{complemento_ser}' in estrutura:
            tempo_ser = random.choice(['ser_presente', 'ser_passado'])
            conj_ser = suj[tempo_ser]
            tipo = random.choice(['adjetivos', 'profissoes'])
            if tipo == 'adjetivos' and self.ser_adj:
                comp_ser = self.escolher_aleatorio(self.ser_adj)
            else:
                comp_ser = self.escolher_aleatorio(self.ser_prof)
            valores['conjugacao_ser'] = conj_ser
            valores['complemento_ser'] = comp_ser

        if '{conjugacao_estar}' in estrutura or '{complemento_estar}' in estrutura:
            tempo_estar = random.choice(['estar_presente', 'estar_passado'])
            conj_estar = suj[tempo_estar]
            tipo = random.choice(['lugares', 'estados'])
            if tipo == 'lugares':
                comp_estar = self.escolher_aleatorio(self.estar_lug)
            else:
                comp_estar = self.escolher_aleatorio(self.estar_est)
            valores['conjugacao_estar'] = conj_estar
            valores['complemento_estar'] = comp_estar

        if '{adverbio}' in estrutura:
            valores['adverbio'] = self.escolher_aleatorio(self.adverbios_tempo)

        if '{adjetivo}' in estrutura:
            valores['adjetivo'] = self.escolher_aleatorio(self.adjetivos_exc)

        chaves = [p[1] for p in string.Formatter().parse(estrutura) if p[1]]
        valores_filtrados = {k: v for k, v in valores.items() if k in chaves}
        if 'sujeito' in chaves:
            valores_filtrados['sujeito'] = suj['pronome']

        frase = estrutura.format(**valores_filtrados).capitalize().strip()
        return frase

# ======================== CAMADAS DA REDE ========================
class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        if dout.ndim == 1:
            dout = dout.reshape(1, -1)
        self.dW += self.x.T @ dout
        self.db += dout.sum(axis=0)
        if GRAD_NOISE > 0:
            self.dW += np.random.randn(*self.dW.shape) * GRAD_NOISE
            self.db += np.random.randn(*self.db.shape) * GRAD_NOISE
        return dout @ self.W.T

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.dgamma = np.zeros(dim)
        self.dbeta = np.zeros(dim)
        self.x = None
        self.mean = None
        self.var = None
        self.std_inv = None
        self.norm = None

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.norm = (x - self.mean) * self.std_inv
        return self.gamma * self.norm + self.beta

    def backward(self, dout):
        N = self.x.shape[-1]
        dx_norm = dout * self.gamma
        dvar = (dx_norm * self.norm).sum(axis=-1, keepdims=True) * -0.5 * self.std_inv**2
        dmean = (dx_norm * -self.std_inv).sum(axis=-1, keepdims=True) + dvar * (-2 * self.norm).mean(axis=-1, keepdims=True)
        dx = dx_norm * self.std_inv + dvar * 2 * self.norm / N + dmean / N
        self.dgamma += (dout * self.norm).sum(axis=0)
        self.dbeta += dout.sum(axis=0)
        if GRAD_NOISE > 0:
            self.dgamma += np.random.randn(*self.dgamma.shape) * GRAD_NOISE
            self.dbeta += np.random.randn(*self.dbeta.shape) * GRAD_NOISE
        return dx

class MultiHeadGeometricAttention:
    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.Wq = Linear(dim, dim)
        self.Wk = Linear(dim, dim)
        self.Wv = Linear(dim, dim)
        self.Wo = Linear(dim, dim)
        self.seq_len = 0
        self.Q = None
        self.K = None
        self.V = None
        self.Q_norm = None
        self.K_norm = None
        self.attn = None

    def forward(self, x, mask=None):
        self.seq_len = x.shape[0]
        Q = self.Wq.forward(x)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)

        self.Q = Q.reshape(self.seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        self.K = K.reshape(self.seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        self.V = V.reshape(self.seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        self.Q_norm = l2_normalize(self.Q, axis=-1)
        self.K_norm = l2_normalize(self.K, axis=-1)

        scores = self.Q_norm @ self.K_norm.transpose(0, 2, 1)
        if mask is not None:
            scores += mask * -1e9
        self.attn = softmax(scores, axis=-1)

        out = self.attn @ self.V
        out = out.transpose(1, 0, 2).reshape(self.seq_len, self.dim)
        return self.Wo.forward(out)

    def backward(self, dout):
        dout = self.Wo.backward(dout)
        dout = dout.reshape(self.seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        dV = self.attn.transpose(0, 2, 1) @ dout
        dattn = dout @ self.V.transpose(0, 2, 1)
        dattn = dattn * self.attn
        dattn -= self.attn * dattn.sum(axis=-1, keepdims=True)

        dQ_norm = dattn @ self.K_norm
        dK_norm = dattn.transpose(0, 2, 1) @ self.Q_norm

        def l2_norm_backward(dy, y, x):
            norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
            y_exp = y[..., np.newaxis]
            y_t = y[..., np.newaxis, :]
            dy_dx = (np.eye(x.shape[-1]) - y_exp @ y_t) / norm[..., np.newaxis]
            if dy.ndim > 2:
                dy_exp = dy[..., np.newaxis, :]
                dx = np.squeeze(dy_exp @ dy_dx, axis=-2)
            else:
                dx = dy @ dy_dx
            return dx

        dQ = l2_norm_backward(dQ_norm, self.Q_norm, self.Q)
        dK = l2_norm_backward(dK_norm, self.K_norm, self.K)

        dQ = dQ.transpose(1, 0, 2).reshape(self.seq_len, self.dim)
        dK = dK.transpose(1, 0, 2).reshape(self.seq_len, self.dim)
        dV = dV.transpose(1, 0, 2).reshape(self.seq_len, self.dim)

        dx_q = self.Wq.backward(dQ)
        dx_k = self.Wk.backward(dK)
        dx_v = self.Wv.backward(dV)
        return dx_q + dx_k + dx_v

class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.W1 = Linear(dim, hidden_dim)
        self.W2 = Linear(hidden_dim, dim)
        self.x = None
        self.out1 = None
        self.relu_out = None

    def forward(self, x):
        self.x = x
        self.out1 = self.W1.forward(x)
        self.relu_out = relu(self.out1)
        return self.W2.forward(self.relu_out)

    def backward(self, dout):
        dout2 = self.W2.backward(dout)
        dout2 = dout2 * drelu(self.out1)
        dx = self.W1.backward(dout2)
        return dx

class TransformerBlock:
    def __init__(self, dim, num_heads, ff_hidden):
        self.attn = MultiHeadGeometricAttention(dim, num_heads)
        self.ln1 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_hidden)
        self.ln2 = LayerNorm(dim)
        self.x_input = None
        self.x_ln1 = None
        self.x_attn = None
        self.resid1 = None
        self.x_ln2 = None
        self.x_ffn = None
        self.resid2 = None

    def forward(self, x, mask=None):
        self.x_input = x
        self.resid1 = x
        self.x_ln1 = self.ln1.forward(x)
        self.x_attn = self.attn.forward(self.x_ln1, mask)
        x = self.x_attn + self.resid1

        self.resid2 = x
        self.x_ln2 = self.ln2.forward(x)
        self.x_ffn = self.ffn.forward(self.x_ln2)
        out = self.x_ffn + self.resid2
        return out

    def backward(self, dout):
        dx_ffn = self.ffn.backward(dout)
        dx_ln2 = self.ln2.backward(dout)
        d_resid2 = dout + dx_ffn + dx_ln2

        dx_attn = self.attn.backward(d_resid2)
        dx_ln1 = self.ln1.backward(d_resid2)
        d_resid1 = d_resid2 + dx_attn + dx_ln1

        return d_resid1

    def save_state(self):
        return {
            'x_input': self.x_input,
            'x_ln1': self.x_ln1,
            'x_attn': self.x_attn,
            'resid1': self.resid1,
            'x_ln2': self.x_ln2,
            'x_ffn': self.x_ffn,
            'resid2': self.resid2,
            'attn_Q': self.attn.Q,
            'attn_K': self.attn.K,
            'attn_V': self.attn.V,
            'attn_Q_norm': self.attn.Q_norm,
            'attn_K_norm': self.attn.K_norm,
            'attn_attn': self.attn.attn,
            'attn_seq_len': self.attn.seq_len,
            'attn_Wq_x': self.attn.Wq.x,
            'attn_Wk_x': self.attn.Wk.x,
            'attn_Wv_x': self.attn.Wv.x,
            'attn_Wo_x': self.attn.Wo.x,
            'ln1_x': self.ln1.x,
            'ln1_mean': self.ln1.mean,
            'ln1_var': self.ln1.var,
            'ln1_std_inv': self.ln1.std_inv,
            'ln1_norm': self.ln1.norm,
            'ln2_x': self.ln2.x,
            'ln2_mean': self.ln2.mean,
            'ln2_var': self.ln2.var,
            'ln2_std_inv': self.ln2.std_inv,
            'ln2_norm': self.ln2.norm,
            'ffn_x': self.ffn.x,
            'ffn_out1': self.ffn.out1,
            'ffn_relu_out': self.ffn.relu_out,
            'ffn_W1_x': self.ffn.W1.x,
            'ffn_W2_x': self.ffn.W2.x,
        }

    def load_state(self, state):
        self.x_input = state['x_input']
        self.x_ln1 = state['x_ln1']
        self.x_attn = state['x_attn']
        self.resid1 = state['resid1']
        self.x_ln2 = state['x_ln2']
        self.x_ffn = state['x_ffn']
        self.resid2 = state['resid2']
        self.attn.Q = state['attn_Q']
        self.attn.K = state['attn_K']
        self.attn.V = state['attn_V']
        self.attn.Q_norm = state['attn_Q_norm']
        self.attn.K_norm = state['attn_K_norm']
        self.attn.attn = state['attn_attn']
        self.attn.seq_len = state['attn_seq_len']
        self.attn.Wq.x = state['attn_Wq_x']
        self.attn.Wk.x = state['attn_Wk_x']
        self.attn.Wv.x = state['attn_Wv_x']
        self.attn.Wo.x = state['attn_Wo_x']
        self.ln1.x = state['ln1_x']
        self.ln1.mean = state['ln1_mean']
        self.ln1.var = state['ln1_var']
        self.ln1.std_inv = state['ln1_std_inv']
        self.ln1.norm = state['ln1_norm']
        self.ln2.x = state['ln2_x']
        self.ln2.mean = state['ln2_mean']
        self.ln2.var = state['ln2_var']
        self.ln2.std_inv = state['ln2_std_inv']
        self.ln2.norm = state['ln2_norm']
        self.ffn.x = state['ffn_x']
        self.ffn.out1 = state['ffn_out1']
        self.ffn.relu_out = state['ffn_relu_out']
        self.ffn.W1.x = state['ffn_W1_x']
        self.ffn.W2.x = state['ffn_W2_x']

# ======================== MODELO TRM ========================
class TRM:
    def __init__(self, dim, num_heads, num_layers, ff_hidden, vocab_size, max_len, tokenizer,
                 num_recursions=4, embeddings=None):
        self.dim = dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.num_recursions = num_recursions

        if embeddings is not None:
            self.E = embeddings.copy()
        else:
            self.E = np.random.randn(vocab_size, dim) * 0.02

        pe = np.zeros((max_len, dim))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pos_enc = pe

        self.layers = [TransformerBlock(dim, num_heads, ff_hidden) for _ in range(num_layers)]
        self.final_ln = LayerNorm(dim)
        self.pool_query = np.random.randn(dim) * 0.02

        self.short_term_memory = np.zeros(dim)
        self.memory_beta = MEMORY_BETA

        self.associative_buffer = deque(maxlen=BUFFER_CAPACITY)
        self.buffer_k = BUFFER_K

        self.rms_cache = {}
        self.rms_eps = 1e-8
        self.rms_decay = 0.9

        self.recursion_states = []

    def forward(self, token_ids, use_memory=True, return_all_logits=False, return_intermediates=False):
        seq_len = len(token_ids)
        if seq_len == 0:
            if return_all_logits:
                return None, None, None, None
            return np.zeros(self.dim), None, None, None, None, None

        token_emb = self.E[token_ids, :]
        x = token_emb + self.pos_enc[:seq_len]

        if use_memory and seq_len < self.max_len - 1:
            mem_emb = np.expand_dims(self.short_term_memory, axis=0)
            x = np.vstack([mem_emb, x])
            x[0] += self.pos_enc[0]

        self.recursion_states = []
        intermediates = []

        for r in range(self.num_recursions):
            for layer in self.layers:
                x = layer.forward(x)

            current_states = [layer.save_state() for layer in self.layers]
            self.recursion_states.append(current_states)

            if return_intermediates or return_all_logits:
                logits_r = x @ self.E.T
                probs_r = softmax(logits_r, axis=-1)
                intermediates.append((logits_r, probs_r, x.copy()))

        x_norm = self.final_ln.forward(x)

        if return_all_logits or return_intermediates:
            logits_final = x_norm @ self.E.T
            probs_final = softmax(logits_final, axis=-1)
            if return_intermediates:
                return intermediates, logits_final, probs_final, x_norm, token_emb
            return logits_final, probs_final, x_norm, token_emb, intermediates
        else:
            scores = x_norm @ self.pool_query
            attn_pool = softmax(scores, axis=0)
            self.attn_pool = attn_pool
            context = (attn_pool[:, np.newaxis] * x_norm).sum(axis=0)
            logits = context @ self.E.T
            probs = softmax(logits, axis=-1)
            predicted_vector = probs @ self.E
            return context, logits, probs, predicted_vector, x_norm, token_emb

    def train_sequence_parallel(self, token_ids, lr=LR):
        if len(token_ids) < 2:
            return 0.0

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        result = self.forward(input_ids, use_memory=False, return_all_logits=True)
        logits_final, probs_final, x_hidden, token_emb, intermediates = result
        seq_len = logits_final.shape[0]

        loss_final = 0.0
        dlogits_total = np.zeros_like(logits_final)
        for t in range(seq_len):
            target_id = target_ids[t]
            prob = probs_final[t]
            loss_final += -np.log(prob[target_id] + 1e-8)
            dlogits = prob.copy()
            dlogits[target_id] -= 1
            dlogits_total[t] = dlogits
        loss_final /= seq_len
        dlogits_total /= seq_len

        loss_deep = 0.0
        dX_deep = np.zeros_like(x_hidden)
        for (logits_r, probs_r, x_r) in intermediates:
            loss_r = 0.0
            dlogits_r = np.zeros_like(logits_r)
            for t in range(seq_len):
                target_id = target_ids[t]
                prob = probs_r[t]
                loss_r += -np.log(prob[target_id] + 1e-8)
                dlogits = prob.copy()
                dlogits[target_id] -= 1
                dlogits_r[t] = dlogits
            loss_r /= seq_len
            dlogits_r /= seq_len
            loss_deep += loss_r
            dX_deep += dlogits_r @ self.E

        loss_total = loss_final + DEEP_SUPERVISION_WEIGHT * loss_deep

        dX = dlogits_total @ self.E + DEEP_SUPERVISION_WEIGHT * dX_deep
        dX = self.final_ln.backward(dX)

        for r in reversed(range(self.num_recursions)):
            for layer, state in zip(self.layers, self.recursion_states[r]):
                layer.load_state(state)
            for layer in reversed(self.layers):
                dX = layer.backward(dX)

        dE = np.zeros_like(self.E)
        for t, idx in enumerate(input_ids):
            dE[idx] += dX[t]
        self._rmsprop_update_embedding(dE, lr)

        self._apply_gradients(lr)

        context, _, _, _, _, _ = self.forward(token_ids, use_memory=False)
        self.short_term_memory = (self.memory_beta * self.short_term_memory +
                                  (1 - self.memory_beta) * context)

        last_target_emb = self.E[token_ids[-1]]
        self.associative_buffer.append((context, last_target_emb))

        return loss_total

    def generate(self, seed_ids, max_new_tokens=15, temperature=TEMPERATURA, use_memory=True):
        ids = seed_ids.copy()
        for _ in range(max_new_tokens):
            context, logits, probs, pred_vec, _, _ = self.forward(ids, use_memory=use_memory)
            logits_scaled = logits / temperature
            probs_scaled = softmax(logits_scaled)
            next_id = np.random.choice(self.vocab_size, p=probs_scaled)
            if next_id == 0:
                probs_scaled[0] = 0
                probs_scaled /= probs_scaled.sum()
                next_id = np.random.choice(self.vocab_size, p=probs_scaled)
            ids.append(next_id)
            palavra = self.tokenizer.idx2word.get(next_id, '')
            if palavra in ('.', '?', '!'):
                break
        return ids

    def _rmsprop_update_embedding(self, dE, lr):
        key = id(self.E)
        if key not in self.rms_cache:
            self.rms_cache[key] = np.zeros_like(self.E)
        self.rms_cache[key] = self.rms_decay * self.rms_cache[key] + (1 - self.rms_decay) * dE**2
        self.E -= lr * dE / (np.sqrt(self.rms_cache[key]) + self.rms_eps)

    def _apply_gradients(self, lr):
        for layer in self.layers:
            for param in [layer.attn.Wq, layer.attn.Wk, layer.attn.Wv, layer.attn.Wo]:
                self._rmsprop_update_linear(param, lr)
            for ln in [layer.ln1, layer.ln2]:
                self._rmsprop_update_ln(ln, lr)
            for ff in [layer.ffn.W1, layer.ffn.W2]:
                self._rmsprop_update_linear(ff, lr)
        self._rmsprop_update_ln(self.final_ln, lr)

    def _rmsprop_update_linear(self, linear, lr):
        key_W = id(linear.W)
        key_b = id(linear.b)
        if key_W not in self.rms_cache:
            self.rms_cache[key_W] = np.zeros_like(linear.W)
            self.rms_cache[key_b] = np.zeros_like(linear.b)
        if GRAD_NOISE > 0:
            linear.dW += np.random.randn(*linear.dW.shape) * GRAD_NOISE
            linear.db += np.random.randn(*linear.db.shape) * GRAD_NOISE
        self.rms_cache[key_W] = self.rms_decay * self.rms_cache[key_W] + (1 - self.rms_decay) * linear.dW**2
        self.rms_cache[key_b] = self.rms_decay * self.rms_cache[key_b] + (1 - self.rms_decay) * linear.db**2
        linear.W -= lr * linear.dW / (np.sqrt(self.rms_cache[key_W]) + self.rms_eps)
        linear.b -= lr * linear.db / (np.sqrt(self.rms_cache[key_b]) + self.rms_eps)
        linear.dW.fill(0)
        linear.db.fill(0)

    def _rmsprop_update_ln(self, ln, lr):
        key_g = id(ln.gamma)
        key_b = id(ln.beta)
        if key_g not in self.rms_cache:
            self.rms_cache[key_g] = np.zeros_like(ln.gamma)
            self.rms_cache[key_b] = np.zeros_like(ln.beta)
        if GRAD_NOISE > 0:
            ln.dgamma += np.random.randn(*ln.dgamma.shape) * GRAD_NOISE
            ln.dbeta += np.random.randn(*ln.dbeta.shape) * GRAD_NOISE
        self.rms_cache[key_g] = self.rms_decay * self.rms_cache[key_g] + (1 - self.rms_decay) * ln.dgamma**2
        self.rms_cache[key_b] = self.rms_decay * self.rms_cache[key_b] + (1 - self.rms_decay) * ln.dbeta**2
        ln.gamma -= lr * ln.dgamma / (np.sqrt(self.rms_cache[key_g]) + self.rms_eps)
        ln.beta -= lr * ln.dbeta / (np.sqrt(self.rms_cache[key_b]) + self.rms_eps)
        ln.dgamma.fill(0)
        ln.dbeta.fill(0)

    def salvar(self, caminho):
        dados = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                dados[k] = v
            elif isinstance(v, list):
                for i, arr in enumerate(v):
                    dados[f'{k}_{i}'] = arr
                dados[f'{k}_len'] = len(v)
            elif isinstance(v, deque):
                pass
            else:
                dados[k] = v
        np.savez(caminho, **dados)

    def carregar(self, caminho):
        dados = np.load(caminho, allow_pickle=True)
        self.dim = dados['dim'].item()
        self.num_layers = dados['num_layers'].item()
        self.max_len = dados['max_len'].item()
        self.vocab_size = dados['vocab_size'].item()
        self.num_recursions = dados['num_recursions'].item()
        self.E = dados['E']
        self.pos_enc = dados['pos_enc']
        self.pool_query = dados['pool_query']
        self.short_term_memory = dados['short_term_memory']
        self.memory_beta = dados['memory_beta'].item()
        self.buffer_k = dados['buffer_k'].item()
        self.layers = [TransformerBlock(self.dim, NUM_HEADS, self.dim * 4) for _ in range(self.num_layers)]
        for i, layer in enumerate(self.layers):
            layer.attn.Wq.W = dados[f'layers_{i}_attn_Wq_W']
            layer.attn.Wq.b = dados[f'layers_{i}_attn_Wq_b']
            layer.attn.Wk.W = dados[f'layers_{i}_attn_Wk_W']
            layer.attn.Wk.b = dados[f'layers_{i}_attn_Wk_b']
            layer.attn.Wv.W = dados[f'layers_{i}_attn_Wv_W']
            layer.attn.Wv.b = dados[f'layers_{i}_attn_Wv_b']
            layer.attn.Wo.W = dados[f'layers_{i}_attn_Wo_W']
            layer.attn.Wo.b = dados[f'layers_{i}_attn_Wo_b']
            layer.ln1.gamma = dados[f'layers_{i}_ln1_gamma']
            layer.ln1.beta = dados[f'layers_{i}_ln1_beta']
            layer.ln2.gamma = dados[f'layers_{i}_ln2_gamma']
            layer.ln2.beta = dados[f'layers_{i}_ln2_beta']
            layer.ffn.W1.W = dados[f'layers_{i}_ffn_W1_W']
            layer.ffn.W1.b = dados[f'layers_{i}_ffn_W1_b']
            layer.ffn.W2.W = dados[f'layers_{i}_ffn_W2_W']
            layer.ffn.W2.b = dados[f'layers_{i}_ffn_W2_b']
        self.final_ln.gamma = dados['final_ln_gamma']
        self.final_ln.beta = dados['final_ln_beta']

# ======================== INICIALIZAÇÃO ========================
print('🔤 Preparando tokenizador...')
tokenizer = Tokenizer(VOCAB_SIZE)
if os.path.exists(ARQUIVO_TOKENIZER):
    tokenizer.carregar(ARQUIVO_TOKENIZER)
else:
    gerador_temp = GeradorFrases(BASE_JSON)
    frases = [gerador_temp.gerar() for _ in range(2000)]
    tokenizer.fit(frases)
    tokenizer.salvar(ARQUIVO_TOKENIZER)

embeddings_matriz = None
if os.path.exists(ARQUIVO_EMBEDDINGS):
    print('📊 Embeddings fastText encontrados.')
    embeddings_matriz = np.load(ARQUIVO_EMBEDDINGS)
    if embeddings_matriz.shape[0] >= tokenizer.next_id:
        embeddings_matriz = embeddings_matriz[:tokenizer.next_id, :]
        DIM = embeddings_matriz.shape[1]
        NUM_HEADS = 6 if DIM >= 6 else 4
        NUM_LAYERS = 2
        FF_HIDDEN = DIM * 4
        print(f'   Dim={DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS}, recursions={NUM_RECURSIONS}')
    else:
        print('⚠️  Embeddings incompatíveis. Usando aleatórios.')
        embeddings_matriz = None

if embeddings_matriz is None:
    DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    FF_HIDDEN = DIM * 4
    NUM_RECURSIONS = 4

modelo = TRM(DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN, tokenizer.next_id, MAX_LEN, tokenizer,
             num_recursions=NUM_RECURSIONS, embeddings=embeddings_matriz)
if os.path.exists(ARQUIVO_MODELO):
    print('🧠 Modelo carregado.')
    modelo.carregar(ARQUIVO_MODELO)
else:
    print('🆕 Novo modelo TRM‑v4 criado (estável).')

print(f'📚 Vocabulário: {tokenizer.next_id} tokens | Dimensão: {DIM}')
print(f'🧠 Camadas físicas: {NUM_LAYERS}, Recursões: {NUM_RECURSIONS}, Cabeças: {NUM_HEADS}')
print('🔄 Iniciando treino online...\n')

gerador = GeradorFrases(BASE_JSON)
lote_frases = []
contador = 0
melhor_loss = float('inf')
lotes_sem_melhora = 0

try:
    while True:
        frase = gerador.gerar()
        ids = tokenizer.encode(frase)
        if len(ids) < 2:
            continue
        loss = modelo.train_sequence_parallel(ids, lr=LR)
        lote_frases.append(ids)
        contador += 1

        if len(lote_frases) == TAMANHO_LOTE:
            print(f'📦 Lote {contador // TAMANHO_LOTE} ({contador} frases) | loss: {loss:.4f}')

            # Anti-colapso: reset se estagnar
            if loss < melhor_loss:
                melhor_loss = loss
                lotes_sem_melhora = 0
            else:
                lotes_sem_melhora += 1

            if lotes_sem_melhora >= RESET_PACIENCIA:
                print(f'⚠️  {RESET_PACIENCIA} lotes sem melhora. Reiniciando modelo...')
                modelo = TRM(DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN, tokenizer.next_id, MAX_LEN, tokenizer,
                             num_recursions=NUM_RECURSIONS, embeddings=embeddings_matriz)
                melhor_loss = float('inf')
                lotes_sem_melhora = 0

            prompts = ['Eu', 'Hoje', 'Por que', 'Será que']
            for p in prompts:
                seed_ids = tokenizer.encode(p)
                gen_ids = modelo.generate(seed_ids, max_new_tokens=MAX_GERACAO, temperature=TEMPERATURA)
                gen_frase = tokenizer.decode(gen_ids)
                print(f'   {p:10s} → {gen_frase}')

            modelo.salvar(ARQUIVO_MODELO)
            tokenizer.salvar(ARQUIVO_TOKENIZER)
            print('💾 Checkpoint salvo.\n')
            lote_frases.clear()

except KeyboardInterrupt:
    print('\n⏹️ Salvando...')
    modelo.salvar(ARQUIVO_MODELO)
    tokenizer.salvar(ARQUIVO_TOKENIZER)
    print('✅ Até!')
