"""
TGp4 v2.0 — BPE + Acentos + KV Cache + Amostragem por Persona.

Mudanças sobre a versão original:
1) Tokenizador BPE (HuggingFace tokenizers) com preservação de acentos.
   Fallback para regex se não instalado.
2) KV Cache na atenção — geração 10-100x mais rápida.
3) Amostragem estocástica (multinomial) com configs por persona:
   - [logico]: temp baixa, top-k baixo (factual)
   - [roteiro]: temp alta, top-k alto (criativo)

Instalação:
    pip install tokenizers
"""
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Dispositivo Ativo: {device}")

# =====================================================================
# 1. TOKENIZADOR BPE COM ACENTOS
# =====================================================================
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    _HAS_BPE = True
except ImportError:
    _HAS_BPE = False
    print("⚠️  tokenizers não instalado. Usando regex fallback. (pip install tokenizers)")


class TokenizadorTopologico:
    SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>",
               "<usr>", "<bot>", "[logico]", "[roteiro]"]

    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.vocabulario = {}
        self.idx_para_word = {}
        self._bpe = None

    def _normalizar(self, texto: str) -> str:
        """Lowercase + preserva acentos + normaliza espaços."""
        t = texto.lower()
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def construir_vocab(self, corpus_total):
        textos = [self._normalizar(t) for t in corpus_total]

        if _HAS_BPE:
            tok = Tokenizer(BPE(unk_token="<unk>"))
            tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            tok.decoder = ByteLevelDecoder()
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=self.SPECIAL,
                show_progress=False,
                min_frequency=1,
            )
            tok.train_from_iterator(textos, trainer=trainer)
            self._bpe = tok
            vocab = tok.get_vocab()
            self.vocabulario = dict(vocab)
            self.idx_para_word = {i: t for t, i in vocab.items()}
            print(f"🔤 BPE treinado | vocab={len(vocab)}")
        else:
            tokens = []
            for t in textos:
                tokens.extend(re.findall(r'<[^>]+>|\[[^\]]+\]|\w+|[^\w\s]', t, flags=re.UNICODE))
            vocab_set = sorted(set(tokens))
            for tag in self.SPECIAL:
                if tag not in vocab_set:
                    vocab_set.append(tag)
            self.vocabulario = {t: i for i, t in enumerate(vocab_set)}
            self.idx_para_word = {i: t for t, i in self.vocabulario.items()}

    def tokenizar(self, texto: str):
        t = self._normalizar(texto)
        if self._bpe is not None:
            return self._bpe.encode(t).ids
        tokens = re.findall(r'<[^>]+>|\[[^\]]+\]|\w+|[^\w\s]', t, flags=re.UNICODE)
        return [self.vocabulario.get(w, self.vocabulario['<unk>']) for w in tokens]

    def processar_linhas(self, linhas, max_len=64):
        dados = []
        for texto in linhas:
            ids = self.tokenizar(texto)
            if len(ids) < max_len:
                ids += [self.vocabulario['<pad>']] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            dados.append(ids)
        return dados

    def decode(self, ids):
        ids = [int(i) for i in ids if 0 <= int(i) < len(self.idx_para_word)]
        if self._bpe is not None:
            filtrado = [i for i in ids if self.idx_para_word.get(i, '') not in self.SPECIAL]
            texto = self._bpe.decode(filtrado)
            return texto.strip()
        words = [self.idx_para_word.get(i, '') for i in ids]
        words = [w for w in words if w and w not in self.SPECIAL]
        return re.sub(r'\s+([.,!?])', r'\1', ' '.join(words)).strip()

    @property
    def pad_id(self):
        return self.vocabulario['<pad>']


# =====================================================================
# 2. ROTEADOR GDI (inalterado — original)
# =====================================================================
class DualPersonaGDIRouter(nn.Module):
    def __init__(self, dim_emb, num_experts=4, eps=1e-9):
        super().__init__()
        self.num_experts = num_experts
        self.eps = eps
        self.expert_nodes = nn.Parameter(torch.randn(num_experts, dim_emb) * 0.02)
        self.proj_topo = nn.Linear(dim_emb, dim_emb)

    def forward(self, x, modo='logico'):
        h = self.proj_topo(x)
        p_x = F.softmax(h, dim=-1)
        p_e = F.softmax(self.expert_nodes, dim=-1)
        p_x_exp = p_x.unsqueeze(-2)
        p_e_exp = p_e.unsqueeze(0).unsqueeze(0)
        kl_div = torch.sum(
            p_x_exp * (torch.log(p_x_exp + self.eps) - torch.log(p_e_exp + self.eps)),
            dim=-1
        )
        router_weights = F.softmax(-kl_div, dim=-1)

        mask = torch.zeros_like(router_weights)
        if modo == 'logico':
            mask[..., :2] = 1.0
        elif modo == 'roteiro':
            mask[..., 2:] = 1.0
        else:
            mask[..., :] = 1.0

        router_weights = router_weights * mask
        router_weights = router_weights / (torch.sum(router_weights, dim=-1, keepdim=True) + self.eps)

        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            P = torch.mean(router_weights, dim=[0, 1])
            top1 = torch.argmax(router_weights, dim=-1)
            tokens_per_expert = F.one_hot(top1, num_classes=self.num_experts).float()
            f = torch.mean(tokens_per_expert, dim=[0, 1])
            aux_loss = self.num_experts * torch.sum(P * f)

        return router_weights, aux_loss


# =====================================================================
# 3. ATENÇÃO CAUSAL COM KV CACHE
# =====================================================================
class CausalAttentionWithCache(nn.Module):
    def __init__(self, dim_emb, num_heads, dropout=0.1):
        super().__init__()
        assert dim_emb % num_heads == 0
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        self.head_dim = dim_emb // num_heads
        self.scale = self.head_dim ** -0.5

        self.Wq = nn.Linear(dim_emb, dim_emb, bias=False)
        self.Wk = nn.Linear(dim_emb, dim_emb, bias=False)
        self.Wv = nn.Linear(dim_emb, dim_emb, bias=False)
        self.Wo = nn.Linear(dim_emb, dim_emb, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask_attn=None, kv_cache=None, use_cache=False):
        B, T, D = x.shape

        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            K_past, V_past = kv_cache
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)

        T_total = K.size(2)
        T_past = T_total - T
        new_cache = (K, V) if use_cache else None

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Máscara causal considerando cache
        if T_past > 0:
            i_idx = torch.arange(T, device=x.device).unsqueeze(1)
            k_idx = torch.arange(T_total, device=x.device).unsqueeze(0)
            causal = k_idx <= (i_idx + T_past)
        else:
            causal = torch.tril(torch.ones(T, T_total, device=x.device, dtype=torch.bool))

        causal = causal.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~causal, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.Wo(out)

        if use_cache:
            return out, new_cache
        return out


# =====================================================================
# 4. CAMADA MoE COM CACHE
# =====================================================================
class DeepTransformerMoELayer(nn.Module):
    def __init__(self, dim_emb, dim_ff, num_experts, num_heads=4):
        super().__init__()
        self.dim_emb = dim_emb
        self.attn = CausalAttentionWithCache(dim_emb, num_heads)
        self.router = DualPersonaGDIRouter(dim_emb, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_emb, dim_ff), nn.GELU(), nn.Linear(dim_ff, dim_emb)
            ) for _ in range(num_experts)
        ])
        self.norm1 = nn.LayerNorm(dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)

    def forward(self, x, mask_attn=None, modo='logico', kv_cache=None, use_cache=False):
        if use_cache:
            attn_out, new_cache = self.attn(x, mask_attn=mask_attn,
                                            kv_cache=kv_cache, use_cache=True)
        else:
            attn_out = self.attn(x, mask_attn=mask_attn)
            new_cache = None

        x = self.norm1(x + attn_out)

        r_probs, aux_loss = self.router(x, modo=modo)
        exp_outs = torch.stack([exp(x) for exp in self.experts], dim=-1)
        moe_out = torch.sum(r_probs.unsqueeze(-2) * exp_outs, dim=-1)
        x = self.norm2(x + moe_out)

        if use_cache:
            return x, aux_loss, new_cache
        return x, aux_loss


# =====================================================================
# 5. MODELO COMPLETO
# =====================================================================
class MOLEDualRuntimeDeep(nn.Module):
    def __init__(self, vocab_size, dim_emb=256, dim_ff=512, num_experts=4,
                 num_layers=3, max_seq_len=256, num_heads=4):
        super().__init__()
        self.dim_emb = dim_emb
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, dim_emb, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, dim_emb)
        self.layers = nn.ModuleList([
            DeepTransformerMoELayer(dim_emb, dim_ff, num_experts, num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(dim_emb, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, x, modo='logico', kv_caches=None, use_cache=False):
        B, T = x.shape

        if kv_caches is not None and kv_caches[0] is not None:
            T_past = kv_caches[0][0].size(2)
        else:
            T_past = 0

        pos = (torch.arange(T, device=x.device) + T_past).clamp_max(self.max_seq_len - 1)
        h = self.embedding(x) + self.pos_embedding(pos)

        total_len = T_past + T
        mask = torch.tril(torch.ones(T, total_len, device=x.device))

        total_aux_loss = torch.tensor(0.0, device=x.device)
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            cache_in = kv_caches[i] if kv_caches is not None else None
            if use_cache:
                h, aux_loss, new_cache = layer(h, mask_attn=mask, modo=modo,
                                                kv_cache=cache_in, use_cache=True)
                new_caches.append(new_cache)
            else:
                h, aux_loss = layer(h, mask_attn=mask, modo=modo)
            total_aux_loss = total_aux_loss + aux_loss

        logits = self.lm_head(h)

        if use_cache:
            return logits, new_caches, total_aux_loss
        return logits, None, total_aux_loss


# =====================================================================
# 6. DATASET E PÓS-PROCESSAMENTO
# =====================================================================
class DatasetSimples(Dataset):
    def __init__(self, dados):
        self.samples = dados
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx][:-1], dtype=torch.long),
                torch.tensor(self.samples[idx][1:], dtype=torch.long))


def limpar_pontuacao(texto):
    if not texto:
        return texto

    correcoes = {
        r'\bpadrao\b': 'padrão', r'\bpadroes\b': 'padrões',
        r'\bchao\b': 'chão', r'\bpedaco\b': 'pedaço',
        r'\bvoce\b': 'você', r'\botimo\b': 'ótimo',
        r'\bbrasilia\b': 'Brasília', r'\bcoracao\b': 'coração',
        r'\bpais\b': 'país', r'\bvarias\b': 'várias',
        r'\bte\b': 'até'
    }
    for errado, certo in correcoes.items():
        texto = re.sub(errado, certo, texto, flags=re.IGNORECASE)

    texto = re.sub(r'\s+([.,!?])', r'\1', texto)
    texto = re.sub(r'([.,!?])([^\s])', r'\1 \2', texto)
    if texto:
        texto = texto[0].upper() + texto[1:]

    def capitalizar(match):
        return match.group(1) + match.group(2).upper()
    texto = re.sub(r'([.!?]\s+)([a-zçáéíóúâêôãõ])', capitalizar, texto)
    return texto.strip()


# =====================================================================
# 7. CONFIGURAÇÕES DE AMOSTRAGEM POR PERSONA
# =====================================================================
SAMPLING_CONFIG = {
    'logico':  {'temperature': 0.3, 'top_k': 20, 'top_p': 0.85, 'rep_penalty': 1.1},
    'roteiro': {'temperature': 0.7, 'top_k': 40, 'top_p': 0.95, 'rep_penalty': 1.15},
}


# =====================================================================
# 8. TREINADOR
# =====================================================================
class TGp5Trainer:
    def __init__(self, dim_emb=256, dim_ff=512, num_experts=4, num_layers=3,
                 max_seq_len=64, lr=0.001, vocab_size=8000):
        self.dim_emb = dim_emb
        self.dim_ff = dim_ff
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.lr = lr

        self.tok = TokenizadorTopologico(vocab_size=vocab_size)
        self.modelo = None
        self.optimizer = None
        self.criterion = None

    def preparar_dados(self, corpus_logico, corpus_roteiro):
        self.tok.construir_vocab(corpus_logico + corpus_roteiro)
        vocab_size = len(self.tok.vocabulario)

        loader_logico = DataLoader(
            DatasetSimples(self.tok.processar_linhas(corpus_logico, self.max_seq_len)),
            batch_size=32, shuffle=True
        )
        loader_roteiro = DataLoader(
            DatasetSimples(self.tok.processar_linhas(corpus_roteiro, self.max_seq_len)),
            batch_size=32, shuffle=True
        )

        self.modelo = MOLEDualRuntimeDeep(
            vocab_size=vocab_size,
            dim_emb=self.dim_emb,
            dim_ff=self.dim_ff,
            num_experts=self.num_experts,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
        ).to(device)

        self.optimizer = optim.AdamW(self.modelo.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tok.vocabulario['<pad>'])

        return loader_logico, loader_roteiro, vocab_size

    def treinar(self, loader_logico, loader_roteiro, vocab_size, epocas=10):
        print("\n🔗 Treinando TGp5...")
        for epoca in range(1, epocas + 1):
            self.modelo.train()
            loss_l, loss_r = 0.0, 0.0

            for x_b, y_b in loader_logico:
                x_b, y_b = x_b.to(device), y_b.to(device)
                self.optimizer.zero_grad()
                logits, _, aux = self.modelo(x_b, modo='logico')
                loss = self.criterion(logits.view(-1, vocab_size), y_b.view(-1)) + (0.01 * aux)
                loss.backward()
                self.optimizer.step()
                loss_l += loss.item()

            for x_b, y_b in loader_roteiro:
                x_b, y_b = x_b.to(device), y_b.to(device)
                self.optimizer.zero_grad()
                logits, _, aux = self.modelo(x_b, modo='roteiro')
                loss = self.criterion(logits.view(-1, vocab_size), y_b.view(-1)) + (0.01 * aux)
                loss.backward()
                self.optimizer.step()
                loss_r += loss.item()

            if epoca % 5 == 0 or epoca == 1:
                print(f"  Época {epoca:2d}/{epocas} | Lógico: {loss_l/len(loader_logico):.4f} | Roteiro: {loss_r/len(loader_roteiro):.4f}")

    def gerar_resposta(self, pergunta, modo='logico'):
        """
        Geração com KV cache + amostragem estocástica.
        Config por persona em SAMPLING_CONFIG.
        """
        self.modelo.eval()
        cfg = SAMPLING_CONFIG.get(modo, SAMPLING_CONFIG['logico'])

        prompt = f"<usr> {pergunta} <bot> [{modo}]"
        tokens = self.tok.tokenizar(prompt)
        eos_id = self.tok.vocabulario.get('<eos>', None)
        forbidden = [
            self.tok.vocabulario.get(t) for t in
            ['<usr>', '<bot>', '<pad>', '<unk>', '[logico]', '[roteiro]']
            if t in self.tok.vocabulario
        ]

        gerados = tokens.copy()

        with torch.no_grad():
            # Forward inicial do prompt para popular o cache
            input_seq = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, kv_caches, _ = self.modelo(input_seq, modo=modo, use_cache=True)

            for step in range(40):
                next_logits = logits[0, -1, :].clone()

                for tag in forbidden:
                    if tag is not None and tag < len(next_logits):
                        next_logits[tag] = float('-inf')

                # Penalidade de repetição
                counts = {}
                for t in gerados[len(tokens):]:
                    counts[t] = counts.get(t, 0) + 1
                for t_id, count in counts.items():
                    if t_id != eos_id and t_id < len(next_logits):
                        next_logits[t_id] -= (cfg['rep_penalty'] * count)

                # Temperatura
                next_logits = next_logits / max(cfg['temperature'], 1e-5)

                # Top-k
                if cfg['top_k'] > 0:
                    values, _ = torch.topk(next_logits, min(cfg['top_k'], next_logits.size(-1)))
                    threshold = values[-1]
                    next_logits[next_logits < threshold] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)

                # Top-p (nucleus)
                if 0 < cfg['top_p'] < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > cfg['top_p']
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    sorted_probs[remove] = 0.0
                    probs_filtrado = torch.zeros_like(probs)
                    probs_filtrado.scatter_(0, sorted_idx, sorted_probs)
                    probs = probs_filtrado / probs_filtrado.sum().clamp_min(1e-9)

                # Amostragem estocástica (não argmax)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == eos_id:
                    break

                gerados.append(next_token)

                # Forward do token novo reusando cache
                input_token = torch.tensor([[next_token]], dtype=torch.long, device=device)
                logits, kv_caches, _ = self.modelo(
                    input_token, modo=modo, kv_caches=kv_caches, use_cache=True
                )

                # Limita cache ao tamanho máximo
                if kv_caches[0][0].size(2) > self.max_seq_len:
                    kv_caches = [
                        (k[:, :, -self.max_seq_len:], v[:, :, -self.max_seq_len:])
                        for k, v in kv_caches
                    ]

        tokens_resposta = gerados[len(tokens):]
        texto = self.tok.decode(tokens_resposta)
        return limpar_pontuacao(texto)


# =====================================================================
# 9. EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("🧠 TGp5 v2.0 — BPE + KV Cache + Amostragem por Persona")
    print("=" * 65)

    CORPUS_BASE_LOGICO = [
        "<usr> oi , tudo bem ? <bot> [logico] sim , operando com estabilidade e pronto para auxiliar . <eos>",
        "<usr> o que é uma pedra ? <bot> [logico] pedra é um mineral sólido e duro formado por rochas e elementos da terra . <eos>",
        "<usr> como funciona uma rede neural ? <bot> [logico] uma rede neural conecta camadas de neurônios artificiais para propagar dados e reconhecer padrões complexos . <eos>",
        "<usr> qual a capital do brasil ? <bot> [logico] brasília é a capital federal do brasil , planejada e inaugurada em 1960 . <eos>",
        "<usr> o que é um pato ? <bot> [logico] pato é uma ave aquática com penas e bico . <eos>"
        "<usr> o pato  pode voar ? <bot> [logico] pato é uma ave que pode voar . <eos>"
    ] * 100

    CORPUS_BASE_ROTEIRO = [
        "<usr> oi , tudo bem ? <bot> [roteiro] sim , eu estou ótimo ! pronto para conversar sobre qualquer assunto que você quiser hoje . <eos>",
        "<usr> o que é uma pedra ? <bot> [roteiro] olha , uma pedra pode parecer só um pedaço de chão duro , mas é a verdadeira história da terra guardada em forma de rocha ! <eos>",
        "<usr> como funciona uma rede neural ? <bot> [roteiro] imagina várias mentes pequenas conversando entre si , passando sinais elétricos até que de repente elas compreendem o padrão ! <eos>",
        "<usr> qual a capital do brasil ? <bot> [roteiro] é brasília , uma cidade futurista em formato de avião perdida no coração do nosso país ! <eos>",
        "<usr> o que é um pato ? <bot> [roteiro] é um bicho nadador cheio de penas que vive fazendo barulho e nadando pelas lagoas ! <eos>"
        "<usr> o pato  pode voar? <bot> [roteiro] pensei em um ave que pode voa pelo mundo inteiro,alguns patos faz isso! <eos>"
    ] * 100

    trainer = TGp5Trainer(dim_emb=256, dim_ff=512, num_experts=4, num_layers=3,
                          max_seq_len=64, vocab_size=8000)
    loader_logico, loader_roteiro, vocab_size = trainer.preparar_dados(
        CORPUS_BASE_LOGICO, CORPUS_BASE_ROTEIRO
    )

    trainer.treinar(loader_logico, loader_roteiro, vocab_size, epocas=10)

    print("\n" + "=" * 65)
    print("🎭 GERAÇÃO COM AMOSTRAGEM POR PERSONA:")
    print("=" * 65)

    perguntas = [
        "oi, tudo bem?",
        "o que é uma pedra?",
        "como funciona uma rede neural?",
        "qual a capital do brasil?",
        "o pato ele pode voar?"
    ]

    for p in perguntas:
        print(f"\n📥 Entrada: {p}")
        print(f"📘 Lógico  : {trainer.gerar_resposta(p, modo='logico')}")
        print(f"🎭 Roteiro : {trainer.gerar_resposta(p, modo='roteiro')}")
