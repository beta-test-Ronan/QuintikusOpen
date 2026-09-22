import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# =====================================================================
# #________________________ 1. CONFIGURAÇÕES ________________________#
# =====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Dispositivo Ativo: {device}")

# =====================================================================
# #________________________ 2. MODELO MOE ___________________________#
# =====================================================================
class DualPersonaGDIRouter(nn.Module):
    """
    Roteador Geométrico baseado em Divergência KL (GDI).
    Direciona o fluxo de tensores para os especialistas específicos do modo.
    """
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
        kl_div = torch.sum(p_x_exp * (torch.log(p_x_exp + self.eps) - torch.log(p_e_exp + self.eps)), dim=-1)
        router_weights = F.softmax(-kl_div, dim=-1)

        mask = torch.zeros_like(router_weights)
        if modo == 'logico': 
            mask[..., :2] = 1.0  # Especialistas 0 e 1 ativados para o modo Lógico
        elif modo == 'roteiro': 
            mask[..., 2:] = 1.0  # Especialistas 2 e 3 ativados para o modo Roteiro
        else: 
            mask[..., :] = 1.0

        router_weights = router_weights * mask
        router_weights = router_weights / (torch.sum(router_weights, dim=-1, keepdim=True) + self.eps)

        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            P = torch.mean(router_weights, dim=[0, 1])
            top1_experts = torch.argmax(router_weights, dim=-1)
            tokens_per_expert = F.one_hot(top1_experts, num_classes=self.num_experts).float()
            f = torch.mean(tokens_per_expert, dim=[0, 1])
            aux_loss = self.num_experts * torch.sum(P * f)
            
        return router_weights, aux_loss

class DeepTransformerMoELayer(nn.Module):
    """
    Camada de Transformer com Atenção e Bloco de Especialistas (MoE).
    """
    def __init__(self, dim_emb, dim_ff, num_experts):
        super().__init__()
        self.dim_emb = dim_emb
        self.Wq = nn.Linear(dim_emb, dim_emb, bias=False)
        self.Wk = nn.Linear(dim_emb, dim_emb, bias=False)
        self.Wv = nn.Linear(dim_emb, dim_emb, bias=False)
        self.router = DualPersonaGDIRouter(dim_emb, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim_emb, dim_ff), nn.GELU(), nn.Linear(dim_ff, dim_emb)) 
            for _ in range(num_experts)
        ])
        self.norm1 = nn.LayerNorm(dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)

    def forward(self, x, mask_attn=None, modo='logico'):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_emb ** 0.5)
        if mask_attn is not None:
            scores = scores.masked_fill(mask_attn == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        x = self.norm1(x + torch.matmul(attn, V))

        r_probs, aux_loss = self.router(x, modo=modo)
        exp_outs = torch.stack([exp(x) for exp in self.experts], dim=-1)
        moe_out = torch.sum(r_probs.unsqueeze(-2) * exp_outs, dim=-1)
        return self.norm2(x + moe_out), aux_loss

class MOLEDualRuntimeDeep(nn.Module):
    """
    Rede Neural do TGp5 MoE contendo múltiplas camadas de atenção e roteamento.
    """
    def __init__(self, vocab_size, dim_emb=256, dim_ff=512, num_experts=4, num_layers=3, max_seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_emb, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, dim_emb)
        self.layers = nn.ModuleList([DeepTransformerMoELayer(dim_emb, dim_ff, num_experts) for _ in range(num_layers)])
        self.lm_head = nn.Linear(dim_emb, vocab_size)

    def forward(self, x, modo='logico'):
        seq_len = x.shape[1]
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            h, aux_loss = layer(h, mask_attn=mask, modo=modo)
            total_aux_loss += aux_loss

        return self.lm_head(h), None, total_aux_loss, None

# =====================================================================
# #________________________ 3. PROCESSAMENTO ________________________#
# =====================================================================
class TokenizadorTopologico:
    """
    Tokenizador personalizado para conversão de texto em índices do vocabulário.
    """
    def __init__(self):
        self.vocabulario = OrderedDict()
        self.idx_para_word = {}

    def construir_vocab(self, corpus_total):
        texto_geral = " ".join(corpus_total).lower()
        tokens_raw = re.findall(r'<usr>|<bot>|<eos>|\[logico\]|\[roteiro\]|\w+|[^\w\s]', texto_geral)
        vocab_set = sorted(list(set(tokens_raw)))
        for tag in ['<pad>', '<unk>']:
            if tag not in vocab_set: 
                vocab_set.append(tag)
        self.vocabulario = {t: i for i, t in enumerate(vocab_set)}
        self.idx_para_word = {i: t for i, t in enumerate(vocab_set)}

    def processar_linhas(self, linhas, max_len=48):
        dados = []
        for texto in linhas:
            words = re.findall(r'<usr>|<bot>|<eos>|\[logico\]|\[roteiro\]|\w+|[^\w\s]', texto.lower())
            indices = [self.vocabulario.get(w, self.vocabulario['<unk>']) for w in words]
            if len(indices) < max_len: 
                indices += [self.vocabulario['<pad>']] * (max_len - len(indices))
            else: 
                indices = indices[:max_len]
            dados.append(indices)
        return dados

    def tokenizar(self, texto: str):
        words = re.findall(r'<usr>|<bot>|<eos>|\[logico\]|\[roteiro\]|\w+|[^\w\s]', texto.lower())
        return [self.vocabulario.get(w, self.vocabulario['<unk>']) for w in words]

class DatasetSimples(Dataset):
    def __init__(self, dados): 
        self.samples = dados
    def __len__(self): 
        return len(self.samples)
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx][:-1], dtype=torch.long), torch.tensor(self.samples[idx][1:], dtype=torch.long)

def limpar_pontuacao(texto):
    """
    Pós-processamento ortográfico básico e formatação de saída.
    """
    if not texto: return texto
    
    correcoes = {
        r'\bpadrao\b': 'padrão', r'\bpadroes\b': 'padrões',
        r'\bchao\b': 'chão', r'\bpedaco\b': 'pedaço',
        r'\bvoce\b': 'você', r'\botimo\b': 'ótimo',
        r'\bbrasilia\b': 'Brasília', r'\bcoracao\b': 'coração',
        r'\bpais\b': 'país', r'\bvarias\b': 'várias',
        r'\bte\b': 'até', r'\b(e)\b(?=\s+um|\s+uma|\s+a|\s+o|\s+brasilia|\s+a\s+verdadeira)': 'é'
    }
    for errado, certo in correcoes.items():
        texto = re.sub(errado, certo, texto, flags=re.IGNORECASE)

    texto = re.sub(r'\s+([.,!?])', r'\1', texto)
    texto = re.sub(r'([.,!?])([^\s])', r'\1 \2', texto)
    texto = texto[0].upper() + texto[1:]
    
    def capitalizar_apos_pontuacao(match):
        return match.group(1) + match.group(2).upper()
    
    texto = re.sub(r'([.!?]\s+)([a-zçáéíóúâêôãõ])', capitalizar_apos_pontuacao, texto)
    return texto.strip()

# =====================================================================
# #________________________ 4. TREINADOR & ORQUESTRADOR ____________#
# =====================================================================
class TGp5Trainer:
    """
    Classe responsável por orquestrar o dataset, treinamento e inferência com controle de temperatura.
    """
    def __init__(self, dim_emb=256, dim_ff=512, num_experts=4, num_layers=3, max_seq_len=64, lr=0.001):
        self.dim_emb = dim_emb
        self.dim_ff = dim_ff
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.lr = lr
        
        self.tok = TokenizadorTopologico()
        self.modelo = None
        self.optimizer = None
        self.criterion = None

    def preparar_dados(self, corpus_logico, corpus_roteiro):
        self.tok.construir_vocab(corpus_logico + corpus_roteiro)
        vocab_size = len(self.tok.vocabulario)

        loader_logico = DataLoader(DatasetSimples(self.tok.processar_linhas(corpus_logico)), batch_size=32, shuffle=True)
        loader_roteiro = DataLoader(DatasetSimples(self.tok.processar_linhas(corpus_roteiro)), batch_size=32, shuffle=True)

        self.modelo = MOLEDualRuntimeDeep(
            vocab_size=vocab_size, 
            dim_emb=self.dim_emb, 
            dim_ff=self.dim_ff, 
            num_experts=self.num_experts, 
            num_layers=self.num_layers, 
            max_seq_len=self.max_seq_len
        ).to(device)

        self.optimizer = optim.AdamW(self.modelo.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tok.vocabulario['<pad>'])

        return loader_logico, loader_roteiro, vocab_size

    def treinar(self, loader_logico, loader_roteiro, vocab_size, epocas=10):
        print("\n🔗 Treinando SLM...")
        for epoca in range(1, epocas + 1):
            self.modelo.train()
            loss_l, loss_r = 0.0, 0.0

            for x_b, y_b in loader_logico:
                x_b, y_b = x_b.to(device), y_b.to(device)
                self.optimizer.zero_grad()
                logits, _, aux, _ = self.modelo(x_b, modo='logico')
                loss = self.criterion(logits.view(-1, vocab_size), y_b.view(-1)) + (0.01 * aux)
                loss.backward()
                self.optimizer.step()
                loss_l += loss.item()

            for x_b, y_b in loader_roteiro:
                x_b, y_b = x_b.to(device), y_b.to(device)
                self.optimizer.zero_grad()
                logits, _, aux, _ = self.modelo(x_b, modo='roteiro')
                loss = self.criterion(logits.view(-1, vocab_size), y_b.view(-1)) + (0.01 * aux)
                loss.backward()
                self.optimizer.step()
                loss_r += loss.item()

            if epoca % 5 == 0 or epoca == 1:
                print(f"  Época {epoca:2d}/{epocas} | Loss Lógico: {loss_l/len(loader_logico):.4f} | Loss Roteiro: {loss_r/len(loader_roteiro):.4f}")

    def gerar_resposta(self, pergunta, modo='logico', temperatura=0.3, rep_penalty=1.2):
        """
        Inferencia autônoma do modelo neural com controle fino de temperatura:
          - Temperatura baixa (ex: 0.1 - 0.3): Mais factual, estável e direto (Ideal para o Modo Lógico).
          - Temperatura mais alta (ex: 0.6 - 0.8): Mais criativo e variado (Ideal para o Modo Roteiro).
        """
        self.modelo.eval()
        prompt = f"<usr> {pergunta} <bot> [{modo}]"
        tokens = self.tok.tokenizar(prompt)
        eos_id = self.tok.vocabulario.get('<eos>', None)
        forbidden = [self.tok.vocabulario.get(t) for t in ['<usr>', '<bot>', '<pad>', '<unk>', '[logico]', '[roteiro]'] if t in self.tok.vocabulario]

        gerados = tokens.copy()
        with torch.no_grad():
            for step in range(35):
                input_seq = torch.tensor([gerados], dtype=torch.long, device=device)
                logits, _, _, _ = self.modelo(input_seq, modo=modo)
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
                        next_logits[t_id] -= (rep_penalty * count)

                # Aplicação da Temperatura
                probs = torch.softmax(next_logits / max(temperatura, 1e-5), dim=-1)
                next_token = torch.argmax(probs).item()
                if next_token == eos_id: 
                    break
                gerados.append(next_token)

        resposta_crua = [self.tok.idx_para_word.get(i, '') for i in gerados[len(tokens):] if i not in forbidden and i != eos_id]
        texto_junto = " ".join(resposta_crua)
        
        return limpar_pontuacao(texto_junto)

# =====================================================================
# #________________________ 5. EXECUÇÃO PRINCIPAL ____________________#
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("🧠 TGp5 MoE — SLM AUTÔNOMO DE LINGUAGEM E PERSONALIDADE")
    print("=" * 65)

    CORPUS_BASE_LOGICO = [
        "<usr> oi , tudo bem ? <bot> [logico] sim , operando com estabilidade e pronto para auxiliar . <eos>",
        "<usr> o que é uma pedra ? <bot> [logico] pedra é um mineral sólido e duro formado por rochas e elementos da terra . <eos>",
        "<usr> como funciona uma rede neural ? <bot> [logico] uma rede neural conecta camadas de neurônios artificiais para propagar dados e reconhecer padrões complexos . <eos>",
        "<usr> qual a capital do brasil ? <bot> [logico] brasília é a capital federal do brasil , planejada e inaugurada em 1960 . <eos>",
        "<usr> o que é um pato ? <bot> [logico] pato é uma ave aquática com penas e bico . <eos>"
    ] * 100 

    CORPUS_BASE_ROTEIRO = [
        "<usr> oi , tudo bem ? <bot> [roteiro] sim , eu estou ótimo ! pronto para conversar sobre qualquer assunto que você quiser hoje . <eos>",
        "<usr> o que é uma pedra ? <bot> [roteiro] olha , uma pedra pode parecer só um pedaço de chão duro , mas é a verdadeira história da terra guardada em forma de rocha ! <eos>",
        "<usr> como funciona uma rede neural ? <bot> [roteiro] imagina várias mentes pequenas conversando entre si , passando sinais elétricos até que de repente elas compreendem o padrão ! <eos>",
        "<usr> qual a capital do brasil ? <bot> [roteiro] é brasília , uma cidade futurista em formato de avião perdida no coração do nosso país ! <eos>",
        "<usr> o que é um pato ? <bot> [roteiro] é um bicho nadador cheio de penas que vive fazendo barulho e nadando pelas lagoas ! <eos>"
    ] * 100

    # Inicialização da Classe Orquestradora
    trainer = TGp5Trainer(dim_emb=256, dim_ff=512, num_experts=4, num_layers=3)
    loader_logico, loader_roteiro, vocab_size = trainer.preparar_dados(CORPUS_BASE_LOGICO, CORPUS_BASE_ROTEIRO)
    
    # #________________________ TREINAMENTO ________________________#
    trainer.treinar(loader_logico, loader_roteiro, vocab_size, epocas=10)

    # #________________________ TESTE DE GERACAO ____________________#
    print("\n" + "=" * 65)
    print("🎭 TESTANDO GERAÇÃO AUTÔNOMA COM AJUSTE FINO DE TEMPERATURA:")
    print("=" * 65)

    perguntas = [
        "oi, tudo bem?",
        "o que é uma pedra?",
        "como funciona uma rede neural?",
        "qual a capital do brasil?",
        "o que é um patos?"
    ]
    
    for p in perguntas:
        print(f"\n📥 Entrada: {p}")
        # Modo Lógico: Temperatura baixa (0.2) -> Factual e Consistente
        print(f"📘 Modo Lógico  (Temp 0.2): {trainer.gerar_resposta(p, modo='logico', temperatura=0.2)}")
        # Modo Roteiro: Temperatura ajustada (0.6) -> Fluidez e Expressividade
        print(f"🎭 Modo Roteiro (Temp 0.6): {trainer.gerar_resposta(p, modo='roteiro', temperatura=0.6)}")
