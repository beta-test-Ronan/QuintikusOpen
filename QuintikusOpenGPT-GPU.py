import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Configuração de Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Rodando no dispositivo: {device}")

# ============================================
# 1. CARREGAMENTO E TOKENIZAÇÃO (BPE)
# ============================================
class ProcessadorTextoBPE:
    def __init__(self):
        # Usa o tokenizer cl100k_base (padrão GPT-4/GPT-3.5)
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab

    def tokenizar_texto(self, texto: str):
        return self.enc.encode(texto, allowed_special={"<|endoftext|>"})

    def decodificar(self, ids):
        return self.enc.decode(ids)

class DatasetTextoGrande(Dataset):
    def __init__(self, tokens_ids, seq_len=32):
        self.seq_len = seq_len
        # Quebra os tokens em blocos sequenciais
        self.samples = []
        for i in range(0, len(tokens_ids) - seq_len, seq_len):
            chunk = tokens_ids[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ============================================
# 2. BLOCO MOE ESCALONADO (4 EXPERTS)
# ============================================
class CamadaMoE(nn.Module):
    def __init__(self, dim_emb=256, dim_ff=512, num_experts=4):
        super().__init__()
        self.router = nn.Linear(dim_emb, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_emb, dim_ff),
                nn.GELU(),
                nn.Linear(dim_ff, dim_emb)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        r_logits = self.router(x)
        r_probs = torch.softmax(r_logits, dim=-1)
        exp_outs = torch.stack([exp(x) for exp in self.experts], dim=-1)
        return torch.sum(r_probs.unsqueeze(-2) * exp_outs, dim=-1)

class BlocoTransformerMoE(nn.Module):
    def __init__(self, dim_emb=256, dim_ff=512, num_experts=4):
        super().__init__()
        self.Wq = nn.Linear(dim_emb, dim_emb, bias=False)
        self.Wk = nn.Linear(dim_emb, dim_emb, bias=False)
        self.Wv = nn.Linear(dim_emb, dim_emb, bias=False)
        self.norm1 = nn.LayerNorm(dim_emb)
        self.moe = CamadaMoE(dim_emb, dim_ff, num_experts)
        self.norm2 = nn.LayerNorm(dim_emb)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # Self-Attention Causal
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        x = self.norm1(x + context)
        moe_out = self.moe(x)
        out = self.norm2(x + moe_out)
        return out

class MOLEModelv7(nn.Module):
    def __init__(self, vocab_size, dim_emb=256, num_layers=4, num_experts=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.layers = nn.ModuleList([
            BlocoTransformerMoE(dim_emb=dim_emb, dim_ff=512, num_experts=num_experts)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(dim_emb, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        return self.lm_head(out)

# ============================================
# 3. EXECUÇÃO DO TREINO
# ============================================
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("🧠 MOLE v7.0 - TREINAMENTO DE GRANDE ESCALA (BPE + MOE ESCALONADO)")
    print("=" * 60)

    # NOME DO SEU ARQUIVO DE DADOS (ex: 'dataset_70k.txt')
    caminho_arquivo = '/kaggle/input/datasets/tironanbastos/mega-corpus/bd-mega.txt'

    # Se não houver arquivo local, gera um conjunto demonstrativo para teste de pipeline
    if not os.path.exists(caminho_arquivo):
        print("⚠️ Arquivo local não encontrado. Gerando amostra demonstrativa de teste...")
        amostras = [
            "A inteligência artificial transforma a maneira como desenvolvemos software e analisamos dados no dia a dia.",
            "O aprendizado de máquina exige pipelines bem estruturados e dados limpos para garantir boa generalização.",
            "Processamento de linguagem natural permite que modelos de linguagem entendam e gerem texto com alta fluidez."
        ]
        texto_completo = "\n".join(amostras * 20000) # Simula volume grande
    else:
        print(f"📖 Lendo dataset do arquivo: {caminho_arquivo}")
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            texto_completo = f.read()

    processador = ProcessadorTextoBPE()
    print("🔤 Tokenizando texto com BPE...")
    tokens_ids = processador.tokenizar_texto(texto_completo)
    print(f"📚 Total de tokens processados: {len(tokens_ids):,}")

    dataset = DatasetTextoGrande(tokens_ids, seq_len=32)
    val_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    modelo = MOLEModelv7(
        vocab_size=processador.vocab_size,
        dim_emb=256,
        num_layers=4,
        num_experts=4
    ).to(device)

    optimizer = optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    print(f"\n🔥 Treinando modelo de escala ({epochs} épocas em GPU)...")

    for epoca in range(1, epochs + 1):
        modelo.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = modelo(x_b)
            loss = criterion(logits.view(-1, processador.vocab_size), y_b.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        modelo.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = modelo(x_b)
                loss = criterion(logits.view(-1, processador.vocab_size), y_b.view(-1))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"  Época {epoca:2d}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("\n✨ TESTE DE GERAÇÃO COM TEXTO REAL:")
    modelo.eval()
    semente = "A inteligência artificial"
    tokens_semente = processador.tokenizar_texto(semente)
    input_ids = torch.tensor([tokens_semente], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(25):
            logits = modelo(input_ids)
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    texto_gerado = processador.decodificar(input_ids[0].tolist())
    print(f"Entrada: '{semente}' -> Gerado: '{texto_gerado}'")
