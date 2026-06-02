"""
Transformer Quantization Study — Tiny Shakespeare character-level GPT.

Trains a mini GPT on the Tiny Shakespeare dataset, then compares
quantization configs using the Study API with perplexity as the metric.

Configs: int4 per_block/per_channel/per_tensor, hadamard, smoothquant, NF4, int8.

Run:  PYTHONPATH=. python scripts/transformer_quant_study.py
"""

from __future__ import annotations

import copy
import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.session import Study, QuantConfig


# ---------------------------------------------------------------------------
# Data: Tiny Shakespeare (character-level)
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "/tmp/tinyshakespeare.txt"


def download_shakespeare():
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    with open(DATA_PATH) as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, encode, decode


def make_dataloaders(block_size=128, batch_size=64):
    train_data, val_data, vocab_size, _, _ = download_shakespeare()

    def _loader(data_tensor, shuffle):
        # Create (x, y) pairs where y is x shifted by 1
        n = (len(data_tensor) - 1) // block_size * block_size
        xs = data_tensor[:n].view(-1, block_size)
        ys = data_tensor[1:n + 1].view(-1, block_size)
        ds = TensorDataset(xs, ys)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _loader(train_data, True)
    val_loader = _loader(val_data, False)
    return train_loader, val_loader, vocab_size


# ---------------------------------------------------------------------------
# Mini GPT model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool(),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_heads=4, n_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, train_loader, val_loader, epochs=3, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        train_ppl = math.exp(total_loss / total_tokens)

        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                val_loss += F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1),
                    reduction="sum",
                ).item()
                val_tokens += y.numel()
        val_ppl = math.exp(val_loss / val_tokens)
        best_loss = min(best_loss, val_loss / val_tokens)

        print(f"  Epoch {epoch + 1}: train_ppl={train_ppl:.2f}  val_ppl={val_ppl:.2f}")

    print(f"  Best val loss: {best_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Eval function
# ---------------------------------------------------------------------------


def eval_fn(model, data):
    model.eval()
    if isinstance(data, list):
        with torch.no_grad():
            for batch in data:
                model(batch)
        return {}
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in data:
            logits = model(x)
            total_loss += F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                reduction="sum",
            ).item()
            total_tokens += y.numel()
    ppl = math.exp(total_loss / total_tokens)
    return {"perplexity": ppl}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(42)

    # -- Data --
    print("=== Preparing data ===")
    train_loader, val_loader, vocab_size = make_dataloaders(block_size=128, batch_size=32)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # -- Model --
    model = MiniGPT(vocab_size=vocab_size, d_model=192, n_heads=3, n_layers=4, block_size=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # -- Train --
    print("\n=== Training MiniGPT ===")
    model = train(model, train_loader, val_loader, epochs=4, lr=3e-4)

    # FP32 baseline
    model.eval()
    val_loss, val_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            val_loss += F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum",
            ).item()
            val_tokens += y.numel()
    fp32_ppl = math.exp(val_loss / val_tokens)
    print(f"  FP32 val perplexity: {fp32_ppl:.2f}")

    # Save weights
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(weights_dir, "shakespeare_gpt.pt"))

    # -- Calibration data --
    calib_data = []
    for x, _y in train_loader:
        calib_data.append(x)
        if len(calib_data) >= 4:
            break

    # -- Configs --
    configs = [
        # W4A4 granularity sweep
        QuantConfig(
            name="int4-pb32 (W4A4)",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="int4-pc (W4A4)",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="int4-pt (W4A4)",
            w_format="int4", a_format="int4",
            w_granularity="per_tensor", a_granularity="per_tensor",
            quantize_nonlinear=False,
        ),
        # Hadamard
        QuantConfig(
            name="int4-pb32-had",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            transform="hadamard",
            quantize_nonlinear=False,
        ),
        # SmoothQuant
        QuantConfig(
            name="int4-pb32-sq",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            transform="smoothquant",
            quantize_nonlinear=False,
        ),
        # W2A2
        QuantConfig(
            name="int2-pt (W2A2)",
            w_format="int2", a_format="int2",
            w_granularity="per_tensor", a_granularity="per_tensor",
            quantize_nonlinear=False,
        ),
        # NF4
        QuantConfig(
            name="nf4-pc",
            w_format="nf4",
            w_granularity="per_channel", w_axis=0,
            weight_only=True,
            quantize_nonlinear=False,
        ),
        # INT8 reference
        QuantConfig(
            name="int8-pc (W8A8)",
            w_format="int8", a_format="int8",
            w_granularity="per_channel", a_granularity="per_channel",
            quantize_nonlinear=False,
        ),
    ]

    # -- Run Study --
    print("\n=== Running Quantization Study ===")
    model_copy = copy.deepcopy(model)
    study = Study(configs, model=model_copy)

    report = study.run(
        calib_data,
        eval_data=val_loader,
        eval_fn=eval_fn,
        outputs="default",
    )

    print("\n===== print_summary() =====")
    report.print_summary()

    print("\n===== Per-Config Detail =====")
    for part in report.parts:
        for r in report._results[part]:
            print(f"\n--- {r.name} ---")
            print(r.summary())
            print(r.accuracy_table())


if __name__ == "__main__":
    main()
