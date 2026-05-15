"""
Transformer AG News E2E Experiment: MXINT4 W4A4 + SmoothQuant / Hadamard via Study API.

Trains a small Transformer classifier on AG News, then compares quantization configs:
  - int4 per_channel W4A4 (no transform)
  - int4 per_channel W4A4 + Hadamard
  - int4 per_channel W4A4 + SmoothQuant
  - int4 per_block(32) W4A4 (MX-style, no transform)
  - int4 per_block(32) W4A4 + Hadamard
  - int4 per_block(32) W4A4 + SmoothQuant
  - int8 per_channel W8A8 (reference)

Only the Study API is used — no low-level Session/QuantSession calls.
Run:  PYTHONPATH=. python scripts/transformer_agnews_study.py
"""

from __future__ import annotations

import copy
import csv
import os
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.session import Study, QuantConfig


# ---------------------------------------------------------------------------
# AG News download
# ---------------------------------------------------------------------------

AG_NEWS_TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
AG_NEWS_TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
DATA_DIR = "/tmp/agnews_data"


def _download_agnews():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(train_path):
        print(f"Downloading AG News train set to {train_path} ...")
        urllib.request.urlretrieve(AG_NEWS_TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        print(f"Downloading AG News test set to {test_path} ...")
        urllib.request.urlretrieve(AG_NEWS_TEST_URL, test_path)
    return train_path, test_path


# ---------------------------------------------------------------------------
# Tokeniser / vocabulary
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocab(train_path: str, max_vocab: int = 10000) -> dict[str, int]:
    word_freq: dict[str, int] = {}
    with open(train_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            text = row[1] + " " + row[2]
            for token in text.lower().split():
                word_freq[token] = word_freq.get(token, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda kv: -kv[1])
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in sorted_words[: max_vocab - 2]:
        vocab[word] = len(vocab)
    return vocab


def tokenise(text: str, vocab: dict[str, int], max_len: int = 128) -> list[int]:
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in text.lower().split()]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def load_agnews(path: str, vocab: dict[str, int], max_len: int = 128, limit: int | None = None):
    texts: list[list[int]] = []
    labels: list[int] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            label = int(row[0]) - 1  # AG News labels are 1-4 → 0-3
            text = row[1] + " " + row[2]
            texts.append(tokenise(text, vocab, max_len))
            labels.append(label)
            if limit and len(labels) >= limit:
                break
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # (batch, d_model) — mean pooling
        return self.classifier(x)  # (batch, num_classes)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(model, train_loader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        acc = correct / total
        print(f"  Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}  acc={acc:.4f}")


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Eval function (handles both calib/analysis and evaluation phases)
# ---------------------------------------------------------------------------


def eval_fn(model, data):
    """Unified eval function for Study.

    Calibration/analysis phase: *data* is a list of tensors.
    Evaluation phase: *data* is a DataLoader yielding (x, y) batches.
    """
    model.eval()

    if isinstance(data, list):
        with torch.no_grad():
            for batch in data:
                model(batch)
        return {}

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def build_configs() -> list[QuantConfig]:
    return [
        # -- per_channel int4 W4A4 (baseline) --
        QuantConfig(
            name="int4-pc",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            quantize_nonlinear=False,
        ),
        # -- per_channel int4 W4A4 + Hadamard --
        QuantConfig(
            name="int4-pc-had",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            transform="hadamard",
            quantize_nonlinear=False,
        ),
        # -- per_channel int4 W4A4 + SmoothQuant --
        QuantConfig(
            name="int4-pc-sq",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            transform="smoothquant",
            quantize_nonlinear=False,
        ),
        # -- per_block(32) int4 W4A4 (MX-style, no transform) --
        QuantConfig(
            name="int4-pb32",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        # -- per_block(32) int4 W4A4 + Hadamard --
        QuantConfig(
            name="int4-pb32-had",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            transform="hadamard",
            quantize_nonlinear=False,
        ),
        # -- per_block(32) int4 W4A4 + SmoothQuant --
        QuantConfig(
            name="int4-pb32-sq",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            transform="smoothquant",
            quantize_nonlinear=False,
        ),
        # -- int8 per_channel W8A8 (reference) --
        QuantConfig(
            name="int8-pc",
            w_format="int8", a_format="int8",
            w_granularity="per_channel", a_granularity="per_channel",
            quantize_nonlinear=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(42)

    # -- Download & load data --
    train_path, test_path = _download_agnews()

    print("=== Building vocabulary ===")
    vocab = build_vocab(train_path, max_vocab=10000)
    print(f"  Vocabulary size: {len(vocab)}")

    print("=== Loading data ===")
    # Use a subset (20k train) for fast iteration; full 120k takes longer
    train_x, train_y = load_agnews(train_path, vocab, max_len=64, limit=20000)
    test_x, test_y = load_agnews(test_path, vocab, max_len=64, limit=7600)
    print(f"  Train: {train_x.shape}, Test: {test_x.shape}")

    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    # -- Train model --
    print("=== Training Transformer on AG News ===")
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=4,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_len=64,
        dropout=0.1,
    )
    train_model(model, train_loader, epochs=5, lr=1e-3)

    # Baseline FP32 accuracy
    fp32_test_acc = evaluate_model(model, test_loader)
    print(f"  FP32 test accuracy: {fp32_test_acc:.4f}")

    # -- Save model weights --
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "transformer_agnews.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "vocab_size": len(vocab),
        "num_classes": 4,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "max_len": 64,
        "fp32_test_acc": fp32_test_acc,
    }
    torch.save(checkpoint, weights_path)
    print(f"  Weights saved to {weights_path}")

    # Also save vocab for the standalone eval script
    vocab_path = os.path.join(weights_dir, "transformer_agnews_vocab.pt")
    torch.save({"vocab": vocab}, vocab_path)

    # -- Calibration data --
    calib_samples = []
    for x, _y in train_loader:
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    # -- Run Study --
    configs = build_configs()
    print(f"\n=== Running Quantization Study ({len(configs)} configs) ===")
    model_copy = copy.deepcopy(model)
    study = Study(configs, model=model_copy)

    report = study.run(
        calib_samples,
        eval_data=test_loader,
        eval_fn=eval_fn,
        outputs="default",
    )

    # -- Results --
    print("\n===== Study print_summary() =====")
    report.print_summary()

    print("\n===== Per-Config Summary + Accuracy Table =====")
    for part in report.parts:
        for r in report._results[part]:
            print(f"\n--- {r.name} ---")
            print(r.summary())
            print(r.accuracy_table())


if __name__ == "__main__":
    main()
