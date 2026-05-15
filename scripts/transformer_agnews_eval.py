"""
Transformer AG News standalone evaluation: load pre-trained weights and run Study.

Loads the vocabulary and model weights saved by transformer_agnews_study.py,
reconstructs the model, and runs the same quantization configs via Study API.

Run:  PYTHONPATH=. python scripts/transformer_agnews_eval.py [--weights scripts/weights/transformer_agnews.pt]
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.session import Study, QuantConfig


# ---------------------------------------------------------------------------
# AG News download
# ---------------------------------------------------------------------------

AG_NEWS_TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
AG_NEWS_TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
DATA_DIR = "/tmp/agnews_data"


def _ensure_test_data() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        print(f"Downloading AG News test set to {test_path} ...")
        urllib.request.urlretrieve(AG_NEWS_TEST_URL, test_path)
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        print(f"Downloading AG News train set to {train_path} ...")
        urllib.request.urlretrieve(AG_NEWS_TRAIN_URL, train_path)
    return test_path, train_path


# ---------------------------------------------------------------------------
# Tokeniser (mirrors training script)
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenise(text: str, vocab: dict[str, int], max_len: int = 64) -> list[int]:
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in text.lower().split()]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def load_agnews(path: str, vocab: dict[str, int], max_len: int = 64, limit: int | None = None):
    texts: list[list[int]] = []
    labels: list[int] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            label = int(row[0]) - 1
            text = row[1] + " " + row[2]
            texts.append(tokenise(text, vocab, max_len))
            labels.append(label)
            if limit and len(labels) >= limit:
                break
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Transformer model (mirrors training script)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

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
        max_len: int = 64,
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
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


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

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# Configs (mirrors training script)
# ---------------------------------------------------------------------------


def build_configs() -> list[QuantConfig]:
    return [
        QuantConfig(
            name="int4-pc",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="int4-pc-had",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            transform="hadamard",
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="int4-pc-sq",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            transform="smoothquant",
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="int4-pb32",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="int4-pb32-had",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            transform="hadamard",
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="int4-pb32-sq",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            transform="smoothquant",
            quantize_nonlinear=False,
        ),
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
    parser = argparse.ArgumentParser(description="Eval Transformer on AG News with quantization")
    parser.add_argument("--weights", default=None, help="Path to weights checkpoint")
    parser.add_argument("--vocab", default=None, help="Path to vocab file")
    args = parser.parse_args()

    torch.manual_seed(42)

    # -- Resolve weight / vocab paths --
    script_dir = os.path.dirname(__file__)
    weights_dir = os.path.join(script_dir, "weights")
    weights_path = args.weights or os.path.join(weights_dir, "transformer_agnews.pt")
    vocab_path = args.vocab or os.path.join(weights_dir, "transformer_agnews_vocab.pt")

    # -- Load checkpoint --
    print(f"Loading weights from {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]
    hparams = {
        "vocab_size": ckpt["vocab_size"],
        "num_classes": ckpt["num_classes"],
        "d_model": ckpt["d_model"],
        "nhead": ckpt["nhead"],
        "num_layers": ckpt["num_layers"],
        "dim_feedforward": ckpt["dim_feedforward"],
        "max_len": ckpt["max_len"],
    }

    # -- Load vocab --
    if os.path.exists(vocab_path):
        vocab_data = torch.load(vocab_path, map_location="cpu", weights_only=False)
        vocab = vocab_data["vocab"]
    else:
        vocab = ckpt["vocab"]
    print(f"  Vocabulary size: {len(vocab)}, hparams: {hparams}")

    # -- Build model and load weights --
    model = TransformerClassifier(**hparams)
    model.load_state_dict(model_state)
    model.eval()

    # -- Load test data --
    test_path, train_path = _ensure_test_data()
    test_x, test_y = load_agnews(test_path, vocab, max_len=hparams["max_len"], limit=7600)
    print(f"  Test data: {test_x.shape}")

    test_ds = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_ds, batch_size=128)

    # Verify loaded FP32 accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    fp32_acc = correct / total
    print(f"  Loaded model FP32 accuracy: {fp32_acc:.4f}  (saved: {ckpt.get('fp32_test_acc', 'N/A'):.4f})")

    # -- Calibration data (from training data) --
    train_x_small, _ = load_agnews(train_path, vocab, max_len=hparams["max_len"], limit=512)
    calib_ds = TensorDataset(train_x_small, torch.zeros(len(train_x_small), dtype=torch.long))
    calib_loader = DataLoader(calib_ds, batch_size=64)
    calib_samples = []
    for x, _y in calib_loader:
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
