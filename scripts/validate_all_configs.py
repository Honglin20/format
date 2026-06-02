"""
Comprehensive validation: all format × granularity × transform × GPTQ combinations
at W4A4, compared against int4 per_channel baseline.

Loads pre-trained weights, runs Study on both MNIST MLP and Transformer AG News,
checks expectations, and flags anomalies.

Run:  PYTHONPATH=. python scripts/validate_all_configs.py
"""

from __future__ import annotations

import copy
import csv
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.session import Study, QuantConfig


# ══════════════════════════════════════════════════════════════════════════════
# Expectation model
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Expectation:
    """What we expect from a config relative to the baseline (int4 per_channel)."""

    label: str  # short description
    # Directional checks (relative to baseline quant_accuracy)
    should_be_better: bool = False  # quant acc should be > baseline
    should_be_worse: bool = False  # quant acc should be < baseline
    max_drop: float = 1.0  # maximum acceptable accuracy drop (absolute)
    check_fp32_nonzero: bool = True  # fp32_accuracy must not be 0


@dataclass
class ConfigSpec:
    config: QuantConfig
    expectations: list[Expectation] = field(default_factory=list)
    category: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════


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


def run_study(model, configs, calib_samples, eval_loader, *, max_workers=None):
    """Run a Study and return the report."""
    model_copy = copy.deepcopy(model)
    study = Study(configs, model=model_copy)
    return study.run(calib_samples, eval_data=eval_loader, eval_fn=eval_fn, outputs="default")


def get_quant_accuracy(report, name: str) -> float | None:
    """Extract quant_accuracy for a config by name from a StudyReport."""
    for part in report.parts:
        for r in report._results[part]:
            if r.name == name and r.quant_metrics:
                return list(r.quant_metrics.values())[0]
    return None


def get_fp32_accuracy(report, name: str) -> float | None:
    """Extract fp32_accuracy for a config by name."""
    for part in report.parts:
        for r in report._results[part]:
            if r.name == name and r.fp32_metrics:
                return list(r.fp32_metrics.values())[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MNIST model
# ══════════════════════════════════════════════════════════════════════════════


def build_mnist_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def load_mnist(weights_path: str):
    """Load or train the MNIST MLP."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = build_mnist_model()

    if os.path.exists(weights_path):
        print(f"  Loading MNIST weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("  Training MNIST MLP (no cached weights)...")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for ep in range(5):
            model.train()
            for x, y in train_loader:
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"  MNIST FP32 accuracy: {correct / total:.4f}")

    # Calibration batches
    calib_samples = []
    for x, _y in train_loader:
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    return model, test_loader, calib_samples


# ══════════════════════════════════════════════════════════════════════════════
# Transformer / AG News model
# ══════════════════════════════════════════════════════════════════════════════


AG_NEWS_TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
AG_NEWS_TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
DATA_DIR = "/tmp/agnews_data"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _ensure_agnews():
    os.makedirs(DATA_DIR, exist_ok=True)
    for url, name in [(AG_NEWS_TRAIN_URL, "train.csv"), (AG_NEWS_TEST_URL, "test.csv")]:
        p = os.path.join(DATA_DIR, name)
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
    return os.path.join(DATA_DIR, "train.csv"), os.path.join(DATA_DIR, "test.csv")


def _tokenise(text: str, vocab: dict[str, int], max_len: int = 64) -> list[int]:
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in text.lower().split()]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def _load_agnews_csv(path: str, vocab: dict[str, int], max_len: int = 64, limit: int | None = None):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            labels.append(int(row[0]) - 1)
            texts.append(_tokenise(row[1] + " " + row[2], vocab, max_len))
            if limit and len(labels) >= limit:
                break
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=4, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


def load_transformer(weights_path: str):
    """Load the pre-trained Transformer from checkpoint."""
    train_csv, test_csv = _ensure_agnews()
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    hparams = {k: ckpt[k] for k in ["vocab_size", "num_classes", "d_model", "nhead",
                                      "num_layers", "dim_feedforward", "max_len"]}

    test_x, test_y = _load_agnews_csv(test_csv, vocab, max_len=hparams["max_len"], limit=7600)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=128)

    model = TransformerClassifier(**hparams)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    acc = 0
    with torch.no_grad():
        correct, total = 0, 0
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
        acc = correct / total
    print(f"  Transformer FP32 accuracy: {acc:.4f}")

    # Calibration from training data subset
    train_x, _ = _load_agnews_csv(train_csv, vocab, max_len=hparams["max_len"], limit=512)
    calib_loader = DataLoader(TensorDataset(train_x, torch.zeros(len(train_x), dtype=torch.long)), batch_size=64)
    calib_samples = []
    for x, _ in calib_loader:
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    return model, test_loader, calib_samples


# ══════════════════════════════════════════════════════════════════════════════
# Config definitions
# ══════════════════════════════════════════════════════════════════════════════

# Shorthand builders
def _cfg(name, **kw):
    """Build a QuantConfig with int4 W4A4 defaults."""
    defaults = dict(
        name=name,
        w_format="int4", a_format="int4",
        w_granularity="per_channel", a_granularity="per_channel",
        w_axis=-1, a_axis=-1,
        quantize_nonlinear=False,
    )
    defaults.update(kw)
    return QuantConfig(**defaults)


def build_all_configs() -> list[ConfigSpec]:
    """Every config we want to validate, with expectations."""
    specs: list[ConfigSpec] = []

    # ── 0. Reference ──────────────────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int8-pc", w_format="int8", a_format="int8"),
        [Expectation("int8 reference", should_be_better=True)],
        "reference",
    ))

    # ── 1. Baseline ───────────────────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int4-pc"),
        [Expectation("baseline (no check)")],
        "baseline",
    ))

    # ── 2. Granularity ────────────────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int4-pt", w_granularity="per_tensor", a_granularity="per_tensor"),
        [Expectation("per_tensor < per_channel", should_be_worse=True)],
        "granularity",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pb32", w_granularity="per_block", a_granularity="per_block",
             w_block_size=32, a_block_size=32),
        [Expectation("per_block32 >= per_channel", should_be_better=True)],
        "granularity",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pb16", w_granularity="per_block", a_granularity="per_block",
             w_block_size=16, a_block_size=16),
        [Expectation("per_block16 ≈ per_channel")],
        "granularity",
    ))

    # ── 3. Scale storage ──────────────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int4-pc-fp32", scale_storage="fp32"),
        [Expectation("fp32 scale — POT may act as regularizer on small models")],
        "scale_storage",
    ))

    # ── 4. Calibrator ─────────────────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int4-pc-max", calibrator="max"),
        [Expectation("max calibrator <= mse calibrator", should_be_worse=True)],
        "calibrator",
    ))

    # ── 5. Format × per_channel ───────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("fp4-pc", w_format="fp4_e2m1", a_format="fp4_e2m1"),
        [Expectation("fp4 vs int4 comparison")],
        "format",
    ))
    specs.append(ConfigSpec(
        _cfg("nf4-pc", w_format="nf4", a_format="nf4"),
        [Expectation("nf4 vs int4 comparison")],
        "format",
    ))

    # ── 6. Format × per_block ─────────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("fp4-pb32", w_format="fp4_e2m1", a_format="fp4_e2m1",
             w_granularity="per_block", a_granularity="per_block",
             w_block_size=32, a_block_size=32),
        [Expectation("fp4 pb32: should beat fp4 pc")],
        "format_x_granularity",
    ))
    specs.append(ConfigSpec(
        _cfg("nf4-pb32", w_format="nf4", a_format="nf4",
             w_granularity="per_block", a_granularity="per_block",
             w_block_size=32, a_block_size=32),
        [Expectation("nf4 pb32: should beat nf4 pc")],
        "format_x_granularity",
    ))

    # ── 7. Transform × per_channel ────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int4-pc-had", transform="hadamard"),
        [Expectation("hadamard: architecture-dependent")],
        "transform_pc",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pc-sq", transform="smoothquant"),
        [Expectation("smoothquant: Transformer good, MNIST moderate")],
        "transform_pc",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pc-prescale", transform="prescale"),
        [Expectation("prescale ones: should ≈ baseline (identity init)")],
        "transform_pc",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pc-prescale-amax", transform="prescale", prescale_init="amax"),
        [Expectation("prescale amax: per-channel activation-based init")],
        "transform_pc",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pc-prescale-pot", transform="prescale", prescale_init="amax", prescale_pot=True),
        [Expectation("prescale amax+pot: PoT-projected per-channel init")],
        "transform_pc",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pc-adaptive", transform="adaptive"),
        [Expectation("adaptive: should be >= best single transform", should_be_better=True)],
        "transform_pc",
    ))

    # ── 8. Transform × per_block ──────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int4-pb32-had", w_granularity="per_block", a_granularity="per_block",
             w_block_size=32, a_block_size=32, transform="hadamard"),
        [Expectation("hadamard + pb32: architecture-dependent")],
        "transform_pb",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pb32-sq", w_granularity="per_block", a_granularity="per_block",
             w_block_size=32, a_block_size=32, transform="smoothquant"),
        [Expectation("smoothquant + pb32")],
        "transform_pb",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pb32-adaptive", w_granularity="per_block", a_granularity="per_block",
             w_block_size=32, a_block_size=32, transform="adaptive"),
        [Expectation("adaptive + pb32: should be >= best single transform", should_be_better=True)],
        "transform_pb",
    ))

    # ── 9. GPTQ ───────────────────────────────────────────────────────────
    specs.append(ConfigSpec(
        _cfg("int4-pc-gptq", gptq=True),
        [Expectation("GPTQ: should improve or match baseline", should_be_better=True)],
        "gptq",
    ))
    specs.append(ConfigSpec(
        _cfg("int4-pc-sq-gptq", transform="smoothquant", gptq=True),
        [Expectation("SmoothQuant + GPTQ: combined improvement", should_be_better=True)],
        "gptq",
    ))

    # ── 10. Format × per_tensor (sanity: should all be worse than pc) ────
    specs.append(ConfigSpec(
        _cfg("fp4-pt", w_format="fp4_e2m1", a_format="fp4_e2m1",
             w_granularity="per_tensor", a_granularity="per_tensor"),
        [Expectation("fp4 per_tensor < fp4 per_channel", should_be_worse=True)],
        "format_pt",
    ))
    specs.append(ConfigSpec(
        _cfg("nf4-pt", w_format="nf4", a_format="nf4",
             w_granularity="per_tensor", a_granularity="per_tensor"),
        [Expectation("nf4 per_tensor < nf4 per_channel", should_be_worse=True)],
        "format_pt",
    ))

    return specs


# ══════════════════════════════════════════════════════════════════════════════
# Validation runner
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CheckResult:
    config_name: str
    category: str
    fp32_ok: bool
    baseline_acc: float
    quant_acc: float
    delta: float
    checks: list[str]  # pass/fail messages


def validate(model, test_loader, calib_samples, specs: list[ConfigSpec], model_name: str) -> list[CheckResult]:
    """Run all configs and check expectations."""
    all_configs = [s.config for s in specs]
    name_to_spec = {s.config.name: s for s in specs}

    print(f"\n{'='*70}")
    print(f"  Running {len(all_configs)} configs on {model_name}")
    print(f"{'='*70}")

    report = run_study(model, all_configs, calib_samples, test_loader)

    baseline_acc = get_quant_accuracy(report, "int4-pc")
    if baseline_acc is None:
        print("  FATAL: baseline (int4-pc) not found in results!")
        return []

    print(f"  Baseline (int4-pc) quant accuracy: {baseline_acc:.4f}")

    results: list[CheckResult] = []
    for spec in specs:
        name = spec.config.name
        fp32 = get_fp32_accuracy(report, name)
        quant = get_quant_accuracy(report, name)

        if quant is None:
            results.append(CheckResult(name, spec.category, False, baseline_acc, 0.0, 0.0,
                                       [f"  FAIL: no quant_accuracy in result"]))
            continue

        delta = quant - baseline_acc
        checks = []

        for exp in spec.expectations:
            if exp.check_fp32_nonzero and fp32 is not None and fp32 == 0.0:
                checks.append(f"  FAIL: fp32_accuracy is 0.0 (regression detected!)")

            if exp.should_be_better and delta < -0.005:
                checks.append(f"  FAIL: expected better than baseline, got {delta:+.4f}")
            elif exp.should_be_better:
                checks.append(f"  PASS: better than baseline ({delta:+.4f})")

            if exp.should_be_worse and delta > 0.005:
                checks.append(f"  FAIL: expected worse than baseline, got {delta:+.4f}")
            elif exp.should_be_worse:
                checks.append(f"  PASS: worse than baseline ({delta:+.4f})")

            if abs(delta) > exp.max_drop:
                checks.append(f"  WARN: accuracy drop {delta:+.4f} exceeds max {exp.max_drop:+.4f}")

        if not checks:
            checks.append(f"  INFO: delta={delta:+.4f} (no directional expectation)")

        fp32_ok = fp32 is not None and fp32 > 0
        results.append(CheckResult(name, spec.category, fp32_ok, baseline_acc, quant, delta, checks))

    # Print summary table
    print(f"\n{'─'*90}")
    print(f"  {model_name} — Validation Results")
    print(f"{'─'*90}")
    print(f"{'Config':<28} {'Cat':<20} {'FP32':>8} {'Quant':>8} {'Δ base':>8}  Checks")
    print(f"{'─'*90}")

    anomalies = 0
    for r in results:
        fp32_str = "  OK   " if r.fp32_ok else "FAILED!"
        print(f"{r.config_name:<28} {r.category:<20} {fp32_str} {r.quant_acc:8.4f} {r.delta:+8.4f}  {r.checks[0] if r.checks else ''}")
        for chk in r.checks[1:]:
            print(f"{'':>70}{chk}")
        if any("FAIL" in c for c in r.checks):
            anomalies += 1

    print(f"{'─'*90}")
    print(f"  Anomalies: {anomalies}/{len(results)}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    torch.manual_seed(42)
    script_dir = os.path.dirname(__file__)
    weights_dir = os.path.join(script_dir, "weights")

    specs = build_all_configs()
    print(f"Total configs: {len(specs)}")
    for cat in sorted(set(s.category for s in specs)):
        names = [s.config.name for s in specs if s.category == cat]
        print(f"  {cat}: {', '.join(names)}")

    all_results: dict[str, list[CheckResult]] = {}

    # ── MNIST ─────────────────────────────────────────────────────────────
    mnist_weights = os.path.join(weights_dir, "mnist_mlp.pt")
    mnist_model, mnist_test_loader, mnist_calib = load_mnist(mnist_weights)
    all_results["MNIST"] = validate(mnist_model, mnist_test_loader, mnist_calib, specs, "MNIST MLP")

    # ── Transformer ───────────────────────────────────────────────────────
    transformer_weights = os.path.join(weights_dir, "transformer_agnews.pt")
    if os.path.exists(transformer_weights):
        tf_model, tf_test_loader, tf_calib = load_transformer(transformer_weights)
        all_results["Transformer"] = validate(tf_model, tf_test_loader, tf_calib, specs, "Transformer AG News")
    else:
        print(f"\n  SKIP Transformer: weights not found at {transformer_weights}")

    # ── Cross-model anomaly summary ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL ANOMALY SUMMARY")
    print(f"{'='*70}")
    for model_name, results in all_results.items():
        failures = [r for r in results if any("FAIL" in c for c in r.checks)]
        warns = [r for r in results if any("WARN" in c for c in r.checks)]
        fp32_fails = [r for r in results if not r.fp32_ok]
        if failures:
            print(f"\n  {model_name} FAILURES:")
            for r in failures:
                for c in r.checks:
                    if "FAIL" in c:
                        print(f"    {r.config_name}: {c.strip()}")
        if warns:
            print(f"\n  {model_name} WARNINGS:")
            for r in warns:
                for c in r.checks:
                    if "WARN" in c:
                        print(f"    {r.config_name}: {c.strip()}")
        if fp32_fails:
            print(f"\n  {model_name} FP32=0 REGRESSION:")
            for r in fp32_fails:
                print(f"    {r.config_name}")
        if not failures and not warns and not fp32_fails:
            print(f"\n  {model_name}: All checks passed!")


if __name__ == "__main__":
    main()
