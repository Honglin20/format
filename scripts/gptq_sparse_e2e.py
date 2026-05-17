"""GPTQ + Sparse/Group-Sparse E2E Verification.

Loads a pretrained model and runs a Study comparing:
  - int4 per_channel (baseline, no GPTQ)
  - int4 per_channel + GPTQ
  - int4 per_channel + WLEM sparse (outlier_ratio=0.1)
  - int4 per_channel + WLEM sparse + GPTQ
  - int4 per_channel + group sparse (group_ratio=0.2, group_format=int8)
  - int4 per_channel + group sparse + GPTQ
  - int8 per_channel (reference)

Uses the MNIST MLP pretrained weights for fast iteration.
Run:  PYTHONPATH=. python scripts/gptq_sparse_e2e.py
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.session import Study, QuantConfig


# ---------------------------------------------------------------------------
# Model (same architecture as mnist_hadamard_study.py)
# ---------------------------------------------------------------------------


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


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
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(42)

    # -- Data --
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256)

    # -- Load pretrained model --
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    weights_path = os.path.join(weights_dir, "mnist_mlp.pt")

    if not os.path.exists(weights_path):
        print(f"Pretrained weights not found at {weights_path}")
        print("Run `PYTHONPATH=. python scripts/mnist_hadamard_study.py` first to train and save.")
        return

    ckpt = torch.load(weights_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    fp32_acc = ckpt.get("fp32_test_acc", None)
    print(f"Loaded pretrained model from {weights_path}")
    if fp32_acc is not None:
        print(f"  FP32 test accuracy (from checkpoint): {fp32_acc:.4f}")

    # Verify FP32 accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    live_fp32_acc = correct / total
    print(f"  FP32 test accuracy (live): {live_fp32_acc:.4f}")

    # -- Calibration data --
    train_ds = datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    calib_samples = []
    for x, _y in train_loader:
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    # -- Study configs --
    configs = [
        # 1. int4 per_channel (baseline, no GPTQ)
        QuantConfig(
            name="int4-pc",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            quantize_nonlinear=False,
        ),
        # 2. int4 per_channel + GPTQ
        QuantConfig(
            name="int4-pc-gptq",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            gptq=True, gptq_block_size=128,
            quantize_nonlinear=False,
        ),
        # 3. int4 per_channel + WLEM sparse (outlier_ratio=0.1, outlier_format=int8)
        QuantConfig(
            name="int4-pc-sparse",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            outlier_ratio=0.1, outlier_format="int8",
            quantize_nonlinear=False,
        ),
        # 4. int4 per_channel + WLEM sparse + GPTQ
        QuantConfig(
            name="int4-pc-sparse-gptq",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            outlier_ratio=0.1, outlier_format="int8",
            gptq=True, gptq_block_size=128,
            quantize_nonlinear=False,
        ),
        # 5. int4 per_channel + group sparse (group_ratio=0.2, group_format=int8)
        QuantConfig(
            name="int4-pc-gsparse",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            group_ratio=0.2, group_format="int8",
            quantize_nonlinear=False,
        ),
        # 6. int4 per_channel + group sparse + GPTQ
        QuantConfig(
            name="int4-pc-gsparse-gptq",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            w_axis=-1, a_axis=-1,
            group_ratio=0.2, group_format="int8",
            gptq=True, gptq_block_size=128,
            quantize_nonlinear=False,
        ),
        # 7. int8 per_channel (reference)
        QuantConfig(
            name="int8-pc",
            w_format="int8", a_format="int8",
            w_granularity="per_channel", a_granularity="per_channel",
            quantize_nonlinear=False,
        ),
    ]

    # -- Run Study --
    print(f"\n=== Running GPTQ + Sparse Study ({len(configs)} configs) ===")
    study = Study(configs, model=model)

    report = study.run(
        calib_samples,
        eval_data=test_loader,
        eval_fn=eval_fn,
        outputs="default",
    )

    # -- Results --
    print("\n" + "=" * 70)
    print("GPTQ + Sparse/Group-Sparse E2E Results")
    print("=" * 70)

    report.print_summary()

    print("\n===== Per-Config Details =====")
    for part in report.parts:
        for r in report._results[part]:
            print(f"\n--- {r.name} ---")
            print(r.summary())
            print(r.accuracy_table())

    # -- Analysis --
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    # Extract accuracy deltas
    acc_map = {}
    for part in report.parts:
        for r in report._results[part]:
            if r.quant_metrics and "accuracy" in r.quant_metrics:
                acc_map[r.name] = r.quant_metrics["accuracy"]

    if acc_map:
        print(f"\nFP32 baseline: {live_fp32_acc:.4f}")
        for name, acc in sorted(acc_map.items()):
            delta = acc - live_fp32_acc
            print(f"  {name:30s}  acc={acc:.4f}  Δ={delta:+.4f}")

        # GPTQ improvement analysis
        if "int4-pc" in acc_map and "int4-pc-gptq" in acc_map:
            gptq_gain = acc_map["int4-pc-gptq"] - acc_map["int4-pc"]
            print(f"\nGPTQ gain (int4-pc baseline): {gptq_gain:+.4f}")

        if "int4-pc-sparse" in acc_map and "int4-pc-sparse-gptq" in acc_map:
            gptq_sparse_gain = acc_map["int4-pc-sparse-gptq"] - acc_map["int4-pc-sparse"]
            print(f"GPTQ gain (sparse baseline):  {gptq_sparse_gain:+.4f}")

        if "int4-pc-gsparse" in acc_map and "int4-pc-gsparse-gptq" in acc_map:
            gptq_gsparse_gain = acc_map["int4-pc-gsparse-gptq"] - acc_map["int4-pc-gsparse"]
            print(f"GPTQ gain (group sparse baseline): {gptq_gsparse_gain:+.4f}")


if __name__ == "__main__":
    main()
