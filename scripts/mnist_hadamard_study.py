"""
MNIST E2E Experiment: INT4 W4A4 with Hadamard Transform via Study API.

Trains an MLP on MNIST, then compares quantization configs:
  - int4 per_channel W4A4 (no transform)
  - int4 per_channel W4A4 + Hadamard
  - int4 per_block(32) W4A4 (MX-style, no transform)
  - int4 per_block(32) W4A4 + Hadamard
  - int8 per_channel W8A8 (reference)

Only the Study API is used — no low-level Session/QuantSession calls.
Run:  PYTHONPATH=. python scripts/mnist_hadamard_study.py
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.session import Study, QuantConfig


# ---------------------------------------------------------------------------
# Model
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

    # Final eval on training set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in train_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"  Train accuracy: {correct / total:.4f}")


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
        # Calibration / analysis — just run forward passes
        with torch.no_grad():
            for batch in data:
                model(batch)
        return {}

    # Evaluation — compute accuracy
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
    train_ds = datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    # -- Train --
    print("=== Training MLP on MNIST ===")
    model = build_model()
    train_model(model, train_loader, epochs=5)

    # Baseline FP32 accuracy on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    fp32_test_acc = correct / total
    print(f"  FP32 test accuracy: {fp32_test_acc:.4f}")

    # -- Save model weights --
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "mnist_mlp.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "fp32_test_acc": fp32_test_acc},
        weights_path,
    )
    print(f"  Weights saved to {weights_path}")

    # -- Calibration data (subset of training batches) --
    calib_samples = []
    for x, _y in train_loader:
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    # -- Study configs --
    configs = [
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

    # -- Run Study --
    print("\n=== Running Quantization Study ===")
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
