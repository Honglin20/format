"""MNIST E2E regression (eval-only) — loads pre-trained weights, skips training.

Run: PYTHONPATH=. python scripts/mnist_hadamard_eval.py
"""
from __future__ import annotations

import copy, os, sys, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.session import Study, QuantConfig


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 10),
    )


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
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


def main():
    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    # Load pre-trained
    model = build_model()
    weights_path = os.path.join(os.path.dirname(__file__), "weights", "mnist_mlp.pt")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"FP32 test accuracy: {correct / total:.4f}")

    calib_samples = []
    for x, _y in train_loader:
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    configs = [
        QuantConfig(name="int4-pc", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    w_axis=-1, a_axis=-1, quantize_nonlinear=False),
        QuantConfig(name="int4-pc-had", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    w_axis=-1, a_axis=-1, transform="hadamard", quantize_nonlinear=False),
        QuantConfig(name="int4-pc-sq", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    w_axis=-1, a_axis=-1, transform="smoothquant", quantize_nonlinear=False),
        QuantConfig(name="int4-pb32", w_format="int4", a_format="int4",
                    w_granularity="per_block", a_granularity="per_block",
                    w_block_size=32, a_block_size=32, quantize_nonlinear=False),
        QuantConfig(name="int4-pb32-had", w_format="int4", a_format="int4",
                    w_granularity="per_block", a_granularity="per_block",
                    w_block_size=32, a_block_size=32, transform="hadamard", quantize_nonlinear=False),
        QuantConfig(name="int4-pb32-sq", w_format="int4", a_format="int4",
                    w_granularity="per_block", a_granularity="per_block",
                    w_block_size=32, a_block_size=32, transform="smoothquant", quantize_nonlinear=False),
        QuantConfig(name="int8-pc", w_format="int8", a_format="int8",
                    w_granularity="per_channel", a_granularity="per_channel", quantize_nonlinear=False),
    ]

    print("\n=== Running MNIST E2E Regression ===")
    model_copy = copy.deepcopy(model)
    study = Study(configs, model=model_copy)
    report = study.run(calib_samples, eval_data=test_loader, eval_fn=eval_fn, outputs="default")
    report.print_summary()

    # Validate against baseline thresholds
    errors = []
    for part in report.parts:
        for r in report._results[part]:
            acc = r.quant_metrics.get("accuracy", 0)
            fp32 = r.fp32_metrics.get("accuracy", 0)
            name = r.name
            if fp32 == 0:
                errors.append(f"{name}: FP32 accuracy is 0 — REGRESSION")
            delta = acc - fp32
            if name == "int8-pc" and abs(delta) > 0.02:
                errors.append(f"{name}: |Δ|={abs(delta):.4f} > 0.02 — REGRESSION")
            if name.startswith("int4-pb32") and abs(delta) > 0.05:
                errors.append(f"{name}: |Δ|={abs(delta):.4f} > 0.05 — REGRESSION")

    if errors:
        print("\n*** REGRESSION DETECTED ***")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nMNIST E2E regression PASSED")


if __name__ == "__main__":
    main()
