"""GPTQ diagnostic: weight-level MSE analysis.

Directly measure GPTQ's effect on weight quantization quality
for different sparse schemes, bypassing the full Session pipeline.
"""
from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.session import quantize_model
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.formats.base import FormatBase
from src.calibration.gptq_optimizer import GPTQOptimizer
from src.calibration.pipeline import CalibrationSession
from src.calibration.strategies import MaxScaleStrategy
from src.quantize.elemwise import quantize


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def main():
    torch.manual_seed(42)

    # -- Load model --
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    weights_path = os.path.join(weights_dir, "mnist_mlp.pt")
    ckpt = torch.load(weights_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained model, FP32 acc={ckpt['fp32_test_acc']:.4f}")

    # -- Calibration data --
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    calib_samples = []
    for x, _y in train_loader:
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    # -- Schemes to test --
    int4_fmt = FormatBase.from_str("int4")
    int8_fmt = FormatBase.from_str("int8")

    schemes = {
        "int4-pc": QuantScheme(
            format=int4_fmt,
            granularity=GranularitySpec.per_channel(axis=0),
        ),
        "int4-pc-sparse": QuantScheme(
            format=int4_fmt,
            granularity=GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0, outlier_ratio=0.1),
            outlier_format=int8_fmt,
        ),
        "int4-pc-gsparse": QuantScheme(
            format=int4_fmt,
            granularity=GranularitySpec.per_channel(axis=0),
            group_format=int8_fmt,
            group_ratio=0.2,
        ),
    }

    print("\n" + "=" * 80)
    print("Weight-level MSE: naive quantize vs GPTQ quantize")
    print("=" * 80)

    for scheme_name, scheme in schemes.items():
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg, quantize_nonlinear=False)

        # Get first linear layer
        linear = None
        linear_name = None
        for name, mod in qmodel.named_modules():
            if isinstance(mod, nn.Linear) and hasattr(mod, "cfg"):
                linear = mod
                linear_name = name
                break

        if linear is None:
            print(f"  {scheme_name}: No quantized linear found")
            continue

        W_fp32 = linear.weight.data.clone()

        # Naive quantize MSE
        with torch.no_grad():
            W_naive = quantize(W_fp32, scheme)
        mse_naive = (W_fp32 - W_naive).pow(2).mean().item()

        # GPTQ quantize
        gptq = GPTQOptimizer(block_size=128, damp_percent=0.01)
        results = gptq.optimize(qmodel, calib_samples)

        W_gptq = linear.weight.data.clone()
        mse_gptq = (W_fp32 - W_gptq).pow(2).mean().item()

        # Check if GPTQ actually changed the weights
        weight_changed = not torch.equal(W_fp32, W_gptq)

        # After GPTQ, the forward pass re-quantizes via quantize(w, scheme).
        # Let's verify that re-quantization of GPTQ weights is idempotent.
        with torch.no_grad():
            W_gptq_requant = quantize(W_gptq, scheme)
        mse_gptq_requant = (W_fp32 - W_gptq_requant).pow(2).mean().item()
        idempotent = torch.allclose(W_gptq, W_gptq_requant, atol=1e-6)

        print(f"\n--- {scheme_name} ({linear_name}) ---")
        print(f"  W shape: {W_fp32.shape}")
        print(f"  Naive MSE:      {mse_naive:.6f}")
        print(f"  GPTQ MSE:       {mse_gptq:.6f}")
        print(f"  GPTQ re-quant:  {mse_gptq_requant:.6f}")
        print(f"  GPTQ changed W: {weight_changed}")
        print(f"  Idempotent:     {idempotent}")

        if results and linear_name in results:
            meta = results[linear_name]
            print(f"  GPTQ meta: mse_before={meta['mse_before']:.6f}  mse_after={meta['mse_after']:.6f}")

    # -- Now test with full run_quantization to see end-to-end --
    print("\n" + "=" * 80)
    print("Full run_quantization pipeline: accuracy comparison")
    print("=" * 80)

    from src.session._config import QuantConfig
    from src.session._session import run_quantization

    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256)

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

    # Run GPTQ-only (no sparse) to get baseline behavior
    for gptq_flag in [False, True]:
        for sparse_name, outlier_ratio, outlier_fmt, group_ratio, group_fmt in [
            ("none", 0.0, None, 0.0, None),
            ("sparse", 0.1, "int8", 0.0, None),
            ("gsparse", 0.0, None, 0.2, "int8"),
        ]:
            name = f"int4-pc-{sparse_name}" + ("-gptq" if gptq_flag else "")
            cfg = QuantConfig(
                name=name,
                w_format="int4", a_format="int4",
                w_granularity="per_channel", a_granularity="per_channel",
                w_axis=-1, a_axis=-1,
                outlier_ratio=outlier_ratio,
                outlier_format=outlier_fmt,
                group_ratio=group_ratio,
                group_format=group_fmt,
                gptq=gptq_flag,
                gptq_block_size=128,
                quantize_nonlinear=False,
            )
            qmodel, _, result = run_quantization(
                copy.deepcopy(model), cfg, calib_samples,
                eval_data=test_loader, eval_fn=eval_fn,
                outputs="default",
            )
            acc = result.quant_metrics.get("accuracy", float("nan")) if result.quant_metrics else float("nan")
            qsnr = sum(result.qsnr_per_layer.values()) / len(result.qsnr_per_layer) if result.qsnr_per_layer else float("nan")
            print(f"  {name:30s}  acc={acc:.4f}  avg_qsnr={qsnr:.1f} dB")


if __name__ == "__main__":
    main()
