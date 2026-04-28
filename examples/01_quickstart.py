"""
01 — Basic Quantization: configure formats and quantize a model.

Shows the four main configuration styles:
  A. Uniform config (one scheme for all ops)
  B. Per-layer dict config (glob matching)
  C. MX block-wise (per_block granularity)
  D. NF4 weight-only (lookup table)

Run:  python examples/01_quickstart.py
"""
import torch

from _model import ToyMLP
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.mapping.quantize_model import quantize_model


def demo_uniform_int8():
    """A — Single int8 config for all ops."""
    print("─" * 50)
    print("A — Uniform int8 (per_tensor)")
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

    model = ToyMLP()
    qmodel = quantize_model(model, cfg=cfg)
    qmodel.eval()

    x = torch.randn(2, 128)
    with torch.no_grad():
        y = qmodel(x)
    print(f"  input: {x.shape} → output: {y.shape}")
    print(f"  output sample: {y[0, :4].tolist()}")
    print(f"  OK — uniform int8 quantized model runs")


def demo_per_layer_dict():
    """B — Per-layer config with glob matching."""
    print("\n" + "─" * 50)
    print("B — Per-layer dict (glob matching)")

    int8_scheme = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=GranularitySpec.per_tensor(),
    )
    fp8_scheme = QuantScheme(
        format=FormatBase.from_str("fp8_e4m3"),
        granularity=GranularitySpec.per_tensor(),
    )
    int8_cfg = OpQuantConfig(input=int8_scheme, weight=int8_scheme, output=int8_scheme)
    fp8_cfg = OpQuantConfig(input=fp8_scheme, weight=fp8_scheme, output=fp8_scheme)

    cfg_dict = {
        "fc1": fp8_cfg,      # first layer gets fp8
        "fc2": int8_cfg,     # second layer gets int8
        "ln":  OpQuantConfig(input=int8_scheme, output=int8_scheme),
    }

    model = ToyMLP()
    qmodel = quantize_model(model, cfg=cfg_dict)
    qmodel.eval()

    x = torch.randn(2, 128)
    with torch.no_grad():
        y = qmodel(x)
    print(f"  fc1 → QuantizedLinear (fp8), fc2 → QuantizedLinear (int8)")
    print(f"  output sample: {y[0, :4].tolist()}")
    print(f"  OK — per-layer dict config works")


def demo_mx_blockwise():
    """C — MX-style block-wise quantization (per_block 32)."""
    print("\n" + "─" * 50)
    print("C — MX block-wise (fp4_e2m1, block_size=32)")

    fmt = FormatBase.from_str("fp4_e2m1")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_block(32))
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

    model = ToyMLP()
    qmodel = quantize_model(model, cfg=cfg)
    qmodel.eval()

    x = torch.randn(2, 128)
    with torch.no_grad():
        y = qmodel(x)
    print(f"  block_size=32, fp4_e2m1 format")
    print(f"  output sample: {y[0, :4].tolist()}")
    print(f"  OK — MX block-wise quantized model runs")


def demo_nf4_weight_only():
    """D — NF4 weight-only quantization (QLoRA LUT)."""
    print("\n" + "─" * 50)
    print("D — NF4 weight-only (QLoRA lookup table)")

    nf4_fmt = FormatBase.from_str("nf4")
    nf4_scheme = QuantScheme(nf4_fmt, GranularitySpec.per_channel(axis=0))
    cfg = OpQuantConfig(weight=nf4_scheme)  # only weight is quantized

    model = ToyMLP()
    qmodel = quantize_model(model, cfg=cfg)
    qmodel.eval()

    x = torch.randn(2, 128)
    with torch.no_grad():
        y = qmodel(x)
    print(f"  NF4 16-level normal-optimized LUT, per_channel on weight only")
    print(f"  output sample: {y[0, :4].tolist()}")
    print(f"  OK — NF4 weight-only quantized model runs")


def demo_mode_comparison():
    """Side-by-side: fp32 vs int8 vs nf4 output comparison."""
    print("\n" + "─" * 50)
    print("E — fp32 vs int8 vs nf4 output comparison")

    int8_scheme = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=GranularitySpec.per_tensor(),
    )
    int8_cfg = OpQuantConfig(input=int8_scheme, weight=int8_scheme, output=int8_scheme)

    nf4_fmt = FormatBase.from_str("nf4")
    nf4_scheme = QuantScheme(nf4_fmt, GranularitySpec.per_channel(axis=0))
    nf4_cfg = OpQuantConfig(weight=nf4_scheme)

    x = torch.randn(4, 128)

    # fp32 baseline
    fp32_model = ToyMLP()
    fp32_model.eval()
    with torch.no_grad():
        fp32_out = fp32_model(x)

    # int8
    q_int8 = quantize_model(ToyMLP(), cfg=int8_cfg)
    q_int8.load_state_dict(fp32_model.state_dict(), strict=False)
    q_int8.eval()
    with torch.no_grad():
        int8_out = q_int8(x)

    # nf4 weight-only
    q_nf4 = quantize_model(ToyMLP(), cfg=nf4_cfg)
    q_nf4.load_state_dict(fp32_model.state_dict(), strict=False)
    q_nf4.eval()
    with torch.no_grad():
        nf4_out = q_nf4(x)

    # Compare
    mse_int8 = (fp32_out - int8_out).pow(2).mean().item()
    mse_nf4 = (fp32_out - nf4_out).pow(2).mean().item()
    print(f"  fp32 output sample:  {fp32_out[0, :4].tolist()}")
    print(f"  int8 output sample:  {int8_out[0, :4].tolist()}")
    print(f"  nf4  output sample:  {nf4_out[0, :4].tolist()}")
    print(f"  MSE(fp32, int8): {mse_int8:.6f}")
    print(f"  MSE(fp32, nf4):  {mse_nf4:.6f}")
    print(f"  OK — all three produce valid outputs")


if __name__ == "__main__":
    demo_uniform_int8()
    demo_per_layer_dict()
    demo_mx_blockwise()
    demo_nf4_weight_only()
    demo_mode_comparison()
    print("\n" + "=" * 50)
    print("All 01_quickstart demos passed.")
