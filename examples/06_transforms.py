"""
06 — Quantization Transforms: Hadamard + SmoothQuant.

Transforms apply reversible pre/post-processing to reduce quantization error.
Run:  PYTHONPATH=. python examples/06_transforms.py
"""
import torch

from _model import ToyMLP
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.mapping.quantize_model import quantize_model
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform


def measure_mse(qmodel, fp32_model, x):
    """Return MSE between fp32 and quantized outputs."""
    with torch.no_grad():
        fp32_out = fp32_model(x)
        q_out = qmodel(x)
    return (fp32_out - q_out).pow(2).mean().item()


def main():
    print("=" * 55)
    print("Quantization Transforms")
    print("=" * 55)

    x = torch.randn(8, 128)

    fp32_model = ToyMLP()
    fp32_model.eval()
    sd = fp32_model.state_dict()

    # ── 1. Hadamard Transform ─────────────────────────────────────
    print("\n1. Hadamard Rotation (int4 vs int4+Hadamard)")

    # int4 without transform (baseline)
    int4_fmt = FormatBase.from_str("int4")
    int4_scheme = QuantScheme(format=int4_fmt, granularity=GranularitySpec.per_tensor())
    int4_cfg = OpQuantConfig(input=(int4_scheme,), weight=(int4_scheme,), output=(int4_scheme,))

    q_int4 = quantize_model(ToyMLP(), cfg=int4_cfg)
    q_int4.load_state_dict(sd, strict=False)
    q_int4.eval()
    mse_int4 = measure_mse(q_int4, fp32_model, x)

    # int4 with Hadamard rotation
    hadamard_scheme = QuantScheme(
        format=int4_fmt,
        granularity=GranularitySpec.per_channel(axis=-1),
        transform=HadamardTransform(),
    )
    hadamard_cfg = OpQuantConfig(input=(hadamard_scheme,), weight=(hadamard_scheme,), output=(hadamard_scheme,))

    q_had = quantize_model(ToyMLP(), cfg=hadamard_cfg)
    q_had.load_state_dict(sd, strict=False)
    q_had.eval()
    mse_had = measure_mse(q_had, fp32_model, x)

    reduction = (1 - mse_had / mse_int4) * 100 if mse_int4 > 0 else 0
    print(f"   MSE(int4, no transform):      {mse_int4:.6f}")
    print(f"   MSE(int4, Hadamard):           {mse_had:.6f}")
    print(f"   Reduction:                     {reduction:.1f}%")

    # ── 2. SmoothQuant Transform (per-layer) ──────────────────────
    print("\n2. SmoothQuant (int8 + per-layer SmoothQuant scaling)")

    # int8 without transform
    int8_fmt = FormatBase.from_str("int8")
    int8_scheme = QuantScheme(format=int8_fmt, granularity=GranularitySpec.per_tensor())
    int8_cfg = OpQuantConfig(input=(int8_scheme,), weight=(int8_scheme,), output=(int8_scheme,))

    q_int8 = quantize_model(ToyMLP(), cfg=int8_cfg)
    q_int8.load_state_dict(sd, strict=False)
    q_int8.eval()
    mse_int8 = measure_mse(q_int8, fp32_model, x)

    # Calibrate SmoothQuant from fc1's input activations + fc1 weight
    calib_x = torch.randn(16, 128)
    with torch.no_grad():
        calib_act = fp32_model.ln(calib_x)  # activation before fc1, shape (16, 128)

    sq_transform = SmoothQuantTransform.from_calibration(
        X_act=calib_act,
        W=fp32_model.fc1.weight.data,  # (512, 128)
        alpha=0.5,
    )
    # Scale shape is (128,) — fc1 input feature dimension

    sq_scheme = QuantScheme(
        format=int8_fmt,
        granularity=GranularitySpec.per_channel(axis=-1),
        transform=sq_transform,
    )
    sq_input_cfg = OpQuantConfig(input=(sq_scheme,))  # apply only to fc1 input

    # Per-layer config: SmoothQuant on fc1 input only, rest is plain int8
    per_layer_cfg = {
        "fc1": OpQuantConfig(
            input=(sq_scheme,), weight=(int8_scheme,), output=(int8_scheme,),
        ),
        "fc2": int8_cfg,
        "ln":  OpQuantConfig(input=(int8_scheme,), output=(int8_scheme,)),
    }

    q_sq = quantize_model(ToyMLP(), cfg=per_layer_cfg)
    q_sq.load_state_dict(sd, strict=False)
    q_sq.eval()
    mse_sq = measure_mse(q_sq, fp32_model, x)

    reduction2 = (1 - mse_sq / mse_int8) * 100 if mse_int8 > 0 else 0
    print(f"   MSE(int8, no transform):      {mse_int8:.6f}")
    print(f"   MSE(int8, SmoothQuant fc1):   {mse_sq:.6f}")
    print(f"   Reduction:                     {reduction2:.1f}%")

    # ── 3. Verify transforms are reversible ───────────────────────
    print("\n3. Transform reversibility check")

    # Hadamard is self-inverse
    had = HadamardTransform()
    t = torch.randn(4, 128)
    t_had = had.forward(t)
    t_back = had.inverse(t_had)
    had_error = (t - t_back).abs().max().item()
    print(f"   Hadamard round-trip max error: {had_error:.2e}")

    # SmoothQuant scale
    t_sq = sq_transform.forward(t)
    t_sq_back = sq_transform.inverse(t_sq)
    sq_error = (t - t_sq_back).abs().max().item()
    print(f"   SmoothQuant round-trip error:  {sq_error:.2e}")

    # ── 4. Comparison summary ─────────────────────────────────────
    print("\n4. Comparison summary")
    print(f"   {'Config':<28} {'MSE':>12} {'vs baseline':>15}")
    print(f"   {'-'*28} {'-'*12} {'-'*15}")
    print(f"   {'int4 (no transform)':<28} {mse_int4:>12.6f} {'—':>15}")
    print(f"   {'int4 + Hadamard':<28} {mse_had:>12.6f} {f'-{reduction:.1f}%':>15}")
    print(f"   {'int8 (no transform)':<28} {mse_int8:>12.6f} {'—':>15}")
    print(f"   {'int8 + SmoothQuant fc1':<28} {mse_sq:>12.6f} {f'-{reduction2:.1f}%':>15}")

    print("\n" + "=" * 55)
    print("Transform examples complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
