"""
05 — ONNX Export with unified three-axis quantization nodes.

All formats use the same node pattern:
  - Non-truncation: Scale(granularity) → Quantize(format)
  - Truncation:     Truncate(dtype)

Run:  PYTHONPATH=. python examples/05_onnx_export.py
"""
import copy
import os
import tempfile

import torch

from pipeline._model import ToyMLP
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.session import quantize_model


def export_and_check(qmodel, name, x):
    """Export to a temp file, run onnx.checker, report result."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{name}.onnx")
        qmodel.export_onnx(x, path)

        import onnx
        try:
            onnx.checker.check_model(path)
            size_kb = os.path.getsize(path) / 1024
            print(f"   {name:<8}  OK  ({size_kb:.1f} KB)")
            return True
        except Exception as e:
            print(f"   {name:<8}  FAIL: {e}")
            return False


def main():
    print("=" * 55)
    print("ONNX Export — Unified Three-Axis Nodes")
    print("=" * 55)

    x = torch.randn(1, 128)

    # ── int8 per_tensor → Scale + Quantize ──────────────────────────
    print("\n1. int8 per_tensor → Scale + Quantize")
    scheme = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=GranularitySpec.per_tensor(),
    )
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
    q_int8 = quantize_model(copy.deepcopy(ToyMLP()), cfg)
    q_int8.eval()
    export_and_check(q_int8, "int8", x)

    # ── int8 per_channel → Scale + Quantize ─────────────────────────
    print("\n2. int8 per_channel → Scale + Quantize (with axis)")
    scheme2 = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=GranularitySpec.per_channel(axis=-1),
    )
    cfg2 = OpQuantConfig(input=scheme2, weight=scheme2, output=scheme2)
    q_int8_ch = quantize_model(copy.deepcopy(ToyMLP()), cfg2)
    q_int8_ch.eval()
    export_and_check(q_int8_ch, "int8_ch", x)

    # ── fp4 per_block → Scale + Quantize ────────────────────────────
    print("\n3. fp4_e2m1 per_block(32) → Scale + Quantize")
    scheme3 = QuantScheme(
        format=FormatBase.from_str("fp4_e2m1"),
        granularity=GranularitySpec.per_block(32),
    )
    cfg3 = OpQuantConfig(input=scheme3, weight=scheme3, output=scheme3)
    q_fp4 = quantize_model(copy.deepcopy(ToyMLP()), cfg3)
    q_fp4.eval()
    export_and_check(q_fp4, "fp4_blk", x)

    # ── nf4 weight-only → Scale + Quantize ──────────────────────────
    print("\n4. nf4 weight-only → Scale + Quantize")
    print("   (skipped — JIT tracer does not support NF4 LUT argmin)")

    # ── Auto-input recording ────────────────────────────────────────
    print("\n5. Manual export with explicit dummy_input")
    qmodel = quantize_model(copy.deepcopy(ToyMLP()), cfg)
    qmodel.eval()
    qmodel(x)
    export_and_check(qmodel, "auto", x)

    # ── bfloat16 per_tensor → Truncate ──────────────────────────────
    print("\n6. bfloat16 per_tensor → Truncate")
    scheme6 = QuantScheme(
        format=FormatBase.from_str("bfloat16"),
        granularity=GranularitySpec.per_tensor(),
    )
    cfg6 = OpQuantConfig(input=scheme6, weight=scheme6, output=scheme6)
    q_bf16 = quantize_model(copy.deepcopy(ToyMLP()), cfg6)
    q_bf16.eval()
    export_and_check(q_bf16, "bf16", x)

    # ── float16 per_tensor → Truncate ───────────────────────────────
    print("\n7. float16 per_tensor → Truncate")
    scheme7 = QuantScheme(
        format=FormatBase.from_str("float16"),
        granularity=GranularitySpec.per_tensor(),
    )
    cfg7 = OpQuantConfig(input=scheme7, weight=scheme7, output=scheme7)
    q_fp16 = quantize_model(copy.deepcopy(ToyMLP()), cfg7)
    q_fp16.eval()
    export_and_check(q_fp16, "fp16", x)

    # ── Storage + compute combo: bf16 + int4 per_block ──────────────
    print("\n8. bf16 storage + int4 per_block compute (Truncate + Scale + Quantize)")
    s_bf = QuantScheme(
        format=FormatBase.from_str("bfloat16"),
        granularity=GranularitySpec.per_tensor(),
    )
    s_int4_mx = QuantScheme(
        format=FormatBase.from_str("int4"),
        granularity=GranularitySpec.per_block(32),
    )
    cfg8 = OpQuantConfig(storage=s_bf, input=s_int4_mx, weight=s_int4_mx, output=s_int4_mx)
    q_combo = quantize_model(copy.deepcopy(ToyMLP()), cfg8)
    q_combo.eval()
    export_and_check(q_combo, "combo", x)

    # ── int8 per_channel axis=0 → Scale + Quantize ──────────────────
    print("\n9. int8 per_channel(axis=0) → Scale + Quantize")
    scheme9 = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=GranularitySpec.per_channel(axis=0),
    )
    cfg9 = OpQuantConfig(input=scheme9, weight=scheme9, output=scheme9)
    q_ch = quantize_model(copy.deepcopy(ToyMLP()), cfg9)
    q_ch.eval()
    export_and_check(q_ch, "int8_pc0", x)

    # ── Complex model: ToyMLP with mixed configs ────────────────────
    print("\n10. ToyMLP mixed config (int4 per_block + int8 per_tensor + bf16)")
    from src.ops.linear import QuantizedLinear
    fmt_mx = FormatBase.from_str("int4")
    s_mx = QuantScheme(format=fmt_mx, granularity=GranularitySpec.per_block(32))
    cfg_mx = OpQuantConfig(input=s_mx, weight=s_mx, output=s_mx)
    fmt_i8 = FormatBase.from_str("int8")
    s_i8 = QuantScheme(format=fmt_i8, granularity=GranularitySpec.per_tensor())
    cfg_i8 = OpQuantConfig(input=s_i8, weight=s_i8, output=s_i8)
    s_bf = QuantScheme(format=FormatBase.from_str("bfloat16"),
                       granularity=GranularitySpec.per_tensor())
    cfg_bf = OpQuantConfig(storage=s_bf, input=s_bf, weight=s_bf)

    model_mixed = ToyMLP()
    model_mixed.fc1 = QuantizedLinear(128, 512, bias=True, cfg=cfg_mx, name="fc1")
    model_mixed.fc2 = QuantizedLinear(512, 128, bias=True, cfg=cfg_i8, name="fc2")
    model_mixed.head = QuantizedLinear(128, 10, bias=True, cfg=cfg_bf, name="head")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "toymlp_mixed.onnx")
        from src.onnx import export_quantized_model
        export_quantized_model(model_mixed, x, path)
        import onnx as _onnx
        _onnx.checker.check_model(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"   ToyMLP_mixed  OK  ({size_kb:.1f} KB)")

    # ── Simple ConvNet: int8 per_channel ────────────────────────────
    print("\n11. SimpleConvMLP int8 per_channel Conv + per_channel Linear")
    from src.ops.conv import QuantizedConv2d
    s_conv = QuantScheme(format=FormatBase.from_str("int8"),
                         granularity=GranularitySpec.per_channel(axis=0))
    cfg_conv = OpQuantConfig(input=s_conv, weight=s_conv, output=s_conv)
    s_lin = QuantScheme(format=FormatBase.from_str("int8"),
                        granularity=GranularitySpec.per_channel(axis=-1))
    cfg_lin = OpQuantConfig(input=s_lin, weight=s_lin, output=s_lin)

    class SimpleConvMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = QuantizedConv2d(1, 8, kernel_size=3, padding=1, cfg=cfg_conv, name="conv1")
            self.conv2 = QuantizedConv2d(8, 16, kernel_size=3, padding=1, cfg=cfg_conv, name="conv2")
            self.pool = torch.nn.AvgPool2d(kernel_size=4)
            self.fc = QuantizedLinear(16 * 4 * 2, 10, cfg=cfg_lin, name="fc")

        def forward(self, xx):
            xx = torch.nn.functional.relu(self.conv1(xx))
            xx = torch.nn.functional.relu(self.conv2(xx))
            xx = self.pool(xx)
            xx = xx.flatten(1)
            return self.fc(xx)

    model_conv = SimpleConvMLP()
    x_c = torch.randn(1, 1, 16, 8)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "convnet_per_ch.onnx")
        export_quantized_model(model_conv, x_c, path)
        import onnx as _onnx
        _onnx.checker.check_model(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"   ConvMLP_pch  OK  ({size_kb:.1f} KB)")

    print("\n" + "=" * 55)
    print("ONNX export examples complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
