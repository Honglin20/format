"""
Phase 5 ONNX export tests.
All tests verify graph structure (node types, attributes), not runtime correctness.
"""
import io
import pytest
import torch
import onnx
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.op_config import OpQuantConfig
from src.ops.linear import QuantizedLinear
from src.ops.conv import QuantizedConv2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export(model, x):
    """Export model to ONNX in-memory; return loaded onnx.ModelProto."""
    buf = io.BytesIO()
    torch.onnx.export(
        model, (x,), buf,
        opset_version=17,
        custom_opsets={"com.microxscaling": 1},
    )
    buf.seek(0)
    return onnx.load(buf)


def _node_ops(onnx_model):
    """Return list of (domain, op_type) for every node (excluding Constant)."""
    return [
        (n.domain or "onnx", n.op_type)
        for n in onnx_model.graph.node
        if n.op_type != "Constant"
    ]


def _has_op(onnx_model, op_type, domain="onnx"):
    return any(n.op_type == op_type and (n.domain or "onnx") == domain
               for n in onnx_model.graph.node)


def _standard_cfg(fmt_name, granularity=None):
    fmt = FormatBase.from_str(fmt_name)
    gran = granularity or GranularitySpec.per_tensor()
    s = QuantScheme(format=fmt, granularity=gran)
    return OpQuantConfig(input=(s,), weight=(s,), output=(s,))


def _mx_cfg(fmt_name="fp4_e2m1", block_size=32):
    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_block(block_size)
    s = QuantScheme(format=fmt, granularity=gran)
    return OpQuantConfig(input=(s,), weight=(s,), output=(s,))


# ---------------------------------------------------------------------------
# Task 1: _is_standard_format / _emit_quantize_node unit tests
# ---------------------------------------------------------------------------

def test_is_standard_format_int8_per_tensor():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("int8"),
                    granularity=GranularitySpec.per_tensor())
    assert _is_standard_format(s) is True


def test_is_standard_format_int4_per_channel():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("int4"),
                    granularity=GranularitySpec.per_channel(axis=0))
    assert _is_standard_format(s) is True


def test_is_standard_format_fp8_per_tensor():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("fp8_e4m3"),
                    granularity=GranularitySpec.per_tensor())
    assert _is_standard_format(s) is True


def test_is_standard_format_fp4_per_block_is_false():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("fp4_e2m1"),
                    granularity=GranularitySpec.per_block(32))
    assert _is_standard_format(s) is False


def test_is_standard_format_int8_per_block_is_false():
    """int8 with PER_BLOCK (MX style) is NOT standard — block quantization is custom."""
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("int8"),
                    granularity=GranularitySpec.per_block(32))
    assert _is_standard_format(s) is False


# ---------------------------------------------------------------------------
# Task 2: LinearFunction ONNX export
# ---------------------------------------------------------------------------

def test_linear_standard_format_uses_qdq():
    """int8 per_tensor → QuantizeLinear/DequantizeLinear nodes in graph."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "QuantizeLinear"), "Expected QDQ for int8"
    assert _has_op(onnx_model, "DequantizeLinear"), "Expected QDQ for int8"
    assert not _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
        "int8 should NOT use MxQuantize"


def test_linear_mx_format_uses_custom_op():
    """fp4_e2m1 per_block → com.microxscaling::MxQuantize nodes."""
    cfg = _mx_cfg("fp4_e2m1", block_size=32)
    model = QuantizedLinear(32, 64, cfg=cfg)
    x = torch.randn(2, 32)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
        "Expected MxQuantize for fp4 per_block"
    assert not _has_op(onnx_model, "QuantizeLinear"), \
        "fp4 per_block should NOT use QDQ"


def test_linear_export_checker_passes():
    """onnx.checker.check_model() passes for int8 linear export."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)  # raises if invalid


def test_linear_no_bias_exports_cleanly():
    """Linear without bias exports without error."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, bias=False, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)  # must not raise
    onnx.checker.check_model(onnx_model)


def test_linear_no_quantization_exports_cleanly():
    """Passthrough model (no cfg) exports as plain Gemm/MatMul."""
    model = QuantizedLinear(8, 16)  # no cfg → passthrough
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)
    assert not _has_op(onnx_model, "QuantizeLinear"), \
        "Passthrough model should have no QDQ"


# ---------------------------------------------------------------------------
# Task 3: ConvFunction ONNX export
# ---------------------------------------------------------------------------

def test_conv2d_standard_format_uses_qdq():
    """int8 per_tensor on Conv2d → QDQ nodes."""
    from src.ops.conv import QuantizedConv2d
    cfg = _standard_cfg("int8")
    model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 4, 8, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "QuantizeLinear"), "Expected QDQ for int8 conv"
    assert _has_op(onnx_model, "DequantizeLinear")


def test_conv2d_mx_format_uses_custom_op():
    """fp4_e2m1 per_block on Conv2d → MxQuantize nodes."""
    from src.ops.conv import QuantizedConv2d
    cfg = _mx_cfg("fp4_e2m1", block_size=32)
    model = QuantizedConv2d(32, 64, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 32, 8, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "MxQuantize", "com.microxscaling")
    assert not _has_op(onnx_model, "QuantizeLinear")


def test_conv2d_export_checker_passes():
    """onnx.checker passes for int8 Conv2d export."""
    from src.ops.conv import QuantizedConv2d
    cfg = _standard_cfg("int8")
    model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 4, 8, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)


# ---------------------------------------------------------------------------
# Task 4: export_quantized_model() end-to-end
# ---------------------------------------------------------------------------

def test_export_quantized_model_linear(tmp_path):
    """export_quantized_model() writes a valid .onnx file for a quantized linear model."""
    from src.onnx import export_quantized_model
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    out_path = str(tmp_path / "model.onnx")
    export_quantized_model(model, x, out_path)  # must not raise
    loaded = onnx.load(out_path)
    onnx.checker.check_model(loaded)
    assert _has_op(loaded, "QuantizeLinear")


def test_export_quantized_model_mixed(tmp_path):
    """Export a small mixed model: one Linear + one Conv2d, both quantized."""
    from src.onnx import export_quantized_model
    from src.ops.conv import QuantizedConv2d

    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cfg_int8 = _standard_cfg("int8")
            self.conv = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg_int8)
            self.linear = QuantizedLinear(8 * 8 * 8, 16, cfg=cfg_int8)

        def forward(self, x):
            x = self.conv(x)
            x = x.flatten(1)
            return self.linear(x)

    model = SmallModel()
    x = torch.randn(1, 4, 8, 8)
    out_path = str(tmp_path / "mixed.onnx")
    export_quantized_model(model, x, out_path)
    loaded = onnx.load(out_path)
    onnx.checker.check_model(loaded)
    assert _has_op(loaded, "QuantizeLinear")
    assert _has_op(loaded, "Conv")
