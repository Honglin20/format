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
    return OpQuantConfig(input=s, weight=s, output=s)


def _mx_cfg(fmt_name="fp4_e2m1", block_size=32):
    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_block(block_size)
    s = QuantScheme(format=fmt, granularity=gran)
    return OpQuantConfig(input=s, weight=s, output=s)


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


# ---------------------------------------------------------------------------
# Multi-input type tests
# ---------------------------------------------------------------------------


class TestMultiInput:
    """ONNX export with list / tuple / dict dummy_input."""

    @staticmethod
    def _make_two_input_model():
        from src.ops.linear import QuantizedLinear
        cfg = _standard_cfg("int8")

        class TwoInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = QuantizedLinear(8, 16, cfg=cfg)
                self.linear2 = QuantizedLinear(8, 16, cfg=cfg)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)

        return TwoInputModel()

    def test_tuple_input(self):
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        buf = io.BytesIO()
        torch.onnx.export(model, (x, y), buf, opset_version=17,
                          custom_opsets={"com.microxscaling": 1})
        buf.seek(0)
        onnx_model = onnx.load(buf)
        onnx.checker.check_model(onnx_model)
        assert len(onnx_model.graph.input) >= 2

    def test_dict_input(self):
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        buf = io.BytesIO()
        torch.onnx.export(model, {"x": x, "y": y}, buf, opset_version=17,
                          custom_opsets={"com.microxscaling": 1})
        buf.seek(0)
        onnx_model = onnx.load(buf)
        onnx.checker.check_model(onnx_model)

    def test_export_quantized_model_with_tuple(self, tmp_path):
        from src.onnx import export_quantized_model
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        out = str(tmp_path / "two_input.onnx")
        export_quantized_model(model, (x, y), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_export_quantized_model_with_list(self, tmp_path):
        from src.onnx import export_quantized_model
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        out = str(tmp_path / "list_input.onnx")
        export_quantized_model(model, [x, y], out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_export_quantized_model_with_dict(self, tmp_path):
        from src.onnx import export_quantized_model
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        out = str(tmp_path / "dict_input.onnx")
        export_quantized_model(model, {"x": x, "y": y}, out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert len(loaded.graph.input) >= 2

    def test_multi_arg_session_records_and_exports(self, tmp_path):
        from src.session._quant import _QuantSession
        cfg = _standard_cfg("int8")
        model = self._make_two_input_model()
        session = _QuantSession(model, cfg)
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        session(x, y)
        out = str(tmp_path / "multi_arg.onnx")
        session.export_onnx(out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)


# ---------------------------------------------------------------------------
# Task: _last_input auto-recording tests
# ---------------------------------------------------------------------------


class TestAutoInput:
    """_last_input auto-recording for ONNX export without dummy_input."""

    def test_export_after_forward_uses_recorded_input(self, tmp_path):
        """Single-tensor forward → export without dummy_input works."""
        from src.session._quant import _QuantSession
        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        session = _QuantSession(model, cfg)
        x = torch.randn(2, 8)
        session(x)  # records _last_input
        out = str(tmp_path / "auto.onnx")
        session.export_onnx(out)  # no dummy_input
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_export_without_input_raises(self):
        """No forward + no dummy_input → ValueError."""
        from src.session._quant import _QuantSession
        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        session = _QuantSession(model, cfg)
        with pytest.raises(ValueError, match="No dummy_input"):
            session.export_onnx("nowhere.onnx")


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


# ---------------------------------------------------------------------------
# Task: NF4 (lookup-table format) ONNX export
# ---------------------------------------------------------------------------


class TestNF4Export:
    """NF4 lookup-table format ONNX export."""

    def test_nf4_per_tensor_emits_nf4_quantize(self):
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "NF4Quantize", "com.microxscaling"), \
            "NF4 should emit NF4Quantize custom op"
        assert not _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
            "NF4 should NOT use MxQuantize"
        assert not _has_op(onnx_model, "QuantizeLinear"), \
            "NF4 should NOT use QDQ"

    def test_nf4_per_block_emits_nf4_quantize(self):
        """NF4 + per_block → still NF4Quantize (NF4 is block-agnostic in ONNX)."""
        fmt = FormatBase.from_str("nf4")
        gran = GranularitySpec.per_block(32)
        s = QuantScheme(format=fmt, granularity=gran)
        cfg = OpQuantConfig(input=s, weight=s)
        model = QuantizedLinear(32, 64, cfg=cfg)
        x = torch.randn(2, 32)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "NF4Quantize", "com.microxscaling")

    def test_nf4_export_checker_passes(self):
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)

    def test_nf4_levels_attribute(self):
        """NF4Quantize node must carry the levels_f attribute with 16 values."""
        from src.formats.lookup_formats import NF4Format
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        for n in onnx_model.graph.node:
            if n.op_type == "NF4Quantize":
                levels = [a for a in n.attribute if a.name == "levels"]
                assert len(levels) == 1
                assert len(levels[0].floats) == len(NF4Format.NF4_LEVELS)
                return
        pytest.fail("No NF4Quantize node found")


# ---------------------------------------------------------------------------
# Task: Calibration scale wiring into ONNX QDQ nodes
# ---------------------------------------------------------------------------


class TestScaleWiring:
    """Real calibration scales embedded in ONNX QDQ nodes."""

    def test_calibrated_int8_exports_real_scale(self, tmp_path):
        """Calibrated int8 model → QDQ nodes use real calibration scale (not 1.0)."""
        from src.session._quant import _QuantSession
        from src.calibration.strategies import MaxScaleStrategy

        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        session = _QuantSession(model, cfg, calibrator=MaxScaleStrategy())

        # Run calibration so _output_scale buffers are registered
        x = torch.randn(2, 8)
        with session.calibrate():
            session(x)

        out = str(tmp_path / "scaled.onnx")
        session.export_onnx(out)

        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

        from onnx import numpy_helper
        # QDQ scale values are embedded in Constant op nodes as float tensor attrs,
        # not as graph initializers.  Find at least one non-1.0 scalar constant.
        # Zero-points (int8 dtype=3, value 0) are excluded by dtype and value.
        found_real_scale = False
        for node in loaded.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t.data_type == 1:  # float32
                        t = attr.t
                        arr = numpy_helper.to_array(t)
                        if arr.ndim == 0 and abs(float(arr.item()) - 1.0) > 0.01:
                            found_real_scale = True
                            break
                if found_real_scale:
                    break
        assert found_real_scale, "All QDQ scales are placeholder 1.0"

    def test_calibrated_int8_scale_flows_through_submodule(self, tmp_path):
        """Same check but with a named submodule (not root-only)."""

        # Use nn.Linear so quantize_model / _MODULE_MAPPING creates the
        # QuantizedLinear with the correct name ("fc") via _make_linear().
        # This exercises the name-based scale lookup in symbolic().

        class WrapperModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x):
                return self.fc(x)

        from src.session._quant import _QuantSession
        from src.calibration.strategies import MaxScaleStrategy
        cfg = _standard_cfg("int8")
        model = WrapperModel()
        session = _QuantSession(model, cfg, calibrator=MaxScaleStrategy())

        x = torch.randn(2, 8)
        with session.calibrate():
            session(x)

        out = str(tmp_path / "submodule_scaled.onnx")
        session.export_onnx(out)

        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

        from onnx import numpy_helper
        found_real_scale = False
        for node in loaded.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t.data_type == 1:
                        t = attr.t
                        arr = numpy_helper.to_array(t)
                        if arr.ndim == 0 and abs(float(arr.item()) - 1.0) > 0.01:
                            found_real_scale = True
                            break
                if found_real_scale:
                    break
        assert found_real_scale, "All QDQ scales are placeholder 1.0 in submodule model"


# ---------------------------------------------------------------------------
# Task 5: Full format coverage matrix
# ---------------------------------------------------------------------------

ALL_STANDARD_FORMATS = ["int8", "int4", "fp8_e4m3", "fp8_e5m2"]
ALL_NARROW_INT_FORMATS = ["int2"]
ALL_MX_FORMATS = ["fp4_e2m1", "fp6_e3m2", "fp6_e2m3"]
ALL_LOOKUP_FORMATS = ["nf4"]
ALL_TRUNC_FORMATS = ["bf16", "fp16"]
ALL_FORMATS = (ALL_STANDARD_FORMATS + ALL_NARROW_INT_FORMATS
               + ALL_MX_FORMATS + ALL_LOOKUP_FORMATS + ALL_TRUNC_FORMATS)


class TestFormatMatrix:
    """Every registered format exports valid ONNX for linear and conv2d."""

    @pytest.mark.parametrize("fmt_name", ALL_STANDARD_FORMATS)
    def test_standard_format_per_tensor_linear(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "QuantizeLinear"), \
            f"{fmt_name} per_tensor should use QDQ"

    @pytest.mark.parametrize("fmt_name", ALL_STANDARD_FORMATS)
    def test_standard_format_per_tensor_conv2d(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
        x = torch.randn(1, 4, 8, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "QuantizeLinear")

    @pytest.mark.parametrize("fmt_name", ALL_MX_FORMATS)
    def test_mx_format_per_block_linear(self, fmt_name):
        cfg = _mx_cfg(fmt_name, block_size=32)
        model = QuantizedLinear(32, 64, cfg=cfg)
        x = torch.randn(2, 32)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
            f"{fmt_name} per_block should use MxQuantize"

    @pytest.mark.parametrize("fmt_name", ALL_MX_FORMATS)
    def test_mx_format_per_block_conv2d(self, fmt_name):
        cfg = _mx_cfg(fmt_name, block_size=32)
        model = QuantizedConv2d(32, 64, kernel_size=3, padding=1, cfg=cfg)
        x = torch.randn(1, 32, 8, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "MxQuantize", "com.microxscaling")

    @pytest.mark.parametrize("fmt_name", ALL_LOOKUP_FORMATS)
    def test_lookup_format_per_tensor_linear(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "NF4Quantize", "com.microxscaling")

    @pytest.mark.parametrize("fmt_name", ALL_TRUNC_FORMATS)
    def test_trunc_format_per_tensor_linear(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        # Truncation formats use MxQuantize custom op
        assert _has_op(onnx_model, "MxQuantize", "com.microxscaling")

    @pytest.mark.parametrize("fmt_name", ALL_FORMATS)
    def test_all_formats_export_passthrough(self, fmt_name):
        """Passthrough (no cfg) exports cleanly regardless of format."""
        model = QuantizedLinear(8, 16)  # no cfg → passthrough
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
