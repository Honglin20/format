"""
ONNX export tests — unified three-axis (Scale + Quantize / Truncate).

All formats use the same node types:
  - Non-truncation: Scale(granularity) → Quantize(format)
  - Truncation:     Truncate(dtype)
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
    buf = io.BytesIO()
    torch.onnx.export(
        model, (x,), buf,
        opset_version=17,
        custom_opsets={"com.microxscaling": 1},
    )
    buf.seek(0)
    return onnx.load(buf)


def _node_ops(onnx_model):
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
    s = QuantScheme(format=fmt, granularity=gran, scale_storage="fp32")
    return OpQuantConfig(input=s, weight=s, output=s)


def _mx_cfg(fmt_name="fp4_e2m1", block_size=32):
    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_block(block_size)
    s = QuantScheme(format=fmt, granularity=gran)
    return OpQuantConfig(input=s, weight=s, output=s)


# Unified node expectations after three-axis refactor
SCALE = ("com.microxscaling", "Scale")
QUANTIZE = ("com.microxscaling", "Quantize")
TRUNCATE = ("com.microxscaling", "Truncate")


# ---------------------------------------------------------------------------
# Task 1: Scale + Quantize / Truncate dispatch tests (replaces _is_standard_format)
# ---------------------------------------------------------------------------

class TestUnifiedDispatch:
    """Every format+grainularity emits the correct unified node pattern."""

    def test_int8_per_tensor_emits_scale_quantize(self):
        s = QuantScheme(format=FormatBase.from_str("int8"),
                        granularity=GranularitySpec.per_tensor())
        assert _has_op(_export(QuantizedLinear(8, 16, cfg=OpQuantConfig(input=s, weight=s)),
                               torch.randn(2, 8)), "Scale", "com.microxscaling")
        assert _has_op(_export(QuantizedLinear(8, 16, cfg=OpQuantConfig(input=s, weight=s)),
                               torch.randn(2, 8)), "Quantize", "com.microxscaling")

    def test_int4_per_channel_emits_scale_quantize(self):
        s = QuantScheme(format=FormatBase.from_str("int4"),
                        granularity=GranularitySpec.per_channel(axis=0))
        m = _export(QuantizedLinear(8, 16, cfg=OpQuantConfig(weight=s)), torch.randn(2, 8))
        assert _has_op(m, "Scale", "com.microxscaling")
        assert _has_op(m, "Quantize", "com.microxscaling")

    def test_fp8_per_tensor_emits_scale_quantize(self):
        s = QuantScheme(format=FormatBase.from_str("fp8_e4m3"),
                        granularity=GranularitySpec.per_tensor())
        m = _export(QuantizedLinear(8, 16, cfg=OpQuantConfig(input=s, weight=s)), torch.randn(2, 8))
        assert _has_op(m, "Scale", "com.microxscaling")
        assert _has_op(m, "Quantize", "com.microxscaling")

    def test_fp4_per_block_emits_scale_quantize(self):
        s = QuantScheme(format=FormatBase.from_str("fp4_e2m1"),
                        granularity=GranularitySpec.per_block(32))
        m = _export(QuantizedLinear(32, 64, cfg=OpQuantConfig(input=s, weight=s)), torch.randn(2, 32))
        assert _has_op(m, "Scale", "com.microxscaling")
        assert _has_op(m, "Quantize", "com.microxscaling")

    def test_int8_per_block_emits_scale_quantize(self):
        s = QuantScheme(format=FormatBase.from_str("int8"),
                        granularity=GranularitySpec.per_block(32))
        m = _export(QuantizedLinear(32, 64, cfg=OpQuantConfig(input=s, weight=s)), torch.randn(2, 32))
        assert _has_op(m, "Scale", "com.microxscaling")
        assert _has_op(m, "Quantize", "com.microxscaling")


# ---------------------------------------------------------------------------
# Task 2: LinearFunction ONNX export
# ---------------------------------------------------------------------------

def test_linear_emits_scale_quantize():
    """int8 per_tensor → Scale + Quantize nodes."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "Scale", "com.microxscaling")
    assert _has_op(onnx_model, "Quantize", "com.microxscaling")


def test_linear_per_block_emits_scale_quantize():
    """fp4_e2m1 per_block → Scale + Quantize nodes."""
    cfg = _mx_cfg("fp4_e2m1", block_size=32)
    model = QuantizedLinear(32, 64, cfg=cfg)
    x = torch.randn(2, 32)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "Scale", "com.microxscaling")
    assert _has_op(onnx_model, "Quantize", "com.microxscaling")


def test_linear_export_checker_passes():
    """onnx.checker passes for quantized linear export."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)


def test_linear_no_bias_exports_cleanly():
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, bias=False, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)


def test_linear_no_quantization_exports_cleanly():
    """Passthrough model exports as plain MatMul."""
    model = QuantizedLinear(8, 16)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)
    assert not _has_op(onnx_model, "Scale", "com.microxscaling")


# ---------------------------------------------------------------------------
# Task 3: ConvFunction ONNX export
# ---------------------------------------------------------------------------

def test_conv2d_emits_scale_quantize():
    cfg = _standard_cfg("int8")
    model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 4, 8, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "Scale", "com.microxscaling")
    assert _has_op(onnx_model, "Quantize", "com.microxscaling")


def test_conv2d_per_block_emits_scale_quantize():
    cfg = _mx_cfg("fp4_e2m1", block_size=32)
    model = QuantizedConv2d(32, 64, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 32, 8, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "Scale", "com.microxscaling")
    assert _has_op(onnx_model, "Quantize", "com.microxscaling")


def test_conv2d_export_checker_passes():
    cfg = _standard_cfg("int8")
    model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 4, 8, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)


# ---------------------------------------------------------------------------
# Task 4: export_quantized_model() end-to-end
# ---------------------------------------------------------------------------

def test_export_quantized_model_linear(tmp_path):
    from src.onnx import export_quantized_model
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    out_path = str(tmp_path / "model.onnx")
    export_quantized_model(model, x, out_path)
    loaded = onnx.load(out_path)
    onnx.checker.check_model(loaded)
    assert _has_op(loaded, "Scale", "com.microxscaling")


# ---------------------------------------------------------------------------
# Multi-input type tests
# ---------------------------------------------------------------------------

class TestMultiInput:

    @staticmethod
    def _make_two_input_model():
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
        from src.onnx import export_quantized_model
        cfg = _standard_cfg("int8")
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        out = str(tmp_path / "multi_arg.onnx")
        export_quantized_model(model, (x, y), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)


# ---------------------------------------------------------------------------
# Auto-input recording tests
# ---------------------------------------------------------------------------

class TestAutoInput:

    def test_export_after_forward_uses_recorded_input(self, tmp_path):
        from src.onnx import export_quantized_model
        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        out = str(tmp_path / "auto.onnx")
        export_quantized_model(model, x, out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_export_without_input_raises(self):
        # This test verifies that calling torch.onnx.export without input raises
        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        with pytest.raises(TypeError):
            torch.onnx.export(model, None, io.BytesIO(), opset_version=17)


def test_export_quantized_model_mixed(tmp_path):
    """Export a small mixed model: one Linear + one Conv2d, both quantized."""
    from src.onnx import export_quantized_model

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
    assert _has_op(loaded, "Scale", "com.microxscaling")
    assert _has_op(loaded, "Conv")


# ---------------------------------------------------------------------------
# NF4 (lookup-table format) ONNX export
# ---------------------------------------------------------------------------

class TestNF4Export:

    def test_nf4_per_tensor_emits_scale_quantize(self):
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "Scale", "com.microxscaling"), "NF4 should have Scale"
        assert _has_op(onnx_model, "Quantize", "com.microxscaling"), "NF4 should have Quantize"

    def test_nf4_per_block_emits_scale_quantize(self):
        fmt = FormatBase.from_str("nf4")
        s = QuantScheme(format=fmt, granularity=GranularitySpec.per_block(32))
        cfg = OpQuantConfig(input=s, weight=s)
        model = QuantizedLinear(32, 64, cfg=cfg)
        x = torch.randn(2, 32)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "Scale", "com.microxscaling")
        assert _has_op(onnx_model, "Quantize", "com.microxscaling")

    def test_nf4_export_checker_passes(self):
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)

    def test_nf4_levels_attribute(self):
        """Quantize node for NF4 must carry the levels_f attribute."""
        from src.formats.lookup_formats import NF4Format
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        for n in onnx_model.graph.node:
            if n.op_type == "Quantize":
                levels = [a for a in n.attribute if a.name == "levels"]
                if levels:
                    assert len(levels[0].floats) == len(NF4Format.NF4_LEVELS)
                    return
        pytest.fail("No Quantize node with levels found for NF4")


# ---------------------------------------------------------------------------
# Calibration scale wiring into ONNX
# ---------------------------------------------------------------------------

class TestScaleWiring:

    def test_calibrated_int8_with_quantize(self, tmp_path):
        """Calibrated int8 model exports Scale + Quantize nodes."""
        from src.onnx import export_quantized_model
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy

        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)

        x = torch.ones(2, 8) * 5.0
        with CalibrationSession(model, MaxScaleStrategy()):
            model(x)

        out = str(tmp_path / "scaled.onnx")
        export_quantized_model(model, x, out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert _has_op(loaded, "Scale", "com.microxscaling")
        assert _has_op(loaded, "Quantize", "com.microxscaling")

    def test_calibrated_scale_with_submodule(self, tmp_path):
        class WrapperModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x):
                return self.fc(x)

        from src.onnx import export_quantized_model
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy
        from src.session._model import quantize_model
        import copy

        cfg = _standard_cfg("int8")
        model = WrapperModel()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)

        x = torch.ones(2, 8) * 5.0
        with CalibrationSession(qmodel, MaxScaleStrategy()):
            qmodel(x)

        out = str(tmp_path / "submodule_scaled.onnx")
        export_quantized_model(qmodel, x, out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert _has_op(loaded, "Scale", "com.microxscaling")


# ---------------------------------------------------------------------------
# Full format coverage matrix
# ---------------------------------------------------------------------------

ALL_STANDARD_FORMATS = ["int8", "int4", "fp8_e4m3", "fp8_e5m2"]
ALL_NARROW_INT_FORMATS = ["int2"]
ALL_MX_FORMATS = ["fp4_e2m1", "fp6_e3m2", "fp6_e2m3"]
ALL_LOOKUP_FORMATS = ["nf4"]
ALL_TRUNC_FORMATS = ["bf16", "fp16"]
ALL_FORMATS = (ALL_STANDARD_FORMATS + ALL_NARROW_INT_FORMATS
               + ALL_MX_FORMATS + ALL_LOOKUP_FORMATS + ALL_TRUNC_FORMATS)


class TestFormatMatrix:

    @pytest.mark.parametrize("fmt_name", ALL_FORMATS)
    def test_all_formats_per_tensor_linear_pass_checker(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)

    @pytest.mark.parametrize("fmt_name", ALL_FORMATS)
    def test_all_formats_per_tensor_conv2d_pass_checker(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
        x = torch.randn(1, 4, 8, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)

    @pytest.mark.parametrize("fmt_name", ALL_FORMATS)
    def test_all_formats_per_block_linear_pass_checker(self, fmt_name):
        cfg = _mx_cfg(fmt_name, block_size=32)
        model = QuantizedLinear(32, 64, cfg=cfg)
        x = torch.randn(2, 32)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)

    @pytest.mark.parametrize("fmt_name", ALL_FORMATS)
    def test_all_formats_per_block_conv2d_pass_checker(self, fmt_name):
        cfg = _mx_cfg(fmt_name, block_size=32)
        model = QuantizedConv2d(32, 64, kernel_size=3, padding=1, cfg=cfg)
        x = torch.randn(1, 32, 8, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)

    # Non-truncation formats → Scale + Quantize
    NON_TRUNC = ALL_STANDARD_FORMATS + ALL_NARROW_INT_FORMATS + ALL_MX_FORMATS + ALL_LOOKUP_FORMATS

    @pytest.mark.parametrize("fmt_name", NON_TRUNC)
    def test_non_trunc_formats_emit_scale_quantize(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "Scale", "com.microxscaling"), \
            f"{fmt_name} should have Scale"
        assert _has_op(onnx_model, "Quantize", "com.microxscaling"), \
            f"{fmt_name} should have Quantize"

    @pytest.mark.parametrize("fmt_name", ALL_TRUNC_FORMATS)
    def test_trunc_formats_emit_truncate(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "Truncate", "com.microxscaling"), \
            f"{fmt_name} should have Truncate"
        assert not _has_op(onnx_model, "Scale", "com.microxscaling"), \
            f"{fmt_name} should NOT have Scale"

    @pytest.mark.parametrize("fmt_name", ALL_FORMATS)
    def test_all_formats_export_passthrough(self, fmt_name):
        """Passthrough (no cfg) exports cleanly regardless of format."""
        model = QuantizedLinear(8, 16)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)


# ---------------------------------------------------------------------------
# Session e2e quantize → calibrate → export_onnx pipeline
# ---------------------------------------------------------------------------

class TestSessionE2EExport:

    def test_session_quantize_calibrate_export(self, tmp_path):
        from src.session._session import run_quantization
        from src.session._config import QuantConfig
        from src.onnx import export_quantized_model

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
                self.linear = torch.nn.Linear(8 * 8 * 8, 16)

            def forward(self, x):
                x = self.conv(x)
                x = x.flatten(1)
                return self.linear(x)

        model = SimpleModel().eval()
        cfg = QuantConfig(
            w_format="int8",
            w_granularity="per_tensor",
            calibrator="max",
        )

        x = torch.randn(1, 4, 8, 8)
        qmodel, fp32_model, result = run_quantization(model, cfg, [x])
        out = str(tmp_path / "session.onnx")
        export_quantized_model(qmodel, x, out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert _has_op(loaded, "Scale", "com.microxscaling"), "Session export should have Scale"


# ---------------------------------------------------------------------------
# Multi-input model ONNX export
# ---------------------------------------------------------------------------

class TestMultiInputModels:

    @staticmethod
    def _int8_cfg():
        fmt = FormatBase.from_str("int8")
        gran = GranularitySpec.per_tensor()
        s = QuantScheme(format=fmt, granularity=gran, scale_storage="fp32")
        return OpQuantConfig(input=s, weight=s)

    def test_multi_arg_auto_record_export(self, tmp_path):
        from src.onnx import export_quantized_model

        class MultiArgModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cfg = TestMultiInputModels._int8_cfg()
                self.fc1 = QuantizedLinear(10, 20, cfg=cfg)
                self.fc2 = QuantizedLinear(10, 20, cfg=cfg)
                self.fc3 = QuantizedLinear(10, 20, cfg=cfg)

            def forward(self, x1, x2, x3):
                return self.fc1(x1) + self.fc2(x2) + self.fc3(x3)

        model = MultiArgModel()
        x1, x2, x3 = torch.randn(2, 10), torch.randn(2, 10), torch.randn(2, 10)
        out = str(tmp_path / "multi_arg.onnx")
        export_quantized_model(model, (x1, x2, x3), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert len(loaded.graph.input) >= 3

    def test_multi_arg_explicit_tuple_export(self, tmp_path):
        from src.onnx import export_quantized_model

        class MultiArgModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cfg = TestMultiInputModels._int8_cfg()
                self.fc1 = QuantizedLinear(10, 20, cfg=cfg)
                self.fc2 = QuantizedLinear(10, 20, cfg=cfg)
                self.fc3 = QuantizedLinear(10, 20, cfg=cfg)

            def forward(self, x1, x2, x3):
                return self.fc1(x1) + self.fc2(x2) + self.fc3(x3)

        model = MultiArgModel()
        x1, x2, x3 = torch.randn(2, 10), torch.randn(2, 10), torch.randn(2, 10)
        out = str(tmp_path / "multi_arg_explicit.onnx")
        export_quantized_model(model, (x1, x2, x3), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert len(loaded.graph.input) >= 3

    def test_list_input_auto_record_export(self, tmp_path):
        from src.onnx import export_quantized_model

        class ListInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cfg = TestMultiInputModels._int8_cfg()
                self.fc1 = QuantizedLinear(10, 20, cfg=cfg)
                self.fc2 = QuantizedLinear(20, 5, cfg=cfg)

            def forward(self, xs):
                x = xs[0] + xs[1]
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = ListInputModel()
        x_list = [torch.randn(2, 10), torch.randn(2, 10)]
        out = str(tmp_path / "list_input.onnx")
        export_quantized_model(model, (x_list,), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_list_input_explicit_tuple_wrap_export(self, tmp_path):
        from src.onnx import export_quantized_model

        class ListInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cfg = TestMultiInputModels._int8_cfg()
                self.fc1 = QuantizedLinear(10, 20, cfg=cfg)
                self.fc2 = QuantizedLinear(20, 5, cfg=cfg)

            def forward(self, xs):
                x = xs[0] + xs[1]
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = ListInputModel()
        out = str(tmp_path / "list_input_explicit.onnx")
        export_quantized_model(
            model,
            ([torch.randn(2, 10), torch.randn(2, 10)],),
            out,
        )
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_dict_kwargs_explicit_export(self, tmp_path):
        from src.onnx import export_quantized_model

        class KwargsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cfg = TestMultiInputModels._int8_cfg()
                self.fc1 = QuantizedLinear(10, 20, cfg=cfg)

            def forward(self, x, y):
                return self.fc1(x + y)

        model = KwargsModel()
        x, y = torch.randn(2, 10), torch.randn(2, 10)
        out = str(tmp_path / "kwargs.onnx")
        export_quantized_model(model, {"x": x, "y": y}, out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_full_session_multi_arg_export(self, tmp_path):
        from src.session._session import run_quantization
        from src.session._config import QuantConfig
        from src.onnx import export_quantized_model

        class MultiArgModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(10, 20)
                self.fc3 = torch.nn.Linear(10, 20)

            def forward(self, x1, x2, x3):
                return self.fc1(x1) + self.fc2(x2) + self.fc3(x3)

        model = MultiArgModel().eval()
        cfg = QuantConfig(
            name="test", w_format="int8", w_granularity="per_tensor",
            a_format="int8", a_granularity="per_tensor", calibrator="max",
        )
        x1, x2, x3 = torch.randn(2, 10), torch.randn(2, 10), torch.randn(2, 10)
        qmodel, fp32_model, result = run_quantization(
            model, cfg, [(x1, x2, x3)],
        )
        out = str(tmp_path / "full_multi_arg.onnx")
        export_quantized_model(qmodel, (x1, x2, x3), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert len(loaded.graph.input) >= 3

    def test_full_session_list_input_export(self, tmp_path):
        from src.session._session import run_quantization
        from src.session._config import QuantConfig
        from src.onnx import export_quantized_model

        class ListInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(20, 5)

            def forward(self, xs):
                x = xs[0] + xs[1]
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = ListInputModel().eval()
        cfg = QuantConfig(
            name="test", w_format="int8", w_granularity="per_tensor",
            a_format="int8", a_granularity="per_tensor", calibrator="max",
        )
        x_list = [torch.randn(2, 10), torch.randn(2, 10)]
        qmodel, fp32_model, result = run_quantization(
            model, cfg, [x_list],
        )
        out = str(tmp_path / "full_list.onnx")
        export_quantized_model(qmodel, (x_list,), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_full_session_kwargs_export(self, tmp_path):
        from src.session._session import run_quantization
        from src.session._config import QuantConfig
        from src.onnx import export_quantized_model

        class KwargsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)

            def forward(self, x, y):
                return self.fc1(x + y)

        model = KwargsModel().eval()
        cfg = QuantConfig(
            name="test", w_format="int8", w_granularity="per_tensor",
            a_format="int8", a_granularity="per_tensor", calibrator="max",
        )
        x, y = torch.randn(2, 10), torch.randn(2, 10)
        qmodel, fp32_model, result = run_quantization(
            model, cfg, [(x, y)],
        )
        out = str(tmp_path / "full_kwargs.onnx")
        export_quantized_model(qmodel, {"x": x, "y": y}, out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_full_session_mx_per_block_multi_input(self, tmp_path):
        from src.session._session import run_quantization
        from src.session._config import QuantConfig
        from src.onnx import export_quantized_model

        class MultiArgModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(10, 20)
                self.fc3 = torch.nn.Linear(10, 20)

            def forward(self, x1, x2, x3):
                return self.fc1(x1) + self.fc2(x2) + self.fc3(x3)

        model = MultiArgModel().eval()
        cfg = QuantConfig(
            name="mxint4", w_format="int4", w_granularity="per_block",
            w_block_size=32, a_format="int4", a_granularity="per_block",
            a_block_size=32, storage_bits=16, storage_kind="bfloat",
            quantize_nonlinear=False,
        )
        x1, x2, x3 = torch.randn(2, 10), torch.randn(2, 10), torch.randn(2, 10)
        qmodel, fp32_model, result = run_quantization(
            model, cfg, [(x1, x2, x3)], keep_fp32=True,
        )
        out = str(tmp_path / "mx_multi_arg.onnx")
        export_quantized_model(qmodel, (x1, x2, x3), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        assert len(loaded.graph.input) >= 3


# ---------------------------------------------------------------------------
# Per-channel ONNX export
# ---------------------------------------------------------------------------

class TestPerChannelExport:

    def test_int8_per_channel_has_scale_axis(self):
        """int8 per_channel → Scale node with axis attribute."""
        fmt = FormatBase.from_str("int8")
        gran = GranularitySpec.per_channel(axis=0)
        s = QuantScheme(format=fmt, granularity=gran)
        cfg = OpQuantConfig(weight=s)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        for n in onnx_model.graph.node:
            if n.op_type == "Scale":
                attrs = {a.name: a for a in n.attribute}
                assert "mode" in attrs and attrs["mode"].s == b"per_channel"
                assert "axis" in attrs, "per_channel Scale should have axis"
                return
        pytest.fail("No Scale node with per_channel mode found")

    @pytest.mark.parametrize("fmt_name", ["int8", "int4", "fp8_e4m3", "fp8_e5m2"])
    def test_standard_format_per_channel_linear(self, fmt_name):
        fmt = FormatBase.from_str(fmt_name)
        gran = GranularitySpec.per_channel(axis=-1)
        s = QuantScheme(format=fmt, granularity=gran)
        cfg = OpQuantConfig(input=s, weight=s)
        model = QuantizedLinear(16, 32, cfg=cfg)
        x = torch.randn(2, 16)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Scale", "com.microxscaling")
        assert _has_op(onnx_model, "Quantize", "com.microxscaling")

    @pytest.mark.parametrize("fmt_name", ["int8", "int4"])
    def test_standard_format_per_channel_conv2d(self, fmt_name):
        fmt = FormatBase.from_str(fmt_name)
        gran = GranularitySpec.per_channel(axis=0)
        s = QuantScheme(format=fmt, granularity=gran)
        cfg = OpQuantConfig(input=s, weight=s)
        model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
        x = torch.randn(1, 4, 8, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Scale", "com.microxscaling")
        assert _has_op(onnx_model, "Quantize", "com.microxscaling")


# ---------------------------------------------------------------------------
# Complex model ONNX export
# ---------------------------------------------------------------------------

class TestComplexModelExport:

    def test_toy_mlp_int8_export(self):
        from pipeline._model import ToyMLP

        fmt = FormatBase.from_str("int8")
        s = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(input=s, weight=s, output=s)

        model = ToyMLP()
        model.fc1 = QuantizedLinear(128, 512, bias=True, cfg=cfg, name="fc1")
        model.fc2 = QuantizedLinear(512, 128, bias=True, cfg=cfg, name="fc2")
        model.head = QuantizedLinear(128, 10, bias=True, cfg=cfg, name="head")

        x = torch.randn(1, 128)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Scale", "com.microxscaling")
        assert _has_op(onnx_model, "Quantize", "com.microxscaling")
        assert _has_op(onnx_model, "MatMul")

    def test_toy_mlp_mixed_config_export(self):
        from pipeline._model import ToyMLP

        fmt_mx = FormatBase.from_str("int4")
        s_mx = QuantScheme(format=fmt_mx, granularity=GranularitySpec.per_block(32))
        cfg_mx = OpQuantConfig(input=s_mx, weight=s_mx, output=s_mx)

        fmt_int8 = FormatBase.from_str("int8")
        s_pt = QuantScheme(format=fmt_int8, granularity=GranularitySpec.per_tensor())
        cfg_pt = OpQuantConfig(input=s_pt, weight=s_pt, output=s_pt)

        fmt_bf = FormatBase.from_str("bfloat16")
        s_bf = QuantScheme(format=fmt_bf, granularity=GranularitySpec.per_tensor())
        cfg_bf = OpQuantConfig(storage=s_bf, input=s_bf, weight=s_bf)

        model = ToyMLP()
        model.fc1 = QuantizedLinear(128, 512, bias=True, cfg=cfg_mx, name="fc1")
        model.fc2 = QuantizedLinear(512, 128, bias=True, cfg=cfg_pt, name="fc2")
        model.head = QuantizedLinear(128, 10, bias=True, cfg=cfg_bf, name="head")

        x = torch.randn(1, 128)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Scale", "com.microxscaling"), "Should have Scale"
        assert _has_op(onnx_model, "Quantize", "com.microxscaling"), "Should have Quantize"
        assert _has_op(onnx_model, "Truncate", "com.microxscaling"), "Should have Truncate from bf16"
        assert _has_op(onnx_model, "MatMul")

    def test_simple_convnet_int8_per_channel_export(self):
        """Conv2d + AvgPool2d + Linear — no AdaptiveAvgPool2d (ONNX-incompatible)."""

        class SimpleConvMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = QuantizedConv2d(1, 8, kernel_size=3, padding=1)
                self.conv2 = QuantizedConv2d(8, 16, kernel_size=3, padding=1)
                self.pool = torch.nn.AvgPool2d(kernel_size=4)
                self.fc = QuantizedLinear(16 * 4 * 2, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x)
                x = torch.nn.functional.relu(x)
                x = self.pool(x)
                x = x.flatten(1)
                return self.fc(x)

        fmt = FormatBase.from_str("int8")
        s_conv = QuantScheme(format=fmt, granularity=GranularitySpec.per_channel(axis=0))
        cfg_conv = OpQuantConfig(input=s_conv, weight=s_conv, output=s_conv)
        s_lin = QuantScheme(format=fmt, granularity=GranularitySpec.per_channel(axis=-1))
        cfg_lin = OpQuantConfig(input=s_lin, weight=s_lin, output=s_lin)

        model = SimpleConvMLP()
        model.conv1.cfg = cfg_conv
        model.conv2.cfg = cfg_conv
        model.fc.cfg = cfg_lin

        x = torch.randn(1, 1, 16, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Scale", "com.microxscaling")
        assert _has_op(onnx_model, "Quantize", "com.microxscaling")
        assert _has_op(onnx_model, "Conv")

    def test_storage_compute_combo_export(self):
        fmt_bf = FormatBase.from_str("bfloat16")
        s_bf = QuantScheme(format=fmt_bf, granularity=GranularitySpec.per_tensor())

        fmt_int4 = FormatBase.from_str("int4")
        s_mx = QuantScheme(format=fmt_int4, granularity=GranularitySpec.per_block(32))

        cfg = OpQuantConfig(storage=s_bf, input=s_mx, weight=s_mx, output=s_mx)

        model = QuantizedLinear(32, 64, cfg=cfg)
        x = torch.randn(2, 32)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Truncate", "com.microxscaling"), "bf16 storage should emit Truncate"
        assert _has_op(onnx_model, "Quantize", "com.microxscaling"), "int4 compute should emit Quantize"

    def test_int2_emits_scale_quantize(self):
        """int2 per_tensor → Scale + Quantize (unified path, no special fallback)."""
        fmt = FormatBase.from_str("int2")
        s = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(input=s, weight=s)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Scale", "com.microxscaling")
        assert _has_op(onnx_model, "Quantize", "com.microxscaling")

    def test_fp4_per_tensor_emits_scale_quantize(self):
        """fp4_e2m1 per_tensor → Scale + Quantize."""
        fmt = FormatBase.from_str("fp4_e2m1")
        s = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(input=s, weight=s)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "Scale", "com.microxscaling")
        assert _has_op(onnx_model, "Quantize", "com.microxscaling")
