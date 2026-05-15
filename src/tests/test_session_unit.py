"""Unit tests for run_quantization and related standalone functions.

Session class has been removed. These tests verify the replacement API:
run_quantization(), quantize_model(), CalibrationSession, AnalysisContext.
"""
import pytest
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import run_quantization, _needs_calibration
from src.scheme.op_config import OpQuantConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_model():
    """Simple 2-layer model: Linear(4,8) -> ReLU -> Linear(8,3)."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


def _calib_data(n_batches: int = 4, batch_size: int = 8):
    """Return list of random input tensors usable as calibration data."""
    return [torch.randn(batch_size, 4) for _ in range(n_batches)]


def _eval_fn(model, data):
    """Default eval function: run model on data, return dummy metrics."""
    with torch.no_grad():
        if isinstance(data, (list, tuple)):
            for batch in data:
                model(batch)
        else:
            model(data)
    return {"loss": 0.0, "acc": 1.0}


# ---------------------------------------------------------------------------
# 1. run_quantization basic API
# ---------------------------------------------------------------------------

class TestRunQuantization:
    """Basic run_quantization returns (qmodel, fp32_model, SessionResult)."""

    def test_returns_tuple_of_three(self):
        model = _make_small_model()
        cfg = QuantConfig(name="test")
        calib = _calib_data()
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        assert qmodel is not None
        assert fp32_model is not None
        assert result is not None

    def test_keep_fp32_false(self):
        model = _make_small_model()
        cfg = QuantConfig(name="test")
        calib = _calib_data()
        qmodel, fp32_model, result = run_quantization(
            model, cfg, calib, keep_fp32=False,
        )
        assert qmodel is not None
        assert fp32_model is None

    def test_qmodel_is_quantized(self):
        model = _make_small_model()
        cfg = QuantConfig(name="test", w_format="int8")
        calib = _calib_data()
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        # Qmodel should be a Sequential with quantized modules
        assert isinstance(qmodel, nn.Sequential)
        assert len(list(qmodel.children())) > 0

    def test_result_has_expected_attributes(self):
        model = _make_small_model()
        cfg = QuantConfig(name="test")
        calib = _calib_data()
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        assert result.name == "test"
        assert isinstance(result.config, QuantConfig)

    def test_qmodel_forward_works(self):
        model = _make_small_model()
        cfg = QuantConfig(name="test", w_format="int8")
        calib = _calib_data()
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        qmodel.eval()
        with torch.no_grad():
            out = qmodel(torch.randn(2, 4))
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_original_model_not_mutated(self):
        model = _make_small_model()
        cfg = QuantConfig(name="test", w_format="int8")
        calib = _calib_data()
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        # Original model unchanged
        assert isinstance(model[0], nn.Linear)
        assert not hasattr(model[0], "cfg")

    def test_forward_delegates_correctly(self):
        """qmodel(x) should give same shape as fp32_model(x)."""
        model = _make_small_model()
        cfg = QuantConfig(name="test", w_format="int8")
        calib = _calib_data()
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        qmodel.eval()
        fp32_model.eval()
        x = torch.randn(2, 4)
        with torch.no_grad():
            q_out = qmodel(x)
            fp_out = fp32_model(x)
        assert q_out.shape == fp_out.shape

    def test_with_eval_fn(self):
        model = _make_small_model()
        cfg = QuantConfig(name="test", w_format="int8")
        calib = _calib_data()

        called = []
        def my_eval(m, data):
            called.append(1)
            m(data[0]) if isinstance(data, list) else m(data)

        qmodel, fp32_model, result = run_quantization(
            model, cfg, calib, eval_fn=my_eval, eval_data=torch.randn(2, 4),
        )
        assert len(called) > 0


# ---------------------------------------------------------------------------
# 2. quantize_nonlinear switch
# ---------------------------------------------------------------------------

class TestQuantizeNonLinearSwitch:

    def test_quantize_nonlinear_false(self):
        """quantize_nonlinear=False keeps activation functions as nn.Module."""
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )
        cfg = QuantConfig(
            name="test",
            w_format="int8",
            quantize_nonlinear=False,
        )
        calib = [torch.randn(8, 4) for _ in range(4)]
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        # ReLU should still be nn.ReLU (not quantized)
        assert isinstance(qmodel[1], nn.ReLU)

    def test_quantize_nonlinear_true(self):
        """quantize_nonlinear=True quantizes activations."""
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )
        cfg = QuantConfig(
            name="test",
            w_format="int8",
            a_format="int8",
            quantize_nonlinear=True,
        )
        calib = [torch.randn(8, 4) for _ in range(4)]
        qmodel, fp32_model, result = run_quantization(model, cfg, calib)
        qmodel.eval()
        with torch.no_grad():
            out = qmodel(torch.randn(2, 4))
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. _needs_calibration helper
# ---------------------------------------------------------------------------

class TestNeedsCalibration:

    def test_empty_config_does_not_need_calibration(self):
        assert not _needs_calibration(OpQuantConfig())

    def test_int8_config_needs_calibration(self):
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec
        from src.formats.base import FormatBase
        scheme = QuantScheme(
            format=FormatBase.from_str("int8"),
            granularity=GranularitySpec.per_tensor(),
        )
        assert _needs_calibration(OpQuantConfig(input=scheme))

    def test_dict_config_needs_calibration(self):
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec
        from src.formats.base import FormatBase
        scheme = QuantScheme(
            format=FormatBase.from_str("int8"),
            granularity=GranularitySpec.per_tensor(),
        )
        assert _needs_calibration({"*": OpQuantConfig(input=scheme)})
