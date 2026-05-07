"""Unit tests for Session atomic execution unit.

Session wraps _QuantSession and orchestrates:
  calibrate -> analyze -> evaluate -> cost

Covers both the full-pipeline ``run()`` and the step-by-step chainable API.
"""
import pytest
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import Session, SessionResult, _needs_calibration
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
# 1. Session construction
# ---------------------------------------------------------------------------

class TestSessionConstruction:
    """Session initialisation stores config and model references."""

    def test_stores_config(self):
        model = _make_small_model()
        config = QuantConfig(name="test_cfg")
        session = Session(model, config)
        assert session._config is config
        assert session._model is model
        assert session._keep_fp32 is True

    def test_keep_fp32_false(self):
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config, keep_fp32=False)
        assert session._keep_fp32 is False


# ---------------------------------------------------------------------------
# 2. Session.run() with basic config
# ---------------------------------------------------------------------------

class TestSessionRunBasic:
    """Session.run() with default options produces a valid SessionResult."""

    def test_returns_session_result(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)
        assert result.config is config

    def test_result_has_name(self):
        model = _make_small_model()
        config = QuantConfig(name="my_config", calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert result.name == "my_config"

    def test_result_metrics_populated(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn, outputs=["accuracy"])
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None
        assert "loss" in result.fp32_metrics
        assert "loss" in result.quant_metrics

    def test_result_delta_computed(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn, outputs=["accuracy"])
        assert result.delta is not None
        assert "loss" in result.delta

    def test_default_outputs_include_qsnr(self):
        """Default outputs=['accuracy', 'qsnr'] should produce qsnr data."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        # qsnr observer should have collected data
        assert len(result.observers_data) > 0

    def test_sq_transforms_none_when_not_smoothquant(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert result.sq_transforms is None


# ---------------------------------------------------------------------------
# 3. Output resolution - specific observer keys
# ---------------------------------------------------------------------------

class TestSessionOutputResolution:
    """Output key strings determine which observers are attached."""

    def test_histogram_output(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["histogram"],
        )
        # observers_data should contain data from HistogramObserver
        assert len(result.observers_data) > 0

    def test_mse_output(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["mse"],
        )
        assert len(result.mse_per_layer) > 0

    def test_error_dist_output(self):
        """error_dist uses distribution + mse observers."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["error_dist"],
        )
        # distribution + mse observers should have collected data
        assert len(result.observers_data) > 0

    def test_unknown_output_raises(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        with pytest.raises(ValueError, match="Unknown output key"):
            session.run(_calib_data(), eval_fn=_eval_fn, outputs=["unknown"])


# ---------------------------------------------------------------------------
# 4. outputs="all" covers all 17 keys
# ---------------------------------------------------------------------------

class TestSessionOutputsAll:
    """outputs='all' runs every registered output spec."""

    def test_outputs_all_runs(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        # Must provide eval_fn because some outputs need evaluation
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs="all",
        )
        assert isinstance(result, SessionResult)
        # observers_data should be populated (many outputs use qsnr/mse)
        assert len(result.observers_data) > 0

    def test_outputs_all_has_cost(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs="all",
        )
        # "cost" output triggers cost estimation
        assert result.cost is not None
        assert result.cost_fp32 is not None


# ---------------------------------------------------------------------------
# 5. eval_data defaults to calib_data
# ---------------------------------------------------------------------------

class TestSessionEvalData:
    """When eval_data is None, it defaults to calib_data."""

    def test_eval_data_defaults_to_calib(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        # Explicitly pass no eval_data
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["accuracy"],
        )
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None

    def test_eval_data_separate(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        eval_data = [torch.randn(4, 4) for _ in range(2)]
        result = session.run(
            _calib_data(), eval_data=eval_data, eval_fn=_eval_fn,
            outputs=["accuracy"],
        )
        assert result.fp32_metrics is not None


# ---------------------------------------------------------------------------
# 6. keep_fp32=False
# ---------------------------------------------------------------------------

class TestSessionNoFP32:
    """keep_fp32=False means fp32_metrics and cost_fp32 are None."""

    def test_fp32_metrics_none(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config, keep_fp32=False)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["accuracy"],
        )
        assert result.fp32_metrics is None
        # quant_metrics should still be populated
        assert result.quant_metrics is not None

    def test_cost_fp32_none(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config, keep_fp32=False)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["cost"],
        )
        assert result.cost is not None
        assert result.cost_fp32 is None


# ---------------------------------------------------------------------------
# 7. Custom eval_fn
# ---------------------------------------------------------------------------

class TestSessionCustomEval:
    """Custom eval_fn produces custom metrics."""

    def test_custom_metrics(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)

        def custom_eval(model, data):
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    for batch in data:
                        model(batch)
                else:
                    model(data)
            return {"custom_metric": 42.0, "another": 3.14}

        result = session.run(
            _calib_data(), eval_fn=custom_eval, outputs=["accuracy"],
        )
        assert result.fp32_metrics == {"custom_metric": 42.0, "another": 3.14}
        assert result.quant_metrics == {"custom_metric": 42.0, "another": 3.14}


# ---------------------------------------------------------------------------
# 8. Calibrator variants
# ---------------------------------------------------------------------------

class TestSessionCalibrators:
    """Different calibrator strings map to different ScaleStrategies."""

    def test_calibrator_percentile(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="percentile")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)

    def test_calibrator_kl(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="kl")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)

    def test_calibrator_mse(self):
        model = _make_small_model()
        config = QuantConfig(calibrator="mse")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)


# ---------------------------------------------------------------------------
# 9. weight_only=True
# ---------------------------------------------------------------------------

class TestSessionWeightOnly:
    """weight_only config should produce a valid result."""

    def test_weight_only_runs(self):
        model = _make_small_model()
        config = QuantConfig(weight_only=True, calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)

    def test_weight_only_with_outputs(self):
        model = _make_small_model()
        config = QuantConfig(weight_only=True, calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["accuracy", "qsnr"],
        )
        assert isinstance(result, SessionResult)


# ---------------------------------------------------------------------------
# 10. Prescale path
# ---------------------------------------------------------------------------

class TestSessionPrescale:
    """Prescale transform path: static init (lsq_steps=0)."""

    def test_prescale_static_runs(self):
        model = _make_small_model()
        config = QuantConfig(transform="prescale", calibrator="max")
        session = Session(model, config, keep_fp32=True)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)

    def test_prescale_with_lsq(self):
        """Full prescale path with LSQ optimization (lsq_steps > 0)."""
        model = _make_small_model()
        config = QuantConfig(
            transform="prescale", lsq_steps=10, lsq_lr=0.01,
            calibrator="max",
        )
        session = Session(model, config, keep_fp32=True)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)

    def test_prescale_lsq_requires_fp32(self):
        """LSQ optimization needs fp32_model (keep_fp32=True)."""
        model = _make_small_model()
        config = QuantConfig(
            transform="prescale", lsq_steps=5, calibrator="max",
        )
        session = Session(model, config, keep_fp32=False)
        # Should raise RuntimeError from _QuantSession.optimize_scales
        with pytest.raises(RuntimeError, match="keep_fp32"):
            session.run(_calib_data(), eval_fn=_eval_fn)


# ---------------------------------------------------------------------------
# 11. SmoothQuant path
# ---------------------------------------------------------------------------

class TestSessionSmoothQuant:
    """SmoothQuant transform path: SQ calibration + weight fusion."""

    def test_smoothquant_runs(self):
        model = _make_small_model()
        config = QuantConfig(transform="smoothquant", calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)

    def test_sq_transforms_cached(self):
        """SessionResult should contain the computed SQ transforms."""
        model = _make_small_model()
        config = QuantConfig(transform="smoothquant", calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert result.sq_transforms is not None
        # At least the Linear layers should have SQ transforms
        assert len(result.sq_transforms) > 0

    def test_smoothquant_forward_pass(self):
        """Forward pass metrics should be populated after smoothquant session."""
        model = _make_small_model()
        config = QuantConfig(transform="smoothquant", calibrator="max")
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None
        assert result.sq_transforms is not None

    def test_smoothquant_with_outputs(self):
        model = _make_small_model()
        config = QuantConfig(transform="smoothquant", calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn,
            outputs=["accuracy", "qsnr"],
        )
        assert isinstance(result, SessionResult)


# ---------------------------------------------------------------------------
# 12. lsq_steps > 0 without prescale raises (validated by QuantConfig)
# ---------------------------------------------------------------------------

class TestSessionLSQValidation:
    """QuantConfig validates that lsq_steps > 0 requires prescale."""

    def test_lsq_without_prescale_raises(self):
        with pytest.raises(
            ValueError,
            match="lsq_steps > 0 requires transform='prescale'",
        ):
            QuantConfig(transform="hadamard", lsq_steps=10)

    def test_lsq_with_smoothquant_raises(self):
        with pytest.raises(
            ValueError,
            match="lsq_steps > 0 requires transform='prescale'",
        ):
            QuantConfig(transform="smoothquant", lsq_steps=10)


# ---------------------------------------------------------------------------
# 13. SessionResult fields completeness
# ---------------------------------------------------------------------------

class TestSessionResultFields:
    """All SessionResult fields should be properly populated."""

    def test_all_fields_present(self):
        model = _make_small_model()
        config = QuantConfig(name="full_test", calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs="all",
        )
        assert result.name == "full_test"
        assert result.config is config
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None
        assert result.delta is not None
        assert isinstance(result.qsnr_per_layer, dict)
        assert isinstance(result.mse_per_layer, dict)
        assert isinstance(result.observers_data, dict)
        assert result.cost is not None
        assert result.cost_fp32 is not None
        assert result.sq_transforms is None  # not smoothquant

    def test_default_fields_not_none(self):
        """Non-observing outputs should still give valid empty dicts."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        result = session.run(
            _calib_data(), eval_fn=_eval_fn, outputs=["cost"],
        )
        # qsnr_per_layer and mse_per_layer should be empty (no observers)
        assert result.qsnr_per_layer == {}
        assert result.mse_per_layer == {}
        # observers_data should be empty (no observers)
        assert result.observers_data == {}


# ---------------------------------------------------------------------------
# 14. Session.run() without eval_fn (direct calibration only)
# ---------------------------------------------------------------------------

class TestSessionNoEvalFn:
    """Without eval_fn, Session still calibrates via direct model calls."""

    def test_calibrate_with_direct_calls(self):
        """Calibration should work using direct model(batch) calls."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        # Single tensor calibration data (not a list)
        single_batch = torch.randn(8, 4)
        result = session.run(single_batch, eval_fn=None)
        assert isinstance(result, SessionResult)
        # Without eval_fn and without outputs requesting eval, no metrics
        assert result.fp32_metrics is None
        assert result.quant_metrics is None


# ---------------------------------------------------------------------------
# 15. Integration: prescale with pot_amax init
# ---------------------------------------------------------------------------

class TestSessionPrescaleInitModes:
    """Prescale with different init modes."""

    def test_prescale_amax_init(self):
        model = _make_small_model()
        config = QuantConfig(
            transform="prescale", prescale_init="amax",
            calibrator="max",
        )
        session = Session(model, config, keep_fp32=True)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)

    def test_prescale_pot_amax_init(self):
        model = _make_small_model()
        config = QuantConfig(
            transform="prescale", prescale_init="pot_amax",
            calibrator="max",
        )
        session = Session(model, config, keep_fp32=True)
        result = session.run(_calib_data(), eval_fn=_eval_fn)
        assert isinstance(result, SessionResult)


# ===========================================================================
# 16. Chainable step-by-step API
# ===========================================================================


class TestSessionChainableAPI:
    """Session exposes a chainable API: .quantize() → .calibrate() → ..."""

    def test_quantize_makes_qmodel_available(self):
        """After .quantize(), session.qmodel is accessible."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        session.quantize()
        assert session.qmodel is not None
        assert isinstance(session.qmodel, nn.Module)

    def test_quantize_makes_fp32_model_available(self):
        """After .quantize() with keep_fp32=True, fp32_model is accessible."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config, keep_fp32=True)
        session.quantize()
        assert session.fp32_model is not None

    def test_fp32_model_none_when_keep_fp32_false(self):
        """After .quantize() with keep_fp32=False, fp32_model is None."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config, keep_fp32=False)
        session.quantize()
        assert session.fp32_model is None

    def test_call_delegates_to_quant_session(self):
        """session(x) delegates to the quantized model."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        session.quantize().calibrate(_calib_data())
        x = torch.randn(4, 4)
        out = session(x)
        assert out.shape == (4, 3)

    def test_qmodel_before_quantize_raises(self):
        """Accessing .qmodel before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            _ = session.qmodel

    def test_fp32_model_before_quantize_raises(self):
        """Accessing .fp32_model before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            _ = session.fp32_model

    def test_call_before_quantize_raises(self):
        """Calling session(x) before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            session(torch.randn(4, 4))

    def test_calibrate_before_quantize_raises(self):
        """Calling .calibrate() before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            session.calibrate(_calib_data())

    def test_analyze_before_quantize_raises(self):
        """Calling .analyze() before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            session.analyze(_calib_data())

    def test_evaluate_before_quantize_raises(self):
        """Calling .evaluate() before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            session.evaluate(_calib_data(), _eval_fn)

    def test_cost_before_quantize_raises(self):
        """Calling .cost() before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            session.cost()

    def test_result_before_quantize_raises(self):
        """Accessing .result before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            _ = session.result

    def test_chainable_returns_self(self):
        """Each step method returns self for chaining."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        data = _calib_data()

        assert session.quantize() is session
        assert session.calibrate(data) is session
        assert session.analyze(data) is session
        assert session.evaluate(data, _eval_fn) is session
        assert session.cost() is session

    def test_full_chain_produces_result(self):
        """Chaining all steps produces a valid SessionResult via .result."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        data = _calib_data()

        session.quantize()
        session.calibrate(data)
        session.analyze(data, outputs=["accuracy", "qsnr"])
        session.evaluate(data, _eval_fn)
        session.cost()

        result = session.result
        assert isinstance(result, SessionResult)
        assert result.config is config
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None
        assert result.delta is not None
        assert result.cost is not None

    def test_full_chain_one_liner(self):
        """Chained one-liner produces a valid SessionResult."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        data = _calib_data()

        result = (
            Session(model, config)
            .quantize()
            .calibrate(data)
            .analyze(data, outputs=["accuracy", "qsnr"])
            .evaluate(data, _eval_fn)
            .cost()
            .result
        )
        assert isinstance(result, SessionResult)
        assert result.fp32_metrics is not None

    def test_mode_property(self):
        """session.mode reflects current inference mode."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        session.quantize()
        assert session.mode == "quant"
        session.use_fp32()
        assert session.mode == "fp32"
        session.use_quant()
        assert session.mode == "quant"

    def test_use_fp32_before_quantize_raises(self):
        """Calling .use_fp32() before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            session.use_fp32()

    def test_use_quant_before_quantize_raises(self):
        """Calling .use_quant() before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            session.use_quant()

    def test_mode_before_quantize_raises(self):
        """Accessing .mode before .quantize() raises RuntimeError."""
        model = _make_small_model()
        config = QuantConfig()
        session = Session(model, config)
        with pytest.raises(RuntimeError, match="Call .quantize\\(\\) first"):
            _ = session.mode


# ===========================================================================
# 17. MX per_block no-op calibration
# ===========================================================================


class TestSessionMXNoCalibration:
    """MX per_block formats skip calibration (scales computed dynamically)."""

    def test_mx_per_block_calibrate_is_noop(self):
        """Calibrate on MX per_block config should be a no-op (no crash)."""
        model = _make_small_model()
        config = QuantConfig(
            w_format="fp4_e2m1",
            w_granularity="per_block",
            w_block_size=32,
            a_format="fp4_e2m1",
            a_granularity="per_block",
            a_block_size=32,
            calibrator="max",
        )
        session = Session(model, config)
        session.quantize()
        # calibrate should be a no-op for MX
        session.calibrate(_calib_data())
        assert session.qmodel is not None

    def test_mx_per_block_run_still_works(self):
        """run() should work on MX per_block config (calibration skipped)."""
        model = _make_small_model()
        config = QuantConfig(
            w_format="fp4_e2m1",
            w_granularity="per_block",
            w_block_size=32,
            a_format="fp4_e2m1",
            a_granularity="per_block",
            a_block_size=32,
            calibrator="max",
        )
        session = Session(model, config)
        result = session.run(_calib_data(), eval_fn=_eval_fn, outputs=["accuracy"])
        assert isinstance(result, SessionResult)
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None

    def test_mx_per_block_forward_pass(self):
        """Forward pass works on MX quantized model without calibration."""
        model = _make_small_model()
        config = QuantConfig(
            w_format="fp4_e2m1",
            w_granularity="per_block",
            w_block_size=32,
            a_format="fp4_e2m1",
            a_granularity="per_block",
            a_block_size=32,
        )
        session = Session(model, config)
        session.quantize()
        x = torch.randn(4, 4)
        out = session(x)
        assert out.shape == (4, 3)


# ===========================================================================
# 18. _needs_calibration helper
# ===========================================================================


class TestNeedsCalibration:
    """Unit tests for _needs_calibration()."""

    def test_per_tensor_needs_calibration(self):
        cfg = QuantConfig(
            w_granularity="per_tensor",
            a_granularity="per_tensor",
        ).to_op_config()
        assert _needs_calibration(cfg) is True

    def test_per_channel_needs_calibration(self):
        cfg = QuantConfig(
            w_granularity="per_channel",
            a_granularity="per_channel",
        ).to_op_config()
        assert _needs_calibration(cfg) is True

    def test_all_per_block_no_calibration(self):
        cfg = QuantConfig(
            w_format="fp4_e2m1",
            w_granularity="per_block",
            w_block_size=32,
            a_format="fp4_e2m1",
            a_granularity="per_block",
            a_block_size=32,
        ).to_op_config()
        assert _needs_calibration(cfg) is False

    def test_hybrid_needs_calibration(self):
        """If ANY scheme is not MX, calibration is needed."""
        cfg = QuantConfig(
            w_format="fp4_e2m1",
            w_granularity="per_block",
            w_block_size=32,
            a_format="int8",
            a_granularity="per_tensor",
        ).to_op_config()
        assert _needs_calibration(cfg) is True


# ===========================================================================
# 19. SessionResult accessor methods
# ===========================================================================


class TestSessionResultAccessors:
    """SessionResult accessor methods: summary, accuracy_table, top_k_qsnr, layer_report."""

    @staticmethod
    def _make_result(**overrides):
        """Build a SessionResult with sensible defaults for testing."""
        defaults = dict(
            name="test-cfg",
            config=QuantConfig(),
            fp32_metrics={"loss": 1.0, "acc": 0.95},
            quant_metrics={"loss": 1.2, "acc": 0.93},
            delta={"loss": -0.2, "acc": 0.02},
            qsnr_per_layer={
                "layer.0": 15.0,
                "layer.1": 30.0,
                "layer.2": 25.0,
                "layer.3": 40.0,
                "layer.4": 20.0,
            },
            mse_per_layer={
                "layer.0": 0.01,
                "layer.1": 0.001,
                "layer.2": 0.005,
                "layer.3": 0.0001,
                "layer.4": 0.003,
            },
        )
        defaults.update(overrides)
        return SessionResult(**defaults)

    def test_summary_returns_str(self):
        result = self._make_result()
        s = result.summary()
        assert isinstance(s, str)
        assert "test-cfg" in s
        assert "loss" in s
        assert "QSNR" in s

    def test_summary_without_metrics(self):
        result = self._make_result(fp32_metrics=None, quant_metrics=None, delta=None)
        s = result.summary()
        assert isinstance(s, str)
        assert "test-cfg" in s
        # Should not crash and should still show QSNR
        assert "QSNR" in s

    def test_summary_without_qsnr(self):
        result = self._make_result(qsnr_per_layer={})
        s = result.summary()
        assert isinstance(s, str)
        assert "QSNR" not in s

    def test_summary_without_name(self):
        result = self._make_result(name="")
        s = result.summary()
        assert "(unnamed)" in s

    def test_accuracy_table_returns_str(self):
        result = self._make_result()
        t = result.accuracy_table()
        assert isinstance(t, str)
        assert "Metric" in t
        assert "FP32" in t
        assert "Quant" in t
        assert "loss" in t
        assert "acc" in t

    def test_accuracy_table_without_metrics(self):
        result = self._make_result(fp32_metrics=None)
        t = result.accuracy_table()
        assert "no accuracy metrics" in t.lower()

    def test_top_k_qsnr_returns_sorted(self):
        result = self._make_result()
        top = result.top_k_qsnr(k=3)
        assert len(top) == 3
        # Should be sorted ascending (worst first)
        assert top[0][0] == "layer.0"  # 15.0 dB (worst)
        assert top[1][0] == "layer.4"  # 20.0 dB
        assert top[2][0] == "layer.2"  # 25.0 dB

    def test_top_k_qsnr_default_k(self):
        result = self._make_result()
        top = result.top_k_qsnr()
        assert len(top) == 5  # only 5 layers in test data, all returned

    def test_top_k_qsnr_empty(self):
        result = self._make_result(qsnr_per_layer={})
        top = result.top_k_qsnr()
        assert top == []

    def test_layer_report_returns_dataframe(self):
        pytest.importorskip("pandas")
        result = self._make_result()
        df = result.layer_report()
        assert df is not None
        assert list(df.columns) == ["layer", "qsnr_db", "mse"]
        assert len(df) == 5

    def test_layer_report_without_pandas(self, monkeypatch):
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Force re-import so the module sees pandas as unavailable
        import src.session._session as mod
        result = self._make_result()
        df = mod.SessionResult.layer_report(result)
        assert df is None

    def test_layer_report_empty(self):
        pytest.importorskip("pandas")
        result = self._make_result(qsnr_per_layer={}, mse_per_layer={})
        df = result.layer_report()
        assert len(df) == 0


# ===========================================================================
# 20. quantize_nonlinear switch — e2e tests
# ===========================================================================


def _make_model_with_all_op_types():
    """Model covering matmul, norm, activation, and pool op types.

    Returns a model with: Linear, Conv2d, BatchNorm2d, LayerNorm, ReLU,
    SiLU, Softmax, AdaptiveAvgPool2d, and Sigmoid.
    """
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 32),
        nn.LayerNorm(32),
        nn.SiLU(),
        nn.Linear(32, 10),
        nn.Softmax(dim=1),
        nn.Sigmoid(),
    )


def _get_module_types(model):
    """Return a dict mapping module name → class name for all children."""
    result = {}
    for child_name, child in model.named_children():
        result[child_name] = type(child).__name__
        # Also recurse into Sequential containers
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in child.named_children():
                result[f"{child_name}.{sub_name}"] = type(sub_child).__name__
    return result


class TestQuantizeNonLinearSwitch:
    """End-to-end tests for quantize_nonlinear switch."""

    # ------------------------------------------------------------------
    # quantize_nonlinear=False — module type verification
    # ------------------------------------------------------------------

    def test_nonlinear_modules_skipped(self):
        """quantize_nonlinear=False leaves norm/activation/pool as original types."""
        from src.session._model import quantize_model
        from src.scheme.op_config import OpQuantConfig

        model = _make_model_with_all_op_types()
        cfg = QuantConfig(w_format="int8").to_op_config()

        qmodel = quantize_model(model, cfg, quantize_nonlinear=False)

        types = _get_module_types(qmodel)
        # Matmul modules should be replaced
        assert types["0"].startswith("Quantized")   # Conv2d
        assert types["5"].startswith("Quantized")   # Linear
        assert types["8"].startswith("Quantized")   # Linear

        # Nonlinear modules should stay as original nn.* types
        assert types["1"] == "BatchNorm2d"          # NOT QuantizedBatchNorm2d
        assert types["2"] == "ReLU"                  # NOT QuantizedReLU
        assert types["3"] == "AdaptiveAvgPool2d"    # NOT QuantizedAdaptiveAvgPool2d
        assert types["6"] == "LayerNorm"            # NOT QuantizedLayerNorm
        assert types["7"] == "SiLU"                  # NOT QuantizedSiLU

    def test_only_matmul_types_in_matmul_tuple(self):
        """_MATMUL_TYPES contains exactly the matmul module classes."""
        from src.session._model import _MATMUL_TYPES
        import torch.nn as nn

        expected = {
            nn.Linear,
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        }
        assert set(_MATMUL_TYPES) == expected

    # ------------------------------------------------------------------
    # quantize_nonlinear=True (default) — backward compat
    # ------------------------------------------------------------------

    def test_default_quantizes_all_modules(self):
        """quantize_nonlinear=True (default) replaces all supported module types."""
        from src.session._model import quantize_model

        model = _make_model_with_all_op_types()
        cfg = QuantConfig(w_format="int8").to_op_config()

        qmodel = quantize_model(model, cfg)  # default: quantize_nonlinear=True

        types = _get_module_types(qmodel)
        # All supported modules should be "Quantized*" (Flatten has no quantized
        # counterpart and stays as-is).
        unsupported = {"Flatten"}
        for name, type_name in types.items():
            if type_name in unsupported:
                continue
            assert type_name.startswith("Quantized"), \
                f"{name} ({type_name}) should be Quantized*"

    # ------------------------------------------------------------------
    # Forward pass works
    # ------------------------------------------------------------------

    def test_forward_pass_quantize_nonlinear_false(self):
        """Forward pass succeeds with quantize_nonlinear=False."""
        from src.session._model import quantize_model

        model = _make_model_with_all_op_types()
        model.eval()
        cfg = QuantConfig(w_format="int8", calibrator="max").to_op_config()

        qmodel = quantize_model(model, cfg, quantize_nonlinear=False)
        qmodel.eval()

        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (2, 10)  # Softmax + Sigmoid → 10-dim
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_pass_quantize_nonlinear_true(self):
        """Forward pass succeeds with quantize_nonlinear=True."""
        from src.session._model import quantize_model

        model = _make_model_with_all_op_types()
        model.eval()
        cfg = QuantConfig(w_format="int8", calibrator="max").to_op_config()

        qmodel = quantize_model(model, cfg, quantize_nonlinear=True)
        qmodel.eval()

        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (2, 10)
        assert not torch.isnan(out).any()

    def test_forward_values_differ_between_modes(self):
        """quantize_nonlinear=False vs True produce different output values."""
        import copy
        from src.session._model import quantize_model

        cfg = QuantConfig(w_format="int8", calibrator="max").to_op_config()

        model_false = _make_model_with_all_op_types()
        model_false.eval()
        qmodel_false = quantize_model(
            model_false, cfg, quantize_nonlinear=False,
        )
        qmodel_false.eval()

        model_true = _make_model_with_all_op_types()
        model_true.load_state_dict(model_false.state_dict())
        model_true.eval()
        qmodel_true = quantize_model(
            model_true, cfg, quantize_nonlinear=True,
        )
        qmodel_true.eval()

        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            out_false = qmodel_false(x)
            out_true = qmodel_true(x)

        # Outputs differ because nonlinear modules are quantized in one
        # but not the other.
        assert not torch.allclose(out_false, out_true)

    # ------------------------------------------------------------------
    # Session high-level API integration
    # ------------------------------------------------------------------

    def test_session_quantize_nonlinear_false_runs(self):
        """Session.run() works with quantize_nonlinear=False."""
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 5),
        )
        config = QuantConfig(
            name="matmul-only",
            w_format="int8",
            calibrator="max",
            quantize_nonlinear=False,
        )
        session = Session(model, config)

        def eval_fn(m, data):
            with torch.no_grad():
                for batch in data:
                    m(batch)
            return {"loss": 0.5, "acc": 0.9}

        calib = [torch.randn(2, 3, 8, 8) for _ in range(4)]
        result = session.run(calib, eval_fn=eval_fn, outputs=["accuracy", "qsnr"])

        assert isinstance(result, SessionResult)
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None

        # Verify module types: BatchNorm and ReLU are untouched
        types = _get_module_types(session.qmodel)
        assert types["1"] == "BatchNorm2d"
        assert types["2"] == "ReLU"
        assert types["0"].startswith("Quantized")   # Conv2d quantized
        assert types["4"].startswith("Quantized")   # Linear quantized

    def test_session_chainable_api_quantize_nonlinear_false(self):
        """Chainable API works with quantize_nonlinear=False."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        config = QuantConfig(
            name="chain-test",
            w_format="int8",
            calibrator="max",
            quantize_nonlinear=False,
        )
        session = Session(model, config)
        data = [torch.randn(4, 10) for _ in range(4)]

        session.quantize().calibrate(data).analyze(
            data, outputs=["qsnr"],
        )

        result = session.result
        assert isinstance(result, SessionResult)

        # ReLU should be untouched
        types = _get_module_types(session.qmodel)
        assert types["1"] == "ReLU"
        assert types["0"].startswith("Quantized")
        assert types["2"].startswith("Quantized")

    def test_session_forward_quantize_nonlinear_false(self):
        """session(x) works with quantize_nonlinear=False."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        config = QuantConfig(
            w_format="int8",
            calibrator="max",
            quantize_nonlinear=False,
        )
        session = Session(model, config)
        session.quantize()

        x = torch.randn(3, 10)
        out = session(x)
        assert out.shape == (3, 5)
        assert not torch.isnan(out).any()

    # ------------------------------------------------------------------
    # QuantConfig validation
    # ------------------------------------------------------------------

    def test_quantize_nonlinear_default(self):
        """quantize_nonlinear defaults to True."""
        cfg = QuantConfig()
        assert cfg.quantize_nonlinear is True

    def test_quantize_nonlinear_field_stored(self):
        """quantize_nonlinear is stored as a field value."""
        cfg = QuantConfig(quantize_nonlinear=False)
        assert cfg.quantize_nonlinear is False

    def test_quantize_nonlinear_persists_in_result(self):
        """SessionResult.config.quantize_nonlinear reflects the config value."""
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
        config = QuantConfig(
            w_format="int8", calibrator="max", quantize_nonlinear=False,
        )
        session = Session(model, config)
        result = session.run(
            [torch.randn(2, 4)],
            eval_fn=lambda m, d: (([m(b) for b in d]), {"x": 0.0})[1],
            outputs=[],
        )
        assert result.config.quantize_nonlinear is False

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_model_with_only_matmul_modules(self):
        """quantize_nonlinear=False on a matmul-only model still quantizes everything."""
        from src.session._model import quantize_model

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))
        cfg = QuantConfig(w_format="int8").to_op_config()

        qmodel = quantize_model(model, cfg, quantize_nonlinear=False)

        types = _get_module_types(qmodel)
        for name, type_name in types.items():
            assert type_name.startswith("Quantized"), \
                f"{name} should be quantized (matmul-only model)"

    def test_model_with_only_nonlinear_modules(self):
        """quantize_nonlinear=False on a nonlinear-only model skips everything."""
        from src.session._model import quantize_model

        model = nn.Sequential(nn.ReLU(), nn.Sigmoid(), nn.Tanh())
        cfg = QuantConfig(w_format="int8").to_op_config()

        qmodel = quantize_model(model, cfg, quantize_nonlinear=False)

        types = _get_module_types(qmodel)
        assert types["0"] == "ReLU"
        assert types["1"] == "Sigmoid"
        assert types["2"] == "Tanh"

    def test_per_block_mx_with_quantize_nonlinear_false(self):
        """MX per_block format + quantize_nonlinear=False works correctly."""
        from src.session._model import quantize_model

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )
        model.eval()

        cfg = QuantConfig(
            w_format="fp4_e2m1",
            w_granularity="per_block",
            w_block_size=32,
            a_format="fp4_e2m1",
            a_granularity="per_block",
            a_block_size=32,
            quantize_nonlinear=False,
        ).to_op_config()

        qmodel = quantize_model(model, cfg, quantize_nonlinear=False)
        qmodel.eval()

        x = torch.randn(4, 4)
        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

        types = _get_module_types(qmodel)
        assert types["0"].startswith("Quantized")
        assert types["1"] == "BatchNorm1d"
        assert types["2"] == "ReLU"
        assert types["3"].startswith("Quantized")

    def test_storage_bits_nonlinear_true_vs_false_difference(self):
        """With storage_bits>0, quantize_nonlinear toggle changes nonlinear behaviour."""
        import copy
        from src.session._model import quantize_model

        cfg = QuantConfig(
            w_format="int8",
            storage_bits=16,
            storage_kind="bfloat",
            calibrator="max",
        ).to_op_config()

        model_false = _make_model_with_all_op_types()
        model_false.eval()
        qmodel_false = quantize_model(
            model_false, cfg, quantize_nonlinear=False,
        )
        qmodel_false.eval()

        model_true = _make_model_with_all_op_types()
        model_true.load_state_dict(model_false.state_dict())
        model_true.eval()
        qmodel_true = quantize_model(
            model_true, cfg, quantize_nonlinear=True,
        )
        qmodel_true.eval()

        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            out_false = qmodel_false(x)
            out_true = qmodel_true(x)

        # Storage quant + nonlinear quant → difference
        assert not torch.allclose(out_false, out_true)
