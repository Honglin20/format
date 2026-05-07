"""Unit tests for Session atomic execution unit.

Session wraps QuantSession and orchestrates:
  calibrate -> analyze -> evaluate -> cost
"""
import pytest
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import Session, SessionResult
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
        # Should raise RuntimeError from QuantSession.optimize_scales
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
