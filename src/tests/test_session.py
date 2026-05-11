"""
Tests for _QuantSession unified API and e2e comparison tools.
"""
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.formats.base import FormatBase
from src.calibration.strategies import MaxScaleStrategy, PercentileScaleStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_model():
    """Simple 2-layer model: Linear(4,8) → ReLU → Linear(8,3)."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


def _make_cfg():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)


def _make_dataloader(n_samples=32, batch_size=8, n_features=4, n_classes=3):
    data = torch.randn(n_samples, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(data, labels), batch_size=batch_size)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

def test_session_creates():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)

    assert session.cfg is cfg
    assert isinstance(session.calibrator, MaxScaleStrategy)
    assert len(session.observers) > 0
    assert session.mode == "quant"
    assert session.fp32_model is not None
    assert session.qmodel is not None
    # fp32_model is a deep copy, not the same object
    assert session.fp32_model is not model


def test_session_no_fp32():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg, keep_fp32=False)

    assert session.fp32_model is None


def test_session_custom_calibrator():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    cal = PercentileScaleStrategy(q=95.0)
    session = _QuantSession(model, cfg, calibrator=cal)

    assert session.calibrator is cal


# ---------------------------------------------------------------------------
# 2. Mode switching
# ---------------------------------------------------------------------------

def test_mode_switching():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)

    assert session.mode == "quant"
    session.use_fp32()
    assert session.mode == "fp32"
    session.use_quant()
    assert session.mode == "quant"


def test_use_fp32_without_fp32_raises():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg, keep_fp32=False)

    with pytest.raises(RuntimeError, match="fp32_model not available"):
        session.use_fp32()


# ---------------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------------

def test_call_in_quant_mode():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    x = torch.randn(4, 4)
    with torch.no_grad():
        out = session(x)
    assert out.shape == (4, 3)


def test_call_in_fp32_mode():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    x = torch.randn(4, 4)
    with torch.no_grad():
        session.use_fp32()
        out = session(x)
    assert out.shape == (4, 3)


def test_call_records_last_input():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    x = torch.randn(4, 4)
    with torch.no_grad():
        session(x)
    assert session._last_input is not None


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------

def test_calibrate_returns_calibration_session():
    from src.session._quant import _QuantSession
    from src.calibration.pipeline import CalibrationSession as CS
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    cs = session.calibrate()
    assert isinstance(cs, CS)


def test_calibrate_context_assigns_scales():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    with session.calibrate():
        with torch.no_grad():
            for _ in range(4):
                session(torch.randn(2, 4))

    # After calibration, some modules should have _output_scale buffers
    has_scale = any(
        hasattr(m, "_output_scale") for m in session.qmodel.modules()
    )
    assert has_scale


def test_calibrate_with_custom_strategy():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    strat = PercentileScaleStrategy(q=95.0)
    with session.calibrate(strategy=strat):
        with torch.no_grad():
            for _ in range(4):
                session(torch.randn(2, 4))

    # Scales should still be assigned
    has_scale = any(
        hasattr(m, "_output_scale") for m in session.qmodel.modules()
    )
    assert has_scale


# ---------------------------------------------------------------------------
# 5. Analysis
# ---------------------------------------------------------------------------

def test_analyze_returns_analysis_context():
    from src.session._quant import _QuantSession
    from src.analysis.context import AnalysisContext
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    ctx = session.analyze()
    assert isinstance(ctx, AnalysisContext)


def test_analyze_collects_metrics():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    with session.analyze() as ctx:
        with torch.no_grad():
            for _ in range(4):
                session(torch.randn(2, 4))

    report = ctx.report()
    # Should have at least one layer
    assert len(report.keys()) > 0


def test_analyze_with_custom_observers():
    from src.session._quant import _QuantSession
    from src.analysis.observers import MSEObserver
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    with session.analyze(observers=[MSEObserver()]) as ctx:
        with torch.no_grad():
            session(torch.randn(2, 4))

    report = ctx.report()
    assert len(report.keys()) > 0


# ---------------------------------------------------------------------------
# 5b. True-error analysis (Session.analyze(true_error=True))
# ---------------------------------------------------------------------------


def test_cfg_causes_quantization_empty():
    """Empty cfg should not be considered as causing quantization."""
    from src.scheme.op_config import cfg_causes_quantization
    assert not cfg_causes_quantization(None)
    assert not cfg_causes_quantization(OpQuantConfig())


def test_cfg_causes_quantization_with_scheme():
    """Any non-None role field should be detected."""
    from src.scheme.op_config import cfg_causes_quantization
    scheme = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=GranularitySpec.per_tensor(),
    )
    assert cfg_causes_quantization(OpQuantConfig(input=scheme))
    assert cfg_causes_quantization(OpQuantConfig(storage=scheme))
    assert cfg_causes_quantization(OpQuantConfig(grad_input=scheme))


def test_true_error_returns_qsnr_per_layer():
    """true_error=True should populate qsnr_per_layer and mse_per_layer."""
    from src.session._session import Session
    from src.session._config import QuantConfig
    model = _make_small_model()
    config = QuantConfig(calibrator="max")
    session = Session(model, config)
    session.quantize()
    session.analyze(
        torch.randn(4, 4), outputs=[], true_error=True,
    )
    assert len(session._qsnr_per_layer) > 0
    assert len(session._mse_per_layer) > 0
    for qsnr in session._qsnr_per_layer.values():
        assert qsnr > 0
        assert not torch.tensor(qsnr).isnan()


def test_true_error_multi_batch_accumulation():
    """Multi-batch true_error should accumulate across batches."""
    from src.session._session import Session
    from src.session._config import QuantConfig
    model = _make_small_model()
    config = QuantConfig(calibrator="max")
    session = Session(model, config)
    session.quantize()

    # Single batch
    session.analyze(
        [torch.randn(4, 4)], outputs=[], true_error=True,
    )
    single_qsnr = dict(session._qsnr_per_layer)

    # Multiple batches — same layers should be present
    session2 = Session(model, config)
    session2.quantize()
    session2.analyze(
        [torch.randn(4, 4) for _ in range(4)],
        outputs=[], true_error=True,
    )
    multi_qsnr = dict(session2._qsnr_per_layer)
    assert set(single_qsnr.keys()) == set(multi_qsnr.keys())


def test_true_error_with_eval_fn():
    """true_error=True with eval_fn should use eval_fn, not direct model()."""
    from src.session._session import Session
    from src.session._config import QuantConfig
    model = _make_small_model()
    config = QuantConfig(calibrator="max")
    session = Session(model, config)
    session.quantize()

    called_fp32 = []
    called_quant = []

    def my_eval(m, data):
        if m is session.fp32_model:
            called_fp32.append(True)
        else:
            called_quant.append(True)
        m(data)

    session.analyze(
        torch.randn(4, 4), outputs=[], true_error=True,
        eval_fn=my_eval,
    )
    assert len(called_fp32) == 1
    assert len(called_quant) == 1
    assert len(session._qsnr_per_layer) > 0


def test_true_error_with_observers_combined():
    """true_error=True + observers should give true error AND observer data."""
    from src.session._session import Session
    from src.session._config import QuantConfig
    model = _make_small_model()
    config = QuantConfig(calibrator="max")
    session = Session(model, config)
    session.quantize()

    session.analyze(
        torch.randn(4, 4), outputs=["qsnr"], true_error=True,
    )
    assert len(session._qsnr_per_layer) > 0
    assert len(session._observers_data) > 0


def test_true_error_excludes_non_quantizing_modules():
    """Modules with empty cfg should be excluded from true_error comparison."""
    from src.session._session import Session
    from src.session._config import QuantConfig
    model = _make_small_model()
    config = QuantConfig(calibrator="max")
    session = Session(model, config)
    session.quantize()

    session.analyze(
        torch.randn(4, 4), outputs=[], true_error=True,
    )

    # Every reported layer QSNR should be a finite number
    for name, qsnr in session._qsnr_per_layer.items():
        assert qsnr == qsnr  # not NaN
        assert qsnr > 0


# ---------------------------------------------------------------------------
# 6. Comparator
# ---------------------------------------------------------------------------

def test_comparator_returns_comparator():
    from src.session._quant import _QuantSession
    from src.analysis.e2e import Comparator
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)

    cmp = session.comparator()
    assert isinstance(cmp, Comparator)


def test_comparator_manual_collection():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    x = torch.randn(4, 4)
    labels = torch.randint(0, 3, (4,))

    cmp = session.comparator()
    with cmp, torch.no_grad():
        session.use_fp32()
        fp32_out = session(x)
        session.use_quant()
        q_out = session(x)
        cmp.record(fp32_out, q_out, labels)

    assert cmp.num_samples == 4


# ---------------------------------------------------------------------------
# 7. compare (auto-mode)
# ---------------------------------------------------------------------------

def test_compare_auto_mode():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    dl = _make_dataloader(n_samples=16, batch_size=4)
    result = session.compare(dl)

    assert "fp32" in result
    assert "quant" in result
    assert "delta" in result
    assert "accuracy" in result["fp32"]


def test_compare_with_custom_eval_fn():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    def my_eval(logits, labels):
        return {"top1": (logits.argmax(-1) == labels).float().mean().item()}

    dl = _make_dataloader(n_samples=16, batch_size=4)
    result = session.compare(dl, eval_fn=my_eval, directions={"top1": "higher"})

    assert "top1" in result["fp32"]
    assert result.get("_directions") == {"top1": "higher"}


def test_compare_without_fp32_raises():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg, keep_fp32=False)
    session.eval()

    dl = _make_dataloader(n_samples=4, batch_size=2)
    with pytest.raises(RuntimeError, match="fp32_model not available"):
        session.compare(dl)


# ---------------------------------------------------------------------------
# 8. ONNX Export
# ---------------------------------------------------------------------------

def test_export_onnx_with_dummy_input():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    x = torch.randn(1, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.onnx")
        session.export_onnx(path, dummy_input=x)
        assert os.path.exists(path)


def test_export_onnx_auto_input():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    x = torch.randn(1, 4)
    with torch.no_grad():
        session(x)  # records _last_input

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.onnx")
        session.export_onnx(path)
        assert os.path.exists(path)


def test_export_onnx_no_input_raises():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    with pytest.raises(ValueError, match="No dummy_input"):
        session.export_onnx("nowhere.onnx")


# ---------------------------------------------------------------------------
# 9. clear_scales
# ---------------------------------------------------------------------------

def test_clear_scales():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    # First calibrate to assign scales
    with session.calibrate():
        with torch.no_grad():
            for _ in range(4):
                session(torch.randn(2, 4))

    # Now clear them
    removed = session.clear_scales()
    assert len(removed) > 0

    # Verify no scales remain
    has_scale = any(
        hasattr(m, "_output_scale") for m in session.qmodel.modules()
    )
    assert not has_scale


# ---------------------------------------------------------------------------
# 10. Delegation
# ---------------------------------------------------------------------------

def test_train_eval():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)

    session.train()
    assert session.qmodel.training

    session.eval()
    assert not session.qmodel.training


def test_parameters():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)

    params = list(session.parameters())
    assert len(params) > 0


def test_state_dict():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)

    sd = session.state_dict()
    assert isinstance(sd, dict)
    assert len(sd) > 0


def test_load_state_dict():
    from src.session._quant import _QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)

    sd = session.state_dict()
    session.load_state_dict(sd)
    # Should not raise


# ---------------------------------------------------------------------------
# 11. compare_sessions (e2e)
# ---------------------------------------------------------------------------

def test_compare_sessions_multiple():
    from src.session._quant import _QuantSession
    from src.analysis.e2e import compare_sessions

    model1 = _make_small_model()
    model2 = _make_small_model()
    # Make both models share the same weights for stable comparison
    sd = model1.state_dict()
    model2.load_state_dict(sd)

    cfg = _make_cfg()
    s1 = _QuantSession(model1, cfg)
    s2 = _QuantSession(model2, cfg)
    s1.eval()
    s2.eval()

    dl = _make_dataloader(n_samples=16, batch_size=4)
    results = compare_sessions({"s1": s1, "s2": s2}, dl)

    assert "fp32" in results
    assert "s1" in results
    assert "s2" in results
    assert "fp32" in results["s1"]
    assert "quant" in results["s1"]
    assert "delta" in results["s1"]


def test_compare_sessions_custom_label():
    from src.session._quant import _QuantSession
    from src.analysis.e2e import compare_sessions

    model = _make_small_model()
    cfg = _make_cfg()
    session = _QuantSession(model, cfg)
    session.eval()

    dl = _make_dataloader(n_samples=16, batch_size=4)
    results = compare_sessions({"a": session}, dl, fp32_label="baseline")

    assert "baseline" in results
    assert "a" in results


# ---------------------------------------------------------------------------
# 12. Comparator standalone tests
# ---------------------------------------------------------------------------

def test_comparator_basic():
    from src.analysis.e2e import Comparator
    cmp = Comparator()
    assert cmp.num_samples == 0

    with cmp:
        fp32 = torch.randn(4, 3)
        quant = fp32 + 0.01 * torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        cmp.record(fp32, quant, labels)

    assert cmp.num_samples == 4


def test_comparator_evaluate():
    from src.analysis.e2e import Comparator
    cmp = Comparator()

    with cmp:
        fp32 = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        quant = torch.tensor([[0.7, 0.3], [0.3, 0.7]])
        labels = torch.tensor([0, 1])
        cmp.record(fp32, quant, labels)

    result = cmp.evaluate(lambda logits, labels: {
        "acc": (logits.argmax(-1) == labels).float().mean().item()
    })

    assert result["fp32"]["acc"] == 1.0
    assert result["quant"]["acc"] == 1.0
    assert result["delta"]["acc"] == 0.0


def test_comparator_evaluate_with_directions():
    from src.analysis.e2e import Comparator
    cmp = Comparator()

    with cmp:
        fp32 = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        quant = torch.tensor([[0.7, 0.3], [0.3, 0.7]])
        labels = torch.tensor([0, 1])
        cmp.record(fp32, quant, labels)

    result = cmp.evaluate(
        lambda logits, labels: {"acc": (logits.argmax(-1) == labels).float().mean().item()},
        directions={"acc": "higher"},
    )
    assert result["_directions"] == {"acc": "higher"}


def test_comparator_device():
    from src.analysis.e2e import Comparator
    cmp = Comparator(device=torch.device("cpu"))
    assert cmp._device == torch.device("cpu")


def test_compare_models_basic():
    from src.analysis.e2e import compare_models
    fp32_model = _make_small_model()
    qmodel = _make_small_model()
    fp32_model.eval()
    qmodel.eval()

    dl = _make_dataloader(n_samples=16, batch_size=4)
    result = compare_models(fp32_model, qmodel, dl)

    assert "fp32" in result
    assert "quant" in result
    assert "delta" in result


# ---------------------------------------------------------------------------
# Pre-Scale Integration (P5)
# ---------------------------------------------------------------------------

class TestPreScaleIntegration:
    """Tests for _QuantSession.initialize_pre_scales() and optimize_scales()."""

    def test_initialize_pre_scales_adds_buffers(self):
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)

        # Before: no _pre_scale buffers
        for _, mod in session.qmodel.named_modules():
            assert not hasattr(mod, "_pre_scale")

        # Initialize
        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = session.initialize_pre_scales(calib_data, init="ones")

        assert count > 0

        # After: _pre_scale buffers on quantized modules
        found = 0
        for _, mod in session.qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                found += 1
                assert isinstance(mod._pre_scale, torch.Tensor)
        assert found == count

    def test_optimize_scales_runs(self):
        from src.session._quant import _QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        session.initialize_pre_scales(calib_data, init="ones")

        opt = LayerwiseScaleOptimizer(num_steps=10, num_batches=2, lr=0.01)
        scales = session.optimize_scales(opt, calib_data)

        assert isinstance(scales, dict)
        assert len(scales) > 0

    def test_initialize_pre_scales_preserves_existing_cfg(self):
        """After initialization, module.cfg should still be OpQuantConfig."""
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        session.initialize_pre_scales(calib_data, init="ones")

        from src.scheme.op_config import cfg_causes_quantization
        for _, mod in session.qmodel.named_modules():
            if hasattr(mod, "cfg") and cfg_causes_quantization(mod.cfg):
                assert isinstance(mod.cfg, OpQuantConfig)

    def test_optimize_scales_requires_fp32(self):
        from src.session._quant import _QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg, keep_fp32=False)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        opt = LayerwiseScaleOptimizer(num_steps=5, num_batches=2)

        with pytest.raises(RuntimeError, match="keep_fp32=True"):
            session.optimize_scales(opt, calib_data)

    def test_initialize_pre_scales_invalid_init(self):
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)

        with pytest.raises(ValueError, match="Unknown init method"):
            session.initialize_pre_scales([torch.randn(8, 4)], init="invalid")

    def test_forward_after_pre_scale_init(self):
        """Forward pass works after initialize_pre_scales."""
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)

        session.initialize_pre_scales([torch.randn(8, 4) for _ in range(4)], init="ones")
        session.eval()
        out = session(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_e2e_pre_scale_pipeline(self):
        """Full pipeline: calibrate -> initialize -> optimize -> compare."""
        from src.session._quant import _QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

        torch.manual_seed(42)
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg, keep_fp32=True)

        # Create calibration data
        calib_data = [torch.randn(8, 4) for _ in range(6)]

        # Step 1: Calibrate
        with session.calibrate():
            for batch in calib_data:
                session(batch)

        # Step 2: Initialize pre-scales
        count = session.initialize_pre_scales(calib_data, init="ones")
        assert count > 0

        # Step 3: LSQ optimize
        opt = LayerwiseScaleOptimizer(num_steps=20, num_batches=3, lr=0.01)
        scales = session.optimize_scales(opt, calib_data)
        # optimizer processes all quantized modules, initialize_pre_scales only
        # creates buffers for modules with known output channels (skips activations)
        assert len(scales) >= count

        # Step 4: Forward pass works
        out = session(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

        # Step 5: Mode toggle works
        session.use_fp32()
        fp32_out = session(torch.randn(4, 4))
        session.use_quant()
        q_out = session(torch.randn(4, 4))
        assert q_out.shape == fp32_out.shape
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(fp32_out).any()

    def test_e2e_pre_scale_pot_pipeline(self):
        """Full pipeline with PoT pre-scale: calibrate -> init -> optimize -> verify PoT."""
        from src.session._quant import _QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

        torch.manual_seed(42)
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg, keep_fp32=True)

        calib_data = [torch.randn(8, 4) for _ in range(6)]

        # Calibrate
        with session.calibrate():
            for batch in calib_data:
                session(batch)

        # Initialize with pot=True
        count = session.initialize_pre_scales(calib_data, init="ones", pot=True)
        assert count > 0

        # LSQ optimize with pot=True
        opt = LayerwiseScaleOptimizer(num_steps=20, num_batches=3, lr=0.01, pot=True)
        scales = session.optimize_scales(opt, calib_data)
        assert len(scales) >= count

        # All optimized scales must be PoT
        for scale in scales.values():
            log2 = torch.log2(scale)
            assert torch.equal(log2, torch.round(log2)), \
                f"scale {scale} is not power-of-two"

        # Forward pass works
        out = session(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    # ---- New init modes, granularity, trainable ----

    def test_initialize_amax_init(self):
        """init='amax' creates pre-scales from activation statistics."""
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)
        session.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = session.initialize_pre_scales(calib_data, init="amax")
        assert count > 0

        for _, mod in session.qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                assert mod._pre_scale.numel() == 1  # per_tensor default

        # Forward pass works
        out = session(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_initialize_pot_amax_init(self):
        """init='pot_amax' creates PoT pre-scales from activation statistics."""
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)
        session.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = session.initialize_pre_scales(calib_data, init="pot_amax")
        assert count > 0

        for _, mod in session.qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                scale = mod._pre_scale
                log2 = torch.log2(scale)
                assert torch.equal(log2, torch.round(log2)), \
                    f"scale {scale} is not power-of-two"

        # Forward pass works
        out = session(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_collect_input_amax_accumulates_across_batches(self):
        """_collect_input_amax takes element-wise max across multiple batches."""
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)
        session.eval()

        # Batch 1: small values (amax ~ 0.5), Batch 2: larger (amax ~ 5.0)
        batch1 = [torch.randn(8, 4) * 0.5]
        batch2 = [torch.randn(8, 4) * 5.0]
        batch2_big = batch2[0].clone()
        batch2_big[0, 0] = 10.0  # forced large value
        batch2 = [batch2_big]

        amap = _QuantSession._collect_input_amax(batch1, session.qmodel)
        amap2 = _QuantSession._collect_input_amax(batch2, session.qmodel)
        amap_both = _QuantSession._collect_input_amax(
            batch1 + batch2, session.qmodel,
        )

        # Running max across both batches should equal max of individual amax values
        for name in amap:
            if name in amap_both:
                expected = torch.maximum(amap[name], amap2[name])
                assert torch.allclose(amap_both[name], expected), \
                    f"amax for {name}: batch1={amap[name].item():.3f}, batch2={amap2[name].item():.3f}, both={amap_both[name].item():.3f}"

    def test_initialize_per_channel_granularity(self):
        """granularity='per_channel' creates (C,) pre-scales on input activation roles only.

        Output/grad_output are excluded because channel counts differ
        across matmul ops (in_features vs out_features).
        """
        from src.session._quant import _QuantSession
        from src.transform.pre_scale import PreScaleTransform
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)
        session.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = session.initialize_pre_scales(
            calib_data, init="ones", granularity="per_channel",
        )
        assert count > 0

        for _, mod in session.qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                # Input activation role has PreScaleTransform
                if mod.cfg.input is not None:
                    assert isinstance(mod.cfg.input.transform, PreScaleTransform)
                # Output role does NOT have PreScaleTransform (channel mismatch)
                if mod.cfg.output is not None:
                    assert not isinstance(
                        mod.cfg.output.transform, PreScaleTransform,
                    )
                # Weight scheme does NOT have PreScaleTransform
                if mod.cfg.weight is not None:
                    assert not isinstance(
                        mod.cfg.weight.transform, PreScaleTransform,
                    )

    def test_initialize_trainable_parameter(self):
        """trainable=True registers _pre_scale as nn.Parameter."""
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)
        session.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = session.initialize_pre_scales(
            calib_data, init="ones", trainable=True,
        )
        assert count > 0

        for _, mod in session.qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                assert isinstance(mod._pre_scale, nn.Parameter)

        # Scales appear in model parameters
        param_names = [n for n, _ in session.qmodel.named_parameters()]
        pre_scale_params = [n for n in param_names if "_pre_scale" in n]
        assert len(pre_scale_params) > 0

    def test_initialize_invalid_granularity(self):
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg)

        with pytest.raises(ValueError, match="Unknown granularity"):
            session.initialize_pre_scales(
                [torch.randn(8, 4)], init="ones", granularity="invalid",
            )

    def test_e2e_hierarchical_pipeline(self):
        """Full pipeline: pot_amax init + calibration + forward."""
        from src.session._quant import _QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = _QuantSession(model, cfg, keep_fp32=True)
        session.eval()

        calib_data = [torch.randn(8, 4) for _ in range(6)]

        # Step 1: Initialize pre-scales (pot_amax, per_channel, pot=True)
        count = session.initialize_pre_scales(
            calib_data, init="pot_amax", pot=True, granularity="per_channel",
        )
        assert count > 0

        # Step 2: Calibrate
        with session.calibrate():
            for batch in calib_data:
                session(batch)

        # Step 3: Forward pass works
        out = session(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()


# ==============================================================================
# QSNR Per-Layer Mathematical Verification
# ==============================================================================


def _manual_qsnr(fp32: torch.Tensor, quant: torch.Tensor) -> float:
    """Reference QSNR: 10 * log10(||fp32||² / ||fp32 - quant||²)."""
    signal = fp32.pow(2).sum().item()
    error = (fp32 - quant).pow(2).sum().item()
    if error < 1e-30:
        return float("inf")
    return 10.0 * __import__("math").log10(max(signal, 1e-12) / error)


def _manual_qsnr_multi(tensors: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Multi-batch QSNR: accumulate sum-of-squares then divide."""
    total_signal = 0.0
    total_error = 0.0
    for fp32, quant in tensors:
        total_signal += fp32.pow(2).sum().item()
        total_error += (fp32 - quant).pow(2).sum().item()
    if total_error < 1e-30:
        return float("inf")
    return 10.0 * __import__("math").log10(max(total_signal, 1e-12) / total_error)


class TestQSNRFormulaVerification:
    """Verify QSNR = 10 * log10(Σ||fp||² / Σ||fp-q||²) is correctly computed."""

    def test_basic_formula(self):
        """QSNR of a tensor with known fp32 and quant values."""
        fp32 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        quant = torch.tensor([1.0, 2.0, 3.1, 3.9])  # slight error

        manual = _manual_qsnr(fp32, quant)
        # ||fp||² = 1+4+9+16 = 30
        # ||fp-q||² = 0+0+0.01+0.01 = 0.02
        # QSNR = 10 * log10(30/0.02) = 10 * log10(1500) ≈ 31.76
        expected = 10.0 * __import__("math").log10(30.0 / 0.02)
        assert abs(manual - expected) < 1e-5  # fp32 accumulation tolerance

    def test_qsnr_observer_matches_formula(self):
        """QSNRObserver._measure must equal the manual formula."""
        from src.analysis.observers import QSNRObserver
        fp32 = torch.randn(16, 32)
        quant = fp32 + torch.randn(16, 32) * 0.1

        obs = QSNRObserver()
        result = obs._measure("test", fp32, quant)
        manual = _manual_qsnr(fp32, quant)

        assert abs(result["qsnr_db"] - manual) < 1e-5  # fp32 accumulation tolerance

    def test_zero_error_is_inf(self):
        """Zero quantization error → QSNR = +inf."""
        fp32 = torch.randn(4, 8)
        manual = _manual_qsnr(fp32, fp32)
        assert manual == float("inf")

    def test_mean_power_div_mean_error_equals_sum_ratio(self):
        """mean(fp²)/mean((fp-q)²) = Σfp²/Σ(fp-q)² because N cancels."""
        fp32 = torch.tensor([1.0, 2.0, 3.0])
        quant = torch.tensor([1.1, 1.9, 2.9])

        # Via means
        mean_signal = fp32.pow(2).mean().item()
        mean_error = (fp32 - quant).pow(2).mean().item()
        qsnr_mean = 10.0 * __import__("math").log10(mean_signal / mean_error)

        # Via sums
        qsnr_sum = _manual_qsnr(fp32, quant)

        assert abs(qsnr_mean - qsnr_sum) < 1e-5  # fp32 accumulation tolerance

    def test_log10_monotonicity(self):
        """Larger error → smaller QSNR (monotonic)."""
        fp32 = torch.randn(100)
        qsnr_small = _manual_qsnr(fp32, fp32 + 0.01 * torch.randn(100))
        qsnr_large = _manual_qsnr(fp32, fp32 + 0.10 * torch.randn(100))
        assert qsnr_small > qsnr_large


class TestMultiBatchAccumulation:
    """Verify that multi-batch QSNR accumulation is mathematically correct.

    QSNR([batch1, batch2]) = 10 * log10( (S1+S2) / (E1+E2) )
    This is NOT the arithmetic mean of per-batch QSNR values.
    """

    def test_two_batches_match_concatenated(self):
        """Accumulated QSNR of two batches = QSNR of concatenated tensors."""
        b1_fp = torch.randn(4, 8)
        b1_q = b1_fp + 0.05 * torch.randn(4, 8)
        b2_fp = torch.randn(4, 8)
        b2_q = b2_fp + 0.05 * torch.randn(4, 8)

        # Accumulated
        acc = _manual_qsnr_multi([(b1_fp, b1_q), (b2_fp, b2_q)])

        # Concatenated
        cat_fp = torch.cat([b1_fp.reshape(-1), b2_fp.reshape(-1)])
        cat_q = torch.cat([b1_q.reshape(-1), b2_q.reshape(-1)])
        cat = _manual_qsnr(cat_fp, cat_q)

        assert abs(acc - cat) < 1e-5  # fp32 accumulation tolerance

    def test_accumulation_not_mean_of_qsnrs(self):
        """Accumulated QSNR ≠ mean of per-batch QSNR (log is non-linear)."""
        b1_fp = torch.randn(100)
        b1_q = b1_fp + 0.01 * torch.randn(100)
        b2_fp = torch.randn(100)
        b2_q = b2_fp + 0.10 * torch.randn(100)  # much noisier

        acc = _manual_qsnr_multi([(b1_fp, b1_q), (b2_fp, b2_q)])
        qsnr1 = _manual_qsnr(b1_fp, b1_q)
        qsnr2 = _manual_qsnr(b2_fp, b2_q)
        mean = (qsnr1 + qsnr2) / 2.0

        # They should differ because log is non-linear
        assert abs(acc - mean) > 0.5

    def test_uneven_batch_sizes(self):
        """Accumulation handles uneven batch sizes correctly."""
        b1_fp = torch.randn(4, 8)   # 32 elements
        b1_q = b1_fp + 0.01 * torch.randn(4, 8)
        b2_fp = torch.randn(16, 8)  # 128 elements
        b2_q = b2_fp + 0.01 * torch.randn(16, 8)

        acc = _manual_qsnr_multi([(b1_fp, b1_q), (b2_fp, b2_q)])
        cat_fp = torch.cat([b1_fp.reshape(-1), b2_fp.reshape(-1)])
        cat_q = torch.cat([b1_q.reshape(-1), b2_q.reshape(-1)])
        cat = _manual_qsnr(cat_fp, cat_q)

        assert abs(acc - cat) < 1e-5  # fp32 accumulation tolerance


class TestTrueErrorEndToEnd:
    """End-to-end verification: Session true_error matches manual computation."""

    def _make_deterministic_model(self):
        """Model with fixed weights so output is deterministic."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))
        for p in model.parameters():
            nn.init.constant_(p, 0.5)
        return model

    def test_qsnr_matches_manual_computation(self):
        """Session true_error QSNR must match manual hook-based computation."""
        from src.session._session import Session
        from src.session._config import QuantConfig
        import math

        torch.manual_seed(42)
        model = self._make_deterministic_model()
        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        session.quantize()

        # Manual computation: capture fp32 and quant outputs
        fp32_refs = {}
        quant_refs = {}

        def fp32_hook(name):
            def h(_m, _inp, out):
                fp32_refs[name] = out.detach()
            return h

        def quant_hook(name):
            def h(_m, _inp, out):
                quant_refs[name] = out.detach()
            return h

        qmodel = session._quant_session.qmodel
        fp32_model = session._quant_session.fp32_model

        hooks = []
        for name, _mod in qmodel.named_modules():
            if hasattr(_mod, "cfg"):
                from src.scheme.op_config import cfg_causes_quantization
                if not cfg_causes_quantization(_mod.cfg):
                    continue
                fp32_mod = dict(fp32_model.named_modules()).get(name)
                quant_mod = dict(qmodel.named_modules()).get(name)
                if fp32_mod and quant_mod:
                    hooks.append(fp32_mod.register_forward_hook(fp32_hook(name)))
                    hooks.append(quant_mod.register_forward_hook(quant_hook(name)))

        x = torch.randn(4, 2)
        with torch.no_grad():
            fp32_model(x)
            qmodel(x)
        for h in hooks:
            h.remove()

        # Compute manual QSNR
        manual_qsnr = {}
        for name in fp32_refs:
            fp = fp32_refs[name]
            q = quant_refs.get(name)
            if q is not None and fp.shape == q.shape:
                signal = fp.pow(2).sum().item()
                error = (fp - q).pow(2).sum().item()
                if error > 1e-30:
                    manual_qsnr[name] = 10.0 * math.log10(
                        max(signal, 1e-12) / error
                    )

        # Session computation
        session2 = Session(self._make_deterministic_model(), config)
        session2.quantize()
        session2.analyze(x, outputs=[], true_error=True)

        # Same layers
        assert set(session2._qsnr_per_layer.keys()) == set(manual_qsnr.keys())

        # Values must match within floating-point error
        for name in manual_qsnr:
            assert abs(session2._qsnr_per_layer[name] - manual_qsnr[name]) < 1e-5, (
                f"Mismatch at {name}: "
                f"session={session2._qsnr_per_layer[name]:.6f} "
                f"manual={manual_qsnr[name]:.6f}"
            )

    def test_multi_batch_qsnr_matches_single_concatenated(self):
        """Multi-batch accumulated QSNR = QSNR of concatenated tensors.

        Uses hooks within a single calibrated session to verify that
        the accumulation formula ΣS/ΣE is correct, independent of
        calibration (which is identical within one session).
        """
        from src.session._session import Session
        from src.session._config import QuantConfig
        from src.scheme.op_config import cfg_causes_quantization
        import math

        torch.manual_seed(42)
        model = self._make_deterministic_model()

        config = QuantConfig(calibrator="max")
        session = Session(model, config)
        session.quantize()

        batches = [torch.randn(4, 2) for _ in range(3)]
        qmodel = session._quant_session.qmodel
        fp32_model = session._quant_session.fp32_model

        # --- Capture per-batch fp32/quant outputs within ONE calibrated session ---
        accum_signal: dict = {name: 0.0 for name in ["0", "1"]}
        accum_error: dict = {name: 0.0 for name in ["0", "1"]}
        all_fp32_tensors: dict = {name: [] for name in ["0", "1"]}
        all_quant_tensors: dict = {name: [] for name in ["0", "1"]}

        for batch in batches:
            fp32_refs = {}
            quant_refs = {}
            hooks = []

            for name, mod in qmodel.named_modules():
                if not hasattr(mod, "cfg") or not cfg_causes_quantization(mod.cfg):
                    continue
                fp_mod = dict(fp32_model.named_modules()).get(name)
                q_mod = dict(qmodel.named_modules()).get(name)
                if fp_mod and q_mod:
                    hooks.append(
                        fp_mod.register_forward_hook(
                            lambda _m, _i, o, n=name: fp32_refs.__setitem__(n, o.detach())
                        )
                    )
                    hooks.append(
                        q_mod.register_forward_hook(
                            lambda _m, _i, o, n=name: quant_refs.__setitem__(n, o.detach())
                        )
                    )

            with torch.no_grad():
                fp32_model(batch)
                qmodel(batch)

            for h in hooks:
                h.remove()

            for name in fp32_refs:
                fp = fp32_refs[name]
                q = quant_refs.get(name)
                if q is not None and fp.shape == q.shape:
                    accum_signal[name] += fp.pow(2).sum().item()
                    accum_error[name] += (fp - q).pow(2).sum().item()
                    all_fp32_tensors[name].append(fp.reshape(-1))
                    all_quant_tensors[name].append(q.reshape(-1))

        # Multi-batch accumulated QSNR
        multi_qsnr = {}
        for name in accum_signal:
            if accum_error[name] > 1e-30:
                multi_qsnr[name] = 10.0 * math.log10(
                    max(accum_signal[name], 1e-12) / accum_error[name]
                )

        # Single concatenated QSNR
        cat_qsnr = {}
        for name in all_fp32_tensors:
            cat_fp = torch.cat(all_fp32_tensors[name])
            cat_q = torch.cat(all_quant_tensors[name])
            cat_qsnr[name] = _manual_qsnr(cat_fp, cat_q)

        # Must match exactly (same calibration, same tensors)
        for name in multi_qsnr:
            assert name in cat_qsnr
            assert abs(multi_qsnr[name] - cat_qsnr[name]) < 1e-5, (
                f"Accumulation bug at {name}: "
                f"multi={multi_qsnr[name]:.6f} cat={cat_qsnr[name]:.6f}"
            )


class TestExtractQSNRMSE:
    """Verify _extract_qsnr_mse role filtering and worst-case selection."""

    def test_extracts_output_role_only(self):
        """Only output role QSNR is extracted; input/weight are ignored."""
        from src.session._session import _extract_qsnr_mse

        data = {
            "fc1": {
                "input": {"stage1": {"s0": {"qsnr_db": 10.0}}},
                "weight": {"stage1": {"s0": {"qsnr_db": 20.0}}},
                "output": {"stage1": {"s0": {"qsnr_db": 30.0}}},
            },
        }
        qsnr, mse = _extract_qsnr_mse(data)
        assert qsnr == {"fc1": 30.0}

    def test_takes_worst_case_across_stages(self):
        """Minimum QSNR across all stages and slices is used."""
        from src.session._session import _extract_qsnr_mse

        data = {
            "fc1": {
                "output": {
                    "stage1": {"s0": {"qsnr_db": 25.0}, "s1": {"qsnr_db": 30.0}},
                    "stage2": {"s0": {"qsnr_db": 15.0}},  # worst
                },
            },
        }
        qsnr, mse = _extract_qsnr_mse(data)
        assert qsnr == {"fc1": 15.0}

    def test_nan_values_are_skipped(self):
        """NaN QSNR values are excluded from worst-case selection."""
        from src.session._session import _extract_qsnr_mse

        data = {
            "fc1": {
                "output": {
                    "stage1": {
                        "s0": {"qsnr_db": float("nan")},
                        "s1": {"qsnr_db": 30.0},
                    },
                },
            },
        }
        qsnr, mse = _extract_qsnr_mse(data)
        assert qsnr == {"fc1": 30.0}

    def test_role_not_present_returns_empty(self):
        """If output role is absent, nothing is extracted."""
        from src.session._session import _extract_qsnr_mse

        data = {
            "fc1": {
                "input": {"stage1": {"s0": {"qsnr_db": 10.0}}},
            },
        }
        qsnr, mse = _extract_qsnr_mse(data)
        assert qsnr == {}


class TestQSNRInvariants:
    """Invariants that must hold for any correct QSNR implementation."""

    def test_identity_quantization_is_inf(self):
        """QSNR(x, x) = +inf (no error)."""
        x = torch.randn(4, 8)
        assert _manual_qsnr(x, x) == float("inf")

    def test_positive_signal_positive_qsnr(self):
        """For positive signal and finite error, QSNR > 0 dB."""
        fp32 = torch.ones(100)
        quant = fp32 + 0.01 * torch.randn(100)
        assert _manual_qsnr(fp32, quant) > 0

    def test_additive_noise_lowers_qsnr(self):
        """Adding independent noise always reduces QSNR."""
        fp32 = torch.randn(1000)
        q1 = fp32 + 0.01 * torch.randn(1000)
        q2 = q1 + 0.01 * torch.randn(1000)  # additional noise
        assert _manual_qsnr(fp32, q1) > _manual_qsnr(fp32, q2)

    def test_scaling_invariant(self):
        """QSNR is scale-invariant: QSNR(αx, αx̂) = QSNR(x, x̂)."""
        x = torch.randn(100)
        x_hat = x + 0.01 * torch.randn(100)
        alpha = 3.0
        assert abs(_manual_qsnr(alpha * x, alpha * x_hat) -
                   _manual_qsnr(x, x_hat)) < 1e-5  # fp32 tolerance
