"""
Tests for QuantSession unified API and e2e comparison tools.
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
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)

    assert session.cfg is cfg
    assert isinstance(session.calibrator, MaxScaleStrategy)
    assert len(session.observers) > 0
    assert session.mode == "quant"
    assert session.fp32_model is not None
    assert session.qmodel is not None
    # fp32_model is a deep copy, not the same object
    assert session.fp32_model is not model


def test_session_no_fp32():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg, keep_fp32=False)

    assert session.fp32_model is None


def test_session_custom_calibrator():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    cal = PercentileScaleStrategy(q=95.0)
    session = QuantSession(model, cfg, calibrator=cal)

    assert session.calibrator is cal


# ---------------------------------------------------------------------------
# 2. Mode switching
# ---------------------------------------------------------------------------

def test_mode_switching():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)

    assert session.mode == "quant"
    session.use_fp32()
    assert session.mode == "fp32"
    session.use_quant()
    assert session.mode == "quant"


def test_use_fp32_without_fp32_raises():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg, keep_fp32=False)

    with pytest.raises(RuntimeError, match="fp32_model not available"):
        session.use_fp32()


# ---------------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------------

def test_call_in_quant_mode():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    x = torch.randn(4, 4)
    with torch.no_grad():
        out = session(x)
    assert out.shape == (4, 3)


def test_call_in_fp32_mode():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    x = torch.randn(4, 4)
    with torch.no_grad():
        session.use_fp32()
        out = session(x)
    assert out.shape == (4, 3)


def test_call_records_last_input():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    x = torch.randn(4, 4)
    with torch.no_grad():
        session(x)
    assert session._last_input is not None


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------

def test_calibrate_returns_calibration_session():
    from src.session import QuantSession
    from src.calibration.pipeline import CalibrationSession as CS
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    cs = session.calibrate()
    assert isinstance(cs, CS)


def test_calibrate_context_assigns_scales():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
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
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
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
    from src.session import QuantSession
    from src.analysis.context import AnalysisContext
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    ctx = session.analyze()
    assert isinstance(ctx, AnalysisContext)


def test_analyze_collects_metrics():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    with session.analyze() as ctx:
        with torch.no_grad():
            for _ in range(4):
                session(torch.randn(2, 4))

    report = ctx.report()
    # Should have at least one layer
    assert len(report.keys()) > 0


def test_analyze_with_custom_observers():
    from src.session import QuantSession
    from src.analysis.observers import MSEObserver
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    with session.analyze(observers=[MSEObserver()]) as ctx:
        with torch.no_grad():
            session(torch.randn(2, 4))

    report = ctx.report()
    assert len(report.keys()) > 0


# ---------------------------------------------------------------------------
# 6. Comparator
# ---------------------------------------------------------------------------

def test_comparator_returns_comparator():
    from src.session import QuantSession
    from src.analysis.e2e import Comparator
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)

    cmp = session.comparator()
    assert isinstance(cmp, Comparator)


def test_comparator_manual_collection():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
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
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    dl = _make_dataloader(n_samples=16, batch_size=4)
    result = session.compare(dl)

    assert "fp32" in result
    assert "quant" in result
    assert "delta" in result
    assert "accuracy" in result["fp32"]


def test_compare_with_custom_eval_fn():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    def my_eval(logits, labels):
        return {"top1": (logits.argmax(-1) == labels).float().mean().item()}

    dl = _make_dataloader(n_samples=16, batch_size=4)
    result = session.compare(dl, eval_fn=my_eval, directions={"top1": "higher"})

    assert "top1" in result["fp32"]
    assert result.get("_directions") == {"top1": "higher"}


def test_compare_without_fp32_raises():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg, keep_fp32=False)
    session.eval()

    dl = _make_dataloader(n_samples=4, batch_size=2)
    with pytest.raises(RuntimeError, match="fp32_model not available"):
        session.compare(dl)


# ---------------------------------------------------------------------------
# 8. ONNX Export
# ---------------------------------------------------------------------------

def test_export_onnx_with_dummy_input():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    x = torch.randn(1, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.onnx")
        session.export_onnx(path, dummy_input=x)
        assert os.path.exists(path)


def test_export_onnx_auto_input():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    x = torch.randn(1, 4)
    with torch.no_grad():
        session(x)  # records _last_input

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.onnx")
        session.export_onnx(path)
        assert os.path.exists(path)


def test_export_onnx_no_input_raises():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
    session.eval()

    with pytest.raises(ValueError, match="No dummy_input"):
        session.export_onnx("nowhere.onnx")


# ---------------------------------------------------------------------------
# 9. clear_scales
# ---------------------------------------------------------------------------

def test_clear_scales():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
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
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)

    session.train()
    assert session.qmodel.training

    session.eval()
    assert not session.qmodel.training


def test_parameters():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)

    params = list(session.parameters())
    assert len(params) > 0


def test_state_dict():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)

    sd = session.state_dict()
    assert isinstance(sd, dict)
    assert len(sd) > 0


def test_load_state_dict():
    from src.session import QuantSession
    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)

    sd = session.state_dict()
    session.load_state_dict(sd)
    # Should not raise


# ---------------------------------------------------------------------------
# 11. compare_sessions (e2e)
# ---------------------------------------------------------------------------

def test_compare_sessions_multiple():
    from src.session import QuantSession
    from src.analysis.e2e import compare_sessions

    model1 = _make_small_model()
    model2 = _make_small_model()
    # Make both models share the same weights for stable comparison
    sd = model1.state_dict()
    model2.load_state_dict(sd)

    cfg = _make_cfg()
    s1 = QuantSession(model1, cfg)
    s2 = QuantSession(model2, cfg)
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
    from src.session import QuantSession
    from src.analysis.e2e import compare_sessions

    model = _make_small_model()
    cfg = _make_cfg()
    session = QuantSession(model, cfg)
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
    """Tests for QuantSession.initialize_pre_scales() and optimize_scales()."""

    def test_initialize_pre_scales_adds_buffers(self):
        from src.session import QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg)

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
        from src.session import QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        session.initialize_pre_scales(calib_data, init="ones")

        opt = LayerwiseScaleOptimizer(num_steps=10, num_batches=2, lr=0.01)
        scales = session.optimize_scales(opt, calib_data)

        assert isinstance(scales, dict)
        assert len(scales) > 0

    def test_initialize_pre_scales_preserves_existing_cfg(self):
        """After initialization, module.cfg should still be OpQuantConfig."""
        from src.session import QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        session.initialize_pre_scales(calib_data, init="ones")

        for _, mod in session.qmodel.named_modules():
            if hasattr(mod, "cfg") and not getattr(mod, "_is_passthrough", False):
                assert isinstance(mod.cfg, OpQuantConfig)

    def test_optimize_scales_requires_fp32(self):
        from src.session import QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg, keep_fp32=False)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        opt = LayerwiseScaleOptimizer(num_steps=5, num_batches=2)

        with pytest.raises(RuntimeError, match="keep_fp32=True"):
            session.optimize_scales(opt, calib_data)

    def test_initialize_pre_scales_invalid_init(self):
        from src.session import QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg)

        with pytest.raises(ValueError, match="Unknown init method"):
            session.initialize_pre_scales([torch.randn(8, 4)], init="invalid")

    def test_forward_after_pre_scale_init(self):
        """Forward pass works after initialize_pre_scales."""
        from src.session import QuantSession
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg)

        session.initialize_pre_scales([torch.randn(8, 4) for _ in range(4)], init="ones")
        session.eval()
        out = session(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_e2e_pre_scale_pipeline(self):
        """Full pipeline: calibrate -> initialize -> optimize -> compare."""
        from src.session import QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

        torch.manual_seed(42)
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg, keep_fp32=True)

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
        from src.session import QuantSession
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

        torch.manual_seed(42)
        model = _make_small_model()
        cfg = _make_cfg()
        session = QuantSession(model, cfg, keep_fp32=True)

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
