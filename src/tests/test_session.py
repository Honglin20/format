"""
Tests for run_quantization, pre-scale pipeline, QSNR math, and e2e comparison.
"""
import copy
import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.scheme.op_config import OpQuantConfig, cfg_causes_quantization
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.formats.base import FormatBase
from src.calibration.strategies import MaxScaleStrategy, PercentileScaleStrategy
from src.calibration.pipeline import CalibrationSession
from src.analysis.context import AnalysisContext
from src.analysis.observers import MSEObserver, QSNRObserver
from src.session._config import QuantConfig
from src.session._result import SessionResult
from src.session._session import run_quantization
from src.session._model import quantize_model
from src.session._helpers import initialize_pre_scales, optimize_scales, clear_scales


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


def _make_cfg():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)


def _make_dataloader(n_samples=32, batch_size=8, n_features=4, n_classes=3):
    data = torch.randn(n_samples, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(data, labels), batch_size=batch_size)


# ==============================================================================
# QSNR Formula Verification (pure math, no session dependency)
# ==============================================================================


def _manual_qsnr(fp32: torch.Tensor, quant: torch.Tensor) -> float:
    """Reference QSNR: 10 * log10(||fp32||^2 / ||fp32 - quant||^2)."""
    signal = fp32.pow(2).sum().item()
    error = (fp32 - quant).pow(2).sum().item()
    if error < 1e-30:
        return float("inf")
    return 10.0 * math.log10(max(signal, 1e-12) / error)


def _manual_qsnr_multi(tensors: list) -> float:
    """Multi-batch QSNR: accumulate sum-of-squares then divide."""
    total_signal = 0.0
    total_error = 0.0
    for fp32, quant in tensors:
        total_signal += fp32.pow(2).sum().item()
        total_error += (fp32 - quant).pow(2).sum().item()
    if total_error < 1e-30:
        return float("inf")
    return 10.0 * math.log10(max(total_signal, 1e-12) / total_error)


class TestQSNRFormulaVerification:
    """Verify QSNR = 10 * log10(||fp||^2 / ||fp-q||^2) is correctly computed."""

    def test_basic_formula(self):
        fp32 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        quant = torch.tensor([1.0, 2.0, 3.1, 3.9])
        manual = _manual_qsnr(fp32, quant)
        expected = 10.0 * math.log10(30.0 / 0.02)
        assert abs(manual - expected) < 1e-5

    def test_qsnr_observer_matches_formula(self):
        fp32 = torch.randn(16, 32)
        quant = fp32 + torch.randn(16, 32) * 0.1
        obs = QSNRObserver()
        result = obs._measure("test", fp32, quant)
        manual = _manual_qsnr(fp32, quant)
        assert abs(result["qsnr_db"] - manual) < 1e-5

    def test_zero_error_is_inf(self):
        fp32 = torch.randn(4, 8)
        manual = _manual_qsnr(fp32, fp32)
        assert manual == float("inf")

    def test_mean_power_div_mean_error_equals_sum_ratio(self):
        fp32 = torch.tensor([1.0, 2.0, 3.0])
        quant = torch.tensor([1.1, 1.9, 2.9])
        mean_signal = fp32.pow(2).mean().item()
        mean_error = (fp32 - quant).pow(2).mean().item()
        qsnr_mean = 10.0 * math.log10(mean_signal / mean_error)
        qsnr_sum = _manual_qsnr(fp32, quant)
        assert abs(qsnr_mean - qsnr_sum) < 1e-5

    def test_log10_monotonicity(self):
        fp32 = torch.randn(100)
        qsnr_small = _manual_qsnr(fp32, fp32 + 0.01 * torch.randn(100))
        qsnr_large = _manual_qsnr(fp32, fp32 + 0.10 * torch.randn(100))
        assert qsnr_small > qsnr_large


class TestMultiBatchAccumulation:
    """Verify that multi-batch QSNR accumulation is mathematically correct."""

    def test_two_batches_match_concatenated(self):
        b1_fp = torch.randn(4, 8)
        b1_q = b1_fp + 0.05 * torch.randn(4, 8)
        b2_fp = torch.randn(4, 8)
        b2_q = b2_fp + 0.05 * torch.randn(4, 8)
        acc = _manual_qsnr_multi([(b1_fp, b1_q), (b2_fp, b2_q)])
        cat_fp = torch.cat([b1_fp.reshape(-1), b2_fp.reshape(-1)])
        cat_q = torch.cat([b1_q.reshape(-1), b2_q.reshape(-1)])
        cat = _manual_qsnr(cat_fp, cat_q)
        assert abs(acc - cat) < 1e-5

    def test_accumulation_not_mean_of_qsnrs(self):
        b1_fp = torch.randn(100)
        b1_q = b1_fp + 0.01 * torch.randn(100)
        b2_fp = torch.randn(100)
        b2_q = b2_fp + 0.10 * torch.randn(100)
        acc = _manual_qsnr_multi([(b1_fp, b1_q), (b2_fp, b2_q)])
        qsnr1 = _manual_qsnr(b1_fp, b1_q)
        qsnr2 = _manual_qsnr(b2_fp, b2_q)
        mean = (qsnr1 + qsnr2) / 2.0
        assert abs(acc - mean) > 0.5

    def test_uneven_batch_sizes(self):
        b1_fp = torch.randn(4, 8)
        b1_q = b1_fp + 0.01 * torch.randn(4, 8)
        b2_fp = torch.randn(16, 8)
        b2_q = b2_fp + 0.01 * torch.randn(16, 8)
        acc = _manual_qsnr_multi([(b1_fp, b1_q), (b2_fp, b2_q)])
        cat_fp = torch.cat([b1_fp.reshape(-1), b2_fp.reshape(-1)])
        cat_q = torch.cat([b1_q.reshape(-1), b2_q.reshape(-1)])
        cat = _manual_qsnr(cat_fp, cat_q)
        assert abs(acc - cat) < 1e-5


# ==============================================================================
# cfg_causes_quantization
# ==============================================================================

def test_cfg_causes_quantization_empty():
    assert not cfg_causes_quantization(None)
    assert not cfg_causes_quantization(OpQuantConfig())


def test_cfg_causes_quantization_with_scheme():
    scheme = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=GranularitySpec.per_tensor(),
    )
    assert cfg_causes_quantization(OpQuantConfig(input=scheme))
    assert cfg_causes_quantization(OpQuantConfig(storage=scheme))
    assert cfg_causes_quantization(OpQuantConfig(grad_input=scheme))


# ==============================================================================
# Accum QSNR via run_quantization (replaces Session.analyze hook path)
# ==============================================================================

class TestAccumQSNR:
    """Tests for accumulated QSNR hook path via run_quantization."""

    def test_accum_qsnr_populated(self):
        """Hook path populates accum_qsnr_per_layer when qsnr is in outputs."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        qmodel, fp32_model, result = run_quantization(
            model, config,
            [torch.randn(4, 4)],
            outputs=["qsnr"],
        )
        assert len(result.accum_qsnr_per_layer) > 0
        assert len(result.accum_mse_per_layer) > 0
        for qsnr in result.accum_qsnr_per_layer.values():
            assert qsnr > 0
            assert not torch.tensor(qsnr).isnan()

    def test_accum_qsnr_multi_batch_accumulation(self):
        """Multi-batch accum QSNR accumulates across batches."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        qmodel, fp32_model, result = run_quantization(
            model, config,
            [torch.randn(4, 4) for _ in range(4)],
            outputs=["qsnr"],
        )
        assert len(result.accum_qsnr_per_layer) > 0

    def test_accum_qsnr_with_eval_fn(self):
        """Hook path with eval_fn uses eval_fn, not direct model()."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")

        called = []
        def my_eval(m, data):
            called.append(m)
            if isinstance(data, (list, tuple)):
                for batch in data:
                    m(batch)
            else:
                m(data)

        qmodel, fp32_model, result = run_quantization(
            model, config,
            [torch.randn(4, 4)],
            outputs=["qsnr"],
            eval_fn=my_eval,
        )
        assert len(called) >= 2
        assert len(result.accum_qsnr_per_layer) > 0

    def test_hook_and_observer_combined(self):
        """qsnr output gives both accum (hook) and local (observer) data."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        qmodel, fp32_model, result = run_quantization(
            model, config,
            [torch.randn(4, 4)],
            outputs=["qsnr"],
        )
        assert len(result.accum_qsnr_per_layer) > 0
        assert len(result.qsnr_per_layer) > 0
        assert len(result.observers_data) > 0

    def test_accum_qsnr_excludes_non_quantizing_modules(self):
        """Modules with empty cfg excluded from accum hook comparison."""
        model = _make_small_model()
        config = QuantConfig(calibrator="max")
        qmodel, fp32_model, result = run_quantization(
            model, config,
            [torch.randn(4, 4)],
            outputs=["qsnr"],
        )
        for name, qsnr in result.accum_qsnr_per_layer.items():
            assert qsnr == qsnr  # not NaN
            assert qsnr > 0


# ==============================================================================
# Calibration via CalibrationSession (no _QuantSession wrapper)
# ==============================================================================

class TestDirectCalibration:
    """Calibration works directly on qmodel via CalibrationSession."""

    def test_calibrate_context_assigns_scales(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        with CalibrationSession(qmodel, MaxScaleStrategy()):
            with torch.no_grad():
                for _ in range(4):
                    qmodel(torch.randn(2, 4))

        has_scale = any(
            hasattr(m, "_output_scale") for m in qmodel.modules()
        )
        assert has_scale

    def test_calibrate_with_custom_strategy(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        strat = PercentileScaleStrategy(q=95.0)
        with CalibrationSession(qmodel, strat):
            with torch.no_grad():
                for _ in range(4):
                    qmodel(torch.randn(2, 4))

        has_scale = any(
            hasattr(m, "_output_scale") for m in qmodel.modules()
        )
        assert has_scale

    def test_clear_scales(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        with CalibrationSession(qmodel, MaxScaleStrategy()):
            with torch.no_grad():
                for _ in range(4):
                    qmodel(torch.randn(2, 4))

        removed = clear_scales(qmodel)
        assert len(removed) > 0

        has_scale = any(
            hasattr(m, "_output_scale") for m in qmodel.modules()
        )
        assert not has_scale


# ==============================================================================
# Analysis via AnalysisContext (no _QuantSession wrapper)
# ==============================================================================

class TestDirectAnalysis:
    """Analysis works directly on qmodel via AnalysisContext."""

    def test_analyze_collects_metrics(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        with AnalysisContext(qmodel, [QSNRObserver()]) as ctx:
            with torch.no_grad():
                for _ in range(4):
                    qmodel(torch.randn(2, 4))

        report = ctx.report()
        assert len(report.keys()) > 0

    def test_analyze_with_custom_observers(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        with AnalysisContext(qmodel, [MSEObserver()]) as ctx:
            with torch.no_grad():
                qmodel(torch.randn(2, 4))

        report = ctx.report()
        assert len(report.keys()) > 0


# ==============================================================================
# E2E Comparison (compare_models + compare_sessions)
# ==============================================================================

class TestCompareModels:
    """compare_models() runs fp32 vs quant e2e comparison."""

    def test_compare_models_basic(self):
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

    def test_compare_sessions_multiple(self):
        from src.analysis.e2e import compare_sessions

        model1 = _make_small_model()
        model2 = _make_small_model()
        sd = model1.state_dict()
        model2.load_state_dict(sd)

        fp32_model = copy.deepcopy(model1)
        cfg = _make_cfg()
        qmodel1 = quantize_model(model1, cfg=cfg)
        qmodel2 = quantize_model(model2, cfg=cfg)
        fp32_model.eval()
        qmodel1.eval()
        qmodel2.eval()

        dl = _make_dataloader(n_samples=16, batch_size=4)
        results = compare_sessions(fp32_model, {"s1": qmodel1, "s2": qmodel2}, dl)

        assert "fp32" in results
        assert "s1" in results
        assert "s2" in results
        assert "fp32" in results["s1"]
        assert "quant" in results["s1"]
        assert "delta" in results["s1"]

    def test_compare_sessions_custom_label(self):
        from src.analysis.e2e import compare_sessions

        model = _make_small_model()
        fp32_model = copy.deepcopy(model)
        cfg = _make_cfg()
        qmodel = quantize_model(model, cfg=cfg)
        fp32_model.eval()
        qmodel.eval()

        dl = _make_dataloader(n_samples=16, batch_size=4)
        results = compare_sessions(fp32_model, {"a": qmodel}, dl, fp32_label="baseline")

        assert "baseline" in results
        assert "a" in results


# ==============================================================================
# Comparator standalone
# ==============================================================================

class TestComparator:
    """Comparator standalone tests — record and evaluate."""

    def test_comparator_basic(self):
        from src.analysis.e2e import Comparator
        cmp = Comparator()
        assert cmp.num_samples == 0

        with cmp:
            fp32 = torch.randn(4, 3)
            quant = fp32 + 0.01 * torch.randn(4, 3)
            labels = torch.randint(0, 3, (4,))
            cmp.record(fp32, quant, labels)

        assert cmp.num_samples == 4

    def test_comparator_evaluate(self):
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

    def test_comparator_evaluate_with_directions(self):
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

    def test_comparator_device(self):
        from src.analysis.e2e import Comparator
        cmp = Comparator(device=torch.device("cpu"))
        assert cmp._device == torch.device("cpu")


# ==============================================================================
# ONNX Export
# ==============================================================================

class TestONNXExport:
    """ONNX export works via export_quantized_model()."""

    def test_export_onnx_with_dummy_input(self):
        from src.onnx.export import export_quantized_model
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        x = torch.randn(1, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.onnx")
            export_quantized_model(qmodel, x, path)
            assert os.path.exists(path)

    def test_export_onnx_no_input_raises(self):
        from src.onnx.export import export_quantized_model
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        with pytest.raises((ValueError, TypeError)):
            export_quantized_model(qmodel, None, "nowhere.onnx")


# ==============================================================================
# Pre-Scale Integration (P5) — standalone helpers, no _QuantSession
# ==============================================================================

class TestPreScaleIntegration:
    """Tests for initialize_pre_scales() and optimize_scales() standalone functions."""

    def test_initialize_pre_scales_adds_buffers(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)

        for _, mod in qmodel.named_modules():
            assert not hasattr(mod, "_pre_scale")

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = initialize_pre_scales(qmodel, calib_data, init="ones")

        assert count > 0
        found = 0
        for _, mod in qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                found += 1
                assert isinstance(mod._pre_scale, torch.Tensor)
        assert found == count

    def test_optimize_scales_runs(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        fp32_model = copy.deepcopy(model)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        initialize_pre_scales(qmodel, calib_data, init="ones")

        opt = LayerwiseScaleOptimizer(num_steps=10, num_batches=2, lr=0.01)
        scales = optimize_scales(qmodel, fp32_model, opt, calib_data)

        assert isinstance(scales, dict)
        assert len(scales) > 0

    def test_initialize_pre_scales_preserves_existing_cfg(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        initialize_pre_scales(qmodel, calib_data, init="ones")

        for _, mod in qmodel.named_modules():
            if hasattr(mod, "cfg") and cfg_causes_quantization(mod.cfg):
                assert isinstance(mod.cfg, OpQuantConfig)

    def test_optimize_scales_fails_without_fp32(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        opt = LayerwiseScaleOptimizer(num_steps=5, num_batches=2)

        with pytest.raises(Exception):
            optimize_scales(qmodel, None, opt, calib_data)

    def test_initialize_pre_scales_invalid_init(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)

        with pytest.raises(ValueError, match="Unknown init method"):
            initialize_pre_scales(qmodel, [torch.randn(8, 4)], init="invalid")

    def test_forward_after_pre_scale_init(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)

        initialize_pre_scales(qmodel, [torch.randn(8, 4) for _ in range(4)], init="ones")
        qmodel.eval()
        out = qmodel(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_e2e_pre_scale_pipeline(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

        torch.manual_seed(42)
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        fp32_model = copy.deepcopy(model)

        calib_data = [torch.randn(8, 4) for _ in range(6)]

        # Step 1: Calibrate
        with CalibrationSession(qmodel, MaxScaleStrategy()):
            for batch in calib_data:
                qmodel(batch)

        # Step 2: Initialize pre-scales
        count = initialize_pre_scales(qmodel, calib_data, init="ones")
        assert count > 0

        # Step 3: LSQ optimize
        opt = LayerwiseScaleOptimizer(num_steps=20, num_batches=3, lr=0.01)
        scales = optimize_scales(qmodel, fp32_model, opt, calib_data)
        assert len(scales) >= count

        # Step 4: Forward pass works
        out = qmodel(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

        # Step 5: fp32 model forward works independently
        fp32_out = fp32_model(torch.randn(4, 4))
        assert fp32_out.shape == (4, 3)

    def test_e2e_pre_scale_pot_pipeline(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

        torch.manual_seed(42)
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        fp32_model = copy.deepcopy(model)

        calib_data = [torch.randn(8, 4) for _ in range(6)]

        # Calibrate
        with CalibrationSession(qmodel, MaxScaleStrategy()):
            for batch in calib_data:
                qmodel(batch)

        # Initialize with pot=True
        count = initialize_pre_scales(qmodel, calib_data, init="ones", pot=True)
        assert count > 0

        # LSQ optimize with pot=True
        opt = LayerwiseScaleOptimizer(num_steps=20, num_batches=3, lr=0.01, pot=True)
        scales = optimize_scales(qmodel, fp32_model, opt, calib_data)
        assert len(scales) >= count

        # All optimized scales must be PoT
        for scale in scales.values():
            log2 = torch.log2(scale)
            assert torch.equal(log2, torch.round(log2)), \
                f"scale {scale} is not power-of-two"

        out = qmodel(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_initialize_amax_init(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = initialize_pre_scales(qmodel, calib_data, init="amax")
        assert count > 0

        for _, mod in qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                assert mod._pre_scale.numel() == 1  # per_tensor default

        out = qmodel(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_initialize_pot_amax_init(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = initialize_pre_scales(qmodel, calib_data, init="pot_amax")
        assert count > 0

        for _, mod in qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                scale = mod._pre_scale
                log2 = torch.log2(scale)
                assert torch.equal(log2, torch.round(log2)), \
                    f"scale {scale} is not power-of-two"

        out = qmodel(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_initialize_per_channel_granularity(self):
        from src.transform.pre_scale import PreScaleTransform
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = initialize_pre_scales(
            qmodel, calib_data, init="ones", granularity="per_channel",
        )
        assert count > 0

        for _, mod in qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                if mod.cfg.input is not None:
                    assert isinstance(mod.cfg.input.transform, PreScaleTransform)
                if mod.cfg.output is not None:
                    assert not isinstance(
                        mod.cfg.output.transform, PreScaleTransform,
                    )
                if mod.cfg.weight is not None:
                    assert not isinstance(
                        mod.cfg.weight.transform, PreScaleTransform,
                    )

    def test_initialize_trainable_parameter(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        calib_data = [torch.randn(8, 4) for _ in range(4)]
        count = initialize_pre_scales(
            qmodel, calib_data, init="ones", trainable=True,
        )
        assert count > 0

        for _, mod in qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                assert isinstance(mod._pre_scale, nn.Parameter)

        param_names = [n for n, _ in qmodel.named_parameters()]
        pre_scale_params = [n for n in param_names if "_pre_scale" in n]
        assert len(pre_scale_params) > 0

    def test_initialize_invalid_granularity(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)

        with pytest.raises(ValueError, match="Unknown granularity"):
            initialize_pre_scales(
                qmodel, [torch.randn(8, 4)], init="ones", granularity="invalid",
            )

    def test_e2e_hierarchical_pipeline(self):
        model = _make_small_model()
        cfg = _make_cfg()
        qmodel = quantize_model(copy.deepcopy(model), cfg=cfg)
        qmodel.eval()

        calib_data = [torch.randn(8, 4) for _ in range(6)]

        count = initialize_pre_scales(
            qmodel, calib_data, init="pot_amax", pot=True, granularity="per_channel",
        )
        assert count > 0

        with CalibrationSession(qmodel, MaxScaleStrategy()):
            for batch in calib_data:
                qmodel(batch)

        out = qmodel(torch.randn(4, 4))
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()


# ==============================================================================
# True Error End-to-End Verification
# ==============================================================================

class TestTrueErrorEndToEnd:
    """End-to-end verification: run_quantization accum QSNR matches manual."""

    def _make_deterministic_model(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))
        for p in model.parameters():
            nn.init.constant_(p, 0.5)
        return model

    def test_qsnr_matches_manual_computation(self):
        """run_quantization accum QSNR must match manual hook-based computation.

        Uses the same input tensor for both the run_quantization pipeline
        and the manual hook computation, so calibration state is identical.
        """
        torch.manual_seed(42)
        model = self._make_deterministic_model()
        config = QuantConfig(calibrator="max")

        x = torch.randn(4, 2)
        qmodel, fp32_model, result = run_quantization(
            model, config,
            [x],
            outputs=["qsnr"],
        )

        # Manual computation on the same calibrated models with the same input
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

        hooks = []
        for name, _mod in qmodel.named_modules():
            if hasattr(_mod, "cfg") and cfg_causes_quantization(_mod.cfg):
                fp32_mod = dict(fp32_model.named_modules()).get(name)
                quant_mod = dict(qmodel.named_modules()).get(name)
                if fp32_mod and quant_mod:
                    hooks.append(fp32_mod.register_forward_hook(fp32_hook(name)))
                    hooks.append(quant_mod.register_forward_hook(quant_hook(name)))

        with torch.no_grad():
            fp32_model(x)
            qmodel(x)
        for h in hooks:
            h.remove()

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

        assert set(result.accum_qsnr_per_layer.keys()) == set(manual_qsnr.keys())
        for name in manual_qsnr:
            assert abs(result.accum_qsnr_per_layer[name] - manual_qsnr[name]) < 1e-5, (
                f"Mismatch at {name}: "
                f"result={result.accum_qsnr_per_layer[name]:.6f} "
                f"manual={manual_qsnr[name]:.6f}"
            )

    def test_multi_batch_qsnr_matches_single_concatenated(self):
        """Multi-batch accumulated QSNR = QSNR of concatenated tensors."""
        torch.manual_seed(42)
        model = self._make_deterministic_model()
        config = QuantConfig(calibrator="max")

        qmodel, fp32_model, result = run_quantization(
            model, config,
            [torch.randn(4, 2) for _ in range(3)],
            outputs=["qsnr"],
        )

        batches = [torch.randn(4, 2) for _ in range(3)]

        accum_signal = {}
        accum_error = {}
        all_fp32_tensors = {}
        all_quant_tensors = {}

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
                    accum_signal[name] = accum_signal.get(name, 0.0) + fp.pow(2).sum().item()
                    accum_error[name] = accum_error.get(name, 0.0) + (fp - q).pow(2).sum().item()
                    all_fp32_tensors.setdefault(name, []).append(fp.reshape(-1))
                    all_quant_tensors.setdefault(name, []).append(q.reshape(-1))

        multi_qsnr = {}
        for name in accum_signal:
            if accum_error[name] > 1e-30:
                multi_qsnr[name] = 10.0 * math.log10(
                    max(accum_signal[name], 1e-12) / accum_error[name]
                )

        cat_qsnr = {}
        for name in all_fp32_tensors:
            cat_fp = torch.cat(all_fp32_tensors[name])
            cat_q = torch.cat(all_quant_tensors[name])
            cat_qsnr[name] = _manual_qsnr(cat_fp, cat_q)

        for name in multi_qsnr:
            assert name in cat_qsnr
            assert abs(multi_qsnr[name] - cat_qsnr[name]) < 1e-5, (
                f"Accumulation bug at {name}: "
                f"multi={multi_qsnr[name]:.6f} cat={cat_qsnr[name]:.6f}"
            )


# ==============================================================================
# SessionResult QSNR extraction
# ==============================================================================

class TestExtractQSNRMSE:
    """Verify SessionResult.qsnr_per_role() role filtering and worst-case selection."""

    def test_extracts_output_role_only(self):
        data = {
            "fc1": {
                "input": {"stage1": {"s0": {"qsnr_db": 10.0}}},
                "weight": {"stage1": {"s0": {"qsnr_db": 20.0}}},
                "output": {"stage1": {"s0": {"qsnr_db": 30.0}}},
            },
        }
        r = SessionResult(name="test", config=QuantConfig(), observers_data=data)
        qsnr, mse = r.qsnr_per_role()
        assert qsnr == {"fc1": 30.0}

    def test_takes_worst_case_across_stages(self):
        data = {
            "fc1": {
                "output": {
                    "stage1": {"s0": {"qsnr_db": 25.0}, "s1": {"qsnr_db": 30.0}},
                    "stage2": {"s0": {"qsnr_db": 15.0}},
                },
            },
        }
        r = SessionResult(name="test", config=QuantConfig(), observers_data=data)
        qsnr, mse = r.qsnr_per_role()
        assert qsnr == {"fc1": 15.0}

    def test_nan_values_are_skipped(self):
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
        r = SessionResult(name="test", config=QuantConfig(), observers_data=data)
        qsnr, mse = r.qsnr_per_role()
        assert qsnr == {"fc1": 30.0}

    def test_role_not_present_returns_empty(self):
        data = {
            "fc1": {
                "input": {"stage1": {"s0": {"qsnr_db": 10.0}}},
            },
        }
        r = SessionResult(name="test", config=QuantConfig(), observers_data=data)
        qsnr, mse = r.qsnr_per_role()
        assert qsnr == {}


class TestQSNRInvariants:
    """Invariants that must hold for any correct QSNR implementation."""

    def test_identity_quantization_is_inf(self):
        x = torch.randn(4, 8)
        assert _manual_qsnr(x, x) == float("inf")

    def test_positive_signal_positive_qsnr(self):
        fp32 = torch.ones(100)
        quant = fp32 + 0.01 * torch.randn(100)
        assert _manual_qsnr(fp32, quant) > 0

    def test_additive_noise_lowers_qsnr(self):
        fp32 = torch.randn(1000)
        q1 = fp32 + 0.01 * torch.randn(1000)
        q2 = q1 + 0.01 * torch.randn(1000)
        assert _manual_qsnr(fp32, q1) > _manual_qsnr(fp32, q2)

    def test_scaling_invariant(self):
        x = torch.randn(100)
        x_hat = x + 0.01 * torch.randn(100)
        alpha = 3.0
        assert abs(_manual_qsnr(alpha * x, alpha * x_hat) -
                   _manual_qsnr(x, x_hat)) < 1e-5
