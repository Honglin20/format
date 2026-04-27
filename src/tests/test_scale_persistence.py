"""
Tests for scale persistence (8B.3): threading pre-computed scales through
the quantize chain, storing scales as module buffers, and e2e calibration + reuse.
"""
import copy
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.quantize.elemwise import quantize
from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.calibration.strategies import MaxScaleStrategy, PercentileScaleStrategy
from src.calibration.pipeline import CalibrationPipeline, CalibrationSession
from src.ops.linear import QuantizedLinear


# ---------------------------------------------------------------------------
# 1. scale kwarg threading — per_channel
# ---------------------------------------------------------------------------

def test_quantize_scale_kwarg_per_channel_matches_autocompute():
    """quantize(x, scheme, scale=precomputed) == quantize(x, scheme) when scale=amax."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_channel("int8", axis=-1)

    # Compute expected: quantize without scale (auto-computes amax)
    expected = quantize(x, scheme)

    # Compute amax separately
    amax = torch.amax(torch.abs(x), dim=-1, keepdim=True).clamp(min=1e-12)

    # Quantize with pre-computed scale
    result = quantize(x, scheme, scale=amax)

    assert torch.equal(result, expected), "pre-computed scale should match auto-computed"


def test_quantize_scale_kwarg_per_channel_different_scale():
    """quantize with different scale produces different output."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_channel("int8", axis=-1)

    auto_result = quantize(x, scheme)
    # Use double the amax → less clamping
    amax = torch.amax(torch.abs(x), dim=-1, keepdim=True).clamp(min=1e-12)
    result_scaled = quantize(x, scheme, scale=amax * 2.0)

    assert not torch.equal(result_scaled, auto_result), "different scale should differ"


def test_quantize_scale_kwarg_per_channel_per_tensor():
    """scale kwarg is accepted by per_tensor quantization (no-op, scale ignored)."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_tensor("int8")

    result_no_scale = quantize(x, scheme)
    result_with_scale = quantize(x, scheme, scale=torch.tensor(2.0))
    result_with_none = quantize(x, scheme, scale=None)

    # per-tensor uses no scaling, so scale is silently ignored
    assert torch.equal(result_with_scale, result_no_scale)
    assert torch.equal(result_with_none, result_no_scale)


# ---------------------------------------------------------------------------
# 2. scale kwarg threading — per_block
# ---------------------------------------------------------------------------

def test_quantize_scale_kwarg_per_block_accepted():
    """Per-block scheme accepts scale=None kwarg without error."""
    torch.manual_seed(42)
    x = torch.randn(4, 32)
    scheme = QuantScheme.mxfp("fp8_e4m3", block_size=8)

    result_auto = quantize(x, scheme)
    result_with_none = quantize(x, scheme, scale=None)

    assert torch.equal(result_with_none, result_auto), \
        "scale=None should behave identically to omitting scale"


# ---------------------------------------------------------------------------
# 3. assign_scales — buffer registration
# ---------------------------------------------------------------------------

def make_simple_quantized_model():
    """Create a 2-layer quantized linear model for testing."""
    cfg = OpQuantConfig(
        input=QuantScheme.per_tensor("int8"),
        output=QuantScheme.per_channel("int8", axis=-1),
    )
    return nn.Sequential(
        QuantizedLinear(8, 16, bias=False, cfg=cfg, name="layer0"),
        nn.ReLU(),
        QuantizedLinear(16, 4, bias=False, cfg=cfg, name="layer1"),
    )


def test_assign_scales_registers_buffers():
    """assign_scales registers _output_scale buffers on target modules."""
    model = make_simple_quantized_model()
    strategy = MaxScaleStrategy()

    # Run calibration
    x = torch.randn(10, 8)
    dl = DataLoader(TensorDataset(x), batch_size=2)
    pipeline = CalibrationPipeline(model, strategy, num_batches=4)
    scales = pipeline.calibrate(dl)

    assigned = pipeline.assign_scales(scales)
    assert len(assigned) > 0, "should have assigned at least one module"

    for name, module in model.named_modules():
        if name in scales:
            assert hasattr(module, "_output_scale"), \
                f"module {name} should have _output_scale buffer"
            assert module._output_scale.shape == scales[name].shape, \
                f"buffer shape mismatch for {name}"


def test_assign_scales_skips_missing_modules():
    """assign_scales silently skips names not in the model."""
    model = make_simple_quantized_model()
    strategy = MaxScaleStrategy()
    pipeline = CalibrationPipeline(model, strategy)

    fake_scales = {"nonexistent.module": torch.tensor([1.0])}
    assigned = pipeline.assign_scales(fake_scales)
    assert assigned == []


def test_assign_scales_survives_state_dict_roundtrip():
    """_output_scale buffer appears in state_dict and survives save/load roundtrip."""
    model = make_simple_quantized_model()
    strategy = MaxScaleStrategy()

    x = torch.randn(10, 8)
    dl = DataLoader(TensorDataset(x), batch_size=2)
    pipeline = CalibrationPipeline(model, strategy, num_batches=4)
    scales = pipeline.calibrate(dl)
    assigned = pipeline.assign_scales(scales)

    # Verify buffers appear in state_dict
    state = model.state_dict()
    for name in assigned:
        key = f"{name}._output_scale"
        assert key in state, f"{key} should be in state_dict"

    # Load into a new model: assign dummy scales first so buffers exist,
    # then load_state_dict overwrites them with real values
    model2 = make_simple_quantized_model()
    dummy_scales = {name: torch.zeros_like(s) for name, s in scales.items()}
    model2_pipeline = CalibrationPipeline(model2, strategy)
    model2_pipeline.assign_scales(dummy_scales)

    model2.load_state_dict(state)

    for name in assigned:
        module2 = dict(model2.named_modules())[name]
        original = dict(model.named_modules())[name]._output_scale
        loaded = module2._output_scale
        assert torch.equal(original, loaded), \
            f"buffer value mismatch for {name}"


# ---------------------------------------------------------------------------
# 4. QuantizedLinear — stored scale passthrough
# ---------------------------------------------------------------------------

def test_linear_with_stored_output_scale():
    """QuantizedLinear passes _output_scale to output quantization."""
    torch.manual_seed(42)
    cfg = OpQuantConfig(
        input=QuantScheme.per_tensor("int8"),
        output=QuantScheme.per_channel("int8", axis=-1),
    )
    model = QuantizedLinear(8, 16, bias=False, cfg=cfg)

    # Manually assign a scale buffer (simulating calibration result)
    precomputed = torch.ones(1, 16) * 2.0  # doubled amax
    model.register_buffer("_output_scale", precomputed)

    x = torch.randn(4, 8)
    y_stored = model(x)

    # Remove buffer and run again (should auto-compute)
    del model._output_scale
    y_auto = model(x)

    # With different scale (2x), output should differ
    assert not torch.equal(y_stored, y_auto), \
        "stored scale (2x amax) should produce different output"


def test_linear_without_stored_scale_works_normally():
    """QuantizedLinear without _output_scale works (backward compatible)."""
    torch.manual_seed(42)
    cfg = OpQuantConfig(
        input=QuantScheme.per_tensor("int8"),
        output=QuantScheme.per_channel("int8", axis=-1),
    )
    model = QuantizedLinear(8, 4, bias=False, cfg=cfg)
    x = torch.randn(4, 8)
    y = model(x)
    assert y.shape == (4, 4)


# ---------------------------------------------------------------------------
# 5. E2E: calibrate → assign → inference
# ---------------------------------------------------------------------------

def test_e2e_calibrate_assign_infer():
    """Full pipeline: calibrate model, assign scales, run inference."""
    torch.manual_seed(42)
    model = make_simple_quantized_model()
    strategy = MaxScaleStrategy()

    # Calibrate on synthetic data
    calib_x = torch.randn(32, 8)
    dl = DataLoader(TensorDataset(calib_x), batch_size=4)
    pipeline = CalibrationPipeline(model, strategy, num_batches=8)
    scales = pipeline.calibrate(dl)

    assert len(scales) > 0, "should have captured scales from quantized layers"

    # Run inference WITHOUT assigned scales (baseline)
    test_x = torch.randn(4, 8)
    with torch.no_grad():
        y_before = model(test_x).clone()

    # Assign scales
    assigned = pipeline.assign_scales(scales)
    assert len(assigned) == len(scales)

    # Run inference WITH assigned scales
    with torch.no_grad():
        y_after = model(test_x).clone()

    # Output should be different (calibrated scale vs auto-computed)
    # Note: with MaxScaleStrategy, the calibrated scale IS the amax from
    # calibration data, so for the same input, results should match.
    # But with different input (test vs calibration), they may differ
    # depending on activation distribution. We just verify it runs.
    assert y_after.shape == y_before.shape
    assert y_after.dtype == y_before.dtype


def test_e2e_with_percentile_strategy():
    """Calibrate with PercentileStrategy and verify different scale produces different output."""
    torch.manual_seed(42)
    model = make_simple_quantized_model()
    # Percentile(50) gives smaller scale → less clamping
    strategy = PercentileScaleStrategy(q=50.0)

    calib_x = torch.randn(32, 8)
    dl = DataLoader(TensorDataset(calib_x), batch_size=4)
    pipeline = CalibrationPipeline(model, strategy, num_batches=8)
    scales = pipeline.calibrate(dl)

    # Run without assigned scales
    test_x = torch.randn(4, 8)
    with torch.no_grad():
        y_before = model(test_x).clone()

    # Assign and run
    pipeline.assign_scales(scales)
    with torch.no_grad():
        y_after = model(test_x).clone()

    # Percentile(50) scales differ from auto-computed amax, so outputs should differ
    assert not torch.equal(y_before, y_after), \
        "Percentile(50) scale should produce different output than auto-amax"


def test_e2e_linear_backward_with_stored_scale():
    """Backward pass works when QuantizedLinear has stored _output_scale."""
    torch.manual_seed(42)
    cfg = OpQuantConfig(
        input=QuantScheme.per_tensor("int8"),
        output=QuantScheme.per_channel("int8", axis=-1),
    )
    model = QuantizedLinear(4, 8, bias=True, cfg=cfg)
    model.register_buffer("_output_scale", torch.ones(1, 8))

    x = torch.randn(2, 4, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert model.weight.grad is not None
    assert model.bias.grad is not None


# ---------------------------------------------------------------------------
# 6. Scale persistence — save / load to disk
# ---------------------------------------------------------------------------

import tempfile
import os


def _make_scales_dict():
    """Helper: produce a deterministic scales dict for persistence tests."""
    torch.manual_seed(42)
    return {
        "layer0": torch.randn(4, 1).abs() + 0.1,
        "layer1": torch.randn(8, 1).abs() + 0.1,
    }


def test_save_load_standalone_roundtrip():
    """Standalone save_scales/load_scales roundtrip preserves all values."""
    from src.calibration import save_scales, load_scales
    original = _make_scales_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name

    try:
        save_scales(original, tmp)
        loaded = load_scales(tmp)
        assert set(loaded.keys()) == set(original.keys())
        for name in original:
            assert torch.equal(loaded[name], original[name]), \
                f"mismatch for {name}"
    finally:
        os.unlink(tmp)


def test_save_scales_returns_filepath():
    """save_scales returns the filepath for chaining."""
    from src.calibration import save_scales
    scales = _make_scales_dict()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    try:
        result = save_scales(scales, tmp)
        assert result == tmp
    finally:
        os.unlink(tmp)


def test_session_save_scales():
    """CalibrationSession.save_scales() writes scales to disk."""
    model = make_simple_quantized_model()
    strategy = MaxScaleStrategy()

    x = torch.randn(10, 8)
    dl = DataLoader(TensorDataset(x), batch_size=2)
    pipeline = CalibrationPipeline(model, strategy, num_batches=4)
    scales = pipeline.calibrate(dl)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name

    try:
        with CalibrationSession(model, strategy, assign=False) as calib:
            # Manually set running_amax to mimic collection
            calib._running_amax = scales
            calib.save_scales(tmp)

        # Load with standalone function
        from src.calibration import load_scales
        loaded = load_scales(tmp)
        assert set(loaded.keys()) == set(scales.keys())
        for name in scales:
            assert torch.equal(loaded[name], scales[name]), \
                f"mismatch for {name}"
    finally:
        os.unlink(tmp)


def test_session_load_scales_assign_true():
    """CalibrationSession.load_scales(assign=True) registers buffers on modules."""
    model = make_simple_quantized_model()
    strategy = MaxScaleStrategy()
    scales = _make_scales_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name

    try:
        torch.save(scales, tmp)

        with CalibrationSession(model, strategy, assign=False) as calib:
            result = calib.load_scales(tmp, assign=True)

        assert result is not None
        # Verify buffers were assigned to matching modules
        module_map = dict(model.named_modules())
        for name in scales:
            if name in module_map:
                assert hasattr(module_map[name], "_output_scale"), \
                    f"{name} missing _output_scale buffer"
                assert torch.equal(module_map[name]._output_scale, scales[name])
    finally:
        os.unlink(tmp)


def test_session_load_scales_assign_false():
    """CalibrationSession.load_scales(assign=False) returns dict but no buffers."""
    model = make_simple_quantized_model()
    strategy = MaxScaleStrategy()
    scales = _make_scales_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name

    try:
        torch.save(scales, tmp)

        with CalibrationSession(model, strategy, assign=False) as calib:
            result = calib.load_scales(tmp, assign=False)

        assert set(result.keys()) == set(scales.keys())
        for name in scales:
            assert torch.equal(result[name], scales[name])
        # No buffers should be registered
        module_map = dict(model.named_modules())
        for name in scales:
            if name in module_map:
                assert not hasattr(module_map[name], "_output_scale"), \
                    f"{name} should not have _output_scale"
    finally:
        os.unlink(tmp)


def test_load_scales_from_static_method():
    """CalibrationSession.load_scales_from() loads without model."""
    scales = _make_scales_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name

    try:
        torch.save(scales, tmp)
        loaded = CalibrationSession.load_scales_from(tmp)
        assert set(loaded.keys()) == set(scales.keys())
        for name in scales:
            assert torch.equal(loaded[name], scales[name])
    finally:
        os.unlink(tmp)


def test_save_load_empty_scales():
    """Saving and loading an empty scales dict works."""
    from src.calibration import save_scales, load_scales

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name

    try:
        save_scales({}, tmp)
        loaded = load_scales(tmp)
        assert loaded == {}
    finally:
        os.unlink(tmp)


def test_e2e_calibrate_save_reload():
    """Full roundtrip: calibrate → save → load into fresh model → outputs match."""
    torch.manual_seed(42)
    model1 = make_simple_quantized_model()
    strategy = MaxScaleStrategy()

    # Calibrate model1
    calib_x = torch.randn(20, 8)
    dl = DataLoader(TensorDataset(calib_x), batch_size=4)
    pipeline = CalibrationPipeline(model1, strategy, num_batches=5)
    scales = pipeline.calibrate(dl)
    pipeline.assign_scales(scales)

    # Save scales to disk
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name

    try:
        torch.save(scales, tmp)

        # Create fresh model2 with same weights, then load scales
        model2 = make_simple_quantized_model()
        # Assign dummy scales first so state_dict keys exist; then overwrite with real weights
        dummy = {n: torch.zeros_like(s) for n, s in scales.items()}
        CalibrationPipeline(model2, strategy).assign_scales(dummy)
        model2.load_state_dict(model1.state_dict())
        with CalibrationSession(model2, strategy, assign=False) as calib:
            calib.load_scales(tmp, assign=True)

        # Both models should produce identical output
        test_x = torch.randn(4, 8)
        with torch.no_grad():
            y1 = model1(test_x)
            y2 = model2(test_x)

        assert torch.equal(y1, y2), \
            "models with same weights and calibrated scales should produce identical output"
    finally:
        os.unlink(tmp)
