"""
Tests for CalibrationPipeline (Phase 8B.2).

Verifies that the pipeline correctly:
- Iterates through calibration data
- Collects activation statistics from quantized layers
- Computes final scale factors using ScaleStrategy
- Handles edge cases (empty model, no quantized layers, empty dataloader)
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.scheme.op_config import OpQuantConfig
from src.calibration.strategies import (
    ScaleStrategy,
    MaxScaleStrategy,
    PercentileScaleStrategy,
    MSEScaleStrategy,
    KLScaleStrategy,
)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_model():
    """Create a simple model with a single QuantizedLinear layer (cfg present)."""
    from src.ops.linear import QuantizedLinear
    model = nn.Sequential()
    model.add_module("linear", QuantizedLinear(4, 8, cfg=OpQuantConfig()))
    return model


def _make_dataloader(n_samples=16, batch_size=4, n_features=4):
    """Create DataLoader yielding (input,) tuples."""
    data = torch.randn(n_samples, n_features)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size)


def _make_dataloader_plain_tensor(n_samples=16, batch_size=4, n_features=4):
    """Create DataLoader that yields plain tensors (no tuple wrapping).

    A custom Dataset returning a single tensor causes the DataLoader
    collation to yield a batched tensor directly.
    """
    class _TensorDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]  # returns a 1-D tensor, no tuple

    data = torch.randn(n_samples, n_features)
    dataset = _TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

def test_pipeline_creates():
    """CalibrationPipeline can be instantiated with model + strategy."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    pipeline = CalibrationPipeline(model, MaxScaleStrategy())
    assert pipeline.model is model
    assert isinstance(pipeline.strategy, MaxScaleStrategy)
    assert pipeline.num_batches == 64  # default
    assert pipeline.axis == -1  # default


def test_pipeline_custom_params():
    """CalibrationPipeline accepts custom num_batches and axis."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=10, axis=0)
    assert pipeline.num_batches == 10
    assert pipeline.axis == 0


def test_pipeline_requires_model():
    """CalibrationPipeline rejects missing model."""
    from src.calibration.pipeline import CalibrationPipeline
    with pytest.raises(TypeError, match="required positional argument"):
        CalibrationPipeline()  # no model


# ---------------------------------------------------------------------------
# 2. Statistics collection
# ---------------------------------------------------------------------------

def test_pipeline_collects_stats():
    """After calibration, returns dict with scale tensors for quantized layers."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    dataloader = _make_dataloader(n_samples=8, batch_size=4)
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=2, axis=-1)
    scales = pipeline.calibrate(dataloader)
    assert isinstance(scales, dict)
    assert "linear" in scales
    assert isinstance(scales["linear"], torch.Tensor)
    # axis=-1 for output shape (4, 8) → running amax shape (4, 1)
    assert scales["linear"].shape == (4, 1)
    assert (scales["linear"] > 0).all()


def test_pipeline_multiple_quantized_layers():
    """Pipeline collects stats from all quantized layers in the model."""
    from src.calibration.pipeline import CalibrationPipeline
    from src.ops.linear import QuantizedLinear

    model = nn.Sequential()
    model.add_module("fc1", QuantizedLinear(4, 8, cfg=OpQuantConfig()))
    model.add_module("fc2", QuantizedLinear(8, 2, cfg=OpQuantConfig()))

    dataloader = _make_dataloader(n_samples=8, batch_size=4)
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=2)
    scales = pipeline.calibrate(dataloader)
    assert "fc1" in scales
    assert "fc2" in scales
    assert len(scales) == 2


# ---------------------------------------------------------------------------
# 3. num_batches limit
# ---------------------------------------------------------------------------

def test_pipeline_respects_num_batches():
    """Only processes num_batches iterations (fewer than total available)."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    dataloader = _make_dataloader(n_samples=64, batch_size=2)  # 32 batches total
    # Set num_batches much smaller than total
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=3)
    scales = pipeline.calibrate(dataloader)
    assert "linear" in scales  # still produces scales with partial data


def test_pipeline_num_batches_zero():
    """num_batches=0 processes no data and returns empty dict."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    dataloader = _make_dataloader()
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=0)
    scales = pipeline.calibrate(dataloader)
    assert scales == {}


# ---------------------------------------------------------------------------
# 4. MaxScaleStrategy correctness
# ---------------------------------------------------------------------------

def test_pipeline_max_strategy_scales():
    """With MaxScaleStrategy, scales match expected amax of activations."""
    from src.calibration.pipeline import CalibrationPipeline
    from src.ops.linear import QuantizedLinear

    torch.manual_seed(42)

    # Create model with fixed weights for deterministic output
    model = nn.Sequential()
    lin = QuantizedLinear(4, 8, cfg=OpQuantConfig())
    nn.init.ones_(lin.weight)
    nn.init.zeros_(lin.bias)
    model.add_module("linear", lin)

    # Single batch of known data
    x = torch.ones(2, 4) * 3.0  # (2, 4), all values = 3
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=2)

    # Expected output: F.linear(x, w, b) with w=1 → each output = 3*4 = 12.0
    # running_amax along axis=-1 for shape (2, 8): (2, 1) all 12.0
    # MaxScaleStrategy.compute → clamp(running_amax, 1e-12)
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=1, axis=-1)
    scales = pipeline.calibrate(dataloader)

    assert scales["linear"].shape == (2, 1)
    expected = torch.ones(2, 1) * 12.0
    assert torch.allclose(scales["linear"], expected, atol=1e-6)


def test_pipeline_max_strategy_multiple_batches():
    """Running amax across multiple batches accumulates correctly."""
    from src.calibration.pipeline import CalibrationPipeline
    from src.ops.linear import QuantizedLinear

    torch.manual_seed(42)

    model = nn.Sequential()
    lin = QuantizedLinear(4, 8, cfg=OpQuantConfig())
    nn.init.ones_(lin.weight)
    nn.init.zeros_(lin.bias)
    model.add_module("linear", lin)

    # Two batches with different magnitudes
    x1 = torch.ones(2, 4) * 3.0       # output: 12.0
    x2 = torch.ones(2, 4) * 5.0       # output: 20.0
    data = torch.cat([x1, x2], dim=0)  # (4, 4)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=2)

    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=2, axis=-1)
    scales = pipeline.calibrate(dataloader)

    # running_amax should capture the max: 20.0
    assert scales["linear"].shape == (2, 1)
    expected = torch.ones(2, 1) * 20.0
    assert torch.allclose(scales["linear"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Works with all strategy types
# ---------------------------------------------------------------------------

def test_pipeline_different_strategies():
    """Works with Max, Percentile, MSE, KL strategies."""
    from src.calibration.pipeline import CalibrationPipeline

    # Fresh model + dataloader for each strategy to avoid iterator exhaustion
    strategies = [
        (MaxScaleStrategy(), (4, 1)),
        (PercentileScaleStrategy(q=99.0), (4, 1)),
        (MSEScaleStrategy(n_steps=5), (4, 1)),
        (KLScaleStrategy(n_bins=32, n_steps=5), (1, 1)),
    ]

    for strategy, expected_shape in strategies:
        model = _make_model()
        dataloader = _make_dataloader(n_samples=8, batch_size=4)
        pipeline = CalibrationPipeline(model, strategy, num_batches=2, axis=-1)
        scales = pipeline.calibrate(dataloader)
        assert "linear" in scales, f"Failed for {type(strategy).__name__}"
        assert isinstance(scales["linear"], torch.Tensor)
        assert scales["linear"].shape == expected_shape, \
            f"{type(strategy).__name__}: expected {expected_shape}, got {scales['linear'].shape}"
        assert (scales["linear"] > 0).all(), \
            f"Non-positive scales for {type(strategy).__name__}"


# ---------------------------------------------------------------------------
# 6. Batch format handling
# ---------------------------------------------------------------------------

def test_pipeline_tuple_batch():
    """Handles batches that are (inputs, targets) tuples."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    # DataLoader from TensorDataset yields (tensor,) — a single-element tuple
    dataloader = _make_dataloader(n_samples=4, batch_size=4)
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=1)
    scales = pipeline.calibrate(dataloader)
    assert "linear" in scales


def test_pipeline_plain_tensor_batch():
    """Handles batches that are plain tensors (no tuple wrapping)."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    dataloader = _make_dataloader_plain_tensor(n_samples=4, batch_size=4)
    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=1)
    scales = pipeline.calibrate(dataloader)
    assert "linear" in scales


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

def test_pipeline_no_quantized_layers():
    """Model without quantized layers (no cfg attr) returns empty dict."""
    from src.calibration.pipeline import CalibrationPipeline
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )
    dataloader = _make_dataloader()
    pipeline = CalibrationPipeline(model, MaxScaleStrategy())
    scales = pipeline.calibrate(dataloader)
    assert scales == {}


def test_pipeline_empty_dataloader():
    """Empty dataloader returns empty dict (no crash)."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    empty_dataset = TensorDataset(torch.empty(0, 4))
    empty_loader = DataLoader(empty_dataset, batch_size=4)
    pipeline = CalibrationPipeline(model, MaxScaleStrategy())
    scales = pipeline.calibrate(empty_loader)
    assert scales == {}


def test_pipeline_hooks_removed():
    """All hooks are removed after calibration completes."""
    from src.calibration.pipeline import CalibrationPipeline
    model = _make_model()
    dataloader = _make_dataloader()

    # Count hooks before calibration
    n_hooks_before = len(model.linear._forward_hooks)

    pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=1)
    pipeline.calibrate(dataloader)

    # Count hooks after calibration — should be same as before
    n_hooks_after = len(model.linear._forward_hooks)
    assert n_hooks_after == n_hooks_before, (
        f"Hooks not removed: {n_hooks_before} → {n_hooks_after}"
    )
