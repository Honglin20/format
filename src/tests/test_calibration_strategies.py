"""
Tests for ScaleStrategy abstraction and implementations (P2F-8 / 8B.1).

Verifies that each strategy correctly computes scale factors along a
specified axis, matches existing amax behavior for MaxScaleStrategy,
and that percentile/MSE/KL produce reasonable results.
"""
import pytest
import torch

from src.calibration.strategies import (
    ScaleStrategy,
    MaxScaleStrategy,
    PercentileScaleStrategy,
    MSEScaleStrategy,
    KLScaleStrategy,
)


# ---------------------------------------------------------------------------
# Helper: simple int8-like quantize used by MSE/KL strategies
# ---------------------------------------------------------------------------

def _simple_int8_quantize(x, scale):
    """Simulate int8 quantize/dequantize: normalize, round, rescale."""
    x_norm = x / scale
    x_q = torch.round(x_norm * 127.0)
    x_q = x_q.clamp(-127.0, 127.0)
    x_q = x_q / 127.0 * scale
    return x_q


def _compute_mse(x, scale):
    """MSE between x and its int8-quantized version with given scale."""
    x_q = _simple_int8_quantize(x, scale)
    return (x - x_q).pow(2).mean()


# ---------------------------------------------------------------------------
# 1. MaxScaleStrategy — replicates existing amax behavior
# ---------------------------------------------------------------------------

def test_max_strategy_matches_existing_amax():
    """MaxScaleStrategy output == amax(|x|).clamp(min=1e-12)."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    for axis in (0, 1, -1):
        result = MaxScaleStrategy().compute(x.clone(), axis)
        expected = torch.amax(torch.abs(x), dim=axis, keepdim=True).clamp(min=1e-12)
        assert torch.equal(result, expected), f"mismatch for axis={axis}"


def test_max_strategy_output_shape():
    """Output axis dim = 1, other dims preserved."""
    torch.manual_seed(42)
    x = torch.randn(3, 5, 7)
    result = MaxScaleStrategy().compute(x, axis=1)
    assert result.shape == (3, 1, 7)


def test_max_strategy_positive():
    """All scale values > 0 (clamp guarantees this even for all-zero input)."""
    x = torch.zeros(4, 8)
    result = MaxScaleStrategy().compute(x, axis=1)
    assert (result > 0).all()


# ---------------------------------------------------------------------------
# 2. PercentileScaleStrategy — outlier-excluding scale
# ---------------------------------------------------------------------------

def test_percentile_strategy_range():
    """Percentile(q=50) <= Max; Percentile(q=99) >= Percentile(q=50)."""
    torch.manual_seed(42)
    x = torch.randn(4, 8) * 5 + 2  # wide distribution
    max_scale = MaxScaleStrategy().compute(x, axis=1)
    p50 = PercentileScaleStrategy(q=50.0).compute(x, axis=1)
    p99 = PercentileScaleStrategy(q=99.0).compute(x, axis=1)
    assert (p50 <= max_scale + 1e-6).all()
    assert (p99 >= p50 - 1e-6).all()


def test_percentile_strategy_extremes():
    """q=0 gives min |x|, q=100 gives max |x|."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    axis = 1
    p0 = PercentileScaleStrategy(q=0.0).compute(x, axis=axis)
    p100 = PercentileScaleStrategy(q=100.0).compute(x, axis=axis)
    expected_min = torch.min(torch.abs(x), dim=axis, keepdim=True).values
    expected_max = torch.amax(torch.abs(x), dim=axis, keepdim=True)
    assert torch.allclose(p0, expected_min, atol=1e-6)
    assert torch.allclose(p100, expected_max, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. MSEScaleStrategy — grid-search for minimal-MSE scale
# ---------------------------------------------------------------------------

def test_mse_strategy_reduces_error():
    """MSEScaleStrategy produces lower MSE than a deliberately bad scale."""
    torch.manual_seed(42)
    x = torch.randn(4, 8) * 3  # values spread across [-9, 9]
    strategy = MSEScaleStrategy(n_steps=20)
    scale = strategy.compute(x, axis=1)
    mse_optimal = _compute_mse(x, scale)
    # 0.5 is a poor scale for values with std ~3 (clamps most values)
    bad_scale = torch.full_like(scale, 0.5)
    mse_bad = _compute_mse(x, bad_scale)
    assert mse_optimal < mse_bad, (
        f"MSE optimal={mse_optimal.item():.6f} should be < MSE bad={mse_bad.item():.6f}"
    )


def test_mse_strategy_output_shape():
    """Correct output shape for MSEScaleStrategy."""
    torch.manual_seed(42)
    x = torch.randn(3, 5, 7)
    strategy = MSEScaleStrategy(n_steps=10)
    result = strategy.compute(x, axis=2)
    assert result.shape == (3, 5, 1)


# ---------------------------------------------------------------------------
# 4. KLScaleStrategy — minimize KL divergence
# ---------------------------------------------------------------------------

def test_kl_strategy_output_shape():
    """Correct output shape for KLScaleStrategy."""
    torch.manual_seed(42)
    x = torch.randn(3, 5, 7)
    strategy = KLScaleStrategy(n_bins=128, n_steps=10)
    result = strategy.compute(x, axis=0)
    assert result.shape == (1, 5, 7)


# ---------------------------------------------------------------------------
# 5. Interface & type checks
# ---------------------------------------------------------------------------

def test_all_strategies_are_scale_strategy():
    """Every strategy is an instance of the ScaleStrategy ABC."""
    assert isinstance(MaxScaleStrategy(), ScaleStrategy)
    assert isinstance(PercentileScaleStrategy(q=50.0), ScaleStrategy)
    assert isinstance(MSEScaleStrategy(n_steps=10), ScaleStrategy)
    assert isinstance(KLScaleStrategy(n_bins=128, n_steps=10), ScaleStrategy)


def test_strategy_with_specific_axis():
    """axis=0 vs axis=-1 produce correct shapes for every strategy."""
    torch.manual_seed(42)
    x = torch.randn(4, 8, 16)
    strategies = [
        MaxScaleStrategy(),
        PercentileScaleStrategy(q=50.0),
        MSEScaleStrategy(n_steps=5),
        KLScaleStrategy(n_bins=32, n_steps=5),
    ]
    for strategy in strategies:
        name = type(strategy).__name__
        r0 = strategy.compute(x, axis=0)
        r1 = strategy.compute(x, axis=-1)
        assert r0.shape == (1, 8, 16), f"{name} axis=0: got {r0.shape}"
        assert r1.shape == (4, 8, 1), f"{name} axis=-1: got {r1.shape}"
