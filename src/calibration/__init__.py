"""
Calibration package: pluggable scale strategies for quantization.

Provides ScaleStrategy ABC and built-in implementations:
- MaxScaleStrategy: absmax (default, bit-exact with existing behavior)
- PercentileScaleStrategy: N-th percentile to exclude outliers
- MSEScaleStrategy: grid-search minimizing MSE
- KLScaleStrategy: grid-search minimizing KL divergence

Usage::

    strategy = MaxScaleStrategy()
    scale = strategy.compute(x, axis=1)
    # Use scale in _quantize_per_channel instead of hardcoded amax
"""

from src.calibration.strategies import (
    ScaleStrategy,
    MaxScaleStrategy,
    PercentileScaleStrategy,
    MSEScaleStrategy,
    KLScaleStrategy,
)

__all__ = [
    "ScaleStrategy",
    "MaxScaleStrategy",
    "PercentileScaleStrategy",
    "MSEScaleStrategy",
    "KLScaleStrategy",
]
