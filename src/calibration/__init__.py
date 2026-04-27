"""
Calibration package: pluggable scale strategies and calibration pipeline.

Provides:
- ScaleStrategy ABC + built-in implementations (Max, Percentile, MSE, KL)
- CalibrationPipeline: iterate calibration data, collect activation
  statistics, and compute scale factors.

Usage::

    strategy = MaxScaleStrategy()
    pipeline = CalibrationPipeline(model, strategy, num_batches=8)
    scales = pipeline.calibrate(dataloader)

    # Use scale in _quantize_per_channel instead of hardcoded amax
"""

from src.calibration.strategies import (
    ScaleStrategy,
    MaxScaleStrategy,
    PercentileScaleStrategy,
    MSEScaleStrategy,
    KLScaleStrategy,
)
from src.calibration.pipeline import CalibrationPipeline

__all__ = [
    "ScaleStrategy",
    "MaxScaleStrategy",
    "PercentileScaleStrategy",
    "MSEScaleStrategy",
    "KLScaleStrategy",
    "CalibrationPipeline",
]
