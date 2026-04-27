"""
Calibration package: pluggable scale strategies and calibration session.

Provides:
- ScaleStrategy ABC + built-in implementations (Max, Percentile, MSE, KL)
- CalibrationSession: context manager for activation-scale calibration
- CalibrationPipeline: legacy DataLoader-driven pipeline (backward compat)

Usage::

    # New (recommended) — context manager
    with CalibrationSession(model, MaxScaleStrategy()) as calib:
        for batch in calib_data:
            model(batch)
    # Scales auto-assigned on exit

    # Legacy — DataLoader-driven
    pipeline = CalibrationPipeline(model, strategy, num_batches=8)
    scales = pipeline.calibrate(dataloader)
"""

from src.calibration.strategies import (
    ScaleStrategy,
    MaxScaleStrategy,
    PercentileScaleStrategy,
    MSEScaleStrategy,
    KLScaleStrategy,
)
from src.calibration.pipeline import CalibrationPipeline, CalibrationSession

__all__ = [
    "ScaleStrategy",
    "MaxScaleStrategy",
    "PercentileScaleStrategy",
    "MSEScaleStrategy",
    "KLScaleStrategy",
    "CalibrationPipeline",
    "CalibrationSession",
]
