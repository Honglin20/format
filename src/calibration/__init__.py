"""
Calibration package: pluggable scale strategies and calibration session.

Provides:
- ScaleStrategy ABC + built-in implementations (Max, Percentile, MSE, KL)
- CalibrationSession: context manager for activation-scale calibration
- CalibrationPipeline: legacy DataLoader-driven pipeline (backward compat)
- save_scales / load_scales: standalone scale persistence helpers

Usage::

    # New (recommended) — context manager
    with CalibrationSession(model, MaxScaleStrategy()) as calib:
        for batch in calib_data:
            model(batch)
    # Scales auto-assigned on exit

    # Persist scales to disk
    calib.save_scales("calib_scales.pt")

    # Restore later
    with CalibrationSession(model, MaxScaleStrategy(), assign=False) as calib:
        calib.load_scales("calib_scales.pt")

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
from src.calibration.pipeline import (
    CalibrationPipeline,
    CalibrationSession,
    save_scales,
    load_scales,
)

__all__ = [
    "ScaleStrategy",
    "MaxScaleStrategy",
    "PercentileScaleStrategy",
    "MSEScaleStrategy",
    "KLScaleStrategy",
    "CalibrationPipeline",
    "CalibrationSession",
    "save_scales",
    "load_scales",
]
