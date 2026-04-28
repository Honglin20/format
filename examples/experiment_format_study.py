#!/usr/bin/env python3
"""
Quantization Format Precision Study
====================================

Systematic comparison of MXINT/MXFP/INT-PC/NF4-PC at 8-bit and 4-bit,
with FP32 vs PoT scaling comparison and SmoothQuant/Hadamard transform analysis.

Produces 6 tables and 11 figures across 4 experimental parts (A/B/C/D).

Usage:
    PYTHONPATH=. python examples/experiment_format_study.py

Custom model: Edit the four user-customization functions below
(``build_model``, ``make_calib_data``, ``make_eval_loader``, ``eval_fn``)
to run the study on your own architecture.
"""
import copy
import os
import json
import math
from datetime import datetime
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project imports
from src.formats.base import FormatBase
from src.scheme.transform import IdentityTransform, TransformBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform
from src.transform.pre_scale import PreScaleTransform
from src.analysis.observers import (
    QSNRObserver, MSEObserver, HistogramObserver, DistributionObserver,
)
from src.analysis.report import Report
from src.analysis.correlation import LayerSensitivity, ErrorByDistribution
from src.analysis.e2e import compare_sessions
from src.session import QuantSession
from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
from src.calibration.strategies import MSEScaleStrategy


# ---------------------------------------------------------------------------
# User customization: override these functions for your own model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """Return a fresh instance of the FP32 reference model.

    Default: ToyMLP from ``examples/_model.py`` (2-layer MLP with GeLU,
    LayerNorm, and residual connection).  Replace with your own model
    for custom experiments.
    """
    from examples._model import ToyMLP
    return ToyMLP()


def make_calib_data(
    num_samples: int = 256,
    batch_size: int = 16,
) -> List[torch.Tensor]:
    """Return calibration data as a list of tensors (one per batch).

    The default produces random Gaussian data of shape ``(batch_size, 128)``.
    Replace with real calibration data for meaningful results.

    Args:
        num_samples: Total number of calibration samples.
        batch_size: Samples per batch.

    Returns:
        List of ``(batch_size, ...)`` tensors.
    """
    return [torch.randn(batch_size, 128) for _ in range(num_samples // batch_size)]


def make_eval_loader(
    num_samples: int = 512,
    batch_size: int = 16,
) -> DataLoader:
    """Return a DataLoader for evaluation.

    The default produces random input-label pairs (10 classes).
    Replace with your validation/test DataLoader.

    Args:
        num_samples: Total number of evaluation samples.
        batch_size: Batch size for evaluation.

    Returns:
        ``DataLoader`` yielding ``(input_tensor, label_tensor)`` tuples.
    """
    x = torch.randn(num_samples, 128)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def eval_fn(model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate ``model`` on ``dataloader`` and return metric dict.

    Default: classification accuracy ("accuracy" key).

    Args:
        model: PyTorch model (callable, returns logits).
        dataloader: Yields ``(input, label)`` batches.

    Returns:
        Dict of metric name to scalar value, e.g. ``{"accuracy": 0.92}``.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# Shared granularity spec constants
# ---------------------------------------------------------------------------

PER_T   = GranularitySpec.per_tensor()
PER_C0  = GranularitySpec.per_channel(axis=0)
PER_Cm1 = GranularitySpec.per_channel(axis=-1)
PER_B32 = GranularitySpec.per_block(size=32, axis=-1)


# ---------------------------------------------------------------------------
# Config builder helpers
# ---------------------------------------------------------------------------

def make_op_cfg(
    fmt_name: str,
    granularity: GranularitySpec,
    *,
    transform: Optional[TransformBase] = None,
) -> OpQuantConfig:
    """Create inference-only OpQuantConfig where input, weight, and output
    all use the same scheme.

    Convenience helper for the common case where all three roles share
    a single format + granularity + transform combination.

    Args:
        fmt_name: Format name string (e.g. ``"int8"``, ``"fp8_e4m3"``,
            ``"nf4"``).  Resolved via ``FormatBase.from_str()``.
        granularity: ``GranularitySpec`` (e.g. ``PER_T``, ``PER_Cm1``,
            ``PER_B32``).
        transform: Optional ``TransformBase`` instance.  ``None`` means
            ``IdentityTransform`` (no transform).

    Returns:
        ``OpQuantConfig`` with ``input``, ``weight``, ``output`` set to
        the same scheme, all other fields ``None``.
    """
    fmt = FormatBase.from_str(fmt_name)
    scheme = QuantScheme(
        format=fmt,
        granularity=granularity,
        transform=transform or IdentityTransform(),
    )
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)


def make_op_cfg_weight_only(
    fmt_name: str,
    granularity: GranularitySpec,
    *,
    transform: Optional[TransformBase] = None,
) -> OpQuantConfig:
    """Create weight-only OpQuantConfig (useful for NF4).

    Only the ``weight`` field is set; all other roles remain ``None``
    (no quantization applied to input, output, or backward).

    Args:
        fmt_name: Format name string (e.g. ``"nf4"``).
        granularity: ``GranularitySpec``.
        transform: Optional ``TransformBase`` instance.

    Returns:
        ``OpQuantConfig`` with only ``weight`` set.
    """
    fmt = FormatBase.from_str(fmt_name)
    scheme = QuantScheme(
        format=fmt,
        granularity=granularity,
        transform=transform or IdentityTransform(),
    )
    return OpQuantConfig(weight=scheme)


# ---------------------------------------------------------------------------
# Experiment runner stubs
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: OpQuantConfig,
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
    observers: Optional[List] = None,
    *,
    lsq_steps: int = 0,
    lsq_pot: bool = False,
    lsq_lr: float = 1e-3,
    eval_fn: Optional[Callable] = None,
) -> dict:
    """Run a single quantization experiment and return results.

    Args:
        cfg: ``OpQuantConfig`` for this experiment.
        fp32_model: Reference FP32 model (will be deep-copied).
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.
        observers: List of Observer instances (default: QSNR + MSE).
        lsq_steps: If > 0, run LSQ pre-scale optimization for this many steps.
        lsq_pot: If True, constrain pre-scale to power-of-two during LSQ.
        lsq_lr: Learning rate for LSQ optimizer.
        eval_fn: Optional ``(logits, labels) -> dict`` eval function.
            If ``None``, uses the internal accuracy default.

    Returns:
        Dict with keys ``accuracy``, ``fp32_accuracy``, ``delta``, ``report``,
        ``session``, ``qsnr_per_layer``, ``mse_per_layer``.
    """
    if observers is None:
        observers = [QSNRObserver(), MSEObserver()]

    if not calib_data:
        raise ValueError("calib_data must contain at least one batch")

    session = QuantSession(
        fp32_model, cfg,
        calibrator=MSEScaleStrategy(),
        keep_fp32=True,
    )

    # 1. Calibrate
    with session.calibrate():
        for batch in calib_data:
            session(batch)

    # 2. Optional LSQ pre-scale optimization
    if lsq_steps > 0:
        session.initialize_pre_scales(calib_data, init="ones", pot=lsq_pot)
        opt = LayerwiseScaleOptimizer(
            num_steps=lsq_steps,
            num_batches=len(calib_data),
            optimizer="adam",
            lr=lsq_lr,
            pot=lsq_pot,
        )
        session.optimize_scales(opt, calib_data)

    # 3. Analyze with observers
    with session.analyze(observers=observers) as ctx:
        for batch in calib_data:
            session(batch)
    report = ctx.report()

    # 4. E2E accuracy
    if eval_fn is not None:
        result = session.compare(eval_loader, eval_fn=eval_fn)
    else:
        result = session.compare(eval_loader)

    # 5. Extract per-layer QSNR/MSE summaries
    qsnr_per_layer = _extract_metric_per_layer(report, "qsnr_db")
    mse_per_layer = _extract_metric_per_layer(report, "mse")

    return {
        "accuracy": result["quant"],
        "fp32_accuracy": result["fp32"],
        "delta": result["delta"],
        "report": report,
        "session": session,
        "qsnr_per_layer": qsnr_per_layer,
        "mse_per_layer": mse_per_layer,
    }


def _extract_metric_per_layer(report: Report, metric: str) -> Dict[str, float]:
    """Extract per-layer average of a metric from Report.

    Args:
        report: ``Report`` instance.
        metric: Metric name to extract (e.g. ``"qsnr_db"``, ``"mse"``).

    Returns:
        Dict mapping layer name to average metric value.
    """
    df = report.to_dataframe()
    if isinstance(df, list):
        result = {}
        for row in df:
            name = row.get("layer", "unknown")
            val = row.get(metric)
            if val is not None:
                result.setdefault(name, []).append(val)
        return {k: sum(v) / len(v) for k, v in result.items()}
    else:
        grouped = df.groupby("layer")[metric].mean()
        return grouped.to_dict()


def run_part_a_8bit(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
) -> dict:
    """Part A: Compare MXINT-8, MXFP-8, INT8-PC (all 8-bit, PoT scaling).

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.

    Returns:
        Dict mapping experiment name to result dict, including an
        ``FP32 (baseline)`` entry with the reference accuracy.
    """
    print("\n### Part A: 8-bit Format Comparison ###")
    configs = {
        "MXINT-8": make_op_cfg("int8", PER_B32),
        "MXFP-8":  make_op_cfg("fp8_e4m3", PER_B32),
        "INT8-PC": make_op_cfg("int8", PER_C0),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"  Running {name}...")
        results[name] = run_experiment(cfg, fp32_model, calib_data, eval_loader)
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results


def run_part_b_4bit(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
) -> dict:
    """Part B: Compare MXINT-4, MXFP-4, INT4-PC, NF4-PC (all 4-bit).

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.

    Returns:
        Dict mapping experiment name to result dict, including an
        ``FP32 (baseline)`` entry with the reference accuracy.
    """
    print("\n### Part B: 4-bit Format Comparison ###")
    configs = {
        "MXINT-4": make_op_cfg("int4", PER_B32),
        "MXFP-4":  make_op_cfg("fp4_e2m1", PER_B32),
        "INT4-PC": make_op_cfg("int4", PER_C0),
        "NF4-PC":  make_op_cfg_weight_only("nf4", PER_C0),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"  Running {name}...")
        results[name] = run_experiment(cfg, fp32_model, calib_data, eval_loader)
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results


def run_part_c_pot_scaling(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
) -> dict:
    """Part C: INT per-channel with FP32 scaling vs PoT scaling (8-bit & 4-bit).

    Compares power-of-two (PoT) constrained learning scales against
    unconstrained FP32 scales for both 8-bit and 4-bit INT per-channel
    quantization, using LSQ (Learned Step-size Quantization) optimization.

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.

    Returns:
        Dict mapping experiment name to result dict.
    """
    print("\n### Part C: FP32 vs PoT Scaling ###")
    configs = {
        "INT8-PC-FP32": ("int8", False),
        "INT8-PC-PoT":  ("int8", True),
        "INT4-PC-FP32": ("int4", False),
        "INT4-PC-PoT":  ("int4", True),
    }
    results = {}
    for name, (fmt, pot) in configs.items():
        print(f"  Running {name} (LSQ, pot={pot})...")
        cfg = make_op_cfg(fmt, PER_C0)
        results[name] = run_experiment(
            cfg, fp32_model, calib_data, eval_loader,
            lsq_steps=100, lsq_pot=pot,
        )
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results


def run_format_study(
    build_model: Callable[[], nn.Module],
    make_calib_data: Callable[..., List[torch.Tensor]],
    make_eval_loader: Callable[..., DataLoader],
    eval_fn: Callable[[nn.Module, DataLoader], Dict[str, float]],
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """Main entry point: run all 27 experiments and produce tables/figures.

    Executes Parts A/B/C/D of the study, saving results to a timestamped
    directory under ``output_dir`` (default: ``results/``).

    .. note::
        Stub — returns empty ``dict``.  Full implementation will be
        added in subsequent tasks.

    Args:
        build_model: Callable that returns a fresh FP32 model.
        make_calib_data: Callable for calibration data.
        make_eval_loader: Callable for evaluation DataLoader.
        eval_fn: Evaluation function (model, dataloader) -> metric dict.
        output_dir: Output directory.  ``None`` uses ``"results"``.

    Returns:
        Dict mapping experiment name to result dict.
    """
    _ = build_model, make_calib_data, make_eval_loader, eval_fn, output_dir
    return {}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_format_study(
        build_model, make_calib_data, make_eval_loader, eval_fn,
    )
    print("Study complete. Results:", results)
