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
import argparse
import copy
import os
import json
import math
from datetime import datetime
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Project imports
from src.formats.base import FormatBase
from src.scheme.transform import IdentityTransform, TransformBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform, SmoothQuantWeightTransform
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
# Unified color palette
# ---------------------------------------------------------------------------

# Format-family colours — colourblind-friendly (Wong 2011), distinguishable
# under deuteranopia, protanopia, and tritanopia.
FORMAT_COLORS = {
    "MXINT-8":  "#0072B2",   # blue
    "MXFP-8":   "#D55E00",   # vermillion
    "INT8-PC":  "#009E73",   # bluish green
    "MXINT-4":  "#56B4E9",   # sky blue (same family as MXINT-8)
    "MXFP-4":   "#E69F00",   # orange (same family as MXFP-8)
    "INT4-PC":  "#F0E442",   # yellow
    "NF4-PC":   "#CC79A7",   # reddish purple
}

# Transform variant colours — colourblind-friendly
TRANSFORM_COLORS = {
    "None":        "#0072B2",   # blue
    "SmoothQuant": "#D55E00",   # vermillion
    "Hadamard":    "#009E73",   # bluish green
}

# Histogram channel colours — colourblind-friendly
HIST_COLORS = {
    "fp32_hist":  "#0072B2",   # blue
    "quant_hist": "#D55E00",   # vermillion
    "err_hist":   "#999999",   # grey
}

# Fallback cycle — colourblind-friendly Wong (2011) palette
FALLBACK_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7",
                  "#56B4E9", "#E69F00", "#999999", "#000000", "#E5C494"]


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


def _make_sq_op_cfg(
    fmt_name: str,
    granularity: GranularitySpec,
    sq_transform: TransformBase,
    weight_only: bool,
) -> OpQuantConfig:
    """Create OpQuantConfig with correct SmoothQuant per-role transforms.

    SmoothQuant (Xiao et al., 2023) applies per-channel scaling to both
    activations and weights to maintain mathematical equivalence::

        (X / s) @ (W * s) = X @ W

    - **input** role: ``SmoothQuantTransform`` — ``forward(x) = x / s``
    - **weight** role: ``SmoothQuantWeightTransform`` — ``forward(W) = W * s``
    - **output** role: ``IdentityTransform`` (no transform; the matmul output
      is already compensated by the inverse transforms during dequant)

    Args:
        fmt_name: Format name string.
        granularity: ``GranularitySpec``.
        sq_transform: The per-layer ``SmoothQuantTransform`` (activation-side,
            carries the shared scale ``s``).
        weight_only: If True, only ``input`` + ``weight`` are quantized
            (used for per-channel formats like NF4, INT4-PC).

    Returns:
        ``OpQuantConfig`` with activation-side SmoothQuant on ``input``,
        weight-side compensation on ``weight``, and ``IdentityTransform``
        on ``output``.
    """
    fmt = FormatBase.from_str(fmt_name)
    # Weight compensation: W * s (same scale, inverse operation)
    w_transform = SmoothQuantWeightTransform(sq_transform.scale)
    input_scheme = QuantScheme(
        format=fmt, granularity=granularity, transform=sq_transform,
    )
    weight_scheme = QuantScheme(
        format=fmt, granularity=granularity, transform=w_transform,
    )
    no_tx = IdentityTransform()
    if weight_only:
        return OpQuantConfig(input=input_scheme, weight=weight_scheme)
    else:
        output_scheme = QuantScheme(
            format=fmt, granularity=granularity, transform=no_tx,
        )
        return OpQuantConfig(input=input_scheme, weight=weight_scheme, output=output_scheme)


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
        observers = [
            QSNRObserver(), MSEObserver(),
            HistogramObserver(), DistributionObserver(),
        ]

    if not calib_data:
        raise ValueError("calib_data must contain at least one batch")

    # Deep-copy fp32_model before QuantSession mutates it in-place via
    # quantize_model().  Without this copy, repeated calls with the same
    # fp32_model reference would pass an already-quantized model on the
    # second and later runs, producing identical QSNR for all configs.
    session = QuantSession(
        copy.deepcopy(fp32_model), cfg,
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
    *,
    eval_fn: Optional[Callable] = None,
) -> dict:
    """Part A: Compare MXINT-8, MXFP-8, INT8-PC (all 8-bit, PoT scaling).

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.
        eval_fn: Optional custom evaluation function.

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
        results[name] = run_experiment(
            cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results


def run_part_b_4bit(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
    *,
    eval_fn: Optional[Callable] = None,
) -> dict:
    """Part B: Compare MXINT-4, MXFP-4, INT4-PC, NF4-PC (all 4-bit).

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.
        eval_fn: Optional custom evaluation function.

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
        results[name] = run_experiment(
            cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results


def run_part_c_pot_scaling(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
    *,
    eval_fn: Optional[Callable] = None,
) -> dict:
    """Part C: INT per-channel with FP32 scaling vs PoT scaling (8-bit & 4-bit).

    Compares power-of-two (PoT) constrained learning scales against
    unconstrained FP32 scales for both 8-bit and 4-bit INT per-channel
    quantization, using LSQ (Learned Step-size Quantization) optimization.

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.
        eval_fn: Optional custom evaluation function.

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
            lsq_steps=100, lsq_pot=pot, eval_fn=eval_fn,
        )
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results


def _make_smoothquant_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
) -> Dict[str, TransformBase]:
    """Create per-layer SmoothQuantTransform dict.

    Runs one forward pass through the FP32 model to capture each layer's
    activation and weight, then creates a per-layer SmoothQuantTransform
    with correctly-shaped per-channel scales.

    Args:
        fp32_model: FP32 reference model.
        calib_data: List of calibration batches (first batch used).

    Returns:
        Dict mapping layer name to ``SmoothQuantTransform`` (or
        ``IdentityTransform`` on failure).
    """
    if fp32_model is None:
        return {}

    activations: Dict[str, torch.Tensor] = {}
    weights: Dict[str, torch.Tensor] = {}
    hooks = []

    def _hook(name):
        def fn(module, _input, _output):
            activations[name] = _input[0].detach()
            if hasattr(module, "weight") and module.weight is not None:
                weights[name] = module.weight.data
        return fn

    for name, module in fp32_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(_hook(name)))

    with torch.no_grad():
        fp32_model.eval()
        fp32_model(calib_data[0])

    for h in hooks:
        h.remove()

    per_layer: Dict[str, TransformBase] = {}
    for name in activations:
        if name in weights:
            try:
                per_layer[name] = SmoothQuantTransform.from_calibration(
                    X_act=activations[name], W=weights[name], alpha=0.5,
                )
            except (ValueError, RuntimeError) as e:
                print(f"  Warning: SmoothQuant for {name}: {e}")
                per_layer[name] = IdentityTransform()

    return per_layer


def _build_per_layer_optimal_cfg(
    variant_results: dict,
    sq_transforms: dict,
    fmt_str: str,
    gran: GranularitySpec,
    cfg_builder: Callable,
    weight_only: bool = False,
) -> dict:
    """Build per-layer OpQuantConfig dict choosing best transform per layer by QSNR.

    For each layer in the model, selects the transform variant (None, SmoothQuant,
    or Hadamard) that achieves the highest QSNR score from the variant experiments.

    Args:
        variant_results: Dict mapping ``"None"``, ``"SmoothQuant"``, ``"Hadamard"``
            to their experiment result dicts (which contain ``qsnr_per_layer``).
        sq_transforms: Per-layer SmoothQuantTransform dict from
            ``_make_smoothquant_transforms``.
        fmt_str: Format name string for the config builder.
        gran: ``GranularitySpec`` for the config builder.
        cfg_builder: ``make_op_cfg`` or ``make_op_cfg_weight_only``.
        weight_only: Whether the format is weight-only (NF4, INT4-PC).

    Returns:
        Dict mapping layer name to ``OpQuantConfig``.
    """
    variant_qsnr = {k: v["qsnr_per_layer"] for k, v in variant_results.items()}
    layer_best_tx = _compute_best_transform_per_layer(variant_qsnr)

    tx_map = {
        "None": None,
        "Hadamard": HadamardTransform(),
    }

    per_layer_cfg = {}
    for layer, tx_name in layer_best_tx.items():
        if tx_name == "SmoothQuant":
            sq_tx = sq_transforms.get(layer, IdentityTransform())
            per_layer_cfg[layer] = _make_sq_op_cfg(fmt_str, gran, sq_tx, weight_only)
        else:
            per_layer_cfg[layer] = cfg_builder(fmt_str, gran, transform=tx_map[tx_name])

    return per_layer_cfg


def run_part_d_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
    *,
    eval_fn: Optional[Callable] = None,
) -> dict:
    """Part D: Evaluate SmoothQuant and Hadamard transforms at 4-bit,
    with per-layer optimal transform selection by QSNR.

    For each 4-bit format (MXINT-4, MXFP-4, INT4-PC, NF4-PC), runs three
    transform variants (No transform, SmoothQuant, Hadamard), then builds
    a per-layer optimal configuration that picks the best transform per
    layer based on QSNR.

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.
        eval_fn: Optional custom evaluation function.

    Returns:
        Nested dict mapping format name to dict of variant name to result dict.
        Each variant entry includes an additional ``PerLayerOpt`` entry with
        the per-layer optimal transform result.
    """
    print("\n### Part D: Transform Study at 4-bit ###")
    fmt_configs = [
        ("MXINT-4", "int4", PER_B32, False),
        ("MXFP-4",  "fp4_e2m1", PER_B32, False),
        ("INT4-PC", "int4", PER_C0, False),
        ("NF4-PC",  "nf4", PER_C0, True),
    ]

    all_results = {}

    for fmt_name, fmt_str, gran, weight_only in fmt_configs:
        print(f"\n  == Transform study for {fmt_name} ==")
        builder = make_op_cfg_weight_only if weight_only else make_op_cfg

        # Phase 1: Run each transform variant
        variant_results = {}
        sq_transforms = _make_smoothquant_transforms(fp32_model, calib_data)
        # For the SmoothQuant variant, build per-layer configs so each layer
        # gets its own correctly-shaped companion transform pair:
        #   activation: x / s → quantize → x_q * s  (SmoothQuantTransform)
        #   weight:     W * s → quantize → W_q / s  (SmoothQuantWeightTransform)
        # This preserves mathematical equivalence: (X/s) @ (W*s) = X@W
        sq_per_layer_cfg = {}
        for lname, sq_tx in sq_transforms.items():
            sq_per_layer_cfg[lname] = _make_sq_op_cfg(fmt_str, gran, sq_tx, weight_only)

        # Fallback: non-Linear/Conv modules (e.g. LayerNorm, GELU) that
        # aren't in sq_transforms still need a quantization config — without
        # this, quantize_model gives them _EMPTY_CFG (no quantization at
        # all), which would inflate the SmoothQuant variant's QSNR.
        fallback_cfg = builder(fmt_str, gran, transform=None)
        for mname, _module in fp32_model.named_modules():
            if mname and mname not in sq_per_layer_cfg:
                sq_per_layer_cfg[mname] = fallback_cfg

        # Single-transform variants: build a single OpQuantConfig
        single_tx = {
            "None": None,
            "Hadamard": HadamardTransform(),
        }
        for tx_name, tx in single_tx.items():
            label = f"{fmt_name}-{tx_name}"
            print(f"    Running {label}...")
            cfg = builder(fmt_str, gran, transform=tx)
            variant_results[tx_name] = run_experiment(
                cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
            )

        # SmoothQuant: each layer has its own correctly-shaped transform
        print(f"    Running {fmt_name}-SmoothQuant...")
        variant_results["SmoothQuant"] = run_experiment(
            sq_per_layer_cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )

        # Phase 2: Per-layer optimal
        per_layer_optimal_cfg = _build_per_layer_optimal_cfg(
            variant_results, sq_transforms, fmt_str, gran, builder, weight_only,
        )
        print(f"    Running {fmt_name}-PerLayerOpt...")
        opt_result = run_experiment(
            per_layer_optimal_cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        variant_results["PerLayerOpt"] = opt_result

        all_results[fmt_name] = variant_results

    return all_results


# ---------------------------------------------------------------------------
# Task 7: Table Generation (6 tables)
# ---------------------------------------------------------------------------

def _accuracy_table(results: dict, title: str, output_dir: str, filename: str) -> str:
    """Generic: format accuracy + avg QSNR/MSE table from a results dict."""
    rows = []
    for name, data in results.items():
        acc = data.get("accuracy", {})
        if isinstance(acc, dict) and len(acc) == 1:
            # Single-metric: extract the raw value for clean CSV
            acc_val = list(acc.values())[0]
            acc_str = f"{acc_val:.4f}"
        elif isinstance(acc, dict):
            acc_str = ", ".join(f"{k}: {v:.4f}" for k, v in acc.items())
        elif isinstance(acc, (int, float)):
            acc_str = f"{acc:.4f}"
        else:
            acc_str = str(acc)
        qsnr_dict = data.get("qsnr_per_layer", {})
        mse_dict = data.get("mse_per_layer", {})
        avg_qsnr = sum(qsnr_dict.values()) / max(len(qsnr_dict), 1)
        avg_mse = sum(mse_dict.values()) / max(len(mse_dict), 1)
        rows.append((name, acc_str, avg_qsnr, avg_mse))

    header = f"\n{'='*70}\n{title}\n{'='*70}\n"
    header += f"{'Config':<20} {'Accuracy':<20} {'Avg QSNR (dB)':<15} {'Avg MSE':<15}\n"
    header += "-" * 70 + "\n"
    for row in rows:
        header += f"{row[0]:<20} {row[1]:<20} {row[2]:<15.2f} {row[3]:<15.6f}\n"

    os.makedirs(os.path.dirname(csv_path := f"{output_dir}/tables/{filename}"), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("Config,Accuracy,Avg_QSNR_dB,Avg_MSE\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.6f}\n")

    return header


def generate_table_1(part_a: dict, output_dir: str) -> str:
    """Table 1: 8-bit Format Comparison."""
    return _accuracy_table(
        part_a, "Table 1: 8-bit Format Comparison", output_dir, "table1_8bit.csv",
    )


def generate_table_2(part_b: dict, output_dir: str) -> str:
    """Table 2: 4-bit Format Comparison."""
    return _accuracy_table(
        part_b, "Table 2: 4-bit Format Comparison", output_dir, "table2_4bit.csv",
    )


def generate_table_3(part_c: dict, output_dir: str) -> str:
    """Table 3: FP32 vs PoT accuracy delta.

    Shows each config's accuracy, delta from FP32 baseline, avg QSNR and MSE.
    """
    # Find baseline
    baseline_acc = 0.0
    for name, data in part_c.items():
        if "baseline" in name.lower():
            acc = data.get("accuracy", {})
            if isinstance(acc, dict):
                baseline_acc = float(acc.get("accuracy", 0.0))
            elif isinstance(acc, (int, float)):
                baseline_acc = float(acc)
            break

    rows = []
    for name, data in part_c.items():
        if "baseline" in name.lower():
            continue
        acc = data.get("accuracy", {})
        if isinstance(acc, dict):
            acc_val = float(acc.get("accuracy", 0.0))
            acc_str = ", ".join(f"{k}: {v:.4f}" for k, v in acc.items())
        else:
            acc_val = float(acc) if isinstance(acc, (int, float)) else 0.0
            acc_str = f"{acc_val:.4f}"

        delta = acc_val - baseline_acc
        qsnr_dict = data.get("qsnr_per_layer", {})
        mse_dict = data.get("mse_per_layer", {})
        avg_qsnr = sum(qsnr_dict.values()) / max(len(qsnr_dict), 1)
        avg_mse = sum(mse_dict.values()) / max(len(mse_dict), 1)
        rows.append((name, acc_str, delta, avg_qsnr, avg_mse))

    lines = [f"\n{'='*85}", "Table 3: FP32 vs PoT Scaling", '=' * 85]
    lines.append(f"{'Config':<20} {'Accuracy':<20} {'Delta':<12} "
                 f"{'Avg QSNR (dB)':<15} {'Avg MSE':<15}")
    lines.append("-" * 85)
    for row in rows:
        lines.append(f"{row[0]:<20} {row[1]:<20} {row[2]:<+12.4f} "
                     f"{row[3]:<15.2f} {row[4]:<15.6f}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table3_pot.csv", "w") as f:
        f.write("Config,Accuracy,Delta,Avg_QSNR_dB,Avg_MSE\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.4f},{row[4]:.6f}\n")
    return result


def generate_table_4(part_d: dict, output_dir: str) -> str:
    """Table 4: Format x Transform accuracy matrix."""
    fmt_names = sorted(part_d.keys())
    tx_variants = sorted({tx for fmt_data in part_d.values() for tx in fmt_data})

    def _get_acc(fmt_data: dict, tx: str) -> float:
        if tx not in fmt_data:
            return float("nan")
        acc = fmt_data[tx].get("accuracy", {})
        if isinstance(acc, dict):
            return float(acc.get("accuracy", 0.0))
        return float(acc) if isinstance(acc, (int, float)) else float("nan")

    lines = [f"\n{'='*80}", "Table 4: Format x Transform Accuracy Matrix", '=' * 80]
    header = f"{'Format':<16}"
    for tx in tx_variants:
        header += f" {tx:<20}"
    lines.append(header)
    lines.append("-" * len(header))

    for fmt_name in fmt_names:
        row_str = f"{fmt_name:<16}"
        fmt_data = part_d[fmt_name]
        for tx in tx_variants:
            val = _get_acc(fmt_data, tx)
            if math.isnan(val):
                row_str += f" {'N/A':<20}"
            else:
                row_str += f" {val:<20.4f}"
        lines.append(row_str)
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table4_format_x_transform.csv", "w") as f:
        f.write("Format," + ",".join(tx_variants) + "\n")
        for fmt_name in fmt_names:
            vals = []
            for tx in tx_variants:
                val = _get_acc(part_d[fmt_name], tx)
                vals.append(f"{val:.6f}" if not math.isnan(val) else "N/A")
            f.write(f"{fmt_name}," + ",".join(vals) + "\n")
    return result


def generate_table_5(part_d: dict, output_dir: str) -> str:
    """Table 5: Per-layer optimal transform distribution.

    Counts how many layers picked each transform as the best by QSNR.
    """
    distribution: Dict[str, Dict[str, int]] = {}
    all_tx_set: set = set()

    for fmt_name, fmt_data in part_d.items():
        variant_qsnr: Dict[str, Dict[str, float]] = {}
        for tx_name in ("None", "SmoothQuant", "Hadamard"):
            if tx_name in fmt_data and "qsnr_per_layer" in fmt_data[tx_name]:
                variant_qsnr[tx_name] = fmt_data[tx_name]["qsnr_per_layer"]

        layer_best_tx = _compute_best_transform_per_layer(variant_qsnr)

        tx_counts: Dict[str, int] = defaultdict(int)
        for best_tx in layer_best_tx.values():
            tx_counts[best_tx] += 1

        distribution[fmt_name] = dict(tx_counts)
        all_tx_set.update(tx_counts.keys())

    all_tx = sorted(all_tx_set)
    lines = [f"\n{'='*80}", "Table 5: Per-Layer Optimal Transform Distribution", '=' * 80]
    hdr = f"{'Format':<16}"
    for tx in all_tx:
        hdr += f" {tx:<18}"
    hdr += " Total"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for fmt_name in sorted(distribution.keys()):
        r = f"{fmt_name:<16}"
        total = 0
        for tx in all_tx:
            cnt = distribution[fmt_name].get(tx, 0)
            r += f" {cnt:<18}"
            total += cnt
        r += f" {total}"
        lines.append(r)
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table5_transform_distribution.csv", "w") as f:
        f.write("Format," + ",".join(all_tx) + ",Total\n")
        for fmt_name in sorted(distribution.keys()):
            vals = [str(distribution[fmt_name].get(tx, 0)) for tx in all_tx]
            vals.append(str(sum(distribution[fmt_name].values())))
            f.write(f"{fmt_name}," + ",".join(vals) + "\n")
    return result


def generate_table_6(all_results: dict, output_dir: str) -> str:
    """Table 6: Layer sensitivity top-10 across all experiments."""
    layer_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: {"mse": [], "qsnr": []})

    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict):
                continue
            for key in ("qsnr_per_layer", "mse_per_layer"):
                if key not in config_data:
                    continue
                metric = "qsnr" if "qsnr" in key else "mse"
                for layer, val in config_data[key].items():
                    layer_metrics[layer][metric].append(val)

    ranking = []
    for layer, metrics in layer_metrics.items():
        avg_mse = sum(metrics["mse"]) / max(len(metrics["mse"]), 1) if metrics["mse"] else 0.0
        avg_qsnr = sum(metrics["qsnr"]) / max(len(metrics["qsnr"]), 1) if metrics["qsnr"] else 0.0
        ranking.append((layer, avg_mse, avg_qsnr))
    ranking.sort(key=lambda x: x[1], reverse=True)
    top10 = ranking[:10]

    lines = [f"\n{'='*80}", "Table 6: Top-10 Most Sensitive Layers", '=' * 80]
    lines.append(f"{'#':<4} {'Layer':<28} {'Avg MSE':<18} {'Avg QSNR (dB)':<15}")
    lines.append("-" * 80)
    for i, (layer, mse, qsnr) in enumerate(top10, 1):
        lines.append(f"{i:<4} {layer:<28} {mse:<18.6e} {qsnr:<15.2f}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table6_sensitivity.csv", "w") as f:
        f.write("Rank,Layer,Avg_MSE,Avg_QSNR_dB\n")
        for i, (layer, mse, qsnr) in enumerate(top10, 1):
            f.write(f"{i},{layer},{mse:.6e},{qsnr:.4f}\n")
    return result


# ---------------------------------------------------------------------------
# Task 8: Figure Generation (11 figures)
# ---------------------------------------------------------------------------

def _save_figure(fig, output_dir: str, name: str):
    """Save figure as PNG and PDF."""
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_dir}/figures/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_best_transform_per_layer(
    variant_qsnr: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """Return ``{layer_name: best_transform_name}`` by QSNR.

    For each layer, picks the transform variant (one of the dict keys in
    ``variant_qsnr``) that maximizes per-layer QSNR.  Ties go to the
    first transform encountered in dict insertion order.
    """
    all_layers: set = set()
    for qsnr_dict in variant_qsnr.values():
        all_layers.update(qsnr_dict.keys())
    result: Dict[str, str] = {}
    tx_names = list(variant_qsnr.keys())
    for layer in all_layers:
        result[layer] = max(
            tx_names,
            key=lambda tx: variant_qsnr[tx].get(layer, -float("inf")),
        )
    return result


def _get_acc_val(data) -> float:
    """Extract scalar accuracy value from a result dict entry.

    Returns ``float("nan")`` when the entry is missing or empty, so that
    tables and heatmaps can visually distinguish missing data from zero.
    """
    if not isinstance(data, dict) or not data:
        return float("nan")
    acc = data.get("accuracy", {})
    if isinstance(acc, dict):
        return float(acc.get("accuracy", float("nan")))
    if isinstance(acc, (int, float)):
        return float(acc)
    return float("nan")


def plot_fig1_qsnr_8bit(part_a: dict, output_dir: str):
    """Fig 1: 8-bit per-layer QSNR line chart (3 lines)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in part_a.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        layers = sorted(data["qsnr_per_layer"].keys())
        values = [data["qsnr_per_layer"][l] for l in layers]
        color = FORMAT_COLORS.get(name, FALLBACK_CYCLE[0])
        ax.plot(range(len(layers)), values, marker="o", label=name, linewidth=2,
                color=color)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("Fig 1: Per-Layer QSNR — 8-bit Formats")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_dir, "fig1_qsnr_8bit")


def plot_fig2_qsnr_4bit(part_b: dict, output_dir: str):
    """Fig 2: 4-bit per-layer QSNR line chart (4 lines)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in part_b.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        layers = sorted(data["qsnr_per_layer"].keys())
        values = [data["qsnr_per_layer"][l] for l in layers]
        color = FORMAT_COLORS.get(name, FALLBACK_CYCLE[0])
        ax.plot(range(len(layers)), values, marker="o", label=name, linewidth=2,
                color=color)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("Fig 2: Per-Layer QSNR — 4-bit Formats")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_dir, "fig2_qsnr_4bit")


def plot_fig3_mse_box_8bit(part_a: dict, output_dir: str):
    """Fig 3: 8-bit per-layer MSE boxplot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot, labels = [], []
    colors = []
    for name, data in part_a.items():
        if "baseline" in name.lower() or "mse_per_layer" not in data:
            continue
        mse_vals = list(data["mse_per_layer"].values())
        if mse_vals:
            data_to_plot.append(mse_vals)
            labels.append(name)
            colors.append(FORMAT_COLORS.get(name, FALLBACK_CYCLE[0]))
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax.set_ylabel("MSE")
    ax.set_title("Fig 3: Per-Layer MSE Distribution — 8-bit Formats")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_dir, "fig3_mse_8bit")


def plot_fig4_mse_box_4bit(part_b: dict, output_dir: str):
    """Fig 4: 4-bit per-layer MSE boxplot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot, labels, colors = [], [], []
    for name, data in part_b.items():
        if "baseline" in name.lower() or "mse_per_layer" not in data:
            continue
        mse_vals = list(data["mse_per_layer"].values())
        if mse_vals:
            data_to_plot.append(mse_vals)
            labels.append(name)
            colors.append(FORMAT_COLORS.get(name, FALLBACK_CYCLE[0]))
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax.set_ylabel("MSE")
    ax.set_title("Fig 4: Per-Layer MSE Distribution — 4-bit Formats")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_dir, "fig4_mse_4bit")


def plot_fig5_pot_delta(part_c: dict, output_dir: str):
    """Fig 5: FP32 vs PoT per-layer QSNR delta bar chart."""
    # Group by format base (INT8-PC, INT4-PC)
    formats: Dict[str, dict] = {}
    for name, data in part_c.items():
        if "baseline" in name.lower():
            continue
        base = name.rsplit("-", 1)[0]
        is_pot = "PoT" in name
        formats.setdefault(base, {})[is_pot] = data

    n_groups = len(formats)
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 5),
                             squeeze=False)
    for idx, (fmt_name, fmt_data) in enumerate(sorted(formats.items())):
        ax = axes[0, idx]
        fp32_qsnr = fmt_data.get(False, {}).get("qsnr_per_layer", {})
        pot_qsnr = fmt_data.get(True, {}).get("qsnr_per_layer", {})

        all_layers = sorted(set(list(fp32_qsnr.keys()) + list(pot_qsnr.keys())))
        deltas = [pot_qsnr.get(l, 0) - fp32_qsnr.get(l, 0) for l in all_layers]
        layer_names = [l.replace("module.", "").replace("Quantized", "")
                       for l in all_layers]

        colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.7)
        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("QSNR Delta (PoT – FP32) [dB]")
        ax.set_title(f"{fmt_name}")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 5: PoT Scaling vs FP32 Scaling — Per-Layer QSNR Delta",
                 fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig5_pot_delta")


def plot_fig6_histogram_overlay(all_results: dict, output_dir: str):
    """Fig 6: Three-channel histogram overlay (fp32 / quant / error).

    Extracts histogram data from ``HistogramObserver`` (keys: ``fp32_hist``,
    ``quant_hist``, ``err_hist`` — torch.histc counts) and renders the most
    sensitive layers as overlaid semi-transparent bar charts.
    """
    # Collect histogram data: {layer: {"fp32_hist": ..., "quant_hist": ..., "err_hist": ...}}
    layer_hists: Dict[str, dict] = {}
    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            if not hasattr(report, "_raw"):
                continue
            for layer, roles in report._raw.items():
                if layer in layer_hists:
                    continue
                for role, stages in roles.items():
                    for stage, slices in stages.items():
                        for metrics in slices.values():
                            if "fp32_hist" in metrics and "quant_hist" in metrics:
                                layer_hists[layer] = {
                                    k: metrics[k].cpu() if hasattr(metrics.get(k, None), "cpu")
                                    else metrics.get(k) for k in
                                    ("fp32_hist", "quant_hist", "err_hist")
                                }
                                break
                    if layer in layer_hists:
                        break

    if not layer_hists:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Histogram data not available\n"
                "(Add HistogramObserver to observers in run_experiment)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Fig 6: Activation Histograms (No Data)")
        _save_figure(fig, output_dir, "fig6_histogram")
        return

    # Pick top 3–5 layers with the richest histogram data
    top_layers = sorted(layer_hists.items(),
                        key=lambda x: x[1].get("fp32_hist",
                         torch.tensor(0)).sum().item(), reverse=True)[:5]
    if not top_layers:
        return

    n = len(top_layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (layer, hist_data) in zip(axes[0], top_layers):
        n_bins = 128
        for channel, color, label in [
            ("fp32_hist", "#3498db", "fp32"),
            ("quant_hist", "#e74c3c", "quant"),
            ("err_hist", "#95a5a6", "error"),
        ]:
            counts = hist_data.get(channel)
            if counts is None or not isinstance(counts, (torch.Tensor, np.ndarray)):
                continue
            if isinstance(counts, torch.Tensor):
                counts = counts.float().numpy()
            # torch.histc returns counts for equal-width bins; use index as
            # approximate x-axis (the relative shape matters more than absolute
            # values for visual comparison across channels)
            bin_centers = np.arange(len(counts))
            ax.fill_between(bin_centers, counts, alpha=0.35, color=color,
                            label=label, step="mid")
            ax.plot(bin_centers, counts, color=color, linewidth=0.8)
        ax.set_title(layer, fontsize=9)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 6: Activation Histograms (fp32 / quant / error) — "
                 "Most Sensitive Layers", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig6_histogram")


def plot_fig7_transform_heatmap(part_d: dict, output_dir: str):
    """Fig 7: Format x Transform heatmap."""
    fmt_names = sorted(part_d.keys())
    tx_variants = sorted({tx for fmt_data in part_d.values()
                          for tx in fmt_data})

    matrix = []
    for fmt_name in fmt_names:
        row = []
        for tx in tx_variants:
            row.append(_get_acc_val(part_d[fmt_name].get(tx, {})))
        matrix.append(row)

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = arr[~np.isnan(arr)]
    if len(valid) > 0:
        vmin, vmax = valid.min(), valid.max()
    else:
        vmin, vmax = 0.0, 1.0
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#d3d3d3")  # gray for missing variants
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(tx_variants)))
    ax.set_xticklabels(tx_variants, rotation=45, ha="right")
    ax.set_yticks(range(len(fmt_names)))
    ax.set_yticklabels(fmt_names)

    for i in range(len(fmt_names)):
        for j in range(len(tx_variants)):
            val = matrix[i][j]
            if not math.isnan(val):
                mid = (vmin + vmax) / 2
                text_color = "white" if val < mid else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_title("Fig 7: Format x Transform Accuracy Matrix")
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig7_transform_heatmap")


def plot_fig8_transform_pie(part_d: dict, output_dir: str):
    """Fig 8: Per-layer optimal transform distribution pie chart."""
    n_fmts = len(part_d)
    fig, axes = plt.subplots(1, max(n_fmts, 1),
                             figsize=(5 * max(n_fmts, 1), 5),
                             subplot_kw={"aspect": "equal"})
    if n_fmts == 1:
        axes = [axes]

    pie_colors = TRANSFORM_COLORS

    for ax, (fmt_name, fmt_data) in zip(axes, sorted(part_d.items())):
        if "PerLayerOpt" not in fmt_data:
            ax.text(0.5, 0.5, "No PerLayerOpt data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        variant_qsnr: Dict[str, Dict[str, float]] = {}
        for tx_name in ("None", "SmoothQuant", "Hadamard"):
            if tx_name in fmt_data and "qsnr_per_layer" in fmt_data[tx_name]:
                variant_qsnr[tx_name] = fmt_data[tx_name]["qsnr_per_layer"]

        layer_best_tx = _compute_best_transform_per_layer(variant_qsnr)

        tx_counts: Dict[str, int] = defaultdict(int)
        for best_tx in layer_best_tx.values():
            tx_counts[best_tx] += 1

        labels = list(tx_counts.keys())
        sizes = list(tx_counts.values())
        colors = [pie_colors.get(l, "#95a5a6") for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            colors=colors, startangle=90,
            textprops={"fontsize": 9},
        )
        total = sum(sizes)
        ax.set_title(f"{fmt_name} (n={total})", fontsize=10)

    fig.suptitle("Fig 8: Per-Layer Optimal Transform Distribution", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig8_transform_pie")


def plot_fig9_transform_delta(part_d: dict, output_dir: str):
    """Fig 9: Transform delta QSNR vs baseline, one subplot per format.

    Each format gets its own subplot so that formats with different layer
    counts (e.g. ToyMLP vs transformer) do not produce overlapping bars.
    """
    fmt_names = sorted(part_d.keys())
    n_fmts = len(fmt_names)
    fig, axes = plt.subplots(n_fmts, 1, figsize=(14, 4 * n_fmts), sharex=False)
    if n_fmts == 1:
        axes = [axes]
    colors_tx = TRANSFORM_COLORS

    for ax, fmt_name in zip(axes, fmt_names):
        fmt_data = part_d[fmt_name]
        if "None" not in fmt_data:
            ax.text(0.5, 0.5, "No baseline data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        baseline_qsnr = fmt_data["None"].get("qsnr_per_layer", {})

        x_pos = 0
        tick_positions, tick_labels = [], []
        for tx_name in ("SmoothQuant", "Hadamard"):
            if tx_name not in fmt_data or "qsnr_per_layer" not in fmt_data[tx_name]:
                continue
            tx_qsnr = fmt_data[tx_name]["qsnr_per_layer"]
            all_layers = sorted(set(baseline_qsnr.keys()) | set(tx_qsnr.keys()))
            deltas = [tx_qsnr.get(l, 0) - baseline_qsnr.get(l, 0) for l in all_layers]

            bar_positions = list(range(x_pos, x_pos + len(all_layers)))
            color = colors_tx.get(tx_name, "#95a5a6")
            ax.bar(bar_positions, deltas, color=color, alpha=0.6,
                   label=tx_name)
            tick_positions.append((bar_positions[0] + bar_positions[-1]) / 2
                                  if bar_positions else x_pos)
            tick_labels.append(tx_name)
            x_pos += len(all_layers) + 2
            if len(all_layers) <= 20:
                for i, layer in enumerate(all_layers):
                    ax.text(bar_positions[i], deltas[i],
                            layer.split(".")[-1] if "." in layer else layer,
                            ha="center", va="bottom" if deltas[i] >= 0 else "top",
                            fontsize=4, rotation=90)

        ax.axhline(y=0, color="black", linewidth=0.5)
        if tick_positions:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel("QSNR Delta (dB)")
        ax.set_title(f"{fmt_name}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Fig 9: Transform Impact on Per-Layer QSNR", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig9_transform_delta")


def plot_fig10_error_vs_distribution(all_results: dict, output_dir: str):
    """Fig 10: QSNR vs distribution features scatter (4-panel)."""
    data_points: list = []

    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            if not hasattr(report, "_raw"):
                continue
            for layer, roles in report._raw.items():
                for role, stages in roles.items():
                    for stage, slices in stages.items():
                        for metrics in slices.values():
                            if "qsnr_db" not in metrics or "dynamic_range_bits" not in metrics:
                                continue
                            data_points.append({
                                "qsnr": metrics["qsnr_db"],
                                "dynamic_range": metrics["dynamic_range_bits"],
                                "skewness": metrics.get("skewness", 0),
                                "kurtosis": metrics.get("kurtosis", 0),
                                "sparse_ratio": metrics.get("sparse_ratio", 0),
                                "layer": layer,
                                "role": role,
                                "mse": metrics.get("mse", 1e-10),
                            })

    if not data_points:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
                "Distribution data not available\n"
                "(No DistributionObserver in reports)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Fig 10: QSNR vs Distribution Features")
        _save_figure(fig, output_dir, "fig10_error_vs_dist")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: QSNR vs Dynamic Range (color = sparse_ratio)
    ax = axes[0, 0]
    dr_vals = [d["dynamic_range"] for d in data_points]
    qsnr_vals = [d["qsnr"] for d in data_points]
    sparse_vals = [d["sparse_ratio"] for d in data_points]
    sc = ax.scatter(dr_vals, qsnr_vals, c=sparse_vals,
                    cmap="viridis", alpha=0.6, s=30)
    ax.set_xlabel("Dynamic Range (bits)")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("QSNR vs Dynamic Range\n(color = sparse ratio)")
    fig.colorbar(sc, ax=ax)
    ax.grid(True, alpha=0.3)

    # Panel 2: QSNR vs Skewness (color = kurtosis)
    ax = axes[0, 1]
    skew_vals = [d["skewness"] for d in data_points]
    kurt_vals = [d["kurtosis"] for d in data_points]
    sc = ax.scatter(skew_vals, qsnr_vals, c=kurt_vals,
                    cmap="plasma", alpha=0.6, s=30)
    ax.set_xlabel("Skewness")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("QSNR vs Skewness\n(color = kurtosis)")
    fig.colorbar(sc, ax=ax)
    ax.grid(True, alpha=0.3)

    # Panel 3: MSE (dB) vs Dynamic Range
    ax = axes[1, 0]
    mse_db = [10 * math.log10(max(d["mse"], 1e-20)) for d in data_points]
    ax.scatter(dr_vals, mse_db, alpha=0.6, s=30, c="#e74c3c")
    ax.set_xlabel("Dynamic Range (bits)")
    ax.set_ylabel("MSE (dB)")
    ax.set_title("MSE vs Dynamic Range")
    ax.grid(True, alpha=0.3)

    # Panel 4: Sparsity histogram
    ax = axes[1, 1]
    ax.hist(sparse_vals, bins=20, alpha=0.7, color=FALLBACK_CYCLE[0],
            edgecolor="white")
    ax.set_xlabel("Sparse Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Sparsity Across Layers")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 10: Quantization Error vs Distribution Features", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig10_error_vs_dist")


def plot_fig11_layer_type_qsnr(all_results: dict, output_dir: str):
    """Fig 11: Layer-type grouped QSNR comparison using LayerSensitivity.

    Note:
        This figure degrades for models with sparse layer-type diversity
        (e.g. ToyMLP / MLP-only architectures) because the ``by_layer_type``
        grouping collapses to a single category (``"Linear"``), producing
        boxplots with only one box per panel.
    """
    ltype_qsnr: Dict[str, list] = defaultdict(list)
    ltype_mse: Dict[str, list] = defaultdict(list)

    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            ls = LayerSensitivity(report)
            by_type = ls.by_layer_type()
            for lt, stats in by_type.items():
                ltype_qsnr[lt].append(stats["avg_qsnr_db"])
                ltype_mse[lt].append(stats["avg_mse"])

    if not ltype_qsnr:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Layer type data not available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Fig 11: Layer-Type Grouped Quantization Error")
        _save_figure(fig, output_dir, "fig11_layer_type_qsnr")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_cycle = FALLBACK_CYCLE
    labels = list(ltype_qsnr.keys())

    # QSNR boxplot
    ax = axes[0]
    qsnr_data = [ltype_qsnr[lt] for lt in labels]
    bp = ax.boxplot(qsnr_data, tick_labels=labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors_cycle[i % len(colors_cycle)])
        patch.set_alpha(0.6)
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("Avg QSNR by Layer Type")
    ax.grid(True, alpha=0.3)

    # MSE boxplot (log scale)
    ax = axes[1]
    mse_data = [ltype_mse[lt] for lt in labels]
    bp2 = ax.boxplot(mse_data, tick_labels=labels, patch_artist=True)
    for i, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor(colors_cycle[i % len(colors_cycle)])
        patch.set_alpha(0.6)
    ax.set_ylabel("MSE")
    ax.set_title("Avg MSE by Layer Type")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 11: Layer-Type Grouped Quantization Error", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig11_layer_type_qsnr")


def run_format_study(
    build_model: Callable[[], nn.Module],
    make_calib_data: Callable[..., List[torch.Tensor]],
    make_eval_loader: Callable[..., DataLoader],
    eval_fn: Callable[[nn.Module, DataLoader], Dict[str, float]],
    *,
    output_dir: Optional[str] = None,
    skip_parts: Optional[Dict[str, bool]] = None,
) -> Dict[str, dict]:
    """Main entry point: run all 27 experiments and produce tables/figures.

    Executes Parts A/B/C/D of the study, saving results to a timestamped
    directory under ``output_dir`` (default: ``results/``).

    Args:
        build_model: Callable that returns a fresh FP32 model.
        make_calib_data: Callable for calibration data.
        make_eval_loader: Callable for evaluation DataLoader.
        eval_fn: Evaluation function (model, dataloader) -> metric dict.
        output_dir: Output directory.  ``None`` uses ``"results"``.
        skip_parts: Dict mapping part name (``"A"``, ``"B"``, ``"C"``,
            ``"D"``) to ``True`` to skip that part.  ``None`` runs all.

    Returns:
        Dict mapping experiment name to result dict.
    """
    if skip_parts is None:
        skip_parts = {}

    if output_dir is None:
        output_dir = f"results/format_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)

    print("=" * 60)
    print("Quantization Format Precision Study")
    print(f"Output: {output_dir}")
    print("=" * 60)

    fp32_model = build_model()
    fp32_model.eval()
    calib_data = make_calib_data()
    eval_loader = make_eval_loader()

    all_results: Dict[str, dict] = {}

    if not skip_parts.get("A"):
        print("\n" + "=" * 60)
        print("PART A: 8-bit Format Comparison")
        print("=" * 60)
        all_results["part_a"] = run_part_a_8bit(
            fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        print(generate_table_1(all_results["part_a"], output_dir))
    else:
        print("\n### PART A: SKIPPED ###")

    if not skip_parts.get("B"):
        print("\n" + "=" * 60)
        print("PART B: 4-bit Format Comparison")
        print("=" * 60)
        all_results["part_b"] = run_part_b_4bit(
            fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        print(generate_table_2(all_results["part_b"], output_dir))
    else:
        print("\n### PART B: SKIPPED ###")

    if not skip_parts.get("C"):
        print("\n" + "=" * 60)
        print("PART C: FP32 vs PoT Scaling")
        print("=" * 60)
        all_results["part_c"] = run_part_c_pot_scaling(
            fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        print(generate_table_3(all_results["part_c"], output_dir))
    else:
        print("\n### PART C: SKIPPED ###")

    if not skip_parts.get("D"):
        print("\n" + "=" * 60)
        print("PART D: Transform Study")
        print("=" * 60)
        all_results["part_d"] = run_part_d_transforms(
            fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        print(generate_table_4(all_results["part_d"], output_dir))
        print(generate_table_5(all_results["part_d"], output_dir))
    else:
        print("\n### PART D: SKIPPED ###")

    # Table 6: Layer sensitivity
    print(generate_table_6(all_results, output_dir))

    # Block size sensitivity sweep
    print("\n" + "=" * 60)
    print("BLOCK SIZE SWEEP")
    print("=" * 60)
    all_results["block_sweep"] = run_block_size_sweep(
        fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
    )

    # Figures
    print("\n### Generating Figures ###")
    plot_tasks = [
        (plot_fig1_qsnr_8bit, "part_a", "fig1_qsnr_8bit"),
        (plot_fig2_qsnr_4bit, "part_b", "fig2_qsnr_4bit"),
        (plot_fig3_mse_box_8bit, "part_a", "fig3_mse_8bit"),
        (plot_fig4_mse_box_4bit, "part_b", "fig4_mse_4bit"),
        (plot_fig5_pot_delta, "part_c", "fig5_pot_delta"),
        (plot_fig6_histogram_overlay, None, "fig6_histogram"),
        (plot_fig7_transform_heatmap, "part_d", "fig7_transform_heatmap"),
        (plot_fig8_transform_pie, "part_d", "fig8_transform_pie"),
        (plot_fig9_transform_delta, "part_d", "fig9_transform_delta"),
        (plot_fig10_error_vs_distribution, None, "fig10_error_vs_dist"),
        (plot_fig11_layer_type_qsnr, None, "fig11_layer_type_qsnr"),
    ]
    for fn, part_key, name in plot_tasks:
        if part_key is not None and part_key not in all_results:
            print(f"  {name}: SKIPPED (part not run)")
            continue
        try:
            data = all_results if part_key is None else all_results[part_key]
            fn(data, output_dir)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # Save results JSON
    _save_results_json(all_results, output_dir)

    print(f"\nStudy complete. Results in {output_dir}/")
    return all_results


def _save_results_json(all_results: dict, output_dir: str):
    """Save a serializable subset of results to JSON."""
    serializable: Dict[str, dict] = {}
    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_"):
            continue
        serializable[part_name] = {}
        if isinstance(part_data, dict):
            for config_name, config_data in part_data.items():
                entry: Dict[str, dict] = {}
                if isinstance(config_data, dict):
                    if "accuracy" in config_data:
                        entry["accuracy"] = config_data["accuracy"]
                    if "qsnr_per_layer" in config_data:
                        entry["qsnr_per_layer"] = config_data["qsnr_per_layer"]
                    if "mse_per_layer" in config_data:
                        entry["mse_per_layer"] = config_data["mse_per_layer"]
                serializable[part_name][config_name] = entry

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  results.json: saved")


def run_block_size_sweep(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
    fmt_name: str = "int8",
    *,
    eval_fn: Optional[Callable] = None,
) -> dict:
    """Sweep block sizes for MX format sensitivity analysis.

    Args:
        fp32_model: Reference FP32 model.
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.
        fmt_name: Format name (default ``"int8"``).
        eval_fn: Optional custom evaluation function.

    Returns:
        Dict mapping ``"{fmt_name}-blk{size}"`` to experiment result dict.
    """
    results = {}
    for bs in (16, 32, 64, 128):
        label = f"{fmt_name}-blk{bs}"
        gran = GranularitySpec.per_block(size=bs, axis=-1)
        cfg = make_op_cfg(fmt_name, gran)
        print(f"  Block size {bs}...")
        try:
            results[label] = run_experiment(
                cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
            )
        except Exception as e:
            print(f"    FAILED: {e}")
    return results


# ---------------------------------------------------------------------------
# Results reload / redraw
# ---------------------------------------------------------------------------


def plot_from_results(results_path: str, output_dir: Optional[str] = None):
    """Reload saved results JSON and regenerate all tables and figures.

    Useful for tweaking figure aesthetics without re-running experiments.

    Args:
        results_path: Path to a ``results.json`` file saved by
            :func:`run_format_study`.
        output_dir: Where to write tables and figures.  Defaults to the
            directory containing ``results_path``.
    """
    if output_dir is None:
        output_dir = os.path.dirname(results_path)

    with open(results_path, "r") as f:
        all_results = json.load(f)

    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)

    print(f"Regenerating tables and figures from {results_path}")
    print(f"Output: {output_dir}")

    # Tables
    skipped_parts = set()
    if "part_a" in all_results:
        print(generate_table_1(all_results["part_a"], output_dir))
    else:
        skipped_parts.add("a")
    if "part_b" in all_results:
        print(generate_table_2(all_results["part_b"], output_dir))
    else:
        skipped_parts.add("b")
    if "part_c" in all_results:
        print(generate_table_3(all_results["part_c"], output_dir))
    else:
        skipped_parts.add("c")
    if "part_d" in all_results:
        print(generate_table_4(all_results["part_d"], output_dir))
        print(generate_table_5(all_results["part_d"], output_dir))
    else:
        skipped_parts.add("d")
    print(generate_table_6(all_results, output_dir))

    # Figures
    print("\n### Generating Figures ###")
    plot_tasks = [
        (plot_fig1_qsnr_8bit, "part_a", "fig1_qsnr_8bit"),
        (plot_fig2_qsnr_4bit, "part_b", "fig2_qsnr_4bit"),
        (plot_fig3_mse_box_8bit, "part_a", "fig3_mse_8bit"),
        (plot_fig4_mse_box_4bit, "part_b", "fig4_mse_4bit"),
        (plot_fig5_pot_delta, "part_c", "fig5_pot_delta"),
        (plot_fig6_histogram_overlay, None, "fig6_histogram"),
        (plot_fig7_transform_heatmap, "part_d", "fig7_transform_heatmap"),
        (plot_fig8_transform_pie, "part_d", "fig8_transform_pie"),
        (plot_fig9_transform_delta, "part_d", "fig9_transform_delta"),
        (plot_fig10_error_vs_distribution, None, "fig10_error_vs_dist"),
        (plot_fig11_layer_type_qsnr, None, "fig11_layer_type_qsnr"),
    ]
    for fn, part_key, name in plot_tasks:
        if part_key is not None and part_key not in all_results:
            print(f"  {name}: SKIPPED (part not in results)")
            continue
        try:
            data = all_results if part_key is None else all_results[part_key]
            fn(data, output_dir)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    print(f"\nRegeneration complete. Output in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantization Format Precision Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PYTHONPATH=. python examples/experiment_format_study.py
  PYTHONPATH=. python examples/experiment_format_study.py -o results/my_study --seed 1234
  PYTHONPATH=. python examples/experiment_format_study.py --skip-part-b --skip-part-c
        """,
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: results/ with timestamped subdir)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--calib-samples", type=int, default=256,
        help="Number of calibration samples (default: 256)",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=512,
        help="Number of evaluation samples (default: 512)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for calibration and evaluation (default: 16)",
    )
    parser.add_argument(
        "--skip-part-a", action="store_true",
        help="Skip Part A: Basic quantization",
    )
    parser.add_argument(
        "--skip-part-b", action="store_true",
        help="Skip Part B: Scaling comparison (FP32 vs PoT)",
    )
    parser.add_argument(
        "--skip-part-c", action="store_true",
        help="Skip Part C: Quantize-Quantize comparison",
    )
    parser.add_argument(
        "--skip-part-d", action="store_true",
        help="Skip Part D: Transform evaluation",
    )
    parser.add_argument(
        "--plot-from", default=None, metavar="RESULTS_JSON",
        help="Skip experiments; regenerate tables/figures from a saved results.json",
    )
    args = parser.parse_args()

    if args.plot_from:
        plot_from_results(args.plot_from, args.output_dir)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        skip_parts = {}
        if args.skip_part_a:
            skip_parts["A"] = True
        if args.skip_part_b:
            skip_parts["B"] = True
        if args.skip_part_c:
            skip_parts["C"] = True
        if args.skip_part_d:
            skip_parts["D"] = True

        results = run_format_study(
            build_model,
            lambda: make_calib_data(args.calib_samples, args.batch_size),
            lambda: make_eval_loader(args.eval_samples, args.batch_size),
            eval_fn,
            output_dir=args.output_dir,
            skip_parts=skip_parts or None,
        )
        print("Study complete. Results:", list(results.keys()))
