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

# Refactored imports — viz module (tables, figures, theme, save)
from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS, HIST_COLORS, FALLBACK_CYCLE
from src.viz.save import save_figure
from src.viz.figures import (
    qsnr_bar_chart,
    mse_box_plot,
    pot_delta_bar,
    histogram_overlay,
    transform_heatmap,
    transform_pie,
    transform_delta,
    error_vs_distribution,
    layer_type_qsnr,
    _compute_best_transform_per_layer,
    _get_acc_val,
)
from src.viz.tables import accuracy_table, format_comparison_table
from src.pipeline.runner import _extract_metric_per_layer


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


def build_conv_model() -> nn.Module:
    """Return a fresh instance of the Conv2d reference model for Part D.

    Default: ToyConvNet from ``examples/_model.py`` — contains Conv2d
    (channel_axis=1 in NCHW), BatchNorm2d, ReLU, and a Linear head.
    Accepts the same ``(B, 128)`` input as ToyMLP.

    Used in Part D to exercise the ``channel_axis=1`` code path in
    ``_make_smoothquant_transforms`` (Conv2d branches).  Replace with
    your own CNN architecture for custom experiments.
    """
    from examples._model import ToyConvNet
    return ToyConvNet()


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

    The weight compensation ``W * s`` is a ONE-TIME fusion performed during
    calibration (see ``_fuse_smoothquant_weights``).  After fusion, the weight
    is quantized with ``IdentityTransform`` — the scale is already baked in.

    - **input** role: ``SmoothQuantTransform`` — ``forward(x) = x / s``
    - **weight** role: ``IdentityTransform`` (scale already fused into W)
    - **output** role: ``IdentityTransform``

    Args:
        fmt_name: Format name string.
        granularity: ``GranularitySpec``.
        sq_transform: The per-layer ``SmoothQuantTransform`` (activation-side,
            carries the shared scale ``s``).
        weight_only: If True, only ``input`` + ``weight`` are quantized
            (used for per-channel formats like NF4, INT4-PC).

    Returns:
        ``OpQuantConfig`` with activation-side SmoothQuant on ``input``,
        ``IdentityTransform`` on weight and output.
    """
    fmt = FormatBase.from_str(fmt_name)
    no_tx = IdentityTransform()
    input_scheme = QuantScheme(
        format=fmt, granularity=granularity, transform=sq_transform,
    )
    weight_scheme = QuantScheme(
        format=fmt, granularity=granularity, transform=no_tx,
    )
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
    # NOTE: session.compare() uses its own _default_accuracy(logits, labels)
    # which computes accuracy via argmax -- this matches what the study's
    # eval_fn(model, DataLoader) does internally.  The study's eval_fn is
    # kept for the user-customization API but is NOT forwarded to
    # session.compare because their call signatures are different
    # (session.compare expects (logits, labels), not (model, DataLoader)).
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
    """Create per-layer SmoothQuantTransform dict from a single calibration pass.

    Runs one forward pass through the FP32 model to capture each layer's
    activation and weight, then creates a per-layer SmoothQuantTransform
    with correctly-shaped per-channel scales.

    This function is PURE — it does NOT mutate ``fp32_model``.  Weight fusion
    (``W = W * s``) must be performed separately via
    :func:`_fuse_smoothquant_weights`.

    Args:
        fp32_model: FP32 reference model (not mutated).
        calib_data: List of calibration batches (first batch used).

    Returns:
        Dict mapping layer name to ``SmoothQuantTransform`` (or
        ``IdentityTransform`` on failure).
    """
    if fp32_model is None:
        return {}

    activations: Dict[str, torch.Tensor] = {}
    weights: Dict[str, torch.Tensor] = {}
    channel_axes: Dict[str, int] = {}
    hooks = []

    def _hook(name):
        def fn(module, _input, _output):
            activations[name] = _input[0].detach()
            if hasattr(module, "weight") and module.weight is not None:
                weights[name] = module.weight.data.clone()
        return fn

    for name, module in fp32_model.named_modules():
        if isinstance(module, nn.Linear):
            channel_axes[name] = -1  # activation channel = last dim
            hooks.append(module.register_forward_hook(_hook(name)))
        elif isinstance(module, nn.Conv2d):
            channel_axes[name] = 1   # activation channel = dim 1 (NCHW)
            hooks.append(module.register_forward_hook(_hook(name)))

    with torch.no_grad():
        fp32_model.eval()
        fp32_model(calib_data[0])

    for h in hooks:
        h.remove()

    per_layer: Dict[str, TransformBase] = {}

    for name in activations:
        if name not in weights:
            continue
        try:
            act_axis = channel_axes.get(name, -1)
            sq_t = SmoothQuantTransform.from_calibration(
                X_act=activations[name], W=weights[name], alpha=0.5,
                act_channel_axis=act_axis,
            )
            per_layer[name] = sq_t
        except (ValueError, RuntimeError) as e:
            print(f"  Warning: SmoothQuant for {name}: {e}")
            per_layer[name] = IdentityTransform()

    return per_layer


def _fuse_smoothquant_weights(
    fp32_model: nn.Module,
    sq_transforms: Dict[str, TransformBase],
    *,
    layer_names: Optional[set] = None,
) -> nn.Module:
    """Return a deep copy of ``fp32_model`` with SmoothQuant weight fusion applied.

    For each layer in ``sq_transforms`` (filtered by ``layer_names`` if given),
    applies ``W = W * s`` — the one-time calibration-time weight compensation
    from SmoothQuant (Xiao et al. 2023, eq. 3).  The original ``fp32_model``
    is NOT mutated.

    Args:
        fp32_model: Reference FP32 model (not mutated).
        sq_transforms: Per-layer SmoothQuantTransform dict.
        layer_names: If given, only fuse weights for layers in this set.
                     ``None`` fuses all layers present in ``sq_transforms``.

    Returns:
        Deep copy of ``fp32_model`` with fused weights for the selected layers.
    """
    fused_model = copy.deepcopy(fp32_model)
    module_map = dict(fused_model.named_modules())

    for name, sq_t in sq_transforms.items():
        if layer_names is not None and name not in layer_names:
            continue
        if not isinstance(sq_t, SmoothQuantTransform):
            continue
        module = module_map.get(name)
        if module is None or not hasattr(module, "weight") or module.weight is None:
            continue
        W = module.weight.data
        # w_axis=1: PyTorch standard input-channel axis for both Linear
        # (out, in) and Conv2d (out, in, kH, kW).
        shape = [1] * W.ndim
        shape[1] = -1
        module.weight.data = W * sq_t.scale.view(*shape)

    return fused_model


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

    # Compute SmoothQuant transforms ONCE before the format loop using the
    # original fp32_model (pure, no mutation).  Build a single weight-fused
    # copy for all SmoothQuant variants — this avoids the compounding error
    # that would occur if _make_smoothquant_transforms were called once per
    # format and mutated the shared fp32_model each time.
    print("  Computing SmoothQuant scales...")
    sq_transforms = _make_smoothquant_transforms(fp32_model, calib_data)
    sq_fp32_model = _fuse_smoothquant_weights(fp32_model, sq_transforms)

    all_results = {}

    for fmt_name, fmt_str, gran, weight_only in fmt_configs:
        print(f"\n  == Transform study for {fmt_name} ==")
        builder = make_op_cfg_weight_only if weight_only else make_op_cfg

        # Build per-layer SmoothQuant configs for this format.
        # Activation transform (x / s) lives in the config; weight fusion
        # (W * s) was applied once to sq_fp32_model above.
        # Mathematical equivalence: (X/s) @ (W*s) = X@W
        sq_per_layer_cfg: Dict[str, OpQuantConfig] = {}
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

        # Phase 1 — single-transform variants.
        # None / Hadamard: use the original fp32_model (clean, unfused weights).
        variant_results: Dict[str, dict] = {}
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

        # SmoothQuant: use the weight-fused model (sq_fp32_model) so that
        # (X/s) @ (W*s) = X@W holds.  Using fp32_model here would break
        # the mathematical equivalence because weights would not carry s.
        print(f"    Running {fmt_name}-SmoothQuant...")
        variant_results["SmoothQuant"] = run_experiment(
            sq_per_layer_cfg, sq_fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )

        # Phase 2 — per-layer optimal.
        # Determine which transform wins per layer, then build a model that
        # has weights fused only for layers where SmoothQuant wins.  Layers
        # that prefer None/Hadamard keep their original (unfused) weights.
        variant_qsnr = {k: v["qsnr_per_layer"] for k, v in variant_results.items()}
        layer_best_tx = _compute_best_transform_per_layer(variant_qsnr)
        sq_winning = {n for n, tx_name in layer_best_tx.items() if tx_name == "SmoothQuant"}
        opt_fp32_model = _fuse_smoothquant_weights(
            fp32_model, sq_transforms, layer_names=sq_winning,
        )

        per_layer_optimal_cfg = _build_per_layer_optimal_cfg(
            variant_results, sq_transforms, fmt_str, gran, builder, weight_only,
        )
        print(f"    Running {fmt_name}-PerLayerOpt...")
        opt_result = run_experiment(
            per_layer_optimal_cfg, opt_fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        variant_results["PerLayerOpt"] = opt_result

        all_results[fmt_name] = variant_results

    return all_results


# ---------------------------------------------------------------------------
# Table Generation (6 tables)
#
# Tables 1-2: replaced by accuracy_table() from src.viz.tables
# Tables 3-6: keep inline until ported to tables.py
# ---------------------------------------------------------------------------

# TODO: Port generate_table_3 through generate_table_6 to src/viz/tables.py
#       once the viz module supports special-format tables.


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


def generate_table_4(part_d: dict, output_dir: str, *, suffix: str = "") -> str:
    """Table 4: Format x Transform accuracy matrix.

    Args:
        part_d: Results dict from :func:`run_part_d_transforms`.
        output_dir: Directory for CSV output.
        suffix: Appended to the CSV filename stem (e.g. ``"_conv"`` →
            ``table4_format_x_transform_conv.csv``).
    """
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
    fname = f"table4_format_x_transform{suffix}.csv"
    with open(f"{output_dir}/tables/{fname}", "w") as f:
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
# Main entry point
# ---------------------------------------------------------------------------


def run_format_study(
    build_model: Callable[[], nn.Module],
    make_calib_data: Callable[..., List[torch.Tensor]],
    make_eval_loader: Callable[..., DataLoader],
    eval_fn: Callable[[nn.Module, DataLoader], Dict[str, float]],
    *,
    build_conv_model: Optional[Callable[[], nn.Module]] = None,
    output_dir: Optional[str] = None,
    skip_parts: Optional[Dict[str, bool]] = None,
) -> Dict[str, dict]:
    """Main entry point: run all experiments and produce tables/figures.

    Executes Parts A/B/C/D of the study, saving results to a timestamped
    directory under ``output_dir`` (default: ``results/``).

    When ``build_conv_model`` is provided (default: :func:`build_conv_model`
    returning :class:`ToyConvNet`), Part D is also run on the Conv2d model
    and stored as ``part_d_conv``.  This validates the ``channel_axis=1``
    code path in SmoothQuant that is not exercised by the MLP-only default.

    Args:
        build_model: Callable that returns a fresh FP32 model.
        make_calib_data: Callable for calibration data.
        make_eval_loader: Callable for evaluation DataLoader.
        eval_fn: Evaluation function (model, dataloader) -> metric dict.
        build_conv_model: Callable that returns a Conv2d model for Part D
            Conv validation.  ``None`` skips the Conv2d Part D run.
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
        print(accuracy_table(
            all_results["part_a"],
            title="Table 1: 8-bit Format Comparison",
            output_dir=output_dir, filename="table1_8bit.csv",
        ))
    else:
        print("\n### PART A: SKIPPED ###")

    if not skip_parts.get("B"):
        print("\n" + "=" * 60)
        print("PART B: 4-bit Format Comparison")
        print("=" * 60)
        all_results["part_b"] = run_part_b_4bit(
            fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        print(accuracy_table(
            all_results["part_b"],
            title="Table 2: 4-bit Format Comparison",
            output_dir=output_dir, filename="table2_4bit.csv",
        ))
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
        print("PART D: Transform Study (MLP)")
        print("=" * 60)
        all_results["part_d"] = run_part_d_transforms(
            fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        print(generate_table_4(all_results["part_d"], output_dir))
        print(generate_table_5(all_results["part_d"], output_dir))

        # Part D — Conv2d variant: validates channel_axis=1 code path in
        # SmoothQuant (Conv2d NCHW layout, different from Linear's last-dim).
        # Uses the same calib_data / eval_loader since ToyConvNet accepts
        # (B, 128) input identical to ToyMLP.
        if build_conv_model is not None and not skip_parts.get("D_conv"):
            print("\n" + "=" * 60)
            print("PART D (Conv): Transform Study on Conv2d Model")
            print("=" * 60)
            conv_model = build_conv_model()
            conv_model.eval()
            all_results["part_d_conv"] = run_part_d_transforms(
                conv_model, calib_data, eval_loader, eval_fn=eval_fn,
            )
            print(generate_table_4(all_results["part_d_conv"],
                                   output_dir, suffix="_conv"))
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
        (lambda data, od: qsnr_bar_chart(data, title="Fig 1: Per-Layer QSNR — 8-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_a", "fig1_qsnr_8bit"),
        (lambda data, od: qsnr_bar_chart(data, title="Fig 2: Per-Layer QSNR — 4-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_b", "fig2_qsnr_4bit"),
        (lambda data, od: mse_box_plot(data, title="Fig 3: Per-Layer MSE Distribution — 8-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_a", "fig3_mse_8bit"),
        (lambda data, od: mse_box_plot(data, title="Fig 4: Per-Layer MSE Distribution — 4-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_b", "fig4_mse_4bit"),
        (lambda data, od: pot_delta_bar(data, output_dir=od), "part_c", "fig5_pot_delta"),
        (lambda data, od: histogram_overlay(data, output_dir=od), None, "fig6_histogram"),
        (lambda data, od: transform_heatmap(data, colors=FORMAT_COLORS, output_dir=od), "part_d", "fig7_transform_heatmap"),
        (lambda data, od: transform_pie(data, colors=TRANSFORM_COLORS, output_dir=od), "part_d", "fig8_transform_pie"),
        (lambda data, od: transform_delta(data, colors=TRANSFORM_COLORS, output_dir=od), "part_d", "fig9_transform_delta"),
        (lambda data, od: error_vs_distribution(data, output_dir=od), None, "fig10_error_vs_dist"),
        (lambda data, od: layer_type_qsnr(data, output_dir=od), None, "fig11_layer_type_qsnr"),
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
        print(accuracy_table(
            all_results["part_a"],
            title="Table 1: 8-bit Format Comparison",
            output_dir=output_dir, filename="table1_8bit.csv",
        ))
    else:
        skipped_parts.add("a")
    if "part_b" in all_results:
        print(accuracy_table(
            all_results["part_b"],
            title="Table 2: 4-bit Format Comparison",
            output_dir=output_dir, filename="table2_4bit.csv",
        ))
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
        (lambda data, od: qsnr_bar_chart(data, title="Fig 1: Per-Layer QSNR — 8-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_a", "fig1_qsnr_8bit"),
        (lambda data, od: qsnr_bar_chart(data, title="Fig 2: Per-Layer QSNR — 4-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_b", "fig2_qsnr_4bit"),
        (lambda data, od: mse_box_plot(data, title="Fig 3: Per-Layer MSE Distribution — 8-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_a", "fig3_mse_8bit"),
        (lambda data, od: mse_box_plot(data, title="Fig 4: Per-Layer MSE Distribution — 4-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_b", "fig4_mse_4bit"),
        (lambda data, od: pot_delta_bar(data, output_dir=od), "part_c", "fig5_pot_delta"),
        (lambda data, od: histogram_overlay(data, output_dir=od), None, "fig6_histogram"),
        (lambda data, od: transform_heatmap(data, colors=FORMAT_COLORS, output_dir=od), "part_d", "fig7_transform_heatmap"),
        (lambda data, od: transform_pie(data, colors=TRANSFORM_COLORS, output_dir=od), "part_d", "fig8_transform_pie"),
        (lambda data, od: transform_delta(data, colors=TRANSFORM_COLORS, output_dir=od), "part_d", "fig9_transform_delta"),
        (lambda data, od: error_vs_distribution(data, output_dir=od), None, "fig10_error_vs_dist"),
        (lambda data, od: layer_type_qsnr(data, output_dir=od), None, "fig11_layer_type_qsnr"),
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
        help="Skip Part A: 8-bit Format Comparison",
    )
    parser.add_argument(
        "--skip-part-b", action="store_true",
        help="Skip Part B: 4-bit Format Comparison",
    )
    parser.add_argument(
        "--skip-part-c", action="store_true",
        help="Skip Part C: FP32 vs PoT Scaling",
    )
    parser.add_argument(
        "--skip-part-d", action="store_true",
        help="Skip Part D: Transform evaluation (MLP + Conv2d)",
    )
    parser.add_argument(
        "--skip-part-d-conv", action="store_true",
        help="Skip Part D Conv2d variant (channel_axis=1 validation)",
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
        if args.skip_part_d_conv:
            skip_parts["D_conv"] = True

        results = run_format_study(
            build_model,
            lambda: make_calib_data(args.calib_samples, args.batch_size),
            lambda: make_eval_loader(args.eval_samples, args.batch_size),
            eval_fn,
            build_conv_model=build_conv_model,
            output_dir=args.output_dir,
            skip_parts=skip_parts or None,
        )
        print("Study complete. Results:", list(results.keys()))
