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


def _make_smoothquant_transform(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
) -> TransformBase:
    """Create SmoothQuantTransform from calibration activation + first Linear weight.

    If no Linear module is found in ``fp32_model``, returns ``IdentityTransform()``
    as a safe fallback.

    Args:
        fp32_model: FP32 reference model (used to extract first Linear weight).
        calib_data: List of calibration batches (first batch used as activation).

    Returns:
        ``SmoothQuantTransform`` if a Linear module is found, else
        ``IdentityTransform()``.
    """
    if fp32_model is None:
        return IdentityTransform()
    x_sample = calib_data[0]
    with torch.no_grad():
        fp32_model.eval()
        target_weight = None
        for module in fp32_model.modules():
            if isinstance(module, nn.Linear):
                target_weight = module.weight.data
                break
        if target_weight is None:
            return IdentityTransform()
    return SmoothQuantTransform.from_calibration(
        X_act=x_sample, W=target_weight, alpha=0.5,
    )


def _build_per_layer_optimal_cfg(
    variant_results: dict,
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    fmt_str: str,
    gran: GranularitySpec,
    cfg_builder: Callable,
) -> dict:
    """Build per-layer OpQuantConfig dict choosing best transform per layer by QSNR.

    For each layer in the model, selects the transform variant (None, SmoothQuant,
    or Hadamard) that achieves the highest QSNR score from the variant experiments.

    Args:
        variant_results: Dict mapping ``"None"``, ``"SmoothQuant"``, ``"Hadamard"``
            to their experiment result dicts (which contain ``qsnr_per_layer``).
        fp32_model: FP32 reference model (for SmoothQuant calibration data).
        calib_data: List of calibration batches (for SmoothQuant).
        fmt_str: Format name string for the config builder.
        gran: ``GranularitySpec`` for the config builder.
        cfg_builder: ``make_op_cfg`` or ``make_op_cfg_weight_only``.

    Returns:
        Dict mapping layer name to ``OpQuantConfig``.
    """
    # Determine best transform per layer by QSNR
    layer_best_tx: dict = {}
    for tx_name in ["None", "SmoothQuant", "Hadamard"]:
        for layer, qsnr in variant_results[tx_name]["qsnr_per_layer"].items():
            if qsnr > layer_best_tx.get(layer, ("", -float("inf")))[1]:
                layer_best_tx[layer] = (tx_name, qsnr)

    # Create a single SmoothQuantTransform for all SQ-chosen layers
    sq_transform = _make_smoothquant_transform(fp32_model, calib_data)
    tx_map = {
        "None": None,
        "SmoothQuant": sq_transform,
        "Hadamard": HadamardTransform(),
    }

    per_layer_cfg = {}
    for layer, (tx_name, _) in layer_best_tx.items():
        per_layer_cfg[layer] = cfg_builder(fmt_str, gran, transform=tx_map[tx_name])

    return per_layer_cfg


def run_part_d_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
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
        transforms_to_try = {
            "None": None,
            "SmoothQuant": _make_smoothquant_transform(fp32_model, calib_data),
            "Hadamard": HadamardTransform(),
        }

        for tx_name, tx in transforms_to_try.items():
            label = f"{fmt_name}-{tx_name}"
            print(f"    Running {label}...")
            cfg = builder(fmt_str, gran, transform=tx)
            variant_results[tx_name] = run_experiment(
                cfg, fp32_model, calib_data, eval_loader,
            )

        # Phase 2: Per-layer optimal
        per_layer_optimal_cfg = _build_per_layer_optimal_cfg(
            variant_results, fp32_model, calib_data, fmt_str, gran, builder,
        )
        print(f"    Running {fmt_name}-PerLayerOpt...")
        opt_result = run_experiment(
            per_layer_optimal_cfg, fp32_model, calib_data, eval_loader,
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
        if isinstance(acc, dict):
            acc_str = ", ".join(f"{k}: {v:.4f}" for k, v in acc.items())
        else:
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
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

        all_layers: set = set()
        for qsnr_dict in variant_qsnr.values():
            all_layers.update(qsnr_dict.keys())

        tx_counts: Dict[str, int] = defaultdict(int)
        for layer in all_layers:
            best_tx = max(
                variant_qsnr.keys(),
                key=lambda tx, l=layer: variant_qsnr[tx].get(l, -float("inf")),
            )
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


def _get_acc_val(data) -> float:
    """Extract scalar accuracy value from a result dict entry."""
    acc = data.get("accuracy", {}) if isinstance(data, dict) else {}
    if isinstance(acc, dict):
        return float(acc.get("accuracy", 0.0))
    if isinstance(acc, (int, float)):
        return float(acc)
    return 0.0


def plot_fig1_qsnr_8bit(part_a: dict, output_dir: str):
    """Fig 1: 8-bit per-layer QSNR line chart (3 lines)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in part_a.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        layers = sorted(data["qsnr_per_layer"].keys())
        values = [data["qsnr_per_layer"][l] for l in layers]
        ax.plot(range(len(layers)), values, marker="o", label=name, linewidth=2)
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
        ax.plot(range(len(layers)), values, marker="o", label=name, linewidth=2)
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
    palette = ["#3498db", "#e74c3c", "#2ecc71"]
    for i, (name, data) in enumerate(part_a.items()):
        if "baseline" in name.lower() or "mse_per_layer" not in data:
            continue
        mse_vals = list(data["mse_per_layer"].values())
        if mse_vals:
            data_to_plot.append(mse_vals)
            labels.append(name)
            colors.append(palette[i % len(palette)])
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
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
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, (name, data) in enumerate(part_b.items()):
        if "baseline" in name.lower() or "mse_per_layer" not in data:
            continue
        mse_vals = list(data["mse_per_layer"].values())
        if mse_vals:
            data_to_plot.append(mse_vals)
            labels.append(name)
            colors.append(palette[i % len(palette)])
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
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
    """Fig 6: Histogram overlay for key layers.

    Extracts histograms from analysis reports and displays them for the
    most sensitive layers (highest MSE).
    """
    layer_histograms: Dict[str, dict] = {}
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
                if layer in layer_histograms:
                    continue
                for role, stages in roles.items():
                    for stage, slices in stages.items():
                        for metrics in slices.values():
                            if "hist_bins" in metrics and "hist_counts" in metrics:
                                layer_histograms[layer] = metrics
                                break
                    if layer in layer_histograms:
                        break

    if not layer_histograms:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Histogram data not available\n"
                "(No HistogramObserver in reports)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Fig 6: Activation Histograms (No Data)")
        _save_figure(fig, output_dir, "fig6_histogram")
        return

    # Pick top 5 layers by MSE
    top_layers = sorted(layer_histograms.items(),
                        key=lambda x: x[1].get("mse", 0), reverse=True)[:5]

    n = len(top_layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (layer, hist_data) in zip(axes[0], top_layers):
        bins = hist_data.get("hist_bins", [])
        counts = hist_data.get("hist_counts", [])
        if len(bins) > 1 and len(counts) > 0:
            width = max(bins[1] - bins[0], 1e-6)
            ax.bar(bins[:-1], counts, width=width, alpha=0.7, color="#3498db",
                   edgecolor="white", linewidth=0.5)
        ax.set_title(layer, fontsize=9)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 6: Activation Histograms — Most Sensitive Layers",
                 fontsize=13)
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
    vmin, vmax = arr[~np.isnan(arr)].min(), arr[~np.isnan(arr)].max()
    im = ax.imshow(arr, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

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

    pie_colors = {"None": "#3498db", "SmoothQuant": "#2ecc71",
                  "Hadamard": "#e74c3c"}

    for ax, (fmt_name, fmt_data) in zip(axes, sorted(part_d.items())):
        if "PerLayerOpt" not in fmt_data:
            ax.text(0.5, 0.5, "No PerLayerOpt data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        variant_qsnr: Dict[str, Dict[str, float]] = {}
        for tx_name in ("None", "SmoothQuant", "Hadamard"):
            if tx_name in fmt_data and "qsnr_per_layer" in fmt_data[tx_name]:
                variant_qsnr[tx_name] = fmt_data[tx_name]["qsnr_per_layer"]

        all_layers: set = set()
        for qsnr_dict in variant_qsnr.values():
            all_layers.update(qsnr_dict.keys())

        tx_counts: Dict[str, int] = defaultdict(int)
        for layer in all_layers:
            best_tx = max(
                variant_qsnr.keys(),
                key=lambda tx, l=layer: variant_qsnr[tx].get(l, -float("inf")),
            )
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
    """Fig 9: Transform delta QSNR vs baseline bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = 0
    tick_positions, tick_labels = [], []
    colors_tx = {"SmoothQuant": "#2ecc71", "Hadamard": "#e74c3c"}

    for fmt_name in sorted(part_d.keys()):
        fmt_data = part_d[fmt_name]
        if "None" not in fmt_data:
            continue
        baseline_qsnr = fmt_data["None"].get("qsnr_per_layer", {})

        for tx_name in ("SmoothQuant", "Hadamard"):
            if tx_name not in fmt_data or "qsnr_per_layer" not in fmt_data[tx_name]:
                continue
            tx_qsnr = fmt_data[tx_name]["qsnr_per_layer"]
            all_layers = sorted(set(list(baseline_qsnr.keys()) + list(tx_qsnr.keys())))
            deltas = [tx_qsnr.get(l, 0) - baseline_qsnr.get(l, 0) for l in all_layers]

            bar_positions = list(range(x_pos, x_pos + len(all_layers)))
            color = colors_tx.get(tx_name, "#95a5a6")
            ax.bar(bar_positions, deltas, color=color, alpha=0.6,
                   label=f"{fmt_name}-{tx_name}")
            x_pos += len(all_layers) + 2
            tick_positions.append((bar_positions[0] + bar_positions[-1]) / 2)
            tick_labels.append(f"{fmt_name}\n{tx_name}")

    ax.axhline(y=0, color="black", linewidth=0.5)
    if tick_positions:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("QSNR Delta (dB) vs No Transform")
    ax.set_title("Fig 9: Transform Impact on Per-Layer QSNR")
    ax.grid(True, alpha=0.3)
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
    ax.hist(sparse_vals, bins=20, alpha=0.7, color="#3498db", edgecolor="white")
    ax.set_xlabel("Sparse Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Sparsity Across Layers")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 10: Quantization Error vs Distribution Features", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig10_error_vs_dist")


def plot_fig11_layer_type_qsnr(all_results: dict, output_dir: str):
    """Fig 11: Layer-type grouped QSNR comparison using LayerSensitivity."""
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
    colors_cycle = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    labels = list(ltype_qsnr.keys())

    # QSNR boxplot
    ax = axes[0]
    qsnr_data = [ltype_qsnr[lt] for lt in labels]
    bp = ax.boxplot(qsnr_data, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors_cycle[i % len(colors_cycle)])
        patch.set_alpha(0.6)
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("Avg QSNR by Layer Type")
    ax.grid(True, alpha=0.3)

    # MSE boxplot (log scale)
    ax = axes[1]
    mse_data = [ltype_mse[lt] for lt in labels]
    bp2 = ax.boxplot(mse_data, labels=labels, patch_artist=True)
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
