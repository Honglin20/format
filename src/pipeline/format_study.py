"""
Format Study experiment runner.

Programmatic entry point::

    from src.pipeline.format_study import run_format_study

    results = run_format_study(
        build_model=my_build_fn,
        make_calib_data=my_calib_fn,
        make_eval_loader=my_loader_fn,
        eval_fn=my_eval_fn,
    )

To customise the search space, edit ``src/pipeline/studies/format_study.py``.
"""
from __future__ import annotations

import copy
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform, TransformBase
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform
from src.analysis.observers import (
    QSNRObserver, MSEObserver, HistogramObserver, DistributionObserver,
)
from src.session import QuantSession
from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
from src.calibration.strategies import MSEScaleStrategy
from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS
from src.viz.figures import (
    qsnr_line_chart,
    mse_box_plot,
    pot_delta_bar,
    histogram_overlay,
    transform_heatmap,
    transform_pie,
    transform_delta,
    error_vs_distribution,
    layer_type_qsnr,
    _compute_best_transform_per_layer,
)
from src.viz.tables import accuracy_table
from src.pipeline.runner import extract_metric_per_layer


_PER_B32 = GranularitySpec.per_block(size=32, axis=-1)
_PER_C0 = GranularitySpec.per_channel(axis=0)


# ---------------------------------------------------------------------------
# Config builder helpers
# ---------------------------------------------------------------------------

def make_op_cfg(
    fmt_name: str,
    granularity: GranularitySpec,
    *,
    transform: Optional[TransformBase] = None,
) -> OpQuantConfig:
    """Inference-only config: input / weight / output all share the same scheme."""
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
    """Weight-only config (input / output not quantized). Used for NF4."""
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
    fmt = FormatBase.from_str(fmt_name)
    no_tx = IdentityTransform()
    input_scheme = QuantScheme(format=fmt, granularity=granularity, transform=sq_transform)
    weight_scheme = QuantScheme(format=fmt, granularity=granularity, transform=no_tx)
    if weight_only:
        return OpQuantConfig(input=input_scheme, weight=weight_scheme)
    output_scheme = QuantScheme(format=fmt, granularity=granularity, transform=no_tx)
    return OpQuantConfig(input=input_scheme, weight=weight_scheme, output=output_scheme)


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg,
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
    """Run one quantization experiment and return results.

    Args:
        cfg: OpQuantConfig or dict[layer_name -> OpQuantConfig].
        fp32_model: Reference FP32 model (deep-copied; not mutated).
        calib_data: List of calibration batch tensors.
        eval_loader: Evaluation DataLoader yielding (input, label).
        observers: Override observer list. Default: QSNR + MSE + Histogram + Distribution.
        lsq_steps: If > 0, run LSQ pre-scale optimisation for this many steps.
        lsq_pot: Constrain LSQ scales to power-of-two.
        lsq_lr: Learning rate for LSQ optimiser.
        eval_fn: Accepted for API compatibility; session.compare() handles evaluation.

    Returns:
        Dict with keys: accuracy, fp32_accuracy, delta, report, session,
        qsnr_per_layer, mse_per_layer.
    """
    if observers is None:
        observers = [QSNRObserver(), MSEObserver(), HistogramObserver(), DistributionObserver()]
    if not calib_data:
        raise ValueError("calib_data must contain at least one batch")

    session = QuantSession(
        copy.deepcopy(fp32_model), cfg,
        calibrator=MSEScaleStrategy(),
        keep_fp32=True,
    )

    with session.calibrate():
        for batch in calib_data:
            session(batch)

    if lsq_steps > 0:
        session.initialize_pre_scales(calib_data, init="ones", pot=lsq_pot)
        opt = LayerwiseScaleOptimizer(
            num_steps=lsq_steps, num_batches=len(calib_data),
            optimizer="adam", lr=lsq_lr, pot=lsq_pot,
        )
        session.optimize_scales(opt, calib_data)

    with session.analyze(observers=observers) as ctx:
        for batch in calib_data:
            session(batch)
    report = ctx.report()

    result = session.compare(eval_loader)

    # Cost estimation (P6)
    cost = session.estimate_cost()
    cost_fp32 = session.estimate_cost(fp32=True)

    return {
        "accuracy": result["quant"],
        "fp32_accuracy": result["fp32"],
        "delta": result["delta"],
        "report": report,
        "session": session,
        "qsnr_per_layer": extract_metric_per_layer(report, "qsnr_db"),
        "mse_per_layer": extract_metric_per_layer(report, "mse"),
        "cost": cost,
        "cost_fp32": cost_fp32,
    }


# ---------------------------------------------------------------------------
# Part runners
# ---------------------------------------------------------------------------

def run_part_a_8bit(fp32_model, calib_data, eval_loader, *, eval_fn=None) -> dict:
    """Part A: MXINT-8 / MXFP-8 / INT8-PC comparison."""
    print("\n### Part A: 8-bit Format Comparison ###")
    configs = {
        "MXINT-8": make_op_cfg("int8",      _PER_B32),
        "MXFP-8":  make_op_cfg("fp8_e4m3",  _PER_B32),
        "INT8-PC": make_op_cfg("int8",       _PER_C0),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"  Running {name}...")
        results[name] = run_experiment(cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn)
    results["FP32 (baseline)"] = {"accuracy": results[list(configs)[0]]["fp32_accuracy"]}
    return results


def run_part_b_4bit(fp32_model, calib_data, eval_loader, *, eval_fn=None) -> dict:
    """Part B: MXINT-4 / MXFP-4 / INT4-PC / NF4-PC comparison."""
    print("\n### Part B: 4-bit Format Comparison ###")
    configs = {
        "MXINT-4": make_op_cfg("int4",          _PER_B32),
        "MXFP-4":  make_op_cfg("fp4_e2m1",      _PER_B32),
        "INT4-PC": make_op_cfg("int4",           _PER_C0),
        "NF4-PC":  make_op_cfg_weight_only("nf4", _PER_C0),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"  Running {name}...")
        results[name] = run_experiment(cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn)
    results["FP32 (baseline)"] = {"accuracy": results[list(configs)[0]]["fp32_accuracy"]}
    return results


def run_part_c_pot_scaling(fp32_model, calib_data, eval_loader, *, eval_fn=None) -> dict:
    """Part C: INT per-channel FP32 vs PoT scaling, LSQ-optimised."""
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
        results[name] = run_experiment(
            make_op_cfg(fmt, _PER_C0), fp32_model, calib_data, eval_loader,
            lsq_steps=100, lsq_pot=pot, eval_fn=eval_fn,
        )
    results["FP32 (baseline)"] = {"accuracy": results[list(configs)[0]]["fp32_accuracy"]}
    return results


def _make_smoothquant_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
) -> Dict[str, TransformBase]:
    """One calibration pass → per-layer SmoothQuantTransform. Pure; does not mutate model."""
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
            channel_axes[name] = -1
            hooks.append(module.register_forward_hook(_hook(name)))
        elif isinstance(module, nn.Conv2d):
            channel_axes[name] = 1
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
            per_layer[name] = SmoothQuantTransform.from_calibration(
                X_act=activations[name], W=weights[name], alpha=0.5,
                act_channel_axis=channel_axes.get(name, -1),
            )
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
    """Return a deep copy of fp32_model with W = W * s applied to selected layers."""
    fused = copy.deepcopy(fp32_model)
    module_map = dict(fused.named_modules())
    for name, sq_t in sq_transforms.items():
        if layer_names is not None and name not in layer_names:
            continue
        if not isinstance(sq_t, SmoothQuantTransform):
            continue
        m = module_map.get(name)
        if m is None or not hasattr(m, "weight") or m.weight is None:
            continue
        W = m.weight.data
        shape = [1] * W.ndim
        shape[1] = -1
        m.weight.data = W * sq_t.scale.view(*shape)
    return fused


def _build_per_layer_optimal_cfg(
    variant_results: dict,
    sq_transforms: dict,
    fmt_str: str,
    gran: GranularitySpec,
    cfg_builder: Callable,
    weight_only: bool = False,
) -> dict:
    """Per-layer OpQuantConfig: pick the transform with highest QSNR per layer."""
    layer_best_tx = _compute_best_transform_per_layer(
        {k: v["qsnr_per_layer"] for k, v in variant_results.items()}
    )
    tx_map = {"None": None, "Hadamard": HadamardTransform()}
    per_layer_cfg = {}
    for layer, tx_name in layer_best_tx.items():
        if tx_name == "SmoothQuant":
            sq_tx = sq_transforms.get(layer, IdentityTransform())
            per_layer_cfg[layer] = _make_sq_op_cfg(fmt_str, gran, sq_tx, weight_only)
        else:
            per_layer_cfg[layer] = cfg_builder(fmt_str, gran, transform=tx_map[tx_name])
    return per_layer_cfg


def run_part_d_transforms(fp32_model, calib_data, eval_loader, *, eval_fn=None) -> dict:
    """Part D: None / SmoothQuant / Hadamard at 4-bit + per-layer optimal selection."""
    print("\n### Part D: Transform Study at 4-bit ###")
    fmt_configs = [
        ("MXINT-4", "int4",      _PER_B32, False),
        ("MXFP-4",  "fp4_e2m1",  _PER_B32, False),
        ("INT4-PC", "int4",      _PER_C0,  False),
        ("NF4-PC",  "nf4",       _PER_C0,  True),
    ]

    print("  Computing SmoothQuant scales...")
    sq_transforms = _make_smoothquant_transforms(fp32_model, calib_data)
    sq_fp32_model = _fuse_smoothquant_weights(fp32_model, sq_transforms)

    all_results: Dict[str, dict] = {}
    for fmt_name, fmt_str, gran, weight_only in fmt_configs:
        print(f"\n  == Transform study for {fmt_name} ==")
        builder = make_op_cfg_weight_only if weight_only else make_op_cfg

        sq_per_layer_cfg: Dict[str, OpQuantConfig] = {
            lname: _make_sq_op_cfg(fmt_str, gran, sq_tx, weight_only)
            for lname, sq_tx in sq_transforms.items()
        }
        fallback_cfg = builder(fmt_str, gran, transform=None)
        for mname, _ in fp32_model.named_modules():
            if mname and mname not in sq_per_layer_cfg:
                sq_per_layer_cfg[mname] = fallback_cfg

        variant_results: Dict[str, dict] = {}
        for tx_name, tx in {"None": None, "Hadamard": HadamardTransform()}.items():
            print(f"    Running {fmt_name}-{tx_name}...")
            variant_results[tx_name] = run_experiment(
                builder(fmt_str, gran, transform=tx),
                fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
            )

        print(f"    Running {fmt_name}-SmoothQuant...")
        variant_results["SmoothQuant"] = run_experiment(
            sq_per_layer_cfg, sq_fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
        )

        layer_best_tx = _compute_best_transform_per_layer(
            {k: v["qsnr_per_layer"] for k, v in variant_results.items()}
        )
        sq_winning = {n for n, tx in layer_best_tx.items() if tx == "SmoothQuant"}
        opt_model = _fuse_smoothquant_weights(fp32_model, sq_transforms, layer_names=sq_winning)

        print(f"    Running {fmt_name}-PerLayerOpt...")
        variant_results["PerLayerOpt"] = run_experiment(
            _build_per_layer_optimal_cfg(variant_results, sq_transforms, fmt_str, gran, builder, weight_only),
            opt_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        all_results[fmt_name] = variant_results

    return all_results


def run_block_size_sweep(
    fp32_model,
    calib_data,
    eval_loader,
    fmt_name: str = "int8",
    *,
    eval_fn=None,
) -> dict:
    """Sweep block sizes [16, 32, 64, 128] for sensitivity analysis."""
    results = {}
    for bs in (16, 32, 64, 128):
        gran = GranularitySpec.per_block(size=bs, axis=-1)
        print(f"  Block size {bs}...")
        try:
            results[f"{fmt_name}-blk{bs}"] = run_experiment(
                make_op_cfg(fmt_name, gran), fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
            )
        except Exception as e:
            print(f"    FAILED: {e}")
    return results


# ---------------------------------------------------------------------------
# Table generators (Tables 3–6; Tables 1–2 delegate to src.viz.tables)
# ---------------------------------------------------------------------------

def generate_table_3(part_c: dict, output_dir: str) -> str:
    """Table 3: FP32 vs PoT accuracy delta."""
    baseline_acc = 0.0
    for name, data in part_c.items():
        if "baseline" in name.lower():
            acc = data.get("accuracy", {})
            baseline_acc = float(acc.get("accuracy", 0.0)) if isinstance(acc, dict) else float(acc or 0.0)
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
        qsnr_d = data.get("qsnr_per_layer", {})
        mse_d = data.get("mse_per_layer", {})
        rows.append((
            name, acc_str, acc_val - baseline_acc,
            sum(qsnr_d.values()) / max(len(qsnr_d), 1),
            sum(mse_d.values()) / max(len(mse_d), 1),
        ))

    lines = [f"\n{'='*85}", "Table 3: FP32 vs PoT Scaling", "=" * 85,
             f"{'Config':<20} {'Accuracy':<20} {'Delta':<12} {'Avg QSNR (dB)':<15} {'Avg MSE':<15}",
             "-" * 85]
    for r in rows:
        lines.append(f"{r[0]:<20} {r[1]:<20} {r[2]:<+12.4f} {r[3]:<15.2f} {r[4]:<15.6f}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table3_pot.csv", "w") as f:
        f.write("Config,Accuracy,Delta,Avg_QSNR_dB,Avg_MSE\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.4f},{r[4]:.6f}\n")
    return result


def generate_table_4(part_d: dict, output_dir: str, *, suffix: str = "") -> str:
    """Table 4: Format × Transform accuracy matrix."""
    fmt_names = sorted(part_d.keys())
    tx_variants = sorted({tx for fmt_data in part_d.values() for tx in fmt_data})

    def _acc(fmt_data, tx):
        if tx not in fmt_data:
            return float("nan")
        acc = fmt_data[tx].get("accuracy", {})
        return float(acc.get("accuracy", 0.0)) if isinstance(acc, dict) else (
            float(acc) if isinstance(acc, (int, float)) else float("nan")
        )

    lines = [f"\n{'='*80}", "Table 4: Format x Transform Accuracy Matrix", "=" * 80,
             f"{'Format':<16}" + "".join(f" {tx:<20}" for tx in tx_variants),
             "-" * (16 + 21 * len(tx_variants))]
    for fmt_name in fmt_names:
        row = f"{fmt_name:<16}"
        for tx in tx_variants:
            v = _acc(part_d[fmt_name], tx)
            row += f" {'N/A':<20}" if math.isnan(v) else f" {v:<20.4f}"
        lines.append(row)
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table4_format_x_transform{suffix}.csv", "w") as f:
        f.write("Format," + ",".join(tx_variants) + "\n")
        for fmt_name in fmt_names:
            vals = [
                f"{_acc(part_d[fmt_name], tx):.6f}" if not math.isnan(_acc(part_d[fmt_name], tx)) else "N/A"
                for tx in tx_variants
            ]
            f.write(f"{fmt_name}," + ",".join(vals) + "\n")
    return result


def generate_table_5(part_d: dict, output_dir: str) -> str:
    """Table 5: Per-layer optimal transform distribution."""
    distribution: Dict[str, Dict[str, int]] = {}
    all_tx_set: set = set()

    for fmt_name, fmt_data in part_d.items():
        variant_qsnr = {
            tx: fmt_data[tx]["qsnr_per_layer"]
            for tx in ("None", "SmoothQuant", "Hadamard")
            if tx in fmt_data and "qsnr_per_layer" in fmt_data[tx]
        }
        tx_counts: Dict[str, int] = defaultdict(int)
        for best_tx in _compute_best_transform_per_layer(variant_qsnr).values():
            tx_counts[best_tx] += 1
        distribution[fmt_name] = dict(tx_counts)
        all_tx_set.update(tx_counts.keys())

    all_tx = sorted(all_tx_set)
    hdr = f"{'Format':<16}" + "".join(f" {tx:<18}" for tx in all_tx) + " Total"
    lines = [f"\n{'='*80}", "Table 5: Per-Layer Optimal Transform Distribution", "=" * 80, hdr, "-" * len(hdr)]
    for fmt_name in sorted(distribution.keys()):
        r = f"{fmt_name:<16}"
        total = 0
        for tx in all_tx:
            cnt = distribution[fmt_name].get(tx, 0)
            r += f" {cnt:<18}"
            total += cnt
        lines.append(r + f" {total}")
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
    """Table 6: Top-10 most sensitive layers across all experiments."""
    layer_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: {"mse": [], "qsnr": []})
    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_data in part_data.values():
            if not isinstance(config_data, dict):
                continue
            for key in ("qsnr_per_layer", "mse_per_layer"):
                if key not in config_data:
                    continue
                metric = "qsnr" if "qsnr" in key else "mse"
                for layer, val in config_data[key].items():
                    layer_metrics[layer][metric].append(val)

    ranking = sorted(
        (
            (layer,
             sum(m["mse"]) / max(len(m["mse"]), 1) if m["mse"] else 0.0,
             sum(m["qsnr"]) / max(len(m["qsnr"]), 1) if m["qsnr"] else 0.0)
            for layer, m in layer_metrics.items()
        ),
        key=lambda x: x[1], reverse=True,
    )[:10]

    lines = [f"\n{'='*80}", "Table 6: Top-10 Most Sensitive Layers", "=" * 80,
             f"{'#':<4} {'Layer':<28} {'Avg MSE':<18} {'Avg QSNR (dB)':<15}", "-" * 80]
    for i, (layer, mse, qsnr) in enumerate(ranking, 1):
        lines.append(f"{i:<4} {layer:<28} {mse:<18.6e} {qsnr:<15.2f}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table6_sensitivity.csv", "w") as f:
        f.write("Rank,Layer,Avg_MSE,Avg_QSNR_dB\n")
        for i, (layer, mse, qsnr) in enumerate(ranking, 1):
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
    """Run all format study experiments and produce tables and figures.

    Args:
        build_model: Returns a fresh FP32 model instance.
        make_calib_data: Returns calibration data as a list of tensors.
        make_eval_loader: Returns evaluation DataLoader yielding (input, label).
        eval_fn: ``(model, dataloader) -> dict[str, float]``.
        build_conv_model: Optional Conv2d model for Part D Conv2d validation.
        output_dir: Output directory. Default: ``results/<timestamp>/``.
        skip_parts: Dict mapping ``"A"``, ``"B"``, ``"C"``, ``"D"`` to True to skip.

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

    for part, label, runner, tables in [
        ("A", "8-bit Format Comparison",  run_part_a_8bit,     ["table1"]),
        ("B", "4-bit Format Comparison",  run_part_b_4bit,     ["table2"]),
        ("C", "FP32 vs PoT Scaling",      run_part_c_pot_scaling, ["table3"]),
        ("D", "Transform Study (MLP)",    run_part_d_transforms,  ["table4", "table5"]),
    ]:
        if skip_parts.get(part):
            print(f"\n### PART {part}: SKIPPED ###")
            continue
        print(f"\n{'='*60}\nPART {part}: {label}\n{'='*60}")
        key = f"part_{part.lower()}"
        all_results[key] = runner(fp32_model, calib_data, eval_loader, eval_fn=eval_fn)
        if "table1" in tables:
            print(accuracy_table(all_results[key], title=f"Table 1: {label}",
                                 output_dir=output_dir, filename="table1_8bit.csv"))
        if "table2" in tables:
            print(accuracy_table(all_results[key], title=f"Table 2: {label}",
                                 output_dir=output_dir, filename="table2_4bit.csv"))
        if "table3" in tables:
            print(generate_table_3(all_results[key], output_dir))
        if "table4" in tables:
            print(generate_table_4(all_results[key], output_dir))
        if "table5" in tables:
            print(generate_table_5(all_results[key], output_dir))

    if not skip_parts.get("D") and build_conv_model is not None and not skip_parts.get("D_conv"):
        print(f"\n{'='*60}\nPART D (Conv): Transform Study on Conv2d Model\n{'='*60}")
        conv_model = build_conv_model()
        conv_model.eval()
        all_results["part_d_conv"] = run_part_d_transforms(
            conv_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        print(generate_table_4(all_results["part_d_conv"], output_dir, suffix="_conv"))

    print(generate_table_6(all_results, output_dir))

    print(f"\n{'='*60}\nBLOCK SIZE SWEEP\n{'='*60}")
    all_results["block_sweep"] = run_block_size_sweep(fp32_model, calib_data, eval_loader, eval_fn=eval_fn)

    print("\n### Generating Figures ###")
    _generate_figures(all_results, output_dir)

    _save_results_json(all_results, output_dir)
    print(f"\nStudy complete. Results in {output_dir}/")
    return all_results


def plot_from_results(results_path: str, output_dir: Optional[str] = None):
    """Reload saved results.json and regenerate all tables and figures."""
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    with open(results_path) as f:
        all_results = json.load(f)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    print(f"Regenerating from {results_path} → {output_dir}")

    for key, title, filename in [
        ("part_a", "Table 1: 8-bit Format Comparison", "table1_8bit.csv"),
        ("part_b", "Table 2: 4-bit Format Comparison", "table2_4bit.csv"),
    ]:
        if key in all_results:
            print(accuracy_table(all_results[key], title=title, output_dir=output_dir, filename=filename))
    if "part_c" in all_results:
        print(generate_table_3(all_results["part_c"], output_dir))
    if "part_d" in all_results:
        print(generate_table_4(all_results["part_d"], output_dir))
        print(generate_table_5(all_results["part_d"], output_dir))
    print(generate_table_6(all_results, output_dir))
    _generate_figures(all_results, output_dir)
    print(f"\nRegeneration complete. Output in {output_dir}/")


def _generate_figures(all_results: dict, output_dir: str):
    plot_tasks = [
        (lambda d, od: qsnr_line_chart(d, title="Fig 1: Per-Layer QSNR — 8-bit Formats", colors=FORMAT_COLORS, output_dir=od),  "part_a", "fig1"),
        (lambda d, od: qsnr_line_chart(d, title="Fig 2: Per-Layer QSNR — 4-bit Formats", colors=FORMAT_COLORS, output_dir=od),  "part_b", "fig2"),
        (lambda d, od: mse_box_plot(d,   title="Fig 3: Per-Layer MSE Distribution — 8-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_a", "fig3"),
        (lambda d, od: mse_box_plot(d,   title="Fig 4: Per-Layer MSE Distribution — 4-bit Formats", colors=FORMAT_COLORS, output_dir=od), "part_b", "fig4"),
        (lambda d, od: pot_delta_bar(d, output_dir=od),                                              "part_c", "fig5"),
        (lambda d, od: histogram_overlay(d, output_dir=od),                                          None,     "fig6"),
        (lambda d, od: transform_heatmap(d, colors=FORMAT_COLORS, output_dir=od),                    "part_d", "fig7"),
        (lambda d, od: transform_pie(d,  colors=TRANSFORM_COLORS, output_dir=od),                    "part_d", "fig8"),
        (lambda d, od: transform_delta(d, colors=TRANSFORM_COLORS, output_dir=od),                   "part_d", "fig9"),
        (lambda d, od: error_vs_distribution(d, output_dir=od),                                      None,     "fig10"),
        (lambda d, od: layer_type_qsnr(d, output_dir=od),                                            None,     "fig11"),
    ]
    for fn, part_key, name in plot_tasks:
        if part_key is not None and part_key not in all_results:
            print(f"  {name}: SKIPPED (part not run)")
            continue
        try:
            fn(all_results if part_key is None else all_results[part_key], output_dir)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")


def _save_results_json(all_results: dict, output_dir: str):
    serializable: Dict[str, dict] = {}
    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        serializable[part_name] = {}
        for cfg_name, cfg_data in part_data.items():
            entry: Dict = {}
            if isinstance(cfg_data, dict):
                for key in ("accuracy", "qsnr_per_layer", "mse_per_layer"):
                    if key in cfg_data:
                        entry[key] = cfg_data[key]
            serializable[part_name][cfg_name] = entry
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print("  results.json: saved")
