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

To customise the search space, edit the config builders in this file.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
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
# Transform part helpers
# ---------------------------------------------------------------------------

def _make_smoothquant_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
) -> Dict[str, TransformBase]:
    """Create per-layer SmoothQuantTransform dict from a single calibration pass.

    Runs one forward pass through the FP32 model to capture each layer's
    activation and weight, then creates a per-layer SmoothQuantTransform
    with correctly-shaped per-channel scales.

    This function does NOT mutate ``fp32_model`` weights.  Weight fusion
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
    if not calib_data:
        raise ValueError("calib_data must contain at least one batch")

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
        try:
            fp32_model(calib_data[0])
        finally:
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
            sq_tx = sq_transforms.get(layer)
            if sq_tx is None:
                print(f"  Warning: {layer} selected SmoothQuant but no transform found, falling back to Identity")
                sq_tx = IdentityTransform()
            per_layer_cfg[layer] = _make_sq_op_cfg(fmt_str, gran, sq_tx, weight_only)
        else:
            per_layer_cfg[layer] = cfg_builder(fmt_str, gran, transform=tx_map[tx_name])

    return per_layer_cfg


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
    session: Optional[QuantSession] = None,
) -> dict:
    """Run one quantization experiment and return results.

    Args:
        cfg: OpQuantConfig or dict[layer_name -> OpQuantConfig].
        fp32_model: Reference FP32 model (deep-copied; not mutated).
        calib_data: List of calibration batch tensors.
        eval_loader: Evaluation DataLoader or data passed to eval_fn.
        observers: Override observer list. Default: QSNR + MSE + Histogram + Distribution.
        lsq_steps: If > 0, run LSQ pre-scale optimisation for this many steps.
        lsq_pot: Constrain LSQ scales to power-of-two.
        lsq_lr: Learning rate for LSQ optimiser.
        eval_fn: ``(model, data) -> dict[str, float]``. Controls all
            model interaction (calibration, analysis, evaluation).
            When None, falls back to ``session(data)`` direct inference.
        session: Optional[QuantSession]. When provided, skip session creation
            and use this pre-configured session (caller must set up pre-scales
            etc. before passing).

    Returns:
        Dict with keys: accuracy, fp32_accuracy, delta, report, session,
        qsnr_per_layer, mse_per_layer.
    """
    if observers is None:
        observers = [QSNRObserver(), MSEObserver(), HistogramObserver(), DistributionObserver()]
    if not calib_data:
        raise ValueError("calib_data must contain at least one batch")

    if session is None:
        session = QuantSession(
            copy.deepcopy(fp32_model), cfg,
            calibrator=MSEScaleStrategy(),
            keep_fp32=True,
        )

    # ---- Calibration ----
    print(" calibrating", end="", flush=True)
    with session.calibrate():
        if eval_fn is not None:
            eval_fn(session, calib_data)
        else:
            for batch in calib_data:
                session(batch)

    # ---- LSQ pre-scale optimisation ----
    if lsq_steps > 0:
        print(" lsq", end="", flush=True)
        session.initialize_pre_scales(calib_data, init="ones", pot=lsq_pot)
        opt = LayerwiseScaleOptimizer(
            num_steps=lsq_steps, num_batches=len(calib_data),
            optimizer="adam", lr=lsq_lr, pot=lsq_pot,
        )
        session.optimize_scales(opt, calib_data, eval_fn=eval_fn)

    # ---- Analysis ----
    print(" analyzing", end="", flush=True)
    with session.analyze(observers=observers) as ctx:
        if eval_fn is not None:
            eval_fn(session, calib_data)
        else:
            for batch in calib_data:
                session(batch)
    report = ctx.report()

    # ---- Evaluation ----
    print(" evaluating", end="", flush=True)
    if eval_fn is not None:
        fp32_metrics = eval_fn(session.fp32_model, eval_loader)
        quant_metrics = eval_fn(session, eval_loader)
        delta = {k: quant_metrics.get(k, 0.0) - fp32_metrics.get(k, 0.0)
                 for k in fp32_metrics}
    else:
        result = session.compare(eval_loader)
        fp32_metrics = result["fp32"]
        quant_metrics = result["quant"]
        delta = result["delta"]

    # Cost estimation (P6)
    cost = session.estimate_cost()
    cost_fp32 = session.estimate_cost(fp32=True)

    return {
        "accuracy": quant_metrics,
        "fp32_accuracy": fp32_metrics,
        "delta": delta,
        "report": report,
        "qsnr_per_layer": extract_metric_per_layer(report, "qsnr_db"),
        "mse_per_layer": extract_metric_per_layer(report, "mse"),
        "cost": cost,
        "cost_fp32": cost_fp32,
    }


# ---------------------------------------------------------------------------
# Config-driven part runners
# ---------------------------------------------------------------------------
from src.pipeline.config import resolve_config as _resolve_config


def _now() -> str:
    """Timestamp for progress logging."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def _fmt_desc(v: dict) -> str:
    """Human-readable description of a variant dict."""
    parts = [v["format"], v["granularity"]]
    if v.get("block_size"):
        parts.append(f"bs={v['block_size']}")
    if v.get("axis") is not None and v.get("granularity") == "per_channel":
        parts.append(f"axis={v['axis']}")
    if v.get("weight_only"):
        parts.append("weight_only")
    if v.get("transform"):
        parts.append(f"tx={v['transform']}")
    return ", ".join(parts)


def _run_simple_part(part_cfg: dict, fp32_model, calib_data, eval_loader, eval_fn) -> dict:
    """Run a 'simple' part: one experiment per variant."""
    variants = part_cfg["variants"]
    desc = part_cfg.get("description", "")
    results = {}
    total = len(variants)

    print(f"\n{'='*60}")
    print(f"  {desc}  [{total} variant(s)]")
    print(f"{'='*60}")

    t0 = time.time()
    for i, v in enumerate(variants, 1):
        name = v["name"]
        cfg = _resolve_config(v)
        print(f"  [{i}/{total}] {name} ({_fmt_desc(v)}) ...", end="", flush=True)
        vt0 = time.time()
        results[name] = run_experiment(cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn)
        vt = time.time() - vt0
        acc = results[name].get("accuracy", {})
        if isinstance(acc, dict):
            acc_str = ", ".join(f"{k}={v:.4f}" for k, v in acc.items())
        else:
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
        print(f" done ({vt:.1f}s)  acc={{{acc_str}}}")

    elapsed = time.time() - t0
    baseline_key = list(results.keys())[0] if results else ""
    results["FP32 (baseline)"] = {"accuracy": results[baseline_key]["fp32_accuracy"]} if baseline_key else {}
    print(f"  ---- {desc} complete: {total}/{total} finished, total {elapsed:.1f}s ----")
    return results


def _run_pot_scaling_part(part_cfg: dict, fp32_model, calib_data, eval_loader, eval_fn) -> dict:
    """Run a 'pot_scaling' part: each variant × {pot=False, pot=True}."""
    variants = part_cfg["variants"]
    lsq_steps = part_cfg.get("lsq_steps", 100)
    desc = part_cfg.get("description", "")
    results = {}

    expanded = []
    for v in variants:
        for pot in (False, True):
            suffix = "PoT" if pot else "FP32"
            expanded.append((f"{v['name']}-{suffix}", v, pot))

    total = len(expanded)
    print(f"\n{'='*60}")
    print(f"  {desc}  [{total} variant(s), lsq_steps={lsq_steps}]")
    print(f"{'='*60}")

    t0 = time.time()
    for i, (name, v, pot) in enumerate(expanded, 1):
        cfg = _resolve_config(v)
        print(f"  [{i}/{total}] {name} ({_fmt_desc(v)}, pot={pot}) ...", end="", flush=True)
        vt0 = time.time()
        results[name] = run_experiment(
            cfg, fp32_model, calib_data, eval_loader,
            lsq_steps=lsq_steps, lsq_pot=pot, eval_fn=eval_fn,
        )
        vt = time.time() - vt0
        acc = results[name].get("accuracy", {})
        if isinstance(acc, dict):
            acc_str = ", ".join(f"{k}={v:.4f}" for k, v in acc.items())
        else:
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
        print(f" done ({vt:.1f}s)  acc={{{acc_str}}}")

    elapsed = time.time() - t0
    baseline_key = list(results.keys())[0] if results else ""
    results["FP32 (baseline)"] = {"accuracy": results[baseline_key]["fp32_accuracy"]} if baseline_key else {}
    print(f"  ---- {desc} complete: {total}/{total} finished, total {elapsed:.1f}s ----")
    return results


def _run_transform_part(part_cfg: dict, fp32_model, calib_data, eval_loader, eval_fn) -> dict:
    """Run a 'transform' part: per-format None/Hadamard/SmoothQuant/PerLayerOpt."""
    variants = part_cfg["variants"]
    desc = part_cfg.get("description", "")

    print(f"\n{'='*60}")
    print(f"  {desc}  [{len(variants)} format(s)]")
    print(f"{'='*60}")

    t0 = time.time()
    print(f"  [{_now()}] Computing SmoothQuant scales...", end="", flush=True)
    sq_t0 = time.time()
    sq_transforms = _make_smoothquant_transforms(fp32_model, calib_data)
    sq_fp32_model = _fuse_smoothquant_weights(fp32_model, sq_transforms)
    print(f" done ({time.time() - sq_t0:.1f}s)  [{len(sq_transforms)} layer(s) calibrated]")

    all_results: Dict[str, dict] = {}
    for fmt_idx, v in enumerate(variants, 1):
        fmt_name = v["name"]
        fmt_str = v["format"]
        weight_only = v.get("weight_only", False)
        gran = _resolve_granularity(v)
        builder = make_op_cfg_weight_only if weight_only else make_op_cfg

        print(f"\n  -- [{fmt_idx}/{len(variants)}] Transform study for {fmt_name} ({_fmt_desc(v)}) --")

        # Per-layer SmoothQuant configs
        sq_per_layer_cfg: Dict[str, OpQuantConfig] = {
            lname: _make_sq_op_cfg(fmt_str, gran, sq_tx, weight_only)
            for lname, sq_tx in sq_transforms.items()
        }
        fallback_cfg = builder(fmt_str, gran, transform=None)
        for mname, _ in fp32_model.named_modules():
            if mname and mname not in sq_per_layer_cfg:
                sq_per_layer_cfg[mname] = fallback_cfg

        tx_variants = [
            ("None", None),
            ("Hadamard", HadamardTransform()),
            ("SmoothQuant", "sq"),  # special: uses sq_per_layer_cfg + sq_fp32_model
        ]
        variant_results: Dict[str, dict] = {}

        for tx_name, tx in tx_variants:
            label = f"{fmt_name}-{tx_name}"
            print(f"    {label} ...", end="", flush=True)
            vt0 = time.time()
            if tx_name == "SmoothQuant":
                variant_results[tx_name] = run_experiment(
                    sq_per_layer_cfg, sq_fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
                )
            else:
                variant_results[tx_name] = run_experiment(
                    builder(fmt_str, gran, transform=tx),
                    fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
                )
            vt = time.time() - vt0
            acc = variant_results[tx_name].get("accuracy", {})
            if isinstance(acc, dict):
                acc_str = ", ".join(f"{k}={v:.4f}" for k, v in acc.items())
            else:
                acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
            print(f" done ({vt:.1f}s)  acc={{{acc_str}}}")

        # Per-layer optimal
        print(f"    {fmt_name}-PerLayerOpt ...", end="", flush=True)
        vt0 = time.time()
        layer_best_tx = _compute_best_transform_per_layer(
            {k: v["qsnr_per_layer"] for k, v in variant_results.items()}
        )
        sq_winning = {n for n, tx in layer_best_tx.items() if tx == "SmoothQuant"}
        opt_model = _fuse_smoothquant_weights(fp32_model, sq_transforms, layer_names=sq_winning)

        variant_results["PerLayerOpt"] = run_experiment(
            _build_per_layer_optimal_cfg(variant_results, sq_transforms, fmt_str, gran, builder, weight_only),
            opt_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        vt = time.time() - vt0
        acc = variant_results["PerLayerOpt"].get("accuracy", {})
        if isinstance(acc, dict):
            acc_str = ", ".join(f"{k}={v:.4f}" for k, v in acc.items())
        else:
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
        n_sq = sum(1 for tx in layer_best_tx.values() if tx == "SmoothQuant")
        n_hd = sum(1 for tx in layer_best_tx.values() if tx == "Hadamard")
        n_no = sum(1 for tx in layer_best_tx.values() if tx == "None")
        print(f" done ({vt:.1f}s)  acc={{{acc_str}}}  [per-layer: {n_no} None + {n_hd} Hadamard + {n_sq} SmoothQuant]")

        all_results[fmt_name] = variant_results

    elapsed = time.time() - t0
    print(f"  ---- {desc} complete, total {elapsed:.1f}s ----")
    return all_results


def _run_block_sweep_part(part_cfg: dict, fp32_model, calib_data, eval_loader, eval_fn) -> dict:
    """Run a 'block_sweep' part: sweep block sizes for a format."""
    fmt_name = part_cfg.get("format", "int8")
    block_sizes = part_cfg.get("block_sizes", [16, 32, 64, 128])
    desc = part_cfg.get("description", "")

    print(f"\n{'='*60}")
    print(f"  {desc}  [format={fmt_name}, sizes={block_sizes}]")
    print(f"{'='*60}")

    results = {}
    t0 = time.time()
    for i, bs in enumerate(block_sizes, 1):
        gran = GranularitySpec.per_block(size=bs, axis=-1)
        label = f"{fmt_name}-blk{bs}"
        print(f"  [{i}/{len(block_sizes)}] {label} ...", end="", flush=True)
        vt0 = time.time()
        try:
            results[label] = run_experiment(
                make_op_cfg(fmt_name, gran), fp32_model, calib_data, eval_loader, eval_fn=eval_fn,
            )
            vt = time.time() - vt0
            acc = results[label].get("accuracy", {})
            if isinstance(acc, dict):
                acc_str = ", ".join(f"{k}={v:.4f}" for k, v in acc.items())
            else:
                acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
            print(f" done ({vt:.1f}s)  acc={{{acc_str}}}")
        except Exception as e:
            print(f" FAILED: {e}")
    elapsed = time.time() - t0
    print(f"  ---- {desc} complete, total {elapsed:.1f}s ----")
    return results


def _run_hierarchical_part(part_cfg: dict, fp32_model, calib_data, eval_loader, eval_fn) -> dict:
    """Run a 'hierarchical' part: pre-scale init + standard experiment per variant.

    For each variant:
    1. Resolve config (MX per_block format)
    2. Create QuantSession
    3. session.initialize_pre_scales with part_cfg["pre_scale"] params
    4. Standard calibrate → analyze → evaluate flow
    """
    variants = part_cfg["variants"]
    pre_scale_cfg = part_cfg.get("pre_scale", {})
    desc = part_cfg.get("description", "")
    results = {}
    total = len(variants)

    print(f"\n{'='*60}")
    print(f"  {desc}  [{total} variant(s)]")
    print(f"  Pre-Scale: init={pre_scale_cfg.get('init')}, "
          f"pot={pre_scale_cfg.get('pot')}, "
          f"granularity={pre_scale_cfg.get('granularity')}")
    print(f"{'='*60}")

    t0 = time.time()
    for i, v in enumerate(variants, 1):
        name = v["name"]
        cfg = _resolve_config(v)
        print(f"  [{i}/{total}] {name} ({_fmt_desc(v)}) ...", end="", flush=True)
        vt0 = time.time()

        # Build session
        session = QuantSession(
            copy.deepcopy(fp32_model), cfg,
            calibrator=MSEScaleStrategy(),
            keep_fp32=True,
        )

        # -- Pre-scale initialization (before calibration) --
        ps_init = pre_scale_cfg.get("init")
        if ps_init:
            print(" pre-scale", end="", flush=True)
            session.initialize_pre_scales(
                calib_data,
                init=ps_init,
                pot=pre_scale_cfg.get("pot", False),
                granularity=pre_scale_cfg.get("granularity", "per_tensor"),
                trainable=pre_scale_cfg.get("trainable", False),
                channel_axis=pre_scale_cfg.get("channel_axis", -1),
            )

        # Delegate to run_experiment with pre-configured session
        results[name] = run_experiment(
            cfg, fp32_model, calib_data, eval_loader,
            eval_fn=eval_fn, session=session,
        )

        vt = time.time() - vt0
        acc = results[name].get("accuracy", {})
        if isinstance(acc, dict):
            acc_str = ", ".join(f"{k}={v:.4f}" for k, v in acc.items())
        else:
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
        print(f" done ({vt:.1f}s)  acc={{{acc_str}}}")

    elapsed = time.time() - t0
    baseline_key = list(results.keys())[0] if results else ""
    results["FP32 (baseline)"] = {"accuracy": results[baseline_key]["fp32_accuracy"]} if baseline_key else {}
    print(f"  ---- {desc} complete: {total}/{total} finished, total {elapsed:.1f}s ----")
    return results


def _resolve_granularity(v: dict) -> GranularitySpec:
    """Convert a variant dict's granularity fields into a GranularitySpec."""
    gran_type = v.get("granularity", "per_tensor")
    if gran_type == "per_block":
        return GranularitySpec.per_block(size=v.get("block_size", 32), axis=v.get("axis", -1))
    elif gran_type == "per_channel":
        return GranularitySpec.per_channel(axis=v.get("axis", -1))
    else:
        return GranularitySpec.per_tensor()


# ---------------------------------------------------------------------------
# Table generators (Tables 3–6; Tables 1–2 delegate to src.viz.tables)
# ---------------------------------------------------------------------------

def generate_table_3(part_c: dict, output_dir: str) -> str:
    """Table 3: FP32 vs PoT accuracy delta."""
    baseline_acc = 0.0
    for name, data in part_c.items():
        if name == "FP32 (baseline)":
            acc = data.get("accuracy", {})
            baseline_acc = float(acc.get("accuracy", 0.0)) if isinstance(acc, dict) else float(acc or 0.0)
            break

    rows = []
    for name, data in part_c.items():
        if name == "FP32 (baseline)":
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
             max(m["mse"]) if m["mse"] else 0.0,
             min(m["qsnr"]) if m["qsnr"] else 0.0)
            for layer, m in layer_metrics.items()
        ),
        key=lambda x: x[1], reverse=True,
    )[:10]

    lines = [f"\n{'='*80}", "Table 6: Top-10 Most Sensitive Layers", "=" * 80,
             f"{'#':<4} {'Layer':<28} {'Max MSE':<18} {'Min QSNR (dB)':<15}", "-" * 80]
    for i, (layer, mse, qsnr) in enumerate(ranking, 1):
        lines.append(f"{i:<4} {layer:<28} {mse:<18.6e} {qsnr:<15.2f}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table6_sensitivity.csv", "w") as f:
        f.write("Rank,Layer,Max_MSE,Min_QSNR_dB\n")
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
    config: Optional[dict] = None,
) -> Dict[str, dict]:
    """Run all format study experiments and produce tables and figures.

    Args:
        build_model: Returns a fresh FP32 model instance.
        make_calib_data: Returns calibration data as a list of tensors.
        make_eval_loader: Returns evaluation DataLoader yielding (input, label).
        eval_fn: ``(model, dataloader) -> dict[str, float]``.
        build_conv_model: Optional Conv2d model for Part D Conv2d validation.
        output_dir: Output directory. Default: ``results/<timestamp>/``.
        skip_parts: Dict mapping part key to True to skip (e.g. ``{"part_a": True}``).
        config: Study config dict. Default: ``STUDY_CONFIG`` from ``study_config.py``.

    Returns:
        Dict mapping experiment name to result dict.
    """
    if config is None:
        from src.pipeline.study_config import STUDY_CONFIG as config
    if skip_parts is None:
        skip_parts = {}
    if output_dir is None:
        output_dir = f"results/format_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)

    # ---- Dispatch table for part types ----
    _RUNNERS = {
        "simple":        _run_simple_part,
        "pot_scaling":   _run_pot_scaling_part,
        "transform":     _run_transform_part,
        "block_sweep":   _run_block_sweep_part,
        "hierarchical":  _run_hierarchical_part,
    }
    _TABLE_GENERATORS = {
        "table1": lambda r, od: accuracy_table(r, title="Table 1", output_dir=od, filename="table1_8bit.csv"),
        "table2": lambda r, od: accuracy_table(r, title="Table 2", output_dir=od, filename="table2_4bit.csv"),
        "table3": generate_table_3,
        "table4": generate_table_4,
        "table5": generate_table_5,
    }

    print("=" * 60)
    print(f"  Quantization Format Precision Study")
    print(f"  Output: {output_dir}")
    print(f"  Started: {_now()}")
    print("=" * 60)

    study_t0 = time.time()
    fp32_model = build_model()
    fp32_model.eval()
    calib_data = make_calib_data()
    eval_loader = make_eval_loader()
    all_results: Dict[str, dict] = {}

    # ---- Run configured parts ----
    for part_key, part_cfg in config.items():
        if skip_parts.get(part_key):
            print(f"\n  [{_now()}] PART {part_key}: SKIPPED")
            continue

        part_type = part_cfg.get("type", "simple")
        runner = _RUNNERS.get(part_type)
        if runner is None:
            print(f"  [{_now()}] PART {part_key}: UNKNOWN type '{part_type}', skipped")
            continue

        all_results[part_key] = runner(part_cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn)

        # Generate tables for this part
        table_name = part_cfg.get("table")
        if table_name and table_name in _TABLE_GENERATORS:
            try:
                print(_TABLE_GENERATORS[table_name](all_results[part_key], output_dir))
            except Exception as e:
                print(f"  Warning: table {table_name} generation failed: {e}")

        # Incremental save after part loop
        _save_results_json(all_results, output_dir)

    # ---- Optional Part D on Conv2d model ----
    part_d_cfg = config.get("part_d")
    if part_d_cfg and not skip_parts.get("part_d_conv") and build_conv_model is not None:
        print(f"\n{'='*60}")
        print(f"  Part D (Conv): Transform Study on Conv2d Model")
        print(f"{'='*60}")
        conv_model = build_conv_model()
        conv_model.eval()
        all_results["part_d_conv"] = _run_transform_part(
            part_d_cfg, conv_model, calib_data, eval_loader, eval_fn=eval_fn,
        )
        try:
            print(generate_table_4(all_results["part_d_conv"], output_dir, suffix="_conv"))
        except Exception as e:
            print(f"  Warning: table4_conv generation failed: {e}")

        # Incremental save after Part D Conv
        _save_results_json(all_results, output_dir)

    # ---- Cross-part table (Table 6: sensitivity) ----
    print(generate_table_6(all_results, output_dir))

    # ---- Figures ----
    print(f"\n  [{_now()}] Generating figures ...")
    fig_t0 = time.time()
    _generate_figures(all_results, output_dir)
    print(f"  Figures done ({time.time() - fig_t0:.1f}s)")

    _save_results_json(all_results, output_dir)

    total_elapsed = time.time() - study_t0
    print(f"\n  Study complete. Results in {output_dir}/")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
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
    if "block_sweep" in all_results:
        print(accuracy_table(all_results["block_sweep"],
               title="Block Size Sweep Results", output_dir=output_dir,
               filename="block_sweep.csv"))
    if "part_hierarchical" in all_results:
        print(accuracy_table(all_results["part_hierarchical"],
               title="Hierarchical Pre-Scale Results", output_dir=output_dir,
               filename="hierarchical.csv"))
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
        if not isinstance(part_data, dict):
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
