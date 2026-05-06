"""
Format Study experiment runner — pure orchestration layer.

New codebase entry point::

    from src.pipeline.format_study import run_format_study

    results = run_format_study(
        build_model=my_build_fn,
        make_calib_data=my_calib_fn,
        make_eval_loader=my_loader_fn,
        eval_fn=my_eval_fn,
    )

The legacy helpers ``make_op_cfg``, ``make_op_cfg_weight_only``, and the
SmoothQuant helper functions remain here for test compatibility.
"""
from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

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
    block_sweep_line_chart,
    hierarchical_delta_bar,
    _compute_best_transform_per_layer,
)
from src.viz.tables import (
    accuracy_table,
    pot_delta_table,
    transform_matrix_table,
    transform_distribution_table,
    sensitivity_table,
)
from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS
from src.pipeline.runner import (
    ExperimentResult,
    ExperimentRunner,
    extract_metric_per_layer,
)
from src.pipeline.report import StudyReport


# ---------------------------------------------------------------------------
# Config builder helpers (keep for test compatibility)
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
    """Build per-layer OpQuantConfig with SmoothQuant on input."""
    fmt = FormatBase.from_str(fmt_name)
    no_tx = IdentityTransform()
    input_scheme = QuantScheme(format=fmt, granularity=granularity, transform=sq_transform)
    weight_scheme = QuantScheme(format=fmt, granularity=granularity, transform=no_tx)
    if weight_only:
        return OpQuantConfig(input=input_scheme, weight=weight_scheme)
    output_scheme = QuantScheme(format=fmt, granularity=granularity, transform=no_tx)
    return OpQuantConfig(input=input_scheme, weight=weight_scheme, output=output_scheme)


# ---------------------------------------------------------------------------
# Transform part helpers (keep for test compatibility)
# ---------------------------------------------------------------------------

def _make_smoothquant_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    *,
    eval_fn: Optional[Callable] = None,
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
        calib_data: List of calibration batches.
        eval_fn: ``(model, data) -> Any``. Controls how the model is called
            during activation capture. When None, falls back to
            ``fp32_model(calib_data[0])`` (single-batch direct inference).

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
            if eval_fn is not None:
                eval_fn(fp32_model, calib_data)
            else:
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
# Results serialization (keep for backward compatibility)
# ---------------------------------------------------------------------------

def _save_results_json(all_results: dict, output_dir: str):
    """Save serializable results to results.json."""
    serializable: Dict[str, dict] = {}
    for part_name, part_data in all_results.items():
        if isinstance(part_data, dict):
            serializable[part_name] = {}
            for cfg_name, cfg_data in part_data.items():
                if isinstance(cfg_data, dict):
                    entry: Dict = {}
                    for key in ("accuracy", "qsnr_per_layer", "mse_per_layer"):
                        if key in cfg_data:
                            entry[key] = cfg_data[key]
                    if entry:
                        serializable[part_name][cfg_name] = entry
        elif hasattr(part_data, '__iter__'):
            # ExperimentResult list
            from src.pipeline.runner import ExperimentResult as _ER
            serializable[part_name] = {}
            for r in part_data:
                if isinstance(r, _ER):
                    entry = {}
                    if r.quant_metrics is not None:
                        entry["accuracy"] = r.quant_metrics
                    if r.qsnr_per_layer:
                        entry["qsnr_per_layer"] = r.qsnr_per_layer
                    if r.mse_per_layer:
                        entry["mse_per_layer"] = r.mse_per_layer
                    if entry:
                        serializable[part_name][r.name] = entry
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print("  results.json: saved")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    """Timestamp for progress logging."""
    return datetime.now().strftime("%H:%M:%S")


def _expand_transform_part(
    part_cfg: dict,
    fp32_model: nn.Module,
    calib_data: list,
    eval_fn: Callable,
) -> Tuple[list, Callable]:
    """Pre-expand transform part configs and pre-compute SmoothQuant transforms.

    Returns:
        (expanded_config_list, model_for_part_callback)
    """
    base_configs = part_cfg.get("configs", [])
    has_sq = any(c.get("transform") == "smoothquant" for c in base_configs)
    if not has_sq:
        return base_configs, None

    print(f"  [{_now()}] Computing SmoothQuant scales...", end="", flush=True)
    sq_t0 = time.time()
    sq_transforms = _make_smoothquant_transforms(fp32_model, calib_data, eval_fn=eval_fn)
    sq_fused = _fuse_smoothquant_weights(fp32_model, sq_transforms)
    print(f" done ({time.time() - sq_t0:.1f}s)  [{len(sq_transforms)} layer(s) calibrated]")

    expanded = []
    for v in base_configs:
        fmt_str = v["format"]
        gran = _resolve_granularity(v)
        weight_only = v.get("weight_only", False)
        builder = make_op_cfg_weight_only if weight_only else make_op_cfg
        tx = v.get("transform", "none")

        if tx == "none":
            expanded.append(v)
        elif tx == "hadamard":
            cv = copy.deepcopy(v)
            # Hadamard replaces the transform field
            cv["transform"] = "hadamard"
            # Fix name suffix
            expanded.append(cv)
        elif tx == "smoothquant":
            # Per-layer SmoothQuant configs — generate one config per format
            cv = copy.deepcopy(v)
            cv["name"] = f"{v['name']}-SmoothQuant"
            cv.pop("transform", None)
            expanded.append(cv)
            # Expand to per-layer configs
            for lname, sq_t in sq_transforms.items():
                pl_cfg = copy.deepcopy(v)
                pl_cfg["name"] = f"{v['name']}-SQ-{lname}"
                # per-layer config for SmoothQuant
                pl_cfg.pop("transform", None)
                # We store the SQ info in the config for the model_for_part callback
                pl_cfg["_sq_layer"] = lname
                expanded.append(pl_cfg)

    return expanded, sq_fused


def _resolve_granularity(desc: dict) -> GranularitySpec:
    """Resolve granularity from config descriptor."""
    from src.pipeline.config import _resolve_granularity as _rg
    return _rg(desc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_format_study(
    build_model: Callable[[], nn.Module],
    make_calib_data: Callable[..., list],
    make_eval_loader: Callable[..., DataLoader],
    eval_fn: Callable[[nn.Module, DataLoader], Dict[str, float]],
    *,
    build_conv_model: Optional[Callable[[], nn.Module]] = None,
    output_dir: Optional[str] = None,
    skip_parts: Optional[Dict[str, bool]] = None,
    config: Optional[dict] = None,
) -> Dict[str, List[ExperimentResult]]:
    """Run all format study experiments and produce tables and figures.

    Args:
        build_model: Returns a fresh FP32 model instance.
        make_calib_data: Returns calibration data as a list of tensors.
        make_eval_loader: Returns evaluation DataLoader yielding (input, label).
        eval_fn: ``(model, dataloader) -> dict[str, float]``.
        build_conv_model: Optional Conv2d model for Conv2d validation.
        output_dir: Output directory. Default: ``results/<timestamp>/``.
        skip_parts: Dict mapping part key to True to skip.
        config: Study config dict. Default: ``STUDY_CONFIG`` from ``study_config.py``.

    Returns:
        Dict mapping part name to list of ExperimentResult.
    """
    if config is None:
        from src.pipeline.study_config import STUDY_CONFIG as config
    if skip_parts is None:
        skip_parts = {}
    if output_dir is None:
        output_dir = f"results/format_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)

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

    # ---- Pre-compute SmoothQuant transforms for parts that need them ----
    sq_cache: Dict[str, tuple] = {}  # part_name -> (expanded_configs_or_None, fused_model_or_None)
    for part_key, part_cfg in config.items():
        if skip_parts.get(part_key):
            continue
        base_configs = part_cfg.get("configs", [])
        has_sq = any(c.get("transform") == "smoothquant" for c in base_configs)
        if has_sq:
            expanded, fused = _expand_transform_part(part_cfg, fp32_model, calib_data, eval_fn)
            sq_cache[part_key] = (expanded, fused)

    # ---- Prepare per-part config lists with expanded transform parts ----
    prepared_config: Dict[str, dict] = {}
    for part_key, part_cfg in config.items():
        if skip_parts.get(part_key):
            continue
        if part_key in sq_cache:
            expanded, _ = sq_cache[part_key]
            prepared_config[part_key] = {**part_cfg, "configs": expanded}
        else:
            prepared_config[part_key] = part_cfg

    # ---- model_for_part callback ----
    _model_cache: Dict[str, nn.Module] = {}

    def _model_for_part(part_name: str) -> nn.Module:
        if part_name in _model_cache:
            return copy.deepcopy(_model_cache[part_name])
        if part_name in sq_cache:
            _, fused_model = sq_cache[part_name]
            _model_cache[part_name] = fused_model
            return copy.deepcopy(fused_model)
        return copy.deepcopy(fp32_model)

    # ---- Run experiments ----
    skip_set = {k for k, v in skip_parts.items() if v}
    runner = ExperimentRunner(prepared_config, skip_parts=skip_set)

    all_results = runner.run(
        fp32_model=fp32_model,
        eval_fn=eval_fn,
        calib_data=calib_data,
        eval_data=eval_loader,
        model_for_part=_model_for_part,
    )

    # ---- PerLayerOpt post-processing for transform parts ----
    for part_key in sq_cache:
        if part_key not in all_results:
            continue
        part_results = all_results[part_key]
        # Group results by format to compute PerLayerOpt
        fmt_groups: Dict[str, List[ExperimentResult]] = defaultdict(list)
        for r in part_results:
            base = r.name.rsplit("-", 1)[0] if "-" in r.name else r.name
            fmt_groups[base].append(r)

        for fmt_base, group in fmt_groups.items():
            # Find None/Hadamard/SmoothQuant results
            variant_results: Dict[str, dict] = {}
            for r in group:
                if r.name.endswith("-None"):
                    variant_results["None"] = {
                        "qsnr_per_layer": r.qsnr_per_layer,
                        "accuracy": r.quant_metrics or {},
                    }
                elif r.name.endswith("-Hadamard"):
                    variant_results["Hadamard"] = {
                        "qsnr_per_layer": r.qsnr_per_layer,
                        "accuracy": r.quant_metrics or {},
                    }
                elif r.name.endswith("-SmoothQuant"):
                    variant_results["SmoothQuant"] = {
                        "qsnr_per_layer": r.qsnr_per_layer,
                        "accuracy": r.quant_metrics or {},
                    }

            if len(variant_results) < 2:
                continue  # Not enough data for PerLayerOpt

            # Compute best per-layer config
            base_cfg = None
            for r in group:
                base_cfg = r
                break
            if base_cfg is None:
                continue

            sq_transforms = _make_smoothquant_transforms(fp32_model, calib_data, eval_fn=eval_fn)

            # Get fmt_str and gran from the config
            v_config = None
            for c in config.get(part_key, {}).get("configs", []):
                if c.get("name", "").startswith(fmt_base):
                    v_config = c
                    break
            if v_config is None:
                continue

            fmt_str = v_config["format"]
            gran = _resolve_granularity(v_config)
            weight_only = v_config.get("weight_only", False)
            builder = make_op_cfg_weight_only if weight_only else make_op_cfg

            per_layer_cfg = _build_per_layer_optimal_cfg(
                variant_results, sq_transforms, fmt_str, gran, builder, weight_only,
            )

            # Determine which layers got SQ
            sq_winners = {n for n in sq_transforms if isinstance(sq_transforms.get(n), SmoothQuantTransform)}
            sq_winning_layers = set()
            for layer, tx_name in _compute_best_transform_per_layer(
                {k: v["qsnr_per_layer"] for k, v in variant_results.items()}
            ).items():
                if tx_name == "SmoothQuant":
                    sq_winning_layers.add(layer)

            opt_model = _fuse_smoothquant_weights(fp32_model, sq_transforms, layer_names=sq_winning_layers)

            # Run PerLayerOpt experiment
            opt_name = f"{fmt_base}-PerLayerOpt"
            session = QuantSession(
                copy.deepcopy(opt_model), per_layer_cfg,
                calibrator=MSEScaleStrategy(),
                keep_fp32=True,
            )
            with session.calibrate():
                eval_fn(session, calib_data)
            with session.analyze(observers=[QSNRObserver(), MSEObserver()]) as ctx:
                eval_fn(session, calib_data)
            report = ctx.report()

            fp32_copy = copy.deepcopy(fp32_model)
            fp32_metrics = eval_fn(fp32_copy, eval_loader)
            quant_metrics = eval_fn(session, eval_loader)
            delta = {k: quant_metrics.get(k, 0.0) - fp32_metrics.get(k, 0.0) for k in fp32_metrics}

            pl_result = ExperimentResult(
                name=opt_name,
                fp32_metrics=fp32_metrics,
                quant_metrics=quant_metrics,
                delta=delta,
                qsnr_per_layer=extract_metric_per_layer(report, "qsnr_db"),
                mse_per_layer=extract_metric_per_layer(report, "mse"),
                cost=session.estimate_cost(),
                cost_fp32=session.estimate_cost(fp32=True),
            )
            all_results[part_key].append(pl_result)

        # Incremental save after PerLayerOpt
        _save_results_json(_results_as_old_dict(all_results), output_dir)

    # ---- Report ----
    report = StudyReport(all_results)
    report.print_summary()
    report.save(output_dir, config=config)

    total_elapsed = time.time() - study_t0
    print(f"\n  Study complete. Results in {output_dir}/")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    return all_results


# ---------------------------------------------------------------------------
# Re-generation
# ---------------------------------------------------------------------------

def _results_as_old_dict(
    results: Dict[str, List[ExperimentResult]],
) -> Dict[str, dict]:
    """Convert new-style results to old-style dict for compat with legacy functions."""
    old: Dict[str, dict] = {}
    for part_name, part_results in results.items():
        old[part_name] = {}
        for r in part_results:
            old[part_name][r.name] = {
                "accuracy": r.quant_metrics or {},
                "qsnr_per_layer": r.qsnr_per_layer,
                "mse_per_layer": r.mse_per_layer,
            }
    return old


def plot_from_results(results_path: str, output_dir: Optional[str] = None):
    """Reload saved results.json and regenerate all tables and figures."""
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    with open(results_path) as f:
        all_results = json.load(f)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    print(f"Regenerating from {results_path} → {output_dir}")

    # Tables from saved JSON (old-style dict)
    for key, title, filename in [
        ("part_a", "Table 1: 8-bit Format Comparison", "table1_8bit.csv"),
        ("part_b", "Table 2: 4-bit Format Comparison", "table2_4bit.csv"),
    ]:
        if key in all_results:
            print(accuracy_table(all_results[key], title=title, output_dir=output_dir, filename=filename))
    if "block_sweep" in all_results:
        print(accuracy_table(all_results["block_sweep"],
              title="Block Size Sweep Results", output_dir=output_dir,
              filename="block_sweep.csv"))
    if "part_hierarchical" in all_results:
        print(accuracy_table(all_results["part_hierarchical"],
              title="Hierarchical Pre-Scale Results", output_dir=output_dir,
              filename="hierarchical.csv"))
    if "part_c" in all_results:
        print(pot_delta_table(all_results["part_c"], output_dir))
    if "part_d" in all_results:
        print(transform_matrix_table(all_results["part_d"], output_dir))
        print(transform_distribution_table(all_results["part_d"], output_dir))
    print(sensitivity_table(all_results, output_dir))

    # Figures
    _generate_figures(all_results, output_dir)
    print(f"\nRegeneration complete. Output in {output_dir}/")


def _generate_figures(all_results: dict, output_dir: str):
    """Regenerate all figures from saved results dict."""
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
        (lambda d, od: block_sweep_line_chart(d, output_dir=od),                                     "block_sweep", "fig12"),
        (lambda d, od: hierarchical_delta_bar(d, output_dir=od, colors=FORMAT_COLORS),               "part_hierarchical", "fig13"),
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
