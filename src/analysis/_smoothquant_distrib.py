"""SmoothQuant pre/post distribution comparison.

Captures raw activations from the fused model and compares them against
their smoothed counterparts (X / scale).  Does NOT modify the quantize
pipeline, TransformBase, or observer interfaces — it's a standalone
analysis tool that hooks the model's forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.analysis.observers import DistributionObserver
from src.transform.smooth_quant import SmoothQuantTransform

_HIST_BINS = 64


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SmoothQuantDistribComparison:
    """Per-layer raw-vs-smoothed distribution comparison.

    Holds per-layer activation and weight statistics before and after
    SmoothQuant, plus a ranked summary of the most-improved layers.

    Attributes:
        per_layer: ``{layer_name: {role: {"raw": {...}, "smoothed": {...}}}}``
            where role is ``"activation"`` or ``"weight"``.
        improved_layers: Layer names sorted by activation dynamic range
            reduction (largest reduction first).
        summary: Global aggregates (mean DR reduction, mean outlier reduction).
    """

    per_layer: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = field(
        default_factory=dict
    )
    improved_layers: List[str] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    _DISPLAY_METRICS = (
        "dynamic_range_bits",
        "outlier_ratio",
        "crest_factor",
        "skewness",
    )

    def summary_table(self, top_k: int = 10) -> str:
        """Return a human-readable terminal table of the top-k most
        improved layers.

        Args:
            top_k: Number of layers to show (default 10).
        """
        top = self.improved_layers[:top_k]
        if not top:
            return "No layers with SmoothQuant distribution data."

        header = (
            f"{'Layer':<30s}  {'DR raw':>7s}  {'DR smooth':>9s}  "
            f"{'Δ DR':>6s}  {'Outlier raw':>10s}  {'Outlier smooth':>12s}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for name in top:
            act = self.per_layer.get(name, {}).get("activation", {})
            raw = act.get("raw", {})
            sm = act.get("smoothed", {})
            dr_raw = raw.get("dynamic_range_bits", float("nan"))
            dr_sm = sm.get("dynamic_range_bits", float("nan"))
            dr_delta = dr_raw - dr_sm
            ol_raw = raw.get("outlier_ratio", float("nan"))
            ol_sm = sm.get("outlier_ratio", float("nan"))

            short = name.split(".")[-1] if "." in name else name
            lines.append(
                f"{short:<30s}  {dr_raw:>7.2f}  {dr_sm:>9.2f}  "
                f"{dr_delta:>6.2f}  {ol_raw:>10.4f}  {ol_sm:>12.4f}"
            )

        if self.summary:
            lines.append(sep)
            lines.append(
                f"Mean DR reduction: {self.summary.get('mean_dr_reduction', 0):.2f} bits"
            )
            lines.append(
                f"Mean outlier reduction: {self.summary.get('mean_outlier_reduction', 0):.4f}"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SmoothQuantDistribComparison(layers={len(self.per_layer)}, improved={len(self.improved_layers)})"


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def compare_smoothquant_distributions(
    fp32_model: nn.Module,
    fused_model: nn.Module,
    sq_transforms: Dict[str, SmoothQuantTransform],
    calib_data,
    *,
    eval_fn: Optional[Callable] = None,
    layers: Optional[List[str]] = None,
) -> SmoothQuantDistribComparison:
    """Compare activation and weight distributions before and after SmoothQuant.

    Runs ONE forward pass through *fused_model*, capturing per-layer
    activations via hooks.  Raw stats are computed on the captured
    tensor; smoothed stats are computed on ``tensor / scale`` using the
    per-layer SmoothQuant scale from *sq_transforms*.

    Weights are compared directly: raw = original model weight,
    smoothed = fused model weight (``W * s``).

    Args:
        fp32_model: Original (un-fused) fp32 model, for raw weight stats.
        fused_model: Model with SmoothQuant weight fusion applied
            (``W = W * s``).  Activations from this model are the raw
            (pre-SmoothQuant) values.
        sq_transforms: Per-layer SmoothQuantTransform dict, as returned
            by :meth:`SmoothQuantTransform.from_model_calibration`.
        calib_data: Calibration data.  When *eval_fn* is ``None``, the
            first element is used.
        eval_fn: Optional ``(model, data) -> Any``.  When provided, it
            is invoked as ``eval_fn(fused_model, calib_data)``.  The
            function must NOT iterate over batch dimensions internally
            — the capture hooks only see the last forward pass.
        layers: If given, only compare these layers.

    Returns:
        :class:`SmoothQuantDistribComparison` with per-layer stats,
        improved-layers ranking, and summary aggregates.
    """
    obs = DistributionObserver()
    original_mods = dict(fp32_model.named_modules())
    fused_mods = dict(fused_model.named_modules())

    # --- Determine target layers ---
    layer_set: Optional[set] = set(layers) if layers is not None else None

    target_layers: Dict[str, SmoothQuantTransform] = {}
    for name, sq_t in sq_transforms.items():
        if layer_set is not None and name not in layer_set:
            continue
        if name not in fused_mods:
            continue
        target_layers[name] = sq_t

    if not target_layers:
        return SmoothQuantDistribComparison()

    # --- Hook to capture activations ---
    activations: Dict[str, Tensor] = {}

    def _hook_factory(layer_name: str):
        def _fn(_module, inp, _out):
            activations[layer_name] = inp[0].detach()

        return _fn

    hooks = []
    for name in target_layers:
        mod = fused_mods[name]
        hooks.append(mod.register_forward_hook(_hook_factory(name)))

    # --- One forward pass ---
    with torch.no_grad():
        try:
            if eval_fn is not None:
                eval_fn(fused_model, calib_data)
            elif isinstance(calib_data, (list, tuple)):
                if len(calib_data) == 0:
                    return SmoothQuantDistribComparison()
                fused_model(calib_data[0])
            else:
                fused_model(calib_data)
        finally:
            for h in hooks:
                h.remove()

    if not activations:
        return SmoothQuantDistribComparison()

    # --- Per-layer comparison ---
    per_layer: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    dr_deltas: List[tuple] = []  # (delta, layer_name)

    for name, sq_t in target_layers.items():
        act = activations.get(name)
        if act is None:
            continue

        # -- Activation: raw vs smoothed --
        # raw = activation captured from fused model forward
        raw_act_flat = act.detach().float()
        raw_act_stats = obs._measure(("tensor",), raw_act_flat, raw_act_flat)

        # smoothed = x / s (same semantic as SmoothQuantTransform.forward)
        broadcast_scale = _resolve_scale(act, sq_t)
        smoothed_act_flat = (act / broadcast_scale).detach().float()
        smooth_act_stats = obs._measure(
            ("tensor",), smoothed_act_flat, smoothed_act_flat
        )

        # Histograms for viz
        raw_hist = torch.histc(raw_act_flat, bins=_HIST_BINS).cpu().numpy()
        smooth_hist = torch.histc(smoothed_act_flat, bins=_HIST_BINS).cpu().numpy()
        raw_act_stats["_hist"] = raw_hist
        smooth_act_stats["_hist"] = smooth_hist

        entry: Dict[str, Dict[str, Dict[str, float]]] = {
            "activation": {"raw": raw_act_stats, "smoothed": smooth_act_stats}
        }

        # -- Weight: raw vs smoothed --
        orig_mod = original_mods.get(name)
        fused_mod = fused_mods.get(name)
        if (
            orig_mod is not None
            and fused_mod is not None
            and hasattr(orig_mod, "weight")
            and orig_mod.weight is not None
        ):
            w_raw = orig_mod.weight.data.detach()
            w_fused = fused_mod.weight.data.detach()
            entry["weight"] = {
                "raw": obs._measure(("tensor",), w_raw, w_raw),
                "smoothed": obs._measure(("tensor",), w_fused, w_fused),
            }

        per_layer[name] = entry

        # Delta for ranking (activation dynamic range reduction)
        dr_delta = raw_act_stats["dynamic_range_bits"] - smooth_act_stats[
            "dynamic_range_bits"
        ]
        dr_deltas.append((dr_delta, name))

    # --- Ranking ---
    dr_deltas.sort(key=lambda x: x[0], reverse=True)
    improved_layers = [name for _delta, name in dr_deltas]

    # --- Summary ---
    activ_dr_reductions = []
    activ_outlier_reductions = []
    for layer_data in per_layer.values():
        act_comp = layer_data.get("activation", {})
        r = act_comp.get("raw", {})
        s = act_comp.get("smoothed", {})
        if r and s:
            activ_dr_reductions.append(
                r["dynamic_range_bits"] - s["dynamic_range_bits"]
            )
            activ_outlier_reductions.append(
                r.get("outlier_ratio", 0) - s.get("outlier_ratio", 0)
            )

    summary: Dict[str, float] = {}
    if activ_dr_reductions:
        summary["mean_dr_reduction"] = sum(activ_dr_reductions) / len(
            activ_dr_reductions
        )
    if activ_outlier_reductions:
        summary["mean_outlier_reduction"] = sum(activ_outlier_reductions) / len(
            activ_outlier_reductions
        )

    return SmoothQuantDistribComparison(
        per_layer=per_layer,
        improved_layers=improved_layers,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_scale(act: Tensor, sq_t: SmoothQuantTransform) -> Tensor:
    """Broadcast *sq_t.scale* to match *act* shape, moving to act's device."""
    s = sq_t.scale
    axis = sq_t.channel_axis
    if axis < 0:
        axis = act.ndim + axis
    if s.device != act.device:
        s = s.to(device=act.device)
    shape = [1] * act.ndim
    shape[axis] = -1
    return s.view(*shape)
