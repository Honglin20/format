"""Harness chart bridge — render_chart versions of all analysis capabilities.

Provides U1–U6 + block/provenance harness visualisations alongside existing
matplotlib path.  Each function emits ``render_chart()`` when harness is
available and *optionally* saves a matplotlib figure when ``output_dir`` is
provided.  Existing ``src/viz/`` code is never modified.

Observer requirements (must be attached during ``Session.run()``):
  - U1 (distribution_fit_chart): DistributionFitObserver
  - U2 (intervention_chart): QSNRObserver (via qsnr_by_role)
  - U3 (channel_heterogeneity_chart): PerBlockQSNRObserver
  - U4 (depth_decay_chart): QSNRObserver (via qsnr_by_role)
  - U5 (error_propagation_chart): QSNRObserver (via qsnr_by_role)
  - U6 (block_qsnr_box_chart): PerBlockQSNRObserver
  - block_error_chart / channel_error_chart: PerBlockQSNRObserver
  - error_provenance_chart: QSNRObserver (via qsnr_by_role)

Usage::

    from src.api.harness_charts import all_harness_charts

    # Single-config (SessionResult)
    all_harness_charts(result, label="MXInt8")

    # Individual charts
    from src.api.harness_charts import depth_decay_chart
    depth_decay_chart(result)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session._result import SessionResult
    from src.report._study_report import StudyReport

from src.api._chart_helpers import (
    _chart, _get_per_block_qsnr, _get_per_channel_qsnr,
    _get_dist_metrics, _get_fit_metrics, _block_stats,
    _linear_layer_names, QSNR_REF,
)


# =====================================================================
# U1: DistributionFitObserver chart
# =====================================================================

def distribution_fit_chart(result: "SessionResult", *, label: str = ""):
    """U1: Parameterised distribution fitting results.

    Requires: DistributionFitObserver attached during Session.run().
    """
    obs = result.observers_data
    rows = []
    bar_data = []

    for layer in sorted(obs.keys()):
        for role in sorted(obs[layer].keys()):
            fit = _get_fit_metrics(obs, layer, role)
            if not fit:
                continue
            best = fit.get("best_fit", "N/A")
            ks = fit.get("best_fit_ks")
            ranking = fit.get("fit_ranking", [])

            rows.append({
                "layer": layer, "role": role,
                "best_fit": best,
                "ks_stat": round(ks, 4) if ks is not None else "N/A",
                "n_candidates": len(ranking),
            })

            for dist_name, ks_val in ranking[:3]:
                bar_data.append({
                    "layer": layer, "role": role,
                    "distribution": dist_name,
                    "ks_stat": round(ks_val, 4),
                })

    if rows:
        _chart(rows, "table", x="layer", y="best_fit",
               label=label, title="Distribution Fit: Best Parametric Fit per Layer")

    if bar_data:
        _chart(bar_data, "bar", x="layer", y="ks_stat", hue="distribution",
               label=label, title="Distribution Fit: Top-3 Candidates (KS Statistic)")


# =====================================================================
# U2: InterventionPlanner chart
# =====================================================================

def intervention_chart(result: "SessionResult", *, k: int = 5, label: str = ""):
    """U2: Intervention plan — top-k layers to boost.

    Requires: QSNRObserver (populates qsnr_by_role).
    """
    plan = result.plan.top_k_boost(k=k, role="auto", target_bits=8)
    if not plan.overrides:
        return

    rows = []
    bar_data = []
    changes = plan.metadata.get("changes", {})

    for layer in sorted(plan.overrides):
        info = changes.get(layer, {})
        what = info.get("what", "config override")
        why = info.get("why", "")

        current_qsnr = None
        for role in ("output", "weight", "input"):
            v = result.qsnr_by_role.get(role, {}).get(layer)
            if v is not None:
                current_qsnr = v
                break

        rows.append({
            "layer": layer,
            "current_qsnr": round(current_qsnr, 1) if current_qsnr else "N/A",
            "action": what,
            "reason": why[:50],
        })
        if current_qsnr:
            bar_data.append({"layer": layer, "qsnr_db": round(current_qsnr, 1)})

    if rows:
        _chart(rows, "table", x="layer", y="current_qsnr",
               label=label, title=f"Intervention Plan: Top-{k} Layers to Boost")

    # U2b bar removed: duplicates P1 Accum QSNR bar


# =====================================================================
# U3: Channel Heterogeneity chart
# =====================================================================

def channel_heterogeneity_chart(
    result: "SessionResult",
    layer: str,
    *,
    role: str = "weight",
    label: str = "",
):
    """U3: Per-channel QSNR heterogeneity analysis.

    Requires: PerBlockQSNRObserver attached during Session.run().
    """
    obs = result.observers_data
    channels = _get_per_channel_qsnr(obs, layer, role)

    if not channels:
        channels = _get_per_block_qsnr(obs, layer, role)
        unit_label = "block"
    else:
        unit_label = "channel"

    if not channels:
        return

    vals = list(channels.values())
    n = len(vals)
    mean_v = sum(vals) / n
    std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / n) if n > 1 else 0.0
    threshold = mean_v - std_v
    cv = std_v / abs(mean_v) if mean_v != 0 else float("inf")

    stats_row = [{
        "layer": layer, "role": role, "unit": unit_label,
        "n_units": n, "mean_qsnr": round(mean_v, 1),
        "std_qsnr": round(std_v, 1), "cv": round(cv, 3),
        "min_qsnr": round(min(vals), 1), "max_qsnr": round(max(vals), 1),
        "n_outliers": sum(1 for v in vals if v < threshold),
    }]
    _chart(stats_row, "table", x="layer", y="mean_qsnr",
           label=label,
           title=f"Channel Heterogeneity: {layer} ({role}) — CV={cv:.3f}")

    sorted_ch = sorted(channels.items(), key=lambda x: x[1])[:20]
    bar_data = [
        {f"{unit_label}_idx": idx, "qsnr_db": round(qsnr, 1),
         "is_outlier": "yes" if qsnr < threshold else "no"}
        for idx, qsnr in sorted_ch
    ]
    if bar_data:
        _chart(bar_data, "bar", x=f"{unit_label}_idx", y="qsnr_db",
               hue="is_outlier", label=label,
               title=f"Channel Heterogeneity: {layer} ({role}) — Top-20 Worst")


# =====================================================================
# U4: Depth Decay chart
# =====================================================================

def depth_decay_chart(result: "SessionResult", *, role: str = "output",
                      label: str = ""):
    """U4: Network depth vs QSNR decay trend.

    Requires: QSNRObserver (populates qsnr_by_role for depth ordering).
    """
    data = result.diagnose.depth_decay_data(role)
    if not data:
        return

    line_data = []
    table_data = []
    for depth, layer_name, qsnr in data:
        if math.isfinite(qsnr):
            line_data.append({"depth": depth, "layer": layer_name, "qsnr_db": round(qsnr, 1)})
            table_data.append({"depth": depth, "layer": layer_name, "qsnr_db": round(qsnr, 1)})

    if line_data:
        _chart(line_data, "line", x="depth", y="qsnr_db",
               label=label, title=f"Depth Decay: QSNR vs Layer Depth ({role})")
        _chart(table_data, "table", x="depth", y="qsnr_db",
               label=label, title=f"Depth Decay Table ({role})")


# =====================================================================
# U5: Error Propagation chart
# =====================================================================

def _classify_error_source(accum_qsnr, local_qsnr) -> str:
    """Classify layer error source from accum vs local QSNR gap."""
    if accum_qsnr is None or not math.isfinite(accum_qsnr):
        return "Local"
    diff = abs(accum_qsnr - local_qsnr)
    if diff < 3.0:
        return "Source"
    elif diff < 10.0:
        return "Mixed"
    return "Propagated"


def error_propagation_chart(result: "SessionResult", *, linear_only: bool = True,
                            label: str = ""):
    """U5: Error source classification — Source/Mixed/Propagated per layer.

    Produces a **source classification table** only (no per-role bar —
    see ``error_provenance_chart`` for per-role attribution).

    Requires: QSNRObserver (populates qsnr_by_role).
    """
    obs = result.observers_data
    qsnr_by_role = result.qsnr_by_role or {}
    allowed = _linear_layer_names(obs) if (linear_only and obs) else None

    table_data = []

    for layer in sorted(obs.keys()):
        if allowed is not None and layer not in allowed:
            continue

        role_qsnrs = {}
        for role in ("input", "weight", "output"):
            v = qsnr_by_role.get(role, {}).get(layer)
            if v is not None and math.isfinite(v):
                role_qsnrs[role] = v

        if not role_qsnrs:
            continue

        local_qsnr = role_qsnrs.get("output", min(role_qsnrs.values()))
        accum_qsnr = result.accum_qsnr_per_layer.get(layer)
        source = _classify_error_source(accum_qsnr, local_qsnr)
        dominant = min(role_qsnrs, key=role_qsnrs.get)

        table_data.append({
            "layer": layer,
            "input_qsnr": round(role_qsnrs.get("input", 0), 1) if "input" in role_qsnrs else "N/A",
            "weight_qsnr": round(role_qsnrs.get("weight", 0), 1) if "weight" in role_qsnrs else "N/A",
            "output_qsnr": round(role_qsnrs.get("output", 0), 1) if "output" in role_qsnrs else "N/A",
            "dominant": dominant,
            "error_source": source,
        })

    if table_data:
        _chart(table_data, "table", x="layer", y="output_qsnr",
               label=label, title="Error Source Classification: Source/Mixed/Propagated")


# =====================================================================
# U6: Per-Block QSNR Box Plot
# =====================================================================

def block_qsnr_box_chart(result: "SessionResult", *, linear_only: bool = True,
                         label: str = ""):
    """U6: Cross-layer per-block QSNR distribution via box chart.

    Requires: PerBlockQSNRObserver attached during Session.run().
    """
    obs = result.observers_data
    allowed = _linear_layer_names(obs) if (linear_only and obs) else None

    box_data = []
    stats_rows = []

    for layer_name in sorted(obs.keys()):
        if allowed is not None and layer_name not in allowed:
            continue

        for role in ("input", "weight", "output"):
            blocks = _get_per_block_qsnr(obs, layer_name, role)
            if not blocks:
                continue

            stats = _block_stats(blocks)
            group_label = f"{layer_name} ({role})"

            for _idx, qsnr in blocks.items():
                box_data.append({"group": group_label, "value": round(qsnr, 1)})

            stats_rows.append({"layer": layer_name, "role": role, **stats})

    if box_data:
        _chart(box_data, "box", x="group", y="value",
               label=label, title="Per-Block QSNR Distribution by Layer (Box Plot)")


# U6b stats table removed: redundant with heatmap in layer_deep_dive


# =====================================================================
# Block Error chart
# =====================================================================

def block_error_chart(result: "SessionResult", layer: str, *, role: str = "weight",
                      top_k: int = 10, label: str = ""):
    """Per-block QSNR bar + stats for a single layer.

    Requires: PerBlockQSNRObserver attached during Session.run().
    """
    obs = result.observers_data
    blocks = _get_per_block_qsnr(obs, layer, role)
    if not blocks:
        return

    sorted_all = sorted(blocks.items(), key=lambda x: x[1])
    if len(sorted_all) > 100:
        step = len(sorted_all) // 100
        sampled = sorted_all[::step]
    else:
        sampled = sorted_all

    bar_all = [{"block_idx": idx, "qsnr_db": round(q, 1)} for idx, q in sampled]
    _chart(bar_all, "bar", x="block_idx", y="qsnr_db",
           label=label, title=f"Block QSNR: {layer} ({role}) — All Blocks (sorted)")

    worst = sorted_all[:top_k]
    bar_worst = [{"block_idx": idx, "qsnr_db": round(q, 1)} for idx, q in worst]
    _chart(bar_worst, "bar", x="block_idx", y="qsnr_db",
           label=label, title=f"Block QSNR: {layer} ({role}) — Top-{top_k} Worst Blocks")

    stats = _block_stats(blocks)
    _chart([{"layer": layer, "role": role, **stats}], "table",
           x="layer", y="mean",
           label=label, title=f"Block QSNR Stats: {layer} ({role})")


# =====================================================================
# Channel Error chart
# =====================================================================

def channel_error_chart(result: "SessionResult", layer: str, *, role: str = "input",
                        top_k: int = 20, label: str = ""):
    """Per-channel error bar chart for activations.

    Requires: PerBlockQSNRObserver attached during Session.run().
    """
    obs = result.observers_data
    channels = _get_per_channel_qsnr(obs, layer, role)

    if not channels:
        channels = _get_per_block_qsnr(obs, layer, role)
        unit = "block"
    else:
        unit = "channel"

    if not channels:
        return

    vals = list(channels.values())
    mean_v = sum(vals) / len(vals)
    std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    threshold = mean_v - std_v

    sorted_ch = sorted(channels.items(), key=lambda x: x[1])[:top_k]
    bar_data = [
        {f"{unit}_idx": idx, "qsnr_db": round(qsnr, 1),
         "is_outlier": "yes" if qsnr < threshold else "no"}
        for idx, qsnr in sorted_ch
    ]

    if bar_data:
        _chart(bar_data, "bar", x=f"{unit}_idx", y="qsnr_db", hue="is_outlier",
               label=label, title=f"Channel Error: {layer} ({role}) — Top-{top_k} Worst")


# =====================================================================
# ErrorProvenance chart (per-role attribution — distinct from U5)
# =====================================================================

def error_provenance_chart(result: "SessionResult", *, linear_only: bool = True,
                           label: str = ""):
    """Per-role error attribution bar + accum vs local source classification.

    Unlike ``error_propagation_chart`` (source classification table only),
    this produces a **per-role error contribution bar** plus an
    **accum vs local comparison table**.

    Requires: QSNRObserver (populates qsnr_by_role).
    """
    qsnr_by_role = result.qsnr_by_role or {}
    obs = result.observers_data
    allowed = _linear_layer_names(obs) if (linear_only and obs) else None

    role_bar = []
    source_rows = []

    for layer in sorted(obs.keys()):
        if allowed is not None and layer not in allowed:
            continue

        role_qsnrs = {}
        for role in ("input", "weight", "output"):
            v = qsnr_by_role.get(role, {}).get(layer)
            if v is not None and math.isfinite(v):
                role_qsnrs[role] = v

        if not role_qsnrs:
            continue

        dominant = min(role_qsnrs, key=role_qsnrs.get)

        # Per-role error contribution bar (unique to provenance)
        for role, qsnr in role_qsnrs.items():
            role_bar.append({
                "layer": layer,
                "error_contribution": round(QSNR_REF - qsnr, 2),
                "source": role,
            })

        # Accum vs local source table
        accum = result.accum_qsnr_per_layer.get(layer)
        local = role_qsnrs.get("output", min(role_qsnrs.values()))
        source = _classify_error_source(accum, local)

        source_rows.append({
            "layer": layer,
            "output_qsnr": round(local, 1),
            "accum_qsnr": round(accum, 1) if accum and math.isfinite(accum) else "N/A",
            "dominant_role": dominant,
            "error_source": source,
        })

    if role_bar:
        _chart(role_bar, "bar", x="layer", y="error_contribution", hue="source",
               label=label, title="Error Provenance: Per-Role Error Contribution")

    if source_rows:
        _chart(source_rows, "table", x="layer", y="output_qsnr",
               label=label, title="Error Provenance: Accum vs Local Source Classification")


# =====================================================================
# Multi-config charts (StudyReport input)
# =====================================================================

def _iter_study_results(study_report: "StudyReport"):
    """Iterate all SessionResult objects from a StudyReport via public API."""
    try:
        df = study_report.to_dataframe()
        # to_dataframe() internally iterates _results, confirming data exists
    except Exception:
        return

    # Access via the internal structure — StudyReport doesn't expose
    # a public result iterator yet.  Use results_by_part if available.
    for results_list in getattr(study_report, "_results", {}).values():
        for r in results_list:
            yield r


def cross_config_ranking_chart(study_report: "StudyReport", *, k: int = 5,
                               label: str = ""):
    """Cross-config layer ranking — consistently worst layers across configs.

    Requires: StudyReport with multiple QuantConfig results.
    """
    from src.analysis.cross_config_ranking import CrossConfigLayerRanking
    ranking = CrossConfigLayerRanking.from_study(study_report)

    consistent = ranking.consistent_worst(k=k)
    if consistent:
        bar_data = [{"layer": layer, "avg_qsnr_db": round(qsnr, 1)}
                    for layer, qsnr in consistent]
        _chart(bar_data, "bar", x="layer", y="avg_qsnr_db",
               label=label, title=f"Cross-Config: Top-{k} Consistently Worst Layers")

    dominance = ranking.role_dominance_cross_config(k=k)
    if dominance:
        rows = []
        for entry in dominance:
            layer_name = entry.get("layer", "")
            for cfg_info in entry.get("configs", []):
                rows.append({
                    "layer": layer_name,
                    "config": cfg_info.get("config", ""),
                    "dominant_role": cfg_info.get("dominant_role", ""),
                    "qsnr_db": round(cfg_info.get("qsnr", 0), 1),
                })
        if rows:
            _chart(rows, "table", x="layer", y="qsnr_db",
                   label=label, title="Cross-Config: Role Dominance per Layer")


def transform_effect_chart(study_report: "StudyReport", *, label: str = ""):
    """Transform recovery effect — accuracy gain per config.

    Requires: StudyReport with transform/no-transform config pairs.
    """
    from src.analysis.transform_effect import TransformEffectReport
    report = TransformEffectReport.from_study(study_report)

    recovery = report.per_config_recovery()
    if recovery:
        bar_data = [
            {"config": entry.get("config", ""),
             "recovery_pct": round(entry.get("recovery_pct", 0), 1),
             "transform": entry.get("transform", "")}
            for entry in recovery
        ]
        _chart(bar_data, "bar", x="config", y="recovery_pct", hue="transform",
               label=label, title="Transform Effect: Accuracy Recovery % by Config")


def multi_config_block_chart(study_report: "StudyReport", layer: str,
                             *, role: str = "weight", top_k: int = 20,
                             label: str = ""):
    """Multi-config block error comparison for a single layer.

    Requires: StudyReport with PerBlockQSNRObserver data across configs.
    """
    config_data: Dict[str, Dict[int, float]] = {}

    for r in _iter_study_results(study_report):
        name = r.name or ""
        blocks = _get_per_block_qsnr(r.observers_data, layer, role)
        if blocks:
            config_data[name] = blocks

    if not config_data:
        return

    all_blocks: set = set()
    for bd in config_data.values():
        all_blocks.update(bd.keys())

    avg_qsnr: Dict[int, float] = {}
    for blk in all_blocks:
        vals = [bd[blk] for bd in config_data.values() if blk in bd]
        if vals:
            avg_qsnr[blk] = sum(vals) / len(vals)

    worst = sorted(avg_qsnr.items(), key=lambda x: x[1])[:top_k]
    if not worst:
        return

    bar_data = []
    for idx, _ in worst:
        for cfg_name, bd in sorted(config_data.items()):
            qsnr = bd.get(idx)
            if qsnr is not None:
                bar_data.append({
                    "block_idx": idx, "qsnr_db": round(qsnr, 1),
                    "config": cfg_name,
                })

    if bar_data:
        _chart(bar_data, "bar", x="block_idx", y="qsnr_db", hue="config",
               label=label, title=f"Multi-Config Block Error: {layer} ({role})")


# =====================================================================
# Convenience: run all single-config charts
# =====================================================================

def all_harness_charts(result: "SessionResult", *, label: str = "",
                       output_dir: Optional[str] = None):
    """Emit U2a + U6 harness charts for a SessionResult."""
    intervention_chart(result, label=label)
    block_qsnr_box_chart(result, label=label)

    if output_dir:
        _save_matplotlib_figures(result, output_dir)


def _save_matplotlib_figures(result: "SessionResult", output_dir: str):
    """Save matplotlib figures using existing viz functions."""
    import os
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    from src.viz.block_error_heatmap import block_error_heatmap, channel_error_bar
    import matplotlib.pyplot as plt

    accum = result.accum_qsnr_per_layer
    if not accum:
        return

    sorted_layers = sorted(accum.items(), key=lambda x: x[1])
    for layer, _ in sorted_layers[:3]:
        safe_name = layer.replace(".", "_")
        for role in ("weight", "input"):
            try:
                fig = block_error_heatmap(result, layer, role=role)
                fig.savefig(f"{output_dir}/figures/block_{safe_name}_{role}.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

        try:
            fig = channel_error_bar(result, layer, role="input")
            fig.savefig(f"{output_dir}/figures/channel_{safe_name}_input.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass
