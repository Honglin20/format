"""Parameterised figure generation functions extracted from the format study.

Each function wraps a matplotlib Figure and saves it via
:func:`save_figure`, then returns the Figure for further customisation.

All functions accept keyword-only parameters (``title``, ``colors``,
``output_dir``) instead of hardcoded values, making them reusable across
different experiments.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.viz.save import save_figure
from src.viz.theme import FALLBACK_CYCLE

import torch

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            key=lambda tx, l=layer: variant_qsnr[tx].get(l, -float("inf")),
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


# ---------------------------------------------------------------------------
# Figure 1 & 2 — Per-layer QSNR line chart  (merged)
# ---------------------------------------------------------------------------

def qsnr_line_chart(
    results: dict,
    *,
    title: str,
    colors: dict,
    output_dir: str,
) -> plt.Figure:
    """Per-layer QSNR line chart.

    Args:
        results: Dict mapping series name to dict with ``qsnr_per_layer``.
        title: Chart title.
        colors: Dict mapping series name to colour hex string.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in results.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        layers = sorted(data["qsnr_per_layer"].keys())
        values = [data["qsnr_per_layer"][l] for l in layers]
        color = colors.get(name, FALLBACK_CYCLE[0])
        ax.plot(
            range(len(layers)), values,
            marker="o", label=name, linewidth=2, color=color,
        )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_dir, title.lower().replace(" ", "_"))
    return fig


# ---------------------------------------------------------------------------
# Figure 3 & 4 — Per-layer MSE box plot  (merged)
# ---------------------------------------------------------------------------

def mse_box_plot(
    results: dict,
    *,
    title: str,
    colors: dict,
    output_dir: str,
) -> plt.Figure:
    """Per-layer MSE box plot.

    Args:
        results: Dict mapping series name to dict with ``mse_per_layer``.
        title: Chart title.
        colors: Dict mapping series name to colour hex string.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot, labels, plot_colors = [], [], []
    for name, data in results.items():
        if "baseline" in name.lower() or "mse_per_layer" not in data:
            continue
        mse_vals = list(data["mse_per_layer"].values())
        if mse_vals:
            data_to_plot.append(mse_vals)
            labels.append(name)
            plot_colors.append(colors.get(name, FALLBACK_CYCLE[0]))
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_dir, title.lower().replace(" ", "_"))
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — PoT scaling delta bar chart
# ---------------------------------------------------------------------------

def pot_delta_bar(
    part_c: dict,
    *,
    output_dir: str,
) -> plt.Figure:
    """FP32 vs PoT per-layer QSNR delta bar chart.

    Groups entries by format base (e.g. ``INT8-PC``) and shows two bars
    side-by-side: FP32-scale and PoT-scale.

    Args:
        part_c: Dict of ``{name: data}``.  Names containing ``"PoT"`` are
            treated as PoT-scaled variants of the base format (the part
            before the last ``-``).
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    formats: Dict[str, dict] = {}
    for name, data in part_c.items():
        if "baseline" in name.lower():
            continue
        base = name.rsplit("-", 1)[0]
        is_pot = "PoT" in name
        formats.setdefault(base, {})[is_pot] = data

    n_groups = len(formats)
    fig, axes = plt.subplots(1, max(n_groups, 1), figsize=(7 * max(n_groups, 1), 5),
                             squeeze=False)
    for idx, (fmt_name, fmt_data) in enumerate(sorted(formats.items())):
        ax = axes[0, idx]
        fp32_qsnr = fmt_data.get(False, {}).get("qsnr_per_layer", {})
        pot_qsnr = fmt_data.get(True, {}).get("qsnr_per_layer", {})

        all_layers = sorted(set(list(fp32_qsnr.keys()) + list(pot_qsnr.keys())))
        deltas = [pot_qsnr.get(l, 0) - fp32_qsnr.get(l, 0) for l in all_layers]
        layer_names = [l.replace("module.", "").replace("Quantized", "")
                       for l in all_layers]

        bar_colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=bar_colors, alpha=0.7)
        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("QSNR Delta (PoT – FP32) [dB]")
        ax.set_title(f"{fmt_name}")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("PoT Scaling vs FP32 Scaling — Per-Layer QSNR Delta",
                 fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "pot_delta")
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Histogram overlay
# ---------------------------------------------------------------------------

def histogram_overlay(
    all_results: dict,
    *,
    output_dir: str,
) -> plt.Figure:
    """Three-channel histogram overlay (fp32 / quant / error).

    Extracts histogram data from ``HistogramObserver`` (keys: ``fp32_hist``,
    ``quant_hist``, ``err_hist``) and renders the most sensitive layers as
    overlaid semi-transparent bar charts.

    Args:
        all_results: Nested dict of ``{part: {config: {"report": ...}}}``.
            Reports are expected to have a ``_raw`` attribute (private
            Report API; if Report's internal format changes this function
            must be updated) containing histogram metrics.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
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
                                    k: _to_numpy(metrics.get(k))
                                    for k in ("fp32_hist", "quant_hist", "err_hist")
                                }
                                break
                        if layer in layer_hists:
                            break
                    if layer in layer_hists:
                        break

    if not layer_hists:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Histogram data not available\n"
                "(Add HistogramObserver to observers in run_experiment)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Activation Histograms (No Data)")
        save_figure(fig, output_dir, "histogram_overlay")
        return fig

    # Pick top 3-5 layers with the richest histogram data
    top_layers = sorted(
        layer_hists.items(),
        key=lambda x: x[1].get("fp32_hist", np.array(0)).sum(),
        reverse=True,
    )[:5]
    if not top_layers:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No histogram data found",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        save_figure(fig, output_dir, "histogram_overlay")
        return fig

    n = len(top_layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (layer, hist_data) in zip(axes[0], top_layers):
        for channel, color, label in [
            ("fp32_hist", "#3498db", "fp32"),
            ("quant_hist", "#e74c3c", "quant"),
            ("err_hist", "#95a5a6", "error"),
        ]:
            counts = hist_data.get(channel)
            if counts is None or not isinstance(counts, np.ndarray):
                continue
            bin_centers = np.arange(len(counts))
            ax.fill_between(bin_centers, counts, alpha=0.35, color=color,
                            label=label, step="mid")
            ax.plot(bin_centers, counts, color=color, linewidth=0.8)
        ax.set_title(layer, fontsize=9)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Activation Histograms (fp32 / quant / error) — "
                 "Most Sensitive Layers", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "histogram_overlay")
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — Transform heatmap
# ---------------------------------------------------------------------------

def transform_heatmap(
    part_d: dict,
    *,
    colors: dict | None = None,
    output_dir: str,
) -> plt.Figure:
    """Format x Transform accuracy heatmap.

    Args:
        part_d: Nested dict ``{format: {transform: data}}`` where data
            contains ``accuracy.accuracy``.
        colors: Optional colour mapping (reserved for future extension;
            currently the heatmap uses a ``RdYlGn`` colormap).
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    _ = colors  # reserved for future extension; heatmap uses RdYlGn colormap
    fmt_names = sorted(part_d.keys())
    tx_variants = sorted({tx for fmt_data in part_d.values()
                          for tx in fmt_data})

    if not fmt_names or not tx_variants:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Format x Transform Accuracy Matrix")
        save_figure(fig, output_dir, "transform_heatmap")
        return fig

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
        vmin, vmax = float(valid.min()), float(valid.max())
    else:
        vmin, vmax = 0.0, 1.0
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#d3d3d3")
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
    ax.set_title("Format x Transform Accuracy Matrix")
    fig.tight_layout()
    save_figure(fig, output_dir, "transform_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Figure 8 — Transform pie chart
# ---------------------------------------------------------------------------

def transform_pie(
    part_d: dict,
    *,
    colors: dict,
    output_dir: str,
) -> plt.Figure:
    """Per-layer optimal transform distribution pie chart.

    Args:
        part_d: Nested dict ``{format: {transform: {"qsnr_per_layer": ...}}}``.
            A ``"PerLayerOpt"`` key triggers the pie-chart rendering;
            its value is not used (presence is the signal).
        colors: Dict mapping transform name to colour hex string.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    n_fmts = len(part_d)
    if n_fmts == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Per-Layer Optimal Transform Distribution")
        save_figure(fig, output_dir, "transform_pie")
        return fig

    fig, axes = plt.subplots(
        1, n_fmts,
        figsize=(5 * n_fmts, 5),
        subplot_kw={"aspect": "equal"},
        squeeze=False,
    )

    pie_colors = colors  # preferred; falls back to FALLBACK_CYCLE below

    for ax, (fmt_name, fmt_data) in zip(axes[0], sorted(part_d.items())):
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
        pie_colors_list = [pie_colors.get(l, "#95a5a6") for l in labels]
        _, _, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            colors=pie_colors_list, startangle=90,
            textprops={"fontsize": 9},
        )
        total = sum(sizes)
        ax.set_title(f"{fmt_name} (n={total})", fontsize=10)

    fig.suptitle("Per-Layer Optimal Transform Distribution", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "transform_pie")
    return fig


# ---------------------------------------------------------------------------
# Figure 9 — Transform delta bar chart
# ---------------------------------------------------------------------------

def transform_delta(
    part_d: dict,
    *,
    colors: dict,
    output_dir: str,
) -> plt.Figure:
    """Transform delta QSNR vs baseline, one subplot per format.

    Each format gets its own subplot so that formats with different layer
    counts produce non-overlapping bars.

    Args:
        part_d: Nested dict ``{format: {transform: {"qsnr_per_layer": ...}}}``.
            A key ``"None"`` is used as the baseline.
        colors: Dict mapping transform name to colour hex string.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    fmt_names = sorted(part_d.keys())
    n_fmts = len(fmt_names)
    if n_fmts == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Transform Impact on Per-Layer QSNR")
        save_figure(fig, output_dir, "transform_delta")
        return fig

    fig, axes = plt.subplots(n_fmts, 1, figsize=(14, 4 * n_fmts), sharex=False,
                             squeeze=False)
    colors_tx = colors

    for ax, fmt_name in zip(axes[:, 0], fmt_names):
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

    fig.suptitle("Transform Impact on Per-Layer QSNR", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "transform_delta")
    return fig


# ---------------------------------------------------------------------------
# Figure 10 — Error vs distribution scatter
# ---------------------------------------------------------------------------

def error_vs_distribution(
    all_results: dict,
    *,
    output_dir: str,
) -> plt.Figure:
    """QSNR vs distribution features scatter (4-panel).

    Args:
        all_results: Nested dict ``{part: {config: {"report": ...}}}``.
            Reports are expected to have a ``_raw`` attribute (private
            Report API; if Report's internal format changes this function
            must be updated) with per-slice metrics (``qsnr_db``,
            ``dynamic_range_bits``, etc.).
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
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
        ax.set_title("Quantization Error vs Distribution Features")
        save_figure(fig, output_dir, "error_vs_distribution")
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    dr_vals = [d["dynamic_range"] for d in data_points]
    qsnr_vals = [d["qsnr"] for d in data_points]
    sparse_vals = [d["sparse_ratio"] for d in data_points]
    skew_vals = [d["skewness"] for d in data_points]
    kurt_vals = [d["kurtosis"] for d in data_points]

    # Panel 1: QSNR vs Dynamic Range (color = sparse_ratio)
    ax = axes[0, 0]
    sc = ax.scatter(dr_vals, qsnr_vals, c=sparse_vals,
                    cmap="viridis", alpha=0.6, s=30)
    ax.set_xlabel("Dynamic Range (bits)")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("QSNR vs Dynamic Range\n(color = sparse ratio)")
    fig.colorbar(sc, ax=ax)
    ax.grid(True, alpha=0.3)

    # Panel 2: QSNR vs Skewness (color = kurtosis)
    ax = axes[0, 1]
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

    fig.suptitle("Quantization Error vs Distribution Features", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir, "error_vs_distribution")
    return fig


# ---------------------------------------------------------------------------
# Figure 11 — Layer-type grouped QSNR
# ---------------------------------------------------------------------------

def layer_type_qsnr(
    all_results: dict,
    *,
    output_dir: str,
) -> plt.Figure:
    """Layer-type grouped QSNR comparison.

    Note:
        This figure degrades for models with sparse layer-type diversity
        (e.g. MLP-only architectures) because the ``by_layer_type`` grouping
        collapses to a single category (``"Linear"``).

    Args:
        all_results: Nested dict ``{part: {config: {"report": ...}}}``.
            Reports must be compatible with ``LayerSensitivity``.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    # Avoid hard dependency on analysis module by importing locally
    from src.analysis.correlation import LayerSensitivity

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
        ax.set_title("Layer-Type Grouped Quantization Error")
        save_figure(fig, output_dir, "layer_type_qsnr")
        return fig

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

    fig.suptitle("Layer-Type Grouped Quantization Error", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir, "layer_type_qsnr")
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(value):
    """Convert torch.Tensor to numpy array; pass through numpy arrays."""
    if isinstance(value, torch.Tensor):
        return value.cpu().float().numpy()
    if isinstance(value, np.ndarray):
        return value
    if value is None:
        return None
    return np.asarray(value)
