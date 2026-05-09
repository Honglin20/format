"""Parameterised figure generation functions extracted from the format study.

Each function wraps a matplotlib Figure and saves it via
:func:`save_figure`, then returns the Figure for further customisation.

All functions accept keyword-only parameters (``title``, ``colors``,
``output_dir``) instead of hardcoded values, making them reusable across
different experiments.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.viz._helpers import _compute_best_transform_per_layer
from src.viz.theme import FALLBACK_CYCLE


def save_figure(fig, output_dir: str, name: str) -> str:
    """Save matplotlib Figure as PNG and PDF.

    Args:
        fig: matplotlib Figure.
        output_dir: Output root directory. Figures saved to ``<output_dir>/figures/``.
        name: Base filename without extension.

    Returns:
        Path to the saved PNG file.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return os.path.join(fig_dir, f"{name}.png")

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    Aligns all configs by shared layer names (union of all layers across
    configs) instead of plotting each config independently by sorted index.

    Args:
        results: Dict mapping series name to dict with ``qsnr_per_layer``.
        title: Chart title.
        colors: Dict mapping series name to colour hex string.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect shared x-axis: union of all layer names
    all_layer_names: list = []
    for name, data in results.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        for lname in data["qsnr_per_layer"]:
            if lname not in all_layer_names:
                all_layer_names.append(lname)

    if not all_layer_names:
        plt.close(fig)
        raise ValueError(
            "No QSNR data available in any config. "
            "Ensure QSNRObserver is active during the analysis pass."
        )

    x_positions = range(len(all_layer_names))
    for name, data in results.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        values = [data["qsnr_per_layer"].get(l, float("nan")) for l in all_layer_names]
        color = colors.get(name, FALLBACK_CYCLE[0])
        ax.plot(x_positions, values, marker="o", label=name, linewidth=2, color=color)

    # X-axis labels: short layer names
    short_names = [l.replace("module.", "").replace("Quantized", "") for l in all_layer_names]
    # Truncate to 20 chars for readability
    short_names = [n[:20] for n in short_names]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Layer")
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
    if not data_to_plot:
        plt.close(fig)
        raise ValueError(
            "No MSE data available in any config. "
            "Ensure MSEObserver is active during the analysis pass."
        )
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
    if n_groups == 0:
        raise ValueError(
            "No PoT scaling data available."
        )
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
    ``quant_hist``, ``err_hist``) and renders the most quantization-sensitive
    layers as overlaid semi-transparent bar charts. Sensitivity is determined
    by QSNR (lower = more quantization-sensitive), with a fallback to
    activation magnitude when no QSNR data is available.

    Args:
        all_results: Nested dict of ``{part: {config: {"report": ...}}}``.
            Reports are expected to have an ``iter_slices`` method
            yielding ``(layer, role, stage, slice_key, metrics)`` tuples.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    layer_hists: Dict[str, dict] = {}
    layer_error: Dict[str, float] = {}  # QSNR for sensitivity ranking

    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            if not hasattr(report, "iter_slices"):
                continue
            for layer, role, stage, slice_key, metrics in report.iter_slices():
                if layer not in layer_hists and "fp32_hist" in metrics and "quant_hist" in metrics:
                    layer_hists[layer] = {
                        k: _to_numpy(metrics.get(k))
                        for k in ("fp32_hist", "quant_hist", "err_hist")
                    }
                if layer not in layer_error and "qsnr_db" in metrics:
                    layer_error[layer] = metrics["qsnr_db"]

    if not layer_hists:
        raise ValueError(
            "Histogram data not available. "
            "Add HistogramObserver to observers in the analysis pass."
        )

    # Rank by sensitivity: lowest QSNR first (most quantization-sensitive)
    if layer_error:
        top_layers = sorted(
            layer_hists.items(),
            key=lambda x: layer_error.get(x[0], float("inf")),
        )[:5]
    else:
        print("  Warning: No QSNR data for sensitivity ranking, "
              "falling back to histogram magnitude")
        top_layers = sorted(
            layer_hists.items(),
            key=lambda x: x[1].get("fp32_hist", np.array(0)).sum(),
            reverse=True,
        )[:5]
    if not top_layers:
        raise ValueError(
            "No histogram data found in any layer. "
            "Ensure HistogramObserver is active during the analysis pass."
        )

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
        raise ValueError(
            "No transform study data available. "
            "Ensure the analysis pass includes formats and transforms."
        )

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
        raise ValueError(
            "No format study data available for transform pie chart."
        )

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
        raise ValueError(
            "No transform delta data available. "
            "Ensure the analysis pass includes format and transform variants."
        )

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
            num_layers = len(all_layers)
            if num_layers <= 10:
                # Show all layer names
                for i, layer in enumerate(all_layers):
                    short = layer.split(".")[-1] if "." in layer else layer
                    ax.text(bar_positions[i], deltas[i], short[:12],
                            ha="center", va="bottom" if deltas[i] >= 0 else "top",
                            fontsize=6, rotation=90)
            elif num_layers <= 30:
                # Show every 3rd layer
                for i, layer in enumerate(all_layers):
                    if i % 3 == 0:
                        short = layer.split(".")[-1] if "." in layer else layer
                        ax.text(bar_positions[i], deltas[i], short[:12],
                                ha="center", va="bottom" if deltas[i] >= 0 else "top",
                                fontsize=5, rotation=90)
            else:
                # Show top-5 by absolute delta
                top_indices = sorted(range(len(deltas)),
                                    key=lambda i: abs(deltas[i]), reverse=True)[:5]
                for i in top_indices:
                    short = all_layers[i].split(".")[-1] if "." in all_layers[i] else all_layers[i]
                    ax.text(bar_positions[i], deltas[i], short[:12],
                            ha="center", va="bottom" if deltas[i] >= 0 else "top",
                            fontsize=6, rotation=90, fontweight="bold")

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
            Reports are expected to have an ``iter_slices`` method
            yielding ``(layer, role, stage, slice_key, metrics)`` tuples.
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
            if not hasattr(report, "iter_slices"):
                continue
            for layer, role, stage, slice_key, metrics in report.iter_slices():
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
        raise ValueError(
            "Distribution data not available. "
            "Ensure DistributionObserver and MSEObserver are active "
            "during the analysis pass."
        )

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
        raise ValueError(
            "Layer type data not available. "
            "Ensure QSNRObserver and MSEObserver are active "
            "during the analysis pass."
        )

    # Single layer type degrades to isolated boxplots — fall back to per-layer chart
    if len(ltype_qsnr) == 1:
        single_lt = list(ltype_qsnr.keys())[0]
        print(f"  layer_type_qsnr: only '{single_lt}' layers found, "
              f"falling back to per-layer QSNR chart")

        qsnr_results: dict = {}
        for part_name, part_data in all_results.items():
            if not part_name.startswith("part_") or not isinstance(part_data, dict):
                continue
            for config_name, config_data in part_data.items():
                if not isinstance(config_data, dict) or "report" not in config_data:
                    continue
                report = config_data["report"]
                ls = LayerSensitivity(report)
                per_layer: Dict[str, list] = {}
                for s in ls._samples:
                    per_layer.setdefault(s["layer"], []).append(s.get("qsnr_db", 0))
                qsnr_results[config_name] = {
                    "qsnr_per_layer": {
                        l: sum(v) / max(len(v), 1) for l, v in per_layer.items()
                    }
                }
        return qsnr_line_chart(
            qsnr_results,
            title="Per-Layer QSNR (single layer-type model)",
            colors={},
            output_dir=output_dir,
        )

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
# Figure 12 — Block size sweep line chart
# ---------------------------------------------------------------------------

def block_sweep_line_chart(
    block_sweep: dict,
    *,
    output_dir: str,
) -> plt.Figure:
    """Block size vs per-layer average QSNR line chart.

    One line per block-size configuration, showing how each layer's QSNR
    changes with block size. Useful for understanding the sensitivity of
    different layers to block granularity.

    Args:
        block_sweep: Dict mapping config name (e.g. ``"int8-blk32"``) to
            result dict with ``qsnr_per_layer``.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect shared layers and config data
    entries = []
    for name, data in block_sweep.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        entries.append((name, data["qsnr_per_layer"]))

    if not entries:
        plt.close(fig)
        raise ValueError(
            "No block sweep data available."
        )

    # Compute per-layer avg QSNR for each block size
    sizes, avg_qsnr = [], []
    for name, qsnr_dict in entries:
        try:
            bs = int(name.split("blk")[-1])
        except (ValueError, IndexError):
            bs = 0
        avg = sum(qsnr_dict.values()) / max(len(qsnr_dict), 1)
        sizes.append(bs)
        avg_qsnr.append(avg)

    # Sort by block size
    sorted_pairs = sorted(zip(sizes, avg_qsnr, [e[0] for e in entries]))
    sizes = [p[0] for p in sorted_pairs]
    avg_qsnr = [p[1] for p in sorted_pairs]

    ax.plot(sizes, avg_qsnr, marker="o", linewidth=2, color=FALLBACK_CYCLE[0])
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Average QSNR (dB)")
    ax.set_title("Block Size vs Average QSNR")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)

    save_figure(fig, output_dir, "block_sweep_line")
    return fig


# ---------------------------------------------------------------------------
# Figure 13 — Hierarchical Pre-Scale delta bar chart
# ---------------------------------------------------------------------------

def hierarchical_delta_bar(
    hierarchical: dict,
    *,
    output_dir: str,
    colors: dict | None = None,
) -> plt.Figure:
    """Pre-scale (hierarchical) vs baseline per-layer QSNR delta.

    Shows the benefit of two-level quantization (global PoT pre-scale +
    MX per-block) relative to plain MX quantization at the same bit-width.

    Args:
        hierarchical: Dict mapping config name (e.g. ``"MXINT-8-HIER"``)
            to result dict with ``qsnr_per_layer``. A ``"FP32 (baseline)"``
            entry is skipped.
        output_dir: Output root directory.
        colors: Optional colour mapping for bars.

    Returns:
        matplotlib Figure.
    """
    color_cycle = colors if colors else {}
    entries = [
        (name, data)
        for name, data in hierarchical.items()
        if "baseline" not in name.lower() and "qsnr_per_layer" in data
    ]

    if not entries:
        raise ValueError(
            "No hierarchical study data available."
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Show avg QSNR per variant as grouped bars
    x_positions = range(len(entries))
    values = [
        sum(data["qsnr_per_layer"].values()) / max(len(data["qsnr_per_layer"]), 1)
        for _, data in entries
    ]
    names = [name for name, _ in entries]
    bar_colors = [
        color_cycle.get(name, FALLBACK_CYCLE[i % len(FALLBACK_CYCLE)])
        for i, name in enumerate(names)
    ]

    bars = ax.bar(x_positions, values, color=bar_colors, alpha=0.7, edgecolor="white")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average QSNR (dB)")
    ax.set_title("Hierarchical Pre-Scale — Average Per-Layer QSNR")

    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(fig, output_dir, "hierarchical_delta")
    return fig


# ---------------------------------------------------------------------------
# Figure 14 — Outlier analysis (per-layer bar + outlier vs QSNR scatter)
# ---------------------------------------------------------------------------

def outlier_analysis(
    all_results: dict,
    *,
    output_dir: str,
    role: str = "input",
) -> plt.Figure:
    """Outlier ratio per-layer bar chart + outlier vs QSNR scatter.

    Args:
        all_results: Nested dict ``{part: {config: {"report": ...}}}``.
            Reports must have ``iter_slices`` yielding metrics with
            ``outlier_ratio`` (from DistributionObserver) and optionally
            ``qsnr_db`` (from QSNRObserver).
        output_dir: Output root directory.
        role: Tensor role to plot (default ``"input"``).

    Returns:
        matplotlib Figure.
    """
    data_points = []
    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            if not hasattr(report, "iter_slices"):
                continue
            for layer, r, stage, slice_key, metrics in report.iter_slices():
                if r != role or "outlier_ratio" not in metrics:
                    continue
                data_points.append({
                    "config": config_name,
                    "layer": layer,
                    "outlier_ratio": metrics["outlier_ratio"],
                    "qsnr_db": metrics.get("qsnr_db", float("nan")),
                })

    if not data_points:
        raise ValueError(
            f"Outlier ratio data not available for role {role!r}. "
            "Ensure DistributionObserver is active during the analysis pass."
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    layers = sorted(set(d["layer"] for d in data_points))
    configs = sorted(set(d["config"] for d in data_points))
    x = np.arange(len(layers))
    width = 0.8 / max(len(configs), 1)

    # Panel 1: per-layer outlier ratio bar chart
    for i, cfg in enumerate(configs):
        per_layer = {}
        for d in data_points:
            if d["config"] == cfg:
                per_layer.setdefault(d["layer"], []).append(d["outlier_ratio"])
        values = [sum(per_layer.get(l, [0])) / max(len(per_layer.get(l, [0])), 1)
                  for l in layers]
        color = FALLBACK_CYCLE[i % len(FALLBACK_CYCLE)]
        ax1.bar(x + i * width, values, width, label=cfg, color=color, alpha=0.7)

    ax1.set_xticks(x + width * (len(configs) - 1) / 2)
    short_names = [l.replace("module.", "").replace("Quantized", "")[:20]
                   for l in layers]
    ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Outlier Ratio")
    ax1.set_title(f"Outlier Ratio per Layer — {role}")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: outlier_ratio vs QSNR scatter
    has_qsnr = any(not math.isnan(d["qsnr_db"]) for d in data_points)
    for i, cfg in enumerate(configs):
        cfg_pts = [d for d in data_points if d["config"] == cfg]
        xs = [d["outlier_ratio"] for d in cfg_pts]
        ys = [d["qsnr_db"] for d in cfg_pts] if has_qsnr else [0] * len(cfg_pts)
        color = FALLBACK_CYCLE[i % len(FALLBACK_CYCLE)]
        ax2.scatter(xs, ys, label=cfg, color=color, alpha=0.7, s=40)

    ax2.set_xlabel("Outlier Ratio")
    ax2.set_ylabel("QSNR (dB)" if has_qsnr else "(no QSNR)")
    ax2.set_title(f"Outlier Ratio vs QSNR — {role}")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Outlier Analysis", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, f"outlier_{role}")
    return fig


# ---------------------------------------------------------------------------
# Figure 15 — Per-block QSNR distribution
# ---------------------------------------------------------------------------

def per_block_qsnr(
    all_results: dict,
    *,
    output_dir: str,
    role: str = "input",
) -> plt.Figure:
    """Per-block QSNR statistics: std dev boxplot + min-vs-mean scatter.

    Uses ``qsnr_db_std``, ``qsnr_db_min``, ``qsnr_db_max`` from
    QSNRObserver per-block mode.

    Args:
        all_results: Nested dict ``{part: {config: {"report": ...}}}``.
        output_dir: Output root directory.
        role: Tensor role to plot (default ``"input"``).

    Returns:
        matplotlib Figure.
    """
    layer_data = defaultdict(lambda: defaultdict(list))
    has_min = False
    has_std = False

    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            if not hasattr(report, "iter_slices"):
                continue
            for layer, r, stage, slice_key, metrics in report.iter_slices():
                if r != role:
                    continue
                if "qsnr_db_std" in metrics:
                    layer_data[layer]["qsnr_db_std"].append(metrics["qsnr_db_std"])
                    has_std = True
                if "qsnr_db_min" in metrics:
                    layer_data[layer]["qsnr_db_min"].append(metrics["qsnr_db_min"])
                    layer_data[layer]["qsnr_db"].append(metrics.get("qsnr_db", float("nan")))
                    has_min = True

    if not has_std and not has_min:
        raise ValueError(
            f"Per-block QSNR statistics not available for role {role!r}. "
            "Ensure QSNRObserver is active with per-block granularity."
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: qsnr_db_std per layer boxplot
    if has_std:
        layers = sorted(layer_data.keys())
        std_data = []
        std_labels = []
        for l in layers:
            vals = layer_data[l].get("qsnr_db_std", [])
            if vals:
                std_data.append(vals)
                short = l.replace("module.", "").replace("Quantized", "")[:20]
                std_labels.append(short)

        if std_data:
            ax1.boxplot(std_data, tick_labels=std_labels, patch_artist=True)
            ax1.set_xticklabels(std_labels, rotation=45, ha="right", fontsize=7)

    ax1.set_ylabel("QSNR Std Dev (dB)")
    ax1.set_title(f"Per-Block QSNR Std Dev — {role}")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: qsnr_db_min vs qsnr_db scatter
    if has_min:
        for l in sorted(layer_data.keys()):
            means = layer_data[l].get("qsnr_db", [])
            mins = layer_data[l].get("qsnr_db_min", [])
            if means and mins:
                avg_mean = sum(means) / max(len(means), 1)
                avg_min = sum(mins) / max(len(mins), 1)
                ax2.scatter(avg_mean, avg_min, s=40, alpha=0.7,
                           label=l[:30] if len(layer_data) <= 10 else "")

        if means and mins:
            all_vals = means + mins
            lo, hi = min(all_vals), max(all_vals)
            ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.5)

    ax2.set_xlabel("Mean QSNR (dB)")
    ax2.set_ylabel("Min QSNR (dB)")
    ax2.set_title(f"Min vs Mean QSNR — {role}")
    if len(layer_data) <= 10:
        ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Per-Block QSNR Distribution", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, f"per_block_qsnr_{role}")
    return fig


# ---------------------------------------------------------------------------
# Figure 16 — Distribution features correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(
    all_results: dict,
    *,
    output_dir: str,
) -> plt.Figure:
    """Pearson correlation heatmap of distribution features vs QSNR/MSE.

    Args:
        all_results: Nested dict ``{part: {config: {"report": ...}}}``.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    feat_cols = [
        "crest_factor", "skewness", "kurtosis", "excess_kurtosis",
        "bimodality_coefficient", "sparse_ratio", "dynamic_range_bits",
        "outlier_ratio", "norm_entropy",
    ]
    all_keys = set()
    rows = []

    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            if not hasattr(report, "iter_slices"):
                continue
            for layer, role, stage, slice_key, metrics in report.iter_slices():
                row = {}
                for c in feat_cols + ["qsnr_db", "mse"]:
                    if c in metrics:
                        row[c] = metrics[c]
                        all_keys.add(c)
                if row:
                    rows.append(row)

    available = [c for c in feat_cols + ["qsnr_db", "mse"] if c in all_keys]
    if len(available) < 2:
        raise ValueError(
            "Insufficient distribution feature data for correlation heatmap. "
            "Ensure DistributionObserver is active during the analysis pass."
        )

    # Build matrix
    data = {c: [] for c in available}
    for row in rows:
        for c in available:
            data[c].append(row.get(c, float("nan")))

    arr = np.array([data[c] for c in available])  # [features, samples]
    # Compute pairwise Pearson correlation (skip NaN rows)
    n_feat = len(available)
    corr = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(n_feat):
            mask = ~(np.isnan(arr[i]) | np.isnan(arr[j]))
            if mask.sum() >= 3:
                corr[i, j] = np.corrcoef(arr[i][mask], arr[j][mask])[0, 1]
            else:
                corr[i, j] = float("nan")

    fig, ax = plt.subplots(figsize=(max(10, n_feat * 1.1), max(8, n_feat * 0.9)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(available, fontsize=8)

    for i in range(n_feat):
        for j in range(n_feat):
            v = corr[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=7, color="white" if abs(v) > 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title("Distribution Features x QSNR/MSE Correlation", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "correlation_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Figure 17 — Per-role distribution comparison
# ---------------------------------------------------------------------------

def role_distribution_comparison(
    all_results: dict,
    *,
    output_dir: str,
    roles: tuple = ("input", "weight", "output"),
) -> plt.Figure:
    """Per-role distribution feature comparison boxplots.

    Compares skewness, kurtosis, and normalized entropy across roles.

    Args:
        all_results: Nested dict ``{part: {config: {"report": ...}}}``.
        output_dir: Output root directory.
        roles: Roles to compare (default ``("input", "weight", "output")``).

    Returns:
        matplotlib Figure.
    """
    role_data: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for part_name, part_data in all_results.items():
        if not part_name.startswith("part_") or not isinstance(part_data, dict):
            continue
        for config_name, config_data in part_data.items():
            if not isinstance(config_data, dict) or "report" not in config_data:
                continue
            report = config_data["report"]
            if not hasattr(report, "iter_slices"):
                continue
            for layer, r, stage, slice_key, metrics in report.iter_slices():
                for feat in ("skewness", "kurtosis", "norm_entropy"):
                    if feat in metrics:
                        role_data[r][feat].append(metrics[feat])

    plot_roles = [r for r in roles if r in role_data and role_data[r]]
    if not plot_roles:
        if role_data:
            raise ValueError(
                f"No data found for roles {list(roles)}. "
                f"Roles present: {sorted(role_data.keys())}."
            )
        raise ValueError(
            "Distribution data not available. "
            "Ensure DistributionObserver is active during the analysis pass."
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_cycle = FALLBACK_CYCLE

    for ax, feature, ylabel in [
        (axes[0], "skewness", "Skewness"),
        (axes[1], "kurtosis", "Kurtosis"),
        (axes[2], "norm_entropy", "Normalized Entropy"),
    ]:
        data_groups = []
        labels = []
        for i, r in enumerate(plot_roles):
            vals = role_data[r].get(feature, [])
            if vals:
                data_groups.append(vals)
                labels.append(r)

        if data_groups:
            bp = ax.boxplot(data_groups, tick_labels=labels, patch_artist=True)
            for patch, label in zip(bp["boxes"], labels):
                idx = plot_roles.index(label) if label in plot_roles else 0
                patch.set_facecolor(colors_cycle[idx % len(colors_cycle)])
                patch.set_alpha(0.6)

        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Role")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Distribution Feature Comparison Across Roles", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "role_distribution")
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
