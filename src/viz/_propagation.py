"""Error propagation visualisation: DAG, waterfall, and accum-vs-local scatter.

All functions accept a ``SessionResult`` and return a matplotlib Figure.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def plot_propagation_dag(result) -> plt.Figure:
    """Horizontal bar-style "DAG": each layer is a bar showing QSNR.

    The figure mimics a propagation diagram by showing layers in model order
    (left to right or top to bottom) with colour coding for local QSNR
    (green → yellow → red) and a second overlay for accumulated QSNR.

    When accumulated QSNR data is present, a second series of markers is
    plotted to show the drop across layers.
    """
    accum = result.accum_qsnr_per_layer
    local, _ = result.qsnr_per_role(role="output")

    if not local:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No QSNR data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Error Propagation DAG")
        return fig

    layers = list(local.keys())
    local_vals = [local[n] for n in layers]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.35), 5))

    y = range(len(layers))

    # Local QSNR bars
    colours = _qsnr_colormap(local_vals)
    ax.barh(y, local_vals, color=colours, alpha=0.8, label="Local QSNR")

    # Accum QSNR markers (if available)
    if accum:
        accum_vals = [accum.get(n, float("nan")) for n in layers]
        valid = [(i, v) for i, v in enumerate(accum_vals) if v == v]
        if valid:
            ax.scatter([v for _, v in valid], [i for i, _ in valid],
                       marker="D", color="black", s=30, zorder=5,
                       label="Accum QSNR")

    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(n) for n in layers], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("QSNR (dB)")
    ax.set_title("Error Propagation — Local QSNR per Layer")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # Colour bar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn", norm=plt.Normalize(vmin=min(local_vals), vmax=max(local_vals))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Local QSNR (dB)", fontsize=8)

    fig.tight_layout()
    return fig


def plot_error_waterfall(result) -> plt.Figure:
    """Waterfall chart: accumulated QSNR drops layer by layer.

    Each bar shows the accumulated QSNR after that layer, and the drop
    from the previous layer is annotated.  The first layer starts from
    "FP32" (infinite QSNR, capped at a high value for display).
    """
    accum = result.accum_qsnr_per_layer
    if not accum:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No accumulated QSNR data.\nRun with keep_fp32=True.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Error Waterfall")
        return fig

    layers = list(accum.keys())
    accum_vals = [accum[n] for n in layers]

    # Cap display at 60 dB (close enough to FP32 "infinite")
    display_cap = 60.0
    capped = [min(v, display_cap) for v in accum_vals]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.3), 5))

    x = np.arange(len(layers))
    prev = display_cap
    colours = []
    drops = []
    for v in capped:
        drop = prev - v
        drops.append(drop)
        colours.append(_headroom_color(drop))
        prev = v

    ax.bar(x, capped, color=colours, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Annotate drops over a threshold
    for i, (layer, v, drop) in enumerate(zip(layers, capped, drops)):
        if drop > 2:
            ax.text(i, v + 0.5, f"-{drop:.1f}", ha="center", fontsize=6, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels([_shorten(n) for n in layers], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accum QSNR (dB)")
    ax.set_title("Error Waterfall — Accumulated QSNR per Layer")
    ax.set_ylim(0, display_cap + 5)
    ax.axhline(y=display_cap, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(len(layers) - 1, display_cap, "FP32 ≈ ∞", fontsize=7, color="green",
            va="bottom", ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_local_vs_accum_scatter(result) -> plt.Figure:
    """Scatter plot: local QSNR vs accumulated QSNR per layer.

    Points above the diagonal have headroom (error is propagated from
    upstream).  Points on or below the diagonal are bottlenecks (local
    error dominates).
    """
    accum = result.accum_qsnr_per_layer
    local, _ = result.qsnr_per_role(role="output")

    if not accum or not local:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need both local and accumulated QSNR data.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Match layers
    common = set(accum) & set(local)
    if not common:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No matching layers between local and accumulated QSNR.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    accum_vals = [accum[n] for n in common]
    local_vals = [local[n] for n in common]

    fig, ax = plt.subplots(figsize=(8, 8))

    all_vals = accum_vals + local_vals
    lo, hi = min(all_vals) - 2, max(all_vals) + 2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1, label="y=x (no headroom)")

    headroom = [l - a for l, a in zip(local_vals, accum_vals)]
    colours = [_headroom_color(h) for h in headroom]

    sc = ax.scatter(local_vals, accum_vals, c=headroom, cmap="RdYlGn",
                    edgecolors="black", linewidth=0.3, s=50, zorder=5)

    # Label extreme points
    for i, name in enumerate(common):
        hr = headroom[i]
        if hr < 1 or hr > 15:
            ax.annotate(_shorten(name), (local_vals[i], accum_vals[i]),
                        fontsize=6, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Local QSNR (dB)")
    ax.set_ylabel("Accumulated QSNR (dB)")
    ax.set_title("Local vs Accumulated QSNR")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Headroom (dB)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _shorten(name: str, max_len: int = 25) -> str:
    """Shorten a module name for display."""
    name = name.replace("module.", "").replace("Quantized", "")
    if len(name) > max_len:
        return name[:max_len - 2] + ".."
    return name


def _qsnr_colormap(values: list) -> list:
    """Map QSNR values to RdYlGn colours."""
    lo, hi = min(values), max(values)
    if hi - lo < 1:
        hi = lo + 1
    norm = plt.Normalize(lo, hi)
    cmap = plt.cm.RdYlGn
    return [cmap(norm(v)) for v in values]


def _headroom_color(headroom: float) -> str:
    """Return a colour for a given headroom value (local - accum QSNR)."""
    if headroom < 3:
        return "#d62728"      # red: source
    elif headroom < 10:
        return "#ff7f0e"      # orange: mixed
    else:
        return "#2ca02c"      # green: propagated
