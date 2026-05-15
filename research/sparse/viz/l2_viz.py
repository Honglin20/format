"""L2 visualization: Sparse ratio sweep — QSNR vs ratio vs b_eff.

Reads results/l2_ratio_sweep.json → produces figures/l2_*.png

Run: PYTHONPATH=. python research/sparse/viz/l2_viz.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from research.sparse.viz.common import (
    COLORS, GRANULARITY_LABELS, DISTRIBUTION_LABELS,
    FIG_DOUBLE, FIG_SQUARE, FIG_SINGLE,
    load_results, save_figure, style_legend,
)


def _color_for_granularity(granularity: str) -> str:
    key = f"sparse_{granularity.lower()}"
    return COLORS.get(key, "#9E9E9E")


# ---------------------------------------------------------------------------
# Figure 1: QSNR vs outlier_ratio (line + error band) — faceted by distribution
# ---------------------------------------------------------------------------

def fig_qsnr_vs_ratio(data: list):
    distributions = sorted(set(r["distribution"] for r in data if r["status"] == "ok"))
    granularities = sorted(set(r["granularity"] for r in data if r["status"] == "ok"))

    fig, axes = plt.subplots(1, len(distributions), figsize=(len(distributions) * 5, 4.5),
                             sharey=True)
    if len(distributions) == 1:
        axes = [axes]

    for ax, dist in zip(axes, distributions):
        for gran in granularities:
            points = [(r["outlier_ratio"], r["qsnr_mean"], r["qsnr_std"])
                      for r in data
                      if r["granularity"] == gran and r["distribution"] == dist
                      and r["status"] == "ok" and r["qsnr_mean"] is not None]
            if not points:
                continue
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            errs = [p[2] for p in points]
            color = _color_for_granularity(gran)
            label = GRANULARITY_LABELS.get(gran.upper(), gran)
            ax.plot(xs, ys, "o-", color=color, label=label, markersize=5)
            ax.fill_between(xs, [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            alpha=0.15, color=color)

        ax.set_xlabel("Outlier Ratio")
        ax.set_title(DISTRIBUTION_LABELS.get(dist, dist))
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("QSNR (dB)")
        style_legend(ax)

    fig.suptitle("L2: QSNR vs Outlier Ratio", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "l2_qsnr_vs_ratio.png")


# ---------------------------------------------------------------------------
# Figure 2: Effective bitwidth vs outlier_ratio
# ---------------------------------------------------------------------------

def fig_bitwidth_vs_ratio(data: list):
    granularities = sorted(set(r["granularity"] for r in data if r["status"] == "ok"))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for gran in granularities:
        points = [(r["outlier_ratio"], r["b_eff"])
                  for r in data
                  if r["granularity"] == gran and r["status"] == "ok"
                  and r["b_eff"] is not None]
        points = sorted(set(points), key=lambda p: p[0])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = _color_for_granularity(gran)
        label = GRANULARITY_LABELS.get(gran.upper(), gran)
        ax.plot(xs, ys, "s-", color=color, label=label, markersize=6)

    ax.set_xlabel("Outlier Ratio")
    ax.set_ylabel("Effective Bitwidth (b_eff)")
    ax.set_title("L2: Effective Bitwidth vs Outlier Ratio")
    ax.grid(alpha=0.3)
    style_legend(ax)
    fig.tight_layout()
    save_figure(fig, "l2_bitwidth_vs_ratio.png")


# ---------------------------------------------------------------------------
# Figure 3: QSNR vs b_eff (Pareto frontier) — scatter/line
# ---------------------------------------------------------------------------

def fig_qsnr_vs_bitwidth(data: list):
    granularities = sorted(set(r["granularity"] for r in data if r["status"] == "ok"))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for gran in granularities:
        points = [(r["b_eff"], r["qsnr_mean"])
                  for r in data
                  if r["granularity"] == gran and r["status"] == "ok"
                  and r["qsnr_mean"] is not None and r["b_eff"] is not None]
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = _color_for_granularity(gran)
        label = GRANULARITY_LABELS.get(gran.upper(), gran)
        ax.plot(xs, ys, "o-", color=color, label=label, markersize=6)

    ax.set_xlabel("Effective Bitwidth (b_eff)")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("L2: QSNR vs Effective Bitwidth (Pareto)")
    ax.grid(alpha=0.3)
    style_legend(ax)
    fig.tight_layout()
    save_figure(fig, "l2_qsnr_vs_bitwidth.png")


# ---------------------------------------------------------------------------
# Figure 4: Per-distribution facet — QSNR vs ratio, colored by granularity
# ---------------------------------------------------------------------------

def fig_per_distribution(data: list):
    distributions = sorted(set(r["distribution"] for r in data if r["status"] == "ok"))
    granularities = sorted(set(r["granularity"] for r in data if r["status"] == "ok"))
    n_cols = min(3, len(distributions))
    n_rows = (len(distributions) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        for gran in granularities:
            points = [(r["outlier_ratio"], r["qsnr_mean"])
                      for r in data
                      if r["granularity"] == gran and r["distribution"] == dist
                      and r["status"] == "ok" and r["qsnr_mean"] is not None]
            if not points:
                continue
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            color = _color_for_granularity(gran)
            label = GRANULARITY_LABELS.get(gran.upper(), gran)
            ax.plot(xs, ys, "o-", color=color, label=label, markersize=5)
        ax.set_xlabel("Outlier Ratio")
        ax.set_ylabel("QSNR (dB)")
        ax.set_title(DISTRIBUTION_LABELS.get(dist, dist))
        ax.grid(alpha=0.3)
        style_legend(ax)

    for idx in range(len(distributions), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("L2: QSNR vs Outlier Ratio by Distribution", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "l2_per_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_results("l2_ratio_sweep")
    print("Generating L2 figures...")
    fig_qsnr_vs_ratio(data)
    fig_bitwidth_vs_ratio(data)
    fig_qsnr_vs_bitwidth(data)
    fig_per_distribution(data)
    print("Done.")


if __name__ == "__main__":
    main()
