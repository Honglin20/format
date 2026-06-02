"""L1 visualization: Sparse vs MXINT QSNR comparison.

Reads results/l1_baseline.json → produces figures/l1_*.png

Run: PYTHONPATH=. python research/sparse/viz/l1_viz.py
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


def _color_for(granularity: str, sparse: str) -> str:
    if sparse == "dense" and granularity == "per_block":
        return COLORS["mxint"]
    key = f"sparse_{granularity}" if sparse == "sparse" else granularity
    return COLORS.get(key, "#9E9E9E")


def _label_for(granularity: str, sparse: str) -> str:
    gl = GRANULARITY_LABELS.get(granularity.upper(), granularity)
    if sparse == "sparse":
        return f"{gl} (sparse)"
    elif granularity.upper() == "PER_BLOCK":
        return f"{gl}"
    return f"{gl} (dense)"


# ---------------------------------------------------------------------------
# Figure 1: Grouped bar chart — QSNR by granularity, hue=dense/sparse,
#           faceted by distribution
# ---------------------------------------------------------------------------

def fig_qsnr_comparison(data: list):
    distributions = sorted(set(r["distribution"] for r in data))
    n_dist = len(distributions)
    fig, axes = plt.subplots(1, n_dist, figsize=(n_dist * 5, 5), sharey=True)
    if n_dist == 1:
        axes = [axes]

    for ax, dist in zip(axes, distributions):
        dist_data = [r for r in data if r["distribution"] == dist and r["status"] == "ok"]
        if not dist_data:
            ax.set_title(DISTRIBUTION_LABELS.get(dist, dist))
            continue

        # Group by (granularity, sparse)
        groups = defaultdict(list)
        for r in dist_data:
            key = (r["granularity"], r["sparse"])
            groups[key].append(r["qsnr_mean"])

        # Compute per-group mean across shapes
        labels = []
        means = []
        errors = []
        colors = []
        for (gran, sparse), vals in sorted(groups.items()):
            valid = [v for v in vals if v is not None]
            if not valid:
                continue
            labels.append(_label_for(gran, sparse))
            means.append(np.mean(valid))
            errors.append(np.std(valid))
            colors.append(_color_for(gran, sparse))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=errors, color=colors, capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(DISTRIBUTION_LABELS.get(dist, dist))
        ax.set_ylabel("QSNR (dB)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("L1: Sparse vs Dense QSNR by Granularity and Distribution", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "l1_qsnr_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: Heatmap — ΔQSNR (sparse - dense) by granularity × shape,
#           averaged across distributions
# ---------------------------------------------------------------------------

def fig_qsnr_heatmap(data: list):
    granularities = sorted(set(r["granularity"] for r in data if r["status"] == "ok"))
    shapes = sorted(set(str(r["shape"]) for r in data if r["status"] == "ok"))

    # Compute ΔQSNR matrix
    delta = np.zeros((len(shapes), len(granularities)))
    delta[:] = np.nan

    for i, shape in enumerate(shapes):
        for j, gran in enumerate(granularities):
            dense_vals = []
            sparse_vals = []
            for r in data:
                if (r["granularity"] == gran and str(r["shape"]) == shape
                        and r["status"] == "ok" and r["qsnr_mean"] is not None):
                    if r["sparse"] == "dense":
                        dense_vals.append(r["qsnr_mean"])
                    elif r["sparse"] == "sparse":
                        sparse_vals.append(r["qsnr_mean"])
            if dense_vals and sparse_vals:
                delta[i, j] = np.mean(sparse_vals) - np.mean(dense_vals)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    im = ax.imshow(delta, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=15)
    ax.set_xticks(range(len(granularities)))
    ax.set_xticklabels([GRANULARITY_LABELS.get(g.upper(), g) for g in granularities],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(shapes)))
    ax.set_yticklabels(shapes)
    ax.set_title("L1: ΔQSNR (sparse − dense)")
    fig.colorbar(im, ax=ax, label="ΔQSNR (dB)")
    fig.tight_layout()
    save_figure(fig, "l1_qsnr_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_results("l1_baseline")
    print("Generating L1 figures...")
    fig_qsnr_comparison(data)
    fig_qsnr_heatmap(data)
    print("Done.")


if __name__ == "__main__":
    main()
