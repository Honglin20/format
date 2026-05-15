"""L3 visualization: Bank granularity sweet-spot analysis.

Reads results/l3_bank_sweetspot.json → produces figures/l3_*.png

Run: PYTHONPATH=. python research/sparse/viz/l3_viz.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from research.sparse.viz.common import (
    COLORS, DISTRIBUTION_LABELS,
    FIG_DOUBLE, FIG_SQUARE, FIG_SINGLE,
    load_results, save_figure, style_legend,
)


# ---------------------------------------------------------------------------
# Figure 1: 2D heatmap — bank_size × tensor_dim → QSNR
# ---------------------------------------------------------------------------

def _make_heatmap(data: list, value_key: str, title: str, filename: str,
                  cmap: str = "viridis"):
    distributions = sorted(set(r["distribution"] for r in data if r["status"] == "ok"))
    bank_sizes = sorted(set(r["bank_size"] for r in data))
    tensor_dims = sorted(set(r["tensor_dim"] for r in data))

    n_dist = len(distributions)
    fig, axes = plt.subplots(1, n_dist, figsize=(n_dist * 5.5, 4.5), sharey=True)
    if n_dist == 1:
        axes = [axes]

    for ax, dist in zip(axes, distributions):
        matrix = np.zeros((len(tensor_dims), len(bank_sizes)))
        matrix[:] = np.nan

        for i, td in enumerate(tensor_dims):
            for j, bs in enumerate(bank_sizes):
                for r in data:
                    if (r["bank_size"] == bs and r["tensor_dim"] == td
                            and r["distribution"] == dist
                            and r["status"] == "ok"
                            and r.get(value_key) is not None):
                        matrix[i, j] = r[value_key]
                        break

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(bank_sizes)))
        ax.set_xticklabels([str(bs) for bs in bank_sizes])
        ax.set_yticks(range(len(tensor_dims)))
        ax.set_yticklabels([str(td) for td in tensor_dims])
        ax.set_xlabel("Bank Size")
        if ax == axes[0]:
            ax.set_ylabel("Tensor Dim")
        ax.set_title(DISTRIBUTION_LABELS.get(dist, dist))
        fig.colorbar(im, ax=ax, label=value_key.replace("_", " ").title())

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Figure 2: Line plot — bank_size × tensor_dim, QSNR vs bank_size
# ---------------------------------------------------------------------------

def fig_bank_lineplot(data: list):
    distributions = sorted(set(r["distribution"] for r in data if r["status"] == "ok"))
    tensor_dims = sorted(set(r["tensor_dim"] for r in data))

    fig, axes = plt.subplots(1, len(distributions), figsize=(len(distributions) * 5, 4.5),
                             sharey=True)
    if len(distributions) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tensor_dims)))

    for ax, dist in zip(axes, distributions):
        for idx, td in enumerate(tensor_dims):
            points = [(r["bank_size"], r["qsnr_mean"])
                      for r in data
                      if r["tensor_dim"] == td and r["distribution"] == dist
                      and r["status"] == "ok" and r["qsnr_mean"] is not None]
            if not points:
                continue
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, "o-", color=colors[idx], label=f"dim={td}", markersize=5)

        ax.set_xlabel("Bank Size")
        ax.set_title(DISTRIBUTION_LABELS.get(dist, dist))
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("QSNR (dB)")
        style_legend(ax)

    fig.suptitle("L3: Bank Sweet Spot — QSNR vs Bank Size by Tensor Dimension", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "l3_bank_lineplot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_results("l3_bank_sweetspot")
    print("Generating L3 figures...")
    _make_heatmap(data, "qsnr_mean",
                  "L3: QSNR by Bank Size and Tensor Dimension",
                  "l3_bank_heatmap.png", cmap="viridis")
    _make_heatmap(data, "qsnr_per_b_eff",
                  "L3: QSNR per Effective Bit (Q/b_eff)",
                  "l3_bank_efficiency.png", cmap="plasma")
    fig_bank_lineplot(data)
    print("Done.")


if __name__ == "__main__":
    main()
