#!/usr/bin/env python3
"""Generate QSNR sweep plots for granularity × sparse analysis.

Two figures:
  1. Outlier amplitude sweep: Gaussian(0,1) + outlier at 1×~50× std, 0.5% ratio
  2. Variance sweep: Gaussian(0,σ²) σ=1~10, with fixed ±50 outliers

Usage:
    PYTHONPATH=. python scripts/qsnr_sweep_plots.py

Output:
    docs/guides/visualizations/qsnr-sweep-outlier-amplitude.png
    docs/guides/visualizations/qsnr-sweep-variance.png
"""

import math
import os

import torch

from src.formats import get_format
from src.quantize import quantize
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.transform import IdentityTransform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "guides", "visualizations")

INT4 = get_format("int4")
INT8 = get_format("int8")

TENSOR_SIZE = 4096
SEED = 42

# Outlier amplitude sweep parameters
OUTLIER_AMPLITUDES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
OUTLIER_FRAC = 0.005  # 0.5% of elements are outliers

# Variance sweep parameters
VARIANCES = list(range(1, 11))
FIXED_OUTLIER_VAL = 50.0

# Sparse ratios to plot
ELEM_SPARSE_RATIOS = [0.02, 0.05, 0.10]
GROUP_SPARSE_RATIOS = [0.20, 0.50]

# Granularities: (display_name, gran_for_2d_tensor)
GRANULARITIES = [
    ("per_tensor", GranularitySpec.per_tensor()),
    ("per_channel", GranularitySpec.per_channel(axis=0)),
    ("per_block (size=32)", GranularitySpec.per_block(size=32, axis=-1)),
    ("bank (size=16)", GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_qsnr(original, quantized):
    signal = (original ** 2).mean().item()
    noise = ((original - quantized) ** 2).mean().item()
    if noise < 1e-12:
        return 60.0  # cap
    return 10.0 * math.log10(signal / noise)


def make_tensor_with_outliers(base_std, outlier_val, outlier_frac, seed=SEED):
    """Create a 2D tensor with Gaussian base + fixed-amplitude outliers."""
    n = TENSOR_SIZE * TENSOR_SIZE
    torch.manual_seed(seed)
    x = torch.randn(TENSOR_SIZE, TENSOR_SIZE) * base_std
    n_outliers = max(1, int(n * outlier_frac))
    # Spread outliers uniformly across the tensor
    outlier_indices = torch.randperm(n)[:n_outliers]
    flat = x.flatten()
    # Alternate sign for outliers
    signs = torch.randint(0, 2, (n_outliers,)).float() * 2 - 1
    flat[outlier_indices] = signs * outlier_val
    return flat.reshape(TENSOR_SIZE, TENSOR_SIZE)


def quantize_and_qsnr(tensor, scheme):
    x_q = quantize(tensor, scheme=scheme)
    return compute_qsnr(tensor, x_q)


def make_gran_with_ratio(gran, outlier_ratio=0.0):
    """Create a new GranularitySpec with the same params but different outlier_ratio."""
    return GranularitySpec(
        mode=gran.mode,
        block_size=gran.block_size,
        channel_axis=gran.channel_axis,
        block_axis=gran.block_axis,
        bank_size=gran.bank_size,
        bank_axis=gran.bank_axis,
        outlier_ratio=outlier_ratio,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep(x_values, x_label, tensor_fn, filename, title):
    """Plot QSNR sweep for all granularities × sparse modes.

    Args:
        x_values: list of x-axis values
        x_label: x-axis label
        tensor_fn: callable(x_val) -> 2D tensor
        filename: output filename
        title: figure title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Color scheme
    elem_colors = ["#d62728", "#ff7f0e", "#e377c2"]  # red shades
    group_colors = ["#2ca02c", "#17becf"]  # green shades
    base_color = "#1f1f1f"

    for ax_idx, (gran_name, gran) in enumerate(GRANULARITIES):
        ax = axes[ax_idx]
        ax.set_title(gran_name, fontsize=13, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=11)
        if ax_idx % 2 == 0:
            ax.set_ylabel("QSNR (dB)", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(0, 55)

        # Base: no sparse
        base_qsnrs = []
        for xv in x_values:
            tensor = tensor_fn(xv)
            scheme = QuantScheme(format=INT4, granularity=gran, transform=IdentityTransform())
            q = quantize_and_qsnr(tensor, scheme)
            base_qsnrs.append(q)
        ax.plot(x_values, base_qsnrs, color=base_color, linewidth=2.5, linestyle="--",
                marker="o", markersize=4, label="base (int4)", zorder=5)

        # Element sparse
        for i, ratio in enumerate(ELEM_SPARSE_RATIOS):
            qsnrs = []
            for xv in x_values:
                tensor = tensor_fn(xv)
                gran_r = make_gran_with_ratio(gran, outlier_ratio=ratio)
                scheme = QuantScheme(format=INT4, granularity=gran_r, transform=IdentityTransform(),
                                     outlier_format=INT8)
                q = quantize_and_qsnr(tensor, scheme)
                qsnrs.append(q)
            color = elem_colors[i % len(elem_colors)]
            ax.plot(x_values, qsnrs, color=color, linewidth=1.8,
                    marker="s", markersize=3, label=f"elem sparse (r={ratio})", zorder=4)

        # Group sparse
        for i, ratio in enumerate(GROUP_SPARSE_RATIOS):
            qsnrs = []
            for xv in x_values:
                tensor = tensor_fn(xv)
                scheme = QuantScheme(format=INT4, granularity=gran, transform=IdentityTransform(),
                                     group_format=INT8, group_ratio=ratio)
                q = quantize_and_qsnr(tensor, scheme)
                qsnrs.append(q)
            color = group_colors[i % len(group_colors)]
            ax.plot(x_values, qsnrs, color=color, linewidth=1.8, linestyle="-.",
                    marker="^", markersize=3, label=f"group sparse (r={ratio})", zorder=4)

        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(OUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath} ({os.path.getsize(filepath) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== Figure 1: Outlier amplitude sweep =====
    print("Figure 1: Outlier amplitude sweep...")
    plot_sweep(
        x_values=OUTLIER_AMPLITUDES,
        x_label="Outlier amplitude (× base std)",
        tensor_fn=lambda amp: make_tensor_with_outliers(
            base_std=1.0,
            outlier_val=float(amp),
            outlier_frac=OUTLIER_FRAC,
        ),
        filename="qsnr-sweep-outlier-amplitude.png",
        title="QSNR vs Outlier Amplitude  (4096×4096, Gaussian(0,1) + 0.5% outlier, int4 base)",
    )

    # ===== Figure 2: Variance sweep with fixed outlier =====
    print("Figure 2: Variance sweep with fixed outlier...")
    plot_sweep(
        x_values=VARIANCES,
        x_label="Base distribution std (σ)",
        tensor_fn=lambda sigma: make_tensor_with_outliers(
            base_std=float(sigma),
            outlier_val=FIXED_OUTLIER_VAL,
            outlier_frac=OUTLIER_FRAC,
        ),
        filename="qsnr-sweep-variance.png",
        title=f"QSNR vs Base Variance  (4096×4096, Gaussian(0,σ²) + ±{FIXED_OUTLIER_VAL:.0f} outlier at 0.5%, int4 base)",
    )

    print("Done.")


if __name__ == "__main__":
    main()
