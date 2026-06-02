"""Shared plotting utilities for sparse research visualizations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "mxint": "#2196F3",            # Blue — baseline
    "sparse_per_tensor": "#FF9800",  # Orange
    "sparse_per_channel": "#4CAF50",  # Green
    "sparse_bank": "#E91E63",       # Pink
    "sparse_per_block": "#9C27B0",  # Purple
}

GRANULARITY_LABELS = {
    "PER_TENSOR": "Per-Tensor",
    "PER_CHANNEL": "Per-Channel",
    "PER_BLOCK": "Per-Block (MXINT)",
    "BANK": "Bank(16)",
}

DISTRIBUTION_LABELS = {
    "normal": "Normal",
    "lognormal": "Lognormal(σ=1)",
    "powerlaw": "PowerLaw(α=2.5)",
    "real_weight": "Real Weight",
    "real_activation": "Real Activation",
}

# ---------------------------------------------------------------------------
# Figure sizes
# ---------------------------------------------------------------------------

FIG_SINGLE = (8, 5)
FIG_DOUBLE = (12, 5)
FIG_SQUARE = (8, 8)

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})


def load_results(experiment_name: str, results_dir: str = None) -> dict:
    """Load a results JSON file.

    Args:
        experiment_name: e.g. "l1_baseline"
        results_dir: Path to results directory (default: research/sparse/results)

    Returns:
        Parsed JSON data.
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"
    path = Path(results_dir) / f"{experiment_name}.json"
    with open(path) as f:
        return json.load(f)


def ensure_figures_dir():
    """Return Path to figures directory, creating it if needed."""
    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(fig, name: str):
    """Save figure to figures/ directory with consistent settings."""
    fig_dir = ensure_figures_dir()
    path = fig_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def style_legend(ax, **kwargs):
    """Apply consistent legend styling."""
    ax.legend(framealpha=0.9, edgecolor="gray", fontsize=10, **kwargs)


def annotate_bar(ax, bars, values, fmt=".1f", offset=0.3):
    """Add value labels above bars."""
    for bar, val in zip(bars, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
