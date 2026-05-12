"""Per-role QSNR visualisation: grouped bars and depth decay curves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

_ROLES = ("input", "weight", "output")
_ROLE_COLORS = {"input": "#1f77b4", "weight": "#ff7f0e", "output": "#2ca02c"}


def _shorten(name: str, max_len: int = 22) -> str:
    name = name.replace("module.", "").replace("Quantized", "")
    if len(name) > max_len:
        return name[:max_len - 2] + ".."
    return name


# ---------------------------------------------------------------------------
# plot_per_role_qsnr_bars
# ---------------------------------------------------------------------------

def plot_per_role_qsnr_bars(
    result,
    max_layers: int = 30,
    sort_by: str = "worst",
) -> plt.Figure:
    """Grouped bar chart: input / weight / output QSNR per layer.

    Args:
        result: SessionResult with ``qsnr_by_role`` populated.
        max_layers: Maximum number of layers to display.
        sort_by: ``"worst"`` — sort by the lowest QSNR across all roles.
                 ``"depth"`` — keep model order (from accum_qsnr keys).

    Returns:
        matplotlib Figure.
    """
    qsnr_by_role = result.qsnr_by_role
    if not qsnr_by_role:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No per-role QSNR data available.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Union of all layers
    all_layers = list(dict.fromkeys(
        n for role_map in qsnr_by_role.values() for n in role_map
    ))

    if sort_by == "worst":
        # Sort by the lowest QSNR across all roles
        def _worst(layer):
            return min(
                qsnr_by_role.get(r, {}).get(layer, float("inf"))
                for r in _ROLES
            )
        all_layers.sort(key=_worst)
    # else: keep model order (from dict insertion)

    layers = all_layers[:max_layers]

    x = np.arange(len(layers))
    n_roles = len(_ROLES)
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.3), 5))

    for j, role in enumerate(_ROLES):
        role_map = qsnr_by_role.get(role, {})
        vals = [role_map.get(n, np.nan) for n in layers]
        offset = (j - (n_roles - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=role,
                      color=_ROLE_COLORS[role], alpha=0.85)

        # Mark missing values
        for i, v in enumerate(vals):
            if v != v:
                ax.text(x[i] + offset, 0.5, "N/A", ha="center",
                        fontsize=5, rotation=90, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([_shorten(n) for n in layers], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("QSNR (dB)")
    ax.set_title("Per-Layer QSNR by Role")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_depth_decay
# ---------------------------------------------------------------------------

def plot_depth_decay(
    result,
    role: str = "output",
) -> plt.Figure:
    """QSNR vs depth line plot for a single role.

    Uses the order of layers as they appear in the QSNR dict (which follows
    ``named_modules()`` order = depth).

    Args:
        result: SessionResult.
        role: Which role to plot (``"input"`` / ``"weight"`` / ``"output"``).

    Returns:
        matplotlib Figure.
    """
    role_map = result.qsnr_by_role.get(role, {})
    if not role_map:
        # Fallback to accum for output
        if role == "output" and result.accum_qsnr_per_layer:
            role_map = result.accum_qsnr_per_layer
        else:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"No QSNR data for role='{role}'.",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

    layers = list(role_map.keys())
    vals = [role_map[n] for n in layers]

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.2), 4))
    x = np.arange(len(layers))

    color = _ROLE_COLORS.get(role, "#333333")
    ax.plot(x, vals, "o-", color=color, linewidth=1.2, markersize=4, label=role)
    ax.fill_between(x, vals, alpha=0.1, color=color)

    # Highlight layers below threshold
    threshold = min(vals) + (max(vals) - min(vals)) * 0.3 if vals else 20
    for i, v in enumerate(vals):
        if v < threshold:
            ax.annotate(_shorten(layers[i]), (i, v),
                        fontsize=5, rotation=45, alpha=0.7,
                        xytext=(0, -12), textcoords="offset points",
                        ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(len(layers))], fontsize=7)
    ax.set_xlabel("Layer depth index")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title(f"QSNR vs Depth — {role} role")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
