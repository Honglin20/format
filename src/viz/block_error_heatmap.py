"""Block-level error visualization: heatmaps, bar charts, and cross-config comparisons.

All functions operate on SessionResult or StudyReport and produce matplotlib Figures.
"""
from __future__ import annotations

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_per_unit_qsnr(result: SessionResult, layer: str, role: str):
    """Extract {(unit_type, idx): qsnr_db} from SessionResult observers_data."""
    obs = result.observers_data
    if not obs:
        return {}
    layer_data = obs.get(layer, {})
    role_data = layer_data.get(role, {})
    out = {}
    for _stage, slices in role_data.items():
        for key, metrics in slices.items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            tag, idx = key[0], key[1]
            if tag in ("block", "channel", "bank"):
                qsnr = metrics.get("qsnr_db")
                if qsnr is not None and np.isfinite(qsnr):
                    out[(tag, int(idx))] = qsnr
    return out


# ---------------------------------------------------------------------------
# 1. block_error_heatmap — 2D grid: blocks × channels
# ---------------------------------------------------------------------------

def block_error_heatmap(
    result: SessionResult,
    layer: str,
    role: str = "weight",
    *,
    top_k_blocks: int = 0,
    figsize: tuple = (12, 6),
    cmap: str = "RdYlGn",
    title: str | None = None,
) -> plt.Figure:
    """Render a 2D heatmap of per-block QSNR for a weight/activation tensor.

    For weight tensors with shape [out_features, in_features]:
      - X-axis: block index (along in_features, block_size groups)
      - Y-axis: output channel (out_features)
      - Cell color: QSNR (dB) — red = worse, green = better

    For activations:
      - X-axis: channel index
      - Y-axis: block index
      - Cell color: QSNR (dB)

    When only per-block aggregate data is available (no per-channel breakdown),
    falls back to a 1D bar chart of per-block QSNR.

    Args:
        result: SessionResult with PerBlockQSNRObserver data.
        layer: Module name.
        role: "weight" or "input".
        top_k_blocks: If > 0, annotate the top-k worst blocks.
        figsize: Figure size.
        cmap: Matplotlib colormap name.
        title: Override title.

    Returns:
        matplotlib Figure.
    """
    obs = result.observers_data
    if not obs:
        return _empty_fig(layer, role, "No observer data")

    layer_data = obs.get(layer, {})
    role_data = layer_data.get(role, {})
    if not role_data:
        return _empty_fig(layer, role, f"No data for role '{role}'")

    # Collect all per-unit measurements
    unit_data = {}  # (tag, idx) → qsnr_db
    for _stage, slices in role_data.items():
        for key, metrics in slices.items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            tag, idx = key[0], int(key[1])
            if tag in ("block", "channel", "bank"):
                qsnr = metrics.get("qsnr_db")
                if qsnr is not None and np.isfinite(qsnr):
                    unit_data[(tag, idx)] = qsnr

    if not unit_data:
        return _empty_fig(layer, role, "No per-unit QSNR data")

    tags = {k[0] for k in unit_data}

    # Try to build a 2D grid if we have both block and channel info
    if "block" in tags and "channel" in tags:
        return _heatmap_2d(result, layer, role, obs, top_k_blocks, figsize, cmap, title)
    elif "block" in tags:
        return _heatmap_1d_blocks(unit_data, layer, role, top_k_blocks, figsize, cmap, title)
    elif "channel" in tags:
        return _heatmap_1d_channels(unit_data, layer, role, figsize, cmap, title)
    else:
        return _heatmap_1d_generic(unit_data, layer, role, figsize, cmap, title)


def _heatmap_2d(result, layer, role, obs, top_k, figsize, cmap, title_override):
    """Build a 2D heatmap using the raw weight/activation tensor shape + block indexing.

    For weight [out_ch, in_ch] with block_size bs:
      - reshape weight view to [out_ch, n_blocks, bs]
      - per-block QSNR gives [out_ch * n_blocks] values
      - rearrange to 2D grid [out_ch, n_blocks]
    """
    # Get tensor shape from the first slice's stage
    role_data = obs[layer][role]
    # Determine n_blocks and n_channels from observer keys
    block_indices = [k[1] for k in _collect_keys(role_data, "block")]
    channel_indices = [k[1] for k in _collect_keys(role_data, "channel")]

    if not block_indices or not channel_indices:
        per_unit = _extract_per_unit_qsnr(result, layer, role)
        block_data = {k: v for k, v in per_unit.items() if k[0] == "block"}
        return _heatmap_1d_blocks(block_data, layer, role, top_k, figsize, cmap, title_override)

    n_blocks_total = max(block_indices) + 1
    n_channels = max(channel_indices) + 1

    # For weight: try to figure out grid from quantized module
    # blocks are along in_features, channels are out_features
    # So grid = [n_channels, n_blocks_per_channel]
    n_blocks_per_ch = n_blocks_total // n_channels if n_channels > 0 else n_blocks_total

    if n_blocks_per_ch < 1:
        n_blocks_per_ch = 1

    # Build 2D array from block data
    per_unit = _extract_per_unit_qsnr(result, layer, role)
    grid = np.full((n_channels, n_blocks_per_ch), np.nan)
    for (tag, idx), qsnr in per_unit.items():
        if tag == "block":
            ch = idx // n_blocks_per_ch
            blk = idx % n_blocks_per_ch
            if 0 <= ch < n_channels and 0 <= blk < n_blocks_per_ch:
                grid[ch, blk] = qsnr

    return _render_heatmap(grid, layer, role, "Block Index", "Channel",
                           top_k, figsize, cmap, title_override)


def _heatmap_1d_blocks(unit_data, layer, role, top_k, figsize, cmap, title_override):
    """1D bar chart of per-block QSNR sorted by index."""
    block_qsnrs = {k[1]: v for k, v in unit_data.items() if k[0] == "block"}
    if not block_qsnrs:
        return _empty_fig(layer, role, "No block data")

    indices = sorted(block_qsnrs.keys())
    values = [block_qsnrs[i] for i in indices]

    fig, ax = plt.subplots(figsize=figsize)
    colors = _qsnr_colors(values, cmap)
    bars = ax.bar(indices, values, color=colors, width=1.0, edgecolor="none")

    # Annotate worst blocks
    if top_k > 0:
        worst = sorted(block_qsnrs.items(), key=lambda x: x[1])[:top_k]
        for idx, qsnr in worst:
            ax.annotate(f"{qsnr:.1f}", (idx, qsnr), fontsize=7,
                        ha="center", va="bottom", color="red", fontweight="bold")

    title = title_override or f"Per-Block QSNR: {layer} ({role})"
    ax.set_title(title)
    ax.set_xlabel("Block Index")
    ax.set_ylabel("QSNR (dB)")
    ax.axhline(y=np.nanmean(values), color="gray", linestyle="--", linewidth=0.8,
               label=f"Mean={np.nanmean(values):.1f} dB")
    ax.legend(fontsize=8)

    _add_colorbar(fig, ax, values, cmap)
    fig.tight_layout()
    return fig


def _heatmap_1d_channels(unit_data, layer, role, figsize, cmap, title_override):
    """1D bar chart of per-channel QSNR."""
    ch_qsnrs = {k[1]: v for k, v in unit_data.items() if k[0] == "channel"}
    if not ch_qsnrs:
        return _empty_fig(layer, role, "No channel data")

    indices = sorted(ch_qsnrs.keys())
    values = [ch_qsnrs[i] for i in indices]

    fig, ax = plt.subplots(figsize=figsize)
    colors = _qsnr_colors(values, cmap)
    ax.bar(indices, values, color=colors, width=1.0, edgecolor="none")

    title = title_override or f"Per-Channel QSNR: {layer} ({role})"
    ax.set_title(title)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("QSNR (dB)")
    ax.axhline(y=np.nanmean(values), color="gray", linestyle="--", linewidth=0.8,
               label=f"Mean={np.nanmean(values):.1f} dB")
    ax.legend(fontsize=8)

    _add_colorbar(fig, ax, values, cmap)
    fig.tight_layout()
    return fig


def _heatmap_1d_generic(unit_data, layer, role, figsize, cmap, title_override):
    """Generic 1D bar chart for any unit type."""
    sorted_items = sorted(unit_data.items(), key=lambda x: x[0][1])
    indices = [f"{k[0]}{k[1]}" for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)
    colors = _qsnr_colors(values, cmap)
    ax.bar(range(len(indices)), values, color=colors)

    title = title_override or f"Per-Unit QSNR: {layer} ({role})"
    ax.set_title(title)
    ax.set_ylabel("QSNR (dB)")
    fig.tight_layout()
    return fig


def _render_heatmap(grid, layer, role, xlabel, ylabel, top_k, figsize, cmap_name,
                    title_override):
    """Render a 2D numpy array as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    vmin, vmax = float(valid.min()), float(valid.max())
    if vmin == vmax:
        vmin -= 1
        vmax += 1

    cmap_obj = plt.cm.get_cmap(cmap_name)
    im = ax.imshow(grid, cmap=cmap_obj, aspect="auto", vmin=vmin, vmax=vmax,
                    interpolation="nearest")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = title_override or f"Block Error Heatmap: {layer} ({role})"
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="QSNR (dB)", shrink=0.8)

    # Annotate worst cells
    if top_k > 0:
        flat = [(i, j, grid[i, j]) for i in range(grid.shape[0])
                for j in range(grid.shape[1]) if np.isfinite(grid[i, j])]
        flat.sort(key=lambda x: x[2])
        for i, j, val in flat[:top_k]:
            ax.annotate(f"{val:.0f}", (j, i), fontsize=6, ha="center", va="center",
                        color="white" if val < (vmin + vmax) / 2 else "black",
                        fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. channel_error_bar — per-channel activation QSNR
# ---------------------------------------------------------------------------

def channel_error_bar(
    result: SessionResult,
    layer: str,
    role: str = "input",
    *,
    top_k: int = 20,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Bar chart of per-channel QSNR sorted worst-first.

    Highlights outlier channels in red. Shows which input features
    cause the most quantization error.

    Args:
        result: SessionResult with PerBlockQSNRObserver data.
        layer: Module name.
        role: "input" for activations.
        top_k: Number of channels to display.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    per_unit = _extract_per_unit_qsnr(result, layer, role)
    ch_data = {k[1]: v for k, v in per_unit.items() if k[0] == "channel"}

    if not ch_data:
        # Fallback: use block data as 1D
        block_data = {k[1]: v for k, v in per_unit.items() if k[0] == "block"}
        if not block_data:
            return _empty_fig(layer, role, "No per-unit data")
        ch_data = block_data

    # Sort worst-first, take top_k
    sorted_ch = sorted(ch_data.items(), key=lambda x: x[1])[:top_k]
    if not sorted_ch:
        return _empty_fig(layer, role, "No data")

    indices = [f"ch{idx}" for idx, _ in sorted_ch]
    values = [v for _, v in sorted_ch]

    # Mark outliers: channels > 1 std below mean of ALL channels
    all_values = list(ch_data.values())
    mean_v = np.mean(all_values)
    std_v = np.std(all_values)
    threshold = mean_v - std_v

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#D55E00" if v < threshold else "#0072B2" for v in values]
    bars = ax.bar(range(len(indices)), values, color=colors, edgecolor="none")

    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(indices, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("QSNR (dB)")
    ax.set_title(f"Channel Error: {layer} ({role}) — Top-{top_k} Worst")

    # Reference lines
    ax.axhline(y=mean_v, color="#009E73", linestyle="--", linewidth=0.8,
               label=f"Mean={mean_v:.1f} dB")
    ax.axhline(y=threshold, color="#D55E00", linestyle=":", linewidth=0.8,
               label=f"Outlier threshold={threshold:.1f} dB")
    ax.legend(fontsize=8)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. multi_config_block_comparison — side-by-side across configs
# ---------------------------------------------------------------------------

def multi_config_block_comparison(
    study_report: StudyReport,
    layer: str,
    role: str = "weight",
    *,
    configs: list[str] | None = None,
    top_k: int = 20,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Side-by-side per-block QSNR comparison across configs.

    Shows the same layer's block-level error profile under different
    quantization configs (e.g. W8A8 vs W4A8 vs W4A4).

    Args:
        study_report: StudyReport from a multi-config Study.
        layer: Module name to compare.
        role: "weight" or "input".
        configs: Config names to include (None = all).
        top_k: Number of worst blocks to show per config.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    from src.viz.theme import FALLBACK_CYCLE

    # Collect per-config block data
    config_data: dict[str, dict[int, float]] = {}

    for part_results in study_report._results.values():
        for r in part_results:
            name = r.name or ""
            if configs and name not in configs:
                continue
            per_unit = _extract_per_unit_qsnr(r, layer, role)
            block_qsnr = {k[1]: v for k, v in per_unit.items() if k[0] == "block"}
            if block_qsnr:
                config_data[name] = block_qsnr

    if not config_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No block data for {layer} ({role})",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Find common worst blocks across configs
    all_blocks: set[int] = set()
    for bd in config_data.values():
        all_blocks.update(bd.keys())

    # Rank by average QSNR across configs (worst first)
    avg_qsnr: dict[int, float] = {}
    for blk in all_blocks:
        vals = [bd.get(blk) for bd in config_data.values() if blk in bd]
        if vals:
            avg_qsnr[blk] = np.mean(vals)

    worst_blocks = sorted(avg_qsnr.items(), key=lambda x: x[1])[:top_k]
    if not worst_blocks:
        return _empty_fig(layer, role, "No common blocks")

    block_indices = [idx for idx, _ in worst_blocks]
    cfg_names = sorted(config_data.keys())

    # Grouped bar chart
    n_cfgs = len(cfg_names)
    n_blocks = len(block_indices)
    bar_width = 0.8 / max(n_cfgs, 1)

    fig, ax = plt.subplots(figsize=figsize)

    for ci, cfg_name in enumerate(cfg_names):
        bd = config_data[cfg_name]
        values = [bd.get(idx, np.nan) for idx in block_indices]
        offset = (ci - n_cfgs / 2 + 0.5) * bar_width
        color = FALLBACK_CYCLE[ci % len(FALLBACK_CYCLE)]
        ax.bar([x + offset for x in range(n_blocks)], values,
               width=bar_width, label=cfg_name, color=color, edgecolor="none")

    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels([f"b{idx}" for idx in block_indices],
                        rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("QSNR (dB)")
    ax.set_title(f"Block Error Comparison: {layer} ({role}) — Top-{top_k} Worst Blocks")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _collect_keys(role_data, tag):
    """Extract all (tag, idx) keys matching tag from role_data."""
    keys = []
    for _stage, slices in role_data.items():
        for key in slices:
            if isinstance(key, tuple) and len(key) >= 2 and key[0] == tag:
                keys.append(key)
    return keys


def _qsnr_colors(values, cmap_name):
    """Map QSNR values to colors from a colormap."""
    arr = np.array(values)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return ["gray"] * len(values)
    vmin, vmax = float(valid.min()), float(valid.max())
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    cmap_obj = plt.cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return [cmap_obj(norm(v)) if np.isfinite(v) else (0.8, 0.8, 0.8, 1.0)
            for v in values]


def _add_colorbar(fig, ax, values, cmap_name):
    """Add a compact colorbar to the right of the axes."""
    valid = [v for v in values if np.isfinite(v)]
    if not valid:
        return
    vmin, vmax = min(valid), max(valid)
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    cmap_obj = plt.cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="QSNR (dB)", shrink=0.8)


def _empty_fig(layer, role, reason):
    """Create a figure with a 'no data' message."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, f"No data for {layer} ({role}): {reason}",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_axis_off()
    return fig
