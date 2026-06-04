"""Shared helpers for chart-producing modules (layer_diagnostic, harness_charts).

Internal module — not part of the public API.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

# ── Optional: harness render_chart ───────────────────────────────────

try:
    from harness.tools.chart import render_chart
except ImportError:
    render_chart = None


def _chart(data, chart_type, *, x, y, label="", title="", hue=None,
           series=None, **kw):
    """Emit a chart via render_chart if harness is available."""
    if render_chart is None:
        return
    # Guard: box chart requires >= 3 rows (harness validation)
    if chart_type == "box" and len(data) < 3:
        return
    render_chart(data, chart_type, x=x, y=y, label=label, title=title,
                 hue=hue, series=series, **kw)


# =====================================================================
# Observer data extraction
# =====================================================================

def _get_per_block_qsnr(obs_data: dict, layer: str, role: str) -> Dict[int, float]:
    """Extract per-block QSNR: {block_idx: qsnr_db}.

    Requires: PerBlockQSNRObserver attached during Session.run().
    """
    layer_data = obs_data.get(layer, {})
    stages = layer_data.get(role, {})
    if not stages:
        return {}
    blocks = {}
    for _stage, slices in stages.items():
        for slice_key, metrics in slices.items():
            if not isinstance(slice_key, tuple) or len(slice_key) < 2:
                continue
            tag = slice_key[0]
            if tag == "block" and "qsnr_db" in metrics:
                idx = int(slice_key[1])
                v = metrics["qsnr_db"]
                if math.isfinite(v):
                    blocks[idx] = v
    return blocks


def _get_per_channel_qsnr(obs_data: dict, layer: str, role: str) -> Dict[int, float]:
    """Extract per-channel QSNR: {channel_idx: qsnr_db}.

    Requires: PerBlockQSNRObserver attached during Session.run().
    """
    layer_data = obs_data.get(layer, {})
    stages = layer_data.get(role, {})
    if not stages:
        return {}
    channels = {}
    for _stage, slices in stages.items():
        for slice_key, metrics in slices.items():
            if not isinstance(slice_key, tuple) or len(slice_key) < 2:
                continue
            tag = slice_key[0]
            if tag == "channel" and "qsnr_db" in metrics:
                idx = int(slice_key[1])
                v = metrics["qsnr_db"]
                if math.isfinite(v):
                    channels[idx] = v
    return channels


def _get_dist_metrics(obs_data: dict, layer: str, role: str) -> Optional[dict]:
    """Extract distribution metrics dict for a (layer, role) pair.

    Requires: DistributionObserver attached during Session.run().
    """
    layer_data = obs_data.get(layer, {})
    stages = layer_data.get(role, {})
    if not stages:
        return None
    for _stage, slices in stages.items():
        for _key, metrics in slices.items():
            if "crest_factor" in metrics:
                return metrics
    return None


def _get_hist_data(obs_data: dict, layer: str, role: str) -> Optional[dict]:
    """Extract histogram data for a (layer, role) pair.

    Requires: HistogramObserver attached during Session.run().
    """
    layer_data = obs_data.get(layer, {})
    stages = layer_data.get(role, {})
    if not stages:
        return None
    for _stage, slices in stages.items():
        for _key, metrics in slices.items():
            if "fp32_hist" in metrics:
                return metrics
    return None


def _get_fit_metrics(obs_data: dict, layer: str, role: str) -> Optional[dict]:
    """Extract distribution fit metrics for a (layer, role) pair.

    Requires: DistributionFitObserver attached during Session.run().
    """
    layer_data = obs_data.get(layer, {})
    stages = layer_data.get(role, {})
    if not stages:
        return None
    for _stage, slices in stages.items():
        for _key, metrics in slices.items():
            if "best_fit" in metrics:
                return metrics
    return None


# =====================================================================
# Statistics
# =====================================================================

def _block_stats(blocks: Dict[int, float]) -> dict:
    """Compute mean/std/min/max/p10/p90 from per-block QSNR dict."""
    if not blocks:
        return {}
    vals = list(blocks.values())
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
    sorted_vals = sorted(vals)
    p10 = sorted_vals[max(0, int(n * 0.1))]
    p90 = sorted_vals[min(n - 1, int(n * 0.9))]
    return {
        "mean": round(mean, 1), "std": round(std, 1),
        "min": round(min(vals), 1), "max": round(max(vals), 1),
        "p10": round(p10, 1), "p90": round(p90, 1),
        "n_blocks": n,
    }


# =====================================================================
# Layer filtering
# =====================================================================

_EXCLUDE_KEYWORDS = (
    "norm", "batch_norm", "layer_norm", "group_norm", "rms_norm",
    "relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "pool",
    "BatchNorm", "LayerNorm", "GroupNorm", "RMSNorm",
)

_DIST_KEYS = [
    ("crest_factor", "crest"),
    ("kurtosis", "kurt"),
    ("excess_kurtosis", "ex_kurt"),
    ("outlier_ratio", "ol_pct"),
    ("sparse_ratio", "sparse"),
    ("dynamic_range_bits", "dr_bits"),
    ("norm_entropy", "entropy"),
    ("skewness", "skew"),
    ("bimodality_coefficient", "bimod"),
]


def _linear_layer_names(observers_data: dict) -> set:
    """Return layer names that are NOT norm/activation/pooling modules."""
    names = set()
    for layer in observers_data:
        if not any(kw in layer for kw in _EXCLUDE_KEYWORDS):
            names.add(layer)
    return names


def _filter_qsnr(qsnr_dict: dict, linear_only: bool, observers_data: dict) -> dict:
    """Filter a qsnr dict to linear-only layers if requested."""
    if not linear_only or not observers_data:
        return qsnr_dict
    allowed = _linear_layer_names(observers_data)
    return {k: v for k, v in qsnr_dict.items() if k in allowed}


# =====================================================================
# Block QSNR Heatmap
# =====================================================================

def _infer_heatmap_shape(
    blocks: Dict[int, float],
    layer_name: str,
    role: str,
    block_size: int,
    model: "nn.Module",
) -> Optional[tuple]:
    """Infer 2D heatmap shape from model parameters and block count.

    Weight: (out_features, in_features) → (out_features, in_features // block_size)
    Input:  avg across all leading dims → (D[-2], D[-1] // block_size)
    Output: same as input.

    Returns (n_rows, n_cols) or None if shape cannot be inferred.
    """
    module_map = dict(model.named_modules())
    module = module_map.get(layer_name)

    if module is None or not hasattr(module, "weight") or module.weight is None:
        return None

    last_dim = module.weight.shape[-1]  # in_features for Linear, kW for Conv
    n_blocks_per_row = last_dim // block_size
    if n_blocks_per_row == 0:
        return None

    total_blocks = len(blocks)
    n_rows = total_blocks // n_blocks_per_row

    if n_rows * n_blocks_per_row != total_blocks:
        return None

    return (n_rows, n_blocks_per_row)


def block_qsnr_heatmap(
    blocks: Dict[int, float],
    layer_name: str,
    role: str,
    block_size: int,
    model: "nn.Module",
    *,
    label: str = "",
):
    """Render per-block QSNR as a 2D heatmap.

    For weight: rows = output channels, cols = blocks along input dim.
    For input/output: averaged across all leading dims (batch, channels, etc.),
    keeping only the last 2 dims before block reshape.

    Harness heatmap format: [{x: col, y: row, value: qsnr_db}, ...]
    """
    if not blocks:
        return

    shape = _infer_heatmap_shape(blocks, layer_name, role, block_size, model)
    if shape is None:
        return

    n_rows, n_cols = shape

    # Harness heatmap: max 50 per axis
    if n_rows > 50 or n_cols > 50:
        return

    # Build heatmap data: flat block index → (row, col)
    data = []
    for flat_idx, qsnr in sorted(blocks.items()):
        row = flat_idx // n_cols
        col = flat_idx % n_cols
        if row < n_rows and col < n_cols:
            data.append({
                "row": row,
                "col": col,
                "value": round(qsnr, 1),
            })

    if data:
        _chart(data, "heatmap", x="col", y="row",
               label=label,
               title=f"{layer_name} ({role}) Block QSNR Heatmap "
                     f"[{n_rows}×{n_cols}]")


# =====================================================================
# Constants
# =====================================================================

# Reference QSNR for error contribution calculation.
# Represents near-perfect quantization; error_contribution = ref - qsnr.
QSNR_REF = 60.0
