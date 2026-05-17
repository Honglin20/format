#!/usr/bin/env python3
"""Generate granularity × sparse analysis HTML document.

Usage:
    PYTHONPATH=. python scripts/granularity_sparse_analysis.py

Output:
    docs/guides/example/granularity-sparse-analysis.html
"""

import math
import os
import sys

import torch

from src.formats import get_format, FormatBase
from src.quantize import quantize
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.transform import IdentityTransform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "guides", "example")
OUT_FILE = os.path.join(OUT_DIR, "granularity-sparse-analysis.html")

INT4 = get_format("int4")
INT8 = get_format("int8")

SEED_X1 = 42
SEED_W = 43

# ---------------------------------------------------------------------------
# QSNR
# ---------------------------------------------------------------------------
def compute_qsnr(original, quantized):
    signal = (original ** 2).mean().item()
    noise = ((original - quantized) ** 2).mean().item()
    if noise < 1e-12:
        return float("inf")
    return 10.0 * math.log10(signal / noise)

# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------
def make_tensors():
    torch.manual_seed(SEED_X1)
    x1 = torch.randn(1, 8, 16) * 0.8
    # Inject outliers
    x1[0, 0, 3] = 12.0
    x1[0, 2, 10] = -9.5
    x1[0, 4, 1] = 8.0
    x1[0, 5, 14] = 11.0
    x1[0, 7, 7] = -10.0
    x1[0, 1, 12] = 7.5

    torch.manual_seed(SEED_W)
    W = torch.randn(4, 16) * 0.6
    W[0, 5] = 10.0
    W[1, 11] = -8.0
    W[2, 2] = 9.5
    W[3, 14] = 7.0

    return x1, W

# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def quant_with_scheme(tensor, scheme):
    return quantize(tensor, scheme=scheme)

def make_scheme(fmt, granularity, outlier_ratio=0.0, outlier_format=None,
                group_format=None, group_ratio=0.0):
    return QuantScheme(
        format=fmt,
        granularity=granularity,
        transform=IdentityTransform(),
        round_mode="nearest",
        scale_storage="pot",
        outlier_format=outlier_format,
        group_format=group_format,
        group_ratio=group_ratio,
    )

# Granularity specs
GRAN_PER_TENSOR = GranularitySpec.per_tensor()
GRAN_PER_CHANNEL_X1 = GranularitySpec.per_channel(axis=1)   # For 3D input: channel=seq dim
GRAN_PER_CHANNEL_W = GranularitySpec.per_channel(axis=0)    # For 2D weight: channel=row dim
GRAN_PER_BLOCK = GranularitySpec.per_block(size=8, axis=-1)
GRAN_BANK = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)

# Each entry: (display_name, gran_for_x1, gran_for_W)
GRANULARITIES = [
    ("per_tensor", GRAN_PER_TENSOR, GRAN_PER_TENSOR),
    ("per_channel", GRAN_PER_CHANNEL_X1, GRAN_PER_CHANNEL_W),
    ("per_block (size=8)", GRAN_PER_BLOCK, GRAN_PER_BLOCK),
    ("bank (size=4)", GRAN_BANK, GRAN_BANK),
]

OUTLIER_RATIOS = [0.05, 0.10, 0.20, 0.30]
GROUP_RATIOS = [0.10, 0.30, 0.50, 0.70]

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def val_to_color(v, vmax, base="#d5c8f0"):
    """Purple gradient based on |v|/vmax."""
    ratio = min(abs(v) / max(vmax, 1e-9), 1.0)
    r = int(0x6c + (0xd5 - 0x6c) * (1 - ratio))
    g = int(0x5c + (0xc8 - 0x5c) * (1 - ratio))
    b = int(0xe7 + (0xf0 - 0xe7) * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"

def err_to_color(e, emax):
    """Warm red-orange gradient for error magnitude."""
    if emax < 1e-9:
        return "#ffffff"
    ratio = min(abs(e) / emax, 1.0)
    r = 0xff
    g = int(0x76 + (0xff - 0x76) * (1 - ratio))
    b = int(0x55 + (0xff - 0x55) * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"

# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
def render_tensor_grid(tensor_2d, values_2d=None, colors=None, title="",
                       row_labels=None, col_labels=None,
                       highlight_cells=None, highlight_color="#ff4444",
                       group_borders=None, group_colors=None,
                       show_scale=None):
    """Render a 2D slice as an HTML table.

    Args:
        tensor_2d: 2D tensor to display
        values_2d: optional 2D list of formatted values (default: from tensor_2d)
        colors: optional dict of (r,c) -> color
        highlight_cells: set of (r,c) tuples to highlight
        group_borders: list of (row_slice, col_slice, color) for group backgrounds
        show_scale: list of (r, c, text) to overlay scale info
    """
    rows, cols = tensor_2d.shape
    html = ['<table class="tensor-grid">']

    # Header
    if col_labels:
        html.append('<tr><th></th>')
        for c in col_labels:
            html.append(f'<th>{c}</th>')
        html.append('</tr>')

    for r in range(rows):
        html.append('<tr>')
        if row_labels:
            html.append(f'<td class="row-label">{row_labels[r]}</td>')
        for c in range(cols):
            val = values_2d[r][c] if values_2d else f"{tensor_2d[r,c].item():.2f}"
            bg = ""
            if colors and (r, c) in colors:
                bg = f' style="background-color:{colors[(r,c)]}"'
            elif highlight_cells and (r, c) in highlight_cells:
                bg = f' style="background-color:{highlight_color}"'
            elif group_colors:
                for (rs, cs, color) in group_colors:
                    if r in range(rs.start, rs.stop) and c in range(cs.start, cs.stop):
                        bg = f' style="background-color:{color}"'
                        break
            border_style = ""
            if group_borders:
                for (rs, cs, side) in group_borders:
                    if r in range(rs.start, rs.stop) and c in range(cs.start, cs.stop):
                        border_style = f' border-{side}: 2px solid #333;'

            html.append(f'<td{bg}{border_style}>{val}</td>')
        html.append('</tr>')

    # Scale row
    if show_scale:
        html.append('<tr class="scale-row">')
        if row_labels:
            html.append('<td></td>')
        for r_s, c_s, text in show_scale:
            if c_s == 0:
                html.append(f'<td colspan="{cols}" class="scale-cell">{text}</td>')
                break
        html.append('</tr>')

    html.append('</table>')
    return '\n'.join(html)


def render_weight_grid(W, values=None, colors=None, highlight_cells=None,
                       group_colors=None, show_scale=None, highlight_color="#e17055"):
    """Render weight matrix (4x16) as HTML table."""
    return render_tensor_grid(W, values, colors, title="W",
                              row_labels=[f"ch{i}" for i in range(W.shape[0])],
                              col_labels=[f"{i}" for i in range(W.shape[1])],
                              highlight_cells=highlight_cells,
                              highlight_color=highlight_color,
                              group_colors=group_colors,
                              show_scale=show_scale)


def render_input_grid(x1, values=None, colors=None, highlight_cells=None,
                      group_colors=None, highlight_color="#e17055"):
    """Render input tensor (1, 8, 16) as HTML table (8x16)."""
    slice_2d = x1[0]  # (8, 16)
    return render_tensor_grid(slice_2d, values, colors, title="x1",
                              row_labels=[f"s{i}" for i in range(8)],
                              col_labels=[f"{i}" for i in range(16)],
                              highlight_cells=highlight_cells,
                              highlight_color=highlight_color,
                              group_colors=group_colors)


# ---------------------------------------------------------------------------
# Compute all data
# ---------------------------------------------------------------------------
def compute_all():
    x1, W = make_tensors()
    y_fp32 = x1 @ W.T  # (1, 8, 4)

    results = {}

    # --- Part 2: Base granularities ---
    base_results = {}
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        scheme_x = make_scheme(INT4, gran_x1)
        scheme_w = make_scheme(INT4, gran_w)
        x1_q = quant_with_scheme(x1, scheme_x)
        W_q = quant_with_scheme(W, scheme_w)
        y_q = x1_q @ W_q.T
        base_results[gran_name] = {
            "x1_q": x1_q,
            "W_q": W_q,
            "y_q": y_q,
            "x1_qsnr": compute_qsnr(x1, x1_q),
            "W_qsnr": compute_qsnr(W, W_q),
            "y_qsnr": compute_qsnr(y_fp32, y_q),
            "gran_x1": gran_x1,
            "gran_w": gran_w,
        }
    results["base"] = base_results

    # --- Part 3: Element Sparse ---
    elem_results = {}
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        for ratio in OUTLIER_RATIOS:
            gran_x1_r = GranularitySpec(
                mode=gran_x1.mode,
                block_size=gran_x1.block_size,
                channel_axis=gran_x1.channel_axis,
                block_axis=gran_x1.block_axis,
                bank_size=gran_x1.bank_size,
                bank_axis=gran_x1.bank_axis,
                outlier_ratio=ratio,
            )
            gran_w_r = GranularitySpec(
                mode=gran_w.mode,
                block_size=gran_w.block_size,
                channel_axis=gran_w.channel_axis,
                block_axis=gran_w.block_axis,
                bank_size=gran_w.bank_size,
                bank_axis=gran_w.bank_axis,
                outlier_ratio=ratio,
            )
            key = f"{gran_name}_ratio{ratio}"
            try:
                scheme_x = make_scheme(INT4, gran_x1_r, outlier_format=INT8)
                scheme_w = make_scheme(INT4, gran_w_r, outlier_format=INT8)
                x1_q = quant_with_scheme(x1, scheme_x)
                W_q = quant_with_scheme(W, scheme_w)
                y_q = x1_q @ W_q.T
                elem_results[key] = {
                    "x1_q": x1_q,
                    "W_q": W_q,
                    "y_q": y_q,
                    "x1_qsnr": compute_qsnr(x1, x1_q),
                    "W_qsnr": compute_qsnr(W, W_q),
                    "y_qsnr": compute_qsnr(y_fp32, y_q),
                    "gran_x1": gran_x1_r,
                    "gran_w": gran_w_r,
                    "ratio": ratio,
                }
            except Exception as e:
                elem_results[key] = {"error": str(e), "ratio": ratio}
    results["element_sparse"] = elem_results

    # --- Part 4: Group Sparse ---
    group_results = {}
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        for ratio in GROUP_RATIOS:
            key = f"{gran_name}_ratio{ratio}"
            try:
                scheme_x = make_scheme(INT4, gran_x1, group_format=INT8, group_ratio=ratio)
                scheme_w = make_scheme(INT4, gran_w, group_format=INT8, group_ratio=ratio)
                x1_q = quant_with_scheme(x1, scheme_x)
                W_q = quant_with_scheme(W, scheme_w)
                y_q = x1_q @ W_q.T
                group_results[key] = {
                    "x1_q": x1_q,
                    "W_q": W_q,
                    "y_q": y_q,
                    "x1_qsnr": compute_qsnr(x1, x1_q),
                    "W_qsnr": compute_qsnr(W, W_q),
                    "y_qsnr": compute_qsnr(y_fp32, y_q),
                    "gran_x1": gran_x1,
                    "gran_w": gran_w,
                    "ratio": ratio,
                }
            except Exception as e:
                group_results[key] = {"error": str(e), "ratio": ratio}
    results["group_sparse"] = group_results

    return x1, W, y_fp32, results


# ---------------------------------------------------------------------------
# Get outlier mask for visualization (dynamic, top-k by magnitude)
# ---------------------------------------------------------------------------
def get_outlier_mask(tensor, gran, ratio):
    """Compute per-element outlier mask for visualization."""
    if ratio <= 0:
        return set()

    flat = tensor.flatten()
    n = flat.numel()
    k = max(1, int(n * ratio))
    _, top_indices = torch.topk(flat.abs(), k)
    mask_set = set()
    for idx in top_indices.tolist():
        # Convert flat index to multi-dim
        if tensor.dim() == 3:
            b, r, c = torch.unravel_index(torch.tensor(idx), tensor.shape)
            mask_set.add((r.item(), c.item()))
        elif tensor.dim() == 2:
            r, c = torch.unravel_index(torch.tensor(idx), tensor.shape)
            mask_set.add((r.item(), c.item()))
    return mask_set


def get_group_mask(tensor, gran, ratio):
    """Compute per-group mask for visualization."""
    if ratio <= 0:
        return set()

    # For per_channel: groups are channels (rows)
    if gran.mode == GranularityMode.PER_CHANNEL:
        n_groups = tensor.shape[0] if tensor.dim() == 2 else tensor.shape[1]
        k = min(n_groups, max(1, int(n_groups * ratio)))
        # Score by channel amax
        if tensor.dim() == 2:
            scores = tensor.abs().amax(dim=1)  # (C,)
        else:
            scores = tensor[0].abs().amax(dim=1)  # (C,)
        _, top_indices = torch.topk(scores, k)
        return set(top_indices.tolist())

    # For per_block: groups are blocks along last dim
    if gran.mode == GranularityMode.PER_BLOCK:
        bs = gran.block_size
        if tensor.dim() == 2:
            reshaped = tensor.reshape(tensor.shape[0], -1, bs)
            scores = reshaped.abs().amax(dim=-1).flatten()  # (rows * n_blocks_per_row,)
        else:
            reshaped = tensor[0].reshape(tensor.shape[1], -1, bs)
            scores = reshaped.abs().amax(dim=-1).flatten()  # (rows * n_blocks_per_row,)
        n_groups = scores.numel()
        k = min(n_groups, max(1, int(n_groups * ratio)))
        _, top_indices = torch.topk(scores, k)
        return set(top_indices.tolist())

    # For bank: groups are banks along last dim
    if gran.mode == GranularityMode.BANK:
        bs = gran.bank_size
        if tensor.dim() == 2:
            n_groups = tensor.shape[1] // bs
        else:
            n_groups = tensor.shape[2] // bs
        k = min(n_groups, max(1, int(n_groups * ratio)))
        if tensor.dim() == 2:
            reshaped = tensor.reshape(tensor.shape[0], -1, bs)
            scores = reshaped.abs().amax(dim=(0, 2))  # (n_banks,)
        else:
            reshaped = tensor[0].reshape(tensor.shape[1], -1, bs)
            scores = reshaped.abs().amax(dim=(0, 2))  # (n_banks,)
        _, top_indices = torch.topk(scores, k)
        return set(top_indices.tolist())

    # per_tensor: only 1 group
    return {0} if ratio > 0 else set()


def get_group_bg_cells(tensor, gran, h_groups, group_type="channel"):
    """Get (row_slice, col_slice, color) for group background coloring."""
    cells = []

    if gran.mode == GranularityMode.PER_CHANNEL:
        if tensor.dim() == 2:
            for g in h_groups:
                cells.append((slice(g, g+1), slice(0, tensor.shape[1]), "#c8f7c5"))
        else:
            for g in h_groups:
                cells.append((slice(g, g+1), slice(0, tensor.shape[2]), "#c8f7c5"))

    elif gran.mode == GranularityMode.PER_BLOCK:
        bs = gran.block_size
        if tensor.dim() == 2:
            for g in h_groups:
                row_idx = g // (tensor.shape[1] // bs)
                col_block = g % (tensor.shape[1] // bs)
                c_start = col_block * bs
                cells.append((slice(row_idx, row_idx+1), slice(c_start, c_start+bs), "#c8f7c5"))
        else:
            for g in h_groups:
                row_idx = g // (tensor.shape[2] // bs)
                col_block = g % (tensor.shape[2] // bs)
                c_start = col_block * bs
                cells.append((slice(row_idx, row_idx+1), slice(c_start, c_start+bs), "#c8f7c5"))

    elif gran.mode == GranularityMode.BANK:
        bs = gran.bank_size
        for g in h_groups:
            c_start = g * bs
            if tensor.dim() == 2:
                cells.append((slice(0, tensor.shape[0]), slice(c_start, c_start+bs), "#c8f7c5"))
            else:
                cells.append((slice(0, tensor.shape[1]), slice(c_start, c_start+bs), "#c8f7c5"))

    elif gran.mode == GranularityMode.PER_TENSOR:
        if tensor.dim() == 2:
            cells.append((slice(0, tensor.shape[0]), slice(0, tensor.shape[1]), "#c8f7c5"))
        else:
            cells.append((slice(0, tensor.shape[1]), slice(0, tensor.shape[2]), "#c8f7c5"))

    return cells


# ---------------------------------------------------------------------------
# Scale count
# ---------------------------------------------------------------------------
def scale_count(gran, tensor_shape, sparse=False):
    """Compute number of scale values for a given granularity."""
    if gran.mode == GranularityMode.PER_TENSOR:
        n = 1
    elif gran.mode == GranularityMode.PER_CHANNEL:
        if len(tensor_shape) == 2:
            n = tensor_shape[0]
        else:
            n = tensor_shape[1]
    elif gran.mode == GranularityMode.PER_BLOCK:
        if len(tensor_shape) == 2:
            n = tensor_shape[0] * (tensor_shape[1] // gran.block_size)
        else:
            n = tensor_shape[1] * (tensor_shape[2] // gran.block_size)
    elif gran.mode == GranularityMode.BANK:
        if len(tensor_shape) == 2:
            n = tensor_shape[1] // gran.bank_size
        else:
            n = tensor_shape[2] // gran.bank_size
    else:
        n = 1
    return n * 2 if sparse else n


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
def build_html(x1, W, y_fp32, results):
    vmax_x = x1.abs().max().item()
    vmax_w = W.abs().max().item()

    sections = []

    # ===== CSS =====
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
           max-width: 1100px; margin: 0 auto; padding: 24px; background: #f8f9fb;
           color: #1a1d23; line-height: 1.6; }
    h1 { font-size: 1.8em; margin: 0 0 8px; border-bottom: 2px solid #6c5ce7; padding-bottom: 8px; color: #2d3436; }
    h2 { font-size: 1.4em; margin: 32px 0 12px; color: #6c5ce7; }
    h3 { font-size: 1.15em; margin: 24px 0 8px; color: #2d3436; }
    h4 { font-size: 1.0em; margin: 16px 0 6px; color: #636e72; }
    p, li { margin: 6px 0; }
    code { background: #f0edf7; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #6c5ce7; }
    .tensor-grid { border-collapse: collapse; margin: 8px 0 16px; font-size: 0.82em; font-family: 'SF Mono', 'Consolas', monospace; }
    .tensor-grid td, .tensor-grid th { border: 1px solid #dfe6e9; padding: 4px 6px; text-align: right; min-width: 42px; }
    .tensor-grid th { background: #f0f2f5; font-weight: 600; font-size: 0.85em; color: #636e72; }
    .tensor-grid .row-label { background: #f0f2f5; font-weight: 600; text-align: center; color: #636e72; }
    .tensor-grid .scale-cell { background: #ffeaa7; font-weight: 600; text-align: center; color: #856404; font-size: 0.9em; }
    .section-card { background: white; border: 1px solid #e1e4e8; border-radius: 10px; padding: 24px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .comparison-grid { display: grid; grid-template-columns: 1fr; gap: 20px; margin: 12px 0; }
    .label { display: inline-block; background: #6c5ce7; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; margin-right: 4px; }
    .label-outlier { background: #e17055; }
    .label-h-group { background: #00b894; }
    .label-l-group { background: #636e72; }
    table.qsnr-table { border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
    table.qsnr-table th, table.qsnr-table td { border: 1px solid #dfe6e9; padding: 6px 12px; text-align: center; }
    table.qsnr-table th { background: #f0f2f5; color: #2d3436; }
    table.qsnr-table td.best { background: #c8f7c5; font-weight: 600; }
    table.qsnr-table td.worst { background: #ffb3b3; }
    .legend { margin: 12px 0; padding: 10px 14px; background: #f0f2f5; border-radius: 6px; font-size: 0.9em; }
    .legend-item { display: inline-block; margin-right: 16px; }
    .legend-swatch { display: inline-block; width: 14px; height: 14px; border: 1px solid #b2bec3; border-radius: 2px; vertical-align: middle; margin-right: 4px; }
    .note { background: #fff8e1; border-left: 3px solid #fdcb6e; padding: 8px 12px; margin: 12px 0; font-size: 0.9em; border-radius: 0 6px 6px 0; }
    .error-heatmap .tensor-grid td { min-width: 36px; }
    .nav { background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: white; padding: 10px 16px; border-radius: 8px; margin-bottom: 20px; }
    .nav a { color: #dfe6e9; text-decoration: none; margin-right: 16px; font-size: 0.9em; transition: color 0.2s; }
    .nav a:hover { color: white; }
    .embed-img { width: 100%; max-width: 900px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    @media (max-width: 768px) {
        .comparison-grid { grid-template-columns: 1fr; }
        .tensor-grid { font-size: 0.72em; }
        .tensor-grid td, .tensor-grid th { padding: 2px 3px; min-width: 32px; }
    }
    """

    # ===== Part 0: Terminology =====
    s0 = """
    <div class="section-card">
    <h2>0. 术语</h2>
    <p>本库提供两种互补的 sparse 量化模式，<strong>互斥使用</strong>：</p>
    <table class="qsnr-table">
    <tr><th></th><th>Element Sparse (ADR-012)</th><th>Group Sparse (ADR-013)</th></tr>
    <tr><td><strong>选择单位</strong></td><td>单个元素（per-element）</td><td>粒度组（per-group：channel / block / bank）</td></tr>
    <tr><td><strong>Mask 形状</strong></td><td>与 tensor 相同（per-element bool）</td><td>per-group（如 (C,) 表示 C 个 channel 各 1 bool）</td></tr>
    <tr><td><strong>组内一致性</strong></td><td>同一 group 内可混用 H/L</td><td>一个 group 内统一 H 或 L</td></tr>
    <tr><td><strong>配置方式</strong></td><td><code>outlier_ratio</code> + <code>outlier_format</code></td><td><code>group_format</code> + <code>group_ratio</code></td></tr>
    <tr><td><strong>硬件友好度</strong></td><td>低（需 per-element 索引）</td><td>高（组内一致，无需逐元素索引）</td></tr>
    <tr><td><strong>适用场景</strong></td><td>少量极端离群点</td><td>某些 channel/block 整体更重要</td></tr>
    </table>
    </div>
    """
    sections.append(s0)

    # ===== Part 1: Tensors =====
    x1_slice = x1[0]  # (8, 16)
    # Color by magnitude
    x1_colors = {}
    for r in range(x1_slice.shape[0]):
        for c in range(x1_slice.shape[1]):
            x1_colors[(r, c)] = val_to_color(x1_slice[r, c].item(), vmax_x)

    W_colors = {}
    for r in range(W.shape[0]):
        for c in range(W.shape[1]):
            W_colors[(r, c)] = val_to_color(W[r, c].item(), vmax_w)

    # Mark outliers
    x1_outlier_locs = set()
    for r in range(8):
        for c in range(16):
            if abs(x1_slice[r, c].item()) > 5.0:
                x1_outlier_locs.add((r, c))

    W_outlier_locs = set()
    for r in range(4):
        for c in range(16):
            if abs(W[r, c].item()) > 5.0:
                W_outlier_locs.add((r, c))

    s1_x1 = render_input_grid(x1, colors=x1_colors)
    s1_W = render_weight_grid(W, colors=W_colors)

    s1 = f"""
    <div class="section-card">
    <h2>1. Tensor 定义与原始值</h2>
    <div class="legend">
        <span class="legend-item"><span class="legend-swatch" style="background:#d5c8f0"></span> 正常值</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#6c5ce7"></span> 高 magnitude</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#e17055"></span> 人造 outlier（|v| > 5）</span>
    </div>

    <h3>x1 — matmul input, shape (1, 8, 16)</h3>
    <p>模拟 Transformer 输入：batch=1, seq_len=8, hidden=16。6 个人造 outlier。</p>
    {s1_x1}

    <h3>W — weight, shape (4, 16)</h3>
    <p>out=4 channels, in=16。4 个人造 outlier。</p>
    {s1_W}

    <h3>y = x1 @ W<sup>T</sup> — FP32 输出, shape (1, 8, 4)</h3>
    <p>作为后续输出 QSNR 的参考基准。</p>
    </div>
    """
    sections.append(s1)

    # ===== Part 2: Base granularities =====
    s2_parts = ['<div class="section-card">', '<h2>2. 基础粒度（int4，无 sparse）</h2>']
    s2_parts.append("""
    <div class="legend">
        <span class="legend-item"><span class="legend-swatch" style="background:#d5c8f0"></span> 同一 group 共享 scale</span>
        <span class="legend-item">标量值 → 量化后值（误差）</span>
    </div>
    """)

    for gran_name, gran_x1, gran_w in GRANULARITIES:
        br = results["base"][gran_name]
        x1_q = br["x1_q"]
        W_q = br["W_q"]

        # Build quantized value display
        x1_slice = x1[0]
        x1_q_slice = x1_q[0]
        x1_vals = []
        for r in range(8):
            row = []
            for c in range(16):
                orig = x1_slice[r, c].item()
                qval = x1_q_slice[r, c].item()
                err = abs(orig - qval)
                row.append(f"{qval:.2f}<br><span style='font-size:0.75em;color:#999'>({err:.2f})</span>")
            x1_vals.append(row)

        W_vals = []
        for r in range(4):
            row = []
            for c in range(16):
                orig = W[r, c].item()
                qval = W_q[r, c].item()
                err = abs(orig - qval)
                row.append(f"{qval:.2f}<br><span style='font-size:0.75em;color:#999'>({err:.2f})</span>")
            W_vals.append(row)

        # Group coloring for base granularity
        x1_gcolors = _gran_group_colors(x1, gran_x1, base=True)
        W_gcolors = _gran_group_colors(W, gran_w, base=True)

        x1_grid = render_input_grid(x1, values=x1_vals, group_colors=x1_gcolors)
        W_grid = render_weight_grid(W, values=W_vals, group_colors=W_gcolors)

        n_scales_x = scale_count(gran_x1, x1.shape)
        n_scales_w = scale_count(gran_w, W.shape)

        s2_parts.append(f"""
        <h3>{gran_name}</h3>
        <div class="comparison-grid">
            <div>
                <h4>x1 量化后 (scale 数: {n_scales_x})</h4>
                {x1_grid}
            </div>
            <div>
                <h4>W 量化后 (scale 数: {n_scales_w})</h4>
                {W_grid}
            </div>
        </div>
        """)

    # Summary table
    s2_parts.append('<h3>QSNR 汇总</h3>')
    s2_parts.append('<table class="qsnr-table"><tr><th>粒度</th><th>x1 QSNR (dB)</th><th>W QSNR (dB)</th><th>输出 QSNR (dB)</th><th>x1 scale 数</th><th>W scale 数</th></tr>')
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        br = results["base"][gran_name]
        n_sx = scale_count(gran_x1, x1.shape)
        n_sw = scale_count(gran_w, W.shape)
        s2_parts.append(f'<tr><td>{gran_name}</td><td>{br["x1_qsnr"]:.1f}</td><td>{br["W_qsnr"]:.1f}</td><td>{br["y_qsnr"]:.1f}</td><td>{n_sx}</td><td>{n_sw}</td></tr>')
    s2_parts.append('</table>')
    s2_parts.append('</div>')
    sections.append('\n'.join(s2_parts))

    # ===== Part 3: Element Sparse =====
    s3_parts = ['<div class="section-card">', '<h2>3. Element Sparse（outlier 升级到 int8）</h2>']
    s3_parts.append("""
    <div class="legend">
        <span class="legend-item"><span class="legend-swatch" style="background:#e17055"></span> Outlier（int8 量化）</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#d5c8f0"></span> Normal（int4 量化）</span>
        <span class="legend-item"><code>outlier_format="int8"</code></span>
    </div>
    """)

    # Detailed view for ratio=0.10
    DETAIL_RATIO = 0.10
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        key = f"{gran_name}_ratio{DETAIL_RATIO}"
        er = results["element_sparse"][key]
        if "error" in er:
            s3_parts.append(f'<h3>{gran_name} (ratio={DETAIL_RATIO})</h3><p class="note">不支持: {er["error"]}</p>')
            continue

        x1_q = er["x1_q"]
        W_q = er["W_q"]

        # Outlier masks
        x1_mask = get_outlier_mask(x1, gran_x1, DETAIL_RATIO)
        W_mask = get_outlier_mask(W, gran_w, DETAIL_RATIO)

        # Quantized values with outlier highlighting
        x1_slice = x1[0]
        x1_q_slice = x1_q[0]
        x1_vals = []
        for r in range(8):
            row = []
            for c in range(16):
                orig = x1_slice[r, c].item()
                qval = x1_q_slice[r, c].item()
                err = abs(orig - qval)
                marker = " 🔴" if (r, c) in x1_mask else ""
                row.append(f"{qval:.2f}{marker}<br><span style='font-size:0.75em;color:#999'>({err:.2f})</span>")
            x1_vals.append(row)

        W_vals = []
        for r in range(4):
            row = []
            for c in range(16):
                orig = W[r, c].item()
                qval = W_q[r, c].item()
                err = abs(orig - qval)
                marker = " 🔴" if (r, c) in W_mask else ""
                row.append(f"{qval:.2f}{marker}<br><span style='font-size:0.75em;color:#999'>({err:.2f})</span>")
            W_vals.append(row)

        x1_grid = render_input_grid(x1, values=x1_vals, highlight_cells=x1_mask, highlight_color="#e17055")
        W_grid = render_weight_grid(W, values=W_vals, highlight_cells=W_mask, highlight_color="#e17055")

        # Base comparison
        br = results["base"][gran_name]
        delta_x1 = er["x1_qsnr"] - br["x1_qsnr"]
        delta_W = er["W_qsnr"] - br["W_qsnr"]
        delta_y = er["y_qsnr"] - br["y_qsnr"]

        n_sx_base = scale_count(gran_x1, x1.shape)
        n_sw_base = scale_count(gran_w, W.shape)
        n_sx_sparse = scale_count(gran_x1, x1.shape, sparse=True)
        n_sw_sparse = scale_count(gran_w, W.shape, sparse=True)

        s3_parts.append(f"""
        <h3>{gran_name} (outlier_ratio={DETAIL_RATIO})</h3>
        <div class="comparison-grid">
            <div>
                <h4>x1 (outlier=🔴 → int8, normal → int4)</h4>
                {x1_grid}
            </div>
            <div>
                <h4>W (outlier=🔴 → int8, normal → int4)</h4>
                {W_grid}
            </div>
        </div>
        <table class="qsnr-table">
        <tr><th></th><th>x1 QSNR</th><th>W QSNR</th><th>输出 QSNR</th><th>x1 scale 数</th><th>W scale 数</th></tr>
        <tr><td>无 sparse</td><td>{br["x1_qsnr"]:.1f} dB</td><td>{br["W_qsnr"]:.1f} dB</td><td>{br["y_qsnr"]:.1f} dB</td><td>{n_sx_base}</td><td>{n_sw_base}</td></tr>
        <tr><td>element sparse</td><td>{er["x1_qsnr"]:.1f} dB</td><td>{er["W_qsnr"]:.1f} dB</td><td>{er["y_qsnr"]:.1f} dB</td><td>{n_sx_sparse}</td><td>{n_sw_sparse}</td></tr>
        <tr><td><strong>Δ</strong></td><td class="best">+{delta_x1:.1f}</td><td class="best">+{delta_W:.1f}</td><td class="best">+{delta_y:.1f}</td><td>+{n_sx_sparse - n_sx_base}</td><td>+{n_sw_sparse - n_sw_base}</td></tr>
        </table>
        """)

    # Ratio sweep table
    s3_parts.append('<h3>outlier_ratio 扫描</h3>')
    s3_parts.append('<table class="qsnr-table"><tr><th>粒度</th>')
    for ratio in OUTLIER_RATIOS:
        s3_parts.append(f'<th>ratio={ratio}</th>')
    s3_parts.append('</tr>')

    for gran_name, gran_x1, gran_w in GRANULARITIES:
        s3_parts.append(f'<tr><td>{gran_name}</td>')
        for ratio in OUTLIER_RATIOS:
            key = f"{gran_name}_ratio{ratio}"
            er = results["element_sparse"][key]
            if "error" in er:
                s3_parts.append('<td>N/A</td>')
            else:
                br = results["base"][gran_name]
                delta = er["y_qsnr"] - br["y_qsnr"]
                s3_parts.append(f'<td>{er["y_qsnr"]:.1f} (+{delta:.1f})</td>')
        s3_parts.append('</tr>')
    s3_parts.append('</table>')
    s3_parts.append('<p class="note">括号内为相对无 sparse 基线的 Δ QSNR (dB)。值越大越好。</p>')
    s3_parts.append("""<div class="note">
    <strong>注意：per_block ratio=0.05~0.20 结果相同 (27.0 dB)</strong>：block_size=8 时，每个 block 8 个元素。
    <code>k = max(1, int(8 × ratio))</code>，当 ratio ∈ [0.05, 0.20] 时 k 均为 1（每 block 选 1 个 outlier），
    因此量化结果完全相同。ratio=0.30 时 k 变为 2，QSNR 跳升至 33.3 dB。
    这是离散 k 值的阶梯效应——group 越小，ratio 的有效粒度越粗。
    </div>""")
    s3_parts.append('</div>')
    sections.append('\n'.join(s3_parts))

    # ===== Part 4: Group Sparse =====
    s4_parts = ['<div class="section-card">', '<h2>4. Group Sparse（H 组升级到 int8）</h2>']
    s4_parts.append("""
    <div class="legend">
        <span class="legend-item"><span class="legend-swatch" style="background:#c8f7c5"></span> H group（int8 量化，高精度）</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#d5c8f0"></span> L group（int4 量化，低精度）</span>
        <span class="legend-item"><code>group_format="int8"</code></span>
    </div>
    """)

    DETAIL_GROUP_RATIO = 0.30
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        key = f"{gran_name}_ratio{DETAIL_GROUP_RATIO}"
        gr = results["group_sparse"][key]
        if "error" in gr:
            s4_parts.append(f'<h3>{gran_name} (group_ratio={DETAIL_GROUP_RATIO})</h3><p class="note">不支持: {gr["error"]}</p>')
            continue

        x1_q = gr["x1_q"]
        W_q = gr["W_q"]

        # Group masks
        x1_h_groups = get_group_mask(x1, gran_x1, DETAIL_GROUP_RATIO)
        W_h_groups = get_group_mask(W, gran_w, DETAIL_GROUP_RATIO)

        x1_gcolors = get_group_bg_cells(x1, gran_x1, x1_h_groups)
        W_gcolors = get_group_bg_cells(W, gran_w, W_h_groups)

        # Quantized values
        x1_slice = x1[0]
        x1_q_slice = x1_q[0]
        x1_vals = []
        for r in range(8):
            row = []
            for c in range(16):
                orig = x1_slice[r, c].item()
                qval = x1_q_slice[r, c].item()
                err = abs(orig - qval)
                row.append(f"{qval:.2f}<br><span style='font-size:0.75em;color:#999'>({err:.2f})</span>")
            x1_vals.append(row)

        W_vals = []
        for r in range(4):
            row = []
            for c in range(16):
                orig = W[r, c].item()
                qval = W_q[r, c].item()
                err = abs(orig - qval)
                row.append(f"{qval:.2f}<br><span style='font-size:0.75em;color:#999'>({err:.2f})</span>")
            W_vals.append(row)

        x1_grid = render_input_grid(x1, values=x1_vals, group_colors=x1_gcolors)
        W_grid = render_weight_grid(W, values=W_vals, group_colors=W_gcolors)

        br = results["base"][gran_name]
        delta_x1 = gr["x1_qsnr"] - br["x1_qsnr"]
        delta_W = gr["W_qsnr"] - br["W_qsnr"]
        delta_y = gr["y_qsnr"] - br["y_qsnr"]

        n_h_x = len(x1_h_groups)
        n_h_w = len(W_h_groups)

        s4_parts.append(f"""
        <h3>{gran_name} (group_ratio={DETAIL_GROUP_RATIO})</h3>
        <p>H group 数: x1={n_h_x}, W={n_h_w}</p>
        <div class="comparison-grid">
            <div>
                <h4>x1 (H group=🟢 → int8, L group → int4)</h4>
                {x1_grid}
            </div>
            <div>
                <h4>W (H group=🟢 → int8, L group → int4)</h4>
                {W_grid}
            </div>
        </div>
        <table class="qsnr-table">
        <tr><th></th><th>x1 QSNR</th><th>W QSNR</th><th>输出 QSNR</th></tr>
        <tr><td>无 sparse</td><td>{br["x1_qsnr"]:.1f} dB</td><td>{br["W_qsnr"]:.1f} dB</td><td>{br["y_qsnr"]:.1f} dB</td></tr>
        <tr><td>group sparse</td><td>{gr["x1_qsnr"]:.1f} dB</td><td>{gr["W_qsnr"]:.1f} dB</td><td>{gr["y_qsnr"]:.1f} dB</td></tr>
        <tr><td><strong>Δ</strong></td><td class="best">+{delta_x1:.1f}</td><td class="best">+{delta_W:.1f}</td><td class="best">+{delta_y:.1f}</td></tr>
        </table>
        """)

        # Add per-granularity notes explaining boundary behavior
        if gran_name == "per_tensor":
            s4_parts.append("""<p class="note"><strong>退化行为</strong>：per_tensor 只有 1 个 group，group_ratio &gt; 0 时该 group 必然是 H，
            全部元素用 int8。结果等价于直接 int8 per_tensor 量化，与 group_ratio 具体值无关。</p>""")
        elif gran_name == "per_channel":
            s4_parts.append(f"""<p class="note"><strong>为何效果弱</strong>：x1 有 8 个 channel，仅 {n_h_x} 个被选为 H（int8）。
            但 ch2 (amax=9.5) 和 ch7 (amax=10.0) 仍在 L 组（int4），它们内部的 outlier 继续碾压正常值。
            Group sparse 无法在 channel 内部只拯救少数元素——这是与 element sparse 的核心差异。</p>""")
        elif "bank" in gran_name:
            s4_parts.append(f"""<p class="note"><strong>为何效果弱</strong>：x1 有 4 个 bank，仅 {n_h_x} 个被选为 H（int8）。
            含大值的 bank 如果 amax 排名不够高，仍用 int4，outlier 继续主导该 bank 的 scale。</p>""")

    # Ratio sweep table
    s4_parts.append('<h3>group_ratio 扫描</h3>')
    s4_parts.append('<table class="qsnr-table"><tr><th>粒度</th>')
    for ratio in GROUP_RATIOS:
        s4_parts.append(f'<th>ratio={ratio}</th>')
    s4_parts.append('</tr>')

    for gran_name, gran_x1, gran_w in GRANULARITIES:
        s4_parts.append(f'<tr><td>{gran_name}</td>')
        for ratio in GROUP_RATIOS:
            key = f"{gran_name}_ratio{ratio}"
            gr = results["group_sparse"][key]
            if "error" in gr:
                s4_parts.append('<td>N/A</td>')
            else:
                br = results["base"][gran_name]
                delta = gr["y_qsnr"] - br["y_qsnr"]
                s4_parts.append(f'<td>{gr["y_qsnr"]:.1f} (+{delta:.1f})</td>')
        s4_parts.append('</tr>')
    s4_parts.append('</table>')
    s4_parts.append('<p class="note">括号内为相对无 sparse 基线的 Δ QSNR (dB)。值越大越好。</p>')
    s4_parts.append("""<div class="note">
    <strong>注意 1：per_tensor group sparse 所有 ratio 结果相同 (26.0 dB)</strong>：per_tensor 只有 1 个 group，
    <code>group_ratio &gt; 0</code> 时该唯一 group 必然是 H → 全部元素使用 int8。
    这等价于直接用 int8 做per_tensor 量化，与 ratio 具体值无关。<br>
    <strong>注意 2：per_channel / bank group sparse 效果弱于 element sparse</strong>：group sparse 按 group 整体分配格式。
    以 per_channel 为例，8 个 channel 中仅 2 个被选为 H（int8），但 ch2 (amax=9.5)、ch7 (amax=10.0)
    等含大值的 channel 仍是 L（int4），其内部 outlier 继续碾压同 channel 的正常值。
    Element sparse 不受此限制——它在一个 channel 内可以只拯救少数 outlier 元素。
    <strong>当 group 内 outlier 比例高但 group 整体 amax 排名不够高时，group sparse 会漏掉这些 group</strong>。
    </div>""")
    s4_parts.append('</div>')
    sections.append('\n'.join(s4_parts))

    # ===== Part 5: Error heatmaps =====
    s5_parts = ['<div class="section-card">', '<h2>5. 误差热点图</h2>']
    s5_parts.append('<p>per-element |x - x_q| 热力图，红色越深 = 误差越大。</p>')

    configs_to_show = [
        ("per_tensor", "base", None),
        ("per_tensor", "element_sparse", f"per_tensor_ratio{DETAIL_RATIO}"),
        ("per_tensor", "group_sparse", f"per_tensor_ratio{DETAIL_GROUP_RATIO}"),
        ("per_channel", "base", None),
        ("per_channel", "element_sparse", f"per_channel_ratio{DETAIL_RATIO}"),
        ("per_channel", "group_sparse", f"per_channel_ratio{DETAIL_GROUP_RATIO}"),
    ]

    for gran_name, mode, key in configs_to_show:
        if mode == "base":
            x1_q = results["base"][gran_name]["x1_q"]
            label = f"{gran_name} (无 sparse)"
        elif mode == "element_sparse":
            r = results["element_sparse"][key]
            if "error" in r:
                continue
            x1_q = r["x1_q"]
            label = f"{gran_name} + element sparse (ratio={DETAIL_RATIO})"
        else:
            r = results["group_sparse"][key]
            if "error" in r:
                continue
            x1_q = r["x1_q"]
            label = f"{gran_name} + group sparse (ratio={DETAIL_GROUP_RATIO})"

        err = (x1[0] - x1_q[0]).abs()
        emax = err.max().item()

        err_colors = {}
        err_vals = []
        for r in range(8):
            row = []
            for c in range(16):
                e = err[r, c].item()
                err_colors[(r, c)] = err_to_color(e, emax)
                row.append(f"{e:.2f}")
            err_vals.append(row)

        grid = render_input_grid(x1, values=err_vals, colors=err_colors)
        s5_parts.append(f'<h4>{label} — max error = {emax:.2f}</h4>')
        s5_parts.append(grid)

    s5_parts.append('</div>')
    sections.append('\n'.join(s5_parts))

    # ===== Part 6: Summary matrix =====
    s6_parts = ['<div class="section-card">', '<h2>6. 汇总矩阵</h2>']
    s6_parts.append('<table class="qsnr-table">')
    s6_parts.append('<tr><th>粒度</th><th>无 sparse</th><th>Element Sparse (ratio=0.1)</th><th>Group Sparse (ratio=0.3)</th></tr>')

    for gran_name, gran_x1, gran_w in GRANULARITIES:
        base_qsnr = results["base"][gran_name]["y_qsnr"]
        elem_key = f"{gran_name}_ratio0.1"
        elem_r = results["element_sparse"][elem_key]
        elem_qsnr = elem_r["y_qsnr"] if "error" not in elem_r else "N/A"
        group_key = f"{gran_name}_ratio0.3"
        group_r = results["group_sparse"][group_key]
        group_qsnr = group_r["y_qsnr"] if "error" not in group_r else "N/A"

        base_str = f'{base_qsnr:.1f} dB'
        elem_str = f'{elem_qsnr:.1f} dB' if isinstance(elem_qsnr, float) else elem_qsnr
        group_str = f'{group_qsnr:.1f} dB' if isinstance(group_qsnr, float) else group_qsnr

        s6_parts.append(f'<tr><td>{gran_name}</td><td>{base_str}</td><td>{elem_str}</td><td>{group_str}</td></tr>')

    s6_parts.append('</table>')

    # Scale overhead table
    s6_parts.append('<h3>Scale 开销</h3>')
    s6_parts.append('<table class="qsnr-table">')
    s6_parts.append('<tr><th>粒度</th><th>无 sparse (x1/W)</th><th>Element Sparse (x1/W)</th><th>Group Sparse (x1/W)</th></tr>')
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        n_base_x = scale_count(gran_x1, x1.shape)
        n_base_w = scale_count(gran_w, W.shape)
        n_elem_x = scale_count(gran_x1, x1.shape, sparse=True)
        n_elem_w = scale_count(gran_w, W.shape, sparse=True)
        # Group sparse: same scale count as base (just different format per group)
        n_group_x = n_base_x
        n_group_w = n_base_w
        s6_parts.append(f'<tr><td>{gran_name}</td><td>{n_base_x}/{n_base_w}</td><td>{n_elem_x}/{n_elem_w}</td><td>{n_group_x}/{n_group_w}</td></tr>')
    s6_parts.append('</table>')
    s6_parts.append('<p class="note">Group Sparse 不增加 scale 数量——仅改变每个 group 使用的格式。</p>')
    s6_parts.append('</div>')
    sections.append('\n'.join(s6_parts))

    # ===== Part 7: Decision table =====
    s7 = """
    <div class="section-card">
    <h2>7. Element Sparse vs Group Sparse — 对比决策</h2>
    <table class="qsnr-table">
    <tr><th>维度</th><th>Element Sparse</th><th>Group Sparse</th></tr>
    <tr><td>精度提升机制</td><td>隔离极端离群点，避免 amax 被拉大</td><td>整组升级格式，提升组内所有值的精度</td></tr>
    <tr><td>最佳 ratio</td><td>小 ratio (0.02-0.10)</td><td>中 ratio (0.2-0.5)</td></tr>
    <tr><td>Scale 开销</td><td>翻倍（每组 2 个 amax）</td><td>不变（同数量 amax，不同格式）</td></tr>
    <tr><td>硬件开销</td><td>高（per-element mask 索引）</td><td>低（组内一致，无逐元素索引）</td></tr>
    <tr><td>per_tensor 受益</td><td>大（1 amax → 2 amax，隔离效果显著）</td><td style="background:#ffeaa7">退化（只有 1 个 group → 全部升级为 group_format）</td></tr>
    <tr><td>per_channel 受益</td><td>大（channel 内隔离 + channel 级 scale）</td><td style="background:#ffeaa7">弱（仅部分 channel 升级，含 outlier 的 L channel 仍受损）</td></tr>
    <tr><td>per_block 受益</td><td>小（block 内已有隔离），ratio 存在阶梯效应</td><td>中（重要 block 整体升级），随 ratio 线性增长</td></tr>
    <tr><td>bank 受益</td><td>中（bank 内隔离）</td><td style="background:#ffeaa7">弱（bank 数少，仅少量 bank 升级）</td></tr>
    <tr><td>推荐场景</td><td>LLM 激活、少量极端 outlier</td><td>CNN 权重、channel 间动态范围差异大</td></tr>
    </table>
    <div class="note">
    <strong>关键洞察</strong>：Element Sparse 在本测试中全面优于 Group Sparse，原因是测试 tensor 的 outlier 分布模式——
    每个含有 outlier 的 channel/block 内只有 1-2 个极端值。Element sparse 可以精确隔离这些元素，
    而 Group sparse 要么整组升级（浪费精度），要么不升级（outlier 继续碾压）。
    Group sparse 的优势场景是：<strong>某些 group 整体动态范围都更大</strong>（如 CNN 不同输出通道的权重），
    而非"group 内有少数极端值"的情况。
    </div>
    </div>
    """
    sections.append(s7)

    # ===== Part 8: QSNR Sweep Plots =====
    import base64
    s8_parts = ['<div class="section-card">', '<h2>8. QSNR Sweep — 大规模参数扫描</h2>']
    s8_parts.append('<p>基于 4096×4096 张量的大规模 QSNR 扫描实验，验证 granularity × sparse 模式在不同数据分布下的表现。</p>')
    s8_parts.append('<p>生成脚本：<code>scripts/qsnr_sweep_plots.py</code></p>')

    for img_file, title, desc in [
        ("qsnr-sweep-outlier-amplitude.png",
         "QSNR vs Outlier Amplitude",
         "Gaussian(0,1) 基底 + 0.5% outlier，outlier 幅度从 1× 到 50× 基底标准差。展示 element sparse 和 group sparse 如何随 outlier 增大而保持 QSNR。"),
        ("qsnr-sweep-variance.png",
         "QSNR vs Base Variance",
         "Gaussian(0,σ²) 基底 + ±50 固定 outlier，σ 从 1 到 10。展示当基底方差增大时，sparse 模式的效果变化。"),
    ]:
        img_path = os.path.join(OUT_DIR, img_file)
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
            s8_parts.append(f'<h3>{title}</h3>')
            s8_parts.append(f'<p>{desc}</p>')
            s8_parts.append(f'<img class="embed-img" src="data:image/png;base64,{img_b64}" alt="{title}">')
        else:
            s8_parts.append(f'<h3>{title}</h3>')
            s8_parts.append(f'<p class="note">图片未找到: {img_path}。请先运行 <code>python scripts/qsnr_sweep_plots.py</code> 生成。</p>')

    s8_parts.append('</div>')
    sections.append('\n'.join(s8_parts))

    # ===== Assemble full HTML =====
    nav_links = """
    <div class="nav">
        <a href="#s0">术语</a>
        <a href="#s1">Tensor</a>
        <a href="#s2">基础粒度</a>
        <a href="#s3">Element Sparse</a>
        <a href="#s4">Group Sparse</a>
        <a href="#s5">误差热点</a>
        <a href="#s6">汇总</a>
        <a href="#s7">决策</a>
        <a href="#s8">QSNR Sweep</a>
    </div>
    """

    # Add section IDs — robust approach: find first <div class="section-card"> in each section
    section_ids = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    for i, sid in enumerate(section_ids):
        if i < len(sections):
            sections[i] = sections[i].replace(
                '<div class="section-card">',
                f'<div class="section-card" id="{sid}">',
                1
            )

    body = '\n'.join(sections)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Granularity × Sparse 可视化分析 — microxcaling</title>
<style>{css}</style>
</head>
<body>
<h1>Granularity × Sparse 可视化分析</h1>
<p>基于 int4 格式，4 种粒度 × 2 种 sparse 模式的完整量化分析。所有数值由库内 API 实际计算，可复现。</p>
<p>生成脚本：<code>scripts/granularity_sparse_analysis.py</code> &nbsp;|&nbsp; 种子：x1=42, W=43</p>
{nav_links}
{body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Granularity group coloring helper
# ---------------------------------------------------------------------------
def _gran_group_colors(tensor, gran, base=False):
    """Return list of (row_slice, col_slice, color) for group background."""
    colors = [
        "#d5c8f0", "#f0e8d5", "#d5f0d8", "#f0d5e8",
        "#f0d5d5", "#d5e8f0", "#f0f0d5", "#e8d5f0",
    ]
    cells = []

    if gran.mode == GranularityMode.PER_TENSOR:
        if tensor.dim() == 2:
            cells.append((slice(0, tensor.shape[0]), slice(0, tensor.shape[1]), colors[0]))
        else:
            cells.append((slice(0, tensor.shape[1]), slice(0, tensor.shape[2]), colors[0]))

    elif gran.mode == GranularityMode.PER_CHANNEL:
        if tensor.dim() == 2:
            for i in range(tensor.shape[0]):
                cells.append((slice(i, i+1), slice(0, tensor.shape[1]), colors[i % len(colors)]))
        else:
            for i in range(tensor.shape[1]):
                cells.append((slice(i, i+1), slice(0, tensor.shape[2]), colors[i % len(colors)]))

    elif gran.mode == GranularityMode.PER_BLOCK:
        bs = gran.block_size
        if tensor.dim() == 2:
            n_blocks = tensor.shape[1] // bs
            for r in range(tensor.shape[0]):
                for b in range(n_blocks):
                    c_start = b * bs
                    idx = r * n_blocks + b
                    cells.append((slice(r, r+1), slice(c_start, c_start+bs), colors[idx % len(colors)]))
        else:
            n_blocks = tensor.shape[2] // bs
            for r in range(tensor.shape[1]):
                for b in range(n_blocks):
                    c_start = b * bs
                    idx = r * n_blocks + b
                    cells.append((slice(r, r+1), slice(c_start, c_start+bs), colors[idx % len(colors)]))

    elif gran.mode == GranularityMode.BANK:
        bs = gran.bank_size
        n_banks = (tensor.shape[1] if tensor.dim() == 2 else tensor.shape[2]) // bs
        for b in range(n_banks):
            c_start = b * bs
            if tensor.dim() == 2:
                cells.append((slice(0, tensor.shape[0]), slice(c_start, c_start+bs), colors[b % len(colors)]))
            else:
                cells.append((slice(0, tensor.shape[1]), slice(c_start, c_start+bs), colors[b % len(colors)]))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Computing quantization results...")
    x1, W, y_fp32, results = compute_all()

    print("Building HTML...")
    html = build_html(x1, W, y_fp32, results)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Output: {OUT_FILE}")
    print(f"Size: {os.path.getsize(OUT_FILE) / 1024:.1f} KB")

    # Print summary
    print("\n=== QSNR Summary ===")
    for gran_name, gran_x1, gran_w in GRANULARITIES:
        br = results["base"][gran_name]
        print(f"\n{gran_name}:")
        print(f"  base: x1={br['x1_qsnr']:.1f}, W={br['W_qsnr']:.1f}, y={br['y_qsnr']:.1f} dB")
        for ratio in OUTLIER_RATIOS:
            key = f"{gran_name}_ratio{ratio}"
            er = results["element_sparse"][key]
            if "error" not in er:
                print(f"  elem_sparse(r={ratio}): y={er['y_qsnr']:.1f} dB (Δ={er['y_qsnr']-br['y_qsnr']:+.1f})")
        for ratio in GROUP_RATIOS:
            key = f"{gran_name}_ratio{ratio}"
            gr = results["group_sparse"][key]
            if "error" not in gr:
                print(f"  group_sparse(r={ratio}): y={gr['y_qsnr']:.1f} dB (Δ={gr['y_qsnr']-br['y_qsnr']:+.1f})")


if __name__ == "__main__":
    main()
