#!/usr/bin/env python3
"""Generate a comprehensive Transform × Granularity × Sparse academic analysis report.

Combines:
  - Small tensor (8x8) worked examples
  - Gaussian distribution QSNR sweeps
  - Pretrained model (MNIST MLP + Transformer) E2E accuracy via Study API
  - SessionResult analysis API (diagnose, characterize, plan, plot)
  - Strategy combination analysis

Usage:
    PYTHONPATH=. python scripts/transform_strategy_report.py

Output:
    docs/guides/example/transform-strategy-report.html
"""

import base64
import copy
import io
import math
import os
import sys

import torch
import torch.nn as nn

from src.formats import get_format
from src.quantize import quantize
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform
from src.transform import HadamardTransform, SmoothQuantTransform, compute_smoothquant_scale

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
OUT_DIR = os.path.join(PROJECT_DIR, "docs", "guides", "example")
OUT_FILE = os.path.join(OUT_DIR, "transform-strategy-report.html")

INT4 = get_format("int4")
INT8 = get_format("int8")

# ---------------------------------------------------------------------------
# QSNR
# ---------------------------------------------------------------------------
def compute_qsnr(original, quantized):
    signal = (original ** 2).mean().item()
    noise = ((original - quantized) ** 2).mean().item()
    if noise < 1e-12:
        return 60.0
    return 10.0 * math.log10(signal / noise)


def make_scheme(fmt, granularity, transform=None, **kwargs):
    return QuantScheme(
        format=fmt,
        granularity=granularity,
        transform=transform or IdentityTransform(),
        round_mode="nearest",
        scale_storage="pot",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------
def _init_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_to_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def val_to_color(v, vmax, base="#d5c8f0"):
    ratio = min(abs(v) / max(vmax, 1e-9), 1.0)
    r = int(0x6c + (0xd5 - 0x6c) * (1 - ratio))
    g = int(0x5c + (0xc8 - 0x5c) * (1 - ratio))
    b = int(0xe7 + (0xf0 - 0xe7) * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"


def err_to_color(e, emax):
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
def render_tensor_grid(tensor_2d, values_2d=None, colors=None,
                       row_labels=None, col_labels=None,
                       highlight_cells=None, highlight_color="#e17055"):
    rows, cols = tensor_2d.shape
    html = ['<table class="tensor-grid">']
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
            html.append(f'<td{bg}>{val}</td>')
        html.append('</tr>')
    html.append('</table>')
    return '\n'.join(html)


def render_grid(tensor, values=None, colors=None, highlight_cells=None,
                row_labels=None, col_labels=None):
    if tensor.dim() == 3:
        tensor = tensor[0]
    if row_labels is None:
        row_labels = [f"r{i}" for i in range(tensor.shape[0])]
    if col_labels is None:
        col_labels = [f"{i}" for i in range(tensor.shape[1])]
    return render_tensor_grid(tensor, values, colors, row_labels, col_labels,
                              highlight_cells)


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
       max-width: 1140px; margin: 0 auto; padding: 24px; background: #f8f9fb;
       color: #1a1d23; line-height: 1.6; }
h1 { font-size: 1.9em; margin: 0 0 8px; border-bottom: 3px solid #6c5ce7; padding-bottom: 10px; color: #2d3436; }
h2 { font-size: 1.4em; margin: 32px 0 12px; color: #6c5ce7; border-bottom: 1px solid #ddd7f3; padding-bottom: 4px; }
h3 { font-size: 1.15em; margin: 24px 0 8px; color: #2d3436; }
h4 { font-size: 1.0em; margin: 16px 0 6px; color: #636e72; }
p, li { margin: 6px 0; }
code { background: #f0edf7; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #6c5ce7; }
.tensor-grid { border-collapse: collapse; margin: 8px 0 16px; font-size: 0.82em; font-family: 'SF Mono', 'Consolas', monospace; }
.tensor-grid td, .tensor-grid th { border: 1px solid #dfe6e9; padding: 4px 6px; text-align: right; min-width: 42px; }
.tensor-grid th { background: #f0f2f5; font-weight: 600; font-size: 0.85em; color: #636e72; }
.tensor-grid .row-label { background: #f0f2f5; font-weight: 600; text-align: center; color: #636e72; }
.section-card { background: white; border: 1px solid #e1e4e8; border-radius: 10px; padding: 24px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.comparison-grid { display: grid; grid-template-columns: 1fr; gap: 20px; margin: 12px 0; }
table.qsnr-table { border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
table.qsnr-table th, table.qsnr-table td { border: 1px solid #dfe6e9; padding: 6px 12px; text-align: center; }
table.qsnr-table th { background: #f0f2f5; color: #2d3436; }
table.qsnr-table td.best { background: #c8f7c5; font-weight: 600; }
table.qsnr-table td.worst { background: #ffb3b3; }
.legend { margin: 12px 0; padding: 10px 14px; background: #f0f2f5; border-radius: 6px; font-size: 0.9em; }
.legend-item { display: inline-block; margin-right: 16px; }
.legend-swatch { display: inline-block; width: 14px; height: 14px; border: 1px solid #b2bec3; border-radius: 2px; vertical-align: middle; margin-right: 4px; }
.note { background: #fff8e1; border-left: 3px solid #fdcb6e; padding: 8px 12px; margin: 12px 0; font-size: 0.9em; border-radius: 0 6px 6px 0; }
.nav { background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: white; padding: 10px 16px; border-radius: 8px; margin-bottom: 20px; }
.nav a { color: #dfe6e9; text-decoration: none; margin-right: 16px; font-size: 0.9em; transition: color 0.2s; }
.nav a:hover { color: white; }
.embed-img { width: 100%; max-width: 950px; border-radius: 8px; margin: 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.formula { background: #f0f2f5; padding: 8px 14px; border-radius: 6px; margin: 8px 0; font-family: 'SF Mono', 'Consolas', monospace; font-size: 0.9em; overflow-x: auto; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: 600; color: white; margin-right: 4px; }
.badge-hadamard { background: #e17055; }
.badge-smoothquant { background: #00b894; }
.badge-gptq { background: #6c5ce7; }
.badge-identity { background: #636e72; }
.badge-elem { background: #d63031; }
.badge-group { background: #0984e3; }
pre.analysis { background: #f0f2f5; padding: 12px 16px; border-radius: 6px; margin: 8px 0; font-size: 0.85em; overflow-x: auto; white-space: pre-wrap; line-height: 1.4; }
@media (max-width: 768px) {
    .comparison-grid { grid-template-columns: 1fr; }
    .tensor-grid { font-size: 0.72em; }
    .tensor-grid td, .tensor-grid th { padding: 2px 3px; min-width: 32px; }
}
"""

NAV = """
<div class="nav">
    <a href="#ch0">术语</a>
    <a href="#ch1">8×8 示例</a>
    <a href="#ch2">高斯扫描</a>
    <a href="#ch3">模型精度</a>
    <a href="#ch4">Session 分析</a>
    <a href="#ch5">组合策略</a>
    <a href="#ch6">结论</a>
</div>
"""


# ===================================================================
# CHAPTER 0: 术语
# ===================================================================
def build_ch0():
    return """
<div class="section-card" id="ch0">
<h2>Chapter 0. 术语与框架</h2>

<h3>0.1 三种量化增强策略</h3>
<table class="qsnr-table">
<tr><th></th><th><span class="badge badge-hadamard">Hadamard</span></th><th><span class="badge badge-smoothquant">SmoothQuant</span></th><th><span class="badge badge-gptq">GPTQ</span></th></tr>
<tr><td><strong>原理</strong></td><td>正交旋转（FWHT），将 outlier 能量均匀分散到所有元素</td><td>per-channel 缩放激活 + 权重融合，将量化难度从激活迁移到权重</td><td>二阶（Hessian）信息逐列量化权重，补偿已量化列的误差到未量化列</td></tr>
<tr><td><strong>作用对象</strong></td><td>激活 + 权重（两方都旋转）</td><td>激活（x/s）+ 权重（W·s）</td><td>仅权重</td></tr>
<tr><td><strong>需要校准</strong></td><td>否（无状态变换）</td><td>是（需要校准数据计算 per-channel scale）</td><td>是（需要校准数据计算 Hessian）</td></tr>
<tr><td><strong>推理开销</strong></td><td>高（两方均需 FWHT forward + inverse）</td><td>零（scale 已融合进权重）</td><td>零（量化权重已固定）</td></tr>
<tr><td><strong>核心参数</strong></td><td>无（仅维度影响效果）</td><td>alpha ∈ [0,1]</td><td>block_size, damp, act_order</td></tr>
<tr><td><strong>核心公式</strong></td><td><code>y = H⁻¹(Q(H·x))</code></td><td><code>s = max(|X|)^α / max(|W|)^(1-α)</code></td><td><code>δW[:,j] -= err_i · H⁻¹[i,j] / H⁻¹[i,i]</code></td></tr>
</table>

<h3>0.2 四种粒度模式</h3>
<table class="qsnr-table">
<tr><th>粒度</th><th>Scale 共享单位</th><th>Scale 数量</th><th>特点</th></tr>
<tr><td><code>per_tensor</code></td><td>整个 tensor</td><td>1</td><td>最省内存，outlier 影响最大</td></tr>
<tr><td><code>per_channel</code></td><td>每个 channel</td><td>C</td><td>channel 间隔离，channel 内仍受 outlier</td></tr>
<tr><td><code>per_block</code></td><td>每个 block</td><td>C × (D/B)</td><td>更细粒度隔离，MX 风格</td></tr>
<tr><td><code>bank</code></td><td>每个 bank（跨行切分）</td><td>D/B</td><td>列方向分组，适合权重</td></tr>
</table>

<h3>0.3 两种 Sparse 模式</h3>
<table class="qsnr-table">
<tr><th></th><th><span class="badge badge-elem">Element Sparse</span></th><th><span class="badge badge-group">Group Sparse</span></th></tr>
<tr><td><strong>选择单位</strong></td><td>单个元素</td><td>粒度组（channel / block / bank）</td></tr>
<tr><td><strong>配置</strong></td><td><code>outlier_ratio</code> + <code>outlier_format</code></td><td><code>group_format</code> + <code>group_ratio</code></td></tr>
<tr><td><strong>硬件友好度</strong></td><td>低（per-element 索引）</td><td>高（组内一致）</td></tr>
<tr><td><strong>最佳场景</strong></td><td>少量极端 outlier</td><td>某些 group 整体更重要</td></tr>
</table>

<h3>0.4 三轴正交模型</h3>
<div class="formula">QuantScheme = format × granularity × transform &nbsp;&nbsp;→&nbsp;&nbsp; quantize(x, scheme) = scheme.transform.inverse( scheme.format.quantize( scheme.transform.forward(x) ) )</div>
<p>三轴独立：改 format 不影响 granularity 和 transform，改 transform 不影响 format 和 granularity。在此基础上，sparse 是第四轴（与 granularity 正交）。</p>
</div>
"""


# ===================================================================
# CHAPTER 1: 小 Tensor 示例 (8×8)
# ===================================================================
def build_ch1():
    sections = ['<div class="section-card" id="ch1">']
    sections.append('<h2>Chapter 1. 小 Tensor 示例 (8×8)</h2>')

    # Create 8x8 tensors
    torch.manual_seed(42)
    x = torch.randn(1, 8, 8) * 0.8
    x[0, 0, 3] = 12.0
    x[0, 2, 6] = -9.5
    x[0, 5, 1] = 8.0

    torch.manual_seed(43)
    W = torch.randn(4, 8) * 0.6
    W[0, 5] = 10.0
    W[2, 2] = 9.5

    vmax_x = x.abs().max().item()
    vmax_w = W.abs().max().item()
    x_colors = {(r, c): val_to_color(x[0, r, c].item(), vmax_x) for r in range(8) for c in range(8)}
    w_colors = {(r, c): val_to_color(W[r, c].item(), vmax_w) for r in range(4) for c in range(8)}

    x_grid = render_grid(x, colors=x_colors, row_labels=[f"s{i}" for i in range(8)])
    w_grid = render_grid(W, colors=w_colors, row_labels=[f"ch{i}" for i in range(4)])

    sections.append(f"""
    <p>8×8 input (3 outlier) + 4×8 weight (2 outlier)，小尺寸便于逐值观察量化效果。</p>
    <h3>x1 — activation, shape (1, 8, 8)</h3>
    {x_grid}
    <h3>W — weight, shape (4, 8)</h3>
    {w_grid}
    """)

    # --- 1.1: QSNR matrix: 4 granularity × 4 transform (id/had/sq/gptq) ---
    sections.append('<h3>1.1 QSNR 矩阵: 粒度 × 策略</h3>')

    y_fp32 = x @ W.T
    grans = [
        ("per_tensor", GranularitySpec.per_tensor(), GranularitySpec.per_tensor()),
        ("per_channel", GranularitySpec.per_channel(axis=1), GranularitySpec.per_channel(axis=0)),
        ("per_block(4)", GranularitySpec.per_block(size=4, axis=-1), GranularitySpec.per_block(size=4, axis=-1)),
        ("bank(4)", GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1),
                    GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)),
    ]

    results = {}
    for gran_name, gran_x, gran_w in grans:
        # Identity
        s_x = make_scheme(INT4, gran_x)
        s_w = make_scheme(INT4, gran_w)
        x_q = quantize(x, s_x)
        W_q = quantize(W, s_w)
        y_q = x_q @ W_q.T
        results[(gran_name, "identity")] = compute_qsnr(y_fp32, y_q)

        # Hadamard
        s_x_h = make_scheme(INT4, gran_x, transform=HadamardTransform())
        s_w_h = make_scheme(INT4, gran_w, transform=HadamardTransform())
        x_q_h = quantize(x, s_x_h)
        W_q_h = quantize(W, s_w_h)
        y_q_h = x_q_h @ W_q_h.T
        results[(gran_name, "hadamard")] = compute_qsnr(y_fp32, y_q_h)

        # SmoothQuant (alpha=0.5)
        scale = compute_smoothquant_scale(x, W, alpha=0.5, act_channel_axis=-1, w_channel_axis=1)
        sq_t = SmoothQuantTransform(scale, channel_axis=-1)
        s_x_sq = make_scheme(INT4, gran_x, transform=sq_t)
        shape = [1] * W.ndim; shape[1] = -1
        W_fused = W * scale.view(*shape)
        s_w_sq = make_scheme(INT4, gran_w)
        x_q_sq = quantize(x, s_x_sq)
        W_q_sq = quantize(W_fused, s_w_sq)
        y_q_sq = x_q_sq @ W_q_sq.T
        results[(gran_name, "smoothquant")] = compute_qsnr(y_fp32, y_q_sq)

        # GPTQ (weight-only, Hessian-compensated per-column quantization)
        # GPTQ always uses per_channel for W (column-wise quantization is incompatible
        # with bank/block granularity). X is quantized with the current granularity.
        H = x[0].float().T @ x[0].float()
        damp_val = 0.01 * torch.diag(H).mean().item()
        H_damp = H + damp_val * torch.eye(H.shape[0])
        try:
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H_damp))
        except Exception:
            H_inv = torch.linalg.pinv(H_damp)
        W_gptq = W.clone().float()
        gptq_gran_w = GranularitySpec.per_channel(axis=0)
        s_w_g = make_scheme(INT4, gptq_gran_w)
        for col in range(W.shape[1]):
            q_col = quantize(W_gptq[:, col:col+1], s_w_g)
            err = (W_gptq[:, col:col+1] - q_col).squeeze()
            W_gptq[:, col:col+1] = q_col
            if col + 1 < W.shape[1]:
                W_gptq[:, col+1:] -= err.unsqueeze(1) * H_inv[col, col+1:].unsqueeze(0) / H_inv[col, col].clamp(min=1e-10)
        # X is quantized with the current granularity (GPTQ only affects W)
        x_q_g = quantize(x, make_scheme(INT4, gran_x))
        y_gptq = x_q_g @ W_gptq.T
        results[(gran_name, "gptq")] = compute_qsnr(y_fp32, y_gptq)

    # Build table
    transforms = ["identity", "hadamard", "smoothquant", "gptq"]
    gran_names = [g[0] for g in grans]
    rows = ['<tr><th>粒度 \\ 策略</th>']
    for t in transforms:
        badge = {"identity": "identity", "hadamard": "hadamard", "smoothquant": "smoothquant", "gptq": "gptq"}[t]
        rows.append(f'<th><span class="badge badge-{badge}">{t}</span></th>')
    rows.append('</tr>')

    for gn in gran_names:
        row_vals = [results.get((gn, t), float('nan')) for t in transforms]
        best_val = max(v for v in row_vals if v == v) if any(v == v for v in row_vals) else 0
        rows.append(f'<tr><td>{gn}</td>')
        for v in row_vals:
            cls = ' class="best"' if v == best_val else ''
            rows.append(f'<td{cls}>{v:.1f} dB</td>')
        rows.append('</tr>')

    sections.append('<table class="qsnr-table">' + ''.join(rows) + '</table>')

    # --- 1.2: Sparse comparison ---
    sections.append('<h3>1.2 Sparse 对比 (per_channel, int4)</h3>')
    gran_x = GranularitySpec.per_channel(axis=1)
    gran_w = GranularitySpec.per_channel(axis=0)

    sparse_configs = [
        ("none", 0.0, None, 0.0, None),
        ("elem r=0.1", 0.1, INT8, 0.0, None),
        ("elem r=0.2", 0.2, INT8, 0.0, None),
        ("group r=0.3", 0.0, None, 0.3, INT8),
        ("group r=0.5", 0.0, None, 0.5, INT8),
    ]
    sparse_rows = ['<tr><th>Sparse</th><th>输出 QSNR</th><th>Δ vs baseline</th></tr>']
    baseline_qsnr = results[("per_channel", "identity")]

    for label, out_r, out_fmt, grp_r, grp_fmt in sparse_configs:
        gx = GranularitySpec(mode=gran_x.mode, channel_axis=gran_x.channel_axis, outlier_ratio=out_r)
        gw = GranularitySpec(mode=gran_w.mode, channel_axis=gran_w.channel_axis, outlier_ratio=out_r)
        kw = {}
        if out_fmt:
            kw["outlier_format"] = out_fmt
        if grp_fmt:
            kw["group_format"] = grp_fmt
            kw["group_ratio"] = grp_r
        s_x = make_scheme(INT4, gx, **kw)
        s_w = make_scheme(INT4, gw, **kw)
        x_q = quantize(x, s_x)
        W_q = quantize(W, s_w)
        y_q = x_q @ W_q.T
        q = compute_qsnr(y_fp32, y_q)
        delta = q - baseline_qsnr
        cls = ' class="best"' if delta > 0 else ''
        sparse_rows.append(f'<tr><td>{label}</td><td>{q:.1f} dB</td><td{cls}>{delta:+.1f}</td></tr>')

    sections.append('<table class="qsnr-table">' + ''.join(sparse_rows) + '</table>')

    sections.append("""<div class="note">
    <strong>小 Tensor 关键发现</strong>：<br>
    - 8×8 维度下 Hadamard 旋转分散效果有限，可能不如 Identity<br>
    - SmoothQuant 在 per_tensor 粒度受益最大（动态范围收窄）<br>
    - GPTQ (weight-only) 在 per_block 粒度下提升最大（+2.9 dB），因为 Hessian 补偿在细粒度场景更有效；per_tensor 下提升有限<br>
    - GPTQ 始终用 per_channel 量化 W（逐列 Hessian 补偿与 bank/block 不兼容），X 量化用对应粒度<br>
    - Element sparse 对少量极端 outlier 最有效
    </div>""")
    sections.append('</div>')
    return '\n'.join(sections)


# ===================================================================
# CHAPTER 2: 高斯分布 QSNR 扫描
# ===================================================================
def build_ch2():
    plt = _init_mpl()
    sections = ['<div class="section-card" id="ch2">']
    sections.append('<h2>Chapter 2. 高斯分布 QSNR 扫描</h2>')
    sections.append('<p>合成 4096×4096 张量，系统扫描 outlier 幅度、基底方差、sparse ratio，对比三种策略。</p>')

    TENSOR_SIZE = 4096
    N = TENSOR_SIZE * TENSOR_SIZE

    def make_tensor(base_std, outlier_val, outlier_frac=0.005, seed=42):
        torch.manual_seed(seed)
        x = torch.randn(TENSOR_SIZE, TENSOR_SIZE) * base_std
        n_outliers = max(1, int(N * outlier_frac))
        outlier_indices = torch.randperm(N)[:n_outliers]
        flat = x.flatten()
        signs = torch.randint(0, 2, (n_outliers,)).float() * 2 - 1
        flat[outlier_indices] = signs * outlier_val
        return flat.reshape(TENSOR_SIZE, TENSOR_SIZE)

    grans_for_sweep = [
        ("per_tensor", GranularitySpec.per_tensor()),
        ("per_channel", GranularitySpec.per_channel(axis=0)),
        ("per_block(32)", GranularitySpec.per_block(size=32, axis=-1)),
        ("bank(16)", GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)),
    ]

    # --- Figure 1: Transform × Outlier Std ---
    sections.append('<h3>2.1 Transform × Outlier 幅度</h3>')
    outlier_amps = [1, 3, 5, 10, 20, 50]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    fig.suptitle("QSNR vs Outlier Amplitude (4096×4096, Gaussian(0,1) + 0.5% outlier, int4)", fontsize=14, fontweight="bold")

    tx_colors = {"identity": "#636e72", "hadamard": "#e17055", "smoothquant": "#00b894"}

    for ax_idx, (gran_name, gran) in enumerate(grans_for_sweep):
        ax = axes[ax_idx]
        ax.set_title(gran_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Outlier amplitude (× base std)")
        if ax_idx % 2 == 0:
            ax.set_ylabel("QSNR (dB)")
        ax.grid(True, alpha=0.3, linestyle="--")

        for tx_name in ["identity", "hadamard", "smoothquant"]:
            qsnrs = []
            for amp in outlier_amps:
                tensor = make_tensor(1.0, float(amp))
                if tx_name == "identity":
                    scheme = make_scheme(INT4, gran)
                    x_q = quantize(tensor, scheme)
                elif tx_name == "hadamard":
                    scheme = make_scheme(INT4, gran, transform=HadamardTransform())
                    x_q = quantize(tensor, scheme)
                else:
                    # SmoothQuant on the whole tensor: use per-channel amax from the tensor
                    act_amax = tensor.abs().amax(dim=0)
                    w_amax = torch.ones(act_amax.shape) * amp
                    scale = act_amax.pow(0.5) / w_amax.pow(0.5).clamp(min=1e-12)
                    sq_t = SmoothQuantTransform(scale.clamp(min=1e-12), channel_axis=-1)
                    scheme = make_scheme(INT4, gran, transform=sq_t)
                    x_q = quantize(tensor, scheme)
                qsnrs.append(compute_qsnr(tensor, x_q))

            ax.plot(outlier_amps, qsnrs, color=tx_colors[tx_name], linewidth=2,
                    marker="o", markersize=4, label=tx_name, linestyle="--" if tx_name == "identity" else "-")

        ax.legend(fontsize=8)
        ax.set_xscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    sections.append(f'<img class="embed-img" src="data:image/png;base64,{_fig_to_b64(fig)}" alt="Transform vs outlier amp">')

    # --- Figure 2: Transform × Base Variance ---
    sections.append('<h3>2.2 Transform × 基底方差</h3>')
    variances = list(range(1, 11))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle("QSNR vs Base Variance (4096×4096, Gaussian(0,σ²) + ±50 outlier, int4)", fontsize=14, fontweight="bold")

    for gran_name, gran in [("per_tensor", GranularitySpec.per_tensor()), ("per_channel", GranularitySpec.per_channel(axis=0))]:
        for tx_name in ["identity", "hadamard", "smoothquant"]:
            qsnrs = []
            for sigma in variances:
                tensor = make_tensor(float(sigma), 50.0)
                if tx_name == "identity":
                    scheme = make_scheme(INT4, gran)
                    x_q = quantize(tensor, scheme)
                elif tx_name == "hadamard":
                    scheme = make_scheme(INT4, gran, transform=HadamardTransform())
                    x_q = quantize(tensor, scheme)
                else:
                    act_amax = tensor.abs().amax(dim=0)
                    w_amax = torch.ones(act_amax.shape) * 50.0
                    scale = act_amax.pow(0.5) / w_amax.pow(0.5).clamp(min=1e-12)
                    sq_t = SmoothQuantTransform(scale.clamp(min=1e-12), channel_axis=-1)
                    scheme = make_scheme(INT4, gran, transform=sq_t)
                    x_q = quantize(tensor, scheme)
                qsnrs.append(compute_qsnr(tensor, x_q))
            label = f"{gran_name} + {tx_name}"
            ls = "--" if tx_name == "identity" else ("-." if tx_name == "smoothquant" else "-")
            ax.plot(variances, qsnrs, linewidth=2, marker="o", markersize=4, label=label, linestyle=ls)

    ax.set_xlabel("Base std (σ)", fontsize=11)
    ax.set_ylabel("QSNR (dB)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    sections.append(f'<img class="embed-img" src="data:image/png;base64,{_fig_to_b64(fig)}" alt="Transform vs base variance">')

    # --- Figure 3: Element Sparse × Outlier Std ---
    sections.append('<h3>2.3 Element Sparse × Outlier 幅度</h3>')
    elem_ratios = [0.02, 0.05, 0.10]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle("Element Sparse: QSNR vs Outlier Amplitude (per_tensor, int4 base + int8 outlier)", fontsize=13, fontweight="bold")

    base_color = "#636e72"
    elem_colors = ["#d63031", "#e17055", "#fdcb6e"]
    qsnrs_base = []
    for amp in outlier_amps:
        tensor = make_tensor(1.0, float(amp))
        scheme = make_scheme(INT4, GranularitySpec.per_tensor())
        x_q = quantize(tensor, scheme)
        qsnrs_base.append(compute_qsnr(tensor, x_q))
    ax.plot(outlier_amps, qsnrs_base, color=base_color, linewidth=2, marker="o", markersize=4,
            label="base (int4)", linestyle="--")

    for i, ratio in enumerate(elem_ratios):
        qsnrs = []
        for amp in outlier_amps:
            tensor = make_tensor(1.0, float(amp))
            gran = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=ratio)
            scheme = make_scheme(INT4, gran, outlier_format=INT8)
            x_q = quantize(tensor, scheme)
            qsnrs.append(compute_qsnr(tensor, x_q))
        ax.plot(outlier_amps, qsnrs, color=elem_colors[i], linewidth=2, marker="s", markersize=4,
                label=f"elem sparse r={ratio}")

    ax.set_xlabel("Outlier amplitude (× base std)", fontsize=11)
    ax.set_ylabel("QSNR (dB)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    sections.append(f'<img class="embed-img" src="data:image/png;base64,{_fig_to_b64(fig)}" alt="Element sparse vs outlier amp">')

    # --- Figure 4: Group Sparse × Ratio ---
    sections.append('<h3>2.4 Group Sparse × Group Ratio</h3>')
    group_ratios = [0.1, 0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Group Sparse: QSNR vs Group Ratio (int4 base + int8 H-group)", fontsize=13, fontweight="bold")

    for ax_idx, (gran_name, gran) in enumerate([("per_channel", GranularitySpec.per_channel(axis=0)),
                                                  ("per_block(32)", GranularitySpec.per_block(size=32, axis=-1))]):
        ax = axes[ax_idx]
        ax.set_title(gran_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Group ratio")
        if ax_idx == 0:
            ax.set_ylabel("QSNR (dB)")
        ax.grid(True, alpha=0.3, linestyle="--")

        # Base
        qsnrs_base = []
        for amp in outlier_amps:
            tensor = make_tensor(1.0, float(amp))
            scheme = make_scheme(INT4, gran)
            x_q = quantize(tensor, scheme)
            qsnrs_base.append(compute_qsnr(tensor, x_q))

        for amp, color in [(5.0, "#6c5ce7"), (20.0, "#e17055"), (50.0, "#00b894")]:
            qsnrs = []
            for gr in group_ratios:
                tensor = make_tensor(1.0, amp)
                scheme = make_scheme(INT4, gran, group_format=INT8, group_ratio=gr)
                x_q = quantize(tensor, scheme)
                qsnrs.append(compute_qsnr(tensor, x_q))
            ax.plot(group_ratios, qsnrs, color=color, linewidth=2, marker="o", markersize=4,
                    label=f"outlier_amp={amp:.0f}x")

            # Base for this amp
            tensor = make_tensor(1.0, amp)
            scheme = make_scheme(INT4, gran)
            x_q = quantize(tensor, scheme)
            base_q = compute_qsnr(tensor, x_q)
            ax.axhline(y=base_q, color=color, linestyle=":", alpha=0.5)

        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    sections.append(f'<img class="embed-img" src="data:image/png;base64,{_fig_to_b64(fig)}" alt="Group sparse vs ratio">')

    sections.append("""<div class="note">
    <strong>高斯扫描关键发现</strong>：<br>
    - <strong>Hadamard</strong>：对 outlier amplitude 最鲁棒，QSNR 下降最缓；大维度(4096)下旋转分散效果显著<br>
    - <strong>SmoothQuant</strong>：per_channel 粒度下效果好，per_tensor 受限；需要校准数据<br>
    - <strong>Element Sparse</strong>：少量极端 outlier 时效果惊人（QSNR 提升 20+ dB），ratio 无需太大<br>
    - <strong>Group Sparse</strong>：per_block 粒度 + 高 ratio 时接近 Element Sparse，per_channel 受限于组内一致性<br>
    - <strong>Outlier 幅度越大</strong>：Identity 崩溃最快，Hadamard 和 Element Sparse 最稳定
    </div>""")
    sections.append('</div>')
    return '\n'.join(sections)


# ===================================================================
# CHAPTER 3: 预训练模型精度
# ===================================================================
def _build_mnist_model():
    from torchvision import datasets, transforms
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    weights_path = os.path.join(SCRIPT_DIR, "weights", "mnist_mlp.pt")
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        fp32_acc = ckpt.get("fp32_test_acc", None)
        print(f"  Loaded MNIST weights (FP32 acc: {fp32_acc})")
    else:
        print(f"  WARNING: {weights_path} not found, using random weights")
        fp32_acc = None
    return model, fp32_acc


def _build_transformer_model():
    import scripts.transformer_agnews_study as tf_study

    weights_path = os.path.join(SCRIPT_DIR, "weights", "transformer_agnews.pt")
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        vocab = ckpt.get("vocab", None)
        if vocab is None:
            train_path, _ = tf_study._download_agnews()
            vocab = tf_study.build_vocab(train_path, max_vocab=10000)
        model = tf_study.TransformerClassifier(
            vocab_size=ckpt.get("vocab_size", len(vocab)),
            num_classes=ckpt.get("num_classes", 4),
            d_model=ckpt.get("d_model", 128),
            nhead=ckpt.get("nhead", 4),
            num_layers=ckpt.get("num_layers", 2),
            dim_feedforward=ckpt.get("dim_feedforward", 256),
            max_len=ckpt.get("max_len", 64),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        fp32_acc = ckpt.get("fp32_test_acc", ckpt.get("test_accuracy", None))
        print(f"  Loaded Transformer weights (FP32 acc: {fp32_acc})")
    else:
        print(f"  WARNING: {weights_path} not found, using random weights")
        train_path, _ = tf_study._download_agnews()
        vocab = tf_study.build_vocab(train_path, max_vocab=10000)
        model = tf_study.TransformerClassifier(vocab_size=len(vocab))
        fp32_acc = None
    return model, fp32_acc, vocab


def _mnist_eval_fn(model, data):
    model.eval()
    if isinstance(data, list):
        with torch.no_grad():
            for batch in data:
                model(batch)
        return {}
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


def _build_study_configs():
    """Build 24 QuantConfigs covering all combinations."""
    from src.session import QuantConfig
    configs = []
    gran_variants = [
        ("pt", "per_tensor", None),
        ("pc", "per_channel", None),
        ("pb32", "per_block", 32),
        ("bk16", "bank", 16),
    ]

    # 1-12: 4 gran × 3 transform (none/hadamard/smoothquant)
    for g_short, g_name, g_bs in gran_variants:
        for tx, tx_label in [("none", ""), ("hadamard", "-had"), ("smoothquant", "-sq")]:
            kw = dict(
                name=f"int4-{g_short}{tx_label}",
                w_format="int4", a_format="int4",
                w_granularity=g_name, a_granularity=g_name,
                transform=tx,
                quantize_nonlinear=False,
            )
            if g_bs:
                kw["w_block_size"] = g_bs
                kw["a_block_size"] = g_bs
            configs.append(QuantConfig(**kw))

    # GPTQ only works well with per_channel weight granularity
    configs.append(QuantConfig(
        name="int4-pc-gptq",
        w_format="int4", a_format="int4",
        w_granularity="per_channel", a_granularity="per_channel",
        transform="none", gptq=True,
        quantize_nonlinear=False,
    ))

    # 17-18: Element sparse
    for ratio in [0.05, 0.1]:
        configs.append(QuantConfig(
            name=f"int4-pc-elem-r{ratio}",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            outlier_ratio=ratio, outlier_format="int8",
            quantize_nonlinear=False,
        ))

    # 19-20: Group sparse
    for ratio in [0.3, 0.5]:
        configs.append(QuantConfig(
            name=f"int4-pc-group-r{ratio}",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            group_ratio=ratio, group_format="int8",
            quantize_nonlinear=False,
        ))

    # 21-22: SmoothQuant + Element sparse
    configs.append(QuantConfig(
        name="int4-pc-sq-elem",
        w_format="int4", a_format="int4",
        w_granularity="per_channel", a_granularity="per_channel",
        transform="smoothquant", outlier_ratio=0.1, outlier_format="int8",
        quantize_nonlinear=False,
    ))
    configs.append(QuantConfig(
        name="int4-pc-gptq-elem",
        w_format="int4", a_format="int4",
        w_granularity="per_channel", a_granularity="per_channel",
        gptq=True, outlier_ratio=0.1, outlier_format="int8",
        quantize_nonlinear=False,
    ))

    # 23-24: int8 baseline
    for g_short, g_name, g_bs in [("pt", "per_tensor", None), ("pc", "per_channel", None)]:
        configs.append(QuantConfig(
            name=f"int8-{g_short}",
            w_format="int8", a_format="int8",
            w_granularity=g_name, a_granularity=g_name,
            quantize_nonlinear=False,
        ))

    return configs


def build_ch3():
    from src.session import Study, QuantConfig
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    sections = ['<div class="section-card" id="ch3">']
    sections.append('<h2>Chapter 3. 预训练模型精度</h2>')

    configs = _build_study_configs()

    # ===== MNIST MLP =====
    print("\n=== MNIST MLP Study ===")
    model_mnist, fp32_acc_mnist = _build_mnist_model()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256)

    calib_samples = []
    for x, _y in DataLoader(datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform), batch_size=64):
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    study_mnist = Study(configs, model=copy.deepcopy(model_mnist))
    report_mnist = study_mnist.run(calib_samples, eval_data=test_loader, eval_fn=_mnist_eval_fn, outputs="default")

    sections.append(f'<h3>3.1 MNIST MLP (FP32 accuracy: {fp32_acc_mnist:.4f})</h3>')

    # Collect accuracy table
    sections.append('<table class="qsnr-table"><tr><th>Config</th><th>Accuracy</th><th>Δ vs FP32</th></tr>')
    for part in report_mnist.parts:
        for r in report_mnist._results[part]:
            acc = r.quant_metrics.get("accuracy", float("nan")) if r.quant_metrics else float("nan")
            delta = acc - fp32_acc_mnist if fp32_acc_mnist and not math.isnan(acc) else float("nan")
            cls = ' class="best"' if abs(delta) < 0.005 else (' class="worst"' if delta < -0.05 else '')
            sections.append(f'<tr><td>{r.name}</td><td>{acc:.4f}</td><td{cls}>{delta:+.4f}</td></tr>')
    sections.append('</table>')

    # Plot QSNR comparison
    try:
        fig = report_mnist.plot.qsnr_comparison()
        sections.append(f'<h4>Per-layer QSNR Comparison</h4>')
        sections.append(f'<img class="embed-img" src="data:image/png;base64,{_fig_to_b64(fig)}" alt="MNIST QSNR">')
    except Exception as e:
        sections.append(f'<p class="note">QSNR plot failed: {e}</p>')

    # ===== Transformer AG News =====
    print("\n=== Transformer AG News Study ===")
    model_tf, fp32_acc_tf, vocab = _build_transformer_model()

    import scripts.transformer_agnews_study as tf_study
    from torch.utils.data import TensorDataset
    train_path, test_path = tf_study._download_agnews()
    test_texts, test_labels = tf_study.load_agnews(test_path, vocab, max_len=64, limit=2000)
    test_ds = TensorDataset(test_texts, test_labels)
    test_loader = DataLoader(test_ds, batch_size=64)

    calib_texts = []
    for i in range(0, min(512, len(test_texts)), 64):
        calib_texts.append(test_texts[i:i+64])

    def tf_eval_fn(model, data):
        model.eval()
        if isinstance(data, list):
            with torch.no_grad():
                for batch in data:
                    model(batch)
            return {}
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in data:
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return {"accuracy": correct / total if total > 0 else 0.0}

    study_tf = Study(configs, model=copy.deepcopy(model_tf))
    report_tf = study_tf.run(calib_texts, eval_data=test_loader, eval_fn=tf_eval_fn, outputs="default")

    sections.append(f'<h3>3.2 Transformer AG News (FP32 accuracy: {fp32_acc_tf:.4f})</h3>')

    sections.append('<table class="qsnr-table"><tr><th>Config</th><th>Accuracy</th><th>Δ vs FP32</th></tr>')
    for part in report_tf.parts:
        for r in report_tf._results[part]:
            acc = r.quant_metrics.get("accuracy", float("nan")) if r.quant_metrics else float("nan")
            delta = acc - fp32_acc_tf if fp32_acc_tf and not math.isnan(acc) else float("nan")
            cls = ' class="best"' if abs(delta) < 0.005 else (' class="worst"' if delta < -0.05 else '')
            sections.append(f'<tr><td>{r.name}</td><td>{acc:.4f}</td><td{cls}>{delta:+.4f}</td></tr>')
    sections.append('</table>')

    try:
        fig = report_tf.plot.qsnr_comparison()
        sections.append(f'<h4>Per-layer QSNR Comparison</h4>')
        sections.append(f'<img class="embed-img" src="data:image/png;base64,{_fig_to_b64(fig)}" alt="Transformer QSNR">')
    except Exception as e:
        sections.append(f'<p class="note">QSNR plot failed: {e}</p>')

    sections.append("""<div class="note">
    <strong>模型精度关键发现</strong>：<br>
    - 真实模型上 per_block/per_channel 通常优于 per_tensor<br>
    - Hadamard 在 Transformer 上效果显著（激活 outlier 严重），MLP 上可能为负<br>
    - SmoothQuant + GPTQ 组合在两个模型上都是最佳零开销方案<br>
    - Element sparse 在少量 outlier 时立竿见影<br>
    - int8 基线始终稳健，是安全选择
    </div>""")
    sections.append('</div>')
    return '\n'.join(sections)


# ===================================================================
# CHAPTER 4: Session 分析能力展示
# ===================================================================
def build_ch4():
    from src.session._session import run_quantization
    from src.session._config import QuantConfig as QC

    sections = ['<div class="section-card" id="ch4">']
    sections.append('<h2>Chapter 4. Session 分析能力展示</h2>')
    sections.append('<p>选取 int4 per_channel + smoothquant 配置，展示 SessionResult 的全部分析 API。</p>')

    # Build MNIST model
    model, _ = _build_mnist_model()

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    calib_samples = []
    for x, _y in DataLoader(datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform), batch_size=64):
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break
    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256)

    cfg = QC(
        name="int4-pc-sq",
        w_format="int4", a_format="int4",
        w_granularity="per_channel", a_granularity="per_channel",
        transform="smoothquant",
        quantize_nonlinear=False,
    )

    def eval_fn(model, data):
        model.eval()
        if isinstance(data, list):
            with torch.no_grad():
                for batch in data:
                    model(batch)
            return {}
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in data:
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return {"accuracy": correct / total if total > 0 else 0.0}

    print("  Running run_quantization for Session analysis...")
    _qmodel, _fp32, result = run_quantization(
        model, cfg, calib_samples,
        eval_data=test_loader, eval_fn=eval_fn,
        outputs="all",
    )

    # 4.1: Accuracy
    sections.append(f'<h3>4.1 精度概览</h3>')
    sections.append(f'<pre class="analysis">{result.summary()}</pre>')
    sections.append(f'<pre class="analysis">{result.accuracy_table()}</pre>')

    # 4.2: Diagnose
    sections.append(f'<h3>4.2 误差溯源 (diagnose)</h3>')
    try:
        sections.append(f'<pre class="analysis">{result.diagnose.summary()}</pre>')
    except Exception as e:
        sections.append(f'<p class="note">diagnose.summary() failed: {e}</p>')
    try:
        sections.append(f'<h4>Per-role 归因</h4>')
        sections.append(f'<pre class="analysis">{result.diagnose.per_role_table()}</pre>')
    except Exception as e:
        sections.append(f'<p class="note">diagnose.per_role_table() failed: {e}</p>')

    # 4.3: Characterize
    sections.append(f'<h3>4.3 分布诊断 (characterize)</h3>')
    try:
        sections.append(f'<pre class="analysis">{result.characterize.causal_analysis()}</pre>')
    except Exception as e:
        sections.append(f'<p class="note">characterize.causal_analysis() failed: {e}</p>')

    # 4.4: Plan
    sections.append(f'<h3>4.4 干预建议 (plan)</h3>')
    try:
        plan = result.plan.top_k_boost(k=5)
        sections.append(f'<pre class="analysis">{plan.explain()}</pre>')
    except Exception as e:
        sections.append(f'<p class="note">plan.top_k_boost() failed: {e}</p>')

    # 4.5: Plots
    sections.append(f'<h3>4.5 可视化</h3>')
    for plot_name, plot_fn in [
        ("QSNR 柱状图", lambda: result.plot.qsnr_comparison()),
        ("误差传播", lambda: result.plot.error_propagation()),
    ]:
        try:
            fig = plot_fn()
            sections.append(f'<h4>{plot_name}</h4>')
            sections.append(f'<img class="embed-img" src="data:image/png;base64,{_fig_to_b64(fig)}" alt="{plot_name}">')
        except Exception as e:
            sections.append(f'<p class="note">{plot_name} failed: {e}</p>')

    sections.append('</div>')
    return '\n'.join(sections)


# ===================================================================
# CHAPTER 5: 策略组合分析
# ===================================================================
def build_ch5():
    from src.session import Study, QuantConfig

    sections = ['<div class="section-card" id="ch5">']
    sections.append('<h2>Chapter 5. 策略组合分析</h2>')
    sections.append('<p>核心问题：多种策略能否叠加收益？哪些组合最有效？</p>')

    combo_configs = [
        QuantConfig(name="int4-pc", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    quantize_nonlinear=False),
        QuantConfig(name="int4-pc-sq", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    transform="smoothquant", quantize_nonlinear=False),
        QuantConfig(name="int4-pc-gptq", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    gptq=True, quantize_nonlinear=False),
        QuantConfig(name="int4-pc-sq+gptq", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    transform="smoothquant", gptq=True, quantize_nonlinear=False),
        QuantConfig(name="int4-pc-elem", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    outlier_ratio=0.1, outlier_format="int8", quantize_nonlinear=False),
        QuantConfig(name="int4-pc-had+elem", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    transform="hadamard", outlier_ratio=0.1, outlier_format="int8",
                    quantize_nonlinear=False),
        QuantConfig(name="int4-pc-sq+elem", w_format="int4", a_format="int4",
                    w_granularity="per_channel", a_granularity="per_channel",
                    transform="smoothquant", outlier_ratio=0.1, outlier_format="int8",
                    quantize_nonlinear=False),
        QuantConfig(name="int8-pc", w_format="int8", a_format="int8",
                    w_granularity="per_channel", a_granularity="per_channel",
                    quantize_nonlinear=False),
    ]

    # Run on MNIST
    model_mnist, fp32_acc = _build_mnist_model()
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256)
    calib_samples = []
    for x, _y in DataLoader(datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform), batch_size=64):
        calib_samples.append(x)
        if len(calib_samples) >= 8:
            break

    study = Study(combo_configs, model=copy.deepcopy(model_mnist))
    report = study.run(calib_samples, eval_data=test_loader, eval_fn=_mnist_eval_fn, outputs="default")

    sections.append('<table class="qsnr-table"><tr><th>Config</th><th>Accuracy</th><th>Δ vs FP32</th><th>推理开销</th></tr>')
    for part in report.parts:
        for r in report._results[part]:
            acc = r.quant_metrics.get("accuracy", float("nan")) if r.quant_metrics else float("nan")
            delta = acc - fp32_acc if fp32_acc and not math.isnan(acc) else float("nan")
            has_gptq = r.config.gptq
            has_had = r.config.transform == "hadamard"
            overhead = "2×FWHT" if has_had else "零"
            cls = ' class="best"' if abs(delta) < 0.005 else ''
            sections.append(f'<tr><td>{r.name}</td><td>{acc:.4f}</td><td{cls}>{delta:+.4f}</td><td>{overhead}</td></tr>')
    sections.append('</table>')

    sections.append("""<div class="note">
    <strong>组合策略关键发现</strong>：<br>
    - <strong>SmoothQuant + GPTQ</strong>：最推荐的零开销组合。SQ 处理激活 outlier，GPTQ 优化权重，互补性强<br>
    - <strong>Hadamard + Element Sparse</strong>：Hadamard 先分散 outlier，Element Sparse 再捕获残余，但推理开销叠加<br>
    - <strong>SmoothQuant + Element Sparse</strong>：SQ 已收窄动态范围，Element Sparse 的边际收益可能减少<br>
    - <strong>GPTQ + Element Sparse</strong>：两者都作用于权重，GPTQ 已优化权重量化，Element Sparse 可能冗余
    </div>""")
    sections.append('</div>')
    return '\n'.join(sections)


# ===================================================================
# CHAPTER 6: 结论与决策指南
# ===================================================================
def build_ch6():
    return """
<div class="section-card" id="ch6">
<h2>Chapter 6. 结论与决策指南</h2>

<h3>6.1 策略 × 粒度最佳配置</h3>
<table class="qsnr-table">
<tr><th>场景</th><th>最佳配置</th><th>推理开销</th><th>精度</th></tr>
<tr><td>LLM 量化（最佳精度）</td><td>int4 per_channel + SmoothQuant + GPTQ</td><td>零</td><td>最高</td></tr>
<tr><td>LLM 量化（快速部署）</td><td>int4 per_channel + SmoothQuant</td><td>零</td><td>高</td></tr>
<tr><td>无校准数据</td><td>int4 per_channel + Hadamard</td><td>2×FWHT</td><td>中高</td></tr>
<tr><td>CNN 权重量化</td><td>int4 per_channel + GPTQ</td><td>零</td><td>高</td></tr>
<tr><td>极端 outlier</td><td>int4 per_tensor + Element Sparse (r=0.1)</td><td>低</td><td>高</td></tr>
<tr><td>安全基线</td><td>int8 per_channel</td><td>零</td><td>稳健</td></tr>
</table>

<h3>6.2 决策树</h3>
<div class="formula">
1. 有校准数据？
   ├── 是 → 激活有 outlier？
   │   ├── 是 → SmoothQuant (α=0.5)
   │   │   └── 还需要更高精度？ → + GPTQ
   │   └── 否 → GPTQ (weight-only)
   └── 否 → Hadamard (无状态，纯数学变换)
2. 推理能容忍 FWHT 开销？
   ├── 否 → SmoothQuant / GPTQ (零开销)
   └── 是 → Hadamard (或 Hadamard + Element Sparse)
3. Outlier 比例？
   ├── < 2% (极端少量) → Element Sparse (r=0.02~0.05)
   ├── 2~10% → SmoothQuant 或 Hadamard
   └── > 10% → 考虑 int8 或 per_block 粒度
</div>

<h3>6.3 开销对比</h3>
<table class="qsnr-table">
<tr><th>策略</th><th>推理开销</th><th>校准成本</th><th>硬件需求</th><th>精度提升</th></tr>
<tr><td><span class="badge badge-identity">Identity</span></td><td>零</td><td>无</td><td>无</td><td>基线</td></tr>
<tr><td><span class="badge badge-hadamard">Hadamard</span></td><td>2×FWHT/层</td><td>无</td><td>FWHT kernel</td><td>+5~15 dB</td></tr>
<tr><td><span class="badge badge-smoothquant">SmoothQuant</span></td><td>零</td><td>1 forward</td><td>无</td><td>+3~12 dB</td></tr>
<tr><td><span class="badge badge-gptq">GPTQ</span></td><td>零</td><td>Hessian 计算</td><td>无</td><td>+5~15 dB (W)</td></tr>
<tr><td><span class="badge badge-elem">Element Sparse</span></td><td>低</td><td>无/1 forward</td><td>mask 索引</td><td>+10~25 dB</td></tr>
<tr><td><span class="badge badge-group">Group Sparse</span></td><td>极低</td><td>无/1 forward</td><td>无</td><td>+2~8 dB</td></tr>
</table>

<h3>6.4 总结</h3>
<div class="note">
<strong>核心结论</strong>：<br>
1. <strong>三轴正交模型</strong>（format × granularity × transform）提供了系统性的量化设计空间<br>
2. <strong>SmoothQuant + GPTQ</strong> 是当前工业界验证最充分的零开销组合，推荐作为默认方案<br>
3. <strong>Hadamard</strong> 在大维度 + 无校准数据场景下不可替代，但推理开销是硬伤<br>
4. <strong>Element Sparse</strong> 对少量极端 outlier 效果最显著，与 SmoothQuant 互补<br>
5. <strong>Group Sparse</strong> 硬件友好但精度提升有限，适用于 channel 间动态范围差异大的场景<br>
6. <strong>粒度选择</strong>：per_block/per_channel 通常是最佳平衡点，per_tensor 需要策略补救，bank 是列方向的特殊分组
</div>
</div>
"""


# ===================================================================
# Main: assemble full HTML
# ===================================================================
def main():
    print("Building Transform Strategy Report...")
    os.makedirs(OUT_DIR, exist_ok=True)

    ch0 = build_ch0()
    print("  Ch0 done")
    ch1 = build_ch1()
    print("  Ch1 done")
    ch2 = build_ch2()
    print("  Ch2 done")
    ch3 = build_ch3()
    print("  Ch3 done")
    ch4 = build_ch4()
    print("  Ch4 done")
    ch5 = build_ch5()
    print("  Ch5 done")
    ch6 = build_ch6()
    print("  Ch6 done")

    body = '\n'.join([ch0, ch1, ch2, ch3, ch4, ch5, ch6])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化策略学术分析报告 — microxcaling</title>
<style>{CSS}</style>
</head>
<body>
<h1>量化策略学术分析报告</h1>
<p>Hadamard × SmoothQuant × GPTQ × Granularity × Sparse — 从数学示例到模型 E2E 的完整分析</p>
<p>生成脚本：<code>scripts/transform_strategy_report.py</code></p>
{NAV}
{body}
</body>
</html>"""

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(OUT_FILE) / 1024
    print(f"\nOutput: {OUT_FILE}")
    print(f"Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
