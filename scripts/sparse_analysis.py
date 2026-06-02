"""
Sparse Outlier Isolation — Analysis & Visualization
=====================================================
Phase A: Baseline quantization (no sparse) for 3 granularity modes on a 4x8 tensor.
Phase B: Sparse quantization step-by-step with outlier_ratio=0.1.
Phase C: QSNR vs ratio curves for 3 distributions (Gaussian+outlier, Laplace, Uniform).

Usage: PYTHONPATH=. python scripts/sparse_analysis.py
Output: prints tables to stdout, saves plots to scripts/output_sparse_analysis/
"""
import os, sys, math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.quantize.elemwise import quantize

OUT = "scripts/output_sparse_analysis"
os.makedirs(OUT, exist_ok=True)

FMT = FormatBase.from_str("int4")
FMT_NAME = "int4"

# ═══════════════════════════════════════════════════════════════════════════════
# Test tensor: 4 rows x 8 cols. Two outliers at (0,7)=15.0 and (2,5)=8.0.
# The remaining values are in [-1.5, 1.2].
# ═══════════════════════════════════════════════════════════════════════════════

X = torch.tensor([
    [ 0.5, -0.3,  0.8, -0.2,  1.2, -0.7,  0.1, 15.0],
    [ 0.4,  0.9, -1.1,  0.3, -0.5,  0.6, -0.9,  0.2],
    [-0.8,  0.2, -0.4,  0.7, -0.1,  8.0,  0.3, -0.6],
    [ 0.1, -0.2,  0.5, -0.8,  0.4, -0.3,  0.9, -1.5],
], dtype=torch.float32)

# ── Granularity specs ────────────────────────────────────────────────────────
G_PT  = GranularitySpec.per_tensor()                           # per_tensor
G_PC  = GranularitySpec.per_channel(axis=0)                     # per_channel (rows)
G_PB  = GranularitySpec.per_block(size=4, axis=-1)              # per_block (4 cols)

# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_scheme(granularity):
    """Build a QuantScheme with int4 + pot scale_storage."""
    return QuantScheme(format=FMT, granularity=granularity, scale_storage="pot")

def qsnr(x, x_q):
    """Compute QSNR in dB."""
    sig = torch.mean(x ** 2)
    err = torch.mean((x - x_q) ** 2)
    if err == 0:
        return float("inf")
    return 10 * math.log10(sig.item() / err.item())

def elemwise_quant_scalar(v, mbits=4, max_norm=1.75):
    """Simulate FormatBase.quantize_elemwise for a single float (int4)."""
    s = 1 if v >= 0 else -1
    av = abs(v) * (2 ** (mbits - 2))          # scale by 4 for int4
    r = math.floor(av + 0.5)                   # nearest rounding
    q = s * (r / (2 ** (mbits - 2)))           # unscale
    if abs(q) > max_norm:
        q = s * max_norm
    return q

def per_tensor_amax(x, scale_storage="pot"):
    """Compute per_tensor amax (pot rounded)."""
    a = torch.amax(torch.abs(x)).clamp(min=1e-12)
    if scale_storage == "pot":
        a = 2 ** torch.round(torch.log2(a))
    return a.item()

def per_channel_amax(x, axis=0, scale_storage="pot"):
    """Compute per_channel amax along given axis."""
    dims = [i for i in range(x.ndim) if i != axis]
    a = torch.amax(torch.abs(x), dim=tuple(dims), keepdim=True)
    a = a.clamp(min=1e-12)
    if scale_storage == "pot":
        a = 2 ** torch.round(torch.log2(a))
    return a

def format_array_2d(arr, fmt_str="7.3f"):
    """Format a 2D numpy array as aligned string table."""
    lines = []
    for row in arr:
        lines.append("  [" + ", ".join(f"{v:{fmt_str}}" for v in row) + "]")
    return "\n".join(lines)

def mask_to_str(mask_2d):
    """Render a bool 2D mask as O (outlier) and . (normal)."""
    lines = []
    for row in mask_2d:
        lines.append("  [" + " ".join("O" if v else "." for v in row) + "]")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Baseline — no sparse
# ═══════════════════════════════════════════════════════════════════════════════

def phase_a():
    print("=" * 72)
    print("Phase A — Baseline Quantization (no sparse)")
    print("=" * 72)
    print(f"\nTest tensor ({X.shape[0]}x{X.shape[1]}), format={FMT_NAME}:\n")
    print(format_array_2d(X.numpy()))
    print()

    for label, g in [("per_tensor", G_PT), ("per_channel (axis=0)", G_PC),
                      ("per_block (size=4, axis=-1)", G_PB)]:
        scheme = fmt_scheme(g)
        x_q = quantize(X, scheme)
        err = X - x_q
        qs = qsnr(X, x_q)

        print(f"── {label} ──")
        print(f"  QSNR: {qs:.2f} dB")

        if g.mode == GranularityMode.PER_TENSOR:
            amax = per_tensor_amax(X)
            print(f"  amax (pot): {amax:.1f}")
            print(f"  Normalized: X / {amax:.1f}")
            x_norm = X / amax
            print(format_array_2d(x_norm.numpy(), "8.4f"))
            print(f"  Quantized:")
            print(format_array_2d(x_q.numpy(), "7.2f"))
            print(f"  Error (x - x_q):")
            print(format_array_2d(err.numpy(), "7.2f"))

        elif g.mode == GranularityMode.PER_CHANNEL:
            amax = per_channel_amax(X, axis=0)
            print(f"  amax per channel (pot): {amax.squeeze().tolist()}")
            print(f"  Quantized:")
            print(format_array_2d(x_q.numpy(), "7.2f"))
            print(f"  Error (x - x_q):")
            print(format_array_2d(err.numpy(), "7.2f"))
            # Show which channels are affected
            print(f"  Channel 0 (row 0) has outlier 15.0 → amax dominates")
            print(f"  Channel 2 (row 2) has outlier 8.0  → amax dominates")
            print(f"  Channel 1,3 (rows 1,3) no outliers   → normal quantization")

        elif g.mode == GranularityMode.PER_BLOCK:
            print(f"  Block layout (size=4 along axis=-1):")
            print(f"    8 cols / 4 = 2 blocks per row, 8 blocks total")
            print(f"    Block (row0, cols4-7): [1.2, -0.7, 0.1, 15.0] → outlier dominates")
            print(f"    Block (row2, cols4-7): [-0.1, 8.0, 0.3, -0.6] → outlier dominates")
            print(f"  Quantized:")
            print(format_array_2d(x_q.numpy(), "7.2f"))
            print(f"  Error (x - x_q):")
            print(format_array_2d(err.numpy(), "7.2f"))

        print()

    # ── Summary comparison table ─────────────────────────────────────────
    print("── Baseline Summary ──")
    print(f"  {'Mode':<28} {'QSNR (dB)':>10} {'Values Crushed':>20}")
    print(f"  {'─'*28} {'─'*10} {'─'*20}")

    for label, g in [("per_tensor", G_PT), ("per_channel (axis=0)", G_PC),
                      ("per_block (size=4)", G_PB)]:
        scheme = fmt_scheme(g)
        x_q = quantize(X, scheme)
        qs = qsnr(X, x_q)
        nz_mask = (X != 0)
        crushed = ((X != 0) & (x_q == 0)).sum().item()
        n_crushed = f"{crushed}/{nz_mask.sum().item()}"
        print(f"  {label:<28} {qs:>10.2f} {n_crushed:>20}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: Sparse (outlier_ratio=0.1) step-by-step
# ═══════════════════════════════════════════════════════════════════════════════

def phase_b():
    RATIO = 0.1
    print("=" * 72)
    print(f"Phase B — Sparse Quantization (outlier_ratio={RATIO})")
    print("=" * 72)
    print()

    # ── per_tensor sparse ─────────────────────────────────────────────────
    print(f"── per_tensor + sparse (ratio={RATIO}) ──")
    N = X.numel()                        # 32
    k = max(1, int(N * RATIO))           # 3
    print(f"  N = {N}, k = max(1, int({N} * {RATIO})) = {k}")
    print(f"  → Top-{k} elements by magnitude are isolated as outliers\n")

    # Find top-k
    flat = X.flatten()
    _, top_idx = torch.topk(torch.abs(flat), k)
    mask_flat = torch.zeros(N, dtype=torch.bool)
    mask_flat.scatter_(0, top_idx, True)
    mask = mask_flat.reshape(X.shape)
    print(f"  Outlier positions (O=outlier, .=normal):")
    print(mask_to_str(mask.numpy()))
    print(f"  Outlier values: {flat[top_idx].tolist()}")

    # amax for each group
    amax_o = per_tensor_amax(X * mask.float())
    amax_n = per_tensor_amax(X * (~mask).float())
    print(f"\n  amax_o (outlier group): {amax_o:.1f}")
    print(f"  amax_n (normal group):  {amax_n:.1f}")
    print(f"  Without sparse, amax would be {per_tensor_amax(X):.1f}")
    print(f"  → Normal group scale shrinks from {per_tensor_amax(X):.1f} → {amax_n:.1f}")

    # Quantize
    g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=RATIO)
    x_q_sparse = quantize(X, fmt_scheme(g))
    x_q_base = quantize(X, fmt_scheme(G_PT))
    qs_s = qsnr(X, x_q_sparse)
    qs_b = qsnr(X, x_q_base)

    print(f"\n  Quantized (sparse):")
    print(format_array_2d(x_q_sparse.numpy(), "7.2f"))
    print(f"\n  Quantized (no sparse):")
    print(format_array_2d(x_q_base.numpy(), "7.2f"))
    print(f"\n  QSNR: {qs_b:.2f} dB → {qs_s:.2f} dB (Δ = {qs_s - qs_b:+.2f} dB)")

    # Per-element comparison for interesting positions
    print(f"\n  Key comparisons:")
    print(f"  {'Pos':<12} {'Original':>8} {'NoSparse':>8} {'Sparse':>8} {'Improved?':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            o, b, s = X[r,c].item(), x_q_base[r,c].item(), x_q_sparse[r,c].item()
            if abs(o - b) > 0.01 or abs(o - s) > 0.01:
                improved = "✓" if abs(s - o) < abs(b - o) else ("—" if abs(s - o) == abs(b - o) else "✗")
                print(f"  [{r},{c}]       {o:>8.2f} {b:>8.2f} {s:>8.2f} {improved:>10}")

    print()

    # ── per_channel sparse ─────────────────────────────────────────────────
    print(f"── per_channel (axis=0) + sparse (ratio={RATIO}) ──")
    axis = 0
    C = X.shape[axis]                    # 4
    N_per = X[0].numel()                 # 8
    k_pc = max(1, int(N_per * RATIO))    # 1  (0.1 * 8 = 0.8 → 1)
    print(f"  C = {C} channels, {N_per} elements/channel")
    print(f"  k_per_channel = max(1, int({N_per} * {RATIO})) = {k_pc}")
    print(f"  → Per channel, top-{k_pc} element is isolated\n")

    # Show per-channel top-k
    x_t = X.transpose(0, axis)  # (C, 8)
    _, top_idx_c = torch.topk(torch.abs(x_t.reshape(C, N_per)), k_pc, dim=1)
    mask_c = torch.zeros(C, N_per, dtype=torch.bool)
    mask_c.scatter_(1, top_idx_c, True)
    mask_c_2d = mask_c.reshape(X.shape)

    print(f"  Outlier positions (O=outlier, .=normal):")
    print(mask_to_str(mask_c_2d.numpy()))

    for ch in range(C):
        o_idx = top_idx_c[ch].item()
        o_val = X[ch, o_idx].item()
        print(f"  Channel {ch}: outlier at col {o_idx}, value = {o_val:.1f}")

    # Quantize
    g_pc = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=axis,
                           outlier_ratio=RATIO)
    x_q_pc_s = quantize(X, fmt_scheme(g_pc))
    x_q_pc_b = quantize(X, fmt_scheme(G_PC))
    qs_pc_s = qsnr(X, x_q_pc_s)
    qs_pc_b = qsnr(X, x_q_pc_b)

    # Show per-channel amax comparison
    amax_base = per_channel_amax(X, axis=0)
    amax_o_pc = per_channel_amax(X * mask_c_2d.float(), axis=0)
    amax_n_pc = per_channel_amax(X * (~mask_c_2d).float(), axis=0)
    print(f"\n  Per-channel amax comparison:")
    print(f"  {'Ch':>3} {'amax_base':>10} {'amax_o':>8} {'amax_n':>8} {'shrink':>8}")
    print(f"  {'─'*3} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for ch in range(C):
        b = amax_base.squeeze()[ch].item()
        n = amax_n_pc.squeeze()[ch].item()
        shrink = f"{b/n:.1f}x" if n > 0 else "—"
        print(f"  {ch:>3} {b:>10.1f} {amax_o_pc.squeeze()[ch].item():>8.1f} {n:>8.1f} {shrink:>8}")

    print(f"\n  QSNR: {qs_pc_b:.2f} dB → {qs_pc_s:.2f} dB (Δ = {qs_pc_s - qs_pc_b:+.2f} dB)")

    print(f"\n  Key comparisons:")
    print(f"  {'Pos':<12} {'Original':>8} {'NoSparse':>8} {'Sparse':>8} {'Improved?':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            o, b, s = X[r,c].item(), x_q_pc_b[r,c].item(), x_q_pc_s[r,c].item()
            if abs(o - b) > 0.01 or abs(o - s) > 0.01:
                improved = "✓" if abs(s - o) < abs(b - o) else ("—" if abs(s - o) == abs(b - o) else "✗")
                print(f"  [{r},{c}]       {o:>8.2f} {b:>8.2f} {s:>8.2f} {improved:>10}")

    print()

    # ── per_block sparse ──────────────────────────────────────────────────
    print(f"── per_block (size=4, axis=-1) + sparse (ratio={RATIO}) ──")
    bs = G_PB.block_size                        # 4
    k_pb = max(1, int(bs * RATIO))              # 1
    print(f"  block_size = {bs}")
    print(f"  k_per_block = max(1, int({bs} * {RATIO})) = {k_pb}")
    print(f"  → Per block, top-{k_pb} element is isolated\n")

    g_pb = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=bs,
                           block_axis=-1, outlier_ratio=RATIO)
    x_q_pb_s = quantize(X, fmt_scheme(g_pb))
    x_q_pb_b = quantize(X, fmt_scheme(G_PB))
    qs_pb_s = qsnr(X, x_q_pb_s)
    qs_pb_b = qsnr(X, x_q_pb_b)

    print(f"  Quantized (sparse):")
    print(format_array_2d(x_q_pb_s.numpy(), "7.2f"))
    print(f"\n  Quantized (no sparse):")
    print(format_array_2d(x_q_pb_b.numpy(), "7.2f"))
    print(f"\n  QSNR: {qs_pb_b:.2f} dB → {qs_pb_s:.2f} dB (Δ = {qs_pb_s - qs_pb_b:+.2f} dB)")

    print(f"\n  Key comparisons:")
    print(f"  {'Pos':<12} {'Original':>8} {'NoSparse':>8} {'Sparse':>8} {'Improved?':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            o, b, s = X[r,c].item(), x_q_pb_b[r,c].item(), x_q_pb_s[r,c].item()
            if abs(o - b) > 0.01 or abs(o - s) > 0.01:
                improved = "✓" if abs(s - o) < abs(b - o) else ("—" if abs(s - o) == abs(b - o) else "✗")
                print(f"  [{r},{c}]       {o:>8.2f} {b:>8.2f} {s:>8.2f} {improved:>10}")

    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("── Sparse Summary (ratio=0.1) ──")
    print(f"  {'Mode':<30} {'QSNR base':>10} {'QSNR sparse':>12} {'Δ':>8}")
    print(f"  {'─'*30} {'─'*10} {'─'*12} {'─'*8}")
    for label, g_base, g_sparse in [
        ("per_tensor", G_PT, GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=RATIO)),
        ("per_channel (axis=0)", G_PC, GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0, outlier_ratio=RATIO)),
        ("per_block (size=4)", G_PB, GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=4, block_axis=-1, outlier_ratio=RATIO)),
    ]:
        q_base = qsnr(X, quantize(X, fmt_scheme(g_base)))
        q_sp = qsnr(X, quantize(X, fmt_scheme(g_sparse)))
        print(f"  {label:<30} {q_base:>10.2f} {q_sp:>12.2f} {q_sp - q_base:>+8.2f}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C: QSNR vs ratio curves
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_tensors():
    """Create 3 test tensors (64, 128) with different distributions."""
    torch.manual_seed(42)
    shape = (64, 128)

    # 1. Gaussian + sparse outliers
    gauss = torch.randn(shape)
    # Inject 5 extreme outliers at fixed positions
    outlier_positions = [(0, 0), (0, 1), (10, 50), (30, 100), (50, 60)]
    for r, c in outlier_positions:
        gauss[r, c] = torch.randn(1).item() * 10 + (15 if torch.randn(1).item() > 0 else -15)

    # 2. Laplace (heavy-tailed)
    laplace = torch.distributions.Laplace(0, 2).sample(shape)

    # 3. Uniform
    uniform = torch.rand(shape) * 2 - 1

    return [
        ("Gaussian + outliers (5 spikes)", gauss),
        ("Laplace(0, 2) heavy-tailed", laplace),
        ("Uniform(-1, 1) no outliers", uniform),
    ]


def phase_c():
    print("=" * 72)
    print("Phase C — QSNR vs Outlier Ratio Curves")
    print("=" * 72)

    distributions = generate_test_tensors()
    ratios = np.linspace(0, 0.5, 26)  # 0, 0.02, 0.04, ..., 0.5

    # Per distribution: 3 granularity modes → 3 curves
    modes = [
        ("per_tensor", lambda r: GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=r)),
        ("per_channel", lambda r: GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0, outlier_ratio=r)),
        ("per_block", lambda r: GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=32, block_axis=-1, outlier_ratio=r)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]  # blue, orange, green
    linestyles = ["-", "--", "-."]

    for dist_idx, (dist_name, tensor) in enumerate(distributions):
        ax = axes[dist_idx]
        print(f"\n{dist_name} (shape={tensor.shape}):")

        for mode_idx, (mode_name, make_g) in enumerate(modes):
            qsnrs = []
            for ratio in ratios:
                g = make_g(float(ratio))
                # per_block needs valid block_size even at ratio=0
                if g.mode == GranularityMode.PER_BLOCK and g.block_size <= 0:
                    continue
                try:
                    x_q = quantize(tensor, fmt_scheme(g))
                    qsnrs.append(qsnr(tensor, x_q))
                except Exception:
                    qsnrs.append(float("nan"))

            label = mode_name
            ax.plot(ratios, qsnrs, color=colors[mode_idx], linestyle=linestyles[mode_idx],
                    linewidth=2, label=label, marker=".", markersize=3, markevery=2)

            # Print key points
            base_qsnr = qsnrs[0]
            best_idx = np.nanargmax(qsnrs)
            best_ratio = ratios[best_idx]
            best_qsnr = qsnrs[best_idx]
            print(f"  {mode_name:<20}: base={base_qsnr:.2f} dB  "
                  f"best ratio={best_ratio:.2f} ({best_qsnr:.2f} dB)  "
                  f"Δ={best_qsnr - base_qsnr:+.2f} dB")

        ax.set_title(dist_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("outlier_ratio", fontsize=11)
        ax.set_ylabel("QSNR (dB)", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)

    fig.suptitle("QSNR vs Outlier Ratio — int4, 3 distributions × 3 granularity modes",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT, "qsnr_vs_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    phase_a()
    phase_b()
    phase_c()
    print("\nDone.")
