"""
Transformer encoder error propagation analysis: hook vs observer decomposition.

Demonstrates per-layer error source diagnosis in a multi-layer transformer
where attention matmuls (torch.matmul) are observer-monitored but NOT
hook-monitored, and error accumulates through the depth of the network.

Module coverage:
  - Hooked (in _MODULE_MAPPING):    Linear, LayerNorm, GELU, Softmax
  - Observer-only (patched ops):    attention matmuls (QK^T, attn*V)

Note: both attention matmuls in the same MHA block share the same layer
name (e.g. ``layers.0.self_attn.matmul``), so the observer metrics from
the second call overwrite the first.  This is expected — differentiating
them requires per-call-site naming beyond the current patching system.

Run: python scripts/test_transformer_error_propagation.py
"""

import os
import torch
import torch.nn as nn

from src.session import Session, QuantConfig
from src.report._study_report import StudyReport


# ═══════════════════════════════════════════════════════════════════════════
# Model components
#
# MultiHeadAttention and TransformerEncoderLayer are plain nn.Module
# (not ObservableMixin).  Their child modules (Linear, LayerNorm, GELU,
# Softmax) ARE replaced by quantize_model and contribute to both hook
# and observer data.  The attention matmuls (torch.matmul) are NOT
# modules — they are intercepted only by the patched-op system and appear
# as observer-only entries.
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Standard MHA — attention matmuls are observer-only."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        Q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        K = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        V = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        # Q*K^T  →  patched torch.matmul  (observer-only, no hook)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (hd ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # attn_weights * V  →  patched torch.matmul  (observer-only, no hook)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)

        return self.out_proj(attn_out)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer block.  Residual adds are patched torch.add."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention + residual
        attn_out = self.self_attn(self.norm1(x))
        x = x + attn_out
        # FFN + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


class TransformerEncoder(nn.Module):
    """Stack of N transformer encoder layers."""

    def __init__(self, d_model=128, n_heads=4, d_ff=256, n_layers=4,
                 vocab_size=64):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def calib_data(n_batches=8, bs=4, seq_len=16, d_in=64):
    return [torch.randn(bs, seq_len, d_in) for _ in range(n_batches)]


def count_hooked_modules(model):
    """Count modules whose cfg actually triggers quantization."""
    from src.scheme.op_config import cfg_causes_quantization
    return sum(
        1 for _name, mod in model.named_modules()
        if hasattr(mod, "cfg") and cfg_causes_quantization(mod.cfg)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)

    sep = "=" * 72
    minor_sep = "─" * 72

    print(sep)
    print("  Transformer Encoder — Hook vs Observer Decomposition")
    print(sep)

    # ── Build model ──────────────────────────────────────────────────
    n_layers = 4
    d_model, n_heads, d_ff = 128, 4, 256
    model = TransformerEncoder(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                               n_layers=n_layers, vocab_size=64)

    print(f"\n{minor_sep}")
    print(f"  Architecture: {n_layers}-layer Transformer Encoder")
    print(f"  d_model={d_model}  n_heads={n_heads}  d_ff={d_ff}")
    print(minor_sep)

    # Count submodules per layer
    layer_0 = model.layers[0]
    sub_count = len(list(layer_0.named_modules())) - 1  # exclude self
    hooked_count = count_hooked_modules(layer_0)
    print(f"  Per-layer submodules: {sub_count}  (hooked: ~{hooked_count})")
    print(f"  Total submodules (est.): {2 + n_layers * sub_count}")
    print(f"  Observer-only per layer: ~2  (attention matmuls)")

    # ── Run session ──────────────────────────────────────────────────
    data = calib_data()
    cfg = QuantConfig(
        name="int8_bf16_storage",
        w_format="int8",
        a_format="int8",
        calibrator="max",
        storage_bits=16,
        quantize_nonlinear=False,
    )
    session = Session(model, cfg)
    result = session.run(data, outputs=["qsnr", "mse"])

    # ── Verify hook / observer keys ──────────────────────────────────
    print(f"\n{minor_sep}")
    print("  Hook vs Observer — Key Coverage")
    print(minor_sep)

    hook_keys = sorted(result.accum_qsnr_per_layer.keys())
    local_qsnr, _ = result.qsnr_per_role(role="output")
    observer_keys = sorted(local_qsnr.keys())

    print(f"  Hook layers:     {len(hook_keys)}")
    print(f"  Observer layers: {len(observer_keys)}")
    print(f"  Coverage gap:    {len(observer_keys) - len(hook_keys)} layers")

    # Classify by type
    matched = []
    observer_only = []
    hook_only = []

    for ok in observer_keys:
        found = None
        for hk in hook_keys:
            if ok == hk or ok.startswith(hk + "."):
                found = hk
                break
        if found:
            matched.append((ok, found))
        else:
            observer_only.append(ok)

    for hk in hook_keys:
        found = any(hk == m[1] for m in matched)
        if not found:
            hook_only.append(hk)

    print(f"\n  Matched:         {len(matched)}")
    print(f"  Observer-only:   {len(observer_only)}")
    print(f"  Hook-only:       {len(hook_only)}")

    if observer_only:
        print(f"\n  Observer-only entries:")
        for ok in observer_only:
            print(f"    ⊙ {ok:<50} local={local_qsnr[ok]:.1f} dB")

    # Per-layer accumulated QSNR summary
    print(f"\n{minor_sep}")
    print("  Accumulated QSNR by Layer (hook data)")
    print(minor_sep)
    for hk in hook_keys:
        marker = ""
        if hk in hook_only:
            marker = " [hook-only]"
        print(f"  {hk:<50} {result.accum_qsnr_per_layer[hk]:6.1f} dB{marker}")

    # ── StudyReport correlation ─────────────────────────────────────
    print(f"\n{minor_sep}")
    print("  StudyReport.correlate_hook_observer()")
    print(minor_sep)
    report = StudyReport({"transformer": [result]})
    corr = report.correlate_hook_observer(role="output")

    for cfg_name, info in corr.items():
        n_m = len(info["matched"])
        n_o = len(info["observer_only"])
        n_h = len(info["hook_only"])
        print(f"  Config: {cfg_name}")
        print(f"  Matched: {n_m}  Observer-only: {n_o}  Hook-only: {n_h}")
        print()

        # Show first and last matched layers to illustrate accumulation
        if info["matched"]:
            first = info["matched"][0]
            last = info["matched"][-1]
            total_drop = last[1] - first[1]
            print(f"  First matched:  {first[0]:<40} accum={first[1]:.1f}  local={first[2]:.1f}  headroom={first[2]-first[1]:+.1f}")
            print(f"  Last matched:   {last[0]:<40} accum={last[1]:.1f}  local={last[2]:.1f}  headroom={last[2]-last[1]:+.1f}")
            print(f"  Total accum. drop across {n_m} layers: {total_drop:+.1f} dB")

        if info["observer_only"]:
            print(f"\n  Observer-only entries:")
            for ok, loc in info["observer_only"][:5]:
                print(f"    {ok:<48} local={loc:.1f} dB")
            if len(info["observer_only"]) > 5:
                print(f"    ... and {len(info['observer_only']) - 5} more")

    # ── Terminal table ───────────────────────────────────────────────
    print(f"\n{minor_sep}")
    print("  Terminal Table — report.tables.error_source_analysis()")
    print(minor_sep)
    table = report.tables.error_source_analysis(role="output")
    print(table)

    # ── Generate figures ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = "scripts/output_transformer_error_propagation"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)

    print(f"\n{minor_sep}")
    print("  Figures & Outputs")
    print(minor_sep)

    try:
        fig = report.plot.error_propagation(role="output")
        path = f"{output_dir}/figures/error_propagation.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ error_propagation.png")
    except Exception as e:
        print(f"  ✗ error_propagation failed: {e}")

    try:
        fig = report.plot.accumulated_vs_local(role="output")
        path = f"{output_dir}/figures/accumulated_vs_local.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ accumulated_vs_local.png")
    except Exception as e:
        print(f"  ✗ accumulated_vs_local failed: {e}")

    table_path = f"{output_dir}/tables/error_source.txt"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"  ✓ error_source.txt")

    # Pivoted per-layer QSNR table (rows=layers, cols=hook/observer/delta)
    print(f"  per_layer_qsnr.csv: saved to {output_dir}/tables/per_layer_qsnr.csv")

    # Full save
    report.save(output_dir)
    print(f"  ✓ StudyReport.save() complete → {output_dir}/")

    print(f"\n{sep}")
    print(f"  DONE — All outputs in {output_dir}/")
    print(sep)


if __name__ == "__main__":
    main()
