"""
Error propagation analysis: hook vs observer decomposition.

Demonstrates per-layer error source diagnosis by comparing accumulated
(true_error hook) QSNR with local (QSNRObserver) QSNR.  Uses a model with
a CustomMatMul layer that is observer-monitored (via patched torch.matmul)
but NOT hook-monitored (no cfg attribute), proving hook ⊂ observer.

Run: python scripts/test_error_propagation.py
"""

import os
import torch
import torch.nn as nn

from src.session import Session, QuantConfig
from src.report._study_report import StudyReport


# ═══════════════════════════════════════════════════════════════════════════
# CustomMatMul: plain nn.Module, not in _MODULE_MAPPING
#
#  - hook:  ✗ (no cfg attr → not in _get_quantized_modules)
#  - observer: ✓ (via patched torch.matmul during analysis)
# ═══════════════════════════════════════════════════════════════════════════

class CustomMatMul(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))

    def forward(self, x):
        return torch.matmul(x, self.weight.T)


# ═══════════════════════════════════════════════════════════════════════════
# Test model
# ═══════════════════════════════════════════════════════════════════════════

def build_model():
    return nn.Sequential(
        nn.Linear(64, 128),        # [0] → QuantizedLinear   (hook ✓, observer ✓)
        CustomMatMul(128, 128),    # [1] → stays CustomMatMul (hook ✗, observer ✓)
        nn.ReLU(),                 # [2] → QuantizedReLU      (hook ✓, observer ✓)
        nn.Linear(128, 10),        # [3] → QuantizedLinear    (hook ✓, observer ✓)
    )


def calib_data(n_batches=4, bs=16):
    return [torch.randn(bs, 64) for _ in range(n_batches)]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)

    sep = "=" * 70
    minor_sep = "─" * 70

    print(sep)
    print("  Error Propagation Analysis — Hook vs Observer Decomposition")
    print(sep)

    # ── Build model ──────────────────────────────────────────────────
    model = build_model()
    print(f"\n{minor_sep}")
    print("  Model (4 submodules)")
    print(minor_sep)
    for i, m in enumerate(model):
        w = hasattr(m, "weight")
        print(f"  [{i}] {type(m).__name__:<20} weight={tuple(m.weight.shape) if w else '-'}")

    # ── Run session with high-level API ──────────────────────────────
    data = calib_data()
    cfg = QuantConfig(
        name="int8_bf16_storage",
        w_format="int8",
        a_format="int8",
        calibrator="max",
        storage_bits=16,            # enable bfloat16 storage → output events for Linear
        quantize_nonlinear=False,
    )
    session = Session(model, cfg)
    result = session.run(data, outputs=["qsnr", "mse"])

    # ── Verify hook / observer keys ──────────────────────────────────
    print(f"\n{minor_sep}")
    print("  Hook vs Observer Keys")
    print(minor_sep)
    print(f"  Hook keys (accumulated):     {sorted(result.accum_qsnr_per_layer.keys())}")
    print(f"  Observer raw keys:           {sorted(result.observers_data.keys())}")

    # Extract local QSNR (output role) via public accessor
    local_qsnr, _ = result.qsnr_per_role(role="output")
    print(f"  Local QSNR keys (output):    {sorted(local_qsnr.keys())}")

    # Match observer → hook by prefix
    hook_keys = set(result.accum_qsnr_per_layer.keys())
    all_matched = set()
    observer_only = []
    for ok in sorted(local_qsnr.keys()):
        matched = None
        for hk in hook_keys:
            if ok == hk or ok.startswith(hk + "."):
                matched = hk
                break
        if matched:
            all_matched.add(ok)
            print(f"    ✓ {ok:<28} → hook '{matched}'  local={local_qsnr[ok]:.1f}  accum={result.accum_qsnr_per_layer[matched]:.1f}")
        else:
            observer_only.append(ok)
            print(f"    ⊙ {ok:<28} → NO hook match  (observer-only)  local={local_qsnr[ok]:.1f}")

    print(f"\n  Matched:              {sorted(all_matched)}")
    print(f"  Observer-only:        {sorted(observer_only) if observer_only else '(none)'}")

    if observer_only:
        print("  >> hook ⊂ observer confirmed! <<")
    else:
        print("  >> WARNING: hook ⊂ observer NOT demonstrated <<")

    # ── Build StudyReport & run correlation ──────────────────────────
    print(f"\n{minor_sep}")
    print("  StudyReport.correlate_hook_observer()")
    print(minor_sep)
    report = StudyReport({"test": [result]})
    corr = report.correlate_hook_observer(role="output")

    for cfg_name, info in corr.items():
        print(f"  Config: {cfg_name}")
        print(f"  Matched: {len(info['matched'])} layers")
        for hk, acc, loc in info["matched"]:
            headroom = loc - acc
            print(f"    {hk:<20} accum={acc:.2f} dB  local={loc:.2f} dB  headroom={headroom:+.2f} dB")
        if info["observer_only"]:
            print(f"  Observer-only: {len(info['observer_only'])} layers")
            for ok, loc in info["observer_only"]:
                print(f"    {ok:<20} local={loc:.2f} dB")
        if info["hook_only"]:
            print(f"  Hook-only (no observer data): {len(info['hook_only'])} layers")
            for hk, acc in info["hook_only"]:
                print(f"    {hk:<20} accum={acc:.2f} dB")

    # ── Terminal table ───────────────────────────────────────────────
    print(f"\n{minor_sep}")
    print("  Terminal Table — result.tables.error_source_analysis()")
    print(minor_sep)
    single_table = result.tables.error_source_analysis(role="output")
    print(single_table)

    print(f"\n{minor_sep}")
    print("  Terminal Table — report.tables.error_source_analysis()")
    print(minor_sep)
    table = report.tables.error_source_analysis(role="output")
    print(table)

    # ── Generate figures ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = "scripts/output_error_propagation"
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

    # ── Full save() test ─────────────────────────────────────────────
    report.save(output_dir)
    print(f"  ✓ StudyReport.save() complete → {output_dir}/")

    print(f"\n{sep}")
    print(f"  DONE — All outputs in {output_dir}/")
    print(sep)


if __name__ == "__main__":
    main()
