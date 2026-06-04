"""MXInt Error Analysis — end-to-end quantization error analysis.

Usage:
    python src/api/mxint_error_analysis.py --adapter /path/to/_adapter.py

Adapter contract — the adapter module must define three functions:

    get_model()  -> nn.Module          # FP32 model with weights loaded
    get_eval_fn() -> callable           # eval_fn(model, data) -> Dict[str, float]
    get_data()   -> (calib, eval_data)  # (List[Tensor], Iterable)

eval_fn handles two modes:
    - data is list → calibration pass (forward only, return {})
    - data is DataLoader → evaluation pass (return {"accuracy": ...})

MXInt8 defaults: w_bits=8, a_bits=8, block_size=16.
All configurable via CLI flags.

Charts: P1 Accum QSNR bar (model layer order) → P2 Accuracy table →
P3 Accum vs Local line → P4 Per-Role QSNR bar (≤100 dB) →
P5 Error Attribution waterfall (positive error) →
compare_extreme_layers (top-1 worst + best, dist_overlay + block heatmap) →
⑨ Precision Recovery grouped bar → Distribution table → Causal analysis →
U2 Intervention table + U6 Box plot.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import torch
import torch.nn as nn

from src.session import Session, QuantConfig
from src.analysis.observers import (
    QSNRObserver, MSEObserver,
    DistributionObserver, HistogramObserver, PerBlockQSNRObserver,
)
from src.cost.model_cost import analyze_model_cost
from src.api._chart_helpers import QSNR_REF

# ── MXInt8 defaults (constants) ──────────────────────────────────────
W_BITS = 8
A_BITS = 8
BLOCK_SIZE = 16

# ── Optional: harness render_chart ───────────────────────────────────
try:
    from harness.tools.chart import render_chart
except ImportError:
    render_chart = None


# =====================================================================
# Adapter loader
# =====================================================================

def load_adapter(path: str):
    """Import adapter module from file path."""
    spec = importlib.util.spec_from_file_location("_adapter", path)
    if spec is None:
        raise ImportError(f"Cannot load adapter from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for name in ("get_model", "get_eval_fn", "get_data"):
        if not hasattr(mod, name):
            raise AttributeError(
                f"Adapter {path} missing required function: {name}()"
            )
    return mod


# =====================================================================
# Config builder
# =====================================================================

def build_mxint_config(
    w_bits: int = W_BITS,
    a_bits: int = A_BITS,
    block_size: int = BLOCK_SIZE,
) -> QuantConfig:
    return QuantConfig(
        name=f"mxint{w_bits}",
        w_format=f"int{w_bits}",
        w_granularity="per_block",
        w_block_size=block_size,
        a_format=f"int{a_bits}",
        a_granularity="per_block",
        a_block_size=block_size,
    )


# =====================================================================
# Chart rendering
# =====================================================================

def _chart(data, chart_type, *, x, y, label="MXInt8", title="", hue=None):
    if render_chart is None:
        return
    render_chart(data, chart_type, x=x, y=y, label=label, title=title, hue=hue)


def charts_from_result(result, label: str = "MXInt8", *,
                       model=None, block_size: int = 16):
    """Generate charts from SessionResult via render_chart.

    Uses accumulated QSNR as primary metric (linear-only).
    Only one chart shows local vs accum comparison.
    """
    from src.api.layer_diagnostic import (
        accum_qsnr_bar,
        accum_vs_local_line,
        per_role_local_qsnr,
        error_attribution_waterfall,
        compare_extreme_layers,
    )

    # ── Phase 2: Global overview ─────────────────────────────────────
    # P1: Accum QSNR bar — use model layer order (not sorted by QSNR)
    accum_qsnr_bar(result, label=label, sort_by_model_order=True)

    # P2: Accuracy summary table
    summary_rows = []
    if result.fp32_metrics:
        for k, v in result.fp32_metrics.items():
            q = result.quant_metrics.get(k, "")
            d = result.delta.get(k, "")
            row = {"metric": k, "fp32": v, "quant": q, "delta": d}
            if isinstance(d, (int, float)) and isinstance(v, (int, float)) and v != 0:
                row["relative_delta_pct"] = round(d / v * 100, 4)
            if isinstance(v, (int, float)) and isinstance(q, (int, float)):
                row["quant_pct_of_fp32"] = round(q / v * 100, 4) if v != 0 else ""
            summary_rows.append(row)
    if summary_rows:
        _chart(summary_rows, "table", x="metric", y="fp32",
               label=label, title="Accuracy Summary (Precision Comparison)")

    # P3: Accum vs Local
    accum_vs_local_line(result, label=label)

    # P4: Per-role local QSNR grouped bar (QSNR ≤ 100 dB only)
    per_role_local_qsnr(result, label=label)

    # ── Phase 3: Error attribution ───────────────────────────────────
    # P5: Error attribution waterfall (positive error only)
    error_attribution_waterfall(result, k=10, label=label)

    # ── Phase 4: Extreme layer deep dive ─────────────────────────────
    compare_extreme_layers(result, top_k=1, linear_only=True,
                           model=model, block_size=block_size)


def _worst_layers_with_dominant(result, k: int = 10):
    """Return top-K worst layers with dominant role attribution.

    Returns list of (layer_name, output_qsnr, dominant_role, role_qsnrs).
    """
    qsnr_by_role = result.qsnr_by_role
    if not qsnr_by_role:
        return []

    all_layers = set()
    for role_map in qsnr_by_role.values():
        all_layers.update(role_map.keys())

    scored = []
    for layer in all_layers:
        output_qsnr = qsnr_by_role.get("output", {}).get(layer)
        if output_qsnr is None:
            continue
        role_qsnrs = {}
        for role in ("input", "weight", "output"):
            v = qsnr_by_role.get(role, {}).get(layer)
            if v is not None:
                role_qsnrs[role] = v

        dominant = min(role_qsnrs, key=role_qsnrs.get) if role_qsnrs else "output"
        scored.append((layer, output_qsnr, dominant, role_qsnrs))

    scored.sort(key=lambda x: x[1])
    return scored[:k]


def charts_precision_recovery(
    model, config, calib_data, eval_data, eval_fn,
    baseline_result, label: str = "MXInt8", top_k: int = 5,
):
    """⑨ Precision recovery: grouped bar showing baseline + restored + FP32 accuracy."""

    from src.scheme.op_config import OpQuantConfig

    fp32_metrics = baseline_result.fp32_metrics or {}
    quant_metrics = baseline_result.quant_metrics or {}

    # Total accuracy gap
    fp32_acc = fp32_metrics.get("accuracy")
    quant_acc = quant_metrics.get("accuracy")
    if fp32_acc is None or quant_acc is None:
        print("[bitx] Skipping precision recovery: no accuracy metrics")
        return

    total_gap = fp32_acc - quant_acc
    if abs(total_gap) < 1e-10:
        print("[bitx] Skipping precision recovery: no accuracy gap to recover")
        return

    worst = _worst_layers_with_dominant(baseline_result, k=top_k)
    if not worst:
        return

    print(f"\n[bitx] Precision recovery analysis (top-{top_k} layers → FP32)...")
    print(f"  Baseline: FP32={fp32_acc:.4f}, Quant={quant_acc:.4f}, Gap={total_gap:+.6f}")

    # Per-layer recovery: restore one layer at a time to FP32
    recovery_data = []
    for layer, output_qsnr, dominant, role_qsnrs in worst:
        # Create override: OpQuantConfig() = all None = FP32 (no quantization)
        override_cfg = OpQuantConfig()
        overrides = {layer: override_cfg}

        try:
            sess = Session(
                copy.deepcopy(model),
                config,
                observers=[QSNRObserver()],
                keep_fp32=True,
            )
            boosted_result = sess.run(
                calib_data,
                eval_data=eval_data,
                eval_fn=eval_fn,
                overrides=overrides,
            )
            boosted_acc = boosted_result.quant_metrics.get("accuracy", quant_acc)
        except Exception as e:
            print(f"  {layer}: failed ({e})")
            continue

        boosted_gap = fp32_acc - boosted_acc
        recovery_pct = (total_gap - boosted_gap) / total_gap * 100 if total_gap != 0 else 0

        print(f"  {layer}: quant={quant_acc:.4f} → boosted={boosted_acc:.4f} "
              f"(recovered {recovery_pct:.1f}% of gap, dominant={dominant})")

        recovery_data.append({
            "layer": layer,
            "accuracy": boosted_acc,
            "recovery_pct": round(recovery_pct, 1),
            "dominant_error": dominant,
        })

    if not recovery_data:
        return

    # ⑨ Precision recovery: grouped bar (baseline quant + restored + FP32)
    acc_data = [{"layer": "quant_baseline", "accuracy": quant_acc, "config": "Quantized"}]
    acc_data.append({"layer": "fp32_baseline", "accuracy": fp32_acc, "config": "FP32"})
    for r in recovery_data:
        acc_data.append({
            "layer": r["layer"],
            "accuracy": r["accuracy"],
            "config": "FP32 restore",
        })
    _chart(acc_data, "bar", x="layer", y="accuracy", hue="config",
           label=label,
           title="Precision Recovery (single-layer restore: each bar restores 1 layer to FP32)")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MXInt Error Analysis")
    parser.add_argument("--adapter", required=True,
                        help="Path to adapter.py (get_model, get_eval_fn, get_data)")
    parser.add_argument("--w-bits", type=int, default=W_BITS,
                        help=f"Weight bit width (default: {W_BITS})")
    parser.add_argument("--a-bits", type=int, default=A_BITS,
                        help=f"Activation bit width (default: {A_BITS})")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for per_block granularity (default: {BLOCK_SIZE})")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--skip-recovery", action="store_true",
                        help="Skip precision recovery ablation (saves time)")
    parser.add_argument("--recovery-top-k", type=int, default=5,
                        help="Number of worst layers for recovery analysis (default: 5)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load adapter ─────────────────────────────────────────────────
    print(f"[bitx] Loading adapter: {args.adapter}")
    adapter = load_adapter(args.adapter)

    model = adapter.get_model().to(device).eval()
    eval_fn = adapter.get_eval_fn()
    calib_data, eval_data = adapter.get_data()

    print(f"[bitx] Model: {type(model).__name__}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[bitx] Parameters: {param_count:,}")
    print(f"[bitx] Config: MXInt{args.w_bits} (w={args.w_bits}, a={args.a_bits}, block={args.block_size})")

    # ── Build config ─────────────────────────────────────────────────
    config = build_mxint_config(args.w_bits, args.a_bits, args.block_size)

    # ── Run quantization analysis ────────────────────────────────────
    print("[bitx] Running quantization + analysis...")
    session = Session(
        model,
        config,
        observers=[
            QSNRObserver(), MSEObserver(),
            DistributionObserver(), HistogramObserver(), PerBlockQSNRObserver(),
        ],
        keep_fp32=True,
    )
    result = session.run(
        calib_data,
        eval_data=eval_data,
        eval_fn=eval_fn,
    )

    # ── Print results ────────────────────────────────────────────────
    print("\n=== FP32 Metrics ===")
    if result.fp32_metrics:
        for k, v in result.fp32_metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n=== Quantized Metrics ===")
    if result.quant_metrics:
        for k, v in result.quant_metrics.items():
            print(f"  {k}: {v:.4f}")

    if result.delta:
        print("\n=== Delta (Quant - FP32) ===")
        for k, v in result.delta.items():
            print(f"  {k}: {v:+.6f}")

    print("\n=== Error Provenance ===")
    try:
        print(result.diagnose.summary())
    except Exception as e:
        print(f"(Error provenance unavailable: {e})")

    try:
        print(result.diagnose.per_role_table())
    except Exception as e:
        print(f"(Per-role table unavailable: {e})")

    # ── Render basic charts ──────────────────────────────────────────
    label = f"MXInt{args.w_bits}"
    print("\n[bitx] Generating charts...")
    charts_from_result(result, label=label,
                       model=model, block_size=args.block_size)

    # ── Precision recovery (⑨) ──────────────────────────────────────
    if not args.skip_recovery:
        charts_precision_recovery(
            model, config, calib_data, eval_data, eval_fn,
            baseline_result=result,
            label=label,
            top_k=args.recovery_top_k,
        )

    # ── Cost analysis ────────────────────────────────────────────────
    try:
        cost_report = analyze_model_cost(model)
        cost_report.print_summary()
    except Exception:
        pass

    # ── Layer-level diagnostics ─────────────────────────────────────
    from src.api.layer_diagnostic import (
        distribution_table, diagnosis_report,
    )

    print("\n[bitx] Running layer-level diagnostics...")
    distribution_table(result)
    diagnosis_report(result)

    # ── Harness charts (U2 + U6) ────────────────────────────────────
    from src.api.harness_charts import all_harness_charts
    print("\n[bitx] Generating harness charts (U2 + U6)...")
    all_harness_charts(result, label=label)

    print("\n[bitx] Analysis complete.")
    return result


if __name__ == "__main__":
    main()
