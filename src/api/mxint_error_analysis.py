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
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import torch
import torch.nn as nn

from src.session import Session, QuantConfig
from src.analysis.observers import QSNRObserver, MSEObserver
from src.cost.model_cost import analyze_model_cost

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


def charts_from_result(result, label: str = "MXInt8"):
    """Generate charts from SessionResult via render_chart."""

    # ① Per-layer QSNR (bar)
    if result.qsnr_per_layer:
        data = [
            {"layer": k, "qsnr_db": v}
            for k, v in result.qsnr_per_layer.items()
        ]
        _chart(data, "bar", x="layer", y="qsnr_db",
               label=label, title="Per-Layer QSNR (dB)")

    # ② Per-layer MSE (bar)
    if result.mse_per_layer:
        data = [
            {"layer": k, "mse": v}
            for k, v in result.mse_per_layer.items()
        ]
        _chart(data, "bar", x="layer", y="mse",
               label=label, title="Per-Layer MSE")

    # ③ Error propagation: local vs accumulated (line)
    if result.qsnr_per_layer:
        layers = list(result.qsnr_per_layer.keys())
        data = []
        for i, layer in enumerate(layers):
            local = result.qsnr_per_layer[layer]
            accum = result.accum_qsnr_per_layer.get(layer, 0)
            data.append({"layer_idx": i, "layer": layer, "qsnr_db": local, "type": "local"})
            data.append({"layer_idx": i, "layer": layer, "qsnr_db": accum, "type": "accumulated"})
        if data:
            _chart(data, "line", x="layer_idx", y="qsnr_db", hue="type",
                   label=label, title="Error Propagation: Local vs Accumulated QSNR")

    # ④ Per-role QSNR grouped bar (input / weight / output)
    if result.qsnr_by_role:
        data = []
        for role, layer_map in result.qsnr_by_role.items():
            for layer, qsnr in layer_map.items():
                data.append({"layer": layer, "role": role, "qsnr_db": qsnr})
        if data:
            _chart(data, "bar", x="layer", y="qsnr_db", hue="role",
                   label=label, title="Per-Layer Per-Role QSNR (dB)")

    # ⑤ Cost decomposition (FLOPs per layer)
    if result.cost:
        cost_rows = result.cost.to_dataframe()
        if cost_rows:
            _chart(cost_rows, "bar", x="op_name", y="flops_math",
                   label=label, title="Math FLOPs per Layer")

    # ⑥ Top-K worst layers
    try:
        top_k = result.diagnose.top_k(10, role="output")
        if top_k:
            data = [{"layer": n, "qsnr_db": v} for n, v in top_k]
            _chart(data, "bar", x="layer", y="qsnr_db",
                   label=label, title="Top-10 Worst Layers by QSNR (dB)")
    except Exception:
        pass

    # ⑦ Summary table
    summary_rows = []
    if result.fp32_metrics:
        for k, v in result.fp32_metrics.items():
            summary_rows.append({"metric": k, "fp32": v,
                                 "quant": result.quant_metrics.get(k, ""),
                                 "delta": result.delta.get(k, "")})
    if summary_rows:
        _chart(summary_rows, "table",
               x="metric", y="fp32",
               label=label, title="Accuracy Summary")


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
        observers=[QSNRObserver(), MSEObserver()],
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

    # ── Render charts ────────────────────────────────────────────────
    print("\n[bitx] Generating charts...")
    charts_from_result(result, label=f"MXInt{args.w_bits}")

    # ── Cost analysis ────────────────────────────────────────────────
    try:
        cost_report = analyze_model_cost(model)
        cost_report.print_summary()
    except Exception:
        pass

    print("\n[bitx] Analysis complete.")
    return result


if __name__ == "__main__":
    main()
