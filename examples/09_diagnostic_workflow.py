"""09 — End-to-End Quantisation Diagnostic Workflow.

Demonstrates the full ADR-010 diagnostic closed loop:
    diagnose → characterize → intervene → verify

Also showcases per-operator-type QuantConfig (different configs for
linear, add, etc.) alongside per-layer (per-module) overrides.

Model: ToyMLP (128→512→128→10) — small enough for rapid iteration.

Run:  PYTHONPATH=. python examples/09_diagnostic_workflow.py
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline._model import ToyMLP
from src.analysis.observers import (
    DistributionFitObserver,
    DistributionObserver,
    HistogramObserver,
    MSEObserver,
    QSNRObserver,
)
from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.session import Session, quantize_model
from src.session._config import QuantConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def accuracy_fn(model: nn.Module, data) -> dict:
    """Classification accuracy on a DataLoader, or forward pass on a tensor.

    ``Session.run()`` and ``InterventionAccessor.compare()`` pass the same
    *eval_fn* to calibrate (tensor forward) and evaluate (accuracy on loader),
    so this function handles both.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # DataLoader path — compute accuracy
        if isinstance(data, DataLoader):
            correct = total = 0
            for x, y in data:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)
            return {"accuracy": correct / total if total > 0 else 0.0}

        # Tensor / list path — run a forward pass (calibration)
        if isinstance(data, (list, tuple)):
            for batch in data:
                model(batch.to(device) if isinstance(batch, torch.Tensor) else batch)
        else:
            model(data.to(device) if isinstance(data, torch.Tensor) else data)
        return {}


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: SETUP — Per-Layer + Per-Op-Type Configuration
# ═══════════════════════════════════════════════════════════════════════════

section("PHASE 1: SETUP — Per-Layer + Per-Operator-Type Configuration")

per_t = GranularitySpec.per_tensor()
per_c_w = GranularitySpec.per_channel(axis=0)

i4_s = QuantScheme(format=FormatBase.from_str("int4"), granularity=per_c_w)
i8_s = QuantScheme(format=FormatBase.from_str("int8"), granularity=per_t)

# ── Per-layer (module-level) config ──
# Matches module names via exact match or glob pattern ("fc*").
layer_cfg: dict = {
    "fc1":  OpQuantConfig(weight=i4_s),
    "fc2":  OpQuantConfig(weight=i4_s),
    "head": OpQuantConfig(weight=i8_s),  # classification head gets higher precision
}

# ── Per-op-type (inline-op-level) config ──
# Valid keys: matmul, mm, bmm, linear, add, sub, mul, div, exp, log
op_cfgs: dict = {
    "add": OpQuantConfig(input=i8_s, output=i8_s),
}

print("Per-layer configs:")
for name, cfg in layer_cfg.items():
    w_fmt = str(cfg.weight.format) if cfg.weight else "?"
    w_grn = str(cfg.weight.granularity) if cfg.weight else "?"
    print(f"  {name}: weight={w_fmt} {w_grn}")
print(f"\nPer-op configs:")
for op, cfg in op_cfgs.items():
    print(f"  {op}: input={cfg.input.format} output={cfg.output.format}")

# Apply quantize_model() with both config levels
model_p1 = ToyMLP()
model_p1 = quantize_model(model_p1, cfg=layer_cfg, op_cfgs=op_cfgs)

# Verify forward pass
x = torch.randn(4, 128)
with torch.no_grad():
    y = model_p1(x)
print(f"\nForward pass: input {list(x.shape)} → output {list(y.shape)}")

# List quantized modules
qmodules = [n for n, _ in model_p1.named_modules()
            if n and hasattr(type(_), '__name__') and 'Quantized' in type(_).__name__]
print(f"Quantized module count: {len(qmodules)}")
if qmodules:
    print(f"  e.g. {qmodules[0]}, {qmodules[1]}, ...")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: DIAGNOSE — Error Provenance & QSNR Attribution
# ═══════════════════════════════════════════════════════════════════════════

section("PHASE 2: DIAGNOSE — Error Provenance & QSNR Attribution")

# int4 weight-only per_channel — enough degradation to demonstrate the pipeline
cfg = QuantConfig(
    name="int4-pc-wo",
    w_format="int4",
    w_granularity="per_channel",
    weight_only=True,
    calibrator="mse",
)

# Calibration data (single batch) and evaluation loader
calib_data = torch.randn(128, 128)

eval_x = torch.randn(256, 128)
eval_y = torch.randint(0, 10, (256,))
eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=16)

# Full observer suite
observers = [
    QSNRObserver(),
    MSEObserver(),
    DistributionObserver(),
    HistogramObserver(),
    DistributionFitObserver(),
]

model_p2 = ToyMLP()
session = Session(
    model_p2,
    config=cfg,
    keep_fp32=True,
    observers=observers,
)

output_keys = [
    "accuracy", "qsnr", "mse", "distribution", "histogram", "fit",
]

# Stepwise API — avoids eval_fn conflation between calibrate (model forward)
# and evaluate (accuracy computation).
session.quantize(calib_data=calib_data)
session.calibrate(calib_data)
session.analyze(calib_data, outputs=output_keys)
session.evaluate(eval_loader, accuracy_fn)
result = session.result

# ── Terminal output ──
print(f"\n--- Accuracy ---")
print(result.accuracy_table())

print(f"\n--- QSNR Summary (by role × layer type) ---")
print(result.diagnose.summary())

print(f"\n--- Per-Role Table (worst layers first) ---")
print(result.diagnose.per_role_table(max_layers=20))

print(f"\n--- Top-5 Worst Layers (weight role) ---")
for name, qsnr in result.diagnose.top_k(5, role="weight"):
    print(f"  {name:<35s} {qsnr:.1f} dB")

print(f"\n--- Error Source Analysis (output role) ---")
print(result.tables.error_source_analysis(role="output"))

# Layer report DataFrame (optional, requires pandas)
df = result.layer_report()
if df is not None:
    print(f"\n--- Layer Report (pandas) ---")
    print(df.sort_values("qsnr_db").head(8).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: CHARACTERIZE — Distribution Features & Degradation Mechanisms
# ═══════════════════════════════════════════════════════════════════════════

section("PHASE 3: CHARACTERIZE — Distribution Diagnosis & Causal Analysis")

print(result.characterize.causal_analysis())

# Profile the worst-weight-QSNR layers
worst_weight = result.diagnose.top_k(3, role="weight")
for layer, qsnr in worst_weight:
    print(f"\n--- Profile: {layer} (weight, QSNR={qsnr:.1f} dB) ---")
    print(result.characterize.profile(layer, role="weight"))

# Also profile worst output role layers
worst_output = result.diagnose.top_k(3, role="output")
for layer, qsnr in worst_output:
    print(f"\n--- Profile: {layer} (output, QSNR={qsnr:.1f} dB) ---")
    print(result.characterize.profile(layer, role="output"))

# Distribution taxonomy
print(f"\n--- Distribution Taxonomy ---")
result.report.taxonomy.print()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: VISUALIZE — Diagnostic Figures
# ═══════════════════════════════════════════════════════════════════════════

section("PHASE 4: VISUALIZE — Diagnostic Figures")

out_dir = tempfile.mkdtemp(prefix="quant_diag_")
print(f"Saving figures to: {out_dir}")

# Find the worst layer name for per-layer histogram
worst_layer_name = worst_weight[0][0] if worst_weight else list(result.qsnr_per_layer.keys())[0]

figure_specs: list[tuple[str, Callable[[], plt.Figure]]] = [
    ("01_qsnr_comparison",       lambda: result.plot.qsnr_comparison()),
    ("02_crest_vs_qsnr",         lambda: result.plot.crest_vs_qsnr()),
    ("03_outlier_analysis",      lambda: result.plot.outlier_analysis()),
    ("04_correlation_heatmap",   lambda: result.plot.correlation_heatmap()),
    ("05_per_role_qsnr_bars",    lambda: result.plot.per_role_qsnr_bars()),
    ("06_error_propagation",     lambda: result.plot.error_propagation()),
    ("07_role_distribution",     lambda: result.plot.role_distribution_comparison()),
    ("08_layer_histogram",       lambda: result.plot.layer_histogram(
        worst_layer_name, role="weight")),
]

for name, fig_fn in figure_specs:
    try:
        fig = fig_fn()
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}.png  OK")
    except Exception as exc:
        print(f"  {name}.png  FAILED  ({exc})")

# SmoothQuant distrib comparison is only available with transform="smoothquant"
# — not collected in this int4-pc-wo run.


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: INTERVENE — Generate Precision-Boost Plans
# ═══════════════════════════════════════════════════════════════════════════

section("PHASE 5: INTERVENE — Precision Boost & Transform Plans")

# top_k_boost: fix the 3 worst-weight layers by raising to int8
plan_w = result.plan.top_k_boost(k=3, role="weight", target_bits=8)
print(f"--- Top-3 Weight Boost → int8 ---")
print(plan_w.explain())

# recommend: QSNR-based conservative strategy
plan_rec = result.plan.recommend(strategy="conservative")
print(f"\n--- Recommend (conservative) ---")
print(plan_rec.explain())

# transform_ranking (known stub — see ADR-010)
print(f"\n--- Transform Ranking ---")
print(result.plan.transform_ranking(k=5))
print("  [GAP] transform_ranking — requires original model + calib_data in "
      "SessionResult; stub until implemented")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: COMPARE — Apply Plan & Verify Improvement
# ═══════════════════════════════════════════════════════════════════════════

section("PHASE 6: COMPARE — Before/After Intervention")

comparison = result.intervention.compare(
    ToyMLP(),
    calib_data,
    plan_w,
    eval_data=eval_loader,
    eval_fn=accuracy_fn,
)
print(comparison.summary())

print("\n  [GAP] InterventionComparison.plot — per-layer before/after "
      "QSNR chart not yet available")
print("  [GAP] InterventionComparison — per-layer role distribution "
      "comparison not yet available")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

section("SUMMARY")

print(f"""
Figures saved to: {out_dir}

Capability coverage (ADR-010 diagnostic closed loop):
  diagnose (ErrorProvenance):            OK  summary, per_role_table, top_k, error_source
  characterize (DistributionDiagnosis):  OK  causal_analysis, profile, taxonomy
  intervene (InterventionPlanner):       OK  top_k_boost, recommend
                                         GAP transform_ranking (stub)
  compare (InterventionAccessor):        OK  compare + summary table
                                         GAP plot accessor, per-layer role comparison

Next steps:
  1. Implement transform_ranking (requires model + calib_data in SessionResult)
  2. Add InterventionComparison.plot for before/after QSNR charts
  3. Run with transform='smoothquant' to collect smoothquant_distrib output
""")
