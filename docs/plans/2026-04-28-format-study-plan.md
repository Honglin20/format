# Format Precision Study — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single experiment script that runs all 4 Parts (A/B/C/D) of the format precision study, producing 6 tables + 11 figures as specified in the design doc.

**Architecture:** Model-agnostic framework where users inject their own `build_model`/`calib_data`/`eval_fn`. Internally uses `QuantSession` for each config, `AnalysisContext`+Observers for per-layer data, `matplotlib`+`seaborn` for visualization. Results saved to a timestamped `results/` directory.

**Tech Stack:** PyTorch, existing src/ APIs (QuantSession, AnalysisContext, Observers, compare_sessions), matplotlib, seaborn

**Design doc:** `docs/plans/2026-04-28-format-study-design.md`

---

### Task 1: Experiment script skeleton and user-facing API

**Files:**
- Create: `examples/experiment_format_study.py`

**Goal:** Define the top-level function signature and helper config builders. No experiment logic yet — just the API surface and format/granularity helpers.

**Step 1: Write the skeleton**

Create `examples/experiment_format_study.py` with:

```python
#!/usr/bin/env python3
"""
Quantization Format Precision Study
====================================
Systematic comparison of MXINT/MXFP/INT-PC/NF4-PC at 8-bit and 4-bit,
with FP32 vs PoT scaling comparison and SmoothQuant/Hadamard transform analysis.

Usage:
    PYTHONPATH=. python examples/experiment_format_study.py

Custom model: Edit the `build_model`, `make_calib_data`, `make_eval_loader`,
and `eval_fn` functions at the top of the file.
"""

import copy
import os
import json
import math
from datetime import datetime
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project imports
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform
from src.transform.pre_scale import PreScaleTransform
from src.analysis.observers import (
    QSNRObserver, MSEObserver, HistogramObserver, DistributionObserver
)
from src.analysis.report import Report
from src.analysis.correlation import LayerSensitivity, ErrorByDistribution
from src.analysis.e2e import compare_sessions
from src.calibration.quant_session import QuantSession
from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
from src.calibration.strategies import ScaleStrategy

# ---------------------------------------------------------------------------
# User customization: override these functions for your own model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """Return a fresh instance of your FP32 model."""
    from examples._model import ToyMLP
    return ToyMLP()

def make_calib_data(num_samples: int = 256, batch_size: int = 16) -> List[torch.Tensor]:
    """Return calibration data as a list of tensors (one per batch)."""
    return [torch.randn(batch_size, 128) for _ in range(num_samples // batch_size)]

def make_eval_loader(num_samples: int = 512, batch_size: int = 16) -> DataLoader:
    """Return a DataLoader for evaluation."""
    x = torch.randn(num_samples, 128)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)

def eval_fn(model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate model on dataloader, return dict of metric_name -> value."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

# Shared granularity specs
PER_T = GranularitySpec.per_tensor()
PER_C0 = GranularitySpec.per_channel(axis=0)
PER_Cm1 = GranularitySpec.per_channel(axis=-1)
PER_B32 = GranularitySpec.per_block(size=32, axis=-1)

def make_op_cfg(fmt_name: str, granularity, *, transform=None) -> OpQuantConfig:
    """Create inference-only OpQuantConfig: input + weight + output all use same scheme."""
    fmt = FormatBase.from_str(fmt_name)
    scheme = QuantScheme(format=fmt, granularity=granularity, transform=transform)
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)

def make_op_cfg_weight_only(fmt_name: str, granularity, *, transform=None) -> OpQuantConfig:
    """Weight-only quantization (for NF4)."""
    fmt = FormatBase.from_str(fmt_name)
    scheme = QuantScheme(format=fmt, granularity=granularity, transform=transform)
    return OpQuantConfig(weight=scheme)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: OpQuantConfig,
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
    observers: Optional[List] = None,
    *,
    lsq_steps: int = 0,
    lsq_pot: bool = False,
) -> dict:
    """Run a single experiment configuration and return results."""
    ...


def run_format_study(
    build_model: Callable,
    make_calib_data: Callable,
    make_eval_loader: Callable,
    eval_fn: Callable,
    *,
    output_dir: str = None,
) -> Dict[str, dict]:
    """Main entry point: run all experiments and produce all tables/figures."""
    ...


if __name__ == "__main__":
    results = run_format_study(build_model, make_calib_data, make_eval_loader, eval_fn)
    print("Study complete. Results saved to results/")
```

**Step 2: Verify the skeleton imports and runs without errors**

```bash
PYTHONPATH=. python -c "import examples.experiment_format_study; print('Imports OK')"
```

Expected: `Imports OK` printed, no ImportError.

**Step 3: Commit**

---

### Task 2: Implement `run_experiment()` — single config runner

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Implement the core `run_experiment()` function that creates a `QuantSession`, calibrates, analyzes with observers, and evaluates E2E accuracy.

**Step 1: Implement `run_experiment()`**

```python
def run_experiment(
    cfg: OpQuantConfig,
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_loader: DataLoader,
    observers: Optional[List] = None,
    *,
    lsq_steps: int = 0,
    lsq_pot: bool = False,
    lsq_lr: float = 1e-3,
) -> dict:
    """
    Run one experiment config.

    Args:
        cfg: OpQuantConfig for the experiment.
        fp32_model: Reference FP32 model (will be deep-copied).
        calib_data: List of calibration batches.
        eval_loader: Evaluation DataLoader.
        observers: List of Observer instances (default: QSNR + MSE).
        lsq_steps: If > 0, run LSQ pre-scale optimization for this many steps.
        lsq_pot: If True, constrain pre-scale to power-of-two during LSQ.

    Returns:
        dict with keys: 'accuracy' (per metric), 'report' (Report object),
        'session' (QuantSession), 'qsnr_per_layer', 'mse_per_layer'.
    """
    if observers is None:
        observers = [QSNRObserver(), MSEObserver()]

    model_copy = copy.deepcopy(fp32_model)
    session = QuantSession(
        model_copy, cfg,
        calibrator=ScaleStrategy.MSE(),
        observers=observers,
        keep_fp32=True,
    )

    # 1. Calibrate
    with session.calibrate():
        for batch in calib_data:
            session(batch)

    # 2. Optional LSQ pre-scale optimization
    if lsq_steps > 0:
        session.initialize_pre_scales(calib_data, init="ones", pot=lsq_pot)
        opt = LayerwiseScaleOptimizer(
            num_steps=lsq_steps, num_batches=len(calib_data),
            optimizer="adam", lr=lsq_lr, pot=lsq_pot,
        )
        session.optimize_scales(opt, calib_data)

    # 3. Analyze with observers
    with session.analyze(observers=observers) as ctx:
        for batch in calib_data:
            session(batch)
    report = ctx.report()

    # 4. E2E accuracy
    result = session.compare(eval_loader, eval_fn=eval_fn)

    # 5. Extract per-layer QSNR/MSE summaries
    qsnr_per_layer = _extract_metric_per_layer(report, "qsnr_db")
    mse_per_layer = _extract_metric_per_layer(report, "mse")

    return {
        "accuracy": result["quant"],
        "fp32_accuracy": result["fp32"],
        "delta": result["delta"],
        "report": report,
        "session": session,
        "qsnr_per_layer": qsnr_per_layer,
        "mse_per_layer": mse_per_layer,
    }


def _extract_metric_per_layer(report: Report, metric: str) -> Dict[str, float]:
    """Extract per-layer average of a metric from Report."""
    df = report.to_dataframe()
    if isinstance(df, list):
        # pandas not available — aggregate manually
        result = {}
        for row in df:
            name = row.get("layer", "unknown")
            val = row.get(metric)
            if val is not None:
                result.setdefault(name, []).append(val)
        return {k: sum(v)/len(v) for k, v in result.items()}
    else:
        grouped = df.groupby("layer")[metric].mean()
        return grouped.to_dict()
```

**Step 2: Verify with a quick smoke test using ToyMLP**

```bash
PYTHONPATH=. python -c "
from examples.experiment_format_study import *
from examples._model import ToyMLP

model = ToyMLP()
calib = [torch.randn(16, 128) for _ in range(4)]
dl = make_eval_loader(64, 16)
cfg = make_op_cfg('int8', PER_T)
res = run_experiment(cfg, model, calib, dl)
print('Accuracy:', res['accuracy'])
print('QSNR layers:', list(res['qsnr_per_layer'].keys()))
print('OK')
"
```

Expected: prints accuracy and layer names, no errors.

**Step 3: Commit**

---

### Task 3: Part A — 8-bit format comparison

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Implement `run_part_a_8bit()` — run the 3 configs (MXINT-8, MXFP-8, INT8-PC) and collect results.

**Step 1: Implement Part A**

```python
def run_part_a_8bit(fp32_model, calib_data, eval_loader) -> dict:
    """Part A: Compare MXINT-8, MXFP-8, INT8-PC (all 8-bit, PoT scaling)."""
    configs = {
        "MXINT-8":  make_op_cfg("int8", PER_B32),
        "MXFP-8":   make_op_cfg("fp8_e4m3", PER_B32),
        "INT8-PC":  make_op_cfg("int8", PER_C0),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"  Running {name}...")
        results[name] = run_experiment(cfg, fp32_model, calib_data, eval_loader)
    # Add FP32 baseline
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results
```

**Step 2: Smoke test**

```bash
PYTHONPATH=. python -c "
from examples.experiment_format_study import *
m = ToyMLP(); calib = [torch.randn(16, 128) for _ in range(4)]
dl = make_eval_loader(64, 16)
res = run_part_a_8bit(m, calib, dl)
for k, v in res.items():
    print(k, '->', {mk: f'{mv:.3f}' for mk, mv in v['accuracy'].items()} if 'accuracy' in v else 'baseline')
"
```

Expected: 3+1 config results printed, no errors.

**Step 3: Commit**

---

### Task 4: Part B — 4-bit format comparison

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Implement `run_part_b_4bit()` — MXINT-4, MXFP-4, INT4-PC, NF4-PC.

**Step 1: Implement Part B**

```python
def run_part_b_4bit(fp32_model, calib_data, eval_loader) -> dict:
    """Part B: Compare MXINT-4, MXFP-4, INT4-PC, NF4-PC (all 4-bit)."""
    configs = {
        "MXINT-4":  make_op_cfg("int4", PER_B32),
        "MXFP-4":   make_op_cfg("fp4_e2m1", PER_B32),
        "INT4-PC":  make_op_cfg("int4", PER_C0),
        "NF4-PC":   make_op_cfg_weight_only("nf4", PER_C0),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"  Running {name}...")
        results[name] = run_experiment(cfg, fp32_model, calib_data, eval_loader)
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results
```

**Step 2: Smoke test**

```bash
PYTHONPATH=. python -c "
from examples.experiment_format_study import *
m = ToyMLP(); calib = [torch.randn(16, 128) for _ in range(4)]
dl = make_eval_loader(64, 16)
res = run_part_b_4bit(m, calib, dl)
for k, v in res.items():
    print(k, v.get('accuracy', 'baseline'))
"
```

**Step 3: Commit**

---

### Task 5: Part C — FP32 vs PoT scaling comparison

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Implement `run_part_c_pot_scaling()` — INT8-PC and INT4-PC with LSQ, comparing fp32 scale vs PoT scale.

**Step 1: Implement Part C**

```python
def run_part_c_pot_scaling(fp32_model, calib_data, eval_loader) -> dict:
    """Part C: INT per-channel with FP32 scaling vs PoT scaling (8-bit & 4-bit)."""
    configs = {
        "INT8-PC-FP32": ("int8", False),
        "INT8-PC-PoT":  ("int8", True),
        "INT4-PC-FP32": ("int4", False),
        "INT4-PC-PoT":  ("int4", True),
    }
    results = {}
    for name, (fmt, pot) in configs.items():
        print(f"  Running {name} (LSQ, pot={pot})...")
        cfg = make_op_cfg(fmt, PER_C0)
        results[name] = run_experiment(
            cfg, fp32_model, calib_data, eval_loader,
            lsq_steps=100, lsq_pot=pot,
        )
    results["FP32 (baseline)"] = {
        "accuracy": results[list(configs.keys())[0]]["fp32_accuracy"],
    }
    return results
```

**Step 2: Smoke test**

```bash
PYTHONPATH=. python -c "
from examples.experiment_format_study import *
m = ToyMLP(); calib = [torch.randn(16, 128) for _ in range(4)]
dl = make_eval_loader(64, 16)
res = run_part_c_pot_scaling(m, calib, dl)
for k, v in res.items():
    print(k, v.get('accuracy', 'baseline'))
"
```

**Step 3: Commit**

---

### Task 6: Part D — Transform study at 4-bit

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Implement `run_part_d_transforms()`. For each 4-bit format, run 3 transform variants (Identity, SmoothQuant, Hadamard), then construct per-layer optimal heterogeneous config and run it as the 4th variant.

**Step 1: Implement Part D**

```python
def run_part_d_transforms(fp32_model, calib_data, eval_loader) -> dict:
    """Part D: Evaluate SmoothQuant and Hadamard transforms at 4-bit,
    with per-layer optimal transform selection."""
    fmt_configs = {
        "MXINT-4": ("int4", PER_B32),
        "MXFP-4":  ("fp4_e2m1", PER_B32),
        "INT4-PC": ("int4", PER_C0),
        "NF4-PC":  ("nf4", PER_C0, True),   # weight-only
    }

    transform_variants = {
        "None":      None,
        "SmoothQuant": "smooth",
        "Hadamard":    "hadamard",
    }

    all_results = {}

    for fmt_name, fmt_args in fmt_configs.items():
        print(f"\n  == Transform study for {fmt_name} ==")
        weight_only = len(fmt_args) > 2 and fmt_args[2]
        fmt_str, gran = fmt_args[0], fmt_args[1]
        builder = make_op_cfg_weight_only if weight_only else make_op_cfg

        # Phase 1: Run each transform variant
        variant_results = {}
        for tx_name, tx_spec in transform_variants.items():
            label = f"{fmt_name}-{tx_name}"
            print(f"    Running {label}...")
            transform = _make_transform(tx_spec, fp32_model, calib_data)
            cfg = builder(fmt_str, gran, transform=transform)
            variant_results[tx_name] = run_experiment(
                cfg, fp32_model, calib_data, eval_loader,
            )

        # Phase 2: Per-layer optimal — pick best QSNR transform per layer
        per_layer_optimal = _build_per_layer_optimal(
            variant_results, fp32_model, calib_data,
            fmt_str, gran, builder,
        )
        print(f"    Running {fmt_name}-PerLayerOpt...")
        opt_result = run_experiment(
            per_layer_optimal, fp32_model, calib_data, eval_loader,
        )
        variant_results["PerLayerOpt"] = opt_result

        all_results[fmt_name] = variant_results

    return all_results


def _make_transform(tx_spec, fp32_model, calib_data) -> Optional[object]:
    """Create a transform instance from spec."""
    if tx_spec is None:
        return None  # IdentityTransform is default
    elif tx_spec == "hadamard":
        return HadamardTransform()
    elif tx_spec == "smooth":
        # Use first calibration batch as activation sample
        x_sample = calib_data[0]
        with torch.no_grad():
            fp32_model.eval()
            # Get fc1 weight from the model
            # Actually, from_calibration needs X_act and W
            # We extract act and weight from first Linear layer
            act = fp32_model.ln(x_sample) if hasattr(fp32_model, 'ln') else x_sample
            for name, module in fp32_model.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight.data
                    break
        return SmoothQuantTransform.from_calibration(
            X_act=act, W=weight, alpha=0.5
        )
    else:
        raise ValueError(f"Unknown transform spec: {tx_spec}")


def _build_per_layer_optimal(
    variant_results: dict,
    fp32_model, calib_data,
    fmt_str: str, gran, cfg_builder,
) -> OpQuantConfig:
    """Build a per-layer OpQuantConfig dict choosing the best transform per layer by QSNR."""
    # Determine which transform is best per layer
    # variant_results: {"None": result, "SmoothQuant": result, "Hadamard": result}
    # Each result has qsnr_per_layer: {layer_name: avg_qsnr}

    layer_best_tx = {}
    layers = set()
    for tx_name, res in variant_results.items():
        layers.update(res["qsnr_per_layer"].keys())

    for layer in sorted(layers):
        best_tx = None
        best_qsnr = -float("inf")
        for tx_name in ["None", "SmoothQuant", "Hadamard"]:
            q = variant_results[tx_name]["qsnr_per_layer"].get(layer, -float("inf"))
            if q > best_qsnr:
                best_qsnr = q
                best_tx = tx_name
        layer_best_tx[layer] = best_tx

    # Build per-layer config dict
    # For SmoothQuant layers, need per-layer SmoothQuant transform
    # For now, use a single SmoothQuant transform for all SQ-chosen layers
    # (the transform is embedded in the QuantScheme per config entry)
    per_layer_cfg = {}
    smooth_transform = _make_transform("smooth", fp32_model, calib_data)
    hadamard_transform = _make_transform("hadamard", fp32_model, calib_data)

    tx_map = {
        "None": None,
        "SmoothQuant": smooth_transform,
        "Hadamard": hadamard_transform,
    }

    # Group layers by chosen transform to minimize unique OpQuantConfigs
    tx_groups = defaultdict(list)
    for layer, tx_name in layer_best_tx.items():
        tx_groups[tx_name].append(layer)

    for tx_name, layer_list in tx_groups.items():
        cfg = cfg_builder(fmt_str, gran, transform=tx_map[tx_name])
        for layer in layer_list:
            per_layer_cfg[layer] = cfg

    return per_layer_cfg
```

**Step 2: Smoke test**

```bash
PYTHONPATH=. python -c "
from examples.experiment_format_study import *
m = ToyMLP(); calib = [torch.randn(16, 128) for _ in range(4)]
dl = make_eval_loader(64, 16)
res = run_part_d_transforms(m, calib, dl)
for fmt_name, variants in res.items():
    for tx_name, r in variants.items():
        print(f'{fmt_name}-{tx_name}:', r.get('accuracy', 'N/A'))
"
```

**Step 3: Commit**

---

### Task 7: Table generation (Tables 1-6)

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Implement functions to produce all 6 tables as formatted strings (console output) and CSV files.

**Step 1: Implement table functions**

```python
def generate_table_1(part_a_results: dict) -> str:
    """Table 1: 8-bit format × accuracy + avg QSNR/MSE."""
    ...

def generate_table_2(part_b_results: dict) -> str:
    """Table 2: 4-bit format × accuracy + avg QSNR/MSE."""
    ...

def generate_table_3(part_c_results: dict) -> str:
    """Table 3: FP32 vs PoT accuracy delta (8-bit & 4-bit)."""
    ...

def generate_table_4(part_d_results: dict) -> str:
    """Table 4: Format × Transform matrix (E2E accuracy + avg QSNR)."""
    ...

def generate_table_5(part_d_results: dict) -> str:
    """Table 5: Per-layer optimal transform distribution."""
    ...

def generate_table_6(all_results: dict) -> str:
    """Table 6: Layer sensitivity top-10 ranking."""
    ...
```

Each function:
1. Takes the results dict from the corresponding Part
2. Formats a markdown-style table string (also writes CSV to output dir)
3. Returns the formatted string

**Step 2: Verify on ToyMLP**

Run a mini integration test that exercises all table functions.

**Step 3: Commit**

---

### Task 8: Figure generation (Figs 1-11)

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Implement matplotlib/seaborn plotting functions. Each figure saves to the output `figures/` subdirectory as both PNG and PDF.

**Step 1: Implement plotting functions**

```python
def plot_fig1_qsnr_8bit(part_a_results, output_dir):
    """Fig 1: 8-bit per-layer QSNR line chart (3 lines)."""
    ...

def plot_fig2_qsnr_4bit(part_b_results, output_dir):
    """Fig 2: 4-bit per-layer QSNR line chart (4 lines)."""
    ...

def plot_fig3_mse_box_8bit(part_a_results, output_dir):
    """Fig 3: 8-bit per-layer MSE boxplot (3 groups)."""
    ...

def plot_fig4_mse_box_4bit(part_b_results, output_dir):
    """Fig 4: 4-bit per-layer MSE boxplot (4 groups)."""
    ...

def plot_fig5_pot_delta(part_c_results, output_dir):
    """Fig 5: FP32 vs PoT ΔQSNR bar chart (8bit + 4bit side by side)."""
    ...

def plot_fig6_histogram_overlay(all_results, output_dir):
    """Fig 6: Key layers fp32/quant/error 3-channel histogram overlay (3-5 layers)."""
    ...

def plot_fig7_transform_heatmap(part_d_results, output_dir):
    """Fig 7: Format × Transform heatmap of E2E accuracy."""
    ...

def plot_fig8_transform_pie(part_d_results, output_dir):
    """Fig 8: Per-layer optimal transform distribution pie chart."""
    ...

def plot_fig9_transform_delta(part_d_results, output_dir):
    """Fig 9: Transform ΔQSNR vs baseline bar chart."""
    ...

def plot_fig10_error_vs_distribution(all_results, output_dir):
    """Fig 10: QSNR vs distribution features scatter matrix."""
    ...

def plot_fig11_layer_type_qsnr(all_results, output_dir):
    """Fig 11: Layer-type grouped QSNR comparison."""
    ...
```

Plotting conventions:
- Use `matplotlib` rcParams for consistent font sizes (title=14, label=12, tick=10)
- Save both `.png` (300 dpi) and `.pdf` to `figures/`
- Use `seaborn` style: `sns.set_style("whitegrid")`
- Color palettes: `Set2` for format comparisons, `RdBu` for deltas

**Step 2: Verify on ToyMLP results**

```bash
PYTHONPATH=. python -c "
from examples.experiment_format_study import *
# Run all parts and generate figures
...
"
```

**Step 3: Commit**

---

### Task 9: Main entry point integration + `run_format_study()`

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Wire everything together in `run_format_study()` — runs all 4 Parts, generates all tables and figures, saves results JSON.

**Step 1: Implement `run_format_study()`**

```python
def run_format_study(
    build_model: Callable = build_model,
    make_calib_data: Callable = make_calib_data,
    make_eval_loader: Callable = make_eval_loader,
    eval_fn: Callable = eval_fn,
    *,
    output_dir: str = None,
) -> Dict[str, dict]:
    """Run the complete format precision study."""
    if output_dir is None:
        output_dir = f"results/format_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)

    print("=" * 60)
    print("Format Precision Study")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Build model and data
    fp32_model = build_model()
    fp32_model.eval()
    calib_data = make_calib_data()
    eval_loader = make_eval_loader()

    all_results = {}

    # Part A: 8-bit format comparison
    print("\n### Part A: 8-bit Format Comparison ###")
    all_results["part_a"] = run_part_a_8bit(fp32_model, calib_data, eval_loader)
    print(generate_table_1(all_results["part_a"]))

    # Part B: 4-bit format comparison
    print("\n### Part B: 4-bit Format Comparison ###")
    all_results["part_b"] = run_part_b_4bit(fp32_model, calib_data, eval_loader)
    print(generate_table_2(all_results["part_b"]))

    # Part C: FP32 vs PoT scaling
    print("\n### Part C: FP32 vs PoT Scaling ###")
    all_results["part_c"] = run_part_c_pot_scaling(fp32_model, calib_data, eval_loader)
    print(generate_table_3(all_results["part_c"]))

    # Part D: Transform study
    print("\n### Part D: Transform Study at 4-bit ###")
    all_results["part_d"] = run_part_d_transforms(fp32_model, calib_data, eval_loader)
    print(generate_table_4(all_results["part_d"]))
    print(generate_table_5(all_results["part_d"]))

    # Layer sensitivity
    print("\n### Layer Sensitivity ###")
    all_results["sensitivity"] = _compute_sensitivity(all_results)
    print(generate_table_6(all_results["sensitivity"]))

    # Generate figures
    print("\n### Generating Figures ###")
    plot_fns = [
        (plot_fig1_qsnr_8bit, "fig1_qsnr_8bit"),
        (plot_fig2_qsnr_4bit, "fig2_qsnr_4bit"),
        (plot_fig3_mse_box_8bit, "fig3_mse_8bit"),
        (plot_fig4_mse_box_4bit, "fig4_mse_4bit"),
        (plot_fig5_pot_delta, "fig5_pot_delta"),
        (plot_fig6_histogram_overlay, "fig6_histogram"),
        (plot_fig7_transform_heatmap, "fig7_transform_heatmap"),
        (plot_fig8_transform_pie, "fig8_transform_pie"),
        (plot_fig9_transform_delta, "fig9_transform_delta"),
        (plot_fig10_error_vs_distribution, "fig10_error_vs_dist"),
        (plot_fig11_layer_type_qsnr, "fig11_layer_type_qsnr"),
    ]
    for fn, name in plot_fns:
        try:
            fn(all_results, output_dir)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # Save raw results
    _save_results_json(all_results, output_dir)

    print(f"\nStudy complete. Results in {output_dir}/")
    return all_results
```

**Step 2: Full end-to-end run with ToyMLP**

```bash
PYTHONPATH=. python examples/experiment_format_study.py
```

Expected: All 4 Parts complete, 6 tables printed, 11 figures saved to `results/`.

**Step 3: Commit**

---

### Task 10: Block size sweep sub-analysis

**Files:**
- Modify: `examples/experiment_format_study.py`

**Goal:** Add optional block size sweep (32, 64, 128) for MX formats to check block_size sensitivity. Produces extra table + figure.

**Step 1: Implement block size sweep**

```python
def run_block_size_sweep(fp32_model, calib_data, eval_loader, fmt_name="int8") -> dict:
    """Sweep block sizes for MX format."""
    results = {}
    for bs in [32, 64, 128]:
        label = f"{fmt_name}-blk{bs}"
        gran = GranularitySpec.per_block(size=bs, axis=-1)
        cfg = make_op_cfg(fmt_name, gran)
        print(f"  Block size {bs}...")
        results[label] = run_experiment(cfg, fp32_model, calib_data, eval_loader)
    return results
```

**Step 2: Integrate into `run_format_study()` as optional extra**

**Step 3: Commit**

---

### Task 11: Polish, final test, update CURRENT.md

**Files:**
- Modify: `docs/status/CURRENT.md`
- Modify: `examples/experiment_format_study.py` (any final fixes)

**Goal:** Run full ToyMLP verification, fix any issues, update status.

**Step 1: Run full study on ToyMLP**

```bash
PYTHONPATH=. python examples/experiment_format_study.py 2>&1 | tee study_output.log
```

Check that:
- All 4 Parts complete without error
- All 6 tables are printed
- All 11 figures exist in `results/figures/`
- All CSV files exist in `results/tables/`
- results.json exists in `results/`

**Step 2: Verify no regression on existing tests**

```bash
PYTHONPATH=. python -m pytest src/tests/ -x -q
```

Expected: ALL 1305 tests pass.

**Step 3: Update CURRENT.md**

Mark format-study as complete. Update task status.

**Step 4: Commit**
