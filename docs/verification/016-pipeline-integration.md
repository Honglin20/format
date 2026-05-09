# 016: Pipeline Refactor — End-to-End Integration Contract

**对应测试函数**: `test_integration_output_equivalence()`, `test_module_boundaries()`, `test_search_space_pure_data()`, `test_cli_unchanged()`, `test_backward_compat()`
**验证层级**: Layer 4 — Pipeline Refactor

## 验证内容

The refactored `examples/experiment_format_study.py` must be split into three concern-separated modules (`src/pipeline/`, `src/viz/`, `src/studies/`) while preserving full backward compatibility. This document defines the integration contract — the behavioral guarantees that the refactored code must satisfy, independent of internal module boundaries.

## 1. Output Identity Contract

The refactored example produces a `results.json` with the identical structure as the original. "Identical" is defined as:

### 1.1 Top-level keys

Both original and refactored produce a dict with the same set of keys:

```
["part_a", "part_b", "part_c", "part_d", "part_d_conv", "block_sweep", "part_d_block", "sensitivity"]
```

Each `part_*` entry is a `Dict[str, dict]` mapping config name to result dict.

### 1.2 Config-level keys

Each result dict per config has the same set of keys as the original:

```python
{
    "accuracy": Dict[str, float],         # {"accuracy": 0.82}
    "fp32_accuracy": Dict[str, float],    # {"accuracy": 0.85}
    "delta": Dict[str, float],            # {"accuracy": -0.03}
    "report": Report,                     # AnalysisContext.report()
    "qsnr_per_layer": Dict[str, float],   # {"0.linear": 42.99}
    "mse_per_layer": Dict[str, float],    # {"0.linear": 0.0012}
}
```

### 1.3 JSON-serialized subset

The serialized `results.json` must contain the same structure (accuracy, qsnr_per_layer, mse_per_layer) with the same numeric values for each config entry. Equivalence is checked via:

```python
import json

with open("original/results.json") as f:
    original = json.load(f)
with open("refactored/results.json") as f:
    refactored = json.load(f)

assert original.keys() == refactored.keys()
for part in original:
    assert original[part].keys() == refactored[part].keys()
    for config in original[part]:
        for key in ("accuracy", "qsnr_per_layer", "mse_per_layer"):
            if key in original[part][config]:
                assert original[part][config][key] == refactored[part][config][key]
```

### 1.4 Table CSV columns

All 6 CSV tables have identical column headers and row semantics:

| Table | File | Columns |
|-------|------|---------|
| 1 | `table1_8bit.csv` | `Config,Accuracy,Avg_QSNR_dB,Avg_MSE` |
| 2 | `table2_4bit.csv` | `Config,Accuracy,Avg_QSNR_dB,Avg_MSE` |
| 3 | `table3_pot.csv` | `Config,Accuracy,Delta,Avg_QSNR_dB,Avg_MSE` |
| 4 | `table4_format_x_transform.csv` | `Format,<sorted transform names>` |
| 5 | `table5_transform_distribution.csv` | `Format,<sorted tx names>,Total` |
| 6 | `table6_sensitivity.csv` | `Rank,Layer,Avg_MSE,Avg_QSNR_dB` |

### 1.5 Figures

All 11 figures render without error. Each figure is saved as both `.png` (300 dpi) and `.pdf` (300 dpi) to the `figures/` subdirectory. The output directory must contain:

```
<output_dir>/figures/fig1_qsnr_8bit.png
<output_dir>/figures/fig1_qsnr_8bit.pdf
<output_dir>/figures/fig2_qsnr_4bit.png
...
<output_dir>/figures/fig11_layer_type_qsnr.png
<output_dir>/figures/fig11_layer_type_qsnr.pdf
```

Pixel-level comparison is not required — only that no figure function raises an exception and the files exist with non-zero size.

## 2. Module Boundaries

### 2.1 `src/pipeline/` — Experiment orchestration

```
src/pipeline/
  __init__.py        # Public API: ExperimentRunner, run_experiment, run_part_*
  runner.py          # ExperimentRunner class
  parts.py           # run_part_a_8bit, run_part_b_4bit, run_part_c_pot_scaling,
                     # run_part_d_transforms
  block_sweep.py     # run_block_size_sweep
  utils.py           # helpers: _extract_metric_per_layer, _save_results_json
  smoothquant.py     # _make_smoothquant_transforms, _fuse_smoothquant_weights,
                     # _build_per_layer_optimal_cfg
```

**Boundary constraint**: `src/pipeline/` does NOT import `matplotlib` or `seaborn`. No visualization code lives in pipeline.

```python
# ALLOWED imports in src/pipeline/
from src.session import QuantSession
from src.scheme.op_config import OpQuantConfig
from src.calibration.strategies import MSEScaleStrategy
from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
from src.transform import SmoothQuantTransform, HadamardTransform
from src.analysis.observers import QSNRObserver, MSEObserver, HistogramObserver, DistributionObserver
import copy, json, os
from collections import defaultdict
from typing import Callable, Dict, List, Optional

# FORBIDDEN imports in src/pipeline/
import matplotlib           # Blocked — no viz in pipeline
import seaborn              # Blocked — no viz in pipeline
```

### 2.2 `src/viz/` — Pure visualization

```
src/viz/
  __init__.py    # re-export public API
  theme.py       # FORMAT_COLORS, TRANSFORM_COLORS, HIST_COLORS, FALLBACK_CYCLE
  save.py        # save_figure(fig, output_dir, name) -> str
  tables.py      # accuracy_table, format_comparison_table, pot_scaling_table,
                 # transform_matrix_table, transform_distribution_table,
                 # layer_sensitivity_table
  figures.py     # qsnr_line_chart, mse_box_plot, pot_delta_bar, histogram_overlay,
                 # transform_heatmap, transform_pie, transform_delta,
                 # error_vs_distribution, layer_type_qsnr
```

**Boundary constraint**: `src/viz/` does NOT import `src.pipeline`, `src.session`, or any `QuantSession` symbol. It receives data dicts and returns charts/tables — it does not orchestrate experiments.

```python
# ALLOWED imports in src/viz/
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math, os, json

# FORBIDDEN imports in src/viz/
from src.pipeline ...    # Blocked — viz does not run experiments
from src.session ...     # Blocked — viz does not interact with QuantSession
```

The `_compute_best_transform_per_layer` and `_get_acc_val` helper functions that are shared between pipeline and viz must live in `src/viz/` (since they are consumed by table/figure functions) or in a neutral `src/viz/_helpers.py`. They must NOT be duplicated.

### 2.3 `src/studies/` — Search space as pure data

```
src/studies/
  __init__.py         # re-export public API
  format_study.py     # FORMAT_STUDY — the experiment matrix as pure data
```

**Boundary constraint**: `FORMAT_STUDY` is importable with zero side effects. No model creation, no torch ops, no calibration data construction at import time.

```python
# src/studies/format_study.py

from typing import Dict, List, Optional

FORMAT_STUDY = {
    "part_a": {
        "title": "8-bit Format Comparison",
        "configs": [
            {"name": "MXINT-8",  "format": "int8",     "granularity": "per_block", "block_size": 32, "axis": -1},
            {"name": "MXFP-8",   "format": "fp8_e4m3", "granularity": "per_block", "block_size": 32, "axis": -1},
            {"name": "INT8-PC",  "format": "int8",     "granularity": "per_channel", "axis": 0},
        ],
    },
    "part_b": {
        "title": "4-bit Format Comparison",
        "configs": [
            {"name": "MXINT-4",  "format": "int4",     "granularity": "per_block", "block_size": 32, "axis": -1},
            {"name": "MXFP-4",   "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32, "axis": -1},
            {"name": "INT4-PC",  "format": "int4",     "granularity": "per_channel", "axis": 0},
            {"name": "NF4-PC",   "format": "nf4",      "granularity": "per_channel", "axis": 0, "weight_only": True},
        ],
    },
    "part_c": {
        "title": "FP32 vs PoT Scaling",
        "configs": [
            {"name": "INT8-PC-FP32", "format": "int8", "granularity": "per_channel", "axis": 0, "lsq_pot": False},
            {"name": "INT8-PC-PoT",  "format": "int8", "granularity": "per_channel", "axis": 0, "lsq_pot": True},
            {"name": "INT4-PC-FP32", "format": "int4", "granularity": "per_channel", "axis": 0, "lsq_pot": False},
            {"name": "INT4-PC-PoT",  "format": "int4", "granularity": "per_channel", "axis": 0, "lsq_pot": True},
        ],
    },
    "part_d": {
        "title": "Transform Study at 4-bit",
        "formats": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",  "block_size": 32, "axis": -1, "weight_only": False},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",  "block_size": 32, "axis": -1, "weight_only": False},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "axis": 0,                        "weight_only": False},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "axis": 0,                        "weight_only": True},
        ],
        "transforms": ["None", "SmoothQuant", "Hadamard", "PerLayerOpt"],
        "block_sweep": {
            "format": "int8",
            "block_sizes": [16, 32, 64, 128],
        },
    },
}
```

**Verification**: The following import raises no error and produces no model creation:

```python
from src.studies.format_study import FORMAT_STUDY
assert isinstance(FORMAT_STUDY, dict)
assert "part_a" in FORMAT_STUDY
# No torch tensor, no model instantiation at import time
```

### 2.4 `examples/` — Only bridge

`examples/experiment_format_study.py` is the only module that imports both `pipeline` and `viz`:

```python
# examples/experiment_format_study.py
from src.pipeline import ExperimentRunner, run_part_a_8bit, ...
from src.viz import accuracy_table, qsnr_line_chart, ...
from src.studies.format_study import FORMAT_STUDY

# CLI entry point, no experiment logic, no visualization logic
# Just orchestration: run experiments → collect results → pass to viz
```

## 3. CLI Interface

The CLI must accept exactly the same arguments as the original. Every existing flag works identically:

| Flag | Type | Default | Behavior |
|------|------|---------|----------|
| `-o`, `--output-dir` | `str` | `None` (timestamped subdir) | Output directory |
| `--seed` | `int` | `42` | Random seed |
| `--calib-samples` | `int` | `256` | Calibration sample count |
| `--eval-samples` | `int` | `512` | Evaluation sample count |
| `--batch-size` | `int` | `16` | Batch size |
| `--skip-part-a` | `store_true` | `False` | Skip Part A |
| `--skip-part-b` | `store_true` | `False` | Skip Part B |
| `--skip-part-c` | `store_true` | `False` | Skip Part C |
| `--skip-part-d` | `store_true` | `False` | Skip Part D |
| `--skip-part-d-conv` | `store_true` | `False` | Skip Part D Conv2d variant |
| `--plot-from` | `str` (metavar `RESULTS_JSON`) | `None` | Skip experiments; regenerate from saved JSON |

### 3.1 Argument parsing

Refactored argument parsing must be functionally equivalent to the original `argparse` block:

```python
# Original (line 1932-1987 in monolithic experiment_format_study.py)
parser = argparse.ArgumentParser(...)
parser.add_argument("-o", "--output-dir", ...)
parser.add_argument("--seed", type=int, default=42, ...)
parser.add_argument("--calib-samples", type=int, default=256, ...)
parser.add_argument("--eval-samples", type=int, default=512, ...)
parser.add_argument("--batch-size", type=int, default=16, ...)
parser.add_argument("--skip-part-a", action="store_true", ...)
parser.add_argument("--skip-part-b", action="store_true", ...)
parser.add_argument("--skip-part-c", action="store_true", ...)
parser.add_argument("--skip-part-d", action="store_true", ...)
parser.add_argument("--skip-part-d-conv", action="store_true", ...)
parser.add_argument("--plot-from", default=None, ...)
```

The refactored CLI implementation may live in `examples/experiment_format_study.py` or be delegated to a `cli.py` — but the externally visible flags, types, defaults, and help text must be identical.

### 3.2 `--plot-from` re-entry

`--plot-from RESULTS_JSON` loads a saved `results.json` and regenerates all tables and figures without re-running experiments. The refactored version must preserve this behavior exactly:

```bash
# Run experiments once
PYTHONPATH=. python examples/experiment_format_study.py -o results/my_study

# Later, regenerate figures with different aesthetics
PYTHONPATH=. python examples/experiment_format_study.py --plot-from results/my_study/results.json -o results/regenerated
```

The `--plot-from` path must:
1. Load the saved JSON (accuracy, qsnr_per_layer, mse_per_layer per config)
2. Call the same table and figure functions as the experiment path
3. Support `-o` to redirect output to a different directory

## 4. Backward Compatibility

### 4.1 Direct invocation

```bash
PYTHONPATH=. python examples/experiment_format_study.py
```

This must complete with the same exit code (0 on success) and produce the same output structure.

### 4.2 Import-based usage

```python
from examples.experiment_format_study import (
    build_model, build_conv_model,
    make_calib_data, make_eval_loader, eval_fn,
    run_format_study, run_experiment,
    run_part_a_8bit, run_part_b_4bit, run_part_c_pot_scaling, run_part_d_transforms,
    run_block_size_sweep, plot_from_results,
    generate_table_1, generate_table_2, generate_table_3,
    generate_table_4, generate_table_5, generate_table_6,
    plot_fig1_qsnr_8bit, plot_fig2_qsnr_4bit, plot_fig3_mse_box_8bit,
    plot_fig4_mse_box_4bit, plot_fig5_pot_delta, plot_fig6_histogram_overlay,
    plot_fig7_transform_heatmap, plot_fig8_transform_pie, plot_fig9_transform_delta,
    plot_fig10_error_vs_distribution, plot_fig11_layer_type_qsnr,
    make_op_cfg, make_op_cfg_weight_only,
    FORMAT_COLORS, TRANSFORM_COLORS, HIST_COLORS, FALLBACK_CYCLE,
    PER_T, PER_C0, PER_Cm1, PER_B32,
)
```

All public symbols that the original exported must remain importable from `examples.experiment_format_study`. The refactored version re-exports them:

```python
# examples/experiment_format_study.py
# Public API re-exports
from src.pipeline import (
    ExperimentRunner, run_experiment, run_format_study,
    run_part_a_8bit, run_part_b_4bit, run_part_c_pot_scaling,
    run_part_d_transforms, run_block_size_sweep, plot_from_results,
    make_op_cfg, make_op_cfg_weight_only,
    _make_smoothquant_transforms, _fuse_smoothquant_weights,
    _build_per_layer_optimal_cfg, _compute_best_transform_per_layer,
)
from src.viz import (
    generate_table_1, generate_table_2, generate_table_3,
    generate_table_4, generate_table_5, generate_table_6,
    plot_fig1_qsnr_8bit, plot_fig2_qsnr_4bit, plot_fig3_mse_box_8bit,
    plot_fig4_mse_box_4bit, plot_fig5_pot_delta, plot_fig6_histogram_overlay,
    plot_fig7_transform_heatmap, plot_fig8_transform_pie, plot_fig9_transform_delta,
    plot_fig10_error_vs_distribution, plot_fig11_layer_type_qsnr,
    FORMAT_COLORS, TRANSFORM_COLORS, HIST_COLORS, FALLBACK_CYCLE,
    _get_acc_val, _compute_best_transform_per_layer,
)
from src.scheme.granularity import GranularitySpec
```

## 5. EvalFn Protocol

The `eval_fn` is the user-provided callback that controls all model interaction during the pipeline. Its protocol is:

### 5.1 Signature

```python
def eval_fn(model: nn.Module, data: Any) -> Dict[str, float]:
    """Evaluate model on data.

    The runner calls eval_fn during three phases:
    - calibrate: model may be wrapped in CalibrationSession;
                  data is a single input tensor (from calib_data list)
    - analyze:   model may be wrapped in AnalysisContext;
                  data is a single input tensor (from analyze_data list)
    - evaluate:  model is fp32 or quantized (session.use_fp32() / session.use_quant());
                  data is a DataLoader yielding (input, label) tuples

    Args:
        model: A callable nn.Module. During calibrate/analyze, it may be
               wrapped in a context manager that modifies forward pass
               behavior (hooks, observers). During evaluate, it is a plain
               quantized or fp32 model.
        data: Varies by phase. The runner passes the appropriate data.

    Returns:
        Dict of metric name to scalar float value, e.g. {"accuracy": 0.92}.
    """
```

### 5.2 Phase-specific behavior

| Phase | `model` | `data` | Expected side effects |
|-------|---------|--------|----------------------|
| calibrate | `CalibrationSession`-wrapped model | Single tensor (one batch of calib_data) | Forward pass triggers hook to collect amax. No return value needed. |
| analyze | `AnalysisContext`-wrapped model | Single tensor (one batch of analyze_data) | Forward pass triggers Observer events at quantization points. Observers collect fp32/quant pairs. |
| evaluate (fp32) | Session in fp32 mode (`session.use_fp32()`) | DataLoader of `(input, label)` tuples | Returns metric dict. Model runs in FP32 (no quantization). |
| evaluate (quant) | Session in quant mode (`session.use_quant()`) | DataLoader of `(input, label)` tuples | Returns metric dict. Model runs with quantization applied. |

### 5.3 Contract guarantees

- `eval_fn` is called by the runner, not by the user directly (after initial setup).
- The runner does NOT wrap exceptions from `eval_fn` — they propagate upward.
- `eval_fn` must be callable: `ExperimentRunner.__init__` checks `callable(eval_fn)` and raises `TypeError` if not.
- `eval_fn` return value must be `dict`: `ExperimentRunner` checks after each evaluate call and raises `TypeError` if not.
- During calibrate and analyze phases, the dict return value is **ignored** — the runner only uses the forward pass side effects.
- During evaluate phase, the dict return value is used as the metric result.

### 5.4 Two eval_fn patterns

**Pattern 1 — Simple (default heuristic study)**:

```python
def eval_fn(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}
```

**Pattern 2 — Per-batch (for calibrate/analyze where data is a single tensor)**:

```python
def eval_fn(model, data):
    model.eval()
    with torch.no_grad():
        model(data)  # forward only, no label needed
    return {}  # return value ignored during calibrate/analyze
```

The same `eval_fn` handles both patterns because during calibrate/analyze, `data` is a single tensor, and during evaluate, `data` is a DataLoader. The runner is responsible for passing the correct data type per phase.

## 6. Data Flow Diagram

```
                     FORMAT_STUDY (pure dict)
                     src/studies/format_study.py
                            |
                            v
    build_model() ──> ExperimentRunner.run() ──> results dict
    eval_fn()         src/pipeline/runner.py       (accuracy, qsnr, mse)
    calib_data/         |           |
    eval_loader         |           |
                       v           v
                 calibrate      analyze        evaluate
                 (hooks)     (observers)     (compare)
                      |           |              |
                      v           v              v
                _output_scale  Report()    fp32 vs quant
                buffer write   object       metrics dict
                      |           |              |
                      +-----+-----+--------------+
                            |
                            v
                      all_results dict
                            |
                            v
                    src/viz/ tables + figures
                       |
                       v
              output_dir/tables/*.csv
              output_dir/figures/*.png
              output_dir/figures/*.pdf
              output_dir/results.json
```

## 7. Key Behavioral Contracts

### 7.1 Deep-copy isolation

`ExperimentRunner.run()` deep-copies `fp32_model` for each config entry. Running N configs produces N independent model copies. After `run()` returns, the original `fp32_model` is unmodified.

### 7.2 Frozen state after calibrate

After `session.calibrate()` exits, the computed `_output_scale` is written to each quantized module's buffer. These buffers persist for the lifetime of the session — subsequent `analyze()` and `compare()` calls use the calibrated scales.

### 7.3 Report completeness

The `Report` object from `session.analyze()` contains data for every Observer that was registered. Each Observer's data is keyed by `(layer_name, role, stage, slice_index)`. Empty report (no observers or no forward passes) is a valid state — `report.to_dataframe()` returns an empty list.

### 7.4 Config descriptor immutability

`FOR源自_STUDY` and all descriptor dicts passed to `resolve_config()` are not modified by pipeline functions. The caller retains ownership of the data.

### 7.5 Seed determinism

Setting `--seed N` guarantees the same random calibration data, evaluation data, and model initialization (if the model uses RNG), producing the same `results.json` across runs on the same platform and PyTorch version.

## 8. Test Plan

| Test | What it verifies | How |
|------|-----------------|-----|
| `test_integration_output_equivalence` | Refactored run produces same results.json structure as original | Run both monolithic and refactored versions with ToyMLP; compare JSON keys and numeric values |
| `test_module_boundaries_pipeline` | src/pipeline/ does not import matplotlib or seaborn | `pip install -e . && python -c "from src.pipeline import *"` — inspect module dependencies |
| `test_module_boundaries_viz` | src/viz/ does not import src.pipeline or src.session | `pip install -e . && python -c "from src.viz import *"` — inspect module dependencies |
| `test_search_space_pure_data` | FORMAT_STUDY is importable without side effects | `python -c "from src.studies.format_study import FORMAT_STUDY"` — no model creation, no torch ops |
| `test_cli_args_identical` | All CLI flags match original exactly | Parse args from both versions; compare `parser._option_string_actions` |
| `test_cli_skip_parts` | `--skip-part-*` flags work identically | Run with various skip combinations; verify correct parts missing |
| `test_cli_plot_from` | `--plot-from` regenerates tables/figures from saved JSON | Save results.json from one run; load with `--plot-from`; verify all tables/figures produced |
| `test_backward_compat_imports` | All public symbols importable from examples.experiment_format_study | `python -c "from examples.experiment_format_study import <every symbol>"` |
| `test_backward_compat_run` | `PYTHONPATH=. python examples/experiment_format_study.py` works | Run with `--skip-part-c --skip-part-d`; exit code 0 |
| `test_eval_fn_protocol` | eval_fn contract enforced | Pass non-callable, non-dict-returning, verify TypeError |
| `test_deepcopy_isolation` | fp32_model unmodified after run | Compare model parameters before and after ExperimentRunner.run() |

## 9. Forward Compatibility — Future Extension Points

The refactored module structure anticipates these future extensions:

| Future feature | Module to extend | Extension pattern |
|---------------|-----------------|-------------------|
| New format (fp6, MXFP6) | `src/studies/format_study.py` | Add config entries to FORMAT_STUDY dict |
| New calibrator strategy | `src/pipeline/runner.py` | Add `calibrator` parameter to ExperimentRunner |
| New analysis observer | `src/pipeline/runner.py` | Pass new Observer to `observers` list |
| New viz chart type | `src/viz/figures.py` | Add new figure function, call from example |
| New part (Part E) | `src/pipeline/parts.py` | Add `run_part_e_*` function, call from example |
| Custom study (not FORMAT_STUDY) | `src/studies/` | Create new study file with custom descriptor dict |

No module boundary change is required for any of these extensions.

## 验证结果

- [ ] 运行日期: YYYY-MM-DD
- [ ] 结果: PASS / FAIL
- [ ] 说明
