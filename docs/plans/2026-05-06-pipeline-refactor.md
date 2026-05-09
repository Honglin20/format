# Pipeline Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `format_study.py` (1086 lines) into a three-layer architecture — runner (execute), report (output), format_study (orchestrate) — per the approved design doc.

**Architecture:** `ExperimentRunner` creates Session per config and drives calibrate→analyze→evaluate lifecycle. `StudyReport` adapts `ExperimentResult` to viz/tables functions. `format_study.py` (~100 lines) orchestrates: load config → Runner → Report. SmoothQuant stays in orchestration layer. Per-layer-optimal stays as composition. Output is declaration-driven.

**Tech Stack:** Python 3.10+, PyTorch, matplotlib. Existing deps only. No new packages.

---

### Task 1: Freeze current test baseline

**Files:**
- No changes — verification only

**Step 1: Run current full test suite to establish baseline**

Run: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q`
Expected: 1,416 passed (or similar count)

**Step 2: Note the count**

Record this as the baseline. After refactoring, we must match or exceed it.

---

### Task 2: Add `scale_format` to `resolve_config`

**Files:**
- Modify: `src/pipeline/config.py`
- Test: existing tests should still pass

**Step 1: Add `scale_format` to `resolve_config`**

In `resolve_config()`, after the existing `weight_only` block, add:

```python
scale_format = desc.get("scale_format", "fp32")
if scale_format not in ("fp32", "pot"):
    raise ValueError(f"scale_format must be 'fp32' or 'pot', got {scale_format!r}")
```

And pass it into `GranularitySpec` or `QuantScheme` depending on where it lives. Since `scale_format` affects how the scale is stored (the quantization of the scale itself), and scales are computed per-granularity unit, the cleanest initial approach is to store it on the scheme:

```python
scheme = QuantScheme(
    format=fmt,
    granularity=granularity,
    transform=transform,
    scale_format=scale_format,
)
```

Note: `QuantScheme` might not have `scale_format` yet. If not, check if it needs to be added or if we can store it elsewhere. For now, add the field to `QuantScheme` as optional (default "fp32") — this is backward compatible. If the core scheme doesn't consume it yet, that's fine: the `scale_format` is read by the Runner when it sets up the session.

**Step 2: Run tests to verify no regressions**

Run: `pytest src/tests/test_format_registry.py src/tests/test_format_quantize.py -q`
Expected: all pass

**Step 3: Commit**

```bash
git add src/pipeline/config.py
git commit -m "feat(pipeline): add scale_format to resolve_config

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 3: Rewrite `runner.py` — `ExperimentResult` + simplified `ExperimentRunner`

**Files:**
- Rewrite: `src/pipeline/runner.py`
- Test: existing tests under `src/tests/` — no specific runner tests exist, verified by integration

**Design reference:** `docs/plans/2026-05-06-pipeline-refactor-design.md` lines 89-204

**Step 1: Write `ExperimentResult` dataclass**

```python
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from src.pipeline.config import resolve_config
from src.session import QuantSession
from src.calibration.strategies import MSEScaleStrategy
from src.analysis.observers import QSNRObserver, MSEObserver


@dataclass
class ExperimentResult:
    """Result of a single quantization experiment (one config)."""
    name: str
    fp32_metrics: Optional[Dict[str, float]] = None
    quant_metrics: Optional[Dict[str, float]] = None
    delta: Optional[Dict[str, float]] = None
    qsnr_per_layer: Dict[str, float] = field(default_factory=dict)
    mse_per_layer: Dict[str, float] = field(default_factory=dict)
    cost: Any = None
    cost_fp32: Any = None

    @property
    def avg_qsnr(self) -> float:
        if not self.qsnr_per_layer:
            return float("nan")
        return sum(self.qsnr_per_layer.values()) / len(self.qsnr_per_layer)

    @property
    def avg_mse(self) -> float:
        if not self.mse_per_layer:
            return float("nan")
        return sum(self.mse_per_layer.values()) / len(self.mse_per_layer)
```

**Step 2: Write `extract_metric_per_layer`** (keep existing function, it's already correct)

**Step 3: Write simplified `ExperimentRunner`**

```python
class ExperimentRunner:
    """Execute a search space of quantization configs against a model.

    For each config: resolve → create Session → calibrate → (LSQ) → analyze → evaluate.
    Pure execution — no print, no file I/O.
    """

    def __init__(
        self,
        search_space: dict,
        *,
        skip_parts: Optional[set] = None,
    ):
        self._search_space = search_space
        self._skip = skip_parts or set()

    def run(
        self,
        fp32_model: nn.Module,
        *,
        eval_fn: Callable,
        calib_data: Any,
        eval_data: Any = None,
        observers: list | None = None,
        on_config_done: Optional[Callable[[ExperimentResult], None]] = None,
        model_for_part: Optional[Callable[[str], nn.Module]] = None,
    ) -> Dict[str, List[ExperimentResult]]:
        """Execute all experiments.

        Args:
            fp32_model: Reference FP32 model (deep-copied per config).
            eval_fn: ``(model, data) -> dict[str, float]``.
            calib_data: Data for calibration + analysis forward passes.
            eval_data: Data for evaluation. Defaults to calib_data.
            observers: Override observer list. Default: QSNR + MSE.
            on_config_done: Called after each config with the result (for incremental save).
            model_for_part: Optional callback ``(part_name) -> nn.Module``.
                When a part uses a pre-transformed model (e.g. SmoothQuant),
                this returns the correct model for that part. If None, uses
                ``copy.deepcopy(fp32_model)`` for every config.

        Returns:
            ``{part_name: [ExperimentResult, ...]}``
        """
        if observers is None:
            observers = [QSNRObserver(), MSEObserver()]
        if eval_data is None:
            eval_data = calib_data

        all_results: Dict[str, List[ExperimentResult]] = {}

        for part_name, part_cfg in self._search_space.items():
            if part_name in self._skip:
                continue

            configs = part_cfg.get("configs", [])
            part_results: List[ExperimentResult] = []

            for cfg_desc in configs:
                op_cfg = resolve_config(cfg_desc)

                # Model for this config — pre-transformed or fresh copy
                if model_for_part is not None:
                    model = model_for_part(part_name)
                else:
                    model = copy.deepcopy(fp32_model)

                session = QuantSession(
                    model, op_cfg,
                    calibrator=MSEScaleStrategy(),
                    keep_fp32=True,
                )

                # Phase 1: LSQ (optional)
                lsq_steps = cfg_desc.get("lsq_steps", 0)
                if lsq_steps > 0:
                    from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
                    session.initialize_pre_scales(
                        calib_data,
                        init=cfg_desc.get("lsq_init", "ones"),
                        pot=cfg_desc.get("lsq_pot", False),
                    )
                    opt = LayerwiseScaleOptimizer(
                        num_steps=lsq_steps,
                        num_batches=len(calib_data) if isinstance(calib_data, list) else 1,
                        optimizer="adam",
                        lr=cfg_desc.get("lsq_lr", 1e-3),
                        pot=cfg_desc.get("lsq_pot", False),
                    )
                    session.optimize_scales(opt, calib_data, eval_fn=eval_fn)

                # Phase 2: Calibrate
                with session.calibrate():
                    eval_fn(session, calib_data)

                # Phase 3: Analyze
                with session.analyze(observers=observers) as ctx:
                    eval_fn(session, calib_data)
                report = ctx.report()

                # Phase 4: Evaluate
                fp32_copy = copy.deepcopy(fp32_model)
                fp32_metrics = eval_fn(fp32_copy, eval_data)
                quant_metrics = eval_fn(session, eval_data)
                delta = {
                    k: quant_metrics.get(k, 0.0) - fp32_metrics.get(k, 0.0)
                    for k in fp32_metrics
                }

                result = ExperimentResult(
                    name=cfg_desc["name"],
                    fp32_metrics=fp32_metrics,
                    quant_metrics=quant_metrics,
                    delta=delta,
                    qsnr_per_layer=extract_metric_per_layer(report, "qsnr_db"),
                    mse_per_layer=extract_metric_per_layer(report, "mse"),
                    cost=session.estimate_cost(),
                    cost_fp32=session.estimate_cost(fp32=True),
                )
                part_results.append(result)

                if on_config_done:
                    on_config_done(result)

            all_results[part_name] = part_results

        return all_results
```

**Step 4: Verify the file imports work**

Run: `python -c "from src.pipeline.runner import ExperimentRunner, ExperimentResult, extract_metric_per_layer; print('OK')"`
Expected: "OK" (no import errors)

**Step 5: Commit**

```bash
git add src/pipeline/runner.py
git commit -m "refactor(pipeline): rewrite runner with ExperimentResult dataclass

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 4: Move table generators to `src/viz/tables.py`

**Files:**
- Modify: `src/viz/tables.py` — add `pot_delta_table`, `transform_matrix_table`, `transform_distribution_table`, `sensitivity_table`
- Source: `src/pipeline/format_study.py` — the `generate_table_3/4/5/6` functions

**Step 1: Extract table functions from format_study.py**

The 4 table generators (`generate_table_3` through `generate_table_6`) are pure functions: data in, text + CSV out. Move them to `src/viz/tables.py` with clean names:

- `generate_table_3` → `pot_delta_table(results, output_dir)` — FP32 vs PoT accuracy delta
- `generate_table_4` → `transform_matrix_table(results, output_dir, *, suffix="")` — Format × Transform accuracy matrix
- `generate_table_5` → `transform_distribution_table(results, output_dir)` — Per-layer optimal transform distribution
- `generate_table_6` → `sensitivity_table(all_results, output_dir)` — Top-10 most sensitive layers

Keep the logic exactly the same, only change function names and make them importable from `src.viz.tables`.

**Step 2: Update `src/viz/__init__.py`** to export the new table functions.

**Step 3: Verify imports**

Run: `python -c "from src.viz.tables import pot_delta_table, transform_matrix_table, transform_distribution_table, sensitivity_table; print('OK')"`
Expected: "OK"

**Step 4: Commit**

```bash
git add src/viz/tables.py src/viz/__init__.py
git commit -m "refactor(viz): move table generators from format_study to viz/tables

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 5: Create `src/pipeline/report.py` — `StudyReport`

**Files:**
- Create: `src/pipeline/report.py`
- Test: verify with import check

**Design reference:** `docs/plans/2026-05-06-pipeline-refactor-design.md` lines 208-296

**Step 1: Write the StudyReport class**

```python
"""Study output layer: terminal summary, CSV tables, figures.

Pure output — receives ExperimentResult list, produces formatted text + files.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from src.pipeline.runner import ExperimentResult


# Registry: key → (function, arg_adapter)
_TABLE_REGISTRY = {}
_FIGURE_REGISTRY = {}


def _register_tables():
    from src.viz.tables import (
        accuracy_table,
        pot_delta_table,
        transform_matrix_table,
        transform_distribution_table,
        sensitivity_table,
    )
    _TABLE_REGISTRY.update({
        "accuracy": accuracy_table,
        "pot_delta": pot_delta_table,
        "transform_matrix": transform_matrix_table,
        "transform_distribution": transform_distribution_table,
        "sensitivity": sensitivity_table,
    })


def _register_figures():
    from src.viz.figures import (
        qsnr_line_chart, mse_box_plot, pot_delta_bar,
        histogram_overlay, transform_heatmap, transform_pie,
        transform_delta, error_vs_distribution, layer_type_qsnr,
        block_sweep_line_chart, hierarchical_delta_bar,
    )
    from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS
    _FIGURE_REGISTRY.update({
        "qsnr_line": qsnr_line_chart,
        "mse_box": mse_box_plot,
        "pot_delta_bar": pot_delta_bar,
        "histogram": histogram_overlay,
        "transform_heatmap": transform_heatmap,
        "transform_pie": transform_pie,
        "transform_delta": transform_delta,
        "error_vs_dist": error_vs_distribution,
        "layer_type_qsnr": layer_type_qsnr,
        "block_sweep": block_sweep_line_chart,
        "hierarchical_delta": hierarchical_delta_bar,
    })


def _results_to_viz_dict(results: List[ExperimentResult]) -> dict:
    """Convert ExperimentResult list to dict format expected by viz functions."""
    return {
        r.name: {
            "accuracy": r.quant_metrics,
            "fp32_accuracy": r.fp32_metrics,
            "delta": r.delta,
            "qsnr_per_layer": r.qsnr_per_layer,
            "mse_per_layer": r.mse_per_layer,
        }
        for r in results
    }


class StudyReport:
    """Generate terminal summary, CSV tables, and figures from experiment results."""

    def __init__(self, results: Dict[str, List[ExperimentResult]]):
        self._results = results
        _register_tables()
        _register_figures()

    def print_summary(self) -> None:
        """Print a terminal comparison table for each part."""
        for part_name, part_results in self._results.items():
            if not part_results:
                continue
            print(f"\n=== {part_name} ===")
            print(f"  {'Config':<24} {'Avg QSNR':>10}  {'Avg MSE':>12}")
            print(f"  {'-'*24} {'-'*10}  {'-'*12}")
            for r in part_results:
                print(f"  {r.name:<24} {r.avg_qsnr:>10.2f}  {r.avg_mse:>12.6f}")
            if part_results:
                best = max(part_results, key=lambda r: r.avg_qsnr)
                print(f"\n  Best QSNR: {best.name} ({best.avg_qsnr:.2f} dB)")

    def save(self, output_dir: str, config: dict | None = None) -> None:
        """Save CSV tables and figures to output_dir.

        Args:
            output_dir: Root output directory.
            config: The search_space dict (same as passed to ExperimentRunner).
                    Used to look up per-part ``output`` declarations.
        """
        os.makedirs(output_dir, exist_ok=True)

        for part_name, part_results in self._results.items():
            if not part_results:
                continue

            part_cfg = (config or {}).get(part_name, {})
            output_decl = part_cfg.get("output", {})
            table_keys = output_decl.get("tables", ["accuracy"])
            figure_keys = output_decl.get("figures", ["qsnr_line"])

            viz_dict = _results_to_viz_dict(part_results)

            for tkey in table_keys:
                fn = _TABLE_REGISTRY.get(tkey)
                if fn is not None:
                    try:
                        text = fn(
                            viz_dict,
                            title=f"{part_name}",
                            output_dir=output_dir,
                            filename=f"{part_name}_{tkey}.csv",
                        )
                        print(text)
                    except Exception as e:
                        print(f"  Warning: table '{tkey}' failed for {part_name}: {e}")

            for fkey in figure_keys:
                fn = _FIGURE_REGISTRY.get(fkey)
                if fn is not None:
                    try:
                        fn(viz_dict, output_dir=output_dir)
                        print(f"  {part_name}_{fkey}: OK")
                    except Exception as e:
                        print(f"  Warning: figure '{fkey}' failed for {part_name}: {e}")

        # Save full results JSON
        self._save_json(output_dir)

    def to_serializable(self) -> dict:
        """Return JSON-serializable dict (for programmatic use)."""
        out = {}
        for part_name, part_results in self._results.items():
            out[part_name] = {
                r.name: {
                    "fp32_metrics": r.fp32_metrics,
                    "quant_metrics": r.quant_metrics,
                    "delta": r.delta,
                    "qsnr_per_layer": r.qsnr_per_layer,
                    "mse_per_layer": r.mse_per_layer,
                }
                for r in part_results
            }
        return out

    def _save_json(self, output_dir: str) -> None:
        path = os.path.join(output_dir, "results.json")
        with open(path, "w") as f:
            json.dump(self.to_serializable(), f, indent=2, default=str)
        print(f"  results.json: saved")
```

**Step 2: Verify imports**

Run: `python -c "from src.pipeline.report import StudyReport; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add src/pipeline/report.py
git commit -m "feat(pipeline): add StudyReport — declaration-driven output layer

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 6: Rewrite `format_study.py` — pure orchestration

**Files:**
- Rewrite: `src/pipeline/format_study.py`
- Test: `src/tests/test_format_study_helpers.py` — update imports

**Design reference:** `docs/plans/2026-05-06-pipeline-refactor-design.md` lines 298-330

**Step 1: Write the new `format_study.py`**

The file should be ~100 lines. It should:

1. Import from `runner` (ExperimentRunner), `report` (StudyReport), `config` (resolve_config)
2. Keep `_make_smoothquant_transforms` and `_fuse_smoothquant_weights` as private helpers (they're tested, they're pure utility — move them to a `_transform_utils` section or keep inline)
3. Keep `_build_per_layer_optimal_cfg` as a private helper (tested)
4. `run_format_study()` loads config → creates Runner → runs → StudyReport
5. Keep `plot_from_results()` for offline regeneration

```python
"""
Format Study experiment runner.

Programmatic entry point::

    from src.pipeline.format_study import run_format_study
    results = run_format_study(build_model=..., make_calib_data=..., eval_fn=...)

To customise the search space, edit ``src/pipeline/study_config.py``.
"""
from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform, TransformBase
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform
from src.pipeline.runner import ExperimentRunner, ExperimentResult
from src.pipeline.report import StudyReport
from src.viz.figures import _compute_best_transform_per_layer


# ---------------------------------------------------------------------------
# SmoothQuant helpers (tested, used by orchestration layer)
# ---------------------------------------------------------------------------

def _make_smoothquant_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
    *,
    eval_fn: Optional[Callable] = None,
) -> Dict[str, TransformBase]:
    # ... keep existing implementation exactly as-is ...


def _fuse_smoothquant_weights(
    fp32_model: nn.Module,
    sq_transforms: Dict[str, TransformBase],
    *,
    layer_names: Optional[set] = None,
) -> nn.Module:
    # ... keep existing implementation exactly as-is ...


def _build_per_layer_optimal_cfg(
    variant_results: dict,
    sq_transforms: dict,
    fmt_str: str,
    gran: GranularitySpec,
    weight_only: bool = False,
) -> dict:
    # ... keep existing implementation exactly as-is, adapting cfg_builder to use
    # resolve_config or inline QuantScheme construction ...


# ---------------------------------------------------------------------------
# Config builders (public — used by users writing custom studies)
# ---------------------------------------------------------------------------

def make_op_cfg(
    fmt_name: str,
    granularity: GranularitySpec,
    *,
    transform: Optional[TransformBase] = None,
    scale_format: str = "fp32",
) -> OpQuantConfig:
    """Inference-only config: input / weight / output all share the same scheme."""
    fmt = FormatBase.from_str(fmt_name)
    scheme = QuantScheme(
        format=fmt,
        granularity=granularity,
        transform=transform or IdentityTransform(),
    )
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_format_study(
    build_model: Callable[[], nn.Module],
    make_calib_data: Callable[[], List[torch.Tensor]],
    eval_fn: Callable,
    *,
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
    skip_parts: Optional[set] = None,
    eval_data: Any = None,
) -> Dict[str, List[ExperimentResult]]:
    """Run all format study experiments and produce tables and figures.

    Args:
        build_model: Returns a fresh FP32 model instance.
        make_calib_data: Returns calibration data as a list of tensors.
        eval_fn: ``(model, data) -> dict[str, float]``.
        config: Study config dict. Default: STUDY_CONFIG from study_config.py.
        output_dir: Output directory. Default: ``results/<timestamp>/``.
        skip_parts: Set of part names to skip.
        eval_data: Evaluation data. Default: calib_data.

    Returns:
        ``{part_name: [ExperimentResult, ...]}``
    """
    if config is None:
        from src.pipeline.study_config import STUDY_CONFIG as config
    if output_dir is None:
        output_dir = f"results/format_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if skip_parts is None:
        skip_parts = set()

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    print(f"Format Study — output: {output_dir}")
    print(f"Parts: {[k for k in config if k not in skip_parts]}")

    model = build_model()
    model.eval()
    calib_data = make_calib_data()
    eval_data = eval_data if eval_data is not None else calib_data

    # SmoothQuant pre-processing: detect parts that need it, prepare once
    sq_models: Dict[str, nn.Module] = {}
    sq_transforms_all: Dict[str, Dict[str, TransformBase]] = {}
    for part_name, part_cfg in config.items():
        if part_name in skip_parts:
            continue
        configs = part_cfg.get("configs", [])
        has_sq = any(c.get("transform") == "smoothquant" for c in configs)
        if has_sq:
            print(f"  SmoothQuant prep for {part_name}...", end="", flush=True)
            sq_tx = _make_smoothquant_transforms(model, calib_data, eval_fn=eval_fn)
            sq_transforms_all[part_name] = sq_tx
            sq_models[part_name] = _fuse_smoothquant_weights(model, sq_tx)
            print(f" done ({len(sq_tx)} layers)")

    def _model_for_part(part_name: str) -> nn.Module:
        return sq_models.get(part_name, copy.deepcopy(model))

    # Accumulate results for incremental JSON save
    all_results: Dict[str, List[ExperimentResult]] = {}

    def _save_incremental(result: ExperimentResult):
        """Incremental save callback — appends to results.json after each config."""
        # We append to all_results and re-save the whole json each time.
        # For large studies this is fine (results.json is small).
        serializable = StudyReport(all_results).to_serializable()
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    runner = ExperimentRunner(config, skip_parts=skip_parts)
    all_results = runner.run(
        model,
        eval_fn=eval_fn,
        calib_data=calib_data,
        eval_data=eval_data,
        on_config_done=_save_incremental,
        model_for_part=_model_for_part,
    )

    # Report
    report = StudyReport(all_results)
    report.print_summary()
    report.save(output_dir, config=config)

    elapsed = time.time() - t0
    print(f"\nStudy complete. Results in {output_dir}/")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return all_results


def plot_from_results(results_path: str, output_dir: Optional[str] = None):
    """Reload saved results.json and regenerate figures and tables."""
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    with open(results_path) as f:
        raw = json.load(f)

    # Reconstruct minimal ExperimentResult list per part
    from src.pipeline.runner import ExperimentResult
    results: Dict[str, List[ExperimentResult]] = {}
    for part_name, part_data in raw.items():
        part_results = []
        for cfg_name, cfg_data in part_data.items():
            part_results.append(ExperimentResult(
                name=cfg_name,
                fp32_metrics=cfg_data.get("fp32_metrics"),
                quant_metrics=cfg_data.get("quant_metrics"),
                delta=cfg_data.get("delta"),
                qsnr_per_layer=cfg_data.get("qsnr_per_layer", {}),
                mse_per_layer=cfg_data.get("mse_per_layer", {}),
            ))
        results[part_name] = part_results

    report = StudyReport(results)
    report.print_summary()
    report.save(output_dir)
    print(f"\nRegeneration complete. Output in {output_dir}/")
```

**Step 2: Update `src/tests/test_format_study_helpers.py` imports**

The test file currently imports from `src.pipeline.format_study`. After the refactor, `_make_smoothquant_transforms`, `_fuse_smoothquant_weights`, `_build_per_layer_optimal_cfg`, `make_op_cfg`, `make_op_cfg_weight_only` should still be importable from `src.pipeline.format_study`.

Keep all these names available in the module. `make_op_cfg_weight_only` can be implemented as:

```python
def make_op_cfg_weight_only(
    fmt_name: str,
    granularity: GranularitySpec,
    *,
    transform: Optional[TransformBase] = None,
) -> OpQuantConfig:
    fmt = FormatBase.from_str(fmt_name)
    scheme = QuantScheme(
        format=fmt,
        granularity=granularity,
        transform=transform or IdentityTransform(),
    )
    return OpQuantConfig(weight=scheme)
```

**Step 3: Run existing format study tests to verify**

Run: `pytest src/tests/test_format_study_helpers.py -v`
Expected: all pass

**Step 4: Commit**

```bash
git add src/pipeline/format_study.py
git commit -m "refactor(pipeline): rewrite format_study as pure orchestration layer

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 7: Migrate `study_config.py` to new schema

**Files:**
- Modify: `src/pipeline/study_config.py`

**Step 1: Rewrite STUDY_CONFIG**

Remove `type` fields. Use the new unified schema:

```python
STUDY_CONFIG: dict = {
    "part_a": {
        "description": "8-bit Format Comparison",
        "configs": [
            {"name": "MXINT-8", "format": "int8",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-8",  "format": "fp8_e4m3", "granularity": "per_block",   "block_size": 32},
            {"name": "INT8-PC", "format": "int8",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },
    "part_b": {
        "description": "4-bit Format Comparison",
        "configs": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "axis": -1, "weight_only": True},
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },
    "part_c": {
        "description": "FP32 vs PoT Scaling",
        "configs": [
            {"name": "INT8-PC-FP32", "format": "int8", "granularity": "per_channel", "axis": -1, "lsq_steps": 100, "lsq_pot": False},
            {"name": "INT8-PC-PoT",  "format": "int8", "granularity": "per_channel", "axis": -1, "lsq_steps": 100, "lsq_pot": True},
            {"name": "INT4-PC-FP32", "format": "int4", "granularity": "per_channel", "axis": -1, "lsq_steps": 100, "lsq_pot": False},
            {"name": "INT4-PC-PoT",  "format": "int4", "granularity": "per_channel", "axis": -1, "lsq_steps": 100, "lsq_pot": True},
        ],
        "output": {"tables": ["accuracy", "pot_delta"], "figures": ["qsnr_line", "pot_delta_bar"]},
    },
    "part_d": {
        "description": "4-bit Transform Study (None/Hadamard/SmoothQuant/PerLayerOpt)",
        "configs": [
            {"name": "MXINT-4",  "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXINT-4-Hadamard", "format": "int4", "granularity": "per_block", "block_size": 32, "transform": "hadamard"},
            {"name": "MXINT-4-SQ",       "format": "int4", "granularity": "per_block", "block_size": 32, "transform": "smoothquant"},
            {"name": "MXFP-4",   "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-4-Hadamard",  "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32, "transform": "hadamard"},
            {"name": "MXFP-4-SQ",        "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32, "transform": "smoothquant"},
            {"name": "INT4-PC",  "format": "int4",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
            {"name": "INT4-PC-Hadamard", "format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "fp32", "transform": "hadamard"},
            {"name": "INT4-PC-SQ",       "format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "fp32", "transform": "smoothquant"},
            {"name": "NF4-PC",   "format": "nf4",      "granularity": "per_channel", "axis": -1, "weight_only": True},
            {"name": "NF4-PC-Hadamard",  "format": "nf4", "granularity": "per_channel", "axis": -1, "weight_only": True, "transform": "hadamard"},
            {"name": "NF4-PC-SQ",        "format": "nf4", "granularity": "per_channel", "axis": -1, "weight_only": True, "transform": "smoothquant"},
        ],
        "output": {"tables": ["accuracy", "transform_matrix", "transform_distribution"],
                   "figures": ["qsnr_line", "transform_heatmap", "transform_pie", "transform_delta"]},
    },
    # ... block_sweep and part_hierarchical kept with same structure ...
}
```

**Step 2: Verify config parses without errors**

Run: `python -c "from src.pipeline.study_config import STUDY_CONFIG; print('OK', len(STUDY_CONFIG), 'parts')"`
Expected: "OK N parts"

**Step 3: Commit**

```bash
git add src/pipeline/study_config.py
git commit -m "refactor(pipeline): migrate study_config to unified schema

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 8: Update `src/pipeline/__init__.py` exports

**Files:**
- Modify: `src/pipeline/__init__.py`

**Step 1: Update exports**

```python
from src.pipeline.config import resolve_config
from src.pipeline.format_study import run_format_study, plot_from_results, make_op_cfg
from src.pipeline.runner import ExperimentRunner, ExperimentResult, extract_metric_per_layer
from src.pipeline.report import StudyReport
from src.pipeline.study_config import STUDY_CONFIG

__all__ = [
    "resolve_config",
    "ExperimentRunner",
    "ExperimentResult",
    "extract_metric_per_layer",
    "StudyReport",
    "run_format_study",
    "plot_from_results",
    "make_op_cfg",
    "STUDY_CONFIG",
]
```

**Step 2: Verify**

Run: `python -c "from src.pipeline import ExperimentRunner, ExperimentResult, StudyReport, run_format_study; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add src/pipeline/__init__.py
git commit -m "refactor(pipeline): update __init__ exports for new API

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 9: Clean up old experiment entry point

**Files:**
- Delete: `pipeline/experiment_format_study.py`
- Move `pipeline/_model.py` to `examples/_model.py` (or keep both if tests depend on the old path)

**Step 1: Check what imports `pipeline/experiment_format_study.py`**

Run: `grep -r "pipeline.experiment_format_study\|pipeline/experiment_format_study" src/ --include="*.py"`

**Step 2: Check what imports `pipeline._model`**

Run: `grep -r "pipeline._model\|pipeline/_model" src/ examples/ --include="*.py"`

Expected: tests use `from pipeline._model import ToyMLP`, example uses the same.

**Step 3: Create `examples/_model.py`** by copying `pipeline/_model.py`.

**Step 4: Update `src/tests/test_format_study_helpers.py`** import to use `examples._model` or keep `pipeline._model` for now.

Decision: Keep `pipeline/_model.py` for backward compatibility with tests, but make `examples/_model.py` the canonical location. The old experiment entry point `pipeline/experiment_format_study.py` can be deleted since it's replaced by `examples/format_study_random.py`.

**Step 5: Delete `pipeline/experiment_format_study.py`**

**Step 6: Commit**

```bash
git add examples/_model.py
git rm pipeline/experiment_format_study.py
git commit -m "chore: move model to examples/, delete old experiment entry point

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 10: Create `examples/format_study_random.py`

**Files:**
- Create: `examples/format_study_random.py`

**Design reference:** `docs/plans/2026-05-06-pipeline-refactor-design.md` lines 334-449

**Step 1: Write the example**

Use the exact code from the design doc (lines 338-449). Key points:
- 3-layer MLP (64→128→128→64)
- Random tensor calibration data
- `eval_fn` uses cosine similarity + relative MSE
- 4 parts: `core_4bit_scale` (the key conclusion), `4bit_formats`, `8bit_formats`, `4bit_transform`

**Step 2: Run the example** (dry run with 1 config to verify no crashes)

Run: `python -c "
import torch
import torch.nn as nn
from src.pipeline.runner import ExperimentRunner
from src.pipeline.config import resolve_config
# Quick smoke test: resolve one config, create session
cfg = resolve_config({'format': 'int8', 'granularity': 'per_channel'})
print('Config resolved:', type(cfg).__name__)
"`

Expected: "Config resolved: OpQuantConfig"

**Step 3: Run the full example**

Run: `timeout 300 python examples/format_study_random.py 2>&1 | head -100`

Note: The example will take a few minutes. Run with a timeout to verify it completes at least the first part.

**Step 4: Verify output files exist**

Run: `ls results/random_tensor_study/tables/ results/random_tensor_study/figures/`

Expected: CSV files and PNG files present.

**Step 5: Commit**

```bash
git add examples/format_study_random.py
git commit -m "feat(examples): add format study random tensor validation example

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 11: Update CURRENT.md

**Files:**
- Modify: `docs/status/CURRENT.md`

**Step 1: Update the progress section**

Add a "P8.R2 — Pipeline Refactor" entry. Mark the status.

**Step 2: Commit**

```bash
git add docs/status/CURRENT.md
git commit -m "docs: update CURRENT.md after pipeline refactor

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 12: Final verification — full test suite

**Step 1: Run full test suite**

Run: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q`
Expected: same count or higher than baseline (Task 1)

**Step 2: Verify import chain**

Run: `python -c "
from src.pipeline import (
    resolve_config,
    ExperimentRunner,
    ExperimentResult,
    StudyReport,
    run_format_study,
    plot_from_results,
    STUDY_CONFIG,
)
print('All imports OK')
print(f'Study config has {len(STUDY_CONFIG)} parts')
for k, v in STUDY_CONFIG.items():
    print(f'  {k}: {len(v.get(\"configs\", []))} configs')
"`

**Step 3: If any failures, diagnose and fix before claiming done**

---

## File Change Summary

| File | Change |
|------|--------|
| `src/pipeline/config.py` | Add `scale_format` support |
| `src/pipeline/runner.py` | Rewrite: `ExperimentResult` + `ExperimentRunner` |
| `src/pipeline/report.py` | **NEW**: `StudyReport` |
| `src/pipeline/format_study.py` | Rewrite: ~150 lines orchestration |
| `src/pipeline/study_config.py` | Migrate to unified schema (no `type`) |
| `src/pipeline/__init__.py` | Update exports |
| `src/viz/tables.py` | Add 4 table generators from format_study |
| `src/viz/__init__.py` | Export new table functions |
| `pipeline/experiment_format_study.py` | **DELETE** (replaced by example) |
| `examples/_model.py` | **NEW** (moved from pipeline/) |
| `examples/format_study_random.py` | **NEW**: runnable example |
| `docs/status/CURRENT.md` | Update progress |
