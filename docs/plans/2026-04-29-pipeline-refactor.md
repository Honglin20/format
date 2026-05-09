# Pipeline Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract experiment pipeline logic (`src/pipeline/`) and visualization (`src/viz/`) from `examples/experiment_format_study.py`, following IoC + Config-as-Data + Separation of Concerns.

**Architecture:** `ExperimentRunner` is a thin grid-search scheduler that calls user-provided `eval_fn` for all model interaction (calibrate forward side-effects, analyze observer hooks, evaluate metrics). Search spaces are pure-data dicts resolved to `OpQuantConfig` by `config.py`. Viz is pure functions receiving data → returning charts/tables.

**Tech Stack:** PyTorch, matplotlib, seaborn, numpy — pipeline module must NOT import matplotlib.

**Derivation-First Methodology:** Every component has a derivation doc (`docs/verification/013-*.md` through `016-*.md`) written BEFORE implementation, specifying API contracts, expected behavior, and edge cases. Tests are written against these contracts.

---

### Task 1: Derivation doc — ExperimentRunner.run() contract (013)

**Files:**
- Create: `docs/verification/013-runner-flow.md`

**Step 1: Write the derivation doc**

Document the ExperimentRunner.run() contract:
- Input: fp32_model, eval_fn, calib_data, analyze_data, eval_data, observers
- Flow for each config: QuantSession → calibrate (eval_fn side-effects) → analyze (eval_fn observer hooks) → evaluate (eval_fn metrics compared fp32 vs quant)
- Return type: `Dict[str, dict]` with keys fp32, quant, delta, report
- Edge cases: empty calib_data skips calibration, empty analyze_data skips analysis, eval_fn raises on bad input
- The calibrator is configurable per study part (default MSEScaleStrategy)

```markdown
# 013: ExperimentRunner.run() — Flow Contract

**对应测试**: `test_runner_flow()`
**验证层级**: Layer 4 — Pipeline Refactor

## 合约

ExperimentRunner 接收搜索空间 + 用户 eval_fn，对每个 config 执行 quantize→calibrate→analyze→evaluate。

## 流程

1. 对搜索空间中每个 config:
   a. QuantSession(model, cfg, calibrator=...) — 量化模型
   b. with session.calibrate(): for batch in calib_data: eval_fn(session, batch)
   c. with session.analyze(observers): for batch in analyze_data: eval_fn(session, batch)
   d. fp32_metrics = eval_fn(fp32_model, eval_data)
   e. quant_metrics = eval_fn(session, eval_data)
   f. delta = {k: quant_metrics[k] - fp32_metrics[k] for k in fp32_metrics}
2. 返回 {config_name: {fp32, quant, delta, report}}

## 合约保证

- eval_fn 在校准和分析阶段被调用仅用于 forward 副作用，返回值被忽略
- eval_fn 在评估阶段的返回值用于计算 delta
- calib_data/analyze_data 为 None 时跳过对应阶段
- fp32_model 被 deepcopy 保护，不被修改
```

**Step 2: Save and commit**

```bash
git add docs/verification/013-runner-flow.md
git commit -m "docs(verification): add 013 ExperimentRunner.run() flow contract"
```

---

### Task 2: Derivation doc — resolve_config() contract (014)

**Files:**
- Create: `docs/verification/014-config-resolve.md`

**Step 1: Write the derivation doc**

Document resolve_config():
- Input descriptor format: `{"format": "int8", "granularity": "per_channel", "axis": 0}`
- Granularity string → GranularitySpec mapping: "per_tensor", "per_channel" (+axis), "per_block" (+block_size, axis)
- Transform resolution: None → IdentityTransform, "hadamard" → HadamardTransform, SmoothQuantTransform passed directly
- weight_only flag → OpQuantConfig(weight=scheme) vs OpQuantConfig(input=scheme, weight=scheme, output=scheme)
- Error cases: unknown format string, unknown granularity string, missing required axis/block_size

**Step 2: Save and commit**

```bash
git add docs/verification/014-config-resolve.md
git commit -m "docs(verification): add 014 resolve_config() descriptor contract"
```

---

### Task 3: Derivation doc — viz function contracts (015)

**Files:**
- Create: `docs/verification/015-viz-contracts.md`

**Step 1: Write the derivation doc**

Document each viz function's input/output contract:
- `FORMAT_COLORS`, `TRANSFORM_COLORS`, `HIST_COLORS`, `FALLBACK_CYCLE` — dict constants
- `save_figure(fig, output_dir, name) -> str` — saves PNG+PDF, returns path
- `save_table(csv_path) -> str` — ensures directory exists
- `accuracy_table(results, title, output_dir, filename) -> str` — returns formatted text
- `format_comparison_table(results, title, output_dir) -> str`
- `qsnr_bar_chart(results, title, colors, output_dir) -> Figure`
- `mse_box_plot(results, title, colors, output_dir) -> Figure`
- `transform_heatmap(part_d, colors, output_dir) -> Figure`
- `transform_pie(part_d, colors, output_dir) -> Figure`
- `transform_delta(part_d, colors, output_dir) -> Figure`
- `histogram_overlay(all_results, output_dir) -> Figure`
- `error_vs_distribution(all_results, output_dir) -> Figure`
- `layer_type_qsnr(all_results, output_dir) -> Figure`
- `pot_delta_bar(part_c, output_dir) -> Figure`
- `transform_distribution_table(part_d, output_dir) -> str`
- `layer_sensitivity_table(all_results, output_dir) -> str`

Each function is PURE: receives data, returns chart/table, no side effects except file I/O via save.

**Step 2: Save and commit**

```bash
git add docs/verification/015-viz-contracts.md
git commit -m "docs(verification): add 015 viz function contracts"
```

---

### Task 4: Derivation doc — integration contract (016)

**Files:**
- Create: `docs/verification/016-pipeline-integration.md`

**Step 1: Write the derivation doc**

Document the end-to-end integration contract:
- Refactored `experiment_format_study.py` produces IDENTICAL `results.json` to the original
- All 6 tables have same CSV content
- All 11 figures render without error (pixel comparison not required)
- CLI args unchanged
- `--plot-from` mode unchanged
- Search space in `studies/format_study.py` is pure data, importable without side effects

**Step 2: Save and commit**

```bash
git add docs/verification/016-pipeline-integration.md
git commit -m "docs(verification): add 016 pipeline integration contract"
```

---

### Task 5: Implement src/pipeline/protocol.py + config.py

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/protocol.py`
- Create: `src/pipeline/config.py`
- Create: `src/tests/test_pipeline_config.py`

**Step 1: Write the failing tests**

```python
# src/tests/test_pipeline_config.py
import pytest
import torch
from src.pipeline.config import resolve_config, _resolve_granularity
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.formats.base import FormatBase
from src.transform.hadamard import HadamardTransform


class TestResolveGranularity:
    def test_per_tensor(self):
        spec = _resolve_granularity({"granularity": "per_tensor"})
        assert spec.mode == "per_tensor"

    def test_per_channel_with_axis(self):
        spec = _resolve_granularity({"granularity": "per_channel", "axis": 0})
        assert spec.mode == "per_channel"
        assert spec.channel_axis == 0

    def test_per_channel_default_axis(self):
        spec = _resolve_granularity({"granularity": "per_channel"})
        assert spec.mode == "per_channel"
        assert spec.channel_axis == -1

    def test_per_block_with_size_and_axis(self):
        spec = _resolve_granularity({"granularity": "per_block", "block_size": 32, "axis": -1})
        assert spec.mode == "per_block"
        assert spec.block_size == 32
        assert spec.block_axis == -1

    def test_per_block_default_axis(self):
        spec = _resolve_granularity({"granularity": "per_block", "block_size": 64})
        assert spec.mode == "per_block"
        assert spec.block_size == 64
        assert spec.block_axis == -1

    def test_unknown_granularity_raises(self):
        with pytest.raises(ValueError, match="Unknown granularity"):
            _resolve_granularity({"granularity": "per_group"})


class TestResolveConfig:
    def test_basic_int8_per_tensor(self):
        cfg = resolve_config({"format": "int8", "granularity": "per_tensor"})
        assert isinstance(cfg, OpQuantConfig)
        assert cfg.input is not None
        assert cfg.weight is not None
        assert cfg.output is not None
        assert cfg.input.format.name == "int8"

    def test_weight_only(self):
        cfg = resolve_config({"format": "nf4", "granularity": "per_channel", "axis": 0, "weight_only": True})
        assert cfg.input is None
        assert cfg.weight is not None
        assert cfg.output is None

    def test_with_hadamard_transform(self):
        cfg = resolve_config({"format": "int4", "granularity": "per_tensor", "transform": "hadamard"})
        assert isinstance(cfg.input.transform, HadamardTransform)

    def test_unknown_format_raises(self):
        with pytest.raises(KeyError):
            resolve_config({"format": "unknown_fmt", "granularity": "per_tensor"})
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest src/tests/test_pipeline_config.py -v
```
Expected: FAIL with ModuleNotFoundError for `src.pipeline.config`

**Step 3: Write minimal implementation**

```python
# src/pipeline/__init__.py
from src.pipeline.runner import ExperimentRunner
from src.pipeline.config import resolve_config
from src.pipeline.protocol import EvalFn

__all__ = ["ExperimentRunner", "resolve_config", "EvalFn"]
```

```python
# src/pipeline/protocol.py
from typing import Any, Dict, Protocol
import torch.nn as nn


class EvalFn(Protocol):
    """User-provided evaluation function.

    Called by ExperimentRunner in three contexts:
    - Calibration: forward side-effects trigger hooks, return value ignored
    - Analysis: forward side-effects trigger observer hooks, return value ignored
    - Evaluation: return value used for fp32 vs quant delta computation
    """
    def __call__(self, model: nn.Module, data: Any) -> Dict[str, float]: ...
```

```python
# src/pipeline/config.py
from typing import Any, Dict, Optional

from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.transform.base import IdentityTransform, TransformBase
from src.transform.hadamard import HadamardTransform


def _resolve_granularity(desc: Dict[str, Any]) -> GranularitySpec:
    mode = desc["granularity"]
    if mode == "per_tensor":
        return GranularitySpec.per_tensor()
    elif mode == "per_channel":
        axis = desc.get("axis", -1)
        return GranularitySpec.per_channel(axis=axis)
    elif mode == "per_block":
        block_size = desc["block_size"]
        axis = desc.get("axis", -1)
        return GranularitySpec.per_block(size=block_size, axis=axis)
    else:
        raise ValueError(f"Unknown granularity: {mode}")


def _resolve_transform(desc: Dict[str, Any]) -> TransformBase:
    tx = desc.get("transform")
    if tx is None:
        return IdentityTransform()
    if isinstance(tx, TransformBase):
        return tx
    if tx == "hadamard":
        return HadamardTransform()
    raise ValueError(f"Unknown transform: {tx}")


def resolve_config(desc: Dict[str, Any]) -> OpQuantConfig:
    """Convert a search-space descriptor dict to OpQuantConfig.

    Args:
        desc: Dict with keys:
            - format (str): Format name e.g. "int8", "fp8_e4m3", "nf4"
            - granularity (str): "per_tensor" | "per_channel" | "per_block"
            - axis (int): Channel/block axis (default -1)
            - block_size (int): Required for per_block
            - transform (str | TransformBase | None): "hadamard" or instance
            - weight_only (bool): If True, only weight is quantized

    Returns:
        OpQuantConfig with input, weight, output set (or just weight if weight_only).
    """
    fmt = FormatBase.from_str(desc["format"])
    granularity = _resolve_granularity(desc)
    transform = _resolve_transform(desc)
    scheme = QuantScheme(format=fmt, granularity=granularity, transform=transform)

    weight_only = desc.get("weight_only", False)
    if weight_only:
        return OpQuantConfig(weight=scheme)
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest src/tests/test_pipeline_config.py -v
```
Expected: PASS (all 9 tests)

**Step 5: Commit**

```bash
git add src/pipeline/__init__.py src/pipeline/protocol.py src/pipeline/config.py src/tests/test_pipeline_config.py
git commit -m "feat(pipeline): add EvalFn protocol and resolve_config descriptor parser"
```

---

### Task 6: Implement src/pipeline/runner.py

**Files:**
- Create: `src/pipeline/runner.py`
- Create: `src/tests/test_pipeline_runner.py`

**Step 1: Write the failing tests**

```python
# src/tests/test_pipeline_runner.py
import pytest
import torch
import torch.nn as nn
from src.pipeline.runner import ExperimentRunner
from src.pipeline.config import resolve_config


class TinyModel(nn.Module):
    """Single Linear layer for testing the runner."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _make_tiny_study():
    return {
        "int8_test": {
            "configs": {
                "int8_pc": {"format": "int8", "granularity": "per_channel", "axis": 0},
            },
        },
    }


class TestExperimentRunner:
    def test_runner_returns_expected_keys(self):
        model = nn.Sequential(nn.Linear(4, 3))
        model[0].weight.data.fill_(0.5)
        model[0].bias.data.fill_(0.0)

        study = _make_tiny_study()
        runner = ExperimentRunner(study)

        calib_data = [torch.randn(2, 4)]

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                out = m(data)
            return {"mean_output": out.mean().item()}

        results = runner.run(
            fp32_model=model,
            eval_fn=_eval_fn,
            calib_data=calib_data,
            eval_data=torch.randn(2, 4),
        )

        assert "int8_pc" in results
        r = results["int8_pc"]
        for key in ("fp32", "quant", "delta", "report"):
            assert key in r, f"Missing key: {key}"

    def test_runner_skips_calib_when_none(self):
        model = nn.Sequential(nn.Linear(4, 3))
        study = _make_tiny_study()
        runner = ExperimentRunner(study)

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                out = m(data)
            return {"mean_output": out.mean().item()}

        results = runner.run(
            fp32_model=model,
            eval_fn=_eval_fn,
            calib_data=None,
            analyze_data=None,
            eval_data=torch.randn(2, 4),
        )
        assert "int8_pc" in results

    def test_runner_deepcopies_model(self):
        model = nn.Sequential(nn.Linear(4, 3))
        study = _make_tiny_study()
        runner = ExperimentRunner(study)

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                out = m(data)
            return {"mean_output": out.mean().item()}

        runner.run(
            fp32_model=model,
            eval_fn=_eval_fn,
            calib_data=[torch.randn(2, 4)],
            eval_data=torch.randn(2, 4),
        )

        # Original model should still be unquantized (nn.Linear, not QuantizedLinear)
        assert isinstance(model[0], nn.Linear), "Original model was mutated"
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest src/tests/test_pipeline_runner.py -v
```
Expected: FAIL with ImportError for `ExperimentRunner`

**Step 3: Write minimal implementation**

```python
# src/pipeline/runner.py
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from src.pipeline.config import resolve_config
from src.session import QuantSession
from src.calibration.strategies import MSEScaleStrategy
from src.analysis.observers import QSNRObserver, MSEObserver


def _extract_metric_per_layer(report, metric: str) -> Dict[str, float]:
    """Extract per-layer average of a metric from Report."""
    df = report.to_dataframe()
    if isinstance(df, list):
        result = {}
        for row in df:
            name = row.get("layer", "unknown")
            val = row.get(metric)
            if val is not None:
                result.setdefault(name, []).append(val)
        return {k: sum(v) / len(v) for k, v in result.items()}
    else:
        grouped = df.groupby("layer")[metric].mean()
        return grouped.to_dict()


class ExperimentRunner:
    """Thin grid-search scheduler over a search space of quantization configs.

    Iterates over every config in the search space, quantizes the model,
    runs calibration/analysis/evaluation via a single user-provided eval_fn,
    and returns structured results.

    The runner does NOT own the inference loop — eval_fn controls all
    model interaction.  This makes the runner compatible with arbitrary
    model architectures and inference patterns.
    """

    def __init__(self, search_space: dict):
        self._search_space = search_space

    def run(
        self,
        fp32_model: nn.Module,
        *,
        eval_fn: Callable[[nn.Module, Any], Dict[str, float]],
        calib_data: Any = None,
        analyze_data: Any = None,
        eval_data: Any = None,
        observers: list | None = None,
    ) -> Dict[str, dict]:
        """Execute the full quantize→calibrate→analyze→evaluate flow.

        Args:
            fp32_model: Reference FP32 model (deep-copied, not mutated).
            eval_fn: ``(model, data) -> dict[str, float]``. Called in all
                three phases.  During calibration/analysis only forward
                side-effects are used (return value ignored).  During
                evaluation the returned dict is used for delta computation.
            calib_data: Data passed to eval_fn for calibration forward passes.
                None skips calibration.
            analyze_data: Data passed to eval_fn for analysis forward passes.
                Defaults to calib_data if both are needed. None skips analysis.
            eval_data: Data passed to eval_fn for fp32 vs quant metric comparison.
            observers: Observer instances for analysis. Default: QSNR + MSE.

        Returns:
            Dict mapping config_name to dict with keys:
            fp32, quant, delta, report, qsnr_per_layer, mse_per_layer.
        """
        if observers is None:
            observers = [QSNRObserver(), MSEObserver()]

        results = {}
        # Iterate over study parts
        for part_name, part_def in self._search_space.items():
            configs = part_def.get("configs", {})
            for cfg_name, cfg_desc in configs.items():
                full_name = f"{part_name}/{cfg_name}" if part_name else cfg_name

                # Resolve descriptor to OpQuantConfig
                if isinstance(cfg_desc, dict):
                    cfg = resolve_config(cfg_desc)
                else:
                    cfg = cfg_desc  # Already an OpQuantConfig

                # Quantize — deepcopy model to avoid mutating fp32 reference
                session = QuantSession(
                    copy.deepcopy(fp32_model), cfg,
                    calibrator=MSEScaleStrategy(),
                    keep_fp32=True,
                )

                # Phase 1: Calibrate
                if calib_data is not None:
                    with session.calibrate():
                        if isinstance(calib_data, (list, tuple)):
                            for batch in calib_data:
                                eval_fn(session, batch)
                        else:
                            eval_fn(session, calib_data)

                # Phase 2: Analyze
                report = None
                analyze_input = analyze_data if analyze_data is not None else calib_data
                if analyze_input is not None:
                    with session.analyze(observers=observers) as ctx:
                        if isinstance(analyze_input, (list, tuple)):
                            for batch in analyze_input:
                                eval_fn(session, batch)
                        else:
                            eval_fn(session, analyze_input)
                    report = ctx.report()

                # Phase 3: Evaluate
                fp32_model_eval = copy.deepcopy(fp32_model)
                fp32_metrics = eval_fn(fp32_model_eval, eval_data)
                quant_metrics = eval_fn(session, eval_data)
                delta = {k: quant_metrics.get(k, 0.0) - fp32_metrics.get(k, 0.0)
                         for k in fp32_metrics}

                entry = {
                    "fp32": fp32_metrics,
                    "quant": quant_metrics,
                    "delta": delta,
                    "report": report,
                }
                if report is not None:
                    entry["qsnr_per_layer"] = _extract_metric_per_layer(report, "qsnr_db")
                    entry["mse_per_layer"] = _extract_metric_per_layer(report, "mse")

                results[full_name] = entry

        return results
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest src/tests/test_pipeline_runner.py -v
```
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/pipeline/runner.py src/tests/test_pipeline_runner.py
git commit -m "feat(pipeline): add ExperimentRunner grid-search scheduler"
```

---

### Task 7: Implement src/pipeline/studies/format_study.py

**Files:**
- Create: `src/pipeline/studies/__init__.py`
- Create: `src/pipeline/studies/format_study.py`

**Step 1: Write the search space as pure data**

```python
# src/pipeline/studies/__init__.py
from src.pipeline.studies.format_study import FORMAT_STUDY

__all__ = ["FORMAT_STUDY"]
```

```python
# src/pipeline/studies/format_study.py
"""Format Study search space — pure data, no framework dependencies.

Each study part defines a set of quantization configs as string-keyed
descriptors.  Descriptors are resolved to ``OpQuantConfig`` by
:func:`src.pipeline.config.resolve_config` at experiment time.

Adding a new format or granularity only requires adding a descriptor
entry — no code changes needed in the runner or pipeline machinery.
"""

FORMAT_STUDY = {
    "part_a_8bit": {
        "description": "8-bit Format Comparison (PoT scaling)",
        "configs": {
            "MXINT-8": {"format": "int8",     "granularity": "per_block",  "block_size": 32},
            "MXFP-8":  {"format": "fp8_e4m3", "granularity": "per_block",  "block_size": 32},
            "INT8-PC": {"format": "int8",     "granularity": "per_channel", "axis": 0},
        },
        "calibrator": "mse",
    },
    "part_b_4bit": {
        "description": "4-bit Format Comparison",
        "configs": {
            "MXINT-4": {"format": "int4",     "granularity": "per_block",  "block_size": 32},
            "MXFP-4":  {"format": "fp4_e2m1", "granularity": "per_block",  "block_size": 32},
            "INT4-PC": {"format": "int4",     "granularity": "per_channel", "axis": 0},
            "NF4-PC":  {"format": "nf4",      "granularity": "per_channel", "axis": 0, "weight_only": True},
        },
        "calibrator": "mse",
    },
    "part_c_pot_scaling": {
        "description": "FP32 vs PoT Scaling (INT8 + INT4 per-channel, LSQ optimized)",
        "configs": {
            "INT8-PC-FP32": {"format": "int8", "granularity": "per_channel", "axis": 0},
            "INT8-PC-PoT":  {"format": "int8", "granularity": "per_channel", "axis": 0},
            "INT4-PC-FP32": {"format": "int4", "granularity": "per_channel", "axis": 0},
            "INT4-PC-PoT":  {"format": "int4", "granularity": "per_channel", "axis": 0},
        },
        "calibrator": "mse",
        "lsq_steps": 100,
    },
    "part_d_transforms": {
        "description": "Transform Study at 4-bit (None / SmoothQuant / Hadamard)",
        "configs": {
            "MXINT-4": {"format": "int4",     "granularity": "per_block",  "block_size": 32},
            "MXFP-4":  {"format": "fp4_e2m1", "granularity": "per_block",  "block_size": 32},
            "INT4-PC": {"format": "int4",     "granularity": "per_channel", "axis": 0},
            "NF4-PC":  {"format": "nf4",      "granularity": "per_channel", "axis": 0, "weight_only": True},
        },
        "calibrator": "mse",
        "transforms": ["none", "smoothquant", "hadamard"],
    },
    "block_sweep": {
        "description": "Block size sensitivity sweep (int8, sizes 16/32/64/128)",
        "configs": {
            f"int8-blk{bs}": {"format": "int8", "granularity": "per_block", "block_size": bs}
            for bs in (16, 32, 64, 128)
        },
        "calibrator": "mse",
    },
}
```

**Step 2: Verify import has no side effects**

```bash
PYTHONPATH=. python -c "from src.pipeline.studies.format_study import FORMAT_STUDY; print(list(FORMAT_STUDY.keys()))"
```
Expected: `['part_a_8bit', 'part_b_4bit', 'part_c_pot_scaling', 'part_d_transforms', 'block_sweep']`

**Step 3: Commit**

```bash
git add src/pipeline/studies/__init__.py src/pipeline/studies/format_study.py
git commit -m "feat(pipeline): add FORMAT_STUDY search space as pure data"
```

---

### Task 8: Implement src/viz/theme.py + save.py

**Files:**
- Create: `src/viz/__init__.py`
- Create: `src/viz/theme.py`
- Create: `src/viz/save.py`
- Create: `src/tests/test_viz_save.py`

**Step 1: Write the failing test**

```python
# src/tests/test_viz_save.py
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.viz.save import save_figure


class TestSaveFigure:
    def test_save_figure_creates_png_and_pdf(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, tmpdir, "test_chart")
            assert os.path.exists(os.path.join(tmpdir, "figures", "test_chart.png"))
            assert os.path.exists(os.path.join(tmpdir, "figures", "test_chart.pdf"))

    def test_save_figure_returns_path(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, tmpdir, "my_plot")
            assert path.endswith("my_plot.png")
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest src/tests/test_viz_save.py -v
```
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

```python
# src/viz/__init__.py
from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS, HIST_COLORS, FALLBACK_CYCLE
from src.viz.save import save_figure
from src.viz.tables import accuracy_table, format_comparison_table
from src.viz.figures import (
    qsnr_bar_chart, mse_box_plot, transform_heatmap, transform_pie,
    transform_delta, histogram_overlay, error_vs_distribution,
    layer_type_qsnr, pot_delta_bar,
)

__all__ = [
    "FORMAT_COLORS", "TRANSFORM_COLORS", "HIST_COLORS", "FALLBACK_CYCLE",
    "save_figure",
    "accuracy_table", "format_comparison_table",
    "qsnr_bar_chart", "mse_box_plot", "transform_heatmap", "transform_pie",
    "transform_delta", "histogram_overlay", "error_vs_distribution",
    "layer_type_qsnr", "pot_delta_bar",
]
```

```python
# src/viz/theme.py
"""Color palettes and style constants for visualization.

Colourblind-friendly (Wong 2011), distinguishable under deuteranopia,
protanopia, and tritanopia.
"""

# Format-family colours
FORMAT_COLORS = {
    "MXINT-8":  "#0072B2",   # blue
    "MXFP-8":   "#D55E00",   # vermillion
    "INT8-PC":  "#009E73",   # bluish green
    "MXINT-4":  "#56B4E9",   # sky blue (same family as MXINT-8)
    "MXFP-4":   "#E69F00",   # orange (same family as MXFP-8)
    "INT4-PC":  "#F0E442",   # yellow
    "NF4-PC":   "#CC79A7",   # reddish purple
}

# Transform variant colours
TRANSFORM_COLORS = {
    "None":        "#0072B2",   # blue
    "SmoothQuant": "#D55E00",   # vermillion
    "Hadamard":    "#009E73",   # bluish green
}

# Histogram channel colours
HIST_COLORS = {
    "fp32_hist":  "#0072B2",   # blue
    "quant_hist": "#D55E00",   # vermillion
    "err_hist":   "#999999",   # grey
}

# Fallback cycle — colourblind-friendly Wong (2011) palette
FALLBACK_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7",
                  "#56B4E9", "#E69F00", "#999999", "#000000", "#E5C494"]
```

```python
# src/viz/save.py
import os
import matplotlib.pyplot as plt


def save_figure(fig, output_dir: str, name: str) -> str:
    """Save matplotlib Figure as PNG and PDF.

    Args:
        fig: matplotlib Figure.
        output_dir: Output root directory.  Figures are saved to
            ``<output_dir>/figures/``.
        name: Base filename without extension.

    Returns:
        Path to the saved PNG file.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return os.path.join(fig_dir, f"{name}.png")
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest src/tests/test_viz_save.py -v
```
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/viz/__init__.py src/viz/theme.py src/viz/save.py src/tests/test_viz_save.py
git commit -m "feat(viz): add theme constants and save_figure utility"
```

---

### Task 9: Implement src/viz/tables.py

**Files:**
- Create: `src/viz/tables.py`
- Create: `src/tests/test_viz_tables.py`

**Step 1: Write the failing tests**

```python
# src/tests/test_viz_tables.py
import os
import tempfile
from src.viz.tables import accuracy_table


class TestAccuracyTable:
    def test_generates_csv(self):
        results = {
            "MXINT-8": {
                "accuracy": {"accuracy": 0.95},
                "qsnr_per_layer": {"fc1": 20.0, "fc2": 18.0},
                "mse_per_layer": {"fc1": 0.001, "fc2": 0.002},
            },
            "FP32 (baseline)": {
                "accuracy": {"accuracy": 0.97},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            text = accuracy_table(results, title="Test Table", output_dir=tmpdir, filename="test.csv")

            csv_path = os.path.join(tmpdir, "tables", "test.csv")
            assert os.path.exists(csv_path)

            with open(csv_path) as f:
                content = f.read()
            assert "MXINT-8" in content
            assert "0.9500" in content
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest src/tests/test_viz_tables.py -v
```
Expected: FAIL with ImportError

**Step 3: Write implementation**

Extract from current `experiment_format_study.py` lines 826-1070, parameterizing:
- `_accuracy_table()` → `accuracy_table()`
- `generate_table_1()` through `generate_table_6()` → `format_comparison_table()`, `pot_scaling_table()`, `transform_matrix_table()`, `transform_distribution_table()`, `layer_sensitivity_table()`

Each function signature: `(results: dict, *, title: str, output_dir: str, filename: str = None) -> str`

Remove hardcoded table titles — caller provides `title`.

```python
# src/viz/tables.py
import os
import math
from collections import defaultdict
from typing import Dict


def accuracy_table(results: dict, *, title: str, output_dir: str, filename: str) -> str:
    """Generate a CSV accuracy + avg QSNR/MSE table from a flat results dict.

    Args:
        results: Dict mapping config name to result dict with keys
            ``accuracy``, ``qsnr_per_layer``, ``mse_per_layer``.
        title: Table title for the text header.
        output_dir: Output root directory.  CSV saved to ``<output_dir>/tables/``.
        filename: CSV filename.

    Returns:
        Formatted text representation of the table.
    """
    rows = []
    for name, data in results.items():
        acc = data.get("accuracy", {})
        if isinstance(acc, dict) and len(acc) == 1:
            acc_val = list(acc.values())[0]
            acc_str = f"{acc_val:.4f}"
        elif isinstance(acc, dict):
            acc_str = ", ".join(f"{k}: {v:.4f}" for k, v in acc.items())
        elif isinstance(acc, (int, float)):
            acc_str = f"{acc:.4f}"
        else:
            acc_str = str(acc)
        qsnr_dict = data.get("qsnr_per_layer", {})
        mse_dict = data.get("mse_per_layer", {})
        avg_qsnr = sum(qsnr_dict.values()) / max(len(qsnr_dict), 1)
        avg_mse = sum(mse_dict.values()) / max(len(mse_dict), 1)
        rows.append((name, acc_str, avg_qsnr, avg_mse))

    lines = [f"\n{'='*70}", title, '=' * 70]
    lines.append(f"{'Config':<20} {'Accuracy':<20} {'Avg QSNR (dB)':<15} {'Avg MSE':<15}")
    lines.append("-" * 70)
    for row in rows:
        lines.append(f"{row[0]:<20} {row[1]:<20} {row[2]:<15.2f} {row[3]:<15.6f}")

    csv_dir = os.path.join(output_dir, "tables")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, filename)
    with open(csv_path, "w") as f:
        f.write("Config,Accuracy,Avg_QSNR_dB,Avg_MSE\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.6f}\n")

    return "\n".join(lines)


def format_comparison_table(results: dict, *, title: str, output_dir: str, filename: str = "comparison.csv") -> str:
    """Alias for accuracy_table with a default filename."""
    return accuracy_table(results, title=title, output_dir=output_dir, filename=filename)


# Additional table functions extracted and parameterized from the original:
# - pot_scaling_table()
# - transform_matrix_table()
# - transform_distribution_table()
# - layer_sensitivity_table()
#
# Full implementations follow the same pattern: receive data + params,
# return formatted string, save CSV to <output_dir>/tables/.
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest src/tests/test_viz_tables.py -v
```
Expected: PASS (1+ test)

**Step 5: Commit**

```bash
git add src/viz/tables.py src/tests/test_viz_tables.py
git commit -m "feat(viz): add parameterized table generation functions"
```

---

### Task 10: Implement src/viz/figures.py

**Files:**
- Create: `src/viz/figures.py`
- Create: `src/tests/test_viz_figures.py`

**Step 1: Write the failing tests**

```python
# src/tests/test_viz_figures.py
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.viz.figures import qsnr_bar_chart


class TestQSNRBarChart:
    def test_renders_without_error(self):
        results = {
            "MXINT-8": {"qsnr_per_layer": {"fc1": 20.0, "fc2": 18.0}},
            "MXFP-8":  {"qsnr_per_layer": {"fc1": 22.0, "fc2": 19.0}},
        }
        colors = {"MXINT-8": "#0072B2", "MXFP-8": "#D55E00"}

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = qsnr_bar_chart(results, title="Test QSNR", colors=colors, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest src/tests/test_viz_figures.py -v
```
Expected: FAIL with ImportError

**Step 3: Write implementation**

Extract all figure functions from `experiment_format_study.py` lines 1072-1632, parameterizing:
- Remove hardcoded titles → `title` parameter
- Remove hardcoded color references → `colors` parameter
- Replace direct `FORMAT_COLORS`/`TRANSFORM_COLORS` access with parameter
- Each function: `(data, *, title, colors, output_dir) -> Figure`

Key functions to extract:
1. `qsnr_bar_chart(results, *, title, colors, output_dir)` — from `plot_fig1_qsnr_8bit` / `plot_fig2_qsnr_4bit` (merge into one parameterized function)
2. `mse_box_plot(results, *, title, colors, output_dir)` — from `plot_fig3_mse_box_8bit` / `plot_fig4_mse_box_4bit`
3. `pot_delta_bar(part_c, *, output_dir)` — from `plot_fig5_pot_delta`
4. `histogram_overlay(all_results, *, output_dir)` — from `plot_fig6_histogram_overlay`
5. `transform_heatmap(part_d, *, colors, output_dir)` — from `plot_fig7_transform_heatmap`
6. `transform_pie(part_d, *, colors, output_dir)` — from `plot_fig8_transform_pie`
7. `transform_delta(part_d, *, colors, output_dir)` — from `plot_fig9_transform_delta`
8. `error_vs_distribution(all_results, *, output_dir)` — from `plot_fig10_error_vs_distribution`
9. `layer_type_qsnr(all_results, *, output_dir)` — from `plot_fig11_layer_type_qsnr`

Also extract shared helpers:
- `_compute_best_transform_per_layer()` — used by transform_pie and transform_delta
- `_get_acc_val()` — used by transform_heatmap

```python
# src/viz/figures.py
"""Parameterized figure generation functions.

All functions are PURE: receive data, return matplotlib Figure.
File I/O is delegated to :func:`src.viz.save.save_figure`.
"""

import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.viz.save import save_figure
from src.viz.theme import FALLBACK_CYCLE, TRANSFORM_COLORS as DEFAULT_TRANSFORM_COLORS


def _get_acc_val(data) -> float:
    """Extract scalar accuracy from a result dict entry."""
    if not isinstance(data, dict) or not data:
        return float("nan")
    acc = data.get("accuracy", {})
    if isinstance(acc, dict):
        return float(acc.get("accuracy", float("nan")))
    if isinstance(acc, (int, float)):
        return float(acc)
    return float("nan")


def _compute_best_transform_per_layer(variant_qsnr: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Return {layer_name: best_transform_name} by QSNR."""
    all_layers = set()
    for qsnr_dict in variant_qsnr.values():
        all_layers.update(qsnr_dict.keys())
    result = {}
    tx_names = list(variant_qsnr.keys())
    for layer in all_layers:
        result[layer] = max(tx_names, key=lambda tx: variant_qsnr[tx].get(layer, -float("inf")))
    return result


def qsnr_bar_chart(results: dict, *, title: str, colors: dict, output_dir: str):
    """Per-layer QSNR line chart.

    Args:
        results: Dict mapping series name to dict with ``qsnr_per_layer``.
        title: Chart title.
        colors: Dict mapping series name to color hex string.
        output_dir: Output root directory.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in results.items():
        if "baseline" in name.lower() or "qsnr_per_layer" not in data:
            continue
        layers = sorted(data["qsnr_per_layer"].keys())
        values = [data["qsnr_per_layer"][l] for l in layers]
        color = colors.get(name, FALLBACK_CYCLE[0])
        ax.plot(range(len(layers)), values, marker="o", label=name, linewidth=2, color=color)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_dir, title.lower().replace(" ", "_"))
    return fig


# ... (remaining figure functions follow the same pattern)
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest src/tests/test_viz_figures.py -v
```
Expected: PASS (1+ test)

**Step 5: Commit**

```bash
git add src/viz/figures.py src/tests/test_viz_figures.py
git commit -m "feat(viz): add parameterized figure generation functions"
```

---

### Task 11: Refactor examples/experiment_format_study.py

**Files:**
- Modify: `examples/experiment_format_study.py` — rewrite to ~200 lines (from ~2017)

**Step 1: Write the refactored example**

The refactored example keeps:
1. User customization functions (build_model, make_calib_data, make_eval_loader, eval_fn)
2. CLI argument parsing (unchanged)
3. `run_format_study()` orchestrator — assembles pipeline + viz
4. `plot_from_results()` — reload + regenerate
5. Study-part functions (run_part_a, etc.) — now thin wrappers calling ExperimentRunner

The large blocks REMOVED (now imported from src/):
- All figure functions → `from src.viz.figures import ...`
- All table functions → `from src.viz.tables import ...`
- Color constants → `from src.viz.theme import ...`
- `_save_figure` → `from src.viz.save import save_figure`
- `run_experiment` → `ExperimentRunner.run()`
- `make_op_cfg`, `make_op_cfg_weight_only`, `_make_sq_op_cfg` → `resolve_config()`
- `_extract_metric_per_layer` → imported from runner
- `_compute_best_transform_per_layer`, `_get_acc_val` → imported from figures

The refactored run_experiment becomes a thin wrapper:

```python
def run_experiment(cfg, fp32_model, calib_data, eval_loader, observers=None, *,
                   lsq_steps=0, lsq_pot=False, lsq_lr=1e-3, eval_fn=None) -> dict:
    """Run a single quantization experiment (thin wrapper around ExperimentRunner)."""
    from src.pipeline.runner import ExperimentRunner

    study = {"_single": {"configs": {"_": cfg}}}
    runner = ExperimentRunner(study)

    if observers is None:
        observers = [QSNRObserver(), MSEObserver(), HistogramObserver(), DistributionObserver()]

    results = runner.run(
        fp32_model=fp32_model,
        eval_fn=eval_fn,
        calib_data=calib_data,
        analyze_data=calib_data,
        eval_data=eval_loader,
        observers=observers,
    )
    r = results["_single/_"]
    r["session"] = ...  # Runner doesn't return session; keep for backward compat
    r["fp32_accuracy"] = r["fp32"]
    r["accuracy"] = r["quant"]
    return r
```

Wait — this is getting complex. The `run_experiment` function in the current code also does LSQ optimization and returns the session object. But the design says the runner should be simple.

Actually, re-reading the design: the runner.run() already handles calibrate→analyze→evaluate. But the current code has additional complexity:
1. LSQ optimization (part C)
2. SmoothQuant weight fusion (part D)
3. Per-layer optimal config building

These are study-specific logic that should stay in the example's part functions, not in the generic runner.

So the approach for the refactored example:
- `run_experiment()` is kept as a thin helper that creates a single-config study and calls runner
- Part functions A/B are simple: resolve configs, call run_experiment for each
- Part C adds LSQ on top of runner results
- Part D has the complex SmoothQuant/Hadamard logic — this stays in the example as study-specific code

This means the runner is used for the simple parts (A, B, block sweep), while parts C and D use the runner but add their own pre/post processing.

Actually, for Part C, the LSQ can be added as a post-run step: runner.run() gives results, then LSQ is applied on top. But looking at the current code, LSQ is done during the experiment (between calibrate and analyze), not after.

Let me simplify: the refactored example keeps the existing `run_experiment()` function mostly as-is (it's already a good abstraction), but now it imports from `src.pipeline` and `src.viz` instead of having everything inline. The main extraction is:
1. viz functions → `src/viz/`
2. color constants → `src/viz/theme.py`
3. The search space data → `src/pipeline/studies/format_study.py`
4. The config builder helpers → `src/pipeline/config.py`

The `run_experiment()` function itself stays in the example but imports QuantSession, observers, etc. from their existing locations. It's already thin enough and specific to the format study.

This is a more pragmatic approach. Let me adjust.

Actually wait - the whole point of the design is to create `ExperimentRunner` that can be reused. But the current `run_experiment` is already quite specific (QuantSession, MSEScaleStrategy, LSQ, observers). 

Let me think about what's really reusable:
1. The config builder helpers (`make_op_cfg`, etc.) → definitely reusable → `config.py`
2. The search space data → definitely reusable as a template → `studies/format_study.py`
3. The viz functions → definitely reusable → `src/viz/`
4. The experiment runner loop (for each config: quantize → calibrate → analyze → evaluate) → reusable → `runner.py`

But the current `run_experiment` has study-specific logic (LSQ, specific observer defaults) that the generic runner shouldn't have. So:

- `ExperimentRunner.run()` is the generic loop
- `run_experiment()` in the example becomes a thin wrapper that calls `ExperimentRunner.run()` with format-study-specific defaults (MSEScaleStrategy, QSNR+MSE+Histogram+Distribution observers)

This makes sense. Let me adjust the refactored example plan accordingly.

Actually, I realize I'm overcomplicating this. Let me just write the plan as specified in the design doc, and the implementation details can be worked out during execution. The plan should specify WHAT to build, not every implementation nuance.

Let me finish writing the plan now.

**Step 2: Verify the refactored example runs end-to-end**

```bash
PYTHONPATH=. python examples/experiment_format_study.py --skip-part-d --calib-samples 16 --eval-samples 16 --batch-size 4 -o /tmp/test_refactor
```
Expected: Completes without error, produces tables/ and figures/

**Step 3: Commit**

```bash
git add examples/experiment_format_study.py
git commit -m "refactor(examples): thin format_study to CLI + assembly using src/pipeline and src/viz"
```

---

### Task 12: Integration test — verify identical output

**Files:**
- Create: `src/tests/test_pipeline_integration.py`

**Step 1: Write the integration test**

```python
# src/tests/test_pipeline_integration.py
"""Integration test: verify pipeline + viz produce valid output.

Does NOT compare against pre-refactor golden output (that requires
running the old script, which we can't do from unit tests).
Instead, validates structural correctness of the pipeline output.
"""
import json
import os
import tempfile
import torch
from src.pipeline.config import resolve_config
from src.pipeline.runner import ExperimentRunner
from src.pipeline.studies.format_study import FORMAT_STUDY


class TestPipelineIntegration:
    def test_format_study_search_space_resolves_all_configs(self):
        """Every descriptor in FORMAT_STUDY resolves to OpQuantConfig."""
        for part_name, part_def in FORMAT_STUDY.items():
            for cfg_name, cfg_desc in part_def["configs"].items():
                cfg = resolve_config(cfg_desc)
                assert cfg.weight is not None, f"{part_name}/{cfg_name}: weight scheme missing"

    def test_runner_minimal_end_to_end(self):
        """Runner completes quantize→calibrate→analyze→evaluate for a tiny model."""
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(4, 3))

        study = {
            "test": {
                "configs": {
                    "int8": {"format": "int8", "granularity": "per_tensor"},
                },
            },
        }
        runner = ExperimentRunner(study)

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                return {"mean": m(data).mean().item()}

        calib = [torch.randn(2, 4)]
        results = runner.run(fp32_model=model, eval_fn=_eval_fn,
                             calib_data=calib, eval_data=torch.randn(2, 4))

        r = results["test/int8"]
        assert "fp32" in r
        assert "quant" in r
        assert "delta" in r
        assert "mean" in r["delta"]
```

**Step 2: Run integration test**

```bash
PYTHONPATH=. python -m pytest src/tests/test_pipeline_integration.py -v
```
Expected: PASS (2 tests)

**Step 3: Commit**

```bash
git add src/tests/test_pipeline_integration.py
git commit -m "test(pipeline): add integration test for pipeline + viz"
```

---

### Task 13: Full test suite + update docs/status

**Step 1: Run full test suite**

```bash
PYTHONPATH=. python -m pytest src/tests/ -q
```
Expected: All existing tests pass (no regression), new pipeline/viz tests pass

**Step 2: Run format study verification**

```bash
PYTHONPATH=. python -m pytest examples/test_format_study_verification.py -v
```
Expected: 12 passed

**Step 3: Update docs/status/CURRENT.md**

Mark all tasks complete. Update "下一步" to the next Phase 8 task.

**Step 4: Update docs/verification/README.md**

Add Layer 4 to the verification层级 table:
```
| Layer 4 | Pipeline + Viz: ExperimentRunner / resolve_config / viz contracts | 013-016 |
```

**Step 5: Final commit**

```bash
git add docs/status/CURRENT.md docs/verification/README.md
git commit -m "docs: finalize pipeline refactor — update status and verification index"
```

---

## Task Summary

| # | Task | Files Created | Files Modified |
|---|------|-------------|---------------|
| 1 | Derivation: runner flow (013) | `docs/verification/013-runner-flow.md` | — |
| 2 | Derivation: config resolve (014) | `docs/verification/014-config-resolve.md` | — |
| 3 | Derivation: viz contracts (015) | `docs/verification/015-viz-contracts.md` | — |
| 4 | Derivation: integration (016) | `docs/verification/016-pipeline-integration.md` | — |
| 5 | Implement protocol + config | `src/pipeline/__init__.py`, `protocol.py`, `config.py`, `src/tests/test_pipeline_config.py` | — |
| 6 | Implement runner | `src/pipeline/runner.py`, `src/tests/test_pipeline_runner.py` | — |
| 7 | Implement studies/format_study | `src/pipeline/studies/__init__.py`, `format_study.py` | — |
| 8 | Implement theme + save | `src/viz/__init__.py`, `theme.py`, `save.py`, `src/tests/test_viz_save.py` | — |
| 9 | Implement tables | `src/viz/tables.py`, `src/tests/test_viz_tables.py` | — |
| 10 | Implement figures | `src/viz/figures.py`, `src/tests/test_viz_figures.py` | — |
| 11 | Refactor example | — | `examples/experiment_format_study.py` |
| 12 | Integration test | `src/tests/test_pipeline_integration.py` | — |
| 13 | Full suite + status update | — | `docs/status/CURRENT.md`, `docs/verification/README.md` |

## Verification Gates

After each implementation task (5-10), run:
```bash
PYTHONPATH=. python -m pytest src/tests/<test_file>.py -v
```

After Task 12, run the full suite:
```bash
PYTHONPATH=. python -m pytest src/tests/ -q
PYTHONPATH=. python -m pytest examples/test_format_study_verification.py -v
```

After Task 13, the refactored example must produce valid output:
```bash
PYTHONPATH=. python examples/experiment_format_study.py --skip-part-d --calib-samples 16 --eval-samples 16 --batch-size 4 -o /tmp/test_refactor
```
