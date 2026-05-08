# Redundancy Cleanup — Implementation Plan

**Date**: 2026-05-08
**Branch**: `feature/refactor-src`
**Findings**: `docs/reviews/redundancy-findings-2026-05-08.md`
**Baseline**: 2,034 tests passed (excluding golden)

---

## Execution Model

Each phase follows this cycle:
1. Dispatch parallel agents to make changes
2. Run `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` — must stay at 2,034+
3. Dispatch review agent to inspect changes
4. Only proceed to next phase when phase passes

---

## Phase B1 — Cleanup (14 files, lowest risk)

Removes dead code and fixes placement errors. No behavioral changes.

### Task B1.1: Remove unused imports (5 files)
- `analysis/e2e.py:19` — remove `Tuple` from typing import
- `ops/conv.py:16` — remove `_single, _pair, _triple` import
- `ops/elemwise.py:15` — remove `import torch.nn.functional as F`
- `viz/figures.py:13` — remove `Tuple` from typing import
- `viz/figures.py:44` — remove duplicate `import matplotlib.pyplot as plt`

### Task B1.2: Remove unused __all__ exports (2 files)
- `session/__init__.py` — remove from `__all__`: `resolve_config`, `install_stack_hooks`, `remove_stack_hooks`, `STUDY_CONFIG`, `QuantSession`
- `report/__init__.py` — remove from `__all__`: `PRESETS`, `_OUTPUT_SPEC`

### Task B1.3: Deduplicate `_EMPTY_CFG` (2 files)
- Remove from `session/_model.py:274`
- Add import: `from src.session._context import _EMPTY_CFG` at top of `_model.py`
- Verify no circular import (both are in `session/` package, `_context.py` has no dependency on `_model.py`)

### Task B1.4: Fix `_pot_scale` inline duplicate (1 file)
- `calibration/lsq_optimizer.py:212-213` — replace `2 ** torch.round(torch.log2(pre_scale.data))` with `_pot_scale(pre_scale.data)`
- Add import: `from src.transform.pre_scale import _pot_scale`

### Task B1.5: Relocate `_get_quantized_modules` (3 files)
- Move function from `calibration/lsq_optimizer.py:18` to `session/_model.py`
- Update import in `session/_quant.py` (lines 349, 485)
- Update import in `calibration/lsq_optimizer.py` (line 161 — but it defines it; after move, imports from session)

### Task B1.6: Relocate legacy wrappers to tests (3 files)
- Move `_quantize_elemwise`, `_quantize_bfloat`, `_quantize_fp` from `quantize/elemwise.py` to `tests/_compat.py`
- Update imports in `tests/test_format_quantize.py:15`
- Update imports in `tests/test_elemwise_equiv.py:12-14`

### Task B1.7: Relocate `_compute_best_transform_per_layer` (5 files)
- Move function from `viz/figures.py:51` to new file `viz/_helpers.py`
- Update `viz/figures.py` — import from `_helpers`
- Update `viz/tables.py:11` — import from `_helpers` instead of `.figures`
- Update `session/_per_layer_opt.py:33` — import from `src.viz._helpers`
- Update `viz/__init__.py:11,39` — re-export from `_helpers`

### Phase B1 test: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → must stay 2,034+

---

## Phase B2 — Merge (10 files, medium risk)

Extracts shared abstractions. All patterns are validated by existing tests.

### Task B2.1: Create `_patched_op` factory (1 file)
In `session/_patches.py`:
```python
def _make_patched_op(op_name, orig_fn, op_fn):
    def patched(*args, **kwargs):
        state = _get_state()
        if state is None:
            return orig_fn(*args, **kwargs)
        cfg = state.resolve(op_name)
        if cfg == _EMPTY_CFG:
            return orig_fn(*args, **kwargs)
        name = get_layer_name()
        return op_fn.apply(*args, _name=name, _cfg=cfg)
    return patched
```
Replace all 10 `_patched_*` functions with factory calls.

### Task B2.2: Create `_QuantizedModuleMixin` base (8+ files)
In `ops/` — extract shared `__init__` and `forward` into a mixin class:
```python
class _QuantizedModuleMixin:
    """Shared cfg/inner_scheme validation and storage/compute quantize wrapper."""
    def _init_quant_cfg(self, cfg, inner_scheme, quantize_backprop):
        # Replaces the 10-line __init__ block repeated in every QuantizedXxx.__init__
        ...
    def _forward_quantized(self, input, FunctionClass, **apply_kwargs):
        # Replaces the 20-line forward block repeated in every QuantizedXxx.forward
        ...
```
Apply to: `QuantizedSigmoid`, `QuantizedTanh`, `QuantizedReLU`, `QuantizedReLU6`,
`QuantizedLeakyReLU`, `QuantizedSiLU`, `QuantizedGELU`, `QuantizedAdaptiveAvgPool2d`,
`QuantizedSoftmax`

### Task B2.3: Consolidate Conv/BN factory functions (1 file)
In `session/_model.py`:
```python
def _make_conv(cls, orig, cfg, name, **extra_kwargs):
    """Factory for QuantizedConv{1,2,3}d / QuantizedConvTranspose{1,2,3}d."""
    return cls(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride,
        padding=orig.padding, dilation=orig.dilation,
        groups=orig.groups, bias=orig.bias is not None,
        cfg=cfg, name=name, **extra_kwargs,
    )
```
Replace 6 `_make_conv*` functions + 3 `_make_bn*` functions with this factory.

### Task B2.4: Extract norm quant helpers (1 file)
In `ops/norm.py`, extract:
```python
def _quantize_norm_inputs(input, weight, bias, cfg):
    """Apply storage→compute quantization to norm inputs."""
    ...

def _quantize_norm_gradients(g_output, g_input, g_weight, g_bias, cfg):
    """Apply storage→compute quantization to norm backward grads."""
    ...
```
Used in `LayerNormFunction`, `GroupNormFunction`, `RMSNormFunction`.

### Phase B2 test: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → must stay 2,034+

---

## Phase B3 — Deprecation (8 files, highest risk)

Changes public API surface. Requires deprecation path, not removal.

### Task B3.1: Deprecate `Report` alias (2 files)
- In `analysis/report.py` — add `DeprecationWarning` to `Report` class; recommend `SessionReport`/`StudyReport`
- In `analysis/__init__.py` — keep `Report` export but mark with deprecation comment
- `AnalysisReport` remains as internal type (still used by compare.py, correlation.py, context.py)

### Task B3.2: Deprecate `STUDY_CONFIG` (1 file)
- In `session/study_config.py` — add deprecation comment pointing to `QuantConfig`
- Remove `STUDY_CONFIG` from `session/__init__.py` `__all__` (already done in B1.2)
- Keep import available for backward compat

### Task B3.3: Mark test-only analysis classes as private (4 files)
- `analysis/correlation.py` — prefix `DistributionProfile`, `DistributionTaxonomy`, `ErrorByDistribution`, `LayerSensitivity` with `_`
- `analysis/compare.py` — prefix `compare_formats`, `ComparisonReport` with `_`
- `analysis/eval_performance.py` — prefix `evaluate_performance`, `PerformanceReport` with `_`
- `analysis/e2e.py` — prefix `compare_sessions` with `_`
- Update all test imports to match new names

### Task B3.4: Deduplicate `save_scales`/`load_scales` (1 file)
- In `calibration/pipeline.py` — remove instance methods `CalibrationSession.save_scales()` and `CalibrationSession.load_scales()`
- Keep standalone functions as canonical API
- Update any callers to use standalone functions
- Check tests for instance method usage

### Phase B3 test: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → must stay 2,034+

---

## Verification Gates

After EACH phase:
1. `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` — 2,034 passed
2. Review agent inspects all changed files for correctness
3. No new warnings or errors

Final gate:
1. Full test suite
2. `python -c "from src import Session, Study, QuantConfig, SessionResult"` — public API intact
3. `python -c "import src; print(src.__all__)"` — clean, no deprecated names

---

## Files Changed (Total: ~25)

| Phase | Files |
|-------|-------|
| B1 | analysis/e2e.py, ops/conv.py, ops/elemwise.py, viz/figures.py, session/__init__.py, report/__init__.py, session/_model.py, calibration/lsq_optimizer.py, session/_quant.py, quantize/elemwise.py, tests/_compat.py, tests/test_format_quantize.py, tests/test_elemwise_equiv.py, viz/_helpers.py (new), viz/tables.py, session/_per_layer_opt.py, viz/__init__.py |
| B2 | session/_patches.py, ops/__init__.py, ops/activations.py, ops/pooling.py, ops/softmax.py, ops/norm.py, session/_model.py |
| B3 | analysis/report.py, analysis/__init__.py, session/study_config.py, analysis/correlation.py, analysis/compare.py, analysis/eval_performance.py, analysis/e2e.py, calibration/pipeline.py |
