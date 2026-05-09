# Redundancy Findings — 2026-05-08

Comprehensive redundancy audit of `src/` (excluding tests). Covers duplicate code,
dead exports, misplaced items, overlapping module responsibilities, and unused imports.

---

## Category 1: Duplicate Code Patterns

### 1.1 Duplicate `_EMPTY_CFG = OpQuantConfig()` (HIGH)
- **`src/session/_context.py:18`** — canonical definition
- **`src/session/_model.py:274`** — independent duplicate

Two separate singleton instances. `_patches.py` imports from `_context`; `_model.py` uses its own local copy. Functionally identical but are distinct Python objects.

**Fix:** Remove from `_model.py`; import from `_context`.

### 1.2 Ten `_patched_*` functions with identical template (HIGH)
- **`src/session/_patches.py:111–224`**

Every patched function follows this exact skeleton:
```python
def _patched_<op>(...):
    state = _get_state()
    if state is None: return _orig_torch_<op>(...)
    cfg = state.resolve("<op>")
    if cfg == _EMPTY_CFG: return _orig_torch_<op>(...)
    name = get_layer_name()
    return <OpFunction>.apply(...)
```
Ten copies: `_patched_matmul`, `_patched_mm`, `_patched_bmm`, `_patched_F_linear`,
`_patched_add`, `_patched_sub`, `_patched_mul`, `_patched_div`, `_patched_exp`, `_patched_log`.

**Fix:** Replace with a `_make_patched_op(op_name, orig_fn, op_fn)` factory.

### 1.3 Ten `Quantized*` classes with identical `__init__`/`forward` (HIGH)
- **`src/ops/activations.py`**: `QuantizedSigmoid` (69), `QuantizedTanh` (135), `QuantizedReLU` (209),
  `QuantizedReLU6` (284), `QuantizedLeakyReLU` (359), `QuantizedSiLU` (436), `QuantizedGELU` (533)
- **`src/ops/pooling.py`**: `QuantizedAdaptiveAvgPool2d` (112)
- **`src/ops/softmax.py`**: `QuantizedSoftmax` (72)

Identical `__init__` (cfg/inner_scheme validation, 10 lines) and `forward` (storage→compute quantize
entry/exit, 20 lines) in every class.

**Fix:** Extract a `_QuantizedActivationMixin` or shared base `_QuantizedModule` class.

### 1.4 Six Conv/ConvTranspose factory functions — identical except class (HIGH)
- **`src/session/_model.py:66–123`**

`_make_conv{1,2,3}d` and `_make_conv_transpose{1,2,3}d` differ only in which `QuantizedConv*D`
class they instantiate.

**Fix:** A single `_make_conv_factory(cls, orig, cfg, name)` that accepts the target class.

### 1.5 Three BN factory functions — identical except class (MEDIUM)
- **`src/session/_model.py:126–162`**

`_make_bn{1,2,3}d` differ only in `QuantizedBatchNorm*D` class.

**Fix:** Same factory pattern as 1.4.

### 1.6 Nine Activation factory functions — same template (MEDIUM)
- **`src/session/_model.py:223–270`**

`_make_sigmoid`, `_make_tanh`, `_make_relu`, `_make_relu6`, `_make_leaky_relu`, `_make_silu`,
`_make_gelu`, `_make_softmax`, `_make_adaptive_avg_pool2d` — all construct a QuantizedXxx
with `_activation_cfg(cfg)` and `name=name`.

**Fix:** Could generalize, but per-op param differences (inplace, dim, output_size) make
this lower ROI. Keep as-is for now.

### 1.7 Norm entry/exit quantization blocks — 3x duplicate (MEDIUM)
- **`src/ops/norm.py`**: `LayerNormFunction.forward` (~486), `GroupNormFunction.forward` (~625),
  `RMSNormFunction.forward` (~769)

Identical storage→compute quantization sequence for input, weight, bias (~22 lines each).
Same pattern repeats in backward methods.

**Fix:** Extract shared `_quantize_input_w_b()` and `_quantize_output_gradients()` helpers.

### 1.8 SIMD binary classes — identical skeleton (LOW)
- **`src/ops/elemwise.py:77–273`**

`SIMDAdd`, `SIMDSub`, `SIMDMul`, `SIMDDiv` share the same `forward(ctx, in1, in2, inner_scheme,
quantize_backprop)` / `backward` / `symbolic` skeleton. The variable vs. non-variable scalar
handling at the top of `forward` is identical.

**Fix:** Could extract base class but the dispatch logic differs per op. ROI is marginal.

### 1.9 Duplicate `_pot_scale` — inline vs function (MEDIUM)
- **`src/transform/pre_scale.py:7`** — canonical `_pot_scale()` function
- **`src/calibration/lsq_optimizer.py:212–213`** — inline `2 ** torch.round(torch.log2(...))`

The same PoT rounding is done two different ways.

**Fix:** Use `from src.transform.pre_scale import _pot_scale` in lsq_optimizer.py.

### 1.10 Duplicate `save_scales`/`load_scales` (MEDIUM)
- **`src/calibration/pipeline.py:227,241`** — standalone module-level functions
- **`src/calibration/pipeline.py:111,127`** — instance methods on `CalibrationSession`

Instance methods delegate to standalone functions. Two code paths to the same operation.

**Fix:** Keep standalone functions as canonical; deprecate or remove instance methods.

---

## Category 2: Dead Code / Unused Exports

### 2.1 Unused imports (5 files)

| File | Line | Unused import |
|------|------|---------------|
| `analysis/e2e.py` | 19 | `Tuple` in typing import |
| `ops/conv.py` | 16 | `_single`, `_pair`, `_triple` from `torch.nn.modules.utils` |
| `ops/elemwise.py` | 15 | `import torch.nn.functional as F` |
| `viz/figures.py` | 13 | `Tuple` in typing import |
| `viz/figures.py` | 44 | Duplicate `import matplotlib.pyplot as plt` (also line 18) |

### 2.2 `__all__` exports never imported externally (7 items)

**`src/session/__init__.py`:**
- `resolve_config` — internal only
- `install_stack_hooks` — internal only
- `remove_stack_hooks` — internal only
- `STUDY_CONFIG` — legacy dict, deprecated in B3

**`src/report/__init__.py`:**
- `PRESETS` — internal spec resolution
- `_OUTPUT_SPEC` — already `_`-prefixed, should never have been public

**`src/session/__init__.py`:**
- `QuantSession` — public alias `= _QuantSession`, never imported externally

### 2.3 ~20 analysis/ classes/functions never used outside tests

**`src/analysis/correlation.py`:**
- `DistributionProfile` (6), `DistributionTaxonomy` (86), `ErrorByDistribution` (202) —
  entire public API never instantiated outside tests
- Various methods on these classes never called outside tests

**`src/analysis/compare.py`:**
- `compare_formats()` (261), `ComparisonReport` (43) — never used outside tests

**`src/analysis/eval_performance.py`:**
- `evaluate_performance()` (158), `PerformanceReport` (12) — never used outside tests

**`src/analysis/e2e.py`:**
- `compare_sessions()` (166) — never used outside tests

### 2.4 Legacy test-only wrappers in production code

**`src/quantize/elemwise.py`:**
- `_quantize_elemwise` (49), `_quantize_bfloat` (62), `_quantize_fp` (76) —
  all marked "Legacy: kept for internal equivalence tests only"
- Only imported by `tests/test_format_quantize.py` and `tests/test_elemwise_equiv.py`

---

## Category 3: Cross-Package Boundary Violations

### 3.1 `_get_quantized_modules` in wrong package
- Defined in `src/calibration/lsq_optimizer.py:18`
- Consumed by `src/session/_quant.py:349,485` and `lsq_optimizer.py:161`
- Belongs in `src/session/_model.py` (near `quantize_model`)

### 3.2 `_compute_best_transform_per_layer` in wrong module
- Defined in `src/viz/figures.py:51` (visualization module)
- Consumed by `src/viz/tables.py:11,188`, `src/viz/figures.py:492`, `src/session/_per_layer_opt.py:33,93`
- This is computation logic, not visualization
- Also a private `_` function imported across package boundary (viz → session)

### 3.3 `_pot_scale` inline in calibration instead of calling shared function
- See 1.9 above

---

## Category 4: Overlapping Module Responsibilities

### 4.1 `analysis/report.py` vs `report/` package
- **`analysis/report.py`**: `AnalysisReport` + `Report` alias — wraps raw dict from observers
- **`report/` package**: `SessionReport`, `StudyReport` — typed wrappers for `SessionResult`
- Both provide `to_dataframe()`, `summary()`-like methods, serialization
- `analysis/report.py` is the older, lower-level API. `report/` is the new Output-Driven system.
- `report/` is canonical; `analysis/report.py` should be treated as internal, with `Report` alias deprecated

---

## Category 5: Total Items By Priority

| Priority | Count | Items |
|----------|-------|-------|
| **P0 (B1)** | 15 | Unused imports (5), dead __all__ exports (7), _EMPTY_CFG (1), legacy wrappers (1 relocate), _pot_scale inline (1), _get_quantized_modules relocate (1) |
| **P1 (B2)** | 6 | _patched_* factory (1), _QuantizedModuleMixin (1), Conv/BN factory consolidation (2), norm quant helpers (1), _compute_best_transform relocate (1) |
| **P2 (B3)** | 3 | analysis/report.py deprecation (1), STUDY_CONFIG deprecation (1), save_scales dedup (1) |

Total: **24 distinct items** across 3 phases.
