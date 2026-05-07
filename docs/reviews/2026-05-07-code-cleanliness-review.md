# Code Cleanliness Review — 2026-05-07

**Branch**: `feature/refactor-src`
**Scope**: `src/` directory, all packages
**Method**: Static analysis of architecture, dependencies, dead code, naming, and patterns against CLAUDE.md rules.

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| P1 (high) | 3 | Design issues — code duplication, tech debt, conceptual overlap |
| P2 (medium) | 3 | Code quality — debug artifacts, large files, cross-package clarity |
| P3 (low) | 4 | Consistency — naming, organization, minor cleanup |

**Overall**: The codebase is in good shape. The ADR-008 refactor was executed cleanly — no `src._utils` imports remain, no `src.pipeline` imports remain, no `mx` imports in production code, and the dependency hierarchy (Math → Ops → Integration → Tools) is strictly enforced. The items below are polish, not structural problems.

---

## P1 — Design Issues

### P1.1 `resolve_config()` duplicates `QuantConfig.from_descriptor().to_op_config()`

**File**: `src/session/_config.py:402-504`

`resolve_config()` is ~100 lines that does the same thing as `QuantConfig.from_descriptor(desc).to_op_config()` but with a slightly different code path. The two implementations have already diverged subtly — for example, `from_descriptor` threads `w_axis`/`a_axis` through while `resolve_config` uses a shared `axis` parameter.

**Recommendation**: Make `resolve_config` delegate to `QuantConfig.from_descriptor(desc).to_op_config()`, adding any missing legacy key support (e.g., `axis` → `w_axis`/`a_axis`) to `from_descriptor`. The function should become a thin wrapper, not a parallel implementation.

```python
# Proposed:
def resolve_config(desc: Dict[str, Any]) -> OpQuantConfig:
    return QuantConfig.from_descriptor(desc).to_op_config()
```

### P1.2 `session/study_config.py` uses legacy dict format instead of `QuantConfig`

**File**: `src/session/study_config.py`

The `STUDY_CONFIG` dict and all its sub-configs use the old flat dict format (`format`, `granularity`, `axis`, `lsq_steps`, etc.) when the project now has `QuantConfig` as the standard configuration entry point. It's exported from `session/__init__.py` with the comment "legacy study configuration dict".

**Impact**: Users learn two configuration formats. The legacy dict keys (`scale_format`, `pre_scale_init`, `pre_scale_pot`) differ from `QuantConfig` field names (`scale_storage`, `prescale_init`, `prescale_pot`), causing confusion.

**Recommendation**: Either:
- (A) Migrate `STUDY_CONFIG` to use `QuantConfig(...)` constructors directly, or
- (B) Move `study_config.py` to `src/tests/` or a `tools/` directory and document it as a test/dev artifact only.

Option A is preferred — it would look like:
```python
STUDY_CONFIG = {
    "part_a": {
        "description": "8-bit Format Comparison",
        "configs": [
            QuantConfig(name="MXINT-8", w_format="int8", w_granularity="per_block", w_block_size=32),
            QuantConfig(name="MXFP-8",  w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32),
            ...
        ],
    },
}
```

### P1.3 `analysis/report.py` overlaps with `report/` package

**Files**: `src/analysis/report.py` vs `src/report/`

The old `Report` class in `analysis/report.py` provides analysis-level data access (`.summary()`, `.to_dataframe()`, `.print_summary()`, `.to_json()`), while the new `report/` package provides output-driven reporting (`SessionReport`, `StudyReport`). Both consume dict-structured observer data and both format it for display.

**Current state**: `analysis/report.py` is still actively used internally by `analysis/context.py`, `analysis/compare.py`, and `analysis/correlation.py`. Its API is nested-dict traversal (`report.layer(name)`, `report.roles(layer)`, `report.slices(...)`). The `report/` package operates on `SessionResult`/`List[SessionResult]` instead.

**Recommendation**: These are distinct enough to coexist (different consumers, different data models), but the naming conflict (`Report` vs `SessionReport`/`StudyReport`) should be resolved. Rename `analysis/report.py:Report` → `AnalysisReport` to clarify the distinction. Or, if the analysis package is internal, make `Report` private (`_Report`).

---

## P2 — Code Quality

### P2.1 Debug artifacts in `tools/`

**Files**: `tools/_debug_equiv.py`, `tools/_debug_equiv2.py`, `tools/_debug_equiv3.py`, `tools/_debug_equiv4.py`

Four numbered debug scripts with the `_` prefix (indicating temporary/internal) are checked into the working tree. These appear to be ad-hoc debugging sessions from the equivalence verification work (CURRENT.md mentions "全算子端到端等价性验证通过" on 2026-05-07).

**Recommendation**: Delete the numbered debug files. `tools/verify_layer_equiv.py` is the canonical verification script and is properly named. The `_debug_equivN.py` files add noise and confusion.

### P2.2 `_session.py` is 680 lines

**File**: `src/session/_session.py`

The file mixes three concerns:
1. Module-level helper functions (`_extract_qsnr_mse`, `_make_calibrator`, `_run_model`, `_needs_calibration`) — ~50 lines
2. Observer mapping (`_OBSERVER_MAP`, `_SMOOTH_INPUT_ROLES`) — ~10 lines
3. `Session` class — ~380 lines
4. `SessionResult` dataclass and its accessor methods — ~140 lines

**Recommendation**: Split `SessionResult` and its accessor methods into `session/_result.py`. The helpers (`_extract_qsnr_mse`, `_make_calibrator`, `_run_model`) are shared with `_per_layer_opt.py` — this is fine within a package (private names are convention, not enforcement), so no action needed, but consider a `session/_helpers.py` if the shared surface grows.

### P2.3 `session/_config.py` import of `torch`

**File**: `src/session/_config.py:9`

`import torch` is used only for `torch.tensor([1.0])` in `_make_activation_transform` (L75). This ties a pure-data configuration module to PyTorch.

**Recommendation**: Defer the tensor creation. `_make_activation_transform` could accept a pre-constructed tensor or the dummy SmoothQuantTransform could use a lazy scale instead of creating a tensor eagerly in what's otherwise a pure-data translation layer.

---

## P3 — Consistency and Polish

### P3.1 `src/__init__.py` is empty

**File**: `src/__init__.py`

The top-level package has no re-exports. Users must know the internal package structure (`from src.session import Session`) rather than having a curated public API at `src.` level.

**Recommendation**: Consider adding key public symbols to `src/__init__.py`:
```python
from src.session import Session, Study, QuantConfig, SessionResult, per_layer_optimal
from src.report import SessionReport, StudyReport
```
Or deliberately keep it empty and document that users import from sub-packages. Either choice is valid, but the current empty file is ambiguous — it's unclear whether it's intentional or an oversight.

### P3.2 Inconsistent `__all__` definitions

Half the `__init__.py` files define `__all__`, half don't. Packages with `__all__`: `session/`, `report/`, `analysis/`. Packages without: `ops/`, `formats/`, `quantize/`, `transform/`, `calibration/`, `cost/`, `viz/`, `observer/`, `scheme/`, `onnx/`.

**Recommendation**: Either add `__all__` to all public packages or remove it from all. The CLAUDE.md rule of thumb: if a package is public-facing (`session`, `report`), `__all__` is useful. If internal (`ops`, `quantize`), it's optional. Document this distinction.

### P3.3 `AnalysisSession = AnalysisContext` alias

**File**: `src/analysis/__init__.py:10`

```python
AnalysisSession = AnalysisContext  # new name, backward-compatible alias
```

This alias was added during the phase where "Session" naming was being standardized (Phase 8.R3). Now that `Session` exists in `src/session`, this alias creates confusion — there's `src.session.Session` (the real session), `src.session._quant._QuantSession` (the internal quant session), and `src.analysis.AnalysisSession` (which is actually an AnalysisContext). Three different things called "Session".

**Recommendation**: Deprecate `AnalysisSession` and remove it after a transition period. The `AnalysisContext` name is already correct.

### P3.4 `test_vector_equiv.py` references deleted file

**File**: `src/tests/test_vector_equiv.py:2`

```python
"""Equivalence tests for src/quantize/vector.py vs mx/vector_ops.py."""
```

`src/quantize/vector.py` was deleted (shown in git status). The docstring should be updated to reference `src/ops/vec_ops.py` instead.

---

## Architecture Compliance (Clean)

The following CLAUDE.md rules were verified and are all **clean**:

| Rule | Status |
|------|--------|
| `src/` does not `import mx` (except tests) | ✅ Only test files + `capture_golden.py` |
| No `MxSpecs` dependency in `src/` | ✅ Only in comments |
| `pipeline/` package deleted | ✅ Zero imports remain |
| `_utils/` directory deleted | ✅ Directory gone, zero imports remain |
| `from src._utils import X` (warning pattern) | ✅ None found |
| No `utils/`, `common/`, `misc/`, `shared/`, `tools/` public packages | ✅ |
| Dependency hierarchy (Tools → Integration → Ops → Math) | ✅ No upward imports |
| No `import *` wildcard imports | ✅ None found |
| No TODO/FIXME/HACK/XXX in code | ✅ None found |
| `# noqa` limited to justified cases | ✅ Only 2 instances, both legitimate |
| `QuantConfig.to_op_config()` is pure data transformation | ✅ No side effects |
| Session 2.0 chainable API (`.quantize() → .calibrate() → ...`) | ✅ Consistent return self |
| `_QuantSession` prefixed with `_` (private) | ✅ Correct |
| `storage_bits`/`storage_kind` naming (not `bfloat`/`fp` scalar) | ✅ Consistent |

---

## Summary of Recommended Actions

| Priority | Action | Effort |
|----------|--------|--------|
| P1 | Deduplicate `resolve_config()` → delegate to `QuantConfig.from_descriptor()` | Small (~30 min) |
| P1 | Migrate `STUDY_CONFIG` to `QuantConfig(...)` constructors | Medium (~1 hr) |
| P1 | Rename `analysis/report.py:Report` → `AnalysisReport` or `_Report` | Small (~15 min) |
| P2 | Delete `tools/_debug_equiv.py`, `_debug_equiv2/3/4.py` | Trivial |
| P2 | Split `SessionResult` out of `_session.py` → `_result.py` | Small (~20 min) |
| P2 | Defer `torch.tensor` creation in `_make_activation_transform` | Small (~10 min) |
| P3 | Decide on `src/__init__.py` policy (re-export or stay empty) | Trivial |
| P3 | Standardize `__all__` usage across packages | Small |
| P3 | Deprecate `AnalysisSession` alias | Trivial |
| P3 | Fix `test_vector_equiv.py` docstring reference | Trivial |

**Estimated total**: ~2.5 hours for all P1-P3 items.
