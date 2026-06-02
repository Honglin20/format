# ADR-012 Defect Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 defects found in ADR-012 review: PER_BLOCK outlier_format silently dropped, missing BANK dynamic sparse outlier_format test, and missing static sparse calibration pipeline integration.

**Architecture:** Three independent fixes:
1. Thread `outlier_format` through `_quantize_per_block` → `_quantize_outlier_bank`
2. Add test coverage for BANK dynamic sparse + outlier_format
3. Wire `compute_sparse_mask()` into the session calibration pipeline to produce and store static masks and per-group scales

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Fix PER_BLOCK outlier_format silently dropped

**Files:**
- Modify: `src/formats/base.py:167-168` — pass `outlier_format` to `_quantize_per_block`
- Modify: `src/formats/base.py:435-461` — accept and forward `outlier_format` to `_quantize_outlier_bank`
- Modify: `src/formats/_outlier_utils.py:14-95` — accept `outlier_format` param and use it for outlier group quantization

**Step 1: Write failing test**

```python
# In test_static_sparse.py TestOutlierFormatDynamicSparse class

def test_per_block_sparse_outlier_format(self):
    """PER_BLOCK dynamic sparse uses outlier_format for outlier group."""
    int4 = FormatBase.from_str("int4")
    int8 = FormatBase.from_str("int8")

    x = torch.randn(2, 8)
    g = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=4,
                        block_axis=-1, outlier_ratio=0.25)
    scheme_int4_only = QuantScheme(format=int4, granularity=g, scale_storage="pot")
    scheme_int8_outlier = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                                      outlier_format=int8)

    r_normal = quantize(x, scheme_int4_only)
    r_outlier_fmt = quantize(x, scheme_int8_outlier)

    assert r_outlier_fmt.shape == x.shape
    assert torch.isfinite(r_outlier_fmt).all()
    assert not torch.equal(r_normal, r_outlier_fmt), \
        "outlier_format=int8 should produce different result from int4-only"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_static_sparse.py::TestOutlierFormatDynamicSparse::test_per_block_sparse_outlier_format -v`
Expected: FAIL — results are equal (outlier_format ignored)

**Step 3: Thread outlier_format through PER_BLOCK path**

In `base.py:167-168`, add `outlier_format=outlier_format`:
```python
return self._quantize_per_block(x, granularity, round_mode,
                                  scale=scale, scale_storage=scale_storage,
                                  outlier_format=outlier_format)
```

In `base.py:435`, update `_quantize_per_block` signature to accept `outlier_format=None`.

In `base.py:460-461`, pass `outlier_format` to `_quantize_outlier_bank`:
```python
return _quantize_outlier_bank(
    self, x, granularity, round_mode, scale_storage=scale_storage,
    outlier_format=outlier_format)
```

In `_outlier_utils.py:14`, update `_quantize_outlier_bank` signature to accept `outlier_format=None`.

In `_outlier_utils.py:82-83`, use `outlier_format` for outlier group elemwise:
```python
q_fmt = outlier_format if outlier_format is not None else format_self
A_o = q_fmt.quantize_elemwise(...)
```

**Step 4: Run test to verify it passes**

Run: `pytest src/tests/test_static_sparse.py::TestOutlierFormatDynamicSparse::test_per_block_sparse_outlier_format -v`
Expected: PASS

**Step 5: Commit**

---

### Task 2: Add BANK dynamic sparse + outlier_format test

**Files:**
- Modify: `src/tests/test_static_sparse.py` — add `test_bank_sparse_outlier_format` to `TestOutlierFormatDynamicSparse`

**Step 1: Write the test**

```python
# In TestOutlierFormatDynamicSparse class

def test_bank_sparse_outlier_format(self):
    """BANK dynamic sparse uses outlier_format for outlier group."""
    int4 = FormatBase.from_str("int4")
    int8 = FormatBase.from_str("int8")

    x = torch.randn(2, 16)
    g = GranularitySpec(mode=GranularityMode.BANK, bank_size=8, bank_axis=-1,
                        outlier_ratio=0.25)
    scheme_int4_only = QuantScheme(format=int4, granularity=g, scale_storage="pot")
    scheme_int8_outlier = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                                      outlier_format=int8)

    r_normal = quantize(x, scheme_int4_only)
    r_outlier_fmt = quantize(x, scheme_int8_outlier)

    assert r_outlier_fmt.shape == x.shape
    assert torch.isfinite(r_outlier_fmt).all()
    assert not torch.equal(r_normal, r_outlier_fmt), \
        "outlier_format=int8 should produce different result from int4-only"
```

**Step 2: Run test to verify it passes (code already correct)**

Run: `pytest src/tests/test_static_sparse.py::TestOutlierFormatDynamicSparse::test_bank_sparse_outlier_format -v`
Expected: PASS (BANK path already correctly passes outlier_format)

**Step 3: Commit**

---

### Task 3: Integrate static sparse into session calibration pipeline

**Files:**
- Modify: `src/calibration/pipeline.py` — add sparse mask computation and per-group scale storage
- Modify: `src/session/_session.py` — thread static sparse scales during quantized forward
- Test: `src/tests/test_static_sparse.py` — add session integration test

**Step 1: Add per-sample activation collection in calibration pipeline**

In `CalibrationPipeline.calibrate()`, when `outlier_ratio > 0` on any module's scheme:
- Collect per-sample activations for weight tensors (weights are static, no collection needed)
- For activations, store per-sample outputs in a list during the forward pass
- After calibration pass, call `compute_sparse_mask()` for each module with sparse config

**Step 2: Compute and store static sparse mask + per-group scales**

Add a method `_compute_sparse_state(module, x_calib, granularity, fmt, outlier_format)` that:
1. Calls `compute_sparse_mask(x_calib, fmt, granularity, outlier_ratio)`
2. Computes per-group amax from calibration data using the mask
3. Stores `_output_mask`, `_output_scale_n`, `_output_scale_o` on module (weights)
4. Stores `_input_mask`, `_input_scale_n`, `_input_scale_o` on module (activations, if `static_input_scale`)

**Step 3: Thread static sparse state in quantized forward pass**

In the session's quantized forward, when module has `_output_mask` buffer:
- Read mask and per-group scales
- Pass them to `quantize()` via `mask=`, `scale=`, `scale_o=` kwargs

**Step 4: Write session integration test**

```python
def test_session_static_sparse_per_tensor(self):
    """Session with static sparse per_tensor produces correct output."""
    # Build a small model, configure with outlier_ratio > 0
    # Calibrate with multiple samples
    # Run quantized forward with static sparse
    # Verify output differs from non-sparse, matches manual static sparse
```

**Step 5: Run all tests**

Run: `pytest src/tests/test_static_sparse.py src/tests/test_sparse_mask.py src/tests/test_sparse_generalization.py src/tests/test_bank_granularity.py -v`
Expected: All PASS

**Step 6: Run E2E regression**

Run: `PYTHONPATH=. python scripts/mnist_hadamard_study.py`
Run: `PYTHONPATH=. python scripts/transformer_agnews_study.py`
Expected: Results within tolerance

**Step 7: Commit**
