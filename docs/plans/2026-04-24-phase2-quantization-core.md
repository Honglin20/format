# Phase 2: Quantization Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate specs, elemwise_ops, mx_ops, quantize, vector_ops into src/ with per-function test coverage verifying bit-identical output against old code.

**Architecture:** Faithful migration — rewrite each function, use `src/formats/` instead of `mx/formats.py`, keep 39-key MxSpecs dict structure. Every internal function gets its own test comparing new vs old output.

**Tech Stack:** Python 3.10, PyTorch, pytest

---

### Task 1: `src/specs/` — MxSpecs Configuration System

**Files:**
- Create: `src/specs/specs.py`
- Create: `src/specs/__init__.py`
- Test: `src/tests/test_specs_equiv.py`

**Step 1: Write the failing test — `MxSpecs` defaults**

```python
# src/tests/test_specs_equiv.py
import pytest
from mx.specs import MxSpecs as OldMxSpecs, get_default_mx_specs as old_get_default
from src.specs.specs import MxSpecs, get_default_mx_specs

ALL_SPEC_KEYS = [
    "scale_bits", "w_elem_format", "a_elem_format", "w_elem_format_bp",
    "a_elem_format_bp", "a_elem_format_bp_ex", "a_elem_format_bp_os",
    "mx_flush_fp32_subnorms", "shared_exp_method", "block_size",
    "bfloat", "fp", "bfloat_subnorms", "quantize_backprop",
    "round", "round_m", "round_weight", "round_output",
    "round_grad_weight", "round_grad_input", "round_mx_output",
    "round_mx_input_grad_input", "round_mx_weight_grad_input",
    "round_mx_grad_output_grad_input", "round_mx_input_grad_weight",
    "round_mx_grad_output_grad_weight",
    "softmax_exp2", "vec_use_exp2", "vec_use_recip", "custom_cuda",
]

def test_mx_specs_defaults():
    old = old_get_default()
    new = get_default_mx_specs()
    for key in ALL_SPEC_KEYS:
        assert key in new, f"Key {key!r} missing from new MxSpecs"
        assert new[key] == old[key], f"Default mismatch for {key!r}: {new[key]} != {old[key]}"
    assert len(new) == len(old), f"Key count mismatch: {len(new)} != {len(old)}"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_specs_equiv.py::test_mx_specs_defaults -v`
Expected: FAIL — `src/specs/specs.py` does not exist

**Step 3: Write minimal `MxSpecs` + `get_default_mx_specs`**

Create `src/specs/__init__.py` (empty) and `src/specs/specs.py` with `MxSpecs(UserDict)` containing all 39 default keys and help strings (copy from `mx/specs.py`), plus `get_default_mx_specs()`. No other functions yet.

**Step 4: Run test to verify it passes**

Run: `pytest src/tests/test_specs_equiv.py::test_mx_specs_defaults -v`
Expected: PASS

**Step 5: Write the failing test — `apply_mx_specs`**

```python
def test_apply_mx_specs():
    from mx.specs import apply_mx_specs as old_apply
    from src.specs.specs import apply_mx_specs

    test_cases = [
        {"bfloat": 16},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16},
        {"bfloat": 16, "round": "floor"},
        {"fp": 8, "quantize_backprop": False},
    ]
    for user_specs in test_cases:
        old_result = old_apply(user_specs.copy())
        new_result = apply_mx_specs(user_specs.copy())
        for key in ALL_SPEC_KEYS:
            assert new_result[key] == old_result[key], \
                f"apply_mx_specs mismatch for {key!r} with input {user_specs}: {new_result[key]} != {old_result[key]}"
```

**Step 6: Run test to verify it fails**

Run: `pytest src/tests/test_specs_equiv.py::test_apply_mx_specs -v`
Expected: FAIL — `apply_mx_specs` not defined

**Step 7: Implement `apply_mx_specs`**

Add `apply_mx_specs(mx_specs, default_mx_specs=None)` to `src/specs/specs.py`. Logic: if `mx_specs` is None/empty, return defaults; iterate keys, set non-None values; raise KeyError for unknown keys.

**Step 8: Run test to verify it passes**

Run: `pytest src/tests/test_specs_equiv.py::test_apply_mx_specs -v`
Expected: PASS

**Step 9: Write the failing test — `finalize_mx_specs`**

```python
def test_finalize_mx_specs():
    from mx.specs import finalize_mx_specs as old_finalize
    from src.specs.specs import finalize_mx_specs

    test_cases = [
        {"bfloat": 16},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp4_e2m1", "block_size": 32, "bfloat": 16},
        {"w_elem_format": "int8", "a_elem_format": "int8", "block_size": 32, "bfloat": 16},
        {"bfloat": 16, "quantize_backprop": False},
        {"bfloat": 16, "round": "floor"},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16,
         "w_elem_format_bp": "fp4_e2m1"},
        {},  # Should return None (no quantization)
    ]
    for user_specs in test_cases:
        old_result = old_finalize(user_specs.copy())
        new_result = finalize_mx_specs(user_specs.copy())
        if old_result is None:
            assert new_result is None, f"Expected None for {user_specs}"
        else:
            for key in ALL_SPEC_KEYS:
                assert new_result[key] == old_result[key], \
                    f"finalize mismatch for {key!r} with {user_specs}: {new_result[key]} != {old_result[key]}"
```

**Step 10: Run test to verify it fails**

Run: `pytest src/tests/test_specs_equiv.py::test_finalize_mx_specs -v`
Expected: FAIL

**Step 11: Implement `finalize_mx_specs`**

Add to `src/specs/specs.py`. Logic: early exit if no quantization params set; validate custom_cuda; `assign_if_none` for backward format inheritance and rounding mode inheritance; call `apply_mx_specs`.

**Step 12: Run test to verify it passes**

Run: `pytest src/tests/test_specs_equiv.py::test_finalize_mx_specs -v`
Expected: PASS

**Step 13: Write the failing test — `get_backwards_mx_specs`**

```python
def test_get_backwards_mx_specs():
    from mx.specs import get_backwards_mx_specs as old_get_bw, finalize_mx_specs as old_finalize
    from src.specs.specs import get_backwards_mx_specs, finalize_mx_specs

    specs_with_bp = finalize_mx_specs({"bfloat": 16, "quantize_backprop": True})
    specs_no_bp = finalize_mx_specs({"bfloat": 16, "quantize_backprop": False})

    old_bw_bp = old_get_bw(old_finalize({"bfloat": 16, "quantize_backprop": True}))
    old_bw_no = old_get_bw(old_finalize({"bfloat": 16, "quantize_backprop": False}))
    new_bw_bp = get_backwards_mx_specs(specs_with_bp)
    new_bw_no = get_backwards_mx_specs(specs_no_bp)

    for key in ALL_SPEC_KEYS:
        assert new_bw_bp[key] == old_bw_bp[key], f"backwards_bp mismatch: {key}"
        assert new_bw_no[key] == old_bw_no[key], f"backwards_no_bp mismatch: {key}"
```

**Step 14: Run test to verify it fails**

Expected: FAIL

**Step 15: Implement `get_backwards_mx_specs` and `mx_assert_test`**

**Step 16: Run test to verify it passes**

**Step 17: Commit**

```bash
git add src/specs/ src/tests/test_specs_equiv.py
git commit -m "Phase 2 Task 1: Migrate MxSpecs config system to src/specs/"
```

---

### Task 2: `src/quantize/elemwise.py` — Element-wise Quantization

**Files:**
- Create: `src/quantize/__init__.py`
- Create: `src/quantize/elemwise.py`
- Test: `src/tests/test_elemwise_equiv.py`

**Step 1: Write the failing test — `_round_mantissa`**

```python
# src/tests/test_elemwise_equiv.py
import pytest
import torch
from mx.elemwise_ops import _round_mantissa as old_round_mantissa
from src.quantize.elemwise import round_mantissa

@pytest.mark.parametrize("round_mode", ["nearest", "floor", "even", "dither"])
def test_round_mantissa(round_mode):
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_out = old_round_mantissa(A.clone(), bits=4, round=round_mode)
    new_out = round_mantissa(A.clone(), bits=4, round_mode=round_mode)
    assert torch.equal(old_out, new_out), f"round_mantissa mismatch for {round_mode}"

def test_round_mantissa_clamp():
    A = torch.tensor([10.0, -10.0, 0.0, 3.5])
    old_out = old_round_mantissa(A.clone(), bits=4, round="nearest", clamp=True)
    new_out = round_mantissa(A.clone(), bits=4, round_mode="nearest", clamp=True)
    assert torch.equal(old_out, new_out)

def test_round_mantissa_tie_breaking():
    A = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5])
    old_out = old_round_mantissa(A.clone(), bits=4, round="even")
    new_out = round_mantissa(A.clone(), bits=4, round_mode="even")
    assert torch.equal(old_out, new_out)
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `_safe_lshift`, `_safe_rshift`, `round_mantissa` in `src/quantize/elemwise.py`**

Note: rename `round` parameter to `round_mode`. Keep logic identical otherwise.

**Step 4: Run test to verify it passes**

**Step 5: Write the failing test — `_quantize_elemwise_core`**

```python
from mx.elemwise_ops import _quantize_elemwise_core as old_core
from src.quantize.elemwise import quantize_elemwise_core
from src.formats.base import FormatBase

ALL_ELEM_FORMATS = ["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3",
                    "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "float16", "bfloat16"]

@pytest.mark.parametrize("fmt_name", ALL_ELEM_FORMATS)
def test_quantize_elemwise_core_normal(fmt_name):
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round="nearest")
    new_out = quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round_mode="nearest")
    assert torch.equal(old_out, new_out), f"core mismatch for {fmt_name}"

@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1", "int8"])
def test_quantize_elemwise_core_no_denorm(fmt_name):
    torch.manual_seed(42)
    A = torch.randn(4, 32) * 0.01  # small values to trigger subnormals
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round="nearest", allow_denorm=False)
    new_out = quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                                     round_mode="nearest", allow_denorm=False)
    assert torch.equal(old_out, new_out)

def test_quantize_elemwise_core_nan_inf():
    A = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, 1.0])
    old_out = old_core(A.clone(), 5, 4, 448.0, round="nearest")
    new_out = quantize_elemwise_core(A.clone(), 5, 4, 448.0, round_mode="nearest")
    assert torch.equal(old_out, new_out)
```

**Step 6: Run test to verify it fails**

**Step 7: Implement `quantize_elemwise_core`**

Key changes from old code:
- Replace `_get_min_norm(ebits)` with `compute_min_norm(ebits)` from `src/formats/base.py`
- Replace `RoundingMode.string_enums()` with `_VALID_ROUND_MODES` check
- Parameter `round` → `round_mode`
- Skip custom_cuda path for now (Phase 2 is CPU/Python path only)
- Handle sparse tensors

**Step 8: Run test to verify it passes**

**Step 9: Write the failing test — `quantize_elemwise_op` (public API)**

```python
from mx.elemwise_ops import quantize_elemwise_op as old_quantize
from src.quantize.elemwise import quantize_elemwise_op
from mx.specs import finalize_mx_specs
from src.specs.specs import finalize_mx_specs as new_finalize

MX_CONFIGS = [
    {"bfloat": 16},
    {"bfloat": 12},
    {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16},
]

@pytest.mark.parametrize("config", MX_CONFIGS)
def test_quantize_elemwise_op(config):
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_specs = finalize_mx_specs(config.copy())
    new_specs = new_finalize(config.copy())
    old_out = old_quantize(A.clone(), mx_specs=old_specs)
    new_out = quantize_elemwise_op(A.clone(), mx_specs=new_specs)
    assert torch.equal(old_out, new_out)
```

**Step 10: Run test to verify it fails**

**Step 11: Implement `_quantize_bfloat`, `_quantize_fp`, `quantize_elemwise_op`**

Key changes:
- Replace `_get_max_norm(ebits, mbits)` with `compute_max_norm(ebits, mbits)` from `src/formats/base.py`
- Parameter `round` → `round_mode`

**Step 12: Run test to verify it passes**

**Step 13: Commit**

```bash
git add src/quantize/ src/tests/test_elemwise_equiv.py
git commit -m "Phase 2 Task 2: Migrate element-wise quantization to src/quantize/elemwise.py"
```

---

### Task 3: `src/quantize/mx_quantize.py` — MX Block Quantization

**Files:**
- Create: `src/quantize/mx_quantize.py`
- Test: `src/tests/test_mx_quantize_equiv.py`

**Step 1: Write the failing test — `_shared_exponents`**

```python
# src/tests/test_mx_quantize_equiv.py
import pytest
import torch
from mx.mx_ops import _shared_exponents as old_shared_exp
from src.quantize.mx_quantize import shared_exponents

def test_shared_exponents_max():
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_shared_exp(A.clone(), method="max", axes=[-1])
    new_out = shared_exponents(A.clone(), method="max", axes=[-1])
    assert torch.equal(old_out, new_out)

def test_shared_exponents_none():
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_shared_exp(A.clone(), method="none")
    new_out = shared_exponents(A.clone(), method="none")
    assert torch.equal(old_out, new_out)

def test_shared_exponents_multi_axes():
    torch.manual_seed(42)
    A = torch.randn(2, 4, 64)
    old_out = old_shared_exp(A.clone(), method="max", axes=[-2, -1])
    new_out = shared_exponents(A.clone(), method="max", axes=[-2, -1])
    assert torch.equal(old_out, new_out)
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `shared_exponents` in `src/quantize/mx_quantize.py`**

Key changes: replace `FP32_EXPONENT_BIAS` / `FP32_MIN_NORMAL` from `mx/formats.py` with constants defined locally in the new file.

**Step 4: Run test to verify it passes**

**Step 5: Write the failing test — `_reshape_to_blocks` / `_undo_reshape_to_blocks`**

```python
from mx.mx_ops import _reshape_to_blocks as old_reshape, _undo_reshape_to_blocks as old_undo
from src.quantize.mx_quantize import reshape_to_blocks, undo_reshape_to_blocks

def test_reshape_undo_blocks_no_padding():
    A = torch.randn(2, 64)
    old_A, old_axes, old_orig, old_padded = old_reshape(A.clone(), axes=[-1], block_size=32)
    new_A, new_axes, new_orig, new_padded = reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    assert torch.equal(old_A, new_A)

def test_reshape_undo_blocks_with_padding():
    A = torch.randn(2, 48)  # 48 not divisible by 32
    old_A, old_axes, old_orig, old_padded = old_reshape(A.clone(), axes=[-1], block_size=32)
    new_A, new_axes, new_orig, new_padded = reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    assert torch.equal(old_A, new_A)

def test_reshape_undo_roundtrip():
    A = torch.randn(2, 64)
    reshaped, axes, orig_shape, padded_shape = reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    recovered = undo_reshape_to_blocks(reshaped, padded_shape, orig_shape, axes)
    assert torch.equal(A, recovered)
```

**Step 6: Run test to verify it fails**

**Step 7: Implement `reshape_to_blocks` and `undo_reshape_to_blocks`**

**Step 8: Run test to verify it passes**

**Step 9: Write the failing test — `_quantize_mx`**

```python
from mx.mx_ops import _quantize_mx as old_quantize_mx
from src.quantize.mx_quantize import quantize_mx
from mx.formats import ElemFormat

MX_FORMATS = ["fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "int8", "int4", "int2"]

@pytest.mark.parametrize("fmt", MX_FORMATS)
def test_quantize_mx(fmt):
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_quantize_mx(A.clone(), scale_bits=8, elem_format=ElemFormat.from_str(fmt),
                              block_size=32, axes=[-1], round="nearest")
    new_out = quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                          block_size=32, axes=[-1], round_mode="nearest")
    assert torch.equal(old_out, new_out), f"mx quantize mismatch for {fmt}"

@pytest.mark.parametrize("fmt", ["fp8_e4m3", "int8"])
def test_quantize_mx_block64(fmt):
    torch.manual_seed(42)
    A = torch.randn(4, 128)
    old_out = old_quantize_mx(A.clone(), scale_bits=8, elem_format=ElemFormat.from_str(fmt),
                              block_size=64, axes=[-1], round="nearest")
    new_out = quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                          block_size=64, axes=[-1], round_mode="nearest")
    assert torch.equal(old_out, new_out)
```

**Step 10: Run test to verify it fails**

**Step 11: Implement `quantize_mx` and `quantize_mx_op`**

Key changes:
- Replace `_get_format_params(elem_format)` with `FormatBase.from_str(elem_format)` attribute access
- Replace `RoundingMode.string_enums()` with `_VALID_ROUND_MODES`
- Replace `ElemFormat.from_str()` conversion with direct string usage
- Parameter `round` → `round_mode`
- Skip custom_cuda path for now

**Step 12: Run test to verify it passes**

**Step 13: Write the failing test — `quantize_mx_op` (public API using mx_specs)**

```python
from mx.mx_ops import quantize_mx_op as old_qmx_op
from src.quantize.mx_quantize import quantize_mx_op
from src.specs.specs import finalize_mx_specs as new_finalize

@pytest.mark.parametrize("fmt", MX_FORMATS)
def test_quantize_mx_op(fmt):
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    config = {"w_elem_format": fmt, "a_elem_format": fmt, "block_size": 32, "bfloat": 16}
    old_specs = old_finalize(config.copy())
    new_specs = new_finalize(config.copy())
    old_out = old_qmx_op(A.clone(), mx_specs=old_specs, elem_format=fmt, axes=[-1])
    new_out = quantize_mx_op(A.clone(), mx_specs=new_specs, elem_format=fmt, axes=[-1])
    assert torch.equal(old_out, new_out)
```

**Step 14: Run test to verify it fails**

**Step 15: Implement `quantize_mx_op`** — wraps `quantize_mx` with mx_specs dict access

**Step 16: Run test to verify it passes**

**Step 17: Commit**

```bash
git add src/quantize/mx_quantize.py src/tests/test_mx_quantize_equiv.py
git commit -m "Phase 2 Task 3: Migrate MX block quantization to src/quantize/mx_quantize.py"
```

---

### Task 4: `src/quantize/bfloat_quantize.py` — Differentiable Bfloat Quantization

**Files:**
- Create: `src/quantize/bfloat_quantize.py`
- Test: `src/tests/test_bfloat_quantize_equiv.py`

**Step 1: Write the failing test — `quantize_bfloat` forward**

```python
# src/tests/test_bfloat_quantize_equiv.py
import pytest
import torch
from mx.quantize import quantize_bfloat as old_qbf
from src.quantize.bfloat_quantize import quantize_bfloat
from mx.specs import finalize_mx_specs as old_finalize
from src.specs.specs import finalize_mx_specs as new_finalize

def test_quantize_bfloat_forward():
    torch.manual_seed(42)
    x = torch.randn(4, 32)
    old_specs = old_finalize({"bfloat": 16})
    new_specs = new_finalize({"bfloat": 16})
    old_out = old_qbf(x.clone(), mx_specs=old_specs)
    new_out = quantize_bfloat(x.clone(), mx_specs=new_specs)
    assert torch.equal(old_out, new_out)
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `quantize_bfloat` and `QuantizeBfloatFunction`**

**Step 4: Run test to verify it passes**

**Step 5: Write the failing test — backward pass**

```python
def test_quantize_bfloat_backward():
    torch.manual_seed(42)
    x = torch.randn(4, 32, requires_grad=True)
    old_specs = old_finalize({"bfloat": 16, "quantize_backprop": True})
    x2 = x.clone().detach().requires_grad_(True)
    old_out = old_qbf(x, mx_specs=old_specs)
    old_out.sum().backward()

    new_specs = new_finalize({"bfloat": 16, "quantize_backprop": True})
    new_out = quantize_bfloat(x2, mx_specs=new_specs)
    new_out.sum().backward()
    assert torch.equal(x.grad, x2.grad)
```

**Step 6: Run test to verify it fails**

**Step 7: Implement backward in `QuantizeBfloatFunction`**

**Step 8: Run test to verify it passes**

**Step 9: Commit**

```bash
git add src/quantize/bfloat_quantize.py src/tests/test_bfloat_quantize_equiv.py
git commit -m "Phase 2 Task 4: Migrate differentiable bfloat quantization"
```

---

### Task 5: `src/quantize/vector.py` — Vector Quantization Wrapper

**Files:**
- Create: `src/quantize/vector.py`
- Test: `src/tests/test_vector_equiv.py`

**Step 1: Write the failing test — `vec_quantize`**

```python
# src/tests/test_vector_equiv.py
import pytest
import torch
from mx.vector_ops import vec_quantize as old_vec_q
from src.quantize.vector import vec_quantize
from mx.specs import finalize_mx_specs as old_finalize
from src.specs.specs import finalize_mx_specs as new_finalize

def test_vec_quantize():
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_specs = old_finalize({"bfloat": 16})
    new_specs = new_finalize({"bfloat": 16})
    old_out = old_vec_q(A.clone(), mx_specs=old_specs)
    new_out = vec_quantize(A.clone(), mx_specs=new_specs)
    assert torch.equal(old_out, new_out)
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `vec_quantize`** — simple wrapper around `quantize_elemwise_op`

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add src/quantize/vector.py src/tests/test_vector_equiv.py
git commit -m "Phase 2 Task 5: Migrate vector quantization wrapper"
```

---

### Task 6: Golden Reference Integration Tests

**Files:**
- Test: `src/tests/test_golden_equiv.py`

**Step 1: Write golden reference tests for core quantization functions**

```python
# src/tests/test_golden_equiv.py
import pytest
import torch
import os
from src.quantize.elemwise import quantize_elemwise_op
from src.quantize.mx_quantize import quantize_mx_op
from src.quantize.bfloat_quantize import quantize_bfloat
from src.specs.specs import finalize_mx_specs

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")

# Test elemwise_quantize golden files
ELEMWISE_FORMATS = ["mxfp8_e5m2", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp6_e2m3",
                    "mxfp4_e2m1", "mxint8", "mxint4", "mxint2", "bfloat16"]

@pytest.mark.parametrize("cfg_name", ELEMWISE_FORMATS)
def test_elemwise_quantize_golden(cfg_name):
    golden = torch.load(os.path.join(GOLDEN_DIR, f"elemwise_quantize_{cfg_name}.pt"))
    new_out = quantize_elemwise_op(golden["input"].clone(), mx_specs=finalize_mx_specs(golden["mx_specs"]))
    assert torch.equal(new_out, golden["output"]), f"elemwise golden mismatch for {cfg_name}"

@pytest.mark.parametrize("cfg_name", ELEMWISE_FORMATS)
def test_quantize_bfloat_golden(cfg_name):
    golden = torch.load(os.path.join(GOLDEN_DIR, f"quantize_bfloat_{cfg_name}.pt"))
    new_out = quantize_bfloat(golden["input"].clone(), mx_specs=finalize_mx_specs(golden["mx_specs"]))
    assert torch.equal(new_out, golden["output"]), f"bfloat golden mismatch for {cfg_name}"

MX_QUANTIZE_FORMATS = ["fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3",
                       "fp4_e2m1", "int8", "int4", "int2"]

@pytest.mark.parametrize("fmt", MX_QUANTIZE_FORMATS)
def test_mx_quantize_golden(fmt):
    golden = torch.load(os.path.join(GOLDEN_DIR, f"mx_quantize_{fmt}_bs32.pt"))
    new_out = quantize_mx_op(golden["input"].clone(), mx_specs=finalize_mx_specs(golden["mx_specs"]),
                             elem_format=fmt, axes=[-1])
    assert torch.equal(new_out, golden["output"]), f"mx golden mismatch for {fmt}"
```

**Step 2: Run test to verify it passes**

**Step 3: Commit**

```bash
git add src/tests/test_golden_equiv.py
git commit -m "Phase 2 Task 6: Add golden reference integration tests for quantization core"
```

---

### Task 7: Update `src/quantize/__init__.py` and Final Verification

**Step 1: Create `src/quantize/__init__.py` with public API exports**

```python
from .elemwise import quantize_elemwise_op
from .mx_quantize import quantize_mx_op
from .bfloat_quantize import quantize_bfloat
from .vector import vec_quantize
```

**Step 2: Run ALL tests together**

Run: `pytest src/tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/quantize/__init__.py
git commit -m "Phase 2 Task 7: Finalize quantization module public API"
```
