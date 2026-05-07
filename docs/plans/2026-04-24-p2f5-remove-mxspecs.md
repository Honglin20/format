# P2F-5: Eliminate MxSpecs Dependencies from src/quantize/

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all MxSpecs dependencies in `src/quantize/` with QuantScheme-driven APIs, keeping thin compat wrappers so existing tests pass.

**Architecture:** Three files need changes: `mx_quantize.py` (new `quantize_mx` + compat `quantize_mx_op`), `bfloat_quantize.py` (rewrite autograd Function to store QuantScheme, new `quantize_bfloat` + compat wrapper), `vector.py` (add `scheme` param alongside `mx_specs`, move behavioral flags to explicit params). The pattern follows P2F-4: new QuantScheme API is primary, old mx_specs signature becomes thin compat wrapper.

**Tech Stack:** Python 3.10+, PyTorch, dataclasses (QuantScheme/GranularitySpec)

---

### Task 1: mx_quantize.py — Add `quantize_mx` + rewrite `quantize_mx_op` as compat wrapper

**Files:**
- Modify: `src/quantize/mx_quantize.py:1-230` (entire file)
- Test: `src/tests/test_mx_quantize_equiv.py` (should pass unchanged)

**Step 1: Remove `from src.specs.specs import mx_assert_test`**

Delete the import at line 14. No code in the file needs it after the compat rewrite.

**Step 2: Add `quantize_mx` — the new QuantScheme-driven API**

Insert after `_quantize_mx` (after line 195), before the old `quantize_mx_op`:

```python
def quantize_mx(
    A,
    scheme,
    axes=None,
    scale_bits=8,
    shared_exp_method="max",
    flush_fp32_subnorms=False,
):
    """Quantize tensor A using a QuantScheme (MX block quantization).

    Primary QuantScheme-driven API for shared-exponent block quantization.

    Args:
        A: Input tensor.
        scheme: QuantScheme specifying format, granularity, and round_mode.
            block_size is taken from scheme.granularity.block_size.
        axes: Axes along which to compute shared exponents. Default: -1.
        scale_bits: Bits for shared scale (sign + magnitude). Default: 8.
        shared_exp_method: "max" or "none". Default: "max".
        flush_fp32_subnorms: Flush subnormal FP32 blocks to zero. Default: False.

    Returns:
        Quantized tensor with same shape as A.
    """
    if scheme is None:
        return A

    fmt = scheme.format
    block_size = scheme.block_size
    round_mode = scheme.round_mode

    return _quantize_mx(
        A, scale_bits, fmt,
        block_size=block_size,
        axes=axes, round_mode=round_mode,
        shared_exp_method=shared_exp_method,
        flush_fp32_subnorms=flush_fp32_subnorms,
    )
```

**Step 3: Rewrite `quantize_mx_op` as compat wrapper**

Replace the existing `quantize_mx_op` with a thin wrapper that constructs QuantScheme and delegates to `quantize_mx`:

```python
def quantize_mx_op(
    A,
    mx_specs,
    elem_format=None,
    block_size=None,
    axes=None,
    round_mode="nearest",
    expand_and_reshape=False,
):
    """Compat wrapper: quantize A using mx_specs dict.

    Delegates to quantize_mx() after constructing a QuantScheme from mx_specs.
    Kept for backward compatibility with existing tests (remove in P2F-6).
    """
    if elem_format is None:
        return A

    if block_size is None:
        block_size = mx_specs["block_size"]

    scale_bits = mx_specs["scale_bits"] if mx_specs["scale_bits"] != 0 else 8

    fmt = FormatBase.from_str(elem_format) if isinstance(elem_format, str) else elem_format

    from src.scheme.quant_scheme import QuantScheme
    from src.scheme.granularity import GranularitySpec

    scheme = QuantScheme(
        format=fmt,
        granularity=GranularitySpec.per_block(block_size) if block_size > 0
                     else GranularitySpec.per_tensor(),
        round_mode=round_mode,
    )

    return quantize_mx(
        A, scheme, axes=axes,
        scale_bits=scale_bits,
        shared_exp_method=mx_specs["shared_exp_method"],
        flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
    )
```

Note: `expand_and_reshape` is a dead parameter (never used in function body). Kept in compat signature to avoid breaking callers, but ignored.

**Step 4: Run equivalence tests for mx_quantize**

Run: `pytest src/tests/test_mx_quantize_equiv.py -v`
Expected: All PASS (compat wrapper produces identical results)

**Step 5: Commit**

```bash
git add src/quantize/mx_quantize.py
git commit -m "feat(quantize): add quantize_mx(A, scheme) API, rewrite quantize_mx_op as compat wrapper (P2F-5)"
```

---

### Task 2: bfloat_quantize.py — Rewrite autograd Function to use QuantScheme

**Files:**
- Modify: `src/quantize/bfloat_quantize.py:1-47` (entire file)
- Modify: `src/tests/test_bfloat_quantize_equiv.py:10-12` (update import)
- Test: `src/tests/test_bfloat_quantize_equiv.py` (must pass)

**Step 1: Rewrite file to use QuantScheme, add compat wrapper**

Replace entire `src/quantize/bfloat_quantize.py`:

```python
"""
Differentiable bfloat quantization.

Primary API: quantize_bfloat(x, scheme, ...) — QuantScheme-driven.
Compat API:  quantize_bfloat_from_specs(x, mx_specs, ...) — MxSpecs wrapper (P2F-6 removes).
"""
import torch

from src.quantize.elemwise import quantize


class QuantizeBfloatFunction(torch.autograd.Function):
    """Forward: quantize to bfloat. Backward: quantize gradients to bfloat."""

    @staticmethod
    def forward(ctx, x, scheme, backwards_scheme=None, allow_denorm=True):
        ctx.backwards_scheme = backwards_scheme
        ctx.allow_denorm = allow_denorm
        return quantize(x, scheme, allow_denorm=allow_denorm)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.backwards_scheme is None:
            return (grad_output, None, None, None)
        grad_input = quantize(grad_output, ctx.backwards_scheme,
                              allow_denorm=ctx.allow_denorm)
        return (grad_input, None, None, None)


def quantize_bfloat(x, scheme, backwards_scheme=None, allow_denorm=True):
    """Quantize x using a QuantScheme (bfloat format, differentiable).

    Args:
        x: Input tensor.
        scheme: QuantScheme specifying format, granularity, and round_mode.
        backwards_scheme: QuantScheme for backward pass. If None, backward
            is identity (no quantization). Default: same as scheme.
        allow_denorm: If False, flush subnormal values to zero.

    Returns:
        Quantized tensor with same shape as x.
    """
    if scheme is None:
        return x

    if backwards_scheme is None:
        backwards_scheme = scheme

    return QuantizeBfloatFunction.apply(x, scheme, backwards_scheme, allow_denorm)


def quantize_bfloat_from_specs(x, mx_specs, round_mode=None):
    """Compat wrapper: quantize x using mx_specs dict.

    Constructs QuantScheme from mx_specs and delegates to quantize_bfloat().
    Kept for backward compatibility with existing tests (remove in P2F-6).
    """
    if mx_specs is None:
        return x

    from src.quantize.elemwise import _format_from_mx_specs
    from src.scheme.quant_scheme import QuantScheme
    from src.scheme.granularity import GranularitySpec

    fmt = _format_from_mx_specs(mx_specs)
    if fmt is None:
        return x

    if round_mode is None:
        round_mode = mx_specs["round"]

    allow_denorm = mx_specs.get("bfloat_subnorms", True)

    scheme = QuantScheme(
        format=fmt,
        granularity=GranularitySpec.per_tensor(),
        round_mode=round_mode,
    )

    backwards_scheme = scheme if mx_specs.get("quantize_backprop", True) else None

    return quantize_bfloat(x, scheme, backwards_scheme=backwards_scheme,
                           allow_denorm=allow_denorm)
```

**Step 2: Update test import**

In `src/tests/test_bfloat_quantize_equiv.py`, change line 10-12:

From:
```python
from src.quantize.bfloat_quantize import quantize_bfloat
from mx.specs import finalize_mx_specs as old_finalize
from src.specs.specs import finalize_mx_specs as new_finalize
```

To:
```python
from src.quantize.bfloat_quantize import quantize_bfloat_from_specs
from mx.specs import finalize_mx_specs as old_finalize
from src.specs.specs import finalize_mx_specs as new_finalize
```

And update all calls from `quantize_bfloat(x, mx_specs=...)` to `quantize_bfloat_from_specs(x, mx_specs=...)` throughout the file.

**Step 3: Run equivalence tests for bfloat_quantize**

Run: `pytest src/tests/test_bfloat_quantize_equiv.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/quantize/bfloat_quantize.py src/tests/test_bfloat_quantize_equiv.py
git commit -m "feat(quantize): add quantize_bfloat(x, scheme) API, rewrite old signature as compat wrapper (P2F-5)"
```

---

### Task 3: vector.py — QuantScheme-driven vector functions

**Files:**
- Modify: `src/quantize/vector.py:1-104` (entire file)
- Test: `src/tests/test_vector_equiv.py` (must pass unchanged)

**Step 1: Rewrite vector.py with QuantScheme support**

Replace entire `src/quantize/vector.py`. Strategy: add `scheme` parameter to all functions (takes priority over `mx_specs`). Move `vec_use_exp2`/`vec_use_recip` to explicit params. Internally use `quantize()` for scheme path, `quantize_elemwise_op` for mx_specs compat path.

```python
"""
Non-differentiable vector quantization operations.

Primary API: vec_quantize(input, scheme, ...) — QuantScheme-driven.
Compat API:  vec_quantize(input, mx_specs=..., ...) — MxSpecs wrapper (P2F-6 removes).
"""
import numpy as np
import torch

from src.quantize.elemwise import quantize_elemwise_op, quantize
from src.scheme.quant_scheme import QuantScheme

torch_exp = torch.exp
torch_exp2 = torch.exp2
torch_sqrt = torch.sqrt
torch_tanh = torch.tanh

LN_2_EXACT = 0.69314718056
LOG2_E_BF16 = 1.4453125  # 1 + 2**-2 + 2**-3 + 2**-4 + 2**-7


def _dispatch_quantize(input, scheme=None, mx_specs=None, round_mode=None):
    """Internal: dispatch to quantize() or quantize_elemwise_op()."""
    if scheme is not None:
        return quantize(input, scheme)
    return quantize_elemwise_op(input, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Quantize
# -------------------------------------------------------------------------

def vec_quantize(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(input, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Arithmetic ops
# -------------------------------------------------------------------------

def vec_add(a, b, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(a + b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_sub(a, b, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(a - b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_mul(a, b, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(a * b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_div(a, b, scheme=None, mx_specs=None, round_mode=None, use_recip=False):
    if not use_recip and mx_specs and mx_specs.get('vec_use_recip'):
        use_recip = True
    if use_recip:
        recip_b = vec_recip(b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
        return vec_mul(a, recip_b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    return _dispatch_quantize(a / b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Special math ops
# -------------------------------------------------------------------------

def vec_exp(input, scheme=None, mx_specs=None, round_mode=None, use_exp2=False):
    if not use_exp2 and mx_specs and mx_specs.get('vec_use_exp2'):
        use_exp2 = True
    if use_exp2:
        phi = _dispatch_quantize(LOG2_E_BF16 * input, scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
        phi = vec_exp2(phi, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    else:
        phi = _dispatch_quantize(torch_exp(input), scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
    return phi


def vec_exp2(input, scheme=None, mx_specs=None, round_mode=None):
    if hasattr(torch, 'exp2'):
        phi = _dispatch_quantize(torch_exp2(input), scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
    else:
        phi = _dispatch_quantize(torch_exp(input * LN_2_EXACT), scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
    return phi


def vec_recip(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(1. / input, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_sqrt(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(torch_sqrt(input), scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_tanh(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(torch_tanh(input), scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Reduce ops
# -------------------------------------------------------------------------

def vec_reduce_sum(input, dim, keepdim=False, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(input.sum(dim, keepdim=keepdim),
                              scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_reduce_mean(input, dim, keepdim=False, scheme=None, mx_specs=None, round_mode=None):
    dim = dim if type(dim) is list else [dim]
    denom = np.prod([input.shape[i] for i in dim])

    s = vec_reduce_sum(input, dim, keepdim=keepdim,
                       scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    s = vec_div(s, denom, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    return s
```

Key design decisions:
- `scheme` parameter takes priority over `mx_specs` (when both provided, `scheme` is used)
- `use_exp2` / `use_recip` are explicit keyword params with `False` default; mx_specs compat path reads them from dict if not explicitly set
- `_dispatch_quantize` helper avoids repeating the scheme-vs-mx_specs dispatch in every function
- Existing tests pass unchanged (they use `mx_specs=` parameter)

**Step 2: Run equivalence tests for vector**

Run: `pytest src/tests/test_vector_equiv.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/quantize/vector.py
git commit -m "feat(quantize): add scheme parameter to vector functions, extract behavioral flags as explicit params (P2F-5)"
```

---

### Task 4: Update `__init__.py` exports

**Files:**
- Modify: `src/quantize/__init__.py`

**Step 1: Add new API exports**

Update `src/quantize/__init__.py`:

```python
from .elemwise import quantize_elemwise_op, quantize
from .mx_quantize import quantize_mx_op, quantize_mx
from .bfloat_quantize import quantize_bfloat, quantize_bfloat_from_specs
from .vector import vec_quantize
```

**Step 2: Commit**

```bash
git add src/quantize/__init__.py
git commit -m "feat(quantize): export new QuantScheme-driven APIs from __init__ (P2F-5)"
```

---

### Task 5: Run full test suite + acceptance check

**Step 1: Run all tests**

Run: `pytest src/tests/ -x -q`
Expected: All PASS (311+ tests)

**Step 2: Check MxSpecs import leakage in src/quantize/**

Run:
```bash
grep -rn "from src.specs" src/quantize/
```
Expected: NO matches (all src.specs.specs imports removed from src/quantize/)

**Step 3: Fix any failures**

If tests fail, investigate and fix before proceeding.

---

### Task 6: Dispatch review agent

Use the review agent template from CLAUDE.md Section 5.2 to check:
- Interface compliance with ADR-001 and ADR-004
- Test coverage for new APIs (positive, negative, boundary)
- No silent type errors or missing validation
- `src/` ↔ `mx/` boundary constraints respected
- Compat wrappers are truly thin (no logic duplication)
