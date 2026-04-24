# Phase 2: Quantization Core Migration Design

## Scope

Migrate `specs`, `elemwise_ops`, `mx_ops` into `src/`. Operators remain in `mx/` for Phase 3.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Migration scope | Quantization core only, operators stay | Manageable scope; operators depend on core |
| Format dependency | New code uses `src/formats/` directly | Cohesive; avoids redundant parameter passing |
| Specs structure | Faithful 39-key dict migration | Low risk; operators deeply depend on dict access |
| Migration style | Rewrite each function | Clean code; uses new FormatBase API; tests catch regressions |

## Module Design

### `src/specs/` — Configuration System

Files:
- `specs.py` — `MxSpecs(UserDict)`, all 39 keys, defaults, help strings
- `defaults.py` — `get_default_mx_specs()`, default values dict
- `finalize.py` — `finalize_mx_specs()`, `apply_mx_specs()`, `get_backwards_mx_specs()`
- `utils.py` — `mx_assert_test()`, `add_mx_args()`, `get_mx_specs()` (argparse bridge)

Public API:
- `MxSpecs` — 39-key config dict (identical to old code)
- `get_default_mx_specs()` — returns default MxSpecs
- `apply_mx_specs(user_specs, default)` — merge user config into defaults
- `finalize_mx_specs(specs)` — validate + fill backward/rounding overrides
- `get_backwards_mx_specs(specs)` — return specs for backward pass
- `mx_assert_test(specs)` — assertion mode check

Relationship with Phase 1 `QuantScheme`: QuantScheme is a high-level abstraction (4 fields). MxSpecs is the low-level implementation config (39 keys). Phase 3 operators can use MxSpecs directly or derive it from QuantScheme.

### `src/quantize/` — Quantization Core

**`elemwise.py`** — Element-wise quantization:
- `quantize_elemwise_op(A, mx_specs, round=None)` — public API
- `_round_mantissa(A, bits, round_mode, clamp=False)` — mantissa rounding (4 modes: nearest, floor, even, dither)
- `_quantize_elemwise_core(A, ebits, mbits, emax, max_norm, max_exp, min_exp, ...)` — core algorithm
- `_quantize_bfloat(A, mx_specs, round=None)` — bfloat quantization
- `_quantize_fp(A, mx_specs, round=None)` — custom FP quantization
- `_safe_lshift(A, bits)` / `_safe_rshift(A, bits)` — safe bit shifts

Internal changes from old code:
- Replace `_get_format_params(ElemFormat)` calls with `FormatBase.from_str(name)` attribute access
- Parameter name `round` → `round_mode` (consistent with QuantScheme)
- Use `compute_min_norm(ebits)` / `compute_max_norm(ebits, mbits)` from `src/formats/base.py`

**`mx_quantize.py`** — MX block quantization:
- `quantize_mx_op(A, mx_specs, elem_format, axes, ...)` — public API
- `_shared_exponents(A, method, axes)` — shared exponent calculation
- `_reshape_to_blocks(A, block_size, axes)` — reshape into blocks
- `_undo_reshape_to_blocks(A, padded_shape, original_shape, ...)` — reverse reshape
- `_quantize_mx(A, mx_specs, elem_format, ...)` — core algorithm

Internal changes:
- Replace `_get_format_params()` with `FormatBase.from_str()` for format parameter access
- Use `FormatBase.is_integer` instead of checking ebits == 0

**`bfloat_quantize.py`** — Differentiable bfloat quantization:
- `quantize_bfloat(x, mx_specs, round=None)` — public API
- `QuantizeBfloatFunction(torch.autograd.Function)` — forward calls `quantize_elemwise_op`, backward quantizes gradients

**`vector.py`** — Vector quantization wrapper:
- `vec_quantize(A, mx_specs, round=None)` — wraps `quantize_elemwise_op`, used by SIMD/norm/activation ops

### Dependency Graph

```
src/formats/          ← Phase 1 (complete)
     ↑
src/specs/            ← Phase 2 (new)
     ↑
src/quantize/elemwise.py    ← Phase 2 (new)
     ↑
src/quantize/mx_quantize.py ← Phase 2 (new, depends on elemwise)
src/quantize/bfloat_quantize.py ← Phase 2 (new, depends on elemwise)
src/quantize/vector.py           ← Phase 2 (new, depends on elemwise)
```

## Test Strategy

**Principle: Every function tested independently. New code vs old code on same input → bit-identical output.**

### A. `src/specs/` Tests

| Function | Test |
|----------|------|
| `get_default_mx_specs()` | 39-key comparison against old `get_default_mx_specs()` |
| `apply_mx_specs(user, default)` | Multiple user configs → key-by-key comparison with old code |
| `finalize_mx_specs(specs)` | Each config combo (bfloat-only, MX-only, mixed format, backward overrides, rounding overrides) → key-by-key comparison |
| `get_backwards_mx_specs(specs)` | `quantize_backprop=True/False` → key-by-key comparison of returned specs |
| `mx_assert_test(specs)` | None specs raises in assert mode; valid specs does not raise |

### B. `src/quantize/elemwise.py` Tests

| Function | Test |
|----------|------|
| `_round_mantissa(A, bits, round_mode)` | 4 round modes × inputs (positive, negative, zero, 0.5 tie-breaking, huge values) → bit-identical vs old `_round_mantissa` |
| `_safe_lshift(A, bits)` / `_safe_rshift(A, bits)` | Edge cases (zero, max exponent, negative shift) → bit-identical vs old |
| `_quantize_elemwise_core(A, ebits, mbits, ...)` | All 10 formats × inputs (normal, subnormal, zero, NaN, Inf, sparse) → bit-identical vs old `_quantize_elemwise_core` |
| `_quantize_bfloat(A, mx_specs, round)` | bfloat16/12/10 × multiple inputs → bit-identical vs old `_quantize_bfloat` |
| `_quantize_fp(A, mx_specs, round)` | fp8_e4m3/fp6_e3m2/fp4_e2m1 etc. × multiple inputs → bit-identical vs old `_quantize_fp` |
| `quantize_elemwise_op(A, mx_specs, round)` | Public API, verified against golden reference `elemwise_quantize_*` files |

### C. `src/quantize/mx_quantize.py` Tests

| Function | Test |
|----------|------|
| `_shared_exponents(A, method, axes)` | method="max"/"none" × different axes × different shapes → bit-identical vs old |
| `_reshape_to_blocks()` / `_undo_reshape_to_blocks()` | block_size=32/64 × padding needed / not needed → reshape+undo recovers original tensor |
| `_quantize_mx(A, mx_specs, elem_format, ...)` | All 8 MX formats × block_size=32/64 × axes=[-1]/[-2] → bit-identical vs old |
| `quantize_mx_op(A, mx_specs, elem_format, ...)` | Public API, verified against golden reference `mx_quantize_*` files |

### D. `src/quantize/bfloat_quantize.py` Tests

| Function | Test |
|----------|------|
| `QuantizeBfloatFunction.forward` | Output bit-identical vs old `QuantizeBfloatFunction.forward` |
| `QuantizeBfloatFunction.backward` | `quantize_backprop=True/False` → gradient bit-identical vs old |
| `quantize_bfloat(x, mx_specs, round)` | Public API, verified against golden reference `quantize_bfloat_*` files |

### E. `src/quantize/vector.py` Tests

| Function | Test |
|----------|------|
| `vec_quantize(A, mx_specs, round)` | Multiple formats → output bit-identical vs old `vec_quantize` |

### F. Boundary & Edge Cases (covering all internal functions)

- Subnormal number handling (flush to zero vs preserve)
- NaN / Inf inputs
- Zero tensors, empty tensors
- Block size not dividing tensor dimension (requires padding)
- Sparse tensors
- Values exceeding max_norm / below min_norm
