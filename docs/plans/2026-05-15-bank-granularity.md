# BANK Granularity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `GranularityMode.BANK` — row/column-wise coarse-grained scale grouping where each bank shares one amax across all elements in that bank.

**Architecture:** Add `BANK` to the GranularityMode enum, `bank_size`/`bank_axis` to GranularitySpec, `_quantize_per_bank()` to FormatBase, and dispatch + QuantConfig resolution. BANK uses standard amax (fp32/pot supported), not MX shared exponents, and its reduction crosses the non-bank dimensions (unlike PER_BLOCK which subdivides every dimension).

**Tech Stack:** PyTorch, pytest, existing `FormatBase` / `GranularitySpec` / `QuantConfig` infrastructure

---

## Design Summary

### BANK vs PER_BLOCK semantic difference

For M×N tensor with axis=-1, size=16:

| | PER_BLOCK(block_size=32) | BANK(bank_size=16) |
|---|---|---|
| Reshape | (M, N) → (M, N/32, 32) | (M, N) → (M, N/16, 16) |
| amax reduction dims | dim=-1 (inside each block) | dims=(0, -1) (M + inner bank) |
| amax shape | (M, N/32, 1) | (1, N/16, 1) |
| Scale count | M × N/32 | N/16 |

BANK does NOT subdivide the M dimension — all M rows in a bank column share one amax. PER_BLOCK subdivides every dimension.

### GranularitySpec new fields

```python
bank_size: int = 16       # elements per bank along bank_axis
bank_axis: int = -1       # axis to split into banks (synced from w_axis/a_axis)
```

post_init rules:
- BANK + bank_size <= 0 → error
- BANK + block_size != 0 → error (mutually exclusive)
- BANK + channel_axis != 0 → error
- Not BANK + bank_axis != -1 → error (same pattern as block_axis)

---

### Task 1: Write failing tests — GranularitySpec BANK construction

**Files:**
- Create: `src/tests/test_bank_granularity.py`

**Step 1: Write the failing test file**

```python
"""Tests for BANK granularity mode."""
import pytest
import torch
from src.scheme.granularity import GranularitySpec, GranularityMode


class TestBankGranularitySpec:
    """GranularitySpec construction and validation for BANK mode."""

    def test_bank_mode_constructs(self):
        """BANK mode with default bank_size constructs."""
        g = GranularitySpec(mode=GranularityMode.BANK)
        assert g.mode == GranularityMode.BANK
        assert g.bank_size == 16
        assert g.bank_axis == -1

    def test_bank_with_custom_size(self):
        """BANK with custom bank_size."""
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=8, bank_axis=0)
        assert g.bank_size == 8
        assert g.bank_axis == 0

    def test_bank_requires_positive_size(self):
        """BANK with bank_size <= 0 raises."""
        with pytest.raises(ValueError, match="bank_size"):
            GranularitySpec(mode=GranularityMode.BANK, bank_size=0)

    def test_bank_rejects_block_size(self):
        """BANK with block_size != 0 raises."""
        with pytest.raises(ValueError, match="block_size"):
            GranularitySpec(mode=GranularityMode.BANK, block_size=32)

    def test_bank_rejects_channel_axis(self):
        """BANK with channel_axis != 0 raises."""
        with pytest.raises(ValueError, match="channel_axis"):
            GranularitySpec(mode=GranularityMode.BANK, channel_axis=1)

    def test_non_bank_rejects_bank_axis(self):
        """PER_TENSOR with bank_axis != -1 raises."""
        with pytest.raises(ValueError, match="bank_axis"):
            GranularitySpec(mode=GranularityMode.PER_TENSOR, bank_axis=0)

    def test_bank_per_tensor_factory_unchanged(self):
        """Existing factories still work."""
        g = GranularitySpec.per_tensor()
        assert g.mode == GranularityMode.PER_TENSOR
        assert g.bank_size == 16  # default, not used for PER_TENSOR

    def test_bank_outlier_ratio_stores(self):
        """BANK stores outlier_ratio for future sparse use."""
        g = GranularitySpec(mode=GranularityMode.BANK, outlier_ratio=0.1)
        assert g.outlier_ratio == 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_bank_granularity.py::TestBankGranularitySpec -v`
Expected: FAIL — `AttributeError: GranularityMode.BANK` not defined

**Step 3: Implement GranularityMode + GranularitySpec**

**Modify** `src/scheme/granularity.py`:

```python
class GranularityMode(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"
    BANK = "bank"                    # NEW
    DYNAMIC_GROUP = "dynamic_group"
```

Add fields to `GranularitySpec`:
```python
bank_size: int = 16        # NEW: elements per bank along bank_axis
bank_axis: int = -1        # NEW: axis to split into banks
```

Add post_init validations:
```python
# BANK validations
if self.mode == GranularityMode.BANK and self.bank_size <= 0:
    raise ValueError(
        f"BANK requires bank_size > 0, got {self.bank_size}"
    )
if self.mode == GranularityMode.BANK and self.block_size != 0:
    raise ValueError(
        f"BANK requires block_size=0, got {self.block_size}"
    )
if self.mode == GranularityMode.BANK and self.channel_axis != 0:
    raise ValueError(
        f"BANK requires channel_axis=0, got {self.channel_axis}"
    )
if self.mode not in (GranularityMode.BANK,) and self.bank_axis != -1:
    raise ValueError(
        f"{self.mode.name} requires bank_axis=-1, got {self.bank_axis}"
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest src/tests/test_bank_granularity.py::TestBankGranularitySpec -v`
Expected: 8 PASS

**Step 5: Commit**

```bash
git add src/scheme/granularity.py src/tests/test_bank_granularity.py
git commit -m "feat(granularity): add BANK mode to GranularityMode and GranularitySpec"
```

---

### Task 2: Write failing test — _quantize_per_bank bit-exact

**Files:**
- Modify: `src/tests/test_bank_granularity.py`

**Step 1: Write the failing test**

Add to `test_bank_granularity.py`:

```python
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.quantize.elemwise import quantize


class TestBankQuantizeBitExact:
    """Bit-exact verification of BANK quantization with int4 format."""

    @pytest.fixture
    def x(self):
        # 2x4 tensor: rows = [1,2,3,4], [5,6,7,8]
        return torch.tensor([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0]])

    @pytest.fixture
    def fmt(self):
        return FormatBase.from_str("int4")

    def test_bank_quantize_pot_bit_exact(self, x, fmt):
        """BANK int4 pot: hand-derived expected values."""
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)

        # Hand derivation (see verification doc):
        # Bank 0 (cols 0-1): amax=max(1,2,5,6)=6.0 → pot=8.0
        #   → [2.0, 2.0] row0, [6.0, 6.0] row1
        # Bank 1 (cols 2-3): amax=max(3,4,7,8)=8.0 → pot=8.0
        #   → [4.0, 4.0] row0, [8.0, 8.0] row1
        expected = torch.tensor([[2.0, 2.0, 4.0, 4.0],
                                 [6.0, 6.0, 8.0, 8.0]])
        assert torch.equal(result, expected), \
            f"BANK pot mismatch:\n got {result}\n expected {expected}"

    def test_bank_quantize_fp32_bit_exact(self, x, fmt):
        """BANK int4 fp32: hand-derived expected values."""
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="fp32")
        result = quantize(x, scheme)

        # Bank 0: amax=6.0 (no pot rounding)
        #   → [1.5, 1.5] row0, [4.5, 4.5] row1
        # Bank 1: amax=8.0 (no pot rounding)
        #   → [4.0, 4.0] row0, [8.0, 8.0] row1
        expected = torch.tensor([[1.5, 1.5, 4.0, 4.0],
                                 [4.5, 4.5, 8.0, 8.0]])
        assert torch.equal(result, expected), \
            f"BANK fp32 mismatch:\n got {result}\n expected {expected}"

    def test_bank_axis_0(self, fmt):
        """BANK with axis=0 splits along rows."""
        x = torch.tensor([[1.0, 2.0],
                          [5.0, 6.0],
                          [3.0, 4.0],
                          [7.0, 8.0]])  # 4x2
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=0)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)

        # Bank 0 (rows 0-1): amax=max(1,2,5,6)=6.0→pot=8.0
        # Bank 1 (rows 2-3): amax=max(3,4,7,8)=8.0→pot=8.0
        expected = torch.tensor([[2.0, 2.0],
                                 [6.0, 6.0],
                                 [4.0, 4.0],
                                 [8.0, 8.0]])
        assert torch.equal(result, expected), \
            f"BANK axis=0 mismatch:\n got {result}\n expected {expected}"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_bank_granularity.py::TestBankQuantizeBitExact -v`
Expected: FAIL — `ValueError: Unknown granularity mode: GranularityMode.BANK` (dispatch not implemented yet)

**Step 3: Implement `_quantize_per_bank()` + dispatch**

**Modify** `src/formats/base.py`:

Add `_quantize_per_bank` method after `_quantize_per_channel`:

```python
def _quantize_per_bank(self, x, granularity, round_mode, allow_denorm=True,
                       scale=None, scale_storage="pot"):
    """Per-bank quantization: split along bank_axis into banks.

    Each bank spans ALL elements across non-bank dimensions within its
    bank_axis segment — unlike PER_BLOCK which subdivides every dimension.
    One amax per bank. Supports fp32 and pot scale_storage.
    """
    if torch.jit.is_tracing():
        return x
    axis = granularity.bank_axis
    if axis < 0:
        axis = x.ndim + axis
    if not (0 <= axis < x.ndim):
        raise ValueError(
            f"bank_axis={granularity.bank_axis} out of range "
            f"for tensor with ndim={x.ndim}"
        )

    bank_size = granularity.bank_size
    N_along = x.shape[axis]
    if N_along % bank_size != 0:
        raise ValueError(
            f"Dimension {axis} size {N_along} not divisible "
            f"by bank_size {bank_size}"
        )

    num_banks = N_along // bank_size

    # Reshape: split axis into (num_banks, bank_size)
    new_shape = list(x.shape)
    new_shape[axis] = num_banks
    new_shape.insert(axis + 1, bank_size)
    x_r = x.reshape(new_shape)
    # x_r shape: (..., num_banks, bank_size, ...)
    # bank dim is at position `axis`, inner dim at `axis+1`

    if scale is not None:
        amax = scale
    else:
        # Reduce all dims EXCEPT the bank dim
        dims_to_reduce = [i for i in range(x_r.ndim) if i != axis]
        amax = torch.amax(torch.abs(x_r), dim=tuple(dims_to_reduce), keepdim=True)
        amax = amax.clamp(min=1e-12)

    if scale_storage == "pot":
        amax = 2 ** torch.round(torch.log2(amax))

    x_norm = x_r / amax
    x_q = self.quantize_elemwise(x_norm, round_mode=round_mode,
                                 allow_denorm=allow_denorm)
    x_q = x_q * amax
    return x_q.reshape(x.shape)
```

Add dispatch in `quantize()` method (after PER_BLOCK branch):

```python
elif mode == GranularityMode.BANK:
    return self._quantize_per_bank(x, granularity, round_mode,
                                    allow_denorm=allow_denorm,
                                    scale=scale, scale_storage=scale_storage)
```

**Step 4: Run test to verify it passes**

Run: `pytest src/tests/test_bank_granularity.py::TestBankQuantizeBitExact -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/formats/base.py src/tests/test_bank_granularity.py
git commit -m "feat(quantize): add _quantize_per_bank method and BANK dispatch"
```

---

### Task 3: Write failing test — BANK edge cases

**Files:**
- Modify: `src/tests/test_bank_granularity.py`

**Step 1: Write the failing edge case tests**

```python
class TestBankEdgeCases:
    """Shape preservation, boundary, and error cases for BANK quantization."""

    @pytest.mark.parametrize("shape,axis", [
        ((4, 8), -1),
        ((2, 3, 8), -1),
        ((2, 4, 6, 8), 1),
    ])
    def test_shape_preserved(self, shape, axis):
        """Output shape matches input for various ranks."""
        torch.manual_seed(42)
        x = torch.randn(*shape)
        fmt = FormatBase.from_str("int8")
        # Ensure dimension divisible by bank_size
        dim_size = shape[axis] if axis >= 0 else shape[axis]
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=dim_size // 2, bank_axis=axis)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        assert result.shape == x.shape

    def test_non_divisible_raises(self):
        """Dimension not divisible by bank_size raises."""
        x = torch.randn(4, 7)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=3, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        with pytest.raises(ValueError, match="not divisible"):
            quantize(x, scheme)

    def test_bank_axis_out_of_range_raises(self):
        """bank_axis out of range raises."""
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=5)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        with pytest.raises(ValueError, match="bank_axis"):
            quantize(x, scheme)

    def test_zero_input(self):
        """Zero input → zero output."""
        x = torch.zeros(4, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        assert (result == 0).all()

    def test_static_scale_smoke(self):
        """Static scale with BANK produces finite output."""
        x = torch.randn(2, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        # Pre-computed scale: 2 banks → shape (1, 2, 1) after reshape
        scale = torch.tensor([[[2.0], [3.0]]])  # broadcastable
        result = quantize(x, scheme, scale=scale)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_per_tensor_vs_bank_full_tensor(self):
        """BANK with bank_size covering entire axis = per_tensor behavior."""
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("int8")

        g_pt = GranularitySpec.per_tensor()
        scheme_pt = QuantScheme(format=fmt, granularity=g_pt, scale_storage="pot")
        result_pt = quantize(x, scheme_pt)

        g_bank = GranularitySpec(mode=GranularityMode.BANK, bank_size=8, bank_axis=-1)
        scheme_bank = QuantScheme(format=fmt, granularity=g_bank, scale_storage="pot")
        result_bank = quantize(x, scheme_bank)

        # One bank covering all 8 columns → same amax as per_tensor
        # But BANK also reduces dim 0, so amax is per_tensor.
        # Wait: with bank covering entire axis, num_banks=1, reduction over
        # dims 0 and 2 (all elements) → same as per_tensor amax.
        assert torch.equal(result_pt, result_bank), \
            "BANK covering full axis should match per_tensor"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_bank_granularity.py::TestBankEdgeCases -v`
Expected: Tests for error messages should pass if implementation already has those checks. The `test_bank_axis_out_of_range_raises` might need the axis range check added.

**Step 3: Fix any missing validation in `_quantize_per_bank`**

Ensure the axis range check is in place (already in the implementation from Task 2).

**Step 4: Run all BANK tests**

Run: `pytest src/tests/test_bank_granularity.py -v`
Expected: ~14 PASS

**Step 5: Commit**

```bash
git add src/tests/test_bank_granularity.py src/formats/base.py
git commit -m "test(bank): add edge case and static scale tests for BANK"
```

---

### Task 4: QuantConfig support for BANK

**Files:**
- Modify: `src/session/_config.py`
- Modify: `src/tests/test_bank_granularity.py`

**Step 1: Write failing QuantConfig test**

```python
class TestBankQuantConfig:
    """QuantConfig → OpQuantConfig resolution for BANK mode."""

    def test_quantconfig_bank_resolves(self):
        """QuantConfig with bank granularity produces BANK GranularitySpec."""
        from src.session._config import QuantConfig

        cfg = QuantConfig(
            w_format="int8",
            w_granularity="bank",
            w_block_size=16,
            w_axis=0,
        )
        op_cfg = cfg.to_op_config()
        g = op_cfg.weight.granularity
        assert g.mode == GranularityMode.BANK
        assert g.bank_size == 16
        assert g.bank_axis == 0

    def test_quantconfig_bank_requires_size(self):
        """BANK without block_size raises."""
        from src.session._config import QuantConfig
        with pytest.raises(ValueError, match="bank_size"):
            QuantConfig(w_granularity="bank")

    def test_quantconfig_bank_outlier_ratio(self):
        """BANK + outlier_ratio stores correctly for future sparse."""
        from src.session._config import QuantConfig
        cfg = QuantConfig(
            w_format="int8",
            w_granularity="bank",
            w_block_size=8,
            outlier_ratio=0.1,
        )
        op_cfg = cfg.to_op_config()
        assert op_cfg.weight.granularity.outlier_ratio == 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_bank_granularity.py::TestBankQuantConfig -v`
Expected: FAIL — `ValueError: Invalid w_granularity 'bank'` or similar

**Step 3: Implement QuantConfig BANK support**

**Modify** `src/session/_config.py`:

1. Add to `_VALID_GRANULARITIES`:
```python
_VALID_GRANULARITIES = frozenset({"per_tensor", "per_channel", "per_block", "bank"})
```

2. Add branch in `_resolve_granularity`:
```python
elif granularity == "bank":
    if block_size is None:
        raise ValueError("bank granularity requires bank_size (pass as block_size)")
    return GranularitySpec(
        mode=GranularityMode.BANK,
        bank_size=block_size,
        bank_axis=axis,
        outlier_ratio=outlier_ratio,
    )
```

3. Update `__post_init__` in QuantConfig: add BANK-specific validation:
```python
if self.w_granularity == "bank" and self.w_block_size is None:
    raise ValueError("w_block_size is required when w_granularity='bank' (used as bank_size)")
if self.a_granularity == "bank" and self.a_block_size is None:
    raise ValueError("a_block_size is required when a_granularity='bank' (used as bank_size)")
```

4. Update `from_descriptor` classmethod: `"bank"` granularity thread-through already works via `_resolve_granularity`.

**Step 4: Run QuantConfig tests**

Run: `pytest src/tests/test_bank_granularity.py::TestBankQuantConfig -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/session/_config.py src/tests/test_bank_granularity.py
git commit -m "feat(config): add bank granularity support to QuantConfig and _resolve_granularity"
```

---

### Task 5: Session integration test

**Files:**
- Modify: `src/tests/test_bank_granularity.py`

**Step 1: Write Session integration test**

```python
class TestBankSessionIntegration:
    """Session-level tests: QuantConfig with bank → qmodel end-to-end."""

    @pytest.fixture
    def simple_model(self):
        import torch.nn as nn

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.fc2(x)
                return x

        torch.manual_seed(42)
        return TinyMLP()

    def test_bank_smoke(self, simple_model):
        """QuantConfig bank mode produces valid output."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        cfg = QuantConfig(
            name="test-bank",
            w_format="int8",
            w_granularity="bank",
            w_block_size=4,
            w_axis=0,
            a_granularity="bank",
            a_block_size=2,
            a_axis=-1,
            weight_only=False,
        ).to_op_config()

        x = torch.randn(3, 4)
        qmodel, _, _ = run_quantization(simple_model, cfg, [x], keep_fp32=False)
        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (3, 2)
        assert torch.isfinite(out).all()
```

**Step 2: Run test**

Run: `pytest src/tests/test_bank_granularity.py::TestBankSessionIntegration -v`
Expected: 1 PASS (should work since BANK paths are already in place)

**Step 3: Commit**

```bash
git add src/tests/test_bank_granularity.py
git commit -m "test(bank): add Session integration test for BANK granularity"
```

---

### Task 6: Verification document + full test run

**Files:**
- Create: `docs/verification/020-bank-granularity.md`

**Step 1: Write verification document**

```markdown
# 020: BANK Granularity 量化正确性

**对应测试**: `test_bank_quantize_pot_bit_exact` / `test_bank_quantize_fp32_bit_exact`
**验证层级**: Layer 1（核心量化）

## 格式原理

BANK: 沿 bank_axis 将 tensor 切分为 bank_size 大小的连续组。每个 bank 覆盖该轴段内的所有元素（跨所有非 bank 维度），共享一个 amax。

- format: int4
- granularity: bank, bank_axis=-1, bank_size=2
- scale_storage: pot / fp32

## 给定数据

```python
x = [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]  # shape (2, 4)
```

bank_axis=-1, bank_size=2 → num_banks = 4/2 = 2

## 手工推导 (POT)

reshape: (2, 4) → (2, 2, 2)
```
Bank dim at axis=1, inner dim at axis=2

Bank 0 (cols 0-1): values [1,2,5,6]
  amax = 6.0 → pot = 8.0
  normalized: [1,2,5,6]/8 = [0.125, 0.25, 0.625, 0.75]

  int4 elemwise (mbits=4, max_norm=1.75):
    x_q_elem = [0.25, 0.25, 0.75, 0.75]
    x_q = x_q_elem * 8.0 = [2.0, 2.0, 6.0, 6.0]

Bank 1 (cols 2-3): values [3,4,7,8]
  amax = 8.0 → pot = 8.0
  normalized: [3,4,7,8]/8 = [0.375, 0.5, 0.875, 1.0]

  int4 elemwise:
    x_q_elem = [0.5, 0.5, 1.0, 1.0]
    x_q = x_q_elem * 8.0 = [4.0, 4.0, 8.0, 8.0]
```

**期望值 (POT): `[[2.0, 2.0, 4.0, 4.0], [6.0, 6.0, 8.0, 8.0]]`**

## 手工推导 (FP32)

```
Bank 0: amax=6.0 (no pot rounding)
  [1,2,5,6]/6 = [0.1667, 0.333, 0.833, 1.0]
  int4: [0.25, 0.25, 0.75, 1.0]
  *6 = [1.5, 1.5, 4.5, 6.0]

Bank 1: amax=8.0 (no pot rounding)
  Same as POT → [4.0, 4.0, 8.0, 8.0]
```

**期望值 (FP32): `[[1.5, 1.5, 4.0, 4.0], [4.5, 4.5, 8.0, 8.0]]`**

## BANK vs PER_BLOCK 对比

同输入用 PER_BLOCK(axis=-1, block_size=2, no sparse):
- PER_BLOCK: M×(N/2)=4 个 block，每个 block 内 2 个元素共享 shared_exp
- BANK: N/2=2 个 bank，每个 bank 覆盖 M×2=4 个元素共享 amax

BANK 的 scale 更粗粒度——同一个 bank 跨所有 M 行。

## 验证结果

- [ ] 运行日期: 
- [ ] 结果: 
- [ ] 实际输出与手工推导期望值完全一致（torch.equal）
```

**Step 2: Run full test suite**

Run: `pytest src/tests/test_bank_granularity.py -v`
Expected: ~17 PASS

Run: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow" --ignore=src/tests/test_bank_granularity.py`
Expected: 2,421 passed (no regressions from existing suite)

**Step 3: Commit**

```bash
git add docs/verification/020-bank-granularity.md
git commit -m "docs(verify): add BANK granularity verification document"
```

---

### Task 7: E2E regression

**Step 1: Run E2E regression**

```bash
PYTHONPATH=. python scripts/mnist_hadamard_study.py
PYTHONPATH=. python scripts/transformer_agnews_study.py
```

Expected: Both pass with existing thresholds (no BANK configs in these scripts, regression-only check).

**Step 2: Final commit if E2E passes**

```bash
git commit --allow-empty -m "chore: E2E regression verified after BANK granularity"
```

---

## Summary

| Task | Files | Tests |
|------|-------|-------|
| 1. GranularitySpec + Mode | `granularity.py` | 8 construction tests |
| 2. _quantize_per_bank | `base.py` | 3 bit-exact tests |
| 3. Edge cases | `base.py` | 6 edge/shape/error tests |
| 4. QuantConfig | `_config.py` | 3 config resolution tests |
| 5. Session integration | (tests only) | 1 smoke test |
| 6. Verification doc | `docs/verification/` | — |
| 7. E2E regression | — | MNIST + Transformer |

**Total: ~150 lines implementation, ~21 tests, 1 verification doc**
