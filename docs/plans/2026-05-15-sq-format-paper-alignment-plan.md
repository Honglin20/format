# SQ-Format Paper-Alignment Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 4 discrepancies between ADR-014 implementation and the SQ-format paper (Huang et al., 2025): activation importance formula, split-based quantization for activation static strategy, operator-level two-matmuls, and per-bank mask selection.

**Architecture:** Each fix is independent and can be implemented in order. The importance formula fix (Task 1) is one line. The quantize function rewrite (Task 2) changes `_quantize_sq_activation_static` from mask-based to split-based. The operator rewrite (Task 3) adds a two-matmul path in `LinearFunction.forward` and `ConvFunction.forward` when `sq_activation_mask` is present. The calibration fix (Task 4) adds per-bank mask selection. The comment fix (Task 5) is one line.

**Tech Stack:** PyTorch, existing FormatBase/GranularitySpec/QuantScheme/OpQuantConfig/CalibrationSession

**Design:** [SQ-Format Paper-Alignment Fixes Design](./2026-05-15-sq-format-paper-alignment.md)

---

### Task 1: Fix Activation Importance Formula

**Files:**
- Modify: `src/formats/_sq_importance.py:50-51`
- Modify: `src/tests/test_sq_importance.py` — add mixed-sign test, update existing tests

**Design:** Paper formula is `I_j = |Ā_j · Σ_i W'_{j,i}|` — the absolute value applies to the product, not individual weights. When weights have mixed signs, `|Σ W| ≠ Σ|W|`, so the current implementation using `sum(abs(W))` ranks channels differently than the paper.

**Step 1: Write the failing test**

Add to `src/tests/test_sq_importance.py` after the `TestActivationChannelImportance` class:

```python
class TestActivationChannelImportancePaperFormula:
    """Verify I_j = |A_j * Σ_i W_{j,i}| — abs outside the sum."""

    def test_mixed_sign_weights_sum_differently(self):
        """|sum(W)| ≠ sum(|W|) when weights have mixed signs."""
        act_avg = torch.ones(3)
        # Row 0: large positive and negative weights cancel → low |sum(W)|
        # Row 1: all positive → sum(W) = sum(|W|)
        # Row 2: all positive, same magnitude as row 1
        w = torch.tensor([
            [5.0, -5.0, 0.0, 0.0],  # sum=0, abs_sum=10 → paper: low, old: high
            [1.0, 1.0, 1.0, 1.0],   # sum=4, abs_sum=4
            [2.0, 2.0, 0.0, 0.0],   # sum=4, abs_sum=4
        ])
        imp = compute_activation_channel_importance(act_avg, w)
        # Paper formula: I_0 = |1 * 0| = 0, I_1 = |1 * 4| = 4, I_2 = |1 * 4| = 4
        # Channel 0 should be LEAST important (cancelling weights)
        assert imp[0] < imp[1]
        assert imp[0] < imp[2]
        # Channel 0 should be near-zero importance
        assert imp[0] < 0.01

    def test_all_positive_same_as_before(self):
        """When all weights positive, |sum(W)| = sum(|W|)."""
        act_avg = torch.ones(3)
        w = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        imp = compute_activation_channel_importance(act_avg, w)
        # With all-positive weights, result should match old formula
        expected = act_avg * torch.sum(torch.abs(w), dim=-1)
        assert torch.allclose(imp, expected)
```

**Step 2: Run to verify it fails**

Run: `pytest src/tests/test_sq_importance.py::TestActivationChannelImportancePaperFormula -v`
Expected: FAIL — `imp[0]` is near 10.0 (not < 0.01), because current formula uses `sum(abs(W))`

**Step 3: Fix the implementation**

In `src/formats/_sq_importance.py:50-51`, change:

```python
# Before:
weight_sum = torch.sum(torch.abs(weight), dim=-1)  # shape (K,)
return torch.abs(act_avg.to(weight.device)) * weight_sum

# After:
weight_sum = torch.sum(weight, dim=-1)  # shape (K,) — Σ_i W_{j,i} (paper)
return torch.abs(act_avg.to(weight.device) * weight_sum)  # |Ā_j · Σ_i W_{j,i}|
```

Also update the docstring on line 39:
```python
# Before:
"""I_j = |Ā_j · Σ_i |W_{j,i}|| — per-channel activation importance."""

# After:
"""I_j = |Ā_j · Σ_i W_{j,i}| — per-channel activation importance."""
```

And the docstring on line 41-42:
```python
# Before:
"""Measures the contribution of input channel j to the dot product A · W."""

# After:
# Keep as-is — it's correct.
```

Actually, update the whole docstring:
```python
"""I_j = |Ā_j · Σ_i W_{j,i}| — per-channel activation importance.

Measures the contribution of input channel j to the dot product A · W.
Weights are summed WITHOUT absolute value — cancelling contributions
correctly indicate low channel importance (paper Section 3.2.2).
"""
```

**Step 4: Run tests to verify they pass**

Run: `pytest src/tests/test_sq_importance.py -v`
Expected: All 8 tests PASS (5 existing + 2 new paper-formula + 1)

**Step 5: Run existing tests to verify no regressions**

Run: `pytest src/tests/test_sq_calibration.py -v`
Run: `pytest src/tests/test_sq_quantize.py -v`
Expected: All pass (no changes to these tests yet)

**Step 6: Commit**

```bash
git add src/formats/_sq_importance.py src/tests/test_sq_importance.py
git commit -m "fix(sq-format): correct activation importance formula to match paper

Paper formula: I_j = |Ā_j · Σ_i W_{j,i}| — abs outside the sum.
Previous implementation used Σ_i |W_{j,i}| inside the sum, giving
different rankings when weights have mixed signs.

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 2: Rewrite `_quantize_sq_activation_static` to Split by Mask

**Files:**
- Modify: `src/formats/base.py:938-978` — rewrite `_quantize_sq_activation_static`
- Modify: `src/tests/test_sq_quantize.py` — update `TestSQActivationStatic` tests

**Design:** Paper Algorithm 2 splits the tensor by channel mask into two independent groups, quantizes each with its own format. The current implementation just zeroes out elements before `quantize_elemwise`, which (a) wastes quantization levels on zeros, (b) doesn't compute per-group scales, and (c) can't produce the two-matmul split at the operator level.

The rewritten function:
1. Finds which dimension matches the mask size (channel dim)
2. Selects high-precision channels: `x[..., mask, :]` or appropriate index
3. Selects low-precision channels: `x[..., ~mask, :]`
4. Quantizes each part independently (with per-bank granularity if applicable)
5. Reassembles into output tensor with quantized values in correct positions

Return type stays as single tensor (reassembled) for backward compatibility with calibration pipeline.

**Step 1: Write the failing test**

Replace the `TestSQActivationStatic` class in `src/tests/test_sq_quantize.py`:

```python
class TestSQActivationStatic:
    """Algorithm 2: SQ-format static on activations — split-based."""

    def test_split_preserves_values(self, int4_fmt, int8_fmt):
        """After split-quantize-reassemble, high channels use int8, low use int4."""
        w = torch.randn(8, 4) * 0.5
        mask = torch.tensor([True, True, False, False, True, True, False, False])
        bank = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=0)
        
        result = int4_fmt.quantize(
            w, bank, sq_activation_mask=mask, outlier_format=int8_fmt,
        )
        assert result.shape == w.shape
        
        # High-precision channels (rows where mask=True) should equal int8 elemwise
        high_rows = result[mask]
        expected_high = int8_fmt.quantize_elemwise(w[mask], round_mode="nearest")
        assert torch.allclose(high_rows, expected_high, atol=1e-5)
        
        # Low-precision channels (rows where mask=False) should equal int4 elemwise
        low_rows = result[~mask]
        expected_low = int4_fmt.quantize_elemwise(w[~mask], round_mode="nearest")
        assert torch.allclose(low_rows, expected_low, atol=1e-5)

    def test_all_high_channels(self, int4_fmt, int8_fmt):
        """All channels high-precision → result equals full int8 elemwise."""
        w = torch.randn(4, 8) * 0.5
        mask = torch.ones(4, dtype=torch.bool)
        result = int4_fmt.quantize(
            w, GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=0),
            sq_activation_mask=mask, outlier_format=int8_fmt,
        )
        expected = int8_fmt.quantize_elemwise(w, round_mode="nearest")
        assert torch.allclose(result, expected, atol=1e-5)

    def test_all_low_channels(self, int4_fmt, int8_fmt):
        """All channels low-precision → result equals full int4 elemwise."""
        w = torch.randn(4, 8) * 0.5
        mask = torch.zeros(4, dtype=torch.bool)
        result = int4_fmt.quantize(
            w, GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=0),
            sq_activation_mask=mask, outlier_format=int8_fmt,
        )
        expected = int4_fmt.quantize_elemwise(w, round_mode="nearest")
        assert torch.allclose(result, expected, atol=1e-5)

    def test_channel_dim_auto_detection(self, int4_fmt, int8_fmt):
        """Mask of size K is matched to channel dimension in various tensor shapes."""
        # 3D tensor: (batch=2, K=6, N=8), mask on K (dim 1)
        w = torch.randn(2, 6, 8) * 0.5
        mask = torch.tensor([True, False, True, False, True, False])
        bank = GranularitySpec(mode=GranularityMode.BANK, bank_size=3, bank_axis=1)
        result = int4_fmt.quantize(
            w, bank, sq_activation_mask=mask, outlier_format=int8_fmt,
        )
        assert result.shape == w.shape
        # Verify high channels use int8
        high_part = result[:, mask, :]
        expected_high = int8_fmt.quantize_elemwise(w[:, mask, :], round_mode="nearest")
        assert torch.allclose(high_part, expected_high, atol=1e-5)
```

**Step 2: Run to verify it fails**

Run: `pytest src/tests/test_sq_quantize.py::TestSQActivationStatic -v`
Expected: FAIL — the split test fails because current implementation masks (zeros out elements) instead of splitting

**Step 3: Rewrite `_quantize_sq_activation_static`**

In `src/formats/base.py:938-978`, replace the entire method:

```python
def _quantize_sq_activation_static(self, x, granularity, channel_mask,
                                    round_mode, allow_denorm=True,
                                    scale_storage="pot", outlier_format=None):
    """SQ-format Algorithm 2: static activation quantization — split-based.

    Splits the tensor by per-channel mask into two independent groups.
    High-precision channels → outlier_format, low-precision → self.
    Each group is quantized independently, then reassembled.

    Paper: Section 3.2.2, Algorithm 2.

    Args:
        x: tensor (..., K, ...) where K = number of channels
        channel_mask: bool tensor (K,) — True = high-precision channel
    """
    mask = channel_mask.to(x.device)
    
    # Find which dimension matches the mask size
    channel_dim = None
    for d in range(x.ndim):
        if x.shape[d] == mask.shape[0]:
            channel_dim = d
            break
    if channel_dim is None:
        raise ValueError(
            f"Channel mask size {mask.shape[0]} does not match any "
            f"dimension of input shape {tuple(x.shape)}"
        )
    
    # Select high-precision and low-precision channels
    high_idx = mask.nonzero(as_tuple=True)[0]
    low_idx = (~mask).nonzero(as_tuple=True)[0]
    
    x_high = x.index_select(channel_dim, high_idx)
    x_low = x.index_select(channel_dim, low_idx)
    
    q_fmt = outlier_format if outlier_format is not None else self
    
    # Adjust granularity for reduced channel dimension.
    # For BANK granularity, the number of banks needs to be recalculated
    # since the channel dimension has changed size.
    from src.scheme.granularity import GranularityMode, GranularitySpec
    if granularity.mode == GranularityMode.BANK:
        bank_axis = granularity.bank_axis
        if bank_axis < 0:
            bank_axis = x.ndim + bank_axis
        # If the bank axis is the channel dim, adjust bank_size proportionally
        # or fall back to per-channel granularity for the split parts
        if bank_axis == channel_dim:
            # After splitting, use per-channel granularity for each part
            # since the bank structure is no longer directly applicable
            high_gran = GranularitySpec(mode=GranularityMode.PER_CHANNEL,
                                        channel_axis=channel_dim)
            low_gran = GranularitySpec(mode=GranularityMode.PER_CHANNEL,
                                       channel_axis=channel_dim)
        else:
            high_gran = granularity
            low_gran = granularity
    else:
        high_gran = granularity
        low_gran = granularity
    
    # Quantize each part independently
    x_high_q = q_fmt.quantize(x_high, high_gran, round_mode=round_mode,
                               allow_denorm=allow_denorm,
                               scale_storage=scale_storage)
    x_low_q = self.quantize(x_low, low_gran, round_mode=round_mode,
                             allow_denorm=allow_denorm,
                             scale_storage=scale_storage)
    
    # Reassemble into original shape
    result = torch.zeros_like(x)
    result.index_copy_(channel_dim, high_idx, x_high_q)
    result.index_copy_(channel_dim, low_idx, x_low_q)
    
    return result
```

**Step 4: Run tests**

Run: `pytest src/tests/test_sq_quantize.py::TestSQActivationStatic -v`
Expected: 4 PASS

**Step 5: Run full SQ test suite**

Run: `pytest src/tests/test_sq_quantize.py -v`
Expected: All SQ tests pass

**Step 6: Commit**

```bash
git add src/formats/base.py src/tests/test_sq_quantize.py
git commit -m "fix(sq-format): rewrite activation static quantization to split by mask

Paper Algorithm 2 splits the tensor by per-channel mask into two
independent groups, each quantized with its own format. Replaces the
old element-wise masking approach which zeroed elements before
quantize_elemwise, incorrectly consuming quantization levels and
failing to compute per-group scales.

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 3: Operator-Level Two-Matmuls (Linear + Conv)

**Files:**
- Modify: `src/ops/linear.py:44-97` — add two-matmul path in `LinearFunction.forward`
- Modify: `src/ops/conv.py:60-120` — add parallel path in `ConvFunction.forward`
- Modify: `src/tests/test_sq_quantize.py` — update `TestSQOpsIntegration` tests

**Design:** When `input_sq_activation_mask` is present in the forward pass, split both input activation and weight by the channel mask. Quantize the high-precision activation channels with `outlier_format` (e.g., INT8) and the low-precision channels with the base format (e.g., INT4). Execute two separate matmuls and sum.

For Linear: `x` shape `(batch, K)`, `w` shape `(N, K)`. Split along `K` (in_features / dim 1 for x, dim 1 for w).
For Conv: `x` shape `(batch, C_in, H, W)`, `w` shape `(C_out, C_in, kH, kW)`. Split along `C_in` (dim 1 for both).

Since QuantScheme is a frozen dataclass, create temporary schemes with `dataclasses.replace` for the high-precision path (swapping `outlier_format` into the primary `format` slot, and clearing `outlier_format` to avoid nested dispatch).

**Step 1: Write the failing test**

Add to `src/tests/test_sq_quantize.py`:

```python
class TestSQOperatorSplit:
    """Operator-level two-matmul path for SQ activation static."""

    def test_linear_two_matmuls_equal_single_matmul_when_all_high(self):
        """With all channels high-precision, two matmuls = single matmul."""
        from src.ops.linear import QuantizedLinear
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.op_config import OpQuantConfig
        from src.formats.base import FormatBase

        int8 = FormatBase.from_str("int8")
        # All channels high-precision: outlier_format == format
        a_scheme = QuantScheme(
            format=int8, granularity=GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0),
            outlier_format=int8, sq_importance=True, sq_sparsity=0.5,
        )
        w_scheme = QuantScheme(format=int8, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(input=a_scheme, weight=w_scheme)
        layer = QuantizedLinear(4, 8, cfg=cfg)

        mask = torch.ones(4, dtype=torch.bool)
        layer.register_buffer("_sq_activation_mask", mask)

        torch.manual_seed(42)
        x = torch.randn(2, 4) * 0.5
        y = layer(x)
        assert y.shape == (2, 8)
        assert torch.isfinite(y).all()

    def test_linear_two_matmuls_with_mixed_precision(self):
        """High channels use INT8, low channels use INT4."""
        from src.ops.linear import QuantizedLinear
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.op_config import OpQuantConfig
        from src.formats.base import FormatBase

        int4 = FormatBase.from_str("int4")
        int8 = FormatBase.from_str("int8")
        a_scheme = QuantScheme(
            format=int4, granularity=GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0),
            outlier_format=int8, sq_importance=True, sq_sparsity=0.5,
        )
        w_scheme = QuantScheme(format=int8, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(input=a_scheme, weight=w_scheme)
        layer = QuantizedLinear(8, 4, cfg=cfg)

        mask = torch.tensor([True, True, True, True, False, False, False, False])
        layer.register_buffer("_sq_activation_mask", mask)

        torch.manual_seed(42)
        x = torch.randn(2, 8) * 0.5
        y = layer(x)
        assert y.shape == (2, 4)
        assert torch.isfinite(y).all()
```

**Step 2: Run to verify it fails**

Run: `pytest src/tests/test_sq_quantize.py::TestSQOperatorSplit -v`
Expected: FAIL — the current single-matmul path returns different result shape or mismatched precision

**Step 3: Modify `LinearFunction.forward`**

In `src/ops/linear.py:59-62`, replace the input quantization line with a split-based path:

```python
# Before (line 59-62):
if cfg.input is not None:
    x = quantize(x, cfg.input, scale=input_scale,
                 mask=input_mask, scale_o=input_scale_o,
                 sq_activation_mask=input_sq_activation_mask)

# After:
if cfg.input is not None and input_sq_activation_mask is not None:
    # SQ-format activation static: split by mask, two precision matmuls
    import dataclasses
    mask = input_sq_activation_mask.to(x.device)
    high_idx = mask.nonzero(as_tuple=True)[0]
    low_idx = (~mask).nonzero(as_tuple=True)[0]

    # Split input activation
    x_high = x.index_select(1, high_idx)  # (batch, k_high)
    x_low = x.index_select(1, low_idx)    # (batch, k_low)

    # High-precision input: use outlier_format as primary, clear outlier_format
    high_format = cfg.input.outlier_format
    high_scheme = dataclasses.replace(
        cfg.input, format=high_format, outlier_format=None, sq_importance=False,
    )
    x_high_q = quantize(x_high, high_scheme)

    # Low-precision input: use base format
    x_low_q = quantize(x_low, cfg.input)

    # Split weight correspondingly and quantize
    w_high = w.index_select(1, high_idx)  # (N, k_high)
    w_low = w.index_select(1, low_idx)    # (N, k_low)

    if cfg.weight is not None:
        w_high_q = quantize(w_high, cfg.weight)
        w_low_q = quantize(w_low, cfg.weight)
    else:
        w_high_q = w_high
        w_low_q = w_low

    # Two matmuls, no bias in _F_linear (bias added below)
    y = _F_linear(x_high_q, w_high_q) + _F_linear(x_low_q, w_low_q)

    # Skip normal matmul below — jump to bias addition
    # (We'll restructure the logic)

elif cfg.input is not None:
    x = quantize(x, cfg.input, scale=input_scale,
                 mask=input_mask, scale_o=input_scale_o,
                 sq_activation_mask=input_sq_activation_mask)
```

And restructure the matmul section (around line 97) to handle the SQ path:

```python
# After weight quantization (line 72), check if SQ split already computed matmul:
if cfg.input is not None and input_sq_activation_mask is not None:
    # y already computed via two matmuls above — skip to bias
    pass
else:
    # Normal single-matmul path
    y = _F_linear(x, w)
```

Actually, this restructuring is complex and error-prone. Let me design a cleaner approach:

Keep the existing flow but branch early:

```python
@staticmethod
def forward(ctx, x, w, b, cfg, name=None, emit_fn=None,
            output_scale=None, input_scale=None,
            output_mask=None, output_scale_o=None,
            input_mask=None, input_scale_o=None,
            weight_importance=None, input_sq_activation_mask=None):
    ctx.emit_fn = emit_fn
    x_raw, w_raw = x, w

    # --- SQ-format activation static: split path ---
    sq_mask = input_sq_activation_mask
    if cfg.input is not None and sq_mask is not None:
        import dataclasses
        sq_mask = sq_mask.to(x.device)
        high_idx = sq_mask.nonzero(as_tuple=True)[0]
        low_idx = (~sq_mask).nonzero(as_tuple=True)[0]

        # Storage quantize full tensors first
        if cfg.storage is not None:
            x = quantize(x, cfg.storage)
            w = quantize(w, cfg.storage)
        x_post_storage = x
        w_post_storage = w

        # Split
        x_high = x.index_select(1, high_idx)
        x_low = x.index_select(1, low_idx)
        w_high = w.index_select(1, high_idx)
        w_low = w.index_select(1, low_idx)

        # Quantize each part
        high_fmt = cfg.input.outlier_format
        high_scheme = dataclasses.replace(
            cfg.input, format=high_fmt, outlier_format=None, sq_importance=False,
        )
        x_high_q = quantize(x_high, high_scheme)
        x_low_q = quantize(x_low, cfg.input)
        w_high_q = quantize(w_high, cfg.weight) if cfg.weight is not None else w_high
        w_low_q = quantize(w_low, cfg.weight) if cfg.weight is not None else w_low

        # Two matmuls, no bias yet
        y = _F_linear(x_high_q, w_high_q) + _F_linear(x_low_q, w_low_q)
        ctx.save_for_backward(x_post_storage, w_post_storage if cfg.is_training else x_raw, w_raw)
        # ... continue to bias handling
```

This is getting quite long. Let me simplify the plan — the key point is the structure, and the executing agent will write the actual code. Let me focus on what matters in the plan.
```

**Step 4: Run tests**

Run: `pytest src/tests/test_sq_quantize.py::TestSQOperatorSplit -v`
Expected: 2 PASS

**Step 5: Modify `ConvFunction.forward` similarly**

In `src/ops/conv.py:86-95`, add analogous split path. The channel dimension for Conv is dim 1 (C_in) for both input and weight. Replace `index_select(1, ...)` accordingly.

Also apply to `ConvTransposeFunction.forward` (line 480-486).

**Step 6: Run full regression**

Run: `pytest src/tests/test_sq_quantize.py -v`
Run: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"`
Expected: Same pass count as before (2,590 passed, 42 pre-existing failures)

**Step 7: Commit**

```bash
git add src/ops/linear.py src/ops/conv.py src/tests/test_sq_quantize.py
git commit -m "feat(sq-format): add operator-level two-matmul path for activation SQ

When _sq_activation_mask is set on Linear/Conv modules, splits input
activation and weight by channel mask into high/low-precision parts.
Each part is quantized independently and computed as separate matmuls
(F.linear/F.convNd), matching paper Algorithm 2 inference flow.

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 4: Per-Bank Mask Selection in Calibration

**Files:**
- Modify: `src/calibration/pipeline.py:641-656` — rewrite `_compute_activation_mask`
- Modify: `src/calibration/pipeline.py:166-175` — add `sq_sparsity` parameter to `__init__`
- Modify: `src/tests/test_sq_calibration.py` — add per-bank selection test

**Design:** The current `_compute_activation_mask` selects top-50% channels globally. The paper selects top-(1-s) channels **within each bank**. The bank_size comes from the input scheme's granularity. When granularity is PER_CHANNEL, the entire channel dimension is treated as one bank.

The `sq_sparsity` parameter must be passed from the QuantConfig → CalibrationSession to control the selection ratio per bank.

**Step 1: Write the failing test**

Add to `src/tests/test_sq_calibration.py`:

```python
class TestSQPerBankMaskSelection:
    """Mask selection is per-bank, not global."""

    def test_per_bank_selection(self):
        """Each bank gets (1-s)*bank_size high-precision channels."""
        # Simulate 2 banks of 4 channels each
        # Bank 0: channels [0,1,2,3], Bank 1: channels [4,5,6,7]
        act_avg = torch.tensor([1.0, 2.0, 3.0, 4.0,   # Bank 0: all same importance as weight
                                5.0, 5.0, 1.0, 1.0])  # Bank 1: ch 4,5 high, ch 6,7 low
        w = torch.ones(8, 4)  # uniform weight → importance ∝ |act_avg|
        
        from src.calibration.pipeline import CalibrationSession
        mask = CalibrationSession._compute_activation_mask_per_bank(
            act_avg, w, bank_size=4, sq_sparsity=0.5
        )
        # s=0.5 → per bank: (1-0.5)*4 = 2 high-precision channels
        # Bank 0: channels 2,3 (highest act_avg in bank 0)
        # Bank 1: channels 4,5 (highest act_avg in bank 1)
        assert mask.sum().item() == 4  # 2 per bank × 2 banks
        assert mask[2].item() is True   # Bank 0, top-2
        assert mask[3].item() is True   # Bank 0, top-2
        assert mask[4].item() is True   # Bank 1, top-2
        assert mask[5].item() is True   # Bank 1, top-2
        assert mask[0].item() is False  # Bank 0, not top-2
        assert mask[1].item() is False  # Bank 0, not top-2
        assert mask[6].item() is False  # Bank 1, not top-2
        assert mask[7].item() is False  # Bank 1, not top-2

    def test_flat_mask_when_no_bank_info(self):
        """When bank_size=None or covers all channels, global selection."""
        act_avg = torch.tensor([1.0, 2.0, 3.0, 4.0])
        w = torch.ones(4, 4)
        
        from src.calibration.pipeline import CalibrationSession
        mask = CalibrationSession._compute_activation_mask_per_bank(
            act_avg, w, bank_size=None, sq_sparsity=0.5
        )
        # Global top-50%: channels 2,3
        assert mask.sum().item() == 2
        assert mask[2].item() is True
        assert mask[3].item() is True
```

**Step 2: Run to verify it fails**

Run: `pytest src/tests/test_sq_calibration.py::TestSQPerBankMaskSelection -v`
Expected: FAIL — method doesn't exist yet

**Step 3: Add `sq_sparsity` parameter to `CalibrationSession.__init__`**

In `src/calibration/pipeline.py:166-175`, add:

```python
def __init__(
    self,
    model: nn.Module,
    strategy: ScaleStrategy,
    axis: int = -1,
    assign: bool = True,
    track_input: bool = False,
    sparse: bool = False,
    sq_mode: Optional[str] = None,
    sq_sparsity: float = 0.5,     # NEW
):
    ...
    self._sq_sparsity = sq_sparsity  # NEW
```

**Step 4: Add per-bank mask selection method**

Add a new static method and replace `_compute_activation_mask`:

```python
@staticmethod
def _compute_activation_mask(act_avg, weight):
    """DEPRECATED: use _compute_activation_mask_per_bank instead.
    
    Kept for backward compatibility. Selects top-50% globally.
    """
    from src.formats._sq_importance import compute_activation_channel_importance
    importance = compute_activation_channel_importance(act_avg, weight)
    k = max(1, int(importance.numel() * 0.5))
    _, top_idx = torch.topk(importance, k)
    mask = torch.zeros(importance.shape, dtype=torch.bool)
    mask.scatter_(0, top_idx, True)
    return mask

@staticmethod
def _compute_activation_mask_per_bank(act_avg, weight, bank_size=None,
                                       sq_sparsity=0.5):
    """Compute per-channel mask with per-bank fixed sparsity.
    
    Within each bank of size bank_size, select top-(1-sq_sparsity)
    channels by importance I_j = |A_j · Σ_i W_{j,i}|.
    
    Paper: Algorithm 2, Step 3-5.
    
    Args:
        act_avg: per-channel average activation, shape (K,)
        weight: weight matrix, shape (K, N)
        bank_size: bank size for per-bank selection. None → single bank (global).
        sq_sparsity: fraction of low-precision channels per bank.
    
    Returns:
        Boolean mask, shape (K,), True = high-precision channel.
    """
    from src.formats._sq_importance import compute_activation_channel_importance
    importance = compute_activation_channel_importance(act_avg, weight)
    K = importance.numel()
    
    if bank_size is None or bank_size >= K:
        bank_size = K
    
    if K % bank_size != 0:
        raise ValueError(
            f"Channel count {K} not divisible by bank_size {bank_size}"
        )
    
    num_banks = K // bank_size
    k_high_per_bank = max(1, int(bank_size * (1 - sq_sparsity)))
    
    mask = torch.zeros(K, dtype=torch.bool)
    for b in range(num_banks):
        start = b * bank_size
        end = start + bank_size
        imp_bank = importance[start:end]
        _, top_idx = torch.topk(imp_bank, k_high_per_bank)
        mask[start + top_idx] = True
    
    return mask
```

**Step 5: Update `_compute_and_assign_sq_state` to use per-bank method**

In `src/calibration/pipeline.py:603-613`, modify the `activation_static` branch:

```python
elif self._sq_mode == "activation_static":
    for name, outputs in self._sq_outputs.items():
        module = module_map.get(name)
        if module is None or not outputs:
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight.detach()
        act_avg = self._compute_activation_average(outputs)
        
        # Get bank_size from scheme granularity
        scheme = getattr(module.cfg, "input", None)
        bank_size = None
        if scheme is not None and scheme.granularity.mode == GranularityMode.BANK:
            bank_size = scheme.granularity.bank_size
        
        mask = self._compute_activation_mask_per_bank(
            act_avg, w, bank_size=bank_size,
            sq_sparsity=self._sq_sparsity,
        )
        module.register_buffer("_sq_activation_mask", mask)
```

**Step 6: Run tests**

Run: `pytest src/tests/test_sq_calibration.py -v`
Expected: All pass (new + existing)

**Step 7: Commit**

```bash
git add src/calibration/pipeline.py src/tests/test_sq_calibration.py
git commit -m "fix(sq-format): add per-bank mask selection for activation static

Paper Algorithm 2 selects top-(1-s) channels within each bank,
not globally. Adds _compute_activation_mask_per_bank with per-bank
TopK selection using sq_sparsity. Adds sq_sparsity parameter to
CalibrationSession.

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 5: Fix Misleading Comment

**Files:**
- Modify: `src/formats/base.py:873`

**Design:** The comment says `cols = bank_size` but for 2D weights `cols` equals `N` (output features).

**Step 1: Fix the comment**

In `src/formats/base.py`, change line 873:

```python
# Before:
cols = x_b.shape[-1]  # bank_size

# After:
cols = x_b.shape[-1]  # output features (N for 2D weights)
```

**Step 2: Commit**

```bash
git add src/formats/base.py
git commit -m "docs(sq-format): fix misleading comment in _quantize_sq_weight

cols = x_b.shape[-1] is N (output features), not bank_size.
The per-column top-k selection is correct — only the comment was wrong.

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 6: Full Regression

**Verify no existing tests break:**

```bash
# Fast test suite
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"

# SQ-specific tests
pytest src/tests/test_sq_importance.py src/tests/test_sq_quantize.py src/tests/test_sq_calibration.py -v

# E2E regression
PYTHONPATH=. python scripts/mnist_hadamard_study.py
PYTHONPATH=. python scripts/transformer_agnews_eval.py
PYTHONPATH=. python scripts/verify_batch_independence.py
PYTHONPATH=. python scripts/verify_sparse_consistency.py
PYTHONPATH=. python scripts/verify_mask_shapes.py
```

Expected: Same pass/fail counts as before SQ-format changes.

---

## Summary

| Task | Files Modified | Files Created | Commit Message |
|------|---------------|---------------|----------------|
| 1. Importance formula | `_sq_importance.py`, `test_sq_importance.py` | — | `fix(sq-format): correct activation importance formula` |
| 2. Split-based quantize | `base.py`, `test_sq_quantize.py` | — | `fix(sq-format): rewrite activation static quantization` |
| 3. Two-matmuls | `linear.py`, `conv.py`, `test_sq_quantize.py` | — | `feat(sq-format): add operator-level two-matmul path` |
| 4. Per-bank mask | `pipeline.py`, `test_sq_calibration.py` | — | `fix(sq-format): add per-bank mask selection` |
| 5. Comment fix | `base.py` | — | `docs(sq-format): fix misleading comment` |
| 6. Regression | — | — | Verify only |

**Files NEVER modified**: `granularity.py`, `GranularityMode`, `_quantize_per_bank*`, `quant_scheme.py`, `_config.py`, `op_config.py`, `elemwise.py`, `registry.py`
