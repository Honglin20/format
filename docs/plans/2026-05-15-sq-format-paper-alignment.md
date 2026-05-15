# SQ-Format Paper Alignment Fixes

**Date**: 2026-05-15
**Source**: [SQ-format: A Unified Sparse-Quantized Hardware-friendly Data Format for LLMs](https://arxiv.org/abs/2512.05409) (Huang et al., 2025)
**Related**: ADR-014, ADR-012

## Problem

ADR-014 and its implementation contain 4 discrepancies from the paper:

1. **Activation importance formula**: `compute_activation_channel_importance` uses `sum(|W|)` inside the product instead of `|sum(W)|` — different ranking when weights have mixed signs
2. **`_quantize_sq_activation_static` uses element-wise mask**: Paper Algorithm 2 splits the tensor by channel mask into two independent groups; current code zeroes out elements before `quantize_elemwise`
3. **No per-bank mask selection in quantization**: Paper selects mask per-bank during calibration; quantize function applies the (already per-bank) mask via channel split
4. **Single matmul instead of two**: Paper infers with two parallel matmuls (`W_high @ A_high + W_low @ A_low`); current code does one matmul with pre-masked tensors
5. **Misleading comment** in `_quantize_sq_weight`: `cols = x_b.shape[-1]  # bank_size` actually equals output features for 2D weights

## Design

### Fix 1: Activation Importance Formula

**File**: `src/formats/_sq_importance.py`

```python
# Before: I_j = |A_j| * sum_i |W_{j,i}|
weight_sum = torch.sum(torch.abs(weight), dim=-1)
return torch.abs(act_avg.to(weight.device)) * weight_sum

# After: I_j = |A_j * sum_i W_{j,i}|
weight_sum = torch.sum(weight, dim=-1)
return torch.abs(act_avg.to(weight.device) * weight_sum)
```

Add test with mixed-sign weights to verify `|sum(W)| != sum(|W|)`.

### Fix 2: Split-based Activation Static Quantization

**File**: `src/formats/base.py` — rewrite `_quantize_sq_activation_static`

Instead of masking elements before quantize_elemwise, SPLIT the tensor by channel mask into two independent groups:

```python
def _quantize_sq_activation_static(self, x, granularity, channel_mask, ...):
    # Split by channel mask — True = high precision
    x_high = x[..., channel_mask, :]   # select high-precision channels
    x_low  = x[..., ~channel_mask, :]  # select low-precision channels
    
    q_fmt = outlier_format if outlier_format is not None else self
    
    x_high_q = q_fmt.quantize(x_high, granularity, round_mode, ...)
    x_low_q  = self.quantize(x_low, granularity, round_mode, ...)
    
    # Return split tensors + mask for inference
    return x_high_q, x_low_q, channel_mask
```

Return type changes from single tensor to `(high_part, low_part, mask)` tuple.

### Fix 3: Operator-level Activation Split

**File**: `src/ops/linear.py` — modify `LinearFunction.forward`

When `input_sq_activation_mask` is present, replace the single-matmul path:

```python
# Input: split by mask
x_high = x[:, mask_channels]   # (batch, k_high)
x_low  = x[:, ~mask_channels]  # (batch, k_low)

# Weight: split corresponding columns
w_high = w[:, mask_channels]   # (N, k_high)
w_low  = w[:, ~mask_channels]  # (N, k_low)

# Quantize each part independently
x_high_q = quantize(x_high, cfg.input, outlier_format_override=...)
x_low_q  = quantize(x_low, cfg.input, format_override=...)
w_high_q = quantize(w_high, cfg.weight)
w_low_q  = quantize(w_low, cfg.weight)

# Two matmuls
y = F.linear(x_high_q, w_high_q) + F.linear(x_low_q, w_low_q)
```

The mask determines the channel split. The input scheme's `outlier_format` provides h_high for x_high, and `format` provides h_low for x_low.

**File**: `src/ops/conv.py` — analogous change for Conv2d

### Fix 4: Comment Fix

**File**: `src/formats/base.py:873`

```python
# Before
cols = x_b.shape[-1]  # bank_size

# After
cols = x_b.shape[-1]  # output features (N for 2D weights)
```

### Fix 5: Per-bank Mask Selection in Calibration

**File**: `src/calibration/pipeline.py`

The calibration pipeline already computes per-channel activation averages. Update it to select the mask **per-bank** when in `activation_static` mode:

```
for each bank of A_avg:
    I_j ← |A_avg[j] · sum_i |W[j,:]||  for j in bank
    m_bank ← top-(1-s) channels by I_j within bank
```

The combined mask (concatenation of all bank masks) is the `_sq_activation_mask` buffer stored on the module.

## Files Modified

| File | Change |
|------|--------|
| `src/formats/_sq_importance.py` | Fix activation importance formula |
| `src/formats/base.py` | Rewrite `_quantize_sq_activation_static` to split; fix comment |
| `src/ops/linear.py` | Two-matmul path when `sq_activation_mask` set |
| `src/ops/conv.py` | Two-matmul path for Conv2d |
| `src/calibration/pipeline.py` | Per-bank mask selection |
| `src/tests/test_sq_importance.py` | Add mixed-sign weight test |
| `src/tests/test_sq_quantize.py` | Update tests for split-based return |

## Not In Scope

- Weight row reordering (Algorithm 2 Step 8) — hardware optimization, deferred
- Compact storage / vmask (hardware feature)
- SmoothQuant integration (separate calibration task)
