"""
NF4 operator equivalence tests — norm ops (RMSNorm, LayerNorm, BatchNorm).

Since mx has no NF4, equivalence is verified against independent golden
reference implementations that replicate the exact quantization chain of
each norm operator.

Every vec_* call in the src operator is replaced by:
  raw torch op → golden_lut_quantize(result, levels).

Mathematical derivation per operator is documented inline.
"""
import pytest
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.ops.norm import (
    RMSNormFunction, LayerNormFunction, BatchNormFunction,
)

# ============================================================================
# Shared golden quantization (same as test_ops_nf4_equiv.py)
# ============================================================================

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


def _golden_q(x, levels=None):
    """Golden nearest-neighbor LUT quantization (no src/ dependency)."""
    if levels is None:
        levels = NF4_LEVELS
    levels = levels.to(dtype=x.dtype, device=x.device)

    nan_mask = torch.isnan(x)
    x_safe = torch.where(nan_mask, torch.zeros_like(x), x)
    x_safe = torch.clamp(x_safe, -1.0, 1.0)

    d = torch.abs(x_safe.unsqueeze(-1) - levels.view(*([1] * x_safe.ndim), -1))
    indices = torch.argmin(d, dim=-1)
    result = levels[indices]

    if nan_mask.any():
        result = result.clone()
        result[nan_mask] = float("nan")
    return result


# Shorthand
Q = _golden_q


def _make_input(shape, seed=42):
    """Generate test input scaled to avoid saturation in NF4 range [-1,1]."""
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32) * 0.5


def _make_weight_bias(size, seed=100):
    """Generate weight/bias in [-0.5, 0.5] to stay in NF4 range."""
    torch.manual_seed(seed)
    w = torch.randn(size, dtype=torch.float32) * 0.5
    torch.manual_seed(seed + 1)
    b = torch.randn(size, dtype=torch.float32) * 0.5
    return w, b


# ============================================================================
# RMSNorm golden
# ============================================================================
#
# Mathematical derivation — RMSNorm forward (NF4 per_tensor):
#
#   Given x ∈ R^(N,D), w ∈ R^D, b ∈ R^D, ε > 0:
#
#   Step 1:  x2     = Q(x ⊙ x)                     [vec_mul]
#   Step 2:  x_ms   = Q(mean(x2, dim=-1))            [vec_reduce_mean = Q(Σ/denom)]
#            Internally: s = Q(Σ x2), then Q(s / D)
#   Step 3:  x_mse  = Q(x_ms + ε)                    [vec_add]
#   Step 4:  x_rms  = Q(√x_mse)                      [vec_sqrt]
#   Step 5:  x_inv  = Q(1 / x_rms)                   [vec_recip]
#   Step 6:  x_norm = Q(x ⊙ x_inv)                   [vec_mul]
#   Step 7:  x_scl  = Q(w ⊙ x_norm)                  [vec_mul]
#   Step 8:  out    = Q(x_scl + b)                    [vec_add]
#
#   Where Q(t) = nearest-neighbor LUT quantize with clamp to [-1, 1].
#
# RMSNorm backward (NF4 per_tensor, empty cfg → no entry/exit quantization):
#
#   Given grad_output go ∈ R^(N,D), saved x_norm, x_rms_inv, weight w:
#
#   grad_bias:   Q(Σ go over all dims except -1)     [vec_reduce_sum]
#   grad_weight: Q(Σ Q(go ⊙ x_norm))                 [vec_mul → vec_reduce_sum]
#
#   dx_norm  = Q(go ⊙ w)                             [vec_mul]
#   dx1      = Q(dx_norm ⊙ x_inv)                    [vec_mul]
#   dx_norm2 = Q(dx1 ⊙ x_norm)                       [vec_mul]
#   dx_norm2 = Q(mean(dx_norm2, dim=-1))             [vec_reduce_mean]
#   dx_norm3 = Q(x_norm ⊙ dx_norm2)                  [vec_mul]
#   dx       = dx1 - dx_norm3                         [BARE subtraction, not vec_sub!]
#
#   Note: The bare subtraction on line 815 is a notable difference from other
#   norm backward paths which use vec_sub throughout.
#
#   Exit quantization on grad_input/weight/bias is skipped (cfg empty).


def golden_rmsnorm_fwd(x, weight, bias, eps, levels):
    """RMSNorm forward with NF4 per_tensor quantization at every step."""
    D = x.shape[-1]

    # Step 1: x2 = Q(x * x)
    x2 = Q(x * x, levels)
    # Step 2: x_ms = Q(mean(x2, dim=-1)) → vec_reduce_mean = Q(Q(Σ)/D)
    s = Q(torch.sum(x2, dim=-1, keepdim=True), levels)
    x_ms = Q(s / D, levels)
    # Step 3: x_mse = Q(x_ms + eps)
    x_mse = Q(x_ms + eps, levels)
    # Step 4: x_rms = Q(sqrt(x_mse))
    x_rms = Q(torch.sqrt(x_mse), levels)
    # Step 5: x_rms_inv = Q(1 / x_rms)
    x_rms_inv = Q(1.0 / x_rms, levels)
    # Step 6: x_norm = Q(x * x_rms_inv)
    x_norm = Q(x * x_rms_inv, levels)
    # Step 7: x_scale = Q(weight * x_norm)
    x_scale = Q(weight * x_norm, levels)
    # Step 8: output = Q(x_scale + bias)
    output = Q(x_scale + bias, levels)

    return output, x_norm, x_rms_inv


def golden_rmsnorm_bwd(grad_output, x_norm, x_rms_inv, weight, levels):
    """RMSNorm backward with NF4 per_tensor quantization.

    Returns (grad_input, grad_weight, grad_bias).
    Matches the backward chain exactly, including the bare subtraction.
    """
    sum_axes = list(range(len(grad_output.shape) - 1))
    D = grad_output.shape[-1]

    # grad_bias = Q(sum(go, sum_axes))
    grad_bias = Q(torch.sum(grad_output, dim=sum_axes), levels)

    # grad_weight = Q(sum(Q(go * x_norm), sum_axes))
    gw_temp = Q(grad_output * x_norm, levels)
    grad_weight = Q(torch.sum(gw_temp, dim=sum_axes), levels)

    # dx_norm = Q(go * weight)
    dx_norm = Q(grad_output * weight, levels)
    # dx1 = Q(dx_norm * x_rms_inv)
    dx1 = Q(dx_norm * x_rms_inv, levels)
    # dx_norm2 = Q(dx1 * x_norm)
    dx_norm2 = Q(dx1 * x_norm, levels)
    # dx_norm2 = vec_reduce_mean(dx_norm2, -1) → Q(Q(Σ)/D)
    s2 = Q(torch.sum(dx_norm2, dim=-1, keepdim=True), levels)
    dx_norm2 = Q(s2 / D, levels)
    # dx_norm3 = Q(x_norm * dx_norm2)
    dx_norm3 = Q(x_norm * dx_norm2, levels)
    # grad_input = dx1 - dx_norm3  ← BARE subtraction
    grad_input = dx1 - dx_norm3

    return grad_input, grad_weight, grad_bias


# ============================================================================
# LayerNorm golden
# ============================================================================
#
# Mathematical derivation — LayerNorm forward (NF4 per_tensor):
#
#   Uses _norm_forward with axes=[-1], no groups, no weight_axis:
#
#   Step 1:  x_mean = vec_reduce_mean(x, [-1])       [Q(Q(Σ)/D)]
#   Step 2:  x_shift = Q(x - x_mean)                  [vec_sub]
#   Step 3:  x_sq = Q(x_shift ⊙ x_shift)              [vec_mul]
#   Step 4:  x_var = vec_reduce_mean(x_sq, [-1])      [Q(Q(Σ)/D)]
#   Step 5:  x_vare = Q(x_var + ε)                    [vec_add]
#   Step 6:  x_std = Q(√x_vare)                       [vec_sqrt]
#   Step 7:  x_std_inv = Q(1 / x_std)                 [vec_recip]
#   Step 8:  x_norm = Q(x_shift ⊙ x_std_inv)          [vec_mul]
#   Step 9:  x_scale = Q(w ⊙ x_norm)                  [vec_mul]
#   Step 10: output = Q(x_scale + b)                   [vec_add]
#
# LayerNorm backward (_norm_backward_LN):
#
#   Given: go, saved x_norm, x_vare, weight
#
#   dx_norm = Q(go ⊙ w)                               [vec_mul]
#   x_std = Q(√x_vare)                                [vec_sqrt]
#   x_std_inv = Q(1 / x_std)                          [vec_div]
#   dx_shift = Q(dx_norm ⊙ x_std_inv)                 [vec_mul]
#
#   dx_std_tmp = Q(dx_norm ⊙ x_norm)                  [vec_mul]
#   dx_std_tmp = Q(dx_std_tmp ⊙ x_std)                [vec_mul]
#   dx_std_tmp = vec_reduce_mean(dx_std_tmp, [-1])    [Q(Q(Σ)/D)]
#   x_vare_inv = Q(1 / x_vare)                        [vec_div]
#   dx_std_tmp = Q(dx_std_tmp ⊙ x_vare_inv)           [vec_mul]
#   dx_shift2 = Q(-dx_std_tmp ⊙ x_norm)               [vec_mul]
#
#   dx = Q(dx_shift + dx_shift2)                      [vec_add]
#   dx_mean = vec_reduce_mean(dx, [-1])               [Q(Q(Σ)/D)]
#   dx = Q(dx + (-dx_mean))                            [vec_add]


def golden_layernorm_fwd(x, weight, bias, eps, levels):
    """LayerNorm forward with NF4 per_tensor quantization."""
    D = x.shape[-1]

    # Step 1: x_mean = vec_reduce_mean(x, [-1])
    s = Q(torch.sum(x, dim=-1, keepdim=True), levels)
    x_mean = Q(s / D, levels)
    # Step 2: x_shift = Q(x - x_mean)
    x_shift = Q(x - x_mean, levels)
    # Step 3: x_sq = Q(x_shift * x_shift)
    x_sq = Q(x_shift * x_shift, levels)
    # Step 4: x_var = vec_reduce_mean(x_sq, [-1])
    s2 = Q(torch.sum(x_sq, dim=-1, keepdim=True), levels)
    x_var = Q(s2 / D, levels)
    # Step 5: x_vare = Q(x_var + eps)
    x_vare = Q(x_var + eps, levels)
    # Step 6: x_std = Q(sqrt(x_vare))
    x_std = Q(torch.sqrt(x_vare), levels)
    # Step 7: x_std_inv = Q(1 / x_std)
    x_std_inv = Q(1.0 / x_std, levels)
    # Step 8: x_norm = Q(x_shift * x_std_inv)
    x_norm = Q(x_shift * x_std_inv, levels)
    # Step 9: x_scale = Q(weight * x_norm)
    x_scale = Q(weight * x_norm, levels)
    # Step 10: output = Q(x_scale + bias)
    output = Q(x_scale + bias, levels)

    return output, x_norm, x_vare


def golden_layernorm_bwd(grad_output, x_norm, x_vare, weight, levels):
    """LayerNorm backward (_norm_backward_LN) with NF4 per_tensor."""
    D = grad_output.shape[-1]

    # dx_norm = Q(go * weight)
    dx_norm = Q(grad_output * weight, levels)

    # x_std = Q(sqrt(x_vare))
    x_std = Q(torch.sqrt(x_vare), levels)
    # x_std_inv = Q(1 / x_std)
    x_std_inv = Q(1.0 / x_std, levels)

    # dx_shift = Q(dx_norm * x_std_inv)
    dx_shift = Q(dx_norm * x_std_inv, levels)

    # dx_std_tmp = Q(dx_norm * x_norm)
    dx_std_tmp = Q(dx_norm * x_norm, levels)
    # dx_std_tmp = Q(dx_std_tmp * x_std)
    dx_std_tmp = Q(dx_std_tmp * x_std, levels)
    # dx_std_tmp = vec_reduce_mean(dx_std_tmp, [-1])
    s = Q(torch.sum(dx_std_tmp, dim=-1, keepdim=True), levels)
    dx_std_tmp = Q(s / D, levels)

    # x_vare_inv = Q(1 / x_vare)
    x_vare_inv = Q(1.0 / x_vare, levels)
    # dx_std_tmp = Q(dx_std_tmp * x_vare_inv)
    dx_std_tmp = Q(dx_std_tmp * x_vare_inv, levels)

    # dx_shift2 = Q((-dx_std_tmp) * x_norm)  → vec_mul(-dx_std_tmp, x_norm)
    dx_shift2 = Q((-dx_std_tmp) * x_norm, levels)

    # dx = Q(dx_shift + dx_shift2)
    dx = Q(dx_shift + dx_shift2, levels)

    # dx_mean = vec_reduce_mean(dx, [-1])
    sm = Q(torch.sum(dx, dim=-1, keepdim=True), levels)
    dx_mean = Q(sm / D, levels)

    # dx = Q(dx + (-dx_mean))
    dx = Q(dx + (-dx_mean), levels)

    return dx


# ============================================================================
# BatchNorm golden
# ============================================================================
#
# Mathematical derivation — BatchNorm forward (NF4 per_tensor, training mode, no running stats):
#
#   Uses _norm_forward with sum_axes = [0, 2, 3, ...] and weight_axis=1.
#   Input: x ∈ R^(N,C,H,W), w,b ∈ R^C
#
#   Step 1:  x_mean = vec_reduce_mean(x, [0,2,3,...])  [Q(Q(Σ)/M)]
#            where M = N*H*W*... (all dims except C)
#   Step 2:  x_shift = Q(x - x_mean)                    [vec_sub]
#   Step 3:  x_sq = Q(x_shift ⊙ x_shift)                [vec_mul]
#   Step 4:  x_var = vec_reduce_mean(x_sq, [0,2,3,...]) [Q(Q(Σ)/M)]
#   Step 5:  x_vare = Q(x_var + ε)                      [vec_add]
#   Step 6:  x_std = Q(√x_vare)                         [vec_sqrt]
#   Step 7:  x_std_inv = Q(1 / x_std)                   [vec_recip]
#   Step 8:  x_norm = Q(x_shift ⊙ x_std_inv)            [vec_mul]
#            weight/bias reshaped to (1,C,1,1,...)
#   Step 9:  x_scale = Q(w_reshaped ⊙ x_norm)           [vec_mul]
#   Step 10: output = Q(x_scale + b_reshaped)            [vec_add]
#
# BatchNorm backward (_norm_backward):
#
#   Given: go, saved x_shift, x_std_inv, weight
#   weight reshaped to (1,C,1,1,...)
#
#   dx_norm = Q(go ⊙ w_reshaped)                        [vec_mul]
#   dx_shift = Q(dx_norm ⊙ x_std_inv)                   [vec_mul]
#   dx_mean = vec_reduce_mean(Q(-dx_shift), axes)       [neg → q → Q(Σ)/M]
#            Actually: Q(Q(-dx_shift) sum/M) → 2 q steps
#            Wait: vec_reduce_mean(-dx_shift, ...) → Q(Q(Σ(-dx_shift))/M)
#
#   dx_std = Q(dx_norm ⊙ x_shift)                       [vec_mul]
#   dx_std = vec_reduce_mean(dx_std, axes)              [Q(Q(Σ)/M)]
#   x_vare_inv = Q(x_std_inv ⊙ x_std_inv)               [vec_mul]
#   dx_std = Q(dx_std ⊙ x_vare_inv)                     [vec_mul]
#   dx_std = Q(dx_std ⊙ x_std_inv)                      [vec_mul]
#   dx_shift2 = Q((-dx_std) ⊙ x_shift)                  [vec_mul]
#
#   dx = Q(dx_shift + dx_shift2)                        [vec_add]
#   dx = Q(dx + dx_mean)                                [vec_add]


def _vec_reduce_mean_golden(x, dims, levels):
    """Golden vec_reduce_mean: Q(Q(sum(x, dims)) / denom)."""
    denom = 1
    for d in dims:
        denom *= x.shape[d]
    s = Q(torch.sum(x, dim=dims, keepdim=True), levels)
    return Q(s / denom, levels)


def golden_batchnorm_fwd(x, weight, bias, eps, levels):
    """BatchNorm forward with NF4 per_tensor quantization.

    Uses sum_axes = [0] + list(range(2, x.ndim)), weight_axis=1.
    """
    ndim = x.ndim
    sum_axes = [0] + list(range(2, ndim))

    # Build weight/bias shapes: (1, C, 1, 1, ...)
    w_shape = [1] * ndim
    w_shape[1] = x.shape[1]
    w = weight.view(w_shape)
    b = bias.view(w_shape)

    # Step 1: x_mean = vec_reduce_mean(x, sum_axes)
    x_mean = _vec_reduce_mean_golden(x, sum_axes, levels)
    # Step 2: x_shift = Q(x - x_mean)
    x_shift = Q(x - x_mean, levels)
    # Step 3: x_sq = Q(x_shift * x_shift)
    x_sq = Q(x_shift * x_shift, levels)
    # Step 4: x_var = vec_reduce_mean(x_sq, sum_axes)
    x_var = _vec_reduce_mean_golden(x_sq, sum_axes, levels)
    # Step 5: x_vare = Q(x_var + eps)
    x_vare = Q(x_var + eps, levels)
    # Step 6: x_std = Q(sqrt(x_vare))
    x_std = Q(torch.sqrt(x_vare), levels)
    # Step 7: x_std_inv = Q(1 / x_std)
    x_std_inv = Q(1.0 / x_std, levels)
    # Step 8: x_norm = Q(x_shift * x_std_inv)
    x_norm = Q(x_shift * x_std_inv, levels)
    # Step 9: x_scale = Q(w * x_norm)
    x_scale = Q(w * x_norm, levels)
    # Step 10: output = Q(x_scale + b)
    output = Q(x_scale + b, levels)

    return output, x_shift, x_norm, x_std_inv


def golden_batchnorm_bwd(grad_output, x_shift, x_norm, x_std_inv, weight, levels):
    """BatchNorm backward (_norm_backward) with NF4 per_tensor."""
    ndim = grad_output.ndim
    sum_axes = [0] + list(range(2, ndim))

    w_shape = [1] * ndim
    w_shape[1] = grad_output.shape[1]
    w = weight.view(w_shape)

    # dx_norm = Q(go * w)
    dx_norm = Q(grad_output * w, levels)
    # dx_shift = Q(dx_norm * x_std_inv)
    dx_shift = Q(dx_norm * x_std_inv, levels)

    # dx_mean = vec_reduce_mean(-dx_shift, sum_axes) → Q(Q(Σ(-dx_shift))/M)
    neg_dx_shift = Q(-dx_shift, levels)
    dx_mean = _vec_reduce_mean_golden(neg_dx_shift, sum_axes, levels)

    # dx_std = Q(dx_norm * x_shift)
    dx_std = Q(dx_norm * x_shift, levels)
    # dx_std = vec_reduce_mean(dx_std, sum_axes)
    dx_std = _vec_reduce_mean_golden(dx_std, sum_axes, levels)
    # x_vare_inv = Q(x_std_inv * x_std_inv)
    x_vare_inv = Q(x_std_inv * x_std_inv, levels)
    # dx_std = Q(dx_std * x_vare_inv)
    dx_std = Q(dx_std * x_vare_inv, levels)
    # dx_std = Q(dx_std * x_std_inv)
    dx_std = Q(dx_std * x_std_inv, levels)
    # dx_shift2 = Q((-dx_std) * x_shift)
    dx_shift2 = Q((-dx_std) * x_shift, levels)

    # dx = Q(dx_shift + dx_shift2)
    dx = Q(dx_shift + dx_shift2, levels)
    # dx = Q(dx + dx_mean)
    dx = Q(dx + dx_mean, levels)

    return dx


# ============================================================================
# Test helpers
# ============================================================================

def _nf4_inner():
    """NF4 per_tensor inner scheme."""
    return QuantScheme.per_tensor("nf4")


def _empty_cfg():
    """Empty OpQuantConfig — no storage/entry/exit quantization."""
    return OpQuantConfig()


# ============================================================================
# RMSNorm tests
# ============================================================================

class TestNF4RMSNorm:
    def test_forward_matches_golden(self):
        """RMSNorm forward: 8 vec_ops in inner scheme, empty cfg."""
        x = _make_input((2, 8), seed=1)
        w, b = _make_weight_bias(8, seed=100)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_out = RMSNormFunction.apply(
            x.clone(), w.clone(), b.clone(), eps,
            cfg, scheme, True, None, None,
        )

        gold_out, _, _ = golden_rmsnorm_fwd(x, w, b, eps, NF4_LEVELS)
        assert torch.equal(src_out, gold_out), (
            f"RMSNorm forward mismatch\n"
            f"max diff: {(src_out - gold_out).abs().max()}"
        )

    def test_backward_matches_golden(self):
        """RMSNorm backward: 5 vec_ops + 1 bare subtraction."""
        x = _make_input((2, 8), seed=2)
        w, b = _make_weight_bias(8, seed=101)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_x = x.clone().requires_grad_(True)
        src_w = w.clone().requires_grad_(True)
        src_b = b.clone().requires_grad_(True)

        src_out = RMSNormFunction.apply(
            src_x, src_w, src_b, eps,
            cfg, scheme, True, None, None,
        )
        src_out.sum().backward()

        # Golden forward to get intermediate values
        _, gold_x_norm, gold_x_rms_inv = golden_rmsnorm_fwd(x, w, b, eps, NF4_LEVELS)

        gold_gi, gold_gw, gold_gb = golden_rmsnorm_bwd(
            torch.ones_like(src_out), gold_x_norm, gold_x_rms_inv, w, NF4_LEVELS,
        )

        assert torch.equal(src_x.grad, gold_gi), (
            f"RMSNorm grad_input mismatch\n"
            f"max diff: {(src_x.grad - gold_gi).abs().max()}"
        )
        assert torch.equal(src_w.grad, gold_gw), (
            f"RMSNorm grad_weight mismatch\n"
            f"max diff: {(src_w.grad - gold_gw).abs().max()}"
        )
        assert torch.equal(src_b.grad, gold_gb), (
            f"RMSNorm grad_bias mismatch\n"
            f"max diff: {(src_b.grad - gold_gb).abs().max()}"
        )

    def test_forward_bare_subtraction_in_backward(self):
        """Verify the bare subtraction (line 815) produces expected gradient.

        The RMSNorm backward uses dx1 - dx_norm3 (bare torch.sub) instead of
        vec_sub. This is a unique characteristic of the RMSNorm backward path.
        """
        x = _make_input((2, 8), seed=3)
        w, b = _make_weight_bias(8, seed=102)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_x = x.clone().requires_grad_(True)
        src_w = w.clone().requires_grad_(True)
        src_b = b.clone().requires_grad_(True)

        src_out = RMSNormFunction.apply(
            src_x, src_w, src_b, eps, cfg, scheme, True, None, None,
        )
        src_out.sum().backward()

        # Compute golden backward WITHOUT quantizing the final subtraction
        # (this should NOT match, confirming bare subtraction is intentional)
        _, gold_x_norm, gold_x_rms_inv = golden_rmsnorm_fwd(x, w, b, eps, NF4_LEVELS)

        # Recompute the backward but quantize the final subtraction too
        grad_output = torch.ones_like(src_out)
        dx_norm = Q(grad_output * w, NF4_LEVELS)
        dx1 = Q(dx_norm * gold_x_rms_inv, NF4_LEVELS)
        dx_norm2 = Q(dx1 * gold_x_norm, NF4_LEVELS)
        s2 = Q(torch.sum(dx_norm2, dim=-1, keepdim=True), NF4_LEVELS)
        dx_norm2 = Q(s2 / src_out.shape[-1], NF4_LEVELS)
        dx_norm3 = Q(gold_x_norm * dx_norm2, NF4_LEVELS)
        # Quantized subtraction (what it would be if using vec_sub)
        grad_quantized_sub = Q(dx1 - dx_norm3, NF4_LEVELS)

        # The actual gradient uses bare subtraction → should differ from
        # quantized subtraction (unless the bare subtraction result happens
        # to already be at a quantization level)
        assert not torch.equal(src_x.grad, grad_quantized_sub), (
            "RMSNorm backward uses bare subtraction, but quantized "
            "subtraction produced the same result (should differ)"
        )


# ============================================================================
# LayerNorm tests
# ============================================================================

class TestNF4LayerNorm:
    def test_forward_matches_golden(self):
        """LayerNorm forward: 10 vec_ops via _norm_forward."""
        x = _make_input((2, 8), seed=4)
        w, b = _make_weight_bias(8, seed=103)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_out = LayerNormFunction.apply(
            x.clone(), w.clone(), b.clone(), eps,
            cfg, scheme, True, None, None,
        )

        gold_out, _, _ = golden_layernorm_fwd(x, w, b, eps, NF4_LEVELS)
        assert torch.equal(src_out, gold_out), (
            f"LayerNorm forward mismatch\n"
            f"max diff: {(src_out - gold_out).abs().max()}"
        )

    def test_backward_matches_golden(self):
        """LayerNorm backward: _norm_backward_LN with full quantization."""
        x = _make_input((2, 8), seed=5)
        w, b = _make_weight_bias(8, seed=104)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_x = x.clone().requires_grad_(True)
        src_w = w.clone().requires_grad_(True)
        src_b = b.clone().requires_grad_(True)

        src_out = LayerNormFunction.apply(
            src_x, src_w, src_b, eps,
            cfg, scheme, True, None, None,
        )
        src_out.sum().backward()

        _, gold_x_norm, gold_x_vare = golden_layernorm_fwd(x, w, b, eps, NF4_LEVELS)
        gold_gi = golden_layernorm_bwd(
            torch.ones_like(src_out), gold_x_norm, gold_x_vare, w, NF4_LEVELS,
        )

        assert torch.equal(src_x.grad, gold_gi), (
            f"LayerNorm grad_input mismatch\n"
            f"max diff: {(src_x.grad - gold_gi).abs().max()}"
        )


# ============================================================================
# BatchNorm tests
# ============================================================================

class TestNF4BatchNorm:
    def test_forward_matches_golden(self):
        """BatchNorm forward: 10 vec_ops, training mode, compute stats from input."""
        x = _make_input((2, 4, 3, 3), seed=6)  # NCHW
        w, b = _make_weight_bias(4, seed=105)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_out = BatchNormFunction.apply(
            x.clone(), None, None, w.clone(), b.clone(),
            True, 0.1, eps,  # is_training=True, momentum, eps
            cfg, scheme, True, None, None,
        )

        gold_out, _, _, _ = golden_batchnorm_fwd(x, w, b, eps, NF4_LEVELS)

        assert torch.equal(src_out, gold_out), (
            f"BatchNorm forward mismatch\n"
            f"max diff: {(src_out - gold_out).abs().max()}"
        )

    def test_backward_matches_golden(self):
        """BatchNorm backward: _norm_backward with full quantization."""
        x = _make_input((2, 4, 3, 3), seed=7)
        w, b = _make_weight_bias(4, seed=106)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_x = x.clone().requires_grad_(True)
        src_w = w.clone().requires_grad_(True)
        src_b = b.clone().requires_grad_(True)

        src_out = BatchNormFunction.apply(
            src_x, None, None, src_w, src_b,
            True, 0.1, eps,
            cfg, scheme, True, None, None,
        )
        src_out.sum().backward()

        _, gold_x_shift, gold_x_norm, gold_x_std_inv = golden_batchnorm_fwd(
            x, w, b, eps, NF4_LEVELS,
        )

        gold_gi = golden_batchnorm_bwd(
            torch.ones_like(src_out),
            gold_x_shift, gold_x_norm, gold_x_std_inv, w, NF4_LEVELS,
        )

        assert torch.equal(src_x.grad, gold_gi), (
            f"BatchNorm grad_input mismatch\n"
            f"max diff: {(src_x.grad - gold_gi).abs().max()}"
        )


# ============================================================================
# STE mode (quantize_backprop=False)
# ============================================================================

class TestNF4NormSTE:
    def test_rmsnorm_ste_backward(self):
        """RMSNorm STE backward: no intermediate quantization in backward."""
        x = _make_input((2, 8), seed=10)
        w, b = _make_weight_bias(8, seed=107)
        eps = 1e-5

        scheme = _nf4_inner()
        cfg = _empty_cfg()

        src_x = x.clone().requires_grad_(True)
        src_w = w.clone().requires_grad_(True)
        src_b = b.clone().requires_grad_(True)

        src_out = RMSNormFunction.apply(
            src_x, src_w, src_b, eps, cfg, scheme, False, None, None,
        )
        src_out.sum().backward()

        # STE backward should produce non-zero, finite gradients
        assert src_x.grad is not None
        assert torch.isfinite(src_x.grad).all()
        assert src_x.grad.abs().sum() > 0, "STE gradient should not be all-zero"


# ============================================================================
# Per-channel NF4
# ============================================================================

class TestNF4NormPerChannel:
    """Norm operators with per_channel NF4 quantization on the last axis."""

    def test_rmsnorm_per_channel(self):
        """RMSNorm with per_channel NF4: each channel independently scaled."""
        x = _make_input((2, 8), seed=11)
        w, b = _make_weight_bias(8, seed=108)
        eps = 1e-5

        pc = QuantScheme.per_channel("nf4", axis=-1)
        cfg = _empty_cfg()

        src_out = RMSNormFunction.apply(
            x, w, b, eps, cfg, pc, True, None, None,
        )
        assert src_out.shape == (2, 8)
        assert torch.isfinite(src_out).all()

    def test_rmsnorm_per_channel_vs_per_tensor_differ(self):
        """Per-channel and per-tensor NF4 produce different RMSNorm results."""
        x = _make_input((2, 8), seed=12)
        w, b = _make_weight_bias(8, seed=109)
        eps = 1e-5

        pc = QuantScheme.per_channel("nf4", axis=-1)
        pt = QuantScheme.per_tensor("nf4")
        cfg = _empty_cfg()

        out_pc = RMSNormFunction.apply(x.clone(), w, b, eps, cfg, pc, True, None, None)
        out_pt = RMSNormFunction.apply(x.clone(), w, b, eps, cfg, pt, True, None, None)

        assert not torch.equal(out_pc, out_pt)
