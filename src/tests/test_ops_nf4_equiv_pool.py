"""
NF4 operator equivalence tests — AdaptiveAvgPool2d.

Since mx has no NF4, equivalence is verified against independent golden
reference implementations that replicate the exact quantization chain.

Mathematical derivation documented inline.
"""
import math
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.ops.pooling import AdaptiveAvgPool2dFunction, _start_index, _end_index

# ============================================================================
# Shared golden quantization
# ============================================================================

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


def _golden_q(x, levels=None):
    """Golden nearest-neighbor LUT quantization."""
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


Q = _golden_q


def _make_input(shape, seed=42):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32) * 0.5


# ============================================================================
# AdaptiveAvgPool2d golden
# ============================================================================
#
# Mathematical derivation — Pool forward (NF4 per_tensor):
#
#   Given x ∈ R^(N,C,H,W), output_size (oH, oW):
#
#   For each output pixel (oh, ow):
#     istartH = floor(oh * H / oH),  iendH = ceil((oh+1) * H / oH)
#     istartW = floor(ow * W / oW),  iendW = ceil((ow+1) * W / oW)
#     slice = x[:, :, istartH:iendH, istartW:iendW]   ∈ R^(N,C,kH,kW)
#
#     pixel = vec_reduce_mean(slice, [2, 3])
#           = Q(Q(sum(slice, [2,3])) / (kH * kW))
#
#   All pixels produced by quantized reduce_mean → the entire output is
#   quantized element-by-element.
#
# Pool backward (NF4 per_tensor):
#
#   grad_input initialized to zeros of input shape.
#   For each output pixel (oh, ow):
#     grad_delta = grad_output[:, :, oh, ow] / kH / kW    [BARE div, NOT quantized]
#     expanded = grad_delta.view(N,C,1,1).expand(N,C,kH,kW)
#     grad_input[:, :, istartH:iendH, istartW:iendW] =
#         vec_add(grad_input[slice], expanded, scheme)
#       = Q(grad_input[slice] + expanded)


def golden_pool_fwd(x, output_size, levels):
    """AdaptiveAvgPool2d forward with NF4 per_tensor quantization."""
    sizeB, sizeD, isizeH, isizeW = x.shape
    osizeH, osizeW = output_size
    device = x.device

    output = torch.zeros(sizeB, sizeD, osizeH, osizeW, device=device)

    for oh in range(osizeH):
        istartH = _start_index(oh, osizeH, isizeH)
        iendH = _end_index(oh, osizeH, isizeH)
        kH = iendH - istartH

        for ow in range(osizeW):
            istartW = _start_index(ow, osizeW, isizeW)
            iendW = _end_index(ow, osizeW, isizeW)
            kW = iendW - istartW

            input_slice = x[:, :, istartH:iendH, istartW:iendW]
            # vec_reduce_mean(input_slice, [2, 3])
            # = Q(Q(sum(input_slice, [2,3])) / (kH * kW))
            s = Q(torch.sum(input_slice, dim=[2, 3]), levels)
            output[:, :, oh, ow] = Q(s / (kH * kW), levels)

    return output


def golden_pool_bwd(grad_output, input_shape, output_size, levels):
    """AdaptiveAvgPool2d backward with NF4 per_tensor quantization.

    Replicates the accumulation loop with vec_add at each step.
    """
    sizeB, sizeD, isizeH, isizeW = input_shape
    osizeH, osizeW = output_size
    device = grad_output.device

    grad_input = torch.zeros(sizeB, sizeD, isizeH, isizeW, device=device)

    for oh in range(osizeH):
        istartH = _start_index(oh, osizeH, isizeH)
        iendH = _end_index(oh, osizeH, isizeH)
        kH = iendH - istartH

        for ow in range(osizeW):
            istartW = _start_index(ow, osizeW, isizeW)
            iendW = _end_index(ow, osizeW, isizeW)
            kW = iendW - istartW

            # grad_delta = go / kH / kW — bare division, NOT quantized
            grad_delta = grad_output[:, :, oh, ow] / kH / kW

            # Expand to slice shape
            expanded = grad_delta.view(sizeB, sizeD, 1, 1).expand(sizeB, sizeD, kH, kW)

            # vec_add(grad_input_slice, expanded) = Q(grad_input_slice + expanded)
            current = grad_input[:, :, istartH:iendH, istartW:iendW]
            updated = Q(current + expanded, levels)
            grad_input[:, :, istartH:iendH, istartW:iendW] = updated

    return grad_input


# ============================================================================
# Tests
# ============================================================================

def _nf4_inner():
    return QuantScheme.per_tensor("nf4")


class TestNF4AdaptiveAvgPool2d:
    def test_forward_matches_golden(self):
        x = _make_input((2, 3, 6, 6), seed=1)
        output_size = (2, 2)
        scheme = _nf4_inner()

        src_out = AdaptiveAvgPool2dFunction.apply(
            x, output_size, scheme, True,
        )
        gold_out = golden_pool_fwd(x, output_size, NF4_LEVELS)

        assert torch.equal(src_out, gold_out), (
            f"Pool forward mismatch\n"
            f"max diff: {(src_out - gold_out).abs().max()}"
        )

    def test_backward_matches_golden(self):
        x = _make_input((2, 3, 6, 6), seed=2).requires_grad_(True)
        output_size = (2, 2)
        scheme = _nf4_inner()

        src_out = AdaptiveAvgPool2dFunction.apply(
            x, output_size, scheme, True,
        )
        src_out.sum().backward()

        gold_grad = golden_pool_bwd(
            torch.ones_like(src_out), (2, 3, 6, 6), output_size, NF4_LEVELS,
        )

        assert torch.equal(x.grad, gold_grad), (
            f"Pool backward mismatch\n"
            f"max diff: {(x.grad - gold_grad).abs().max() if x.grad is not None else 'N/A'}"
        )

    def test_forward_single_output(self):
        """Pool to single pixel (global average pooling)."""
        x = _make_input((1, 3, 8, 8), seed=3)
        output_size = (1, 1)
        scheme = _nf4_inner()

        src_out = AdaptiveAvgPool2dFunction.apply(x, output_size, scheme, True)
        gold_out = golden_pool_fwd(x, output_size, NF4_LEVELS)

        assert torch.equal(src_out, gold_out)

    def test_forward_same_size(self):
        """Pool where output_size == input_size → identity-like (still quantized)."""
        x = _make_input((1, 2, 4, 4), seed=4)
        output_size = (4, 4)
        scheme = _nf4_inner()

        src_out = AdaptiveAvgPool2dFunction.apply(x, output_size, scheme, True)
        gold_out = golden_pool_fwd(x, output_size, NF4_LEVELS)

        assert torch.equal(src_out, gold_out)

    def test_backward_single_output(self):
        """Global average pool backward."""
        x = _make_input((1, 3, 4, 4), seed=5).requires_grad_(True)
        output_size = (1, 1)
        scheme = _nf4_inner()

        src_out = AdaptiveAvgPool2dFunction.apply(x, output_size, scheme, True)
        src_out.sum().backward()

        gold_grad = golden_pool_bwd(
            torch.ones_like(src_out), (1, 3, 4, 4), output_size, NF4_LEVELS,
        )
        assert torch.equal(x.grad, gold_grad)


# ============================================================================
# STE mode
# ============================================================================

class TestNF4PoolSTE:
    def test_ste_forward(self):
        x = _make_input((2, 3, 4, 4), seed=10)
        scheme = _nf4_inner()

        out_qbp = AdaptiveAvgPool2dFunction.apply(x, (2, 2), scheme, True)
        out_ste = AdaptiveAvgPool2dFunction.apply(x, (2, 2), scheme, False)
        assert torch.equal(out_qbp, out_ste)

    def test_ste_backward(self):
        x = _make_input((2, 3, 4, 4), seed=11).requires_grad_(True)
        scheme = _nf4_inner()

        out = AdaptiveAvgPool2dFunction.apply(x, (2, 2), scheme, False)
        out.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
