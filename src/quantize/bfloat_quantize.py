"""
Differentiable bfloat quantization.

Rewritten from mx/quantize.py. Key changes:
- Parameter 'round' renamed to 'round_mode'
- Uses src.quantize.elemwise.quantize_elemwise_op
- Uses src.specs for apply_mx_specs, get_backwards_mx_specs, mx_assert_test
"""
import torch

from src.quantize.elemwise import quantize_elemwise_op
from src.specs.specs import apply_mx_specs, get_backwards_mx_specs, mx_assert_test


class QuantizeBfloatFunction(torch.autograd.Function):
    """Forward: quantize to bfloat. Backward: quantize gradients to bfloat."""

    @staticmethod
    def forward(ctx, x, mx_specs, round_mode=None):
        if round_mode is None:
            round_mode = mx_specs["round"]

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.round_mode = round_mode

        return quantize_elemwise_op(x, mx_specs=mx_specs, round_mode=round_mode)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = quantize_elemwise_op(
            grad_output, mx_specs=ctx.mx_specs, round_mode=ctx.round_mode,
        )
        return (grad_input, None, None)


def quantize_bfloat(x, mx_specs, round_mode=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return x

    mx_specs = apply_mx_specs(mx_specs)

    if round_mode is None:
        round_mode = mx_specs["round"]

    return QuantizeBfloatFunction.apply(x, mx_specs, round_mode)
