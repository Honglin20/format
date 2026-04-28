"""
Differentiable bfloat quantization.

Primary API: quantize_bfloat(x, scheme, ...) — QuantScheme-driven.
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
            If None, input is returned unchanged.
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
