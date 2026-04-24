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
