"""
MX block quantization: _quantize_mx, quantize_mx.

The canonical per-block implementation is FormatBase._quantize_per_block().
These wrappers exist for backward compatibility with callers (primarily tests)
that pass legacy mx parameter names.

New code should use::

    quantize(x, QuantScheme.mxfp(fmt, block_size=32))
    # or directly:
    fmt.quantize(x, GranularitySpec.per_block(32), round_mode)
"""
import torch
from src.formats.base import FormatBase
from src.formats._block_utils import (
    FP32_EXPONENT_BIAS,
    _shared_exponents,
    _reshape_to_blocks,
    _undo_reshape_to_blocks,
)


def _quantize_mx(A, scale_bits, elem_format,
                 shared_exp_method="max", axes=None, block_size=0,
                 round_mode="nearest", flush_fp32_subnorms=False, scale=None):
    """Per-block quantize with shared exponents (backward-compat wrapper).

    Delegates to ``FormatBase._quantize_per_block()`` for the standard path
    (shared_exp_method='max', flush_fp32_subnorms=False, block_size>0).
    Non-standard and per_tensor paths are handled by ``_mx_legacy``.
    """
    if elem_format is None:
        return A

    fmt = FormatBase.from_str(elem_format) if isinstance(elem_format, str) else elem_format

    # Per-tensor MX (block_size=0) uses shared exponents without tiling —
    # not the same as _quantize_per_tensor (which skips shared exponents).
    if block_size <= 0 or shared_exp_method != "max" or flush_fp32_subnorms:
        return _mx_legacy(A, fmt, block_size, axes, round_mode,
                          shared_exp_method, flush_fp32_subnorms)

    from src.scheme.granularity import GranularitySpec
    axis = axes[0] if isinstance(axes, list) else axes
    gran = GranularitySpec.per_block(block_size, axis=axis)
    return fmt._quantize_per_block(A, gran, round_mode)


def _mx_legacy(A, fmt, block_size, axes, round_mode,
                shared_exp_method, flush_fp32_subnorms):
    """Non-standard shared_exp / flush / per_tensor paths (mx compat only)."""
    axes = [axes] if isinstance(axes, int) else (axes or [])
    axes = [A.ndim + a if a < 0 else a for a in axes]

    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(A, axes, block_size)

    shared_exp_axes = [a + 1 for a in axes] if block_size > 0 else axes
    shared_exp = _shared_exponents(A, method=shared_exp_method,
                                   axes=shared_exp_axes, ebits=0)

    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    shared_exp = shared_exp - fmt.emax
    scale_emax = 2**(8-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)
    A = fmt.quantize_elemwise(A, round_mode=round_mode,
                               allow_denorm=True, saturate_normals=True)
    A = A * (2**shared_exp)

    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


def quantize_mx(
    A,
    scheme,
    axes=None,
    scale_bits=8,
    shared_exp_method="max",
    flush_fp32_subnorms=False,
):
    """Quantize tensor A using MX block quantization.

    When *scheme.transform* is not IdentityTransform, delegates to the
    standard three-step :func:`quantize` flow (ADR-001 compliant).

    When *scheme.transform* is IdentityTransform, uses the direct MX
    path (``_quantize_mx``) as a fast path.

    Args:
        A: Input tensor.
        scheme: QuantScheme. granularity.mode must be PER_BLOCK or PER_TENSOR.
        axes: Axes for shared exponent computation.
        scale_bits: Bits for shared scale (sign + magnitude). Default: 8.
        shared_exp_method: "max" or "none". Default: "max".
        flush_fp32_subnorms: Flush subnormal FP32 blocks to zero.

    Returns:
        Quantized tensor with same shape as A.

    Raises:
        ValueError: If scheme.granularity is PER_CHANNEL.
    """
    if scheme is None:
        return A

    from src.scheme.granularity import GranularityMode
    from src.scheme.transform import IdentityTransform

    if not isinstance(scheme.transform, IdentityTransform):
        from src.quantize.elemwise import quantize

        return quantize(A, scheme)
    if scheme.granularity.mode == GranularityMode.PER_CHANNEL:
        raise ValueError(
            "quantize_mx does not support PER_CHANNEL granularity. "
            "Use quantize(x, scheme) for per-channel quantization."
        )

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
