"""
MX block quantization: _quantize_mx, quantize_mx.

The canonical per-block implementation is FormatBase._quantize_per_block().
Block utility functions (_reshape_to_blocks, _shared_exponents, etc.) live in
src.formats._block_utils.
"""
import torch
from src.formats.base import FormatBase
from src.formats._block_utils import (
    FP32_EXPONENT_BIAS,
    FP32_MIN_NORMAL,
    _shared_exponents,
    _reshape_to_blocks,
    _undo_reshape_to_blocks,
)


# ---------------------------------------------------------------------------
# Core MX quantization
# ---------------------------------------------------------------------------

def _quantize_mx(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round_mode="nearest",
    flush_fp32_subnorms=False,
    scale=None,
):
    """Quantize tensor A using MX-style per-block shared exponents.

    Args:
        scale: Optional pre-computed shared exponent tensor.  If provided,
            ``_shared_exponents()`` is skipped and this is used directly.
            Must have the correct shape for broadcasting with the
            (possibly tiled) ``A`` tensor.
    """
    # Shortcut for no quantization
    if elem_format is None:
        return A

    if scale_bits <= 0:
        raise ValueError("scale_bits must be > 0")

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # Get format instance
    if isinstance(elem_format, str):
        fmt = FormatBase.from_str(elem_format)
    else:
        fmt = elem_format

    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(A, axes, block_size)

    if scale is not None:
        shared_exp = scale
    else:
        # Quantize
        shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

        # Get shared exponents
        shared_exp = _shared_exponents(
            A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
        )

    # Flush subnormal FP32 inputs to zero
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    # Offset the max exponent by the largest representable exponent
    shared_exp = shared_exp - fmt.emax

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)

    A = fmt.quantize_elemwise(A, round_mode=round_mode,
                              allow_denorm=True, saturate_normals=True)

    A = A * (2**shared_exp)

    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
