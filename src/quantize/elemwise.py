"""
Element-wise quantization.

Primary API: quantize(x, scheme) — QuantScheme-driven unified entry.

Legacy wrappers (kept for internal equivalence tests):
  _quantize_elemwise, _quantize_bfloat, _quantize_fp

Low-level primitives (_quantize_elemwise_core, _round_mantissa, _safe_lshift,
_safe_rshift) are defined in src.formats._core and re-exported here for
backward compatibility with tests.
"""
import torch
from src.formats.base import compute_min_norm, compute_max_norm, FormatBase
from src.formats._core import (
    _elemwise_core,
    _round_mantissa,
    _safe_lshift,
    _safe_rshift,
)

# Backward compatibility alias for tests
_quantize_elemwise_core = _elemwise_core


# ---------------------------------------------------------------------------
# Format-specific wrappers
# ---------------------------------------------------------------------------

def _quantize_elemwise(A, elem_format, round_mode='nearest',
                       saturate_normals=False, allow_denorm=True):
    """Quantize values to a defined format using FormatBase.from_str()."""
    if elem_format is None:
        return A

    fmt = FormatBase.from_str(elem_format) if isinstance(elem_format, str) else elem_format

    return fmt.quantize_elemwise(A, round_mode=round_mode,
                                 allow_denorm=allow_denorm,
                                 saturate_normals=saturate_normals)


def _quantize_bfloat(A, bfloat, round_mode='nearest', allow_denorm=True):
    """Legacy: kept for internal equivalence tests only."""
    if bfloat == 0 or bfloat == 32:
        return A

    max_norm = compute_max_norm(8, bfloat - 7)
    from src.formats.fp_formats import FPFormat
    fmt = FPFormat(name=f"bfloat{bfloat}", ebits=8, mbits=bfloat - 7,
                   max_norm_override=max_norm)
    return fmt.quantize_elemwise(A, round_mode=round_mode,
                                 allow_denorm=allow_denorm,
                                 saturate_normals=False)


def _quantize_fp(A, exp_bits=None, mantissa_bits=None,
                 round_mode='nearest', allow_denorm=True):
    """Legacy: kept for internal equivalence tests only."""
    if exp_bits is None or mantissa_bits is None:
        return A

    max_norm = compute_max_norm(exp_bits, mantissa_bits + 2)
    from src.formats.fp_formats import FPFormat
    fmt = FPFormat(name=f"fp_e{exp_bits}m{mantissa_bits}", ebits=exp_bits,
                   mbits=mantissa_bits + 2, max_norm_override=max_norm)
    return fmt.quantize_elemwise(A, round_mode=round_mode,
                                 allow_denorm=allow_denorm,
                                 saturate_normals=False)


# ---------------------------------------------------------------------------
# QuantScheme-driven unified entry point
# ---------------------------------------------------------------------------

def quantize(x, scheme=None, allow_denorm=True, scale=None):
    """Quantize tensor x using a QuantScheme (format + granularity + transform).

    This is the primary entry point for tensor-level quantization.

    Args:
        x: Input tensor.
        scheme: QuantScheme specifying format, granularity, transform, and round_mode.
            If None, input is returned unchanged (no quantization path).
        allow_denorm: If False, flush subnormal values to zero (float formats only).
        scale: Optional pre-computed scale tensor.  If provided, the format
            uses this scale instead of computing one from ``x``.  Only
            meaningful for PER_CHANNEL and PER_BLOCK granularity.

    Returns:
        Quantized tensor with same shape as x.
    """
    if scheme is None:
        return x
    x_t = scheme.transform.forward(x)
    x_q = scheme.format.quantize(x_t, scheme.granularity, scheme.round_mode,
                                  allow_denorm=allow_denorm, scale=scale,
                                  scale_format=scheme.scale_format)
    return scheme.transform.inverse(x_q)

