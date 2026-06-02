"""
Element-wise quantization.

Primary API: quantize(x, scheme) — QuantScheme-driven unified entry.

Low-level primitives (_quantize_elemwise_core, _round_mantissa, _safe_lshift,
_safe_rshift) are defined in src.formats._core and re-exported here for
backward compatibility with tests.
"""
import threading

import torch
from src.formats.base import FormatBase
from src.formats._core import (
    _elemwise_core,
    _round_mantissa,
    _safe_lshift,
    _safe_rshift,
)

# Thread-local guard preventing Tensor method patches from re-entering
# quantize internals (e.g. torch.abs + add in _elemwise_core would
# otherwise trigger SIMDAdd → quantize → ... → infinite recursion).
_quant_guard = threading.local()


def _enter_quantize():
    _quant_guard.depth = getattr(_quant_guard, 'depth', 0) + 1


def _exit_quantize():
    _quant_guard.depth -= 1


def _is_in_quantize() -> bool:
    return getattr(_quant_guard, 'depth', 0) > 0

# Backward compatibility alias for tests
_quantize_elemwise_core = _elemwise_core


# ---------------------------------------------------------------------------
# QuantScheme-driven unified entry point
# ---------------------------------------------------------------------------

def quantize(x, scheme=None, allow_denorm=True, scale=None, mask=None, scale_o=None,
             group_mask=None, importance=None, sq_sparsity=None,
             sq_activation_mask=None):
    """Quantize tensor x using a QuantScheme (format + granularity + transform).

    This is the primary entry point for tensor-level quantization.

    Args:
        x: Input tensor.
        scheme: QuantScheme specifying format, granularity, transform, and round_mode.
            If None, input is returned unchanged (no quantization path).
        allow_denorm: If False, flush subnormal values to zero (float formats only).
        scale: Optional pre-computed scale tensor (normal-group amax when sparse).
        mask: Optional pre-computed boolean mask for static sparse. True = outlier.
        scale_o: Optional pre-computed scale for outlier group (static sparse).
        group_mask: Optional pre-computed per-group boolean mask (True = H) for
                    static group sparse.

    Returns:
        Quantized tensor with same shape as x.
    """
    if scheme is None:
        return x
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"quantize() expects a Tensor, got {type(x).__name__}: {x!r}"
        )
    # Set re-entrancy guard so Tensor dunder patches don't intercept
    # internal tensor arithmetic inside format/granularity logic.
    _enter_quantize()
    try:
        x_t = scheme.transform.forward(x)
        x_q = scheme.format.quantize(x_t, scheme.granularity, scheme.round_mode,
                                      allow_denorm=allow_denorm, scale=scale,
                                      scale_storage=scheme.scale_storage,
                                      mask=mask, scale_o=scale_o,
                                      outlier_format=scheme.outlier_format,
                                      group_format=scheme.group_format,
                                      group_ratio=scheme.group_ratio,
                                      group_mask=group_mask,
                                      importance=importance,
                                      sq_sparsity=sq_sparsity if sq_sparsity is not None else scheme.sq_sparsity,
                                      sq_activation_mask=sq_activation_mask)
        return scheme.transform.inverse(x_q)
    finally:
        _exit_quantize()

