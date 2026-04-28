"""
Element-wise quantization.

Primary API: quantize(x, scheme) — QuantScheme-driven unified entry.

Internal helpers: _quantize_elemwise_core, _round_mantissa, _safe_lshift, _safe_rshift.
Legacy wrappers (kept for internal equivalence tests):
  _quantize_elemwise, _quantize_bfloat, _quantize_fp
"""
import torch
from src.formats.base import compute_min_norm, compute_max_norm, FormatBase


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def _round_mantissa(A, bits, round_mode, clamp=False):
    if round_mode == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round_mode == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round_mode == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round_mode == "even":
        absA = torch.abs(A)
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise ValueError(f"Unrecognized round_mode {round_mode!r}")

    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------

def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round_mode='nearest',
                            saturate_normals=False, allow_denorm=True):
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")
        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = compute_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))

        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round_mode, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    # handle Inf/NaN
    out[A == float("Inf")] = float("Inf")
    out[A == -float("Inf")] = -float("Inf")
    out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        # Fixed: old code had UnboundLocalError here (used 'output' instead of 'out')
        out = torch.sparse_coo_tensor(sparse_A.indices(), out,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)
        return out

    return out


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
                                  allow_denorm=allow_denorm, scale=scale)
    return scheme.transform.inverse(x_q)

