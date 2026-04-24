"""
Element-wise quantization: round_mantissa, quantize_elemwise_core,
quantize_elemwise_op, _quantize_bfloat, _quantize_fp.

Rewritten from mx/elemwise_ops.py. Key changes:
- Uses src/formats/base.py instead of mx/formats.py
- Parameter 'round' renamed to 'round_mode'
- Uses compute_min_norm/compute_max_norm from src/formats/base.py
- Custom CUDA path omitted (Python/CPU path only for now)
"""
import torch
from src.formats.base import compute_min_norm, compute_max_norm

_VALID_ROUND_MODES = {"nearest", "floor", "even", "dither"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def round_mantissa(A, bits, round_mode, clamp=False):
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

def quantize_elemwise_core(A, bits, exp_bits, max_norm, round_mode='nearest',
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
    out = safe_lshift(out, bits - 2, private_exp)

    out = round_mantissa(out, bits, round_mode, clamp=False)

    # Undo scaling
    out = safe_rshift(out, bits - 2, private_exp)

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
        output = torch.sparse_coo_tensor(sparse_A.indices(), output,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)
        return output

    return out


# ---------------------------------------------------------------------------
# Format-specific wrappers
# ---------------------------------------------------------------------------

def _quantize_bfloat(A, bfloat, round_mode='nearest', allow_denorm=True):
    if bfloat == 0 or bfloat == 32:
        return A

    max_norm = compute_max_norm(8, bfloat - 7)

    return quantize_elemwise_core(
            A, bits=bfloat-7, exp_bits=8, max_norm=max_norm, round_mode=round_mode,
            allow_denorm=allow_denorm)


def _quantize_fp(A, exp_bits=None, mantissa_bits=None,
                 round_mode='nearest', allow_denorm=True):
    if exp_bits is None or mantissa_bits is None:
        return A

    max_norm = compute_max_norm(exp_bits, mantissa_bits + 2)

    return quantize_elemwise_core(
            A, bits=mantissa_bits + 2, exp_bits=exp_bits,
            max_norm=max_norm, round_mode=round_mode, allow_denorm=allow_denorm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_elemwise_op(A, mx_specs, round_mode=None):
    if mx_specs is None:
        return A
    elif round_mode is None:
        round_mode = mx_specs['round']

    if mx_specs['bfloat'] == 16 and round_mode == 'even'\
        and torch.cuda.is_bf16_supported() \
        and mx_specs['bfloat_subnorms'] == True:
        return A.to(torch.bfloat16)

    if mx_specs['bfloat'] > 0 and mx_specs['fp'] > 0:
        raise ValueError("Cannot set both [bfloat] and [fp] in mx_specs.")
    elif mx_specs['bfloat'] > 9:
        A = _quantize_bfloat(A, bfloat=mx_specs['bfloat'], round_mode=round_mode,
                             allow_denorm=mx_specs['bfloat_subnorms'])
    elif mx_specs['bfloat'] > 0 and mx_specs['bfloat'] <= 9:
        raise ValueError("Cannot set [bfloat] <= 9 in mx_specs.")
    elif mx_specs['fp'] > 6:
        A = _quantize_fp(A, exp_bits=5, mantissa_bits=mx_specs['fp'] - 6,
                         round_mode=round_mode, allow_denorm=mx_specs['bfloat_subnorms'])
    elif mx_specs['fp'] > 0 and mx_specs['fp'] <= 6:
        raise ValueError("Cannot set [fp] <= 6 in mx_specs.")
    return A
