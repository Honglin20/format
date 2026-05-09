"""Low-level format quantization primitives shared by formats/ and quantize/.

These are pure numerical operations: given bits/ebits/max_norm, they quantize.
No dependency on FormatBase, QuantScheme, or any quantize/ module.
"""
import torch


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


def _elemwise_core(A, bits, exp_bits, max_norm, round_mode='nearest',
                   saturate_normals=False, allow_denorm=True):
    """Element-wise quantization to a given number representation.

    Pure function: no dependency on FormatBase or QuantScheme.
    Callers are responsible for providing bits/ebits/max_norm.
    """
    from src.formats.base import compute_min_norm

    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")
        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

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

    out = _safe_lshift(out, bits - 2, private_exp)
    out = _round_mantissa(out, bits, round_mode, clamp=False)
    out = _safe_rshift(out, bits - 2, private_exp)

    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                          torch.sign(out) * float("Inf"), out)

    out[A == float("Inf")] = float("Inf")
    out[A == -float("Inf")] = -float("Inf")
    out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        out = torch.sparse_coo_tensor(sparse_A.indices(), out,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)
        return out

    return out
