"""
Non-differentiable vector quantization operations.

Primary API: vec_quantize(input, scheme, ...) — QuantScheme-driven.
"""
import numpy as np
import torch

from src.quantize.elemwise import quantize

torch_exp = torch.exp
torch_exp2 = torch.exp2
torch_sqrt = torch.sqrt
torch_tanh = torch.tanh

LN_2_EXACT = 0.69314718056
LOG2_E_BF16 = 1.4453125  # 1 + 2**-2 + 2**-3 + 2**-4 + 2**-7


def _dispatch_quantize(input, scheme=None):
    """Internal: dispatch to quantize()."""
    return quantize(input, scheme)


# -------------------------------------------------------------------------
# Quantize
# -------------------------------------------------------------------------

def vec_quantize(input, scheme=None):
    return _dispatch_quantize(input, scheme=scheme)


# -------------------------------------------------------------------------
# Arithmetic ops
# -------------------------------------------------------------------------

def vec_add(a, b, scheme=None):
    return _dispatch_quantize(a + b, scheme=scheme)


def vec_sub(a, b, scheme=None):
    return _dispatch_quantize(a - b, scheme=scheme)


def vec_mul(a, b, scheme=None):
    return _dispatch_quantize(a * b, scheme=scheme)


def vec_div(a, b, scheme=None, use_recip=False):
    if use_recip:
        recip_b = vec_recip(b, scheme=scheme)
        return vec_mul(a, recip_b, scheme=scheme)
    return _dispatch_quantize(a / b, scheme=scheme)


# -------------------------------------------------------------------------
# Special math ops
# -------------------------------------------------------------------------

def vec_exp(input, scheme=None, use_exp2=False):
    if use_exp2:
        phi = _dispatch_quantize(LOG2_E_BF16 * input, scheme=scheme)
        phi = vec_exp2(phi, scheme=scheme)
    else:
        phi = _dispatch_quantize(torch_exp(input), scheme=scheme)
    return phi


def vec_exp2(input, scheme=None):
    if hasattr(torch, 'exp2'):
        phi = _dispatch_quantize(torch_exp2(input), scheme=scheme)
    else:
        phi = _dispatch_quantize(torch_exp(input * LN_2_EXACT), scheme=scheme)
    return phi


def vec_recip(input, scheme=None):
    return _dispatch_quantize(1. / input, scheme=scheme)


def vec_sqrt(input, scheme=None):
    return _dispatch_quantize(torch_sqrt(input), scheme=scheme)


def vec_tanh(input, scheme=None):
    return _dispatch_quantize(torch_tanh(input), scheme=scheme)


# -------------------------------------------------------------------------
# Reduce ops
# -------------------------------------------------------------------------

def vec_reduce_sum(input, dim, keepdim=False, scheme=None):
    return _dispatch_quantize(input.sum(dim, keepdim=keepdim), scheme=scheme)


def vec_reduce_mean(input, dim, keepdim=False, scheme=None):
    dim = dim if type(dim) is list else [dim]
    denom = np.prod([input.shape[i] for i in dim])

    s = vec_reduce_sum(input, dim, keepdim=keepdim, scheme=scheme)
    s = vec_div(s, denom, scheme=scheme)
    return s
