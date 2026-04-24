"""
Non-differentiable vector quantization operations.

Primary API: vec_quantize(input, scheme, ...) — QuantScheme-driven.
Compat API:  vec_quantize(input, mx_specs=..., ...) — MxSpecs wrapper (P2F-6 removes).
"""
import numpy as np
import torch

from src.quantize.elemwise import quantize_elemwise_op, quantize
from src.scheme.quant_scheme import QuantScheme

torch_exp = torch.exp
torch_exp2 = torch.exp2
torch_sqrt = torch.sqrt
torch_tanh = torch.tanh

LN_2_EXACT = 0.69314718056
LOG2_E_BF16 = 1.4453125  # 1 + 2**-2 + 2**-3 + 2**-4 + 2**-7


def _dispatch_quantize(input, scheme=None, mx_specs=None, round_mode=None):
    """Internal: dispatch to quantize() or quantize_elemwise_op()."""
    if scheme is not None:
        if mx_specs is not None:
            raise TypeError(
                "Cannot specify both scheme and mx_specs. "
                "Use scheme for QuantScheme-driven API, or mx_specs for compat."
            )
        if round_mode is not None:
            raise TypeError(
                "round_mode is ignored when scheme is provided. "
                "Set round_mode in the QuantScheme instead."
            )
        return quantize(input, scheme)
    return quantize_elemwise_op(input, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Quantize
# -------------------------------------------------------------------------

def vec_quantize(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(input, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Arithmetic ops
# -------------------------------------------------------------------------

def vec_add(a, b, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(a + b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_sub(a, b, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(a - b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_mul(a, b, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(a * b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_div(a, b, scheme=None, mx_specs=None, round_mode=None, use_recip=False):
    if not use_recip and mx_specs and mx_specs.get('vec_use_recip'):
        use_recip = True
    if use_recip:
        recip_b = vec_recip(b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
        return vec_mul(a, recip_b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    return _dispatch_quantize(a / b, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Special math ops
# -------------------------------------------------------------------------

def vec_exp(input, scheme=None, mx_specs=None, round_mode=None, use_exp2=False):
    if not use_exp2 and mx_specs and mx_specs.get('vec_use_exp2'):
        use_exp2 = True
    if use_exp2:
        phi = _dispatch_quantize(LOG2_E_BF16 * input, scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
        phi = vec_exp2(phi, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    else:
        phi = _dispatch_quantize(torch_exp(input), scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
    return phi


def vec_exp2(input, scheme=None, mx_specs=None, round_mode=None):
    if hasattr(torch, 'exp2'):
        phi = _dispatch_quantize(torch_exp2(input), scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
    else:
        phi = _dispatch_quantize(torch_exp(input * LN_2_EXACT), scheme=scheme,
                                 mx_specs=mx_specs, round_mode=round_mode)
    return phi


def vec_recip(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(1. / input, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_sqrt(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(torch_sqrt(input), scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_tanh(input, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(torch_tanh(input), scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Reduce ops
# -------------------------------------------------------------------------

def vec_reduce_sum(input, dim, keepdim=False, scheme=None, mx_specs=None, round_mode=None):
    return _dispatch_quantize(input.sum(dim, keepdim=keepdim),
                              scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)


def vec_reduce_mean(input, dim, keepdim=False, scheme=None, mx_specs=None, round_mode=None):
    dim = dim if type(dim) is list else [dim]
    denom = np.prod([input.shape[i] for i in dim])

    s = vec_reduce_sum(input, dim, keepdim=keepdim,
                       scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    s = vec_div(s, denom, scheme=scheme, mx_specs=mx_specs, round_mode=round_mode)
    return s
