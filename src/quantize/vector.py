"""
Non-differentiable vector quantization operations.

Rewritten from mx/vector_ops.py. Key changes:
- Parameter 'round' renamed to 'round_mode'
- Uses src.quantize.elemwise.quantize_elemwise_op
"""
import numpy as np
import torch

from src.quantize.elemwise import quantize_elemwise_op

torch_exp = torch.exp
torch_exp2 = torch.exp2
torch_sqrt = torch.sqrt
torch_tanh = torch.tanh

LN_2_EXACT = 0.69314718056
LOG2_E_BF16 = 1.4453125  # 1 + 2**-2 + 2**-3 + 2**-4 + 2**-7


def vec_quantize(input, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(input, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Arithmetic ops
# -------------------------------------------------------------------------

def vec_add(a, b, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(a + b, mx_specs=mx_specs, round_mode=round_mode)


def vec_sub(a, b, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(a - b, mx_specs=mx_specs, round_mode=round_mode)


def vec_mul(a, b, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(a * b, mx_specs=mx_specs, round_mode=round_mode)


def vec_div(a, b, mx_specs=None, round_mode=None):
    if mx_specs and mx_specs['vec_use_recip']:
        recip_b = vec_recip(b, mx_specs=mx_specs, round_mode=round_mode)
        return vec_mul(a, recip_b, mx_specs=mx_specs, round_mode=round_mode)
    else:
        return quantize_elemwise_op(a / b, mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Special math ops
# -------------------------------------------------------------------------

def vec_exp(input, mx_specs=None, round_mode=None):
    if mx_specs and mx_specs['vec_use_exp2']:
        phi = quantize_elemwise_op(LOG2_E_BF16 * input,
                                   mx_specs=mx_specs, round_mode=round_mode)
        phi = vec_exp2(phi, mx_specs=mx_specs, round_mode=round_mode)
    else:
        phi = quantize_elemwise_op(torch_exp(input),
                                   mx_specs=mx_specs, round_mode=round_mode)
    return phi


def vec_exp2(input, mx_specs=None, round_mode=None):
    if hasattr(torch, 'exp2'):
        phi = quantize_elemwise_op(torch_exp2(input),
                                   mx_specs=mx_specs, round_mode=round_mode)
    else:
        phi = quantize_elemwise_op(torch_exp(input * LN_2_EXACT),
                                   mx_specs=mx_specs, round_mode=round_mode)
    return phi


def vec_recip(input, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(1. / input, mx_specs=mx_specs, round_mode=round_mode)


def vec_sqrt(input, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(torch_sqrt(input), mx_specs=mx_specs, round_mode=round_mode)


def vec_tanh(input, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(torch_tanh(input), mx_specs=mx_specs, round_mode=round_mode)


# -------------------------------------------------------------------------
# Reduce ops
# -------------------------------------------------------------------------

def vec_reduce_sum(input, dim, keepdim=False, mx_specs=None, round_mode=None):
    return quantize_elemwise_op(input.sum(dim, keepdim=keepdim),
                                mx_specs=mx_specs, round_mode=round_mode)


def vec_reduce_mean(input, dim, keepdim=False, mx_specs=None, round_mode=None):
    dim = dim if type(dim) is list else [dim]
    denom = np.prod([input.shape[i] for i in dim])

    s = vec_reduce_sum(input, dim, keepdim=keepdim,
                       mx_specs=mx_specs, round_mode=round_mode)
    s = vec_div(s, denom, mx_specs=mx_specs, round_mode=round_mode)
    return s
