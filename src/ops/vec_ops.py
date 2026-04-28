"""
Quantized vector operations — scheme-driven equivalents of mx/vector_ops.py.

Each operation computes the result then applies elemwise quantization via the
provided QuantScheme. When scheme=None, the result is returned unquantized
(equivalent to mx's quantize_elemwise_op with bfloat=0, fp=0).

These are building blocks for Norm, Activation, Softmax, and Pool operators
that need per-step quantization.
"""
import numpy as np
import torch

from src.quantize import quantize
from src.scheme.quant_scheme import QuantScheme

_torch_exp = torch.exp
_torch_exp2 = torch.exp2
_torch_sqrt = torch.sqrt
_torch_tanh = torch.tanh

LN_2_EXACT = 0.69314718056
LOG2_E_BF16 = 1.4453125  # 1 + 2**-2 + 2**-3 + 2**-4 + 2**-7


def _q(x: torch.Tensor, scheme) -> torch.Tensor:
    """Quantize if scheme is provided, else pass through."""
    if scheme is None:
        return x
    return quantize(x, scheme)


def vec_quantize(x: torch.Tensor, scheme) -> torch.Tensor:
    return _q(x, scheme)


def vec_add(a: torch.Tensor, b: torch.Tensor, scheme) -> torch.Tensor:
    return _q(a + b, scheme)


def vec_sub(a: torch.Tensor, b: torch.Tensor, scheme) -> torch.Tensor:
    return _q(a - b, scheme)


def vec_mul(a: torch.Tensor, b: torch.Tensor, scheme) -> torch.Tensor:
    return _q(a * b, scheme)


def vec_div(a: torch.Tensor, b: torch.Tensor, scheme,
            use_recip: bool = False) -> torch.Tensor:
    if use_recip:
        recip_b = vec_recip(b, scheme)
        return vec_mul(a, recip_b, scheme)
    return _q(a / b, scheme)


def vec_recip(x: torch.Tensor, scheme) -> torch.Tensor:
    return _q(1.0 / x, scheme)


def vec_sqrt(x: torch.Tensor, scheme) -> torch.Tensor:
    return _q(_torch_sqrt(x), scheme)


def vec_exp(x: torch.Tensor, scheme, use_exp2: bool = False) -> torch.Tensor:
    if use_exp2:
        phi = _q(LOG2_E_BF16 * x, scheme)
        return vec_exp2(phi, scheme)
    return _q(_torch_exp(x), scheme)


def vec_exp2(x: torch.Tensor, scheme) -> torch.Tensor:
    if hasattr(torch, 'exp2'):
        return _q(_torch_exp2(x), scheme)
    return _q(_torch_exp(x * LN_2_EXACT), scheme)


def vec_tanh(x: torch.Tensor, scheme) -> torch.Tensor:
    return _q(_torch_tanh(x), scheme)


def vec_reduce_sum(x: torch.Tensor, dim, keepdim: bool = False,
                   scheme=None) -> torch.Tensor:
    return _q(x.sum(dim, keepdim=keepdim), scheme)


def vec_reduce_mean(x: torch.Tensor, dim, keepdim: bool = False,
                    scheme=None) -> torch.Tensor:
    dim = dim if isinstance(dim, list) else [dim]
    denom = int(np.prod([x.shape[i] for i in dim]))
    s = vec_reduce_sum(x, dim, keepdim=keepdim, scheme=scheme)
    return vec_div(s, denom, scheme)
