"""
Quantized SIMD elementwise operators — inner_scheme-driven.

Bit-exact equivalent to mx/simd_ops.py. Each SIMD class is an
autograd.Function with forward + QAT backward using vec_* helpers.

Public API:
    simd_add, simd_sub, simd_mul, simd_div
    simd_split, simd_square, simd_sqrt, simd_exp, simd_log
    simd_reduce_sum, simd_reduce_mean, simd_norm
"""
import math

import torch
import torch.nn.functional as F

from src.ops.vec_ops import (
    vec_quantize, vec_add, vec_sub, vec_mul, vec_div,
    vec_exp, vec_sqrt,
)

_torch_sum = torch.sum
_torch_mean = torch.mean
_torch_sqrt = torch.sqrt
_torch_exp = torch.exp
_torch_log = torch.log
_torch_square = torch.square


# ---------------------------------------------------------------------------
# Broadcast gradient helper
# ---------------------------------------------------------------------------

def _broadcast_gradient(grad_out, in_shape, scheme=None):
    """Compute gradient of broadcast operation with quantization.

    Matches mx/simd_ops.py::_broadcast_gradient exactly.
    """
    if list(grad_out.shape) == in_shape:
        return grad_out

    assert grad_out.ndim >= len(in_shape)

    reduce_dims = []
    for i in range(grad_out.ndim):
        if i + 1 > len(in_shape):
            reduce_dims.append(-1 - i)
            continue

        dout = grad_out.shape[-1 - i]
        din = in_shape[-1 - i]

        if dout == din:
            pass
        elif din == 1:
            reduce_dims.append(-1 - i)
        else:
            raise ValueError(
                f"simd gradient shape error. grad_out shape {grad_out.shape} "
                f"and input shape {in_shape}"
            )

    if len(reduce_dims) > 0:
        grad_out = vec_quantize(grad_out, scheme=scheme)
        grad_in = _torch_sum(grad_out, dim=reduce_dims)
        grad_in = vec_quantize(grad_in, scheme=scheme)
        return grad_in.view(in_shape)
    else:
        return grad_out.view(in_shape)


# ---------------------------------------------------------------------------
# SIMD Add
# ---------------------------------------------------------------------------

class SIMDAdd(torch.autograd.Function):
    """Fwd: y = x1 + x2   Bwd: dy/dx1 = dy/dx2 = 1"""

    @staticmethod
    def forward(ctx, in1, in2, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        qin1 = vec_quantize(in1, scheme=inner_scheme)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)
            qin2 = vec_quantize(in2, scheme=inner_scheme)
        else:
            ctx.in2_const = True
            qin2 = in2

        return vec_add(qin1, qin2, scheme=inner_scheme)

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        g = vec_quantize(g, scheme=scheme)

        if not ctx.in2_const:
            g1 = _broadcast_gradient(g, ctx.in1_shape, scheme)
            g2 = _broadcast_gradient(g, ctx.in2_shape, scheme)
            return (g1, g2, None, None)
        else:
            return (g, None, None, None)

    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Add", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out


# ---------------------------------------------------------------------------
# SIMD Sub
# ---------------------------------------------------------------------------

class SIMDSub(torch.autograd.Function):
    """Fwd: y = x1 - x2   Bwd: dy/dx1 = 1, dy/dx2 = -1"""

    @staticmethod
    def forward(ctx, in1, in2, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        qin1 = vec_quantize(in1, scheme=inner_scheme)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)
            qin2 = vec_quantize(in2, scheme=inner_scheme)
        else:
            ctx.in2_const = True
            qin2 = in2

        return vec_sub(qin1, qin2, scheme=inner_scheme)

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        if not ctx.in2_const:
            n_g = vec_quantize(-g, scheme=scheme)
            g1 = _broadcast_gradient(g, ctx.in1_shape, scheme)
            g2 = _broadcast_gradient(n_g, ctx.in2_shape, scheme)
            return (g1, g2, None, None)
        else:
            return (g, None, None, None)

    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Sub", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out


# ---------------------------------------------------------------------------
# SIMD Mul
# ---------------------------------------------------------------------------

class SIMDMul(torch.autograd.Function):
    """Fwd: y = x1 * x2   Bwd: dy/dx1 = x2, dy/dx2 = x1"""

    @staticmethod
    def forward(ctx, in1, in2, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        qin1 = vec_quantize(in1, scheme=inner_scheme)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)
            qin2 = vec_quantize(in2, scheme=inner_scheme)

            if quantize_backprop:
                ctx.save_for_backward(qin1, qin2)
            else:
                ctx.save_for_backward(in1, in2)
        else:
            ctx.in2_const = True
            ctx.in2 = in2
            qin2 = in2

            if quantize_backprop:
                ctx.save_for_backward(qin1)
            else:
                ctx.save_for_backward(in1)

        return vec_mul(qin1, qin2, scheme=inner_scheme)

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        g = vec_quantize(g, scheme=scheme)

        if not ctx.in2_const:
            in1, in2 = ctx.saved_tensors
            g1 = vec_mul(g, in2, scheme=scheme)
            g2 = vec_mul(g, in1, scheme=scheme)
            g1 = _broadcast_gradient(g1, ctx.in1_shape, scheme)
            g2 = _broadcast_gradient(g2, ctx.in2_shape, scheme)
            return (g1, g2, None, None)
        else:
            in1, = ctx.saved_tensors
            g1 = vec_mul(g, ctx.in2, scheme=scheme)
            return (g1, None, None, None)

    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Mul", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out


# ---------------------------------------------------------------------------
# SIMD Div
# ---------------------------------------------------------------------------

class SIMDDiv(torch.autograd.Function):
    """Fwd: y = x1 / x2   Bwd: dy/dx1 = 1/x2, dy/dx2 = -x1/(x2^2)"""

    @staticmethod
    def forward(ctx, in1, in2, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        qin1 = vec_quantize(in1, scheme=inner_scheme)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)
            qin2 = vec_quantize(in2, scheme=inner_scheme)
            out = vec_div(qin1, qin2, scheme=inner_scheme)

            if quantize_backprop:
                ctx.save_for_backward(out, qin2)
            else:
                ctx.save_for_backward(out, in2)
        else:
            ctx.in2_const = True
            ctx.in2 = in2
            out = vec_div(qin1, in2, scheme=inner_scheme)

            if quantize_backprop:
                ctx.save_for_backward(qin1)
            else:
                ctx.save_for_backward(in1)

        return out

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        g = vec_quantize(g, scheme=scheme)

        if not ctx.in2_const:
            out, in2 = ctx.saved_tensors
            g1 = vec_div(g, in2, scheme=scheme)
            g2 = vec_div(-out, in2, scheme=scheme)
            g2 = vec_mul(g, g2, scheme=scheme)

            g1 = _broadcast_gradient(g1, ctx.in1_shape, scheme)
            g2 = _broadcast_gradient(g2, ctx.in2_shape, scheme)
            return (g1, g2, None, None)
        else:
            in1, = ctx.saved_tensors
            g1 = vec_div(g, ctx.in2, scheme=scheme)
            return (g1, None, None, None)

    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Div", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out


# ---------------------------------------------------------------------------
# SIMD Split
# ---------------------------------------------------------------------------

class SIMDSplit(torch.autograd.Function):
    """Fwd: x1, x2 = x   Bwd: dy/dx = dy/dx1 + dy/dx2"""

    @staticmethod
    def forward(ctx, in1, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        ctx.set_materialize_grads(False)
        return in1.clone(), in1.clone()

    @staticmethod
    def backward(ctx, g1, g2):
        scheme = ctx.inner_scheme_bw
        if g1 is None:
            return (g2, None, None)
        if g2 is None:
            return (g1, None, None)

        g1 = vec_quantize(g1, scheme=scheme)
        g2 = vec_quantize(g2, scheme=scheme)
        grad_in = vec_add(g1, g2, scheme=scheme)
        return (grad_in, None, None)


# ---------------------------------------------------------------------------
# SIMD Square
# ---------------------------------------------------------------------------

class SIMDSquare(torch.autograd.Function):
    """Fwd: y = x**2   Bwd: dy/dx = 2*x"""

    @staticmethod
    def forward(ctx, in1, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        qin1 = vec_quantize(in1, scheme=inner_scheme)

        if quantize_backprop:
            ctx.save_for_backward(qin1)
        else:
            ctx.save_for_backward(in1)

        return vec_quantize(qin1 ** 2, scheme=inner_scheme)

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        x, = ctx.saved_tensors
        g = vec_quantize(g, scheme=scheme)
        x2 = vec_mul(x, 2, scheme=scheme)
        grad_in = vec_mul(g, x2, scheme=scheme)
        return (grad_in, None, None)


# ---------------------------------------------------------------------------
# SIMD Sqrt
# ---------------------------------------------------------------------------

class SIMDSqrt(torch.autograd.Function):
    """Fwd: y = sqrt(x)   Bwd: dy/dx = 0.5/sqrt(x)"""

    @staticmethod
    def forward(ctx, in1, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        in1 = vec_quantize(in1, scheme=inner_scheme)
        out = vec_sqrt(in1, scheme=inner_scheme)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        sqrt_x, = ctx.saved_tensors
        g = vec_quantize(g, scheme=scheme)
        g = vec_mul(g, 0.5, scheme=scheme)
        grad_in = vec_div(g, sqrt_x, scheme=scheme)
        return (grad_in, None, None)


# ---------------------------------------------------------------------------
# SIMD Exp
# ---------------------------------------------------------------------------

class SIMDExp(torch.autograd.Function):
    """Fwd: y = e^x   Bwd: dy/dx = e^x"""

    @staticmethod
    def forward(ctx, in1, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        in1 = vec_quantize(in1, scheme=inner_scheme)
        out = vec_exp(in1, scheme=inner_scheme)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        exp_x, = ctx.saved_tensors
        g = vec_quantize(g, scheme=scheme)
        g = vec_mul(g, exp_x, scheme=scheme)
        return (g, None, None)

    @staticmethod
    def symbolic(g, in1, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
        out = g.op("Exp", in1)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out


# ---------------------------------------------------------------------------
# SIMD Log
# ---------------------------------------------------------------------------

class SIMDLog(torch.autograd.Function):
    """Fwd: y = log_e(x)   Bwd: dy/dx = 1/x"""

    @staticmethod
    def forward(ctx, in1, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        qin1 = vec_quantize(in1, scheme=inner_scheme)
        out = _torch_log(qin1)
        out = vec_quantize(out, scheme=inner_scheme)

        if quantize_backprop:
            ctx.save_for_backward(qin1)
        else:
            ctx.save_for_backward(in1)

        return out

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        x, = ctx.saved_tensors
        g = vec_quantize(g, scheme=scheme)
        g = vec_div(g, x, scheme=scheme)
        return (g, None, None)

    @staticmethod
    def symbolic(g, in1, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
        out = g.op("Log", in1)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out


# ---------------------------------------------------------------------------
# SIMD ReduceSum
# ---------------------------------------------------------------------------

class SIMDReduceSum(torch.autograd.Function):
    """Fwd: y = sum(x, dim)   Bwd: dy/dx = 1, expanded in summed dims"""

    @staticmethod
    def forward(ctx, in1, dim, keepdim=False, inner_scheme=None, quantize_backprop=True):
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        dim = [dim] if type(dim) == int else dim
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(in1.shape)

        in1 = vec_quantize(in1, scheme=inner_scheme)
        out = vec_quantize(_torch_sum(in1, dim=dim, keepdim=keepdim), scheme=inner_scheme)
        return out

    @staticmethod
    def backward(ctx, g):
        scheme = ctx.inner_scheme_bw
        ndim = len(ctx.input_shape)
        dim = ctx.dim
        dim = [(i + ndim if i < 0 else i) for i in dim]

        g = vec_quantize(g, scheme=scheme)
        if not ctx.keepdim:
            for i in sorted(dim):
                g = g.unsqueeze(i)

        expand_sizes = [-1 for _ in range(ndim)]
        for i in dim:
            expand_sizes[i] = ctx.input_shape[i]

        grad_in = g.expand(expand_sizes)
        return (grad_in, None, None, None, None)


# ---------------------------------------------------------------------------
# User-facing functional APIs
# ---------------------------------------------------------------------------

def simd_add(in1, in2, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return in1 + in2
    return SIMDAdd.apply(in1, in2, inner_scheme, quantize_backprop)


def simd_sub(in1, in2, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return in1 - in2
    return SIMDSub.apply(in1, in2, inner_scheme, quantize_backprop)


def simd_mul(in1, in2, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return in1 * in2
    return SIMDMul.apply(in1, in2, inner_scheme, quantize_backprop)


def simd_div(in1, in2, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return in1 / in2
    return SIMDDiv.apply(in1, in2, inner_scheme, quantize_backprop)


def simd_split(in1, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return in1, in1
    return SIMDSplit.apply(in1, inner_scheme, quantize_backprop)


def simd_square(in1, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return _torch_square(in1)
    return SIMDSquare.apply(in1, inner_scheme, quantize_backprop)


def simd_sqrt(in1, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return _torch_sqrt(in1)
    return SIMDSqrt.apply(in1, inner_scheme, quantize_backprop)


def simd_exp(in1, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return _torch_exp(in1)
    return SIMDExp.apply(in1, inner_scheme, quantize_backprop)


def simd_log(in1, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return _torch_log(in1)
    return SIMDLog.apply(in1, inner_scheme, quantize_backprop)


def simd_reduce_sum(in1, dim=None, keepdim=False, inner_scheme=None, quantize_backprop=True):
    if dim is None:
        dim = list(range(in1.ndim))
    if inner_scheme is None:
        return _torch_sum(in1, dim, keepdim=keepdim)
    return SIMDReduceSum.apply(in1, dim, keepdim, inner_scheme, quantize_backprop)


def simd_reduce_mean(in1, dim=None, keepdim=False, inner_scheme=None, quantize_backprop=True):
    if dim is None:
        dim = list(range(in1.ndim))
    if inner_scheme is None:
        return _torch_mean(in1, dim, keepdim=keepdim)

    dim = dim if type(dim) is list else [dim]
    denom = math.prod([in1.shape[i] for i in dim])

    s = SIMDReduceSum.apply(in1, dim, keepdim, inner_scheme, quantize_backprop)
    return SIMDMul.apply(s, 1.0 / denom, inner_scheme, quantize_backprop)


def simd_norm(in1, keepdim=False, inner_scheme=None, quantize_backprop=True):
    if inner_scheme is None:
        return torch.linalg.norm(in1, keepdim=keepdim)

    in1 = SIMDSquare.apply(in1, inner_scheme, quantize_backprop)
    s = SIMDReduceSum.apply(in1, list(range(in1.ndim)), keepdim, inner_scheme, quantize_backprop)
    return SIMDSqrt.apply(s, inner_scheme, quantize_backprop)
