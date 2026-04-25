"""
Quantized Softmax operator — inner_scheme-driven, bit-exact equivalent to mx/softmax.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scheme.quant_scheme import QuantScheme
from src.analysis.mixin import ObservableMixin
from src.ops.vec_ops import (
    vec_quantize, vec_sub, vec_mul, vec_div,
    vec_exp, vec_exp2, vec_reduce_sum,
)

_f_softmax = F.softmax
LN_2_BF16 = 0.69140625  # ln(2) in bfloat16 precision


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, inner_scheme, softmax_exp2=False,
                quantize_backprop=True, name=None):
        if dim < 0:
            dim = dim + len(input.shape)
        ctx.dim = dim
        ctx.softmax_exp2 = softmax_exp2
        ctx.name = name

        input = vec_quantize(input, inner_scheme)

        # compute max
        max_data, _ = input.max(dim, keepdim=True)

        # subtraction
        input = vec_sub(input, max_data, inner_scheme)

        # exponentiation
        if softmax_exp2:
            output = vec_exp2(input, inner_scheme)
        else:
            output = vec_exp(input, inner_scheme)

        # sum
        output_sum = vec_reduce_sum(output, dim, keepdim=True, scheme=inner_scheme)

        # divide
        output = vec_div(output, output_sum, inner_scheme)

        ctx.save_for_backward(output)
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        scheme = ctx.inner_scheme_bw

        grad_output = vec_quantize(grad_output, scheme)

        # dot product
        grad_input = vec_mul(grad_output, output, scheme)
        # sum
        grad_input = vec_reduce_sum(grad_input, ctx.dim, keepdim=True, scheme=scheme)
        # subtraction
        grad_input = vec_sub(grad_output, grad_input, scheme)
        # elementwise multiplication
        grad_input = vec_mul(output, grad_input, scheme)

        # Adjust for exp2 constant
        if ctx.softmax_exp2:
            grad_input = vec_mul(grad_input, LN_2_BF16, scheme)

        return (grad_input, None, None, None, None, None)


class QuantizedSoftmax(ObservableMixin, nn.Softmax):
    def __init__(self, dim=None, inner_scheme: QuantScheme = None,
                 softmax_exp2: bool = False,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__(dim)
        self.inner_scheme = inner_scheme
        self.softmax_exp2 = softmax_exp2
        self.quantize_backprop = quantize_backprop
        self._analysis_name = name

    def forward(self, input):
        if self.inner_scheme is None:
            return super().forward(input)
        return SoftmaxFunction.apply(
            input, self.dim, self.inner_scheme, self.softmax_exp2,
            self.quantize_backprop, self._analysis_name,
        )
