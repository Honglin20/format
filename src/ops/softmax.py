"""
Quantized Softmax operator — inner_scheme-driven, bit-exact equivalent to mx/softmax.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.analysis.mixin import ObservableMixin
from src.quantize import quantize
from src.ops.vec_ops import (
    vec_quantize, vec_sub, vec_mul, vec_div,
    vec_exp, vec_exp2, vec_reduce_sum,
)

_f_softmax = F.softmax
LN_2_BF16 = 0.69140625  # ln(2) in bfloat16 precision


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, inner_scheme, softmax_exp2=False,
                quantize_backprop=True, name=None, emit_fn=None):
        if dim < 0:
            dim = dim + len(input.shape)
        ctx.dim = dim
        ctx.softmax_exp2 = softmax_exp2
        ctx.name = name
        ctx.emit_fn = emit_fn

        fp_in = input
        input = vec_quantize(input, inner_scheme)
        if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, input, inner_scheme)

        max_data, _ = input.max(dim, keepdim=True)
        input = vec_sub(input, max_data, inner_scheme)

        if softmax_exp2:
            output = vec_exp2(input, inner_scheme)
        else:
            output = vec_exp(input, inner_scheme)

        output_sum = vec_reduce_sum(output, dim, keepdim=True, scheme=inner_scheme)
        output = vec_div(output, output_sum, inner_scheme)

        ctx.save_for_backward(output)
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        scheme = ctx.inner_scheme_bw
        emit_fn = ctx.emit_fn

        fp_go = grad_output
        grad_output = vec_quantize(grad_output, scheme)
        if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_output, scheme)

        grad_input = vec_mul(grad_output, output, scheme)
        grad_input = vec_reduce_sum(grad_input, ctx.dim, keepdim=True, scheme=scheme)
        grad_input = vec_sub(grad_output, grad_input, scheme)
        grad_input = vec_mul(output, grad_input, scheme)

        if ctx.softmax_exp2:
            grad_input = vec_mul(grad_input, LN_2_BF16, scheme)

        return (grad_input, None, None, None, None, None, None)


class QuantizedSoftmax(ObservableMixin, nn.Softmax):
    def __init__(self, dim=None, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 softmax_exp2: bool = False,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__(dim)
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            bw = inner_scheme if quantize_backprop else None
            cfg = OpQuantConfig(input=inner_scheme, grad_input=bw)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self.softmax_exp2 = softmax_exp2
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input
        quantize_backprop = self.cfg.grad_input is not None
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        if self.cfg.storage is not None:
            input = quantize(input, self.cfg.storage)
        result = SoftmaxFunction.apply(
            input, self.dim, inner_scheme, self.softmax_exp2,
            quantize_backprop, self._analysis_name, emit_fn,
        )
        if self.cfg.storage is not None:
            result = quantize(result, self.cfg.storage)
        return result
