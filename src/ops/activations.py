"""
Quantized activation operators: Sigmoid, Tanh, ReLU, ReLU6, LeakyReLU, SiLU, GELU.

inner_scheme-driven, bit-exact equivalent to mx/activations.py.

All activations use inner_scheme (QuantScheme or None) as the sole quantization
config. When inner_scheme is None, the activation is passthrough (no quantization).
For backward, inner_scheme_bw controls whether backward vec_ops are quantized
(None = passthrough when quantize_backprop=False).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.analysis.mixin import ObservableMixin
from src.ops.vec_ops import (
    vec_quantize, vec_add, vec_sub, vec_mul, vec_div,
    vec_exp, vec_recip, vec_tanh,
)

_torch_relu = torch.relu
_torch_relu_ = torch.relu_
_f_relu = F.relu
_f_relu6 = F.relu6
_f_leaky_relu = F.leaky_relu
_f_silu = F.silu
_f_gelu = F.gelu


# ---------------------------------------------------------------------------
# Sigmoid: 1 / (1 + exp(-x))
# ---------------------------------------------------------------------------

class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inner_scheme, quantize_backprop=True, name=None, emit_fn=None):
        ctx.name = name
        ctx.emit_fn = emit_fn

        fp_in = input
        input = vec_quantize(input, inner_scheme)
        if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, input, inner_scheme)
        exp_nx = vec_exp(-input, inner_scheme)
        exp_nx_plus_1 = vec_add(exp_nx, 1., inner_scheme)
        output = vec_recip(exp_nx_plus_1, inner_scheme)

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
        temp = vec_sub(1, output, scheme)
        grad_sigmoid = vec_mul(output, temp, scheme)
        grad_input = vec_mul(grad_sigmoid, grad_output, scheme)

        return (grad_input, None, None, None, None)


class QuantizedSigmoid(ObservableMixin, nn.Sigmoid):
    def __init__(self, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__()
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        return SigmoidFunction.apply(
            input, inner_scheme, quantize_backprop, self._analysis_name, emit_fn,
        )


# ---------------------------------------------------------------------------
# Tanh: torch.tanh, backward: 1 - tanh²
# ---------------------------------------------------------------------------

class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inner_scheme, quantize_backprop=True, name=None, emit_fn=None):
        ctx.name = name
        ctx.emit_fn = emit_fn

        fp_in = input
        input = vec_quantize(input, inner_scheme)
        if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, input, inner_scheme)
        output = vec_tanh(input, inner_scheme)

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
        output2 = vec_mul(output, output, scheme)
        grad_tanh = vec_sub(1, output2, scheme)
        grad_input = vec_mul(grad_tanh, grad_output, scheme)

        return (grad_input, None, None, None, None)


class QuantizedTanh(ObservableMixin, nn.Tanh):
    def __init__(self, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__()
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        return TanhFunction.apply(
            input, inner_scheme, quantize_backprop, self._analysis_name, emit_fn,
        )


# ---------------------------------------------------------------------------
# ReLU: torch.relu, backward: mask * grad_output
# ---------------------------------------------------------------------------

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace, inner_scheme, quantize_backprop=True, name=None, emit_fn=None):
        ctx.name = name
        ctx.emit_fn = emit_fn

        # No need to quantize input first since ReLU just masks
        if inplace:
            ctx.mark_dirty(input)
            input = _torch_relu_(input)
            fp_out = input
            output = vec_quantize(input, inner_scheme)
            if emit_fn: emit_fn("output", 0, "output_post_quant", fp_out, output, inner_scheme)
            input.copy_(output)
            output = input
        else:
            fp_out = _torch_relu(input)
            output = vec_quantize(fp_out, inner_scheme)
            if emit_fn: emit_fn("output", 0, "output_post_quant", fp_out, output, inner_scheme)

        mask = output > 0
        ctx.save_for_backward(mask)
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        scheme = ctx.inner_scheme_bw
        emit_fn = ctx.emit_fn

        zs = torch.zeros([1], dtype=grad_output.dtype, device=grad_output.device)
        grad_input = torch.where(mask, grad_output, zs)
        fp_gi = grad_input
        grad_input = vec_quantize(grad_input, scheme)
        if emit_fn: emit_fn("grad_input", 0, "grad_input_post_quant", fp_gi, grad_input, scheme)

        return (grad_input, None, None, None, None, None)


class QuantizedReLU(ObservableMixin, nn.ReLU):
    def __init__(self, inplace=False, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__(inplace=inplace)
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        return ReLUFunction.apply(
            input, self.inplace, inner_scheme,
            quantize_backprop, self._analysis_name, emit_fn,
        )


# ---------------------------------------------------------------------------
# ReLU6: torch.relu6, backward: mask * grad_output
# ---------------------------------------------------------------------------

class ReLU6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace, inner_scheme, quantize_backprop=True, name=None, emit_fn=None):
        ctx.name = name
        ctx.emit_fn = emit_fn

        if inplace:
            ctx.mark_dirty(input)
            input = _f_relu6(input, inplace=True)
            fp_out = input
            output = vec_quantize(input, inner_scheme)
            if emit_fn: emit_fn("output", 0, "output_post_quant", fp_out, output, inner_scheme)
            input.copy_(output)
            output = input
        else:
            fp_out = _f_relu6(input)
            output = vec_quantize(fp_out, inner_scheme)
            if emit_fn: emit_fn("output", 0, "output_post_quant", fp_out, output, inner_scheme)

        mask = torch.logical_and(output > 0, output < 6)
        ctx.save_for_backward(mask)
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        scheme = ctx.inner_scheme_bw
        emit_fn = ctx.emit_fn

        zs = torch.zeros([1], dtype=grad_output.dtype, device=grad_output.device)
        grad_input = torch.where(mask, grad_output, zs)
        fp_gi = grad_input
        grad_input = vec_quantize(grad_input, scheme)
        if emit_fn: emit_fn("grad_input", 0, "grad_input_post_quant", fp_gi, grad_input, scheme)

        return (grad_input, None, None, None, None, None)


class QuantizedReLU6(ObservableMixin, nn.ReLU6):
    def __init__(self, inplace=False, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__(inplace=inplace)
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        return ReLU6Function.apply(
            input, self.inplace, inner_scheme,
            quantize_backprop, self._analysis_name, emit_fn,
        )


# ---------------------------------------------------------------------------
# LeakyReLU: quantize input, leaky_relu, quantize output
# ---------------------------------------------------------------------------

class LeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, negative_slope, inplace, inner_scheme,
                quantize_backprop=True, name=None, emit_fn=None):
        ctx.negative_slope = negative_slope
        ctx.name = name
        ctx.emit_fn = emit_fn

        fp_in = input
        q_in = vec_quantize(input, inner_scheme)
        if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, q_in, inner_scheme)
        output = _f_leaky_relu(q_in, negative_slope=negative_slope)
        output = vec_quantize(output, inner_scheme)

        if inplace:
            ctx.mark_dirty(input)
            input.copy_(output)
            output = input

        mask = output > 0
        ctx.save_for_backward(mask)
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        scheme = ctx.inner_scheme_bw
        emit_fn = ctx.emit_fn

        fp_go = grad_output
        grad_output = vec_quantize(grad_output, scheme)
        if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_output, scheme)
        grad_neg = vec_mul(grad_output, ctx.negative_slope, scheme)
        grad_input = torch.where(mask, grad_output, grad_neg)

        return (grad_input, None, None, None, None, None, None)


class QuantizedLeakyReLU(ObservableMixin, nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False,
                 cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__(negative_slope=negative_slope, inplace=inplace)
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        return LeakyReLUFunction.apply(
            input, self.negative_slope, self.inplace,
            inner_scheme, quantize_backprop, self._analysis_name, emit_fn,
        )


# ---------------------------------------------------------------------------
# SiLU: x * sigmoid(x), backward: sigmoid(x) + y * (1 - sigmoid(x))
# ---------------------------------------------------------------------------

class SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace, inner_scheme, quantize_backprop=True, name=None, emit_fn=None):
        ctx.name = name
        ctx.emit_fn = emit_fn

        fp_in = input
        q_in = vec_quantize(input, inner_scheme)
        if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, q_in, inner_scheme)
        exp_nx = vec_exp(-q_in, inner_scheme)
        exp_nx_plus_1 = vec_add(exp_nx, 1., inner_scheme)
        sig_x = vec_recip(exp_nx_plus_1, inner_scheme)
        output = vec_mul(q_in, sig_x, inner_scheme)

        if inplace:
            ctx.mark_dirty(input)
            input.copy_(output)
            output = input

        ctx.save_for_backward(output, sig_x)
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y, sig_x, = ctx.saved_tensors
        scheme = ctx.inner_scheme_bw
        emit_fn = ctx.emit_fn

        fp_go = grad_output
        grad_output = vec_quantize(grad_output, scheme)
        if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_output, scheme)
        temp = vec_sub(1., sig_x, scheme)
        temp = vec_mul(y, temp, scheme)
        grad_silu = vec_add(sig_x, temp, scheme)
        grad_input = vec_mul(grad_silu, grad_output, scheme)

        return (grad_input, None, None, None, None, None)


class QuantizedSiLU(ObservableMixin, nn.SiLU):
    def __init__(self, inplace=False, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__(inplace=inplace)
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        return SiLUFunction.apply(
            input, self.inplace, inner_scheme,
            quantize_backprop, self._analysis_name, emit_fn,
        )


# ---------------------------------------------------------------------------
# GELU: x * sigmoid(y), first_order: y=1.702*x, detailed: y=1.5958*(x+0.044715*x³)
# ---------------------------------------------------------------------------

class GELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inner_scheme, first_order_gelu=False,
                quantize_backprop=True, name=None, emit_fn=None):
        ctx.first_order_gelu = first_order_gelu
        ctx.name = name
        ctx.emit_fn = emit_fn

        fp_in = input
        q_in = vec_quantize(input, inner_scheme)
        if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, q_in, inner_scheme)

        if first_order_gelu:
            sigmoid_input = vec_mul(1.703125, q_in, inner_scheme)
        else:
            sigmoid_input = vec_mul(q_in, q_in, inner_scheme)
            sigmoid_input = vec_mul(sigmoid_input, q_in, inner_scheme)
            sigmoid_input = vec_mul(0.044677734, sigmoid_input, inner_scheme)
            sigmoid_input = vec_add(sigmoid_input, q_in, inner_scheme)
            sigmoid_input = vec_mul(1.59375, sigmoid_input, inner_scheme)

        phi = vec_exp(-sigmoid_input, inner_scheme)
        phi = vec_add(phi, 1., inner_scheme)
        phi = vec_recip(phi, inner_scheme)

        if quantize_backprop:
            ctx.save_for_backward(q_in, phi)
        else:
            ctx.save_for_backward(input, phi)
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        return vec_mul(q_in, phi, inner_scheme)

    @staticmethod
    def backward(ctx, grad_output):
        input, phi = ctx.saved_tensors
        scheme = ctx.inner_scheme_bw
        emit_fn = ctx.emit_fn

        fp_go = grad_output
        grad_output = vec_quantize(grad_output, scheme)
        if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_output, scheme)

        dphi = vec_sub(1, phi, scheme)
        dphi = vec_mul(phi, dphi, scheme)

        if ctx.first_order_gelu:
            dphi = vec_mul(1.703125, dphi, scheme)
        else:
            dy = vec_mul(input, input, scheme)
            dy = vec_mul(0.21386719, dy, scheme)
            dy = vec_add(1.59375, dy, scheme)
            dphi = vec_mul(dy, dphi, scheme)

        x_dphi = vec_mul(input, dphi, scheme)
        grad_gelu = vec_add(phi, x_dphi, scheme)
        grad_input = vec_mul(grad_gelu, grad_output, scheme)

        return (grad_input, None, None, None, None, None)


class QuantizedGELU(ObservableMixin, nn.GELU):
    def __init__(self, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None, first_order_gelu=False,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        try:
            super().__init__(approximate='tanh')
        except TypeError:
            super().__init__()
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self.first_order_gelu = first_order_gelu
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return super().forward(input)
        emit_fn = self._emit if self._observers else None
        return GELUFunction.apply(
            input, inner_scheme, self.first_order_gelu,
            quantize_backprop, self._analysis_name, emit_fn,
        )
