"""
QuantizedConv: OpQuantConfig-driven replacement for mx.Conv{1,2,3}d.

Forward + backward are bit-exact equivalent to mx/convolution.py when driven
by the same OpQuantConfig produced by op_config_from_mx_specs.

Two-level quantization model (storage → compute):
- storage: applied first, always per-tensor elemwise (e.g. bfloat16)
- compute: per-role quantization (e.g. fp8 MX per-block)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import grad as nn_grad
from torch.nn.modules.utils import _single, _pair, _triple

from src.scheme.op_config import OpQuantConfig
from src.quantize import quantize
from src.analysis.mixin import ObservableMixin


def _conv_weight(input, weight_shape, grad_output, stride=1, padding=0,
                 dilation=1, groups=1):
    """Compute gradient of conv wrt weight.

    Matches mx/convolution.py's conv_weight helper which works around
    a PyTorch bug in nn.grad.conv2d_weight (pre v1.13).
    """
    num_spatial_dims = input.ndim - 2
    if num_spatial_dims == 1:
        _conv_weight_impl = nn_grad.conv1d_weight
    elif num_spatial_dims == 2:
        _conv_weight_impl = nn_grad.conv2d_weight
    elif num_spatial_dims == 3:
        _conv_weight_impl = nn_grad.conv3d_weight
    else:
        raise ValueError(f"conv_weight does not support ndim={input.ndim}")

    return _conv_weight_impl(
        input, weight_shape, grad_output,
        stride=stride, padding=padding,
        dilation=dilation, groups=groups,
    )


class ConvFunction(torch.autograd.Function):
    """Autograd function for quantized conv with QAT backward.

    Supports Conv1d, Conv2d, Conv3d (detected by input ndim).

    Forward flow:
      1. storage quantize input, weight, bias
      2. compute quantize input
      3. compute quantize weight
      4. F.conv{1,2,3}d(qinput, qweight, qbias, ...)
      5. storage quantize output
      6. compute quantize output
    """

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups,
                cfg: OpQuantConfig, name=None, emit_fn=None):
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.name = name
        ctx.emit_fn = emit_fn

        num_spatial_dims = input.ndim - 2
        assert num_spatial_dims in (1, 2, 3)
        if num_spatial_dims == 1:
            ctx.conv_input = nn_grad.conv1d_input
        elif num_spatial_dims == 2:
            ctx.conv_input = nn_grad.conv2d_input
        else:
            ctx.conv_input = nn_grad.conv3d_input

        input_raw, weight_raw = input, weight

        # input: storage → compute
        if cfg.storage is not None:
            fp_in = input; input = quantize(input, cfg.storage)
            if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, input, cfg.storage)
        input_post_storage = input
        if cfg.input is not None:
            fp_in = input; input = quantize(input, cfg.input)
            if emit_fn: emit_fn("input", 1, "input_pre_quant", fp_in, input, cfg.input)

        # weight: storage → compute
        if cfg.storage is not None:
            fp_wt = weight; weight = quantize(weight, cfg.storage)
            if emit_fn: emit_fn("weight", 0, "weight_pre_quant", fp_wt, weight, cfg.storage)
        weight_post_storage = weight
        if cfg.weight is not None:
            fp_wt = weight; weight = quantize(weight, cfg.weight)
            if emit_fn: emit_fn("weight", 1, "weight_pre_quant", fp_wt, weight, cfg.weight)

        # bias: storage only
        q_bias = bias
        if bias is not None and cfg.storage is not None:
            fp_b = q_bias; q_bias = quantize(q_bias, cfg.storage)
            if emit_fn: emit_fn("bias", 0, "weight_pre_quant", fp_b, q_bias, cfg.storage)

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(input_post_storage, weight_post_storage)
        else:
            ctx.save_for_backward(input_raw, weight_raw)

        ctx.cfg = cfg

        # Compute conv
        if num_spatial_dims == 1:
            output = F.conv1d(input, weight, q_bias, stride, padding, dilation, groups)
        elif num_spatial_dims == 2:
            output = F.conv2d(input, weight, q_bias, stride, padding, dilation, groups)
        else:
            output = F.conv3d(input, weight, q_bias, stride, padding, dilation, groups)

        # Output: storage (bias already included in conv)
        if cfg.storage is not None:
            fp_out = output; output = quantize(output, cfg.storage)
            if emit_fn: emit_fn("output", 0, "output_post_quant", fp_out, output, cfg.storage)

        # Output compute
        if cfg.output is not None:
            fp_out = output; output = quantize(output, cfg.output)
            if emit_fn: emit_fn("output", 1, "output_post_quant", fp_out, output, cfg.output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        emit_fn = ctx.emit_fn

        # grad_output: storage → compute
        if cfg.storage is not None:
            fp_go = grad_output; grad_output = quantize(grad_output, cfg.storage)
            if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_output, cfg.storage)
        if cfg.grad_output is not None:
            fp_go = grad_output; grad_output = quantize(grad_output, cfg.grad_output)
            if emit_fn: emit_fn("grad_output", 1, "grad_output_pre_quant", fp_go, grad_output, cfg.grad_output)

        # grad_weight gemm
        input_gw = input
        if cfg.storage is not None:
            input_gw = quantize(input_gw, cfg.storage)
        if cfg.input_gw is not None:
            input_gw = quantize(input_gw, cfg.input_gw)

        grad_output_gw = grad_output
        if cfg.storage is not None:
            grad_output_gw = quantize(grad_output_gw, cfg.storage)
        if cfg.grad_output_gw is not None:
            grad_output_gw = quantize(grad_output_gw, cfg.grad_output_gw)

        grad_weight = _conv_weight(
            input_gw, weight.shape, grad_output_gw,
            stride=ctx.stride, padding=ctx.padding,
            dilation=ctx.dilation, groups=ctx.groups,
        )

        if cfg.storage is not None:
            fp_gw = grad_weight; grad_weight = quantize(grad_weight, cfg.storage)
            if emit_fn: emit_fn("grad_weight", 0, "grad_weight_post_quant", fp_gw, grad_weight, cfg.storage)
        if cfg.grad_weight is not None:
            fp_gw = grad_weight; grad_weight = quantize(grad_weight, cfg.grad_weight)
            if emit_fn: emit_fn("grad_weight", 1, "grad_weight_post_quant", fp_gw, grad_weight, cfg.grad_weight)

        # grad_input gemm
        weight_gi = weight
        if cfg.storage is not None:
            weight_gi = quantize(weight_gi, cfg.storage)
        if cfg.weight_gi is not None:
            weight_gi = quantize(weight_gi, cfg.weight_gi)

        grad_output_gi = grad_output
        if cfg.storage is not None:
            grad_output_gi = quantize(grad_output_gi, cfg.storage)
        if cfg.grad_output_gi is not None:
            grad_output_gi = quantize(grad_output_gi, cfg.grad_output_gi)

        grad_input = ctx.conv_input(
            input.shape, weight_gi, grad_output_gi,
            stride=ctx.stride, padding=ctx.padding,
            dilation=ctx.dilation, groups=ctx.groups,
        )

        if cfg.storage is not None:
            fp_gi = grad_input; grad_input = quantize(grad_input, cfg.storage)
            if emit_fn: emit_fn("grad_input", 0, "grad_input_post_quant", fp_gi, grad_input, cfg.storage)
        if cfg.grad_input is not None:
            fp_gi = grad_input; grad_input = quantize(grad_input, cfg.grad_input)
            if emit_fn: emit_fn("grad_input", 1, "grad_input_post_quant", fp_gi, grad_input, cfg.grad_input)

        # grad_bias
        grad_bias = None
        if ctx.has_bias:
            sum_axes = [0] + list(range(2, grad_output.ndim))
            grad_bias = grad_output.sum(sum_axes)
            if cfg.storage is not None:
                grad_bias = quantize(grad_bias, cfg.storage)
            if cfg.grad_bias is not None:
                grad_bias = quantize(grad_bias, cfg.grad_bias)

        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None, None, None)

    @staticmethod
    def symbolic(g, input, weight, bias, stride, padding, dilation, groups,
                 cfg, name, emit_fn):
        from src.onnx.helpers import _emit_quantize_node

        if cfg.storage is not None:
            input = _emit_quantize_node(g, input, cfg.storage)
        if cfg.input is not None:
            input = _emit_quantize_node(g, input, cfg.input)

        if cfg.storage is not None:
            weight = _emit_quantize_node(g, weight, cfg.storage)
        if cfg.weight is not None:
            weight = _emit_quantize_node(g, weight, cfg.weight)

        if bias is not None and cfg.storage is not None:
            bias = _emit_quantize_node(g, bias, cfg.storage)
        if bias is not None and cfg.bias is not None:
            bias = _emit_quantize_node(g, bias, cfg.bias)

        weight_sizes = weight.type().sizes()
        kernel_shape = list(weight_sizes[2:]) if weight_sizes is not None else None

        pad_list = list(padding) if hasattr(padding, '__iter__') else [padding]
        onnx_pads = pad_list + pad_list

        conv_kwargs = dict(
            dilations_i=list(dilation) if hasattr(dilation, '__iter__') else [dilation],
            group_i=groups,
            pads_i=onnx_pads,
            strides_i=list(stride) if hasattr(stride, '__iter__') else [stride],
        )
        if kernel_shape is not None:
            conv_kwargs["kernel_shape_i"] = kernel_shape

        if bias is not None:
            output = g.op("Conv", input, weight, bias, **conv_kwargs)
        else:
            output = g.op("Conv", input, weight, **conv_kwargs)

        if cfg.storage is not None:
            output = _emit_quantize_node(g, output, cfg.storage)
        if cfg.output is not None:
            output = _emit_quantize_node(g, output, cfg.output)

        return output


class QuantizedConv2d(ObservableMixin, nn.Conv2d):
    """Drop-in replacement for mx.Conv2d using OpQuantConfig."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cfg: OpQuantConfig = None, name: str = None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.cfg = cfg or OpQuantConfig()
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig():
            return self._conv_forward(x, self.weight, self.bias)

        emit_fn = self._emit if self._observers else None
        return ConvFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.cfg, self._analysis_name, emit_fn,
        )


class QuantizedConv1d(ObservableMixin, nn.Conv1d):
    """Drop-in replacement for mx.Conv1d using OpQuantConfig."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cfg: OpQuantConfig = None, name: str = None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.cfg = cfg or OpQuantConfig()
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig():
            return self._conv_forward(x, self.weight, self.bias)

        emit_fn = self._emit if self._observers else None
        return ConvFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.cfg, self._analysis_name, emit_fn,
        )


class QuantizedConv3d(ObservableMixin, nn.Conv3d):
    """Drop-in replacement for mx.Conv3d using OpQuantConfig."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cfg: OpQuantConfig = None, name: str = None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.cfg = cfg or OpQuantConfig()
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig():
            return self._conv_forward(x, self.weight, self.bias)

        emit_fn = self._emit if self._observers else None
        return ConvFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.cfg, self._analysis_name, emit_fn,
        )


# ---------------------------------------------------------------------------
# ConvTranspose (P3.2)
# ---------------------------------------------------------------------------

class ConvTransposeFunction(torch.autograd.Function):
    """Autograd function for quantized conv_transpose with QAT backward.

    Supports ConvTranspose1d, ConvTranspose2d, ConvTranspose3d.

    Forward flow:
      1. storage quantize input, weight, bias
      2. compute quantize input (axis=1)
      3. compute quantize weight (axis=0 — differs from Conv's axis=1)
      4. F.conv_transposed(qinput, qweight, qbias, ...)
      5. storage + compute quantize output
    """

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding,
                dilation, groups, cfg: OpQuantConfig, name=None, emit_fn=None):
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.name = name
        ctx.emit_fn = emit_fn

        num_spatial_dims = input.ndim - 2
        assert num_spatial_dims in (1, 2, 3)

        input_raw, weight_raw = input, weight

        # input: storage → compute
        if cfg.storage is not None:
            fp_in = input; input = quantize(input, cfg.storage)
            if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in, input, cfg.storage)
        input_post_storage = input
        if cfg.input is not None:
            fp_in = input; input = quantize(input, cfg.input)
            if emit_fn: emit_fn("input", 1, "input_pre_quant", fp_in, input, cfg.input)

        # weight: storage → compute
        if cfg.storage is not None:
            fp_wt = weight; weight = quantize(weight, cfg.storage)
            if emit_fn: emit_fn("weight", 0, "weight_pre_quant", fp_wt, weight, cfg.storage)
        weight_post_storage = weight
        if cfg.weight is not None:
            fp_wt = weight; weight = quantize(weight, cfg.weight)
            if emit_fn: emit_fn("weight", 1, "weight_pre_quant", fp_wt, weight, cfg.weight)

        # bias: storage only
        q_bias = bias
        if bias is not None and cfg.storage is not None:
            fp_b = q_bias; q_bias = quantize(q_bias, cfg.storage)
            if emit_fn: emit_fn("bias", 0, "weight_pre_quant", fp_b, q_bias, cfg.storage)

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(input_post_storage, weight_post_storage)
        else:
            ctx.save_for_backward(input_raw, weight_raw)

        ctx.cfg = cfg

        # Compute conv_transpose
        if num_spatial_dims == 1:
            output = F.conv_transpose1d(input, weight, q_bias, stride, padding,
                                        output_padding, groups, dilation)
        elif num_spatial_dims == 2:
            output = F.conv_transpose2d(input, weight, q_bias, stride, padding,
                                        output_padding, groups, dilation)
        else:
            output = F.conv_transpose3d(input, weight, q_bias, stride, padding,
                                        output_padding, groups, dilation)

        # Output: storage
        if cfg.storage is not None:
            fp_out = output; output = quantize(output, cfg.storage)
            if emit_fn: emit_fn("output", 0, "output_post_quant", fp_out, output, cfg.storage)

        # Output compute
        if cfg.output is not None:
            fp_out = output; output = quantize(output, cfg.output)
            if emit_fn: emit_fn("output", 1, "output_post_quant", fp_out, output, cfg.output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        emit_fn = ctx.emit_fn

        # grad_output: storage → compute
        if cfg.storage is not None:
            fp_go = grad_output; grad_output = quantize(grad_output, cfg.storage)
            if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_output, cfg.storage)
        if cfg.grad_output is not None:
            fp_go = grad_output; grad_output = quantize(grad_output, cfg.grad_output)
            if emit_fn: emit_fn("grad_output", 1, "grad_output_pre_quant", fp_go, grad_output, cfg.grad_output)

        # grad_weight gemm
        input_gw = input
        if cfg.storage is not None:
            input_gw = quantize(input_gw, cfg.storage)
        if cfg.input_gw is not None:
            input_gw = quantize(input_gw, cfg.input_gw)

        grad_output_gw = grad_output
        if cfg.storage is not None:
            grad_output_gw = quantize(grad_output_gw, cfg.storage)
        if cfg.grad_output_gw is not None:
            grad_output_gw = quantize(grad_output_gw, cfg.grad_output_gw)

        grad_weight = _conv_weight(
            grad_output_gw, weight.shape, input_gw,
            stride=ctx.stride, padding=ctx.padding,
            dilation=ctx.dilation, groups=ctx.groups,
        )

        if cfg.storage is not None:
            fp_gw = grad_weight; grad_weight = quantize(grad_weight, cfg.storage)
            if emit_fn: emit_fn("grad_weight", 0, "grad_weight_post_quant", fp_gw, grad_weight, cfg.storage)
        if cfg.grad_weight is not None:
            fp_gw = grad_weight; grad_weight = quantize(grad_weight, cfg.grad_weight)
            if emit_fn: emit_fn("grad_weight", 1, "grad_weight_post_quant", fp_gw, grad_weight, cfg.grad_weight)

        # grad_input: uses F.conv2d (not conv_transpose)
        weight_gi = weight
        if cfg.storage is not None:
            weight_gi = quantize(weight_gi, cfg.storage)
        if cfg.weight_gi is not None:
            weight_gi = quantize(weight_gi, cfg.weight_gi)

        grad_output_gi = grad_output
        if cfg.storage is not None:
            grad_output_gi = quantize(grad_output_gi, cfg.storage)
        if cfg.grad_output_gi is not None:
            grad_output_gi = quantize(grad_output_gi, cfg.grad_output_gi)

        num_spatial_dims = input.ndim - 2
        if num_spatial_dims == 1:
            grad_input = F.conv1d(grad_output_gi, weight_gi, bias=None,
                                  stride=ctx.stride, padding=ctx.padding,
                                  dilation=ctx.dilation, groups=ctx.groups)
        elif num_spatial_dims == 2:
            grad_input = F.conv2d(grad_output_gi, weight_gi, bias=None,
                                  stride=ctx.stride, padding=ctx.padding,
                                  dilation=ctx.dilation, groups=ctx.groups)
        else:
            grad_input = F.conv3d(grad_output_gi, weight_gi, bias=None,
                                  stride=ctx.stride, padding=ctx.padding,
                                  dilation=ctx.dilation, groups=ctx.groups)

        if cfg.storage is not None:
            fp_gi = grad_input; grad_input = quantize(grad_input, cfg.storage)
            if emit_fn: emit_fn("grad_input", 0, "grad_input_post_quant", fp_gi, grad_input, cfg.storage)
        if cfg.grad_input is not None:
            fp_gi = grad_input; grad_input = quantize(grad_input, cfg.grad_input)
            if emit_fn: emit_fn("grad_input", 1, "grad_input_post_quant", fp_gi, grad_input, cfg.grad_input)

        # grad_bias
        grad_bias = None
        if ctx.has_bias:
            sum_axes = [0] + list(range(2, grad_output.ndim))
            grad_bias = grad_output.sum(sum_axes)
            if cfg.storage is not None:
                grad_bias = quantize(grad_bias, cfg.storage)
            if cfg.grad_bias is not None:
                grad_bias = quantize(grad_bias, cfg.grad_bias)

        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None, None, None, None)

    @staticmethod
    def symbolic(g, input, weight, bias, stride, padding, output_padding,
                 dilation, groups, cfg, name, emit_fn):
        from src.onnx.helpers import _emit_quantize_node

        if cfg.storage is not None:
            input = _emit_quantize_node(g, input, cfg.storage)
        if cfg.input is not None:
            input = _emit_quantize_node(g, input, cfg.input)

        if cfg.storage is not None:
            weight = _emit_quantize_node(g, weight, cfg.storage)
        if cfg.weight is not None:
            weight = _emit_quantize_node(g, weight, cfg.weight)

        if bias is not None and cfg.storage is not None:
            bias = _emit_quantize_node(g, bias, cfg.storage)
        if bias is not None and cfg.bias is not None:
            bias = _emit_quantize_node(g, bias, cfg.bias)

        weight_sizes = weight.type().sizes()
        kernel_shape = list(weight_sizes[2:]) if weight_sizes is not None else None

        pad_list = list(padding) if hasattr(padding, '__iter__') else [padding]
        onnx_pads = pad_list + pad_list

        conv_kwargs = dict(
            dilations_i=list(dilation) if hasattr(dilation, '__iter__') else [dilation],
            group_i=groups,
            output_padding_i=list(output_padding) if hasattr(output_padding, '__iter__') else [output_padding],
            pads_i=onnx_pads,
            strides_i=list(stride) if hasattr(stride, '__iter__') else [stride],
        )
        if kernel_shape is not None:
            conv_kwargs["kernel_shape_i"] = kernel_shape

        if bias is not None:
            output = g.op("ConvTranspose", input, weight, bias, **conv_kwargs)
        else:
            output = g.op("ConvTranspose", input, weight, **conv_kwargs)

        if cfg.storage is not None:
            output = _emit_quantize_node(g, output, cfg.storage)
        if cfg.output is not None:
            output = _emit_quantize_node(g, output, cfg.output)

        return output


class QuantizedConvTranspose2d(ObservableMixin, nn.ConvTranspose2d):
    """Drop-in replacement for mx.ConvTranspose2d using OpQuantConfig."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, dilation=1, groups=1,
                 bias=True, cfg: OpQuantConfig = None, name: str = None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         output_padding=output_padding, groups=groups, bias=bias)
        self.cfg = cfg or OpQuantConfig()
        self._analysis_name = name

    def forward(self, x, output_size=None):
        if self.cfg == OpQuantConfig():
            return self._conv_forward(x, self.weight, self.bias)

        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding,
            self.kernel_size, self.dilation,
        )

        emit_fn = self._emit if self._observers else None
        return ConvTransposeFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, output_padding,
            self.dilation, self.groups,
            self.cfg, self._analysis_name, emit_fn,
        )


class QuantizedConvTranspose1d(ObservableMixin, nn.ConvTranspose1d):
    """Drop-in replacement for mx.ConvTranspose1d using OpQuantConfig."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, dilation=1, groups=1,
                 bias=True, cfg: OpQuantConfig = None, name: str = None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         output_padding=output_padding, groups=groups, bias=bias)
        self.cfg = cfg or OpQuantConfig()
        self._analysis_name = name

    def forward(self, x, output_size=None):
        if self.cfg == OpQuantConfig():
            return self._conv_forward(x, self.weight, self.bias)

        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding,
            self.kernel_size, self.dilation,
        )

        emit_fn = self._emit if self._observers else None
        return ConvTransposeFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, output_padding,
            self.dilation, self.groups,
            self.cfg, self._analysis_name, emit_fn,
        )


class QuantizedConvTranspose3d(ObservableMixin, nn.ConvTranspose3d):
    """Drop-in replacement for mx.ConvTranspose3d using OpQuantConfig."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, dilation=1, groups=1,
                 bias=True, cfg: OpQuantConfig = None, name: str = None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         output_padding=output_padding, groups=groups, bias=bias)
        self.cfg = cfg or OpQuantConfig()
        self._analysis_name = name

    def forward(self, x, output_size=None):
        if self.cfg == OpQuantConfig():
            return self._conv_forward(x, self.weight, self.bias)

        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding,
            self.kernel_size, self.dilation,
        )

        emit_fn = self._emit if self._observers else None
        return ConvTransposeFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, output_padding,
            self.dilation, self.groups,
            self.cfg, self._analysis_name, emit_fn,
        )
