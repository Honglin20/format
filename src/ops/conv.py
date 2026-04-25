"""
QuantizedConv: OpQuantConfig-driven replacement for mx.Conv{1,2,3}d.

Forward + backward are bit-exact equivalent to mx/convolution.py when driven
by the same OpQuantConfig produced by op_config_from_mx_specs.

Conv uses axis=1 (channel dim) for MX block quantization on both input
and weight in forward, and different axes in backward gemm operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import grad as nn_grad
from torch.nn.modules.utils import _single, _pair, _triple

from src.scheme.op_config import OpQuantConfig
from src.quantize import quantize
from src.analysis.mixin import ObservableMixin
from src.scheme.granularity import GranularityMode


def _conv_weight(input, weight_shape, grad_output, stride=1, padding=0,
                 dilation=1, groups=1):
    """Compute gradient of conv wrt weight.

    Matches mx/convolution.py's conv_weight helper which works around
    a PyTorch bug in nn.grad.conv2d_weight (pre v1.13).
    """
    num_spatial_dims = input.ndim - 2
    if num_spatial_dims == 1:
        _p = _single
        _conv = F.conv1d
        _conv_weight = nn_grad.conv1d_weight
    elif num_spatial_dims == 2:
        _p = _pair
        _conv = F.conv2d
        _conv_weight = nn_grad.conv2d_weight
    elif num_spatial_dims == 3:
        _p = _triple
        _conv = F.conv3d
        _conv_weight = nn_grad.conv3d_weight
    else:
        raise ValueError(f"conv_weight does not support ndim={input.ndim}")

    return _conv_weight(
        input, weight_shape, grad_output,
        stride=stride, padding=padding,
        dilation=dilation, groups=groups,
    )


class ConvFunction(torch.autograd.Function):
    """Autograd function for quantized conv with QAT backward.

    Supports Conv1d, Conv2d, Conv3d (detected by input ndim).

    Forward flow (matches mx/convolution.py):
      1. elemwise quantize input, weight, bias
      2. MX quantize input along axis=1 (in_channels)
      3. MX quantize weight along axis=1 (in_channels/groups)
      4. F.conv{1,2,3}d(qinput, qweight, qbias, ...)
      5. elemwise quantize output
    """

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups,
                cfg: OpQuantConfig, name=None):
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.name = name

        num_spatial_dims = input.ndim - 2
        assert num_spatial_dims in (1, 2, 3)
        if num_spatial_dims == 1:
            ctx.conv_input = nn_grad.conv1d_input
        elif num_spatial_dims == 2:
            ctx.conv_input = nn_grad.conv2d_input
        else:
            ctx.conv_input = nn_grad.conv3d_input

        input_raw, weight_raw = input, weight

        # Split pipelines into elemwise + MX
        input_elem = tuple(s for s in cfg.input if s.granularity.mode != GranularityMode.PER_BLOCK)
        input_mx = tuple(s for s in cfg.input if s.granularity.mode == GranularityMode.PER_BLOCK)

        weight_elem = tuple(s for s in cfg.weight if s.granularity.mode != GranularityMode.PER_BLOCK)
        weight_mx = tuple(s for s in cfg.weight if s.granularity.mode == GranularityMode.PER_BLOCK)

        # Elemwise quantize
        for s in input_elem:
            input = quantize(input, s)
        input_post_elem = input

        for s in weight_elem:
            weight = quantize(weight, s)
        weight_post_elem = weight

        # MX quantize (conv uses axis=1 for channel dim)
        for s in input_mx:
            input = quantize(input, s)
        for s in weight_mx:
            weight = quantize(weight, s)

        # Bias quantization
        q_bias = None
        if bias is not None:
            q_bias = bias
            for s in cfg.bias:
                q_bias = quantize(q_bias, s)

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(input_post_elem, weight_post_elem)
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

        # Output quantization (single elemwise step — bias already included in conv)
        if len(cfg.output) > 0:
            output = quantize(output, cfg.output[0])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg

        # Quantize grad_output
        for s in cfg.grad_output:
            grad_output = quantize(grad_output, s)

        # --- grad_weight ---
        # input MX along axis=0, grad_output MX along axis=0
        input_gw = input
        for s in cfg.input_gw:
            input_gw = quantize(input_gw, s)

        grad_output_gw = grad_output
        for s in cfg.grad_output_gw:
            grad_output_gw = quantize(grad_output_gw, s)

        grad_weight = _conv_weight(
            input_gw, weight.shape, grad_output_gw,
            stride=ctx.stride, padding=ctx.padding,
            dilation=ctx.dilation, groups=ctx.groups,
        )

        for s in cfg.grad_weight:
            grad_weight = quantize(grad_weight, s)

        # --- grad_input ---
        # weight MX along axis=0, grad_output MX along axis=1
        weight_gi = weight
        for s in cfg.weight_gi:
            weight_gi = quantize(weight_gi, s)

        grad_output_gi = grad_output
        for s in cfg.grad_output_gi:
            grad_output_gi = quantize(grad_output_gi, s)

        grad_input = ctx.conv_input(
            input.shape, weight_gi, grad_output_gi,
            stride=ctx.stride, padding=ctx.padding,
            dilation=ctx.dilation, groups=ctx.groups,
        )

        for s in cfg.grad_input:
            grad_input = quantize(grad_input, s)

        # --- grad_bias ---
        grad_bias = None
        if ctx.has_bias:
            sum_axes = [0] + list(range(2, grad_output.ndim))
            grad_bias = grad_output.sum(sum_axes)
            for s in cfg.grad_bias:
                grad_bias = quantize(grad_bias, s)

        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None, None)


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

        return ConvFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.cfg, self._analysis_name,
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

        return ConvFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.cfg, self._analysis_name,
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

        return ConvFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.cfg, self._analysis_name,
        )


# ---------------------------------------------------------------------------
# ConvTranspose (P3.2)
# ---------------------------------------------------------------------------

class ConvTransposeFunction(torch.autograd.Function):
    """Autograd function for quantized conv_transpose with QAT backward.

    Supports ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
    (detected by input ndim).

    Forward flow (matches mx/transpose_convolution.py):
      1. elemwise quantize input, weight, bias
      2. MX quantize input along axis=1 (in_channels)
      3. MX quantize weight along axis=0 (in_channels — differs from Conv's axis=1)
      4. F.conv_transpose{1,2,3}d(qinput, qweight, qbias, ..., output_padding)
      5. elemwise quantize output

    Backward flow:
      - grad_weight: conv_weight(grad_output, weight.shape, input) with
        input MX axis=0, grad_output MX axis=0
      - grad_input: F.conv{1,2,3}d(grad_output, weight, ...) with
        weight MX axis=1, grad_output MX axis=1
    """

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding,
                dilation, groups, cfg: OpQuantConfig, name=None):
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.name = name

        num_spatial_dims = input.ndim - 2
        assert num_spatial_dims in (1, 2, 3)

        input_raw, weight_raw = input, weight

        # Split pipelines into elemwise + MX
        input_elem = tuple(s for s in cfg.input if s.granularity.mode != GranularityMode.PER_BLOCK)
        input_mx = tuple(s for s in cfg.input if s.granularity.mode == GranularityMode.PER_BLOCK)

        weight_elem = tuple(s for s in cfg.weight if s.granularity.mode != GranularityMode.PER_BLOCK)
        weight_mx = tuple(s for s in cfg.weight if s.granularity.mode == GranularityMode.PER_BLOCK)

        # Elemwise quantize
        for s in input_elem:
            input = quantize(input, s)
        input_post_elem = input

        for s in weight_elem:
            weight = quantize(weight, s)
        weight_post_elem = weight

        # MX quantize (conv_transpose: input axis=1, weight axis=0)
        for s in input_mx:
            input = quantize(input, s)
        for s in weight_mx:
            weight = quantize(weight, s)

        # Bias quantization
        q_bias = None
        if bias is not None:
            q_bias = bias
            for s in cfg.bias:
                q_bias = quantize(q_bias, s)

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(input_post_elem, weight_post_elem)
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

        # Output quantization
        if len(cfg.output) > 0:
            output = quantize(output, cfg.output[0])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg

        # Quantize grad_output
        for s in cfg.grad_output:
            grad_output = quantize(grad_output, s)

        # --- grad_weight ---
        # input MX along axis=0, grad_output MX along axis=0
        input_gw = input
        for s in cfg.input_gw:
            input_gw = quantize(input_gw, s)

        grad_output_gw = grad_output
        for s in cfg.grad_output_gw:
            grad_output_gw = quantize(grad_output_gw, s)

        grad_weight = _conv_weight(
            grad_output_gw, weight.shape, input_gw,
            stride=ctx.stride, padding=ctx.padding,
            dilation=ctx.dilation, groups=ctx.groups,
        )

        for s in cfg.grad_weight:
            grad_weight = quantize(grad_weight, s)

        # --- grad_input ---
        # Uses F.conv2d (not conv_transpose) to compute grad_input
        # weight MX along axis=1, grad_output MX along axis=1
        weight_gi = weight
        for s in cfg.weight_gi:
            weight_gi = quantize(weight_gi, s)

        grad_output_gi = grad_output
        for s in cfg.grad_output_gi:
            grad_output_gi = quantize(grad_output_gi, s)

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

        for s in cfg.grad_input:
            grad_input = quantize(grad_input, s)

        # --- grad_bias ---
        grad_bias = None
        if ctx.has_bias:
            sum_axes = [0] + list(range(2, grad_output.ndim))
            grad_bias = grad_output.sum(sum_axes)
            for s in cfg.grad_bias:
                grad_bias = quantize(grad_bias, s)

        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None, None, None)


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

        return ConvTransposeFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, output_padding,
            self.dilation, self.groups,
            self.cfg, self._analysis_name,
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

        return ConvTransposeFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, output_padding,
            self.dilation, self.groups,
            self.cfg, self._analysis_name,
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

        return ConvTransposeFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, output_padding,
            self.dilation, self.groups,
            self.cfg, self._analysis_name,
        )
