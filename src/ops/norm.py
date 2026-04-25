"""
Quantized Norm operators: BatchNorm, LayerNorm, GroupNorm, RMSNorm.

OpQuantConfig-driven, bit-exact equivalent to mx/{batchnorm,layernorm,groupnorm}.py.

Norm operators differ from matmul-family operators in that every intermediate
arithmetic step is quantized (via vec_ops). The OpQuantConfig provides
entry/exit quantization (input, weight, bias, output, grad_*), while an
additional `inner_scheme` parameter provides the QuantScheme used for all
intermediate vec_op computations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.quantize import quantize
from src.analysis.mixin import ObservableMixin
from src.ops.vec_ops import (
    vec_quantize, vec_add, vec_sub, vec_mul, vec_div,
    vec_recip, vec_sqrt, vec_exp, vec_exp2, vec_tanh,
    vec_reduce_sum, vec_reduce_mean,
)


# ---------------------------------------------------------------------------
# Shared norm forward/backward (mirrors mx/norm_utils.py)
# ---------------------------------------------------------------------------

def _get_group_shape(x, axis, groups):
    H = x.shape[axis]
    assert H % groups == 0
    orig_shape = list(x.shape)
    grouped_shape = list(x.shape)
    grouped_shape[axis] = groups
    grouped_shape.insert(axis + 1, H // groups)
    return orig_shape, grouped_shape


def _norm_forward(x, axes, weight, bias, eps, scheme: QuantScheme,
                  groups=None, weight_axis=None,
                  use_running_stats=False,
                  running_mean=None, running_var=None):
    """Forward pass for BatchNorm, LayerNorm, GroupNorm."""
    if not isinstance(axes, list):
        axes = [axes]

    if weight_axis is not None:
        w_shape = [1 for _ in range(x.ndim)]
        w_shape[weight_axis] = x.shape[weight_axis]
    else:
        w_shape = None

    if groups:
        orig_shape, grouped_shape = _get_group_shape(x, axes[0], groups)
        x = x.view(grouped_shape)
        axes = [a + 1 for a in axes]

    reduced_shape = list(x.shape)
    reduced_shape[0] = 1
    for i in axes:
        reduced_shape[i] = 1

    if not use_running_stats:
        x_mean = vec_reduce_mean(x, axes, keepdim=True, scheme=scheme)
        x_shift = vec_sub(x, x_mean, scheme)
        x_shift_pow2 = vec_mul(x_shift, x_shift, scheme)
        x_var = vec_reduce_mean(x_shift_pow2, axes, keepdim=True, scheme=scheme)
    else:
        assert running_mean is not None
        assert running_var is not None
        x_mean = vec_quantize(running_mean, scheme)
        x_mean = x_mean.view(reduced_shape)
        x_shift = vec_sub(x, x_mean, scheme)
        x_var = vec_quantize(running_var, scheme)
        x_var = x_var.view(reduced_shape)

    x_vare = vec_add(x_var, eps, scheme)
    x_std = vec_sqrt(x_vare, scheme)
    x_std_inv = vec_recip(x_std, scheme)
    x_norm = vec_mul(x_shift, x_std_inv, scheme)

    if groups:
        x_norm = x_norm.view(orig_shape)

    if w_shape is not None:
        weight = weight.view(w_shape)
        bias = bias.view(w_shape)

    x_scale = vec_mul(weight, x_norm, scheme)
    output = vec_add(x_scale, bias, scheme)

    return output, x_shift, x_norm, x_std_inv, x_mean, x_vare


def _norm_backward(grad_output, axes, weight, x_shift, x_std_inv,
                   scheme: QuantScheme, groups=None, weight_axis=None):
    """Backward pass for BatchNorm, GroupNorm."""
    if not isinstance(axes, list):
        axes = [axes]

    if weight_axis is not None:
        w_shape = [1 for _ in range(grad_output.ndim)]
        w_shape[weight_axis] = grad_output.shape[weight_axis]
    else:
        w_shape = None

    if groups:
        orig_shape, grouped_shape = _get_group_shape(grad_output, axes[0], groups)
        axes = [a + 1 for a in axes]

    if w_shape is not None:
        weight = weight.view(w_shape)

    dx_norm = vec_mul(grad_output, weight, scheme)

    if groups:
        dx_norm = dx_norm.view(grouped_shape)

    dx_shift = vec_mul(dx_norm, x_std_inv, scheme)
    dx_mean = vec_reduce_mean(-dx_shift, axes, keepdim=True, scheme=scheme)

    dx_std = vec_mul(dx_norm, x_shift, scheme)
    dx_std = vec_reduce_mean(dx_std, axes, keepdim=True, scheme=scheme)
    x_vare_inv = vec_mul(x_std_inv, x_std_inv, scheme)
    dx_std = vec_mul(dx_std, x_vare_inv, scheme)
    dx_std = vec_mul(dx_std, x_std_inv, scheme)
    dx_shift2 = vec_mul(-dx_std, x_shift, scheme)

    dx = vec_add(dx_shift, dx_shift2, scheme)
    dx = vec_add(dx, dx_mean, scheme)

    if groups:
        dx = dx.view(orig_shape)

    return dx


def _norm_backward_LN(grad_output, axes, weight, x_norm, x_var,
                      scheme: QuantScheme, groups=None, weight_axis=None):
    """Backward pass for LayerNorm (Deepspeed-style)."""
    if not isinstance(axes, list):
        axes = [axes]

    if weight_axis is not None:
        w_shape = [1 for _ in range(grad_output.ndim)]
        w_shape[weight_axis] = grad_output.shape[weight_axis]
    else:
        w_shape = None

    if groups:
        orig_shape, grouped_shape = _get_group_shape(grad_output, axes[0], groups)
        axes = [a + 1 for a in axes]

    if w_shape is not None:
        weight = weight.view(w_shape)

    dx_norm = vec_mul(grad_output, weight, scheme)

    if groups:
        dx_norm = dx_norm.view(grouped_shape)

    x_std = vec_sqrt(x_var, scheme)
    x_std_inv = vec_div(1.0, x_std, scheme)

    dx_shift = vec_mul(dx_norm, x_std_inv, scheme)

    dx_std_tmp = vec_mul(dx_norm, x_norm, scheme)
    dx_std_tmp = vec_mul(dx_std_tmp, x_std, scheme)
    dx_std_tmp = vec_reduce_mean(dx_std_tmp, axes, keepdim=True, scheme=scheme)
    x_vare_inv = vec_div(1.0, x_var, scheme)
    dx_std_tmp = vec_mul(dx_std_tmp, x_vare_inv, scheme)
    dx_shift2 = vec_mul(-dx_std_tmp, x_norm, scheme)

    dx = vec_add(dx_shift, dx_shift2, scheme)
    dx_mean = vec_reduce_mean(dx, axes, keepdim=True, scheme=scheme)
    dx = vec_add(dx, -dx_mean, scheme)

    if groups:
        dx = dx.view(orig_shape)

    return dx


# ---------------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------------

class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, running_mean, running_var, weight, bias,
                is_training, momentum, eps,
                cfg: OpQuantConfig, inner_scheme: QuantScheme,
                quantize_backprop: bool = True, name=None):
        if not is_training:
            assert running_mean is not None
            assert running_var is not None

        ctx.is_training = is_training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.name = name

        # Entry quantization on input, weight, bias
        for s in cfg.input:
            x = quantize(x, s)
        q_weight = weight
        for s in cfg.weight:
            q_weight = quantize(q_weight, s)
        q_bias = bias
        for s in cfg.bias:
            q_bias = quantize(q_bias, s)

        H = x.shape[1]
        sum_axes = [0] + list(range(2, x.ndim))

        output, x_shift, x_norm, x_std_inv, x_mean, x_var = _norm_forward(
            x, sum_axes, q_weight, q_bias, eps, inner_scheme,
            weight_axis=1,
            use_running_stats=not is_training,
            running_mean=running_mean, running_var=running_var,
        )

        # Update running stats
        if is_training and running_mean is not None:
            t1 = vec_mul((1 - momentum), running_mean, inner_scheme)
            t2 = vec_mul(momentum, x_mean.view(H), inner_scheme)
            t3 = vec_add(t1, t2, inner_scheme)
            running_mean.copy_(t3)
        if is_training and running_var is not None:
            t1 = vec_mul((1 - momentum), running_var, inner_scheme)
            t2 = vec_mul(momentum, x_var.view(H), inner_scheme)
            t3 = vec_add(t1, t2, inner_scheme)
            running_var.copy_(t3)

        # Save for backward — always save intermediates; choose quantized or raw weight
        if cfg.is_training:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, q_weight)
        else:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, weight)

        ctx.cfg = cfg
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        ctx.sum_axes = sum_axes

        # Output quantization
        for s in cfg.output:
            output = quantize(output, s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_shift, x_norm, x_std_inv, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        scheme = ctx.inner_scheme_bw

        # Entry quantization on grad_output
        for s in cfg.grad_output:
            grad_output = quantize(grad_output, s)

        sum_axes = ctx.sum_axes

        # grad_bias
        grad_bias = vec_reduce_sum(grad_output, sum_axes, scheme=scheme)

        # grad_weight
        grad_weight = vec_mul(grad_output, x_norm, scheme)
        grad_weight = vec_reduce_sum(grad_weight, sum_axes, scheme=scheme)

        # grad_input
        grad_input = _norm_backward(
            grad_output, sum_axes, weight, x_shift, x_std_inv,
            scheme, weight_axis=1,
        )

        # Exit quantization
        for s in cfg.grad_bias:
            grad_bias = quantize(grad_bias, s)
        for s in cfg.grad_weight:
            grad_weight = quantize(grad_weight, s)
        for s in cfg.grad_input:
            grad_input = quantize(grad_input, s)

        return (grad_input, None, None, grad_weight, grad_bias,
                None, None, None, None, None, None, None)


class QuantizedBatchNorm2d(ObservableMixin, nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 cfg: OpQuantConfig = None, inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True,
                 name: str = None, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum,
                         affine=affine, track_running_stats=track_running_stats,
                         **kwargs)
        self.cfg = cfg or OpQuantConfig()
        self.inner_scheme = inner_scheme
        self.quantize_backprop = quantize_backprop
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig() and self.inner_scheme is None:
            return super().forward(x)

        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        bn_training = self.training or (self.running_mean is None and self.running_var is None)

        return BatchNormFunction.apply(
            x,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias,
            bn_training, exponential_average_factor, self.eps,
            self.cfg, self.inner_scheme, self.quantize_backprop, self._analysis_name,
        )


class QuantizedBatchNorm1d(ObservableMixin, nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 cfg: OpQuantConfig = None, inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True,
                 name: str = None, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum,
                         affine=affine, track_running_stats=track_running_stats,
                         **kwargs)
        self.cfg = cfg or OpQuantConfig()
        self.inner_scheme = inner_scheme
        self.quantize_backprop = quantize_backprop
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig() and self.inner_scheme is None:
            return super().forward(x)

        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        bn_training = self.training or (self.running_mean is None and self.running_var is None)

        return BatchNormFunction.apply(
            x,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias,
            bn_training, exponential_average_factor, self.eps,
            self.cfg, self.inner_scheme, self.quantize_backprop, self._analysis_name,
        )


class QuantizedBatchNorm3d(ObservableMixin, nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 cfg: OpQuantConfig = None, inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True,
                 name: str = None, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum,
                         affine=affine, track_running_stats=track_running_stats,
                         **kwargs)
        self.cfg = cfg or OpQuantConfig()
        self.inner_scheme = inner_scheme
        self.quantize_backprop = quantize_backprop
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig() and self.inner_scheme is None:
            return super().forward(x)

        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        bn_training = self.training or (self.running_mean is None and self.running_var is None)

        return BatchNormFunction.apply(
            x,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias,
            bn_training, exponential_average_factor, self.eps,
            self.cfg, self.inner_scheme, self.quantize_backprop, self._analysis_name,
        )


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps,
                cfg: OpQuantConfig, inner_scheme: QuantScheme,
                quantize_backprop: bool = True, name=None):
        ctx.eps = eps
        ctx.name = name

        # Entry quantization
        for s in cfg.input:
            x = quantize(x, s)
        q_weight = weight
        for s in cfg.weight:
            q_weight = quantize(q_weight, s)
        q_bias = bias
        for s in cfg.bias:
            q_bias = quantize(q_bias, s)

        output, _, x_norm, _, _, x_vare = _norm_forward(
            x, -1, q_weight, q_bias, eps, inner_scheme,
        )

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(x_norm, x_vare, q_weight)
        else:
            ctx.save_for_backward(x_norm, x_vare, weight)

        ctx.cfg = cfg
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        # Output quantization
        for s in cfg.output:
            output = quantize(output, s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, x_vare, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        scheme = ctx.inner_scheme_bw

        # Entry quantization on grad_output
        for s in cfg.grad_output:
            grad_output = quantize(grad_output, s)

        sum_axes = list(range(grad_output.ndim - 1))

        # grad_bias
        grad_bias = vec_reduce_sum(grad_output, sum_axes, scheme=scheme)

        # grad_weight
        grad_weight = vec_mul(grad_output, x_norm, scheme)
        grad_weight = vec_reduce_sum(grad_weight, sum_axes, scheme=scheme)

        # grad_input
        grad_input = _norm_backward_LN(
            grad_output, -1, weight, x_norm, x_vare, scheme,
        )

        # Exit quantization
        for s in cfg.grad_bias:
            grad_bias = quantize(grad_bias, s)
        for s in cfg.grad_weight:
            grad_weight = quantize(grad_weight, s)
        for s in cfg.grad_input:
            grad_input = quantize(grad_input, s)

        return (grad_input, grad_weight, grad_bias, None, None, None, None)


class QuantizedLayerNorm(ObservableMixin, nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 cfg: OpQuantConfig = None, inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True,
                 name: str = None, **kwargs):
        super().__init__(normalized_shape, eps=eps,
                         elementwise_affine=elementwise_affine, **kwargs)
        self.cfg = cfg or OpQuantConfig()
        self.inner_scheme = inner_scheme
        self.quantize_backprop = quantize_backprop
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig() and self.inner_scheme is None:
            return super().forward(x)

        return LayerNormFunction.apply(
            x, self.weight, self.bias, self.eps,
            self.cfg, self.inner_scheme, self.quantize_backprop, self._analysis_name,
        )


# ---------------------------------------------------------------------------
# GroupNorm
# ---------------------------------------------------------------------------

class GroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_groups, weight, bias, eps,
                cfg: OpQuantConfig, inner_scheme: QuantScheme,
                quantize_backprop: bool = True, name=None):
        ctx.num_groups = num_groups
        ctx.eps = eps
        ctx.name = name

        # Entry quantization
        for s in cfg.input:
            x = quantize(x, s)
        q_weight = weight
        for s in cfg.weight:
            q_weight = quantize(q_weight, s)
        q_bias = bias
        for s in cfg.bias:
            q_bias = quantize(q_bias, s)

        sum_axes = list(range(1, x.ndim))

        output, x_shift, x_norm, x_std_inv, _, _ = _norm_forward(
            x, sum_axes, q_weight, q_bias, eps, inner_scheme,
            groups=num_groups, weight_axis=1,
        )

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, q_weight)
        else:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, weight)

        ctx.cfg = cfg
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        # Output quantization
        for s in cfg.output:
            output = quantize(output, s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_shift, x_norm, x_std_inv, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        scheme = ctx.inner_scheme_bw

        # Entry quantization on grad_output
        for s in cfg.grad_output:
            grad_output = quantize(grad_output, s)

        sum_axes = [0] + list(range(2, grad_output.ndim))

        # grad_bias
        grad_bias = vec_reduce_sum(grad_output, sum_axes, scheme=scheme)

        # grad_weight
        grad_weight = vec_mul(grad_output, x_norm, scheme)
        grad_weight = vec_reduce_sum(grad_weight, sum_axes, scheme=scheme)

        # grad_input
        grad_input = _norm_backward(
            grad_output, list(range(1, grad_output.ndim)),
            weight, x_shift, x_std_inv, scheme,
            groups=ctx.num_groups, weight_axis=1,
        )

        # Exit quantization
        for s in cfg.grad_bias:
            grad_bias = quantize(grad_bias, s)
        for s in cfg.grad_weight:
            grad_weight = quantize(grad_weight, s)
        for s in cfg.grad_input:
            grad_input = quantize(grad_input, s)

        return (grad_input, None, grad_weight, grad_bias,
                None, None, None, None)


class QuantizedGroupNorm(ObservableMixin, nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 cfg: OpQuantConfig = None, inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True,
                 name: str = None, **kwargs):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine,
                         **kwargs)
        self.cfg = cfg or OpQuantConfig()
        self.inner_scheme = inner_scheme
        self.quantize_backprop = quantize_backprop
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig() and self.inner_scheme is None:
            return super().forward(x)

        return GroupNormFunction.apply(
            x, self.num_groups, self.weight, self.bias, self.eps,
            self.cfg, self.inner_scheme, self.quantize_backprop, self._analysis_name,
        )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps,
                cfg: OpQuantConfig, inner_scheme: QuantScheme,
                quantize_backprop: bool = True, name=None):
        ctx.eps = eps
        ctx.name = name

        # Entry quantization
        for s in cfg.input:
            x = quantize(x, s)

        # RMSNorm: x_rms = sqrt(mean(x^2) + eps), output = x * (1/x_rms) * weight + bias
        x2 = vec_mul(x, x, inner_scheme)
        x_ms = vec_reduce_mean(x2, -1, keepdim=True, scheme=inner_scheme)
        x_mse = vec_add(x_ms, eps, inner_scheme)
        x_rms = vec_sqrt(x_mse, inner_scheme)
        x_rms_inv = vec_recip(x_rms, inner_scheme)
        x_norm = vec_mul(x, x_rms_inv, inner_scheme)

        q_weight = weight
        for s in cfg.weight:
            q_weight = quantize(q_weight, s)
        q_bias = bias
        for s in cfg.bias:
            q_bias = quantize(q_bias, s)

        x_scale = vec_mul(q_weight, x_norm, inner_scheme)
        output = vec_add(x_scale, q_bias, inner_scheme)

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(x_norm, x_rms_inv, q_weight)
        else:
            ctx.save_for_backward(x_norm, x_rms_inv, weight)

        ctx.cfg = cfg
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None

        # Output quantization
        for s in cfg.output:
            output = quantize(output, s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, x_rms_inv, weight = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        scheme = ctx.inner_scheme_bw

        # Entry quantization on grad_output
        for s in cfg.grad_output:
            grad_output = quantize(grad_output, s)

        sum_axes = list(range(len(grad_output.shape) - 1))

        # grad_bias
        grad_bias = vec_reduce_sum(grad_output, sum_axes, scheme=scheme)

        # grad_weight
        grad_weight = vec_mul(grad_output, x_norm, scheme)
        grad_weight = vec_reduce_sum(grad_weight, sum_axes, scheme=scheme)

        # grad_input
        dx_norm = vec_mul(grad_output, weight, scheme)
        dx1 = vec_mul(dx_norm, x_rms_inv, scheme)
        dx_norm2 = vec_mul(dx1, x_norm, scheme)
        dx_norm2 = vec_reduce_mean(dx_norm2, -1, keepdim=True, scheme=scheme)
        dx_norm3 = vec_mul(x_norm, dx_norm2, scheme)
        # mx/layernorm.py RMSNorm backward: vec_sub(dx1, dx_norm3) has no
        # mx_specs — the final subtraction is NOT quantized
        grad_input = dx1 - dx_norm3

        # Exit quantization
        for s in cfg.grad_bias:
            grad_bias = quantize(grad_bias, s)
        for s in cfg.grad_weight:
            grad_weight = quantize(grad_weight, s)
        for s in cfg.grad_input:
            grad_input = quantize(grad_input, s)

        return (grad_input, grad_weight, grad_bias, None, None, None, None)


class QuantizedRMSNorm(ObservableMixin, nn.LayerNorm):
    """RMSNorm module — no PyTorch equivalent, inherits from LayerNorm."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 cfg: OpQuantConfig = None, inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True,
                 name: str = None, **kwargs):
        super().__init__(normalized_shape, eps=eps,
                         elementwise_affine=elementwise_affine, **kwargs)
        self.cfg = cfg or OpQuantConfig()
        self.inner_scheme = inner_scheme
        self.quantize_backprop = quantize_backprop
        self._analysis_name = name

    def forward(self, x):
        return RMSNormFunction.apply(
            x, self.weight, self.bias, self.eps,
            self.cfg, self.inner_scheme, self.quantize_backprop, self._analysis_name,
        )
