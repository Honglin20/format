"""
quantize_model: recursively replace nn.Module subclasses with Quantized* equivalents.

Matches the behaviour of mx/mx_mapping.py but uses OpQuantConfig instead of mx_specs.
"""
from typing import Dict, Optional, Union

import torch.nn as nn

from src.scheme.op_config import OpQuantConfig

# ---------------------------------------------------------------------------
# Module type → Quantized constructor + param extractor
# ---------------------------------------------------------------------------

def _make_linear(orig: nn.Linear, cfg: OpQuantConfig, name: str):
    from src.ops.linear import QuantizedLinear
    return QuantizedLinear(
        in_features=orig.in_features, out_features=orig.out_features,
        bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_conv1d(orig: nn.Conv1d, cfg: OpQuantConfig, name: str):
    from src.ops.conv import QuantizedConv1d
    return QuantizedConv1d(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding,
        dilation=orig.dilation, groups=orig.groups,
        bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_conv2d(orig: nn.Conv2d, cfg: OpQuantConfig, name: str):
    from src.ops.conv import QuantizedConv2d
    return QuantizedConv2d(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding,
        dilation=orig.dilation, groups=orig.groups,
        bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_conv3d(orig: nn.Conv3d, cfg: OpQuantConfig, name: str):
    from src.ops.conv import QuantizedConv3d
    return QuantizedConv3d(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding,
        dilation=orig.dilation, groups=orig.groups,
        bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_conv_transpose1d(orig: nn.ConvTranspose1d, cfg: OpQuantConfig, name: str):
    from src.ops.conv import QuantizedConvTranspose1d
    return QuantizedConvTranspose1d(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding,
        output_padding=orig.output_padding, dilation=orig.dilation,
        groups=orig.groups, bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_conv_transpose2d(orig: nn.ConvTranspose2d, cfg: OpQuantConfig, name: str):
    from src.ops.conv import QuantizedConvTranspose2d
    return QuantizedConvTranspose2d(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding,
        output_padding=orig.output_padding, dilation=orig.dilation,
        groups=orig.groups, bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_conv_transpose3d(orig: nn.ConvTranspose3d, cfg: OpQuantConfig, name: str):
    from src.ops.conv import QuantizedConvTranspose3d
    return QuantizedConvTranspose3d(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding,
        output_padding=orig.output_padding, dilation=orig.dilation,
        groups=orig.groups, bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_bn1d(orig: nn.BatchNorm1d, cfg: OpQuantConfig, name: str):
    from src.ops.norm import QuantizedBatchNorm1d
    mod = QuantizedBatchNorm1d(
        num_features=orig.num_features, eps=orig.eps,
        momentum=orig.momentum, affine=orig.affine,
        track_running_stats=orig.track_running_stats, cfg=cfg, name=name,
    )
    _copy_bn_state(orig, mod)
    return mod


def _make_bn2d(orig: nn.BatchNorm2d, cfg: OpQuantConfig, name: str):
    from src.ops.norm import QuantizedBatchNorm2d
    mod = QuantizedBatchNorm2d(
        num_features=orig.num_features, eps=orig.eps,
        momentum=orig.momentum, affine=orig.affine,
        track_running_stats=orig.track_running_stats, cfg=cfg, name=name,
    )
    _copy_bn_state(orig, mod)
    return mod


def _make_bn3d(orig: nn.BatchNorm3d, cfg: OpQuantConfig, name: str):
    from src.ops.norm import QuantizedBatchNorm3d
    mod = QuantizedBatchNorm3d(
        num_features=orig.num_features, eps=orig.eps,
        momentum=orig.momentum, affine=orig.affine,
        track_running_stats=orig.track_running_stats, cfg=cfg, name=name,
    )
    _copy_bn_state(orig, mod)
    return mod


def _copy_bn_state(orig, target):
    if orig.affine:
        target.weight.data.copy_(orig.weight.data)
        target.bias.data.copy_(orig.bias.data)
    if orig.track_running_stats:
        target.running_mean.copy_(orig.running_mean)
        target.running_var.copy_(orig.running_var)


def _make_ln(orig: nn.LayerNorm, cfg: OpQuantConfig, name: str):
    from src.ops.norm import QuantizedLayerNorm
    normalized_shape = orig.normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    mod = QuantizedLayerNorm(
        normalized_shape=list(normalized_shape), eps=orig.eps,
        elementwise_affine=orig.elementwise_affine, cfg=cfg, name=name,
    )
    if orig.elementwise_affine:
        mod.weight.data.copy_(orig.weight.data)
        mod.bias.data.copy_(orig.bias.data)
    return mod


def _make_gn(orig: nn.GroupNorm, cfg: OpQuantConfig, name: str):
    from src.ops.norm import QuantizedGroupNorm
    mod = QuantizedGroupNorm(
        num_groups=orig.num_groups, num_channels=orig.num_channels,
        eps=orig.eps, affine=orig.affine, cfg=cfg, name=name,
    )
    if orig.affine:
        mod.weight.data.copy_(orig.weight.data)
        mod.bias.data.copy_(orig.bias.data)
    return mod


def _make_rms_norm(orig, cfg: OpQuantConfig, name: str):
    from src.ops.norm import QuantizedRMSNorm
    normalized_shape = orig.normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    mod = QuantizedRMSNorm(
        normalized_shape=list(normalized_shape), eps=orig.eps,
        elementwise_affine=orig.elementwise_affine, cfg=cfg, name=name,
    )
    if orig.elementwise_affine:
        mod.weight.data.copy_(orig.weight.data)
    return mod


# --- Activation constructors ---

def _make_sigmoid(orig: nn.Sigmoid, cfg: OpQuantConfig, name: str):
    from src.ops.activations import QuantizedSigmoid
    return QuantizedSigmoid(cfg=cfg, name=name)


def _make_tanh(orig: nn.Tanh, cfg: OpQuantConfig, name: str):
    from src.ops.activations import QuantizedTanh
    return QuantizedTanh(cfg=cfg, name=name)


def _make_relu(orig: nn.ReLU, cfg: OpQuantConfig, name: str):
    from src.ops.activations import QuantizedReLU
    return QuantizedReLU(inplace=orig.inplace, cfg=cfg, name=name)


def _make_relu6(orig: nn.ReLU6, cfg: OpQuantConfig, name: str):
    from src.ops.activations import QuantizedReLU6
    return QuantizedReLU6(inplace=orig.inplace, cfg=cfg, name=name)


def _make_leaky_relu(orig: nn.LeakyReLU, cfg: OpQuantConfig, name: str):
    from src.ops.activations import QuantizedLeakyReLU
    return QuantizedLeakyReLU(
        negative_slope=orig.negative_slope, inplace=orig.inplace,
        cfg=cfg, name=name,
    )


def _make_silu(orig: nn.SiLU, cfg: OpQuantConfig, name: str):
    from src.ops.activations import QuantizedSiLU
    return QuantizedSiLU(inplace=orig.inplace, cfg=cfg, name=name)


def _make_gelu(orig: nn.GELU, cfg: OpQuantConfig, name: str):
    from src.ops.activations import QuantizedGELU
    return QuantizedGELU(cfg=cfg, name=name)


def _make_softmax(orig: nn.Softmax, cfg: OpQuantConfig, name: str):
    from src.ops.softmax import QuantizedSoftmax
    return QuantizedSoftmax(dim=orig.dim, cfg=cfg, name=name)


def _make_adaptive_avg_pool2d(orig: nn.AdaptiveAvgPool2d, cfg: OpQuantConfig, name: str):
    from src.ops.pooling import QuantizedAdaptiveAvgPool2d
    return QuantizedAdaptiveAvgPool2d(
        output_size=orig.output_size, cfg=cfg, name=name,
    )


# --- Empty passthrough (no quantization) ---
_EMPTY_CFG = OpQuantConfig()


# ---------------------------------------------------------------------------
# Module mapping table
# ---------------------------------------------------------------------------

_MODULE_MAPPING = {
    nn.Linear: _make_linear,
    nn.Conv1d: _make_conv1d,
    nn.Conv2d: _make_conv2d,
    nn.Conv3d: _make_conv3d,
    nn.ConvTranspose1d: _make_conv_transpose1d,
    nn.ConvTranspose2d: _make_conv_transpose2d,
    nn.ConvTranspose3d: _make_conv_transpose3d,
    nn.BatchNorm1d: _make_bn1d,
    nn.BatchNorm2d: _make_bn2d,
    nn.BatchNorm3d: _make_bn3d,
    nn.LayerNorm: _make_ln,
    nn.GroupNorm: _make_gn,
    nn.Sigmoid: _make_sigmoid,
    nn.Tanh: _make_tanh,
    nn.ReLU: _make_relu,
    nn.ReLU6: _make_relu6,
    nn.LeakyReLU: _make_leaky_relu,
    nn.SiLU: _make_silu,
    nn.GELU: _make_gelu,
    nn.Softmax: _make_softmax,
    nn.AdaptiveAvgPool2d: _make_adaptive_avg_pool2d,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig], None] = None,
    *,
    prefix: str = "",
) -> nn.Module:
    """Recursively replace nn.Module subclasses with Quantized* equivalents.

    Args:
        model: Root nn.Module to quantize.
        cfg: OpQuantConfig for all modules, or a dict mapping name patterns
             (e.g. "fc" or "conv*") to OpQuantConfig.
        prefix: Internal prefix for nested child naming.

    Returns:
        The model with known modules replaced in-place (same model object).
    """
    if cfg is None:
        cfg = _EMPTY_CFG

    # Replace children in-place
    for child_name, child in list(model.named_children()):
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        quantized_child = _replace_module(child, cfg, child_prefix)
        if quantized_child is not None:
            setattr(model, child_name, quantized_child)
        elif isinstance(child, nn.Module):
            quantize_model(child, cfg, prefix=child_prefix)

    return model


def _resolve_cfg(cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig]], name: str) -> OpQuantConfig:
    """Resolve per-module cfg from a dict or singleton."""
    if isinstance(cfg, OpQuantConfig):
        return cfg
    if isinstance(cfg, dict):
        # Exact match first, then wildcard pattern match
        if name in cfg:
            return cfg[name]
        for pattern, c in cfg.items():
            if _pattern_match(name, pattern):
                return c
    return _EMPTY_CFG


def _pattern_match(name: str, pattern: str) -> bool:
    """Simple glob-style matching: 'conv*' matches 'conv1', 'conv_blocks.0', etc."""
    if pattern.endswith("*"):
        return name == pattern[:-1] or name.startswith(pattern[:-1])
    return name == pattern


def _replace_module(
    module: nn.Module,
    cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig]],
    name: str,
):
    """Replace a single module with its quantized version, or return None."""
    # Skip if already quantized (has cfg attribute)
    if hasattr(module, "cfg"):
        return None

    make_fn = _MODULE_MAPPING.get(type(module))
    if make_fn is None:
        return None

    resolved_cfg = _resolve_cfg(cfg, name)
    if resolved_cfg == _EMPTY_CFG and resolved_cfg != cfg:
        # Explicit empty cfg but the user may want passthrough
        pass

    mod = make_fn(module, resolved_cfg, name)

    # Copy weights for modules that have state_dict
    if hasattr(mod, "load_state_dict"):
        # Only copy if the old module also had weights
        try:
            mod.load_state_dict(module.state_dict(), strict=False)
        except Exception:
            pass

    return mod
