"""
quantize_model: unified entry point for all-op quantization.

1. Recursively replaces nn.Module subclasses with Quantized* equivalents
   (nn.Conv2d → QuantizedConv2d, nn.BatchNorm2d → QuantizedBatchNorm2d, etc.)
2. Patches model.forward to auto-wrap in QuantizeContext, so inline ops
   (torch.matmul, torch.add, torch.exp, etc.) are also intercepted.

After quantize_model(model, cfg), simply calling model(x) gives fully
quantized forward + backward. model.export_onnx(x, path) is also added.

Module-level ops → QuantizedXxx classes (explicit, cfg baked into module)
Inline ops         → QuantizeContext wrapping (automatic, no model surgery)
Both paths converge at the same XxxFunction.apply() for bit-exact consistency.
"""
import types
from typing import Dict, Optional, Union

import torch.nn as nn

from src.scheme.op_config import OpQuantConfig
from src.context.quantize_context import QuantizeContext

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


def _resolve_context_cfg(cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig], None]) -> OpQuantConfig:
    """Resolve a single OpQuantConfig for QuantizeContext inline-op quantization.

    If cfg is a dict (per-name module configs), inline ops get _EMPTY_CFG
    (passthrough) unless the user also passes op_cfgs for per-op overrides.
    """
    if isinstance(cfg, OpQuantConfig):
        return cfg
    return _EMPTY_CFG


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
# Forward patching for inline-op quantization
# ---------------------------------------------------------------------------

def _patch_forward(
    model: nn.Module,
    ctx_cfg: OpQuantConfig,
    *,
    op_cfgs: Optional[Dict[str, OpQuantConfig]] = None,
    observers: Optional[list] = None,
) -> None:
    """Replace model.forward with a version auto-wrapped in QuantizeContext.

    Also attaches model.export_onnx(dummy_input, path) convenience method.

    Guarded by model._quantize_forward_patched — calling twice is a no-op.
    """
    if getattr(model, '_quantize_forward_patched', False):
        return

    # model.forward at this point is a bound method of the model.
    # Capture it before reassigning model.forward to our wrapper.
    original_forward = model.forward

    def _wrapped_forward(*args, **kwargs):
        with QuantizeContext(
            model,
            ctx_cfg,
            op_cfgs=op_cfgs,
            observers=observers,
        ):
            return original_forward(*args, **kwargs)

    # Assign as a regular function (not MethodType). nn.Module.__call__
    # calls self.forward(*args, **kwargs) — a plain function here
    # receives exactly the user's arguments, no implicit self.
    # original_forward is already a bound method, so calling it with
    # the same args restores the original behaviour.
    model.forward = _wrapped_forward

    # Store for export_onnx to use without re-entering quantize_model
    model._quantize_cfg = ctx_cfg
    model._quantize_op_cfgs = op_cfgs or {}
    model._quantize_observers = observers or []

    def _export_onnx(self, dummy_input, output_path: str, opset_version: int = 17):
        with QuantizeContext(
            self,
            self._quantize_cfg,
            op_cfgs=self._quantize_op_cfgs,
            observers=self._quantize_observers,
        ) as ctx:
            ctx.export_onnx(dummy_input, output_path, opset_version=opset_version)

    model.export_onnx = types.MethodType(_export_onnx, model)
    model._quantize_forward_patched = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig], None] = None,
    *,
    prefix: str = "",
    op_cfgs: Optional[Dict[str, OpQuantConfig]] = None,
    observers: Optional[list] = None,
    _patch_root: bool = True,
) -> nn.Module:
    """Unified entry point: module replacement + inline-op quantization.

    1. Recursively replaces known nn.Module subclasses with Quantized* equivalents
       (module-level ops like Conv, BN, Norm, Activation, Softmax, Pool).
    2. Patches model.forward to auto-wrap in QuantizeContext, intercepting
       inline torch ops (torch.matmul, torch.add, torch.sub, torch.mul,
       torch.div, torch.exp, torch.log) — no separate context manager needed.

    After calling this function, model(x) produces fully-quantized output.
    model.export_onnx(dummy_input, path) is also added for convenience.

    Args:
        model: Root nn.Module to quantize.
        cfg: OpQuantConfig applied to all modules AND inline ops,
             or a dict mapping name patterns ("fc", "conv*") to OpQuantConfig.
             If a dict, inline ops get no quantization (use op_cfgs for that).
        op_cfgs: Optional per-op-type overrides for inline ops only.
                 Valid keys: "matmul", "mm", "bmm", "linear",
                 "add", "sub", "mul", "div", "exp", "log".
        observers: Optional observers for analysis (same as QuantizeContext).
        prefix: Internal. For nested child naming in recursive calls.
        _patch_root: Internal. Set False to skip forward patching (recursive).

    Returns:
        The same model object, with modules replaced in-place and
        forward patched for inline-op quantization.
    """
    if cfg is None:
        cfg = _EMPTY_CFG

    # Step 1: Replace module subclasses in-place
    for child_name, child in list(model.named_children()):
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        quantized_child = _replace_module(child, cfg, child_prefix)
        if quantized_child is not None:
            setattr(model, child_name, quantized_child)
        elif isinstance(child, nn.Module):
            quantize_model(child, cfg, prefix=child_prefix,
                           op_cfgs=op_cfgs, observers=observers,
                           _patch_root=False)

    # Step 2: Patch forward on the root model only
    if _patch_root:
        ctx_cfg = _resolve_context_cfg(cfg)
        _patch_forward(model, ctx_cfg, op_cfgs=op_cfgs, observers=observers)

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
