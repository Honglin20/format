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
from typing import Dict, List, Optional, Union

import torch.nn as nn

from src.scheme.op_config import OpQuantConfig
from src.session._context import QuantizeContext, _EMPTY_CFG
from src.ops.conv import (
    QuantizedConv1d,
    QuantizedConv2d,
    QuantizedConv3d,
    QuantizedConvTranspose1d,
    QuantizedConvTranspose2d,
    QuantizedConvTranspose3d,
)
from src.ops.norm import QuantizedBatchNorm1d, QuantizedBatchNorm2d, QuantizedBatchNorm3d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_inner_scheme(cfg: OpQuantConfig):
    """Extract inner scheme for norm ops (quantizing every intermediate step).

    - If ``cfg.storage`` exists (bf16/fp8 storage), use it — MX with storage
      quantizes every intermediate result.
    - If ``cfg.storage`` is None and ``cfg.input`` is a per_tensor elemwise
      scheme (compat-style config), use it — input carries the elemwise scheme.
    - If ``cfg.storage`` is None and ``cfg.input`` is a per_block MX compute
      scheme (MX bfloat=0), return None — MX without storage runs norms in fp32.
    """
    if cfg.storage is not None:
        return cfg.storage
    if cfg.input is not None and cfg.input.granularity.mode.name == "PER_TENSOR":
        return cfg.input
    return None


def _is_mx_compute(scheme) -> bool:
    """True if a scheme is MX per_block compute, not elemwise storage."""
    if scheme is None:
        return False
    return scheme.granularity.mode.name == "PER_BLOCK"


def _non_matmul_cfg(cfg: OpQuantConfig) -> OpQuantConfig:
    """Derive an OpQuantConfig for norm ops — strip MX compute, keep elemwise.

    Three cases:
    - Two-level model with storage (e.g. bf16): keep only storage + backward fields.
    - Compat-style config where input carries per_tensor elemwise scheme:
      pass through unchanged (no separate storage field).
    - MX with bfloat=0 (storage=None, input is per_block MX compute):
      return empty — MX applies identity to all vec_* / elemwise operations.
    """
    if cfg.storage is not None:
        return OpQuantConfig(
            storage=cfg.storage,
            grad_output=cfg.grad_output or cfg.storage,
            grad_input=cfg.grad_input or cfg.storage,
            grad_weight=cfg.grad_weight or cfg.storage,
            grad_bias=cfg.grad_bias or cfg.storage,
        )
    # No storage: either compat-style (input is per_tensor elemwise) or MX bfloat=0
    if _is_mx_compute(cfg.input) or _is_mx_compute(cfg.weight):
        return OpQuantConfig()  # MX bfloat=0 — no quantization for non-matmul ops
    return cfg  # compat-style: input/weight carry per_tensor elemwise schemes


def _activation_cfg(cfg: OpQuantConfig) -> OpQuantConfig:
    """Derive an OpQuantConfig for activation/softmax/pool ops — strip MX compute.

    Three cases (same discrimination as ``_non_matmul_cfg``).
    """
    if cfg.storage is not None:
        return OpQuantConfig(
            storage=cfg.storage,
            input=cfg.storage,
            grad_input=cfg.storage,
        )
    if _is_mx_compute(cfg.input) or _is_mx_compute(cfg.weight):
        return OpQuantConfig()  # MX bfloat=0 — no quantization for activation/softmax
    return cfg  # compat-style: input carries per_tensor elemwise scheme


def _nonlinear_true_cfg(cfg: OpQuantConfig) -> OpQuantConfig:
    """Derive an OpQuantConfig that keeps per_block compute for operand entry.

    Used when ``quantize_nonlinear=True`` — norm/activation/pool operands
    receive the same storage -> per_block two-level quantization as matmul ops,
    while backward fields stay storage-only.

    Three cases:
    - Two-level model with storage: keep storage + input/weight/bias per_block,
      populate backward from storage.
    - MX with bfloat=0 (storage=None, input is per_block MX compute):
      keep per_block compute, backward stays None.
    - Compat-style config where input carries per_tensor elemwise scheme:
      pass through unchanged (no separate compute quant to add).
    """
    if cfg.storage is not None:
        return OpQuantConfig(
            storage=cfg.storage,
            input=cfg.input,              # per_block compute kept
            weight=cfg.weight,            # per_block compute kept
            bias=cfg.bias,                # per_block compute kept
            grad_output=cfg.grad_output or cfg.storage,
            grad_input=cfg.grad_input or cfg.storage,
            grad_weight=cfg.grad_weight or cfg.storage,
            grad_bias=cfg.grad_bias or cfg.storage,
        )
    # No storage: either compat-style (input is per_tensor elemwise) or MX bfloat=0
    if _is_mx_compute(cfg.input) or _is_mx_compute(cfg.weight):
        # MX bfloat=0: keep compute, backward stays None
        return OpQuantConfig(
            input=cfg.input,
            weight=cfg.weight,
            bias=cfg.bias,
        )
    return cfg  # compat-style: input/weight carry per_tensor elemwise schemes


# ---------------------------------------------------------------------------
# Module type → Quantized constructor + param extractor
# ---------------------------------------------------------------------------

def _make_linear(orig: nn.Linear, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.linear import QuantizedLinear
    return QuantizedLinear(
        in_features=orig.in_features, out_features=orig.out_features,
        bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_conv(orig, cfg, name, conv_cls, quantize_nonlinear=False):
    """Generic factory for QuantizedConv{1,2,3}d."""
    return conv_cls(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride,
        padding=orig.padding, dilation=orig.dilation,
        groups=orig.groups, bias=orig.bias is not None,
        cfg=cfg, name=name,
    )


def _make_conv_transpose(orig, cfg, name, conv_cls, quantize_nonlinear=False):
    """Generic factory for QuantizedConvTranspose{1,2,3}d."""
    return conv_cls(
        in_channels=orig.in_channels, out_channels=orig.out_channels,
        kernel_size=orig.kernel_size, stride=orig.stride,
        padding=orig.padding, output_padding=orig.output_padding,
        dilation=orig.dilation, groups=orig.groups,
        bias=orig.bias is not None, cfg=cfg, name=name,
    )


def _make_bn(orig, cfg, name, bn_cls, quantize_nonlinear=False):
    """Generic factory for QuantizedBatchNorm{1,2,3}d."""
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = bn_cls(
        num_features=orig.num_features, eps=orig.eps,
        momentum=orig.momentum, affine=orig.affine,
        track_running_stats=orig.track_running_stats,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg), quantize_backprop=norm_cfg.is_training, name=name,
    )
    _copy_bn_state(orig, mod)
    return mod


def _copy_bn_state(orig, target):
    if orig.affine:
        target.weight.data = orig.weight.data.clone()
        target.bias.data = orig.bias.data.clone()
    if orig.track_running_stats:
        target.running_mean = orig.running_mean.clone()
        target.running_var = orig.running_var.clone()


def _make_ln(orig: nn.LayerNorm, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.norm import QuantizedLayerNorm
    normalized_shape = orig.normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = QuantizedLayerNorm(
        normalized_shape=list(normalized_shape), eps=orig.eps,
        elementwise_affine=orig.elementwise_affine,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg), quantize_backprop=norm_cfg.is_training, name=name,
    )
    if orig.elementwise_affine:
        mod.weight.data = orig.weight.data.clone()
        mod.bias.data = orig.bias.data.clone()
    return mod


def _make_gn(orig: nn.GroupNorm, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.norm import QuantizedGroupNorm
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = QuantizedGroupNorm(
        num_groups=orig.num_groups, num_channels=orig.num_channels,
        eps=orig.eps, affine=orig.affine,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg), quantize_backprop=norm_cfg.is_training, name=name,
    )
    if orig.affine:
        mod.weight.data = orig.weight.data.clone()
        mod.bias.data = orig.bias.data.clone()
    return mod


def _make_rms_norm(orig, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.norm import QuantizedRMSNorm
    normalized_shape = orig.normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = QuantizedRMSNorm(
        normalized_shape=list(normalized_shape), eps=orig.eps,
        elementwise_affine=orig.elementwise_affine,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg), quantize_backprop=norm_cfg.is_training, name=name,
    )
    if orig.elementwise_affine:
        mod.weight.data = orig.weight.data.clone()
    return mod


# --- Activation constructors ---

def _make_sigmoid(orig: nn.Sigmoid, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedSigmoid
    return QuantizedSigmoid(cfg=_activation_cfg(cfg), name=name)


def _make_tanh(orig: nn.Tanh, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedTanh
    return QuantizedTanh(cfg=_activation_cfg(cfg), name=name)


def _make_relu(orig: nn.ReLU, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedReLU
    return QuantizedReLU(inplace=orig.inplace, cfg=_activation_cfg(cfg), name=name)


def _make_relu6(orig: nn.ReLU6, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedReLU6
    return QuantizedReLU6(inplace=orig.inplace, cfg=_activation_cfg(cfg), name=name)


def _make_leaky_relu(orig: nn.LeakyReLU, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedLeakyReLU
    return QuantizedLeakyReLU(
        negative_slope=orig.negative_slope, inplace=orig.inplace,
        cfg=_activation_cfg(cfg), name=name,
    )


def _make_silu(orig: nn.SiLU, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedSiLU
    return QuantizedSiLU(inplace=orig.inplace, cfg=_activation_cfg(cfg), name=name)


def _make_gelu(orig: nn.GELU, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedGELU
    return QuantizedGELU(cfg=_activation_cfg(cfg), name=name)


def _make_softmax(orig: nn.Softmax, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.softmax import QuantizedSoftmax
    return QuantizedSoftmax(dim=orig.dim, cfg=_activation_cfg(cfg), name=name)


def _make_adaptive_avg_pool2d(orig: nn.AdaptiveAvgPool2d, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.pooling import QuantizedAdaptiveAvgPool2d
    return QuantizedAdaptiveAvgPool2d(
        output_size=orig.output_size, cfg=_activation_cfg(cfg), name=name,
    )


def _get_quantized_modules(model: nn.Module) -> List[tuple]:
    """Return [(name, module), ...] for all Quantized* modules with cfg."""
    result = []
    for name, module in model.named_modules():
        if hasattr(module, "cfg") and not getattr(module, "_is_passthrough", False):
            result.append((name, module))
    return result


def _resolve_context_cfg(
    cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig], None],
    op_cfgs: Optional[Dict[str, OpQuantConfig]] = None,
) -> OpQuantConfig:
    """Resolve a single OpQuantConfig for QuantizeContext inline-op quantization.

    When cfg is a singleton OpQuantConfig, it is passed through ``_non_matmul_cfg``
    to strip MX per_block compute — SIMD/non-linear ops only receive elemwise
    (storage) quantization, matching MX architecture.  matmul-family inline ops
    get the full config via ``op_cfgs`` (auto-populated in ``quantize_model``).

    When cfg is a dict (per-module configs), the storage scheme from the first
    config that has one is extracted as the default for inline ops.
    """
    if isinstance(cfg, OpQuantConfig):
        # When MX per_block compute is present, strip it from the default cfg.
        # MX never applies per_block compute to SIMD/non-linear operations — only
        # elemwise storage. Use the same discrimination as _non_matmul_cfg:
        #   - storage present → keep storage only
        #   - MX per_block → empty (bfloat=0 → identity)
        #   - compat-style per_tensor → pass through unchanged
        if cfg.storage is not None or _is_mx_compute(cfg.input) or _is_mx_compute(cfg.weight):
            return _non_matmul_cfg(cfg)
        return cfg
    # cfg is a dict — extract storage (or per_tensor input) for inline-op defaults
    storage = None
    if cfg:
        for c in cfg.values():
            if not isinstance(c, OpQuantConfig):
                continue
            if c.storage is not None:
                storage = c.storage
                break
            # Fallback: compat-style configs store the elemwise scheme in `input`
            if (c.input is not None
                    and c.input.granularity is not None
                    and c.input.granularity.mode.name == "PER_TENSOR"):
                storage = c.input
                break
    return OpQuantConfig(storage=storage)


# ---------------------------------------------------------------------------
# Module type classification
# ---------------------------------------------------------------------------

_MATMUL_TYPES = (
    nn.Linear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)

# ---------------------------------------------------------------------------
# Module mapping table
# ---------------------------------------------------------------------------

_MODULE_MAPPING = {
    nn.Linear: lambda orig, cfg, name, **kw: _make_linear(orig, cfg, name, **kw),
    nn.Conv1d: lambda orig, cfg, name, **kw: _make_conv(orig, cfg, name, QuantizedConv1d, **kw),
    nn.Conv2d: lambda orig, cfg, name, **kw: _make_conv(orig, cfg, name, QuantizedConv2d, **kw),
    nn.Conv3d: lambda orig, cfg, name, **kw: _make_conv(orig, cfg, name, QuantizedConv3d, **kw),
    nn.ConvTranspose1d: lambda orig, cfg, name, **kw: _make_conv_transpose(orig, cfg, name, QuantizedConvTranspose1d, **kw),
    nn.ConvTranspose2d: lambda orig, cfg, name, **kw: _make_conv_transpose(orig, cfg, name, QuantizedConvTranspose2d, **kw),
    nn.ConvTranspose3d: lambda orig, cfg, name, **kw: _make_conv_transpose(orig, cfg, name, QuantizedConvTranspose3d, **kw),
    nn.BatchNorm1d: lambda orig, cfg, name, **kw: _make_bn(orig, cfg, name, QuantizedBatchNorm1d, **kw),
    nn.BatchNorm2d: lambda orig, cfg, name, **kw: _make_bn(orig, cfg, name, QuantizedBatchNorm2d, **kw),
    nn.BatchNorm3d: lambda orig, cfg, name, **kw: _make_bn(orig, cfg, name, QuantizedBatchNorm3d, **kw),
    nn.LayerNorm: lambda orig, cfg, name, **kw: _make_ln(orig, cfg, name, **kw),
    nn.GroupNorm: lambda orig, cfg, name, **kw: _make_gn(orig, cfg, name, **kw),
    nn.Sigmoid: lambda orig, cfg, name, **kw: _make_sigmoid(orig, cfg, name, **kw),
    nn.Tanh: lambda orig, cfg, name, **kw: _make_tanh(orig, cfg, name, **kw),
    nn.ReLU: lambda orig, cfg, name, **kw: _make_relu(orig, cfg, name, **kw),
    nn.ReLU6: lambda orig, cfg, name, **kw: _make_relu6(orig, cfg, name, **kw),
    nn.LeakyReLU: lambda orig, cfg, name, **kw: _make_leaky_relu(orig, cfg, name, **kw),
    nn.SiLU: lambda orig, cfg, name, **kw: _make_silu(orig, cfg, name, **kw),
    nn.GELU: lambda orig, cfg, name, **kw: _make_gelu(orig, cfg, name, **kw),
    nn.Softmax: lambda orig, cfg, name, **kw: _make_softmax(orig, cfg, name, **kw),
    nn.AdaptiveAvgPool2d: lambda orig, cfg, name, **kw: _make_adaptive_avg_pool2d(orig, cfg, name, **kw),
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
    quantize_nonlinear: bool = True,
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
        quantize_nonlinear: If False, norm/activation/pool modules are skipped
            and remain in fp32. Default True.
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
        quantized_child = _replace_module(
            child, cfg, child_prefix, quantize_nonlinear=quantize_nonlinear,
        )
        if quantized_child is not None:
            setattr(model, child_name, quantized_child)
        elif isinstance(child, nn.Module):
            quantize_model(child, cfg, prefix=child_prefix,
                           op_cfgs=op_cfgs, observers=observers,
                           quantize_nonlinear=quantize_nonlinear,
                           _patch_root=False)

    # Step 2: Patch forward on the root model only
    if _patch_root:
        ctx_cfg = _resolve_context_cfg(cfg, op_cfgs)

        # When cfg has MX per_block compute, auto-populate matmul-family op_cfgs
        # with the full config. The default ctx_cfg is storage-only (strip MX
        # compute for SIMD/non-linear ops), so matmul inline ops need explicit
        # entries to receive MX per_block quantization.
        _final_op_cfgs = dict(op_cfgs) if op_cfgs else {}
        if isinstance(cfg, OpQuantConfig) and (_is_mx_compute(cfg.input) or _is_mx_compute(cfg.weight)):
            for matmul_op in ("matmul", "mm", "bmm", "linear"):
                if matmul_op not in _final_op_cfgs:
                    _final_op_cfgs[matmul_op] = cfg
        _patch_forward(model, ctx_cfg, op_cfgs=_final_op_cfgs or None, observers=observers)

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
    *,
    quantize_nonlinear: bool = True,
):
    """Replace a single module with its quantized version, or return None.

    When *quantize_nonlinear* is False, non-linear modules still get replaced — but
    the ``_make_*`` functions derive a storage-only ``OpQuantConfig`` via
    ``_non_matmul_cfg`` / ``_activation_cfg``, stripping MX per_block compute so
    only elemwise (storage) quantization is applied.  This matches MX architecture
    where non-linear ops only go through ``quantize_elemwise_op``.

    When *quantize_nonlinear* is True (default), the same storage-only derivation
    applies today.  The flag is reserved for future "extra quantization" steps
    beyond MX (e.g. applying MX per_block compute to non-linear ops).
    """
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

    mod = make_fn(module, resolved_cfg, name, quantize_nonlinear=quantize_nonlinear)

    # Copy weights for modules that have state_dict
    if hasattr(mod, "load_state_dict"):
        # Only copy if the old module also had weights
        try:
            mod.load_state_dict(module.state_dict(), strict=False)
        except Exception:
            pass

    # Preserve device of original module
    try:
        device = next(module.parameters()).device
    except StopIteration:
        try:
            device = next(module.buffers()).device
        except StopIteration:
            device = None
    if device is not None and device.type != 'cpu':
        mod = mod.to(device)

    return mod
