"""Adaptive transform selection: per-layer QSNR-based transform choice.

Hook → forward → estimate QSNR per candidate → pick best per layer.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ops.conv import QuantizedConv2d
from src.ops.linear import QuantizedLinear
from src.quantize import quantize as _quantize_fn
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform
from src.session._config import QuantConfig
from src.session._helpers import (
    _SMOOTH_INPUT_ROLES,
    _run_model,
)
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import (
    SmoothQuantTransform,
    compute_smoothquant_scale,
)

_logger = logging.getLogger(__name__)

_MATMUL_QUANTIZED_TYPES = (QuantizedLinear, QuantizedConv2d)


def _estimate_layer_qsnr(
    module,
    x_act: torch.Tensor,
    W: torch.Tensor,
    base_act,
    base_w,
    candidate: str,
    sq_alpha: float,
    weight_only: bool,
) -> float:
    """Estimate matmul-output QSNR (dB) for a single transform candidate."""
    import math

    if candidate == "none":
        act_tx = IdentityTransform()
        w_tx = IdentityTransform()
        W_est = W
    elif candidate == "hadamard":
        act_tx = HadamardTransform()
        w_tx = HadamardTransform()
        W_est = W
    elif candidate == "smoothquant":
        act_axis = -1 if isinstance(module, QuantizedLinear) else 1
        scale = compute_smoothquant_scale(
            x_act, W, alpha=sq_alpha,
            act_channel_axis=act_axis, w_channel_axis=1,
        )
        act_tx = SmoothQuantTransform(scale, channel_axis=act_axis)
        w_tx = IdentityTransform()
        shape = [1] * W.ndim
        shape[1] = -1
        W_est = W * scale.to(W.device).view(*shape)
    else:
        return -float("inf")

    if not weight_only:
        act_scheme = QuantScheme(
            format=base_act.format,
            granularity=base_act.granularity,
            transform=act_tx,
            round_mode=base_act.round_mode,
            scale_storage=base_act.scale_storage,
        )
    w_scheme = QuantScheme(
        format=base_w.format,
        granularity=base_w.granularity,
        transform=w_tx,
        round_mode=base_w.round_mode,
        scale_storage=base_w.scale_storage,
    )

    try:
        if weight_only:
            W_q = _quantize_fn(W_est, w_scheme)
            mse = (W_q - W_est).pow(2).mean()
            signal = W_est.pow(2).mean()
        elif isinstance(module, QuantizedLinear):
            x_q = _quantize_fn(x_act, act_scheme)
            W_q = _quantize_fn(W_est, w_scheme)
            y_ref = x_act.to(torch.float32) @ W_est.to(torch.float32).T
            y_q = x_q.to(torch.float32) @ W_q.to(torch.float32).T
            mse = (y_q - y_ref).pow(2).mean()
            signal = y_ref.pow(2).mean()
        elif isinstance(module, QuantizedConv2d):
            x_q = _quantize_fn(x_act, act_scheme)
            W_q = _quantize_fn(W_est, w_scheme)
            y_ref = F.conv2d(
                x_act.to(torch.float32), W_est.to(torch.float32),
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
            )
            y_q = F.conv2d(
                x_q.to(torch.float32), W_q.to(torch.float32),
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
            )
            mse = (y_q - y_ref).pow(2).mean()
            signal = y_ref.pow(2).mean()
        else:
            return -float("inf")

        return 10.0 * math.log10(
            max(signal.item(), 1e-12) / max(mse.item(), 1e-12)
        )
    except (RuntimeError, ValueError) as exc:
        _logger.debug(
            "adaptive: QSNR estimation failed for %s on layer %s: %s",
            candidate, getattr(module, "name", "?"), exc,
        )
        return -float("inf")


def _apply_layer_selection(module, tx: str, sq_scale, applied: dict) -> None:
    """Patch *module.cfg* to use the chosen *tx* and fuse SQ weights."""
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        return

    if tx == "none":
        applied["none"] += 1
        return

    _fields = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}

    if tx == "hadamard":
        if cfg.input is not None:
            _fields["input"] = QuantScheme(
                format=cfg.input.format,
                granularity=cfg.input.granularity,
                transform=HadamardTransform(),
                round_mode=cfg.input.round_mode,
                scale_storage=cfg.input.scale_storage,
            )
        _fields["weight"] = QuantScheme(
            format=cfg.weight.format,
            granularity=cfg.weight.granularity,
            transform=HadamardTransform(),
            round_mode=cfg.weight.round_mode,
            scale_storage=cfg.weight.scale_storage,
        )
        module.cfg = OpQuantConfig(**_fields)
        applied["hadamard"] += 1
    elif tx == "smoothquant":
        if sq_scale is None:
            applied["none"] += 1
            return
        act_axis = -1 if isinstance(module, QuantizedLinear) else 1
        sq_t = SmoothQuantTransform(sq_scale, channel_axis=act_axis)
        for role in _SMOOTH_INPUT_ROLES:
            if _fields.get(role) is not None:
                _fields[role] = QuantScheme(
                    format=_fields[role].format,
                    granularity=_fields[role].granularity,
                    transform=sq_t,
                    round_mode=_fields[role].round_mode,
                    scale_storage=_fields[role].scale_storage,
                )
        if hasattr(module, "weight") and module.weight is not None:
            shape = [1] * module.weight.ndim
            shape[1] = -1
            module.weight.data = module.weight.data * sq_scale.to(
                module.weight.device
            ).view(*shape)
        _fields["weight"] = QuantScheme(
            format=cfg.weight.format,
            granularity=cfg.weight.granularity,
            transform=IdentityTransform(),
            round_mode=cfg.weight.round_mode,
            scale_storage=cfg.weight.scale_storage,
        )
        module.cfg = OpQuantConfig(**_fields)
        applied["smoothquant"] += 1


def adaptive_transform_selection(
    qmodel: nn.Module,
    base_config: "QuantConfig",
    calib_data,
    eval_fn: Optional[Callable] = None,
) -> Dict[str, int]:
    """Hook → forward → estimate QSNR per candidate → pick best transform per layer.

    Returns counts of modules assigned to each transform
    (keys: ``"none"``, ``"hadamard"``, ``"smoothquant"``).
    """
    activations: Dict[str, torch.Tensor] = {}

    def _make_hook(layer_name: str):
        def _fn(_module, inp, _out):
            activations[layer_name] = inp[0].detach()
        return _fn

    hooks = []
    for name, module in qmodel.named_modules():
        if not isinstance(module, _MATMUL_QUANTIZED_TYPES):
            continue
        cfg = getattr(module, "cfg", None)
        if cfg is None:
            continue
        if cfg.input is None and cfg.weight is None:
            continue
        hooks.append(module.register_forward_hook(_make_hook(name)))

    if not hooks:
        return {"none": 0, "hadamard": 0, "smoothquant": 0}

    try:
        with torch.no_grad():
            _run_model(qmodel, calib_data, eval_fn)
    finally:
        for h in hooks:
            h.remove()

    if not activations:
        return {"none": 0, "hadamard": 0, "smoothquant": 0}

    module_map = dict(qmodel.named_modules())
    per_layer_choice: Dict[str, tuple] = {}

    for name, x_act in activations.items():
        module = module_map.get(name)
        if module is None or not hasattr(module, "weight") or module.weight is None:
            continue
        cfg = getattr(module, "cfg", None)
        if cfg is None:
            continue

        W = module.weight.data.detach()
        base_act = cfg.input
        base_w = cfg.weight
        weight_only = base_act is None

        best_qsnr = -float("inf")
        best_tx = "none"
        best_sq_scale = None

        for candidate in ("none", "hadamard", "smoothquant"):
            if weight_only and candidate == "smoothquant":
                continue

            qsnr = _estimate_layer_qsnr(
                module, x_act, W, base_act, base_w,
                candidate=candidate,
                sq_alpha=base_config.sq_alpha,
                weight_only=weight_only,
            )
            if qsnr > best_qsnr:
                best_qsnr = qsnr
                best_tx = candidate
                if candidate == "smoothquant":
                    act_axis = -1 if isinstance(module, QuantizedLinear) else 1
                    best_sq_scale = compute_smoothquant_scale(
                        x_act, W, alpha=base_config.sq_alpha,
                        act_channel_axis=act_axis, w_channel_axis=1,
                    )

        per_layer_choice[name] = (best_tx, best_sq_scale)

    applied = {"none": 0, "hadamard": 0, "smoothquant": 0}

    for name, (tx, sq_scale) in per_layer_choice.items():
        module = module_map.get(name)
        if module is None:
            continue
        _apply_layer_selection(module, tx, sq_scale, applied)

    return applied
