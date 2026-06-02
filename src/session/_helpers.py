"""Shared helper functions extracted from Session / _QuantSession.

These are used by _per_layer_opt, calibration, analysis, and user code.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.analysis.observers import (
    DistributionFitObserver,
    DistributionObserver,
    HistogramObserver,
    MSEObserver,
    QSNRObserver,
)
from src.calibration.strategies import (
    KLScaleStrategy,
    MSEScaleStrategy,
    MaxScaleStrategy,
    PercentileScaleStrategy,
    ScaleStrategy,
)


# ---------------------------------------------------------------------------
# QSNR / MSE extraction from observer reports
# ---------------------------------------------------------------------------

def _extract_all_roles_qsnr_mse(
    observers_data: dict,
) -> tuple:
    """Extract per-layer QSNR and MSE for ALL roles in a single pass."""
    qsnr_by_role: Dict[str, Dict[str, float]] = {}
    mse_by_role: Dict[str, Dict[str, float]] = {}

    for _layer, roles in observers_data.items():
        for role, stages in roles.items():
            qsnr_map = qsnr_by_role.setdefault(role, {})
            mse_map = mse_by_role.setdefault(role, {})
            for _stage, slices in stages.items():
                for _slice_key, metrics in slices.items():
                    if "qsnr_db" in metrics:
                        v = metrics["qsnr_db"]
                        if v == v and v != float("-inf"):
                            prev = qsnr_map.get(_layer)
                            if prev is None or v < prev:
                                qsnr_map[_layer] = v
                    if "mse" in metrics:
                        mse_map[_layer] = max(
                            mse_map.get(_layer, 0.0),
                            metrics["mse"],
                        )
    return qsnr_by_role, mse_by_role


def _extract_qsnr_mse(observers_data: dict, *, role: str = "output"):
    """Extract per-layer QSNR and MSE for a single role."""
    qsnr_by_role, mse_by_role = _extract_all_roles_qsnr_mse(observers_data)
    return qsnr_by_role.get(role, {}), mse_by_role.get(role, {})


# ---------------------------------------------------------------------------
# Observer / calibrator registries
# ---------------------------------------------------------------------------

_OBSERVER_MAP: Dict[str, type] = {
    "qsnr": QSNRObserver,
    "mse": MSEObserver,
    "histogram": HistogramObserver,
    "distribution": DistributionObserver,
    "fit": DistributionFitObserver,
}

_SMOOTH_INPUT_ROLES = frozenset({"input", "grad_input", "input_gw"})


def _make_calibrator(name: str) -> ScaleStrategy:
    """Map calibrator string to a ScaleStrategy instance."""
    _mapping = {
        "mse": MSEScaleStrategy,
        "max": MaxScaleStrategy,
        "percentile": PercentileScaleStrategy,
        "kl": KLScaleStrategy,
    }
    return _mapping[name]()


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def _module_device(module: nn.Module) -> torch.device:
    """Get the device of *module* from its parameters or buffers."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        pass
    return torch.device('cpu')


def _run_model(model, data, eval_fn: Optional[Callable] = None) -> None:
    """Run *model* on *data*, forwarding via *eval_fn* when provided."""
    if eval_fn is not None:
        eval_fn(model, data)
    elif isinstance(data, (list, tuple)):
        for batch in data:
            if isinstance(batch, dict):
                model(**batch)
            elif isinstance(batch, tuple):
                model(*batch)
            else:
                model(batch)
    else:
        model(data)


def _needs_calibration(cfg) -> bool:
    """Return False if ALL schemes are MX per_block (scales computed dynamically)."""
    if isinstance(cfg, dict):
        configs = list(cfg.values())
    else:
        configs = [cfg]

    for op_cfg in configs:
        for field_name in op_cfg.__dataclass_fields__:
            scheme = getattr(op_cfg, field_name)
            if scheme is not None and not scheme.granularity.is_mx:
                return True
    return False


# ---------------------------------------------------------------------------
# Channel inference (used by pre_scale)
# ---------------------------------------------------------------------------

def _infer_out_channels(module) -> Optional[int]:
    """Infer output channel count for a module."""
    if hasattr(module, "out_features"):
        return module.out_features
    if hasattr(module, "out_channels"):
        return module.out_channels
    if hasattr(module, "num_features"):
        return module.num_features
    return None


def _infer_in_channels(module) -> Optional[int]:
    """Infer input channel count for a module."""
    if hasattr(module, "in_features"):
        return module.in_features
    if hasattr(module, "in_channels"):
        return module.in_channels
    return None


def _collect_input_amax(
    calib_data: Any,
    model: nn.Module,
    *,
    channel_axis: int = -1,
    granularity: str = "per_tensor",
    eval_fn: Optional[Callable] = None,
) -> Dict[str, torch.Tensor]:
    """Collect per-module input activation max-abs via forward hooks."""
    from src.session._model import _get_quantized_modules

    amax_store: Dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name):
        def _fn(_module, inp, _out):
            x = inp[0].detach()
            if granularity == "per_channel":
                ndim = x.ndim
                ax = channel_axis if channel_axis >= 0 else ndim + channel_axis
                reduce_dims = tuple(d for d in range(ndim) if d != ax)
                batch_amax = torch.amax(torch.abs(x), dim=reduce_dims)
            else:
                batch_amax = torch.amax(torch.abs(x)).reshape(1)
            if name in amax_store:
                amax_store[name] = torch.maximum(amax_store[name], batch_amax)
            else:
                amax_store[name] = batch_amax
        return _fn

    for qname, qmod in _get_quantized_modules(model):
        handles.append(qmod.register_forward_hook(_make_hook(qname)))

    try:
        with torch.no_grad():
            if eval_fn is not None:
                eval_fn(model, calib_data)
            else:
                for batch in calib_data:
                    model(batch)
    finally:
        for h in handles:
            h.remove()

    return amax_store


# ---------------------------------------------------------------------------
# Pre-scale (standalone, converted from Session methods)
# ---------------------------------------------------------------------------

def initialize_pre_scales(
    qmodel: nn.Module,
    calib_data: Any,
    *,
    init: str = "ones",
    pot: bool = False,
    granularity: str = "per_tensor",
    trainable: bool = False,
    channel_axis: int = -1,
    eval_fn: Optional[Callable] = None,
) -> int:
    """Initialize ``_pre_scale`` tensors on all quantized modules in *qmodel*."""
    from src.transform.pre_scale import PreScaleTransform
    from src.session._model import _get_quantized_modules
    from src.calibration.lsq_optimizer import (
        _replace_transform,
        _replace_transform_activation_only,
        _INPUT_ACTIVATION_ROLES,
    )

    if init not in ("ones", "amax", "pot_amax"):
        raise ValueError(f"Unknown init method: {init!r}")
    if granularity not in ("per_tensor", "per_channel"):
        raise ValueError(f"Unknown granularity: {granularity!r}")

    amax_map = None
    if init in ("amax", "pot_amax"):
        amax_map = _collect_input_amax(
            calib_data, qmodel,
            channel_axis=channel_axis,
            granularity=granularity,
            eval_fn=eval_fn,
        )

    count = 0
    for name, module in _get_quantized_modules(qmodel):
        device = _module_device(module)

        if init == "ones":
            if granularity == "per_channel":
                in_channels = _infer_in_channels(module)
                if in_channels is None:
                    continue
                init_scale = torch.ones(in_channels, device=device)
            else:
                init_scale = torch.ones(1, device=device)
        else:
            amax = amax_map.get(name)
            if amax is None:
                continue
            amax = amax.to(device).clamp(min=1e-12)
            init_scale = torch.tensor(1.0, device=device) / amax
            if init == "pot_amax":
                init_scale = 2 ** torch.round(torch.log2(init_scale))

        if trainable:
            module.register_parameter("_pre_scale", nn.Parameter(init_scale))
        else:
            module.register_buffer("_pre_scale", init_scale)

        transform = PreScaleTransform(
            scale=module._pre_scale, pot=pot, channel_axis=channel_axis,
        )
        if granularity == "per_channel":
            module.cfg = _replace_transform_activation_only(
                module.cfg, transform, roles=_INPUT_ACTIVATION_ROLES,
            )
        else:
            module.cfg = _replace_transform(module.cfg, transform)
        count += 1

    return count


def optimize_scales(
    qmodel: nn.Module,
    fp32_model: nn.Module,
    optimizer: "LayerwiseScaleOptimizer",
    calib_data: Any,
    *,
    eval_fn: Optional[Callable] = None,
) -> Dict[str, torch.Tensor]:
    """Run layer-wise LSQ optimization on pre-scale parameters in *qmodel*."""
    return optimizer.optimize(qmodel, fp32_model, calib_data, eval_fn=eval_fn)


def clear_scales(qmodel: nn.Module, calibrator: Optional[ScaleStrategy] = None) -> List[str]:
    """Remove all ``_output_scale`` buffers from *qmodel*."""
    from src.calibration.pipeline import CalibrationSession

    strat = calibrator if calibrator is not None else MaxScaleStrategy()
    cs = CalibrationSession(qmodel, strat, assign=False)
    return cs.clear_scales()
