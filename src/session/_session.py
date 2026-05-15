"""Backward-compat re-exports and standalone quantization pipeline.

The Session class has been removed.  Use ``quantize_model()`` + ``CalibrationSession``
+ ``AnalysisContext`` directly, or call ``run_quantization()`` for the full pipeline.
"""

from __future__ import annotations

import copy
import logging
import math
import time as _time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.analysis.context import AnalysisContext
from src.analysis.e2e import Comparator, compare_models, _default_accuracy
from src.calibration.pipeline import CalibrationSession
from src.calibration.strategies import ScaleStrategy
from src.scheme.op_config import OpQuantConfig, cfg_causes_quantization
from src.session._config import QuantConfig
from src.session._model import quantize_model
from src.session._result import SessionResult

from src.session._helpers import (  # noqa: F401 — re-export for backward compat
    _collect_input_amax,
    _extract_all_roles_qsnr_mse,
    _extract_qsnr_mse,
    _infer_in_channels,
    _infer_out_channels,
    _make_calibrator,
    _module_device,
    _needs_calibration,
    _OBSERVER_MAP,
    _run_model,
    _SMOOTH_INPUT_ROLES,
    clear_scales,
    initialize_pre_scales,
    optimize_scales,
)

from src.transform._adaptive import (  # noqa: F401 — re-export for backward compat
    _apply_layer_selection,
    _estimate_layer_qsnr,
    adaptive_transform_selection,
)

_logger = logging.getLogger(__name__)

_SMOOTH_INPUT_ROLES = frozenset({"input", "grad_input", "input_gw"})


# ---------------------------------------------------------------------------
# Standalone quantization pipeline (replaces Session.run())
# ---------------------------------------------------------------------------

def run_quantization(
    model: nn.Module,
    config: Union[QuantConfig, OpQuantConfig, Dict[str, OpQuantConfig]],
    calib_data,
    *,
    eval_data=None,
    eval_fn: Optional[Callable] = None,
    outputs: Union[str, List[str]] = "default",
    keep_fp32: bool = True,
    overrides: Optional[Dict[str, OpQuantConfig]] = None,
    calibrator: Optional[ScaleStrategy] = None,
    observers: Optional[List] = None,
    op_cfgs: Optional[Dict[str, OpQuantConfig]] = None,
    quantize_nonlinear: bool = True,
    fp32_ref: Optional[nn.Module] = None,
) -> tuple:
    """Run the full quantization pipeline and return (qmodel, fp32_model, result).

    This is the standalone replacement for ``Session(model, cfg).run(...)``.

    Args:
        model: Original FP32 PyTorch model.
        config: ``QuantConfig``, ``OpQuantConfig``, or ``Dict[str, OpQuantConfig]``.
        calib_data: Calibration data.
        eval_data: Optional evaluation data.
        eval_fn: ``(model, data) -> Dict[str, float]``.
        outputs: Output keys (``"default"`` / ``"all"`` / list).
        keep_fp32: Keep a deep copy of the original fp32 model for comparison.
        overrides: Optional per-layer ``OpQuantConfig`` overrides.
        calibrator: ``ScaleStrategy`` for calibration (default: ``MaxScaleStrategy()``).
        observers: Observer list for analysis (default: ``[QSNRObserver()]``).
        op_cfgs: Optional per-op-type overrides for inline ops.
        quantize_nonlinear: If False, norm/activation/pool remain fp32.
        fp32_ref: Optional fp32 reference (used when model is pre-fused).

    Returns:
        ``(qmodel, fp32_model, SessionResult)``.
    """
    from src.report._spec import resolve_outputs as _resolve_outputs, PRESETS

    from src.analysis.observers import QSNRObserver
    from src.calibration.strategies import MaxScaleStrategy
    from src.analysis.observers import (
        DistributionFitObserver,
        DistributionObserver,
        HistogramObserver,
        MSEObserver,
        QSNRObserver,
    )
    from src.transform.smooth_quant import (
        SmoothQuantTransform,
        fuse_smoothquant_weights,
    )
    from src.transform._adaptive import adaptive_transform_selection
    from src.session._helpers import (
        _make_calibrator,
        _needs_calibration,
        _run_model,
        _OBSERVER_MAP,
        _extract_all_roles_qsnr_mse,
    )

    observer_keys, needs_eval, needs_cost = _resolve_outputs(outputs)

    # Resolve default calibrator / observers
    if calibrator is None:
        calibrator = MaxScaleStrategy()
    if observers is None:
        observers = [QSNRObserver()]

    # ---- Handle OpQuantConfig / dict directly ----
    if isinstance(config, (OpQuantConfig, dict)):
        op_cfg = config
        qmodel = quantize_model(
            copy.deepcopy(model), cfg=op_cfg, op_cfgs=op_cfgs,
            quantize_nonlinear=quantize_nonlinear,
        )
        fp32_model = copy.deepcopy(fp32_ref if fp32_ref is not None else model) if keep_fp32 else None
        cfg_for_calib = op_cfg
    else:
        # ---- QuantConfig path ----
        cfg = config  # type: QuantConfig
        sq_transforms = None

        if cfg.transform == "smoothquant":
            if calib_data is None:
                raise ValueError("calib_data is required for smoothquant transform")
            sq_transforms = SmoothQuantTransform.from_model_calibration(
                model, calib_data, alpha=cfg.sq_alpha, eval_fn=eval_fn,
            )
            fused_model = fuse_smoothquant_weights(model, sq_transforms)
            fp32_ref_base = model
        else:
            fused_model = model
            fp32_ref_base = None

        # Build OpQuantConfig
        op_cfg = cfg.to_op_config()
        if overrides:
            op_cfg = {"*": op_cfg, **overrides}

        calibrator = _make_calibrator(cfg.calibrator)

        qmodel = quantize_model(
            copy.deepcopy(fused_model),
            cfg=op_cfg,
            op_cfgs=None,
            quantize_nonlinear=cfg.quantize_nonlinear,
        )
        fp32_model = copy.deepcopy(
            fp32_ref if fp32_ref is not None else (fp32_ref_base if fp32_ref_base is not None else fused_model)
        ) if keep_fp32 else None

        # Patch per-layer SmoothQuant transforms
        if sq_transforms:
            for name, sq_t in sq_transforms.items():
                module = dict(qmodel.named_modules()).get(name)
                if module is None or not hasattr(module, "cfg"):
                    continue
                old: OpQuantConfig = module.cfg
                new_kwargs: Dict[str, Any] = {}
                for f_name in old.__dataclass_fields__:
                    scheme = getattr(old, f_name)
                    if scheme is not None and f_name in _SMOOTH_INPUT_ROLES:
                        new_kwargs[f_name] = type(scheme)(
                            format=scheme.format,
                            granularity=scheme.granularity,
                            transform=sq_t,
                            round_mode=scheme.round_mode,
                            scale_storage=scheme.scale_storage,
                        )
                    else:
                        new_kwargs[f_name] = scheme
                module.cfg = OpQuantConfig(**new_kwargs)

        # GPTQ
        if cfg.gptq:
            if calib_data is None:
                raise ValueError("calib_data is required for GPTQ")
            from src.calibration.gptq_optimizer import GPTQOptimizer
            gptq_opt = GPTQOptimizer(
                block_size=cfg.gptq_block_size,
                damp_percent=cfg.gptq_damp,
                act_order=cfg.gptq_act_order,
            )
            gptq_opt.optimize(qmodel, calib_data, eval_fn=eval_fn)

        # Prescale + LSQ
        if cfg.transform == "prescale":
            if calib_data is None:
                raise ValueError("calib_data is required for prescale transform")
            initialize_pre_scales(
                qmodel, calib_data,
                init=cfg.prescale_init,
                pot=cfg.prescale_pot,
                granularity=cfg.prescale_granularity,
                eval_fn=eval_fn,
            )
            if cfg.lsq_steps > 0:
                from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
                opt = LayerwiseScaleOptimizer(
                    num_steps=cfg.lsq_steps, lr=cfg.lsq_lr,
                )
                optimize_scales(qmodel, fp32_model, opt, calib_data, eval_fn=eval_fn)

        cfg_for_calib = op_cfg

    # ---- Calibrate ----
    if isinstance(config, QuantConfig) and config.transform == "adaptive":
        adaptive_transform_selection(qmodel, config, calib_data, eval_fn)

    if _needs_calibration(cfg_for_calib):
        track_input_flag = config.static_input_scale if isinstance(config, QuantConfig) else False
        _sparse = _any_outlier_ratio(qmodel)
        _sq_mode = config.sq_mode if isinstance(config, QuantConfig) else None
        with CalibrationSession(qmodel, calibrator, track_input=track_input_flag,
                                sparse=_sparse, sq_mode=_sq_mode):
            _run_model(qmodel, calib_data, eval_fn)

    # ---- Analyze ----
    _qsnr_per_layer: Dict[str, float] = {}
    _mse_per_layer: Dict[str, float] = {}
    _qsnr_by_role: Dict[str, Dict[str, float]] = {}
    _mse_by_role: Dict[str, Dict[str, float]] = {}
    _accum_qsnr_per_layer: Dict[str, float] = {}
    _accum_mse_per_layer: Dict[str, float] = {}
    _observers_data: Dict[str, Any] = {}

    can_hook = (
        "qsnr" in observer_keys
        and keep_fp32
        and fp32_model is not None
    )

    if observer_keys or can_hook:
        if can_hook:
            _observers_data, _accum_qsnr_per_layer, _accum_mse_per_layer, _qsnr_by_role, \
                _mse_by_role, _qsnr_per_layer, _mse_per_layer = _run_hook_analysis(
                    qmodel, fp32_model, calib_data, observer_keys, eval_fn,
                )
        elif observer_keys:
            _obs_list = [
                _OBSERVER_MAP[k]() for k in sorted(observer_keys)
                if k in _OBSERVER_MAP
            ]
            if _obs_list:
                with AnalysisContext(qmodel, _obs_list) as ctx:
                    _run_model(qmodel, calib_data, eval_fn)
                _observers_data = ctx.report()._raw
                qsnr_by_role, mse_by_role = _extract_all_roles_qsnr_mse(_observers_data)
                _qsnr_by_role = qsnr_by_role
                _mse_by_role = mse_by_role
                _qsnr_per_layer = qsnr_by_role.get("output", {})
                _mse_per_layer = mse_by_role.get("output", {})

    # ---- Evaluate ----
    fp32_metrics = None
    quant_metrics = None
    delta = None

    if needs_eval and eval_fn is not None:
        if eval_data is None:
            _logger.warning("eval_data not provided — falling back to calib_data")
        _eval_data = eval_data if eval_data is not None else calib_data
        if keep_fp32 and fp32_model is not None:
            fp32_metrics = eval_fn(fp32_model, _eval_data)
        quant_metrics = eval_fn(qmodel, _eval_data)
        if fp32_metrics is not None:
            delta = {
                k: quant_metrics[k] - fp32_metrics[k]
                for k in fp32_metrics
            }

    # ---- Cost ----
    _cost = None
    _cost_fp32 = None
    if needs_cost:
        from src.cost.model_cost import analyze_model_cost
        _cost = analyze_model_cost(qmodel)
        if keep_fp32 and fp32_model is not None:
            _cost_fp32 = analyze_model_cost(fp32_model)

    # ---- Build result ----
    result = SessionResult(
        name=config.name if isinstance(config, QuantConfig) else "",
        config=config if isinstance(config, QuantConfig) else QuantConfig(),
        fp32_metrics=fp32_metrics,
        quant_metrics=quant_metrics,
        delta=delta,
        qsnr_per_layer=_qsnr_per_layer,
        mse_per_layer=_mse_per_layer,
        qsnr_by_role=_qsnr_by_role,
        mse_by_role=_mse_by_role,
        accum_qsnr_per_layer=_accum_qsnr_per_layer,
        accum_mse_per_layer=_accum_mse_per_layer,
        observers_data=_observers_data,
        cost=_cost,
        cost_fp32=_cost_fp32,
    )

    return qmodel, fp32_model, result


def _any_outlier_ratio(qmodel: nn.Module) -> bool:
    """Check if any module's OpQuantConfig uses outlier_ratio > 0."""
    for m in qmodel.modules():
        cfg = getattr(m, "cfg", None)
        if cfg is None or not isinstance(cfg, OpQuantConfig):
            continue
        for fname in cfg.__dataclass_fields__:
            s = getattr(cfg, fname)
            if s is None:
                continue
            gran = getattr(s, "granularity", None)
            if gran is not None and gran.outlier_ratio > 0:
                return True
    return False


def _run_hook_analysis(
    qmodel: nn.Module,
    fp32_model: nn.Module,
    calib_data,
    observer_keys: set,
    eval_fn: Optional[Callable] = None,
):
    """Run hook-based analysis: compare fp32 vs quant outputs with optional observers."""
    from src.session._helpers import _OBSERVER_MAP, _extract_all_roles_qsnr_mse, _run_model

    quant_names = [
        name for name, mod in qmodel.named_modules()
        if hasattr(mod, "cfg") and cfg_causes_quantization(mod.cfg)
    ]
    fp32_name_to_mod = dict(fp32_model.named_modules())
    qname_to_mod = dict(qmodel.named_modules())

    multi_batch = eval_fn is None and isinstance(calib_data, (list, tuple))
    batches = list(calib_data) if multi_batch else [calib_data]

    accum_signal: dict = defaultdict(float)
    accum_error: dict = defaultdict(float)
    accum_count: dict = defaultdict(int)

    observers = [
        _OBSERVER_MAP[k]() for k in sorted(observer_keys)
        if k in _OBSERVER_MAP
    ] if observer_keys else []

    obs_ctx = AnalysisContext(qmodel, observers) if observers else None
    if obs_ctx is not None:
        obs_ctx.__enter__()

    observers_data = {}
    try:
        for batch in batches:
            # fp32 reference forward
            fp32_refs: Dict[str, torch.Tensor] = {}
            fp32_hooks = []
            for name in quant_names:
                mod = fp32_name_to_mod.get(name)
                if mod is None:
                    continue
                def _fp32_hook(_m, _inp, out, n=name):
                    fp32_refs[n] = out.detach()
                fp32_hooks.append(mod.register_forward_hook(_fp32_hook))

            with torch.no_grad():
                if eval_fn is not None:
                    eval_fn(fp32_model, batch)
                elif isinstance(batch, dict):
                    fp32_model(**batch)
                elif isinstance(batch, tuple):
                    fp32_model(*batch)
                else:
                    fp32_model(batch)
            for h in fp32_hooks:
                h.remove()

            # quant forward
            quant_outs: Dict[str, torch.Tensor] = {}
            quant_hooks = []
            for name in quant_names:
                mod = qname_to_mod.get(name)
                if mod is None:
                    continue
                def _quant_hook(_m, _inp, out, n=name):
                    quant_outs[n] = out.detach()
                quant_hooks.append(mod.register_forward_hook(_quant_hook))

            with torch.no_grad():
                if eval_fn is not None:
                    eval_fn(qmodel, batch)
                elif isinstance(batch, dict):
                    qmodel(**batch)
                elif isinstance(batch, tuple):
                    qmodel(*batch)
                else:
                    qmodel(batch)
            for h in quant_hooks:
                h.remove()

            for name, fp in fp32_refs.items():
                q = quant_outs.get(name)
                if q is None or fp.shape != q.shape:
                    continue
                accum_signal[name] += fp.pow(2).sum().item()
                accum_error[name] += (fp - q).pow(2).sum().item()
                accum_count[name] += fp.numel()
    finally:
        if obs_ctx is not None:
            obs_ctx.__exit__(None, None, None)
            observers_data = obs_ctx.report()._raw

    accum_qsnr = {}
    accum_mse = {}
    for name in sorted(accum_signal):
        if accum_count[name] == 0:
            continue
        mean_signal = accum_signal[name] / accum_count[name]
        mean_error = accum_error[name] / accum_count[name]
        if mean_error > 1e-30:
            accum_qsnr[name] = 10.0 * math.log10(
                max(mean_signal, 1e-12) / mean_error
            )
        accum_mse[name] = mean_error

    qsnr_by_role = {}
    mse_by_role = {}
    if observers_data:
        qsnr_by_role, mse_by_role = _extract_all_roles_qsnr_mse(observers_data)

    return (
        observers_data,
        accum_qsnr,
        accum_mse,
        qsnr_by_role,
        mse_by_role,
        qsnr_by_role.get("output", {}),
        mse_by_role.get("output", {}),
    )
