"""Session execution unit and SessionResult dataclass.

Session is the atomic execution unit: one QuantConfig → one SessionResult.
It wraps _QuantSession and supports two usage modes:

1. **Full pipeline** (backward compat)::

    result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)

2. **Step-by-step** (chainable, user-inspectable at each stage)::

    session = Session(model, cfg)
    session.quantize(calib_data=calib_data)          # MX: calib_data optional
    # session.qmodel available for manual inference
    session.calibrate(calib_data, eval_fn=eval_fn)   # MX per_block: no-op
    session.analyze(calib_data, outputs="default")
    session.evaluate(eval_data, eval_fn)
    session.cost()
    result = session.result
    print(result.summary())
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.analysis.observers import (
    DistributionObserver,
    DistributionFitObserver,
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
from src.ops.conv import QuantizedConv2d
from src.ops.linear import QuantizedLinear
from src.quantize import quantize as _quantize_fn
from src.scheme.op_config import OpQuantConfig, cfg_causes_quantization
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform
from src.session._config import QuantConfig
from src.session._quant import _QuantSession
from src.session._result import SessionResult
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import (
    SmoothQuantTransform,
    compute_smoothquant_scale,
    fuse_smoothquant_weights,
)

import logging
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_all_roles_qsnr_mse(
    observers_data: dict,
) -> tuple:
    """Extract per-layer QSNR and MSE for ALL roles in a single pass.

    Iterates the nested observer report dict once and collects per-layer
    worst-case QSNR (minimum across stages/slices) and MSE (maximum across
    stages/slices) for every role present in the data.

    Args:
        observers_data: Raw observer report dict
            (``{layer: {role: {stage: {slice: metrics}}}}``).

    Returns:
        ``(qsnr_by_role, mse_by_role)`` — each is ``Dict[str, Dict[str, float]]``
        mapping role name → layer name → value.
    """
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
                        if v == v and v != float("-inf"):  # exclude NaN and -inf
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
    """Extract per-layer QSNR and MSE for a single role.

    Thin backward-compatible wrapper around :func:`_extract_all_roles_qsnr_mse`.
    Prefer that function for new code to avoid repeated iteration.

    Args:
        observers_data: Raw observer report dict.
        role: Tensor role to extract (``"input"`` / ``"weight"`` /
            ``"output"`` / ``"bias"``). Default ``"output"``.

    Returns:
        ``(qsnr_per_layer, mse_per_layer)`` — each is ``Dict[str, float]``.
    """
    qsnr_by_role, mse_by_role = _extract_all_roles_qsnr_mse(observers_data)
    return qsnr_by_role.get(role, {}), mse_by_role.get(role, {})


# ---------------------------------------------------------------------------
# Observer key -> observer class mapping
# ---------------------------------------------------------------------------

_OBSERVER_MAP: Dict[str, type] = {
    "qsnr": QSNRObserver,
    "mse": MSEObserver,
    "histogram": HistogramObserver,
    "distribution": DistributionObserver,
    "fit": DistributionFitObserver,
}

# Roles that receive SmoothQuantTransform when patching per-layer configs.
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


def _run_model(model, data, eval_fn: Optional[Callable] = None) -> None:
    """Run *model* on *data*, forwarding via *eval_fn* when provided."""
    if eval_fn is not None:
        eval_fn(model, data)
    elif isinstance(data, (list, tuple)):
        for batch in data:
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
# Adaptive transform selection
# ---------------------------------------------------------------------------

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
    """Estimate matmul-output QSNR (dB) for a single transform candidate.

    Constructs candidate ``QuantScheme`` instances, runs real
    :func:`quantize` on the activation and weight, and computes the
    resulting QSNR on the matmul (or conv) output.

    Returns:
        QSNR in dB, or ``-inf`` if estimation failed.
    """
    import math
    import torch.nn.functional as F

    # --- Build candidate schemes ---
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
        w_tx = IdentityTransform()  # fused into weight
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
    """Patch *module.cfg* to use the chosen *tx* and fuse SQ weights.

    *applied* is a ``Dict[str, int]`` mutated in-place to track counts.
    """
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
        # Weight fusion
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


def _adaptive_transform_selection(
    qsession: _QuantSession,
    base_config: QuantConfig,
    calib_data,
    eval_fn=None,
):
    """Hook → forward → estimate QSNR per candidate → pick best transform.

    Returns counts of modules assigned to each transform
    (keys: ``"none"``, ``"hadamard"``, ``"smoothquant"``).
    """
    qmodel = qsession.qmodel

    # ---- 1. Hook to capture activations ---------------------------------
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

    # ---- 2. One forward pass --------------------------------------------
    try:
        with torch.no_grad():
            _run_model(qsession, calib_data, eval_fn)
    finally:
        for h in hooks:
            h.remove()

    if not activations:
        return {"none": 0, "hadamard": 0, "smoothquant": 0}

    # ---- 3. Per-layer candidate evaluation ------------------------------
    module_map = dict(qmodel.named_modules())
    per_layer_choice: Dict[str, tuple] = {}  # name → (best_tx, sq_scale)

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
                    # Recompute scale for winner (save it for apply step)
                    act_axis = -1 if isinstance(module, QuantizedLinear) else 1
                    best_sq_scale = compute_smoothquant_scale(
                        x_act, W, alpha=base_config.sq_alpha,
                        act_channel_axis=act_axis, w_channel_axis=1,
                    )

        per_layer_choice[name] = (best_tx, best_sq_scale)

    # ---- 4. Apply selections --------------------------------------------
    applied = {"none": 0, "hadamard": 0, "smoothquant": 0}

    for name, (tx, sq_scale) in per_layer_choice.items():
        module = module_map.get(name)
        if module is None:
            continue
        _apply_layer_selection(module, tx, sq_scale, applied)

    return applied


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """Atomic execution unit with a layered, chainable API.

    Two usage modes:

    1. **Full pipeline** (backward compat)::

        result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)

    2. **Step-by-step**::

        session = Session(model, cfg)
        session.quantize(calib_data=calib_data)
        # session.qmodel is now accessible for manual inference
        session.calibrate(calib_data)               # MX per_block: no-op
        session.analyze(calib_data, outputs="default")
        session.evaluate(eval_data, eval_fn)
        session.cost()
        result = session.result

    All step methods return ``self`` so calls can be chained::

        session.quantize().calibrate(data).analyze(data).evaluate(data, fn)
    """

    def __init__(
        self,
        model: nn.Module,
        config: QuantConfig,
        *,
        keep_fp32: bool = True,
        overrides: Optional[Dict[str, OpQuantConfig]] = None,
    ):
        self._model = model
        self._config = config
        self._keep_fp32 = keep_fp32
        self._overrides = overrides

        # Lazily initialized by .quantize()
        self._quant_session: Optional[_QuantSession] = None
        self._sq_transforms: Optional[Dict[str, Any]] = None
        self._adaptive_done: bool = False
        self._adaptive_selection: Optional[Dict[str, int]] = None

        # Collected results (populated by each step)
        self._qsnr_per_layer: Dict[str, float] = {}
        self._mse_per_layer: Dict[str, float] = {}
        self._qsnr_by_role: Dict[str, Dict[str, float]] = {}
        self._mse_by_role: Dict[str, Dict[str, float]] = {}
        self._accum_qsnr_per_layer: Dict[str, float] = {}
        self._accum_mse_per_layer: Dict[str, float] = {}
        self._observers_data: Dict[str, Any] = {}
        self._fp32_metrics: Optional[Dict[str, float]] = None
        self._quant_metrics: Optional[Dict[str, float]] = None
        self._delta: Optional[Dict[str, float]] = None
        self._cost: Any = None
        self._cost_fp32: Any = None
        self._sq_distrib_comparison: Optional[Any] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qmodel(self) -> nn.Module:
        """The quantized model (available after ``.quantize()``)."""
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        return self._quant_session.qmodel

    @property
    def fp32_model(self) -> Optional[nn.Module]:
        """The fp32 reference model (available after ``.quantize()`` if
        ``keep_fp32=True``).
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        return self._quant_session.fp32_model

    @property
    def result(self) -> SessionResult:
        """Build and return the :class:`SessionResult` from collected data.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        return SessionResult(
            name=self._config.name,
            config=self._config,
            fp32_metrics=self._fp32_metrics,
            quant_metrics=self._quant_metrics,
            delta=self._delta,
            qsnr_per_layer=self._qsnr_per_layer,
            mse_per_layer=self._mse_per_layer,
            qsnr_by_role=self._qsnr_by_role,
            mse_by_role=self._mse_by_role,
            accum_qsnr_per_layer=self._accum_qsnr_per_layer,
            accum_mse_per_layer=self._accum_mse_per_layer,
            observers_data=self._observers_data,
            cost=self._cost,
            cost_fp32=self._cost_fp32,
            sq_transforms=self._sq_transforms,
            sq_distrib_comparison=self._sq_distrib_comparison,
        )

    # ------------------------------------------------------------------
    # Step 1: Quantize
    # ------------------------------------------------------------------

    def quantize(self, *, calib_data=None, eval_fn=None) -> "Session":
        """Build the quantized model. Must be called first.

        After this method returns, ``session.qmodel`` is available for
        manual inference and ``session(x)`` delegates to the quantized model.

        Args:
            calib_data: Calibration data. Required when ``transform`` is
                ``"smoothquant"`` or ``"prescale"``. Not needed for other
                transforms or MX per_block formats.
            eval_fn: Optional ``(model, data) -> Any`` for custom model
                interaction during SmoothQuant calibration.

        Raises:
            ValueError: If ``calib_data`` is ``None`` but the transform
                requires it.
        """
        # Reset adaptive state so re-quantize re-runs selection
        self._adaptive_done = False
        self._adaptive_selection: Optional[Dict[str, int]] = None

        # ---- SmoothQuant: compute per-channel scales + fuse weights ----
        fp32_ref: Optional[nn.Module] = None
        if self._config.transform == "smoothquant":
            if calib_data is None:
                raise ValueError(
                    "calib_data is required for smoothquant transform"
                )
            self._sq_transforms = SmoothQuantTransform.from_model_calibration(
                self._model,
                calib_data,
                alpha=self._config.sq_alpha,
                eval_fn=eval_fn,
            )
            model = fuse_smoothquant_weights(self._model, self._sq_transforms)
            fp32_ref = self._model  # original, unmodified model as FP32 baseline
        else:
            model = self._model
            self._sq_transforms = None

        # ---- Build OpQuantConfig (per-layer dict when overrides exist) ----
        op_cfg = self._config.to_op_config()
        if self._overrides:
            # "*" pattern matches any module name not explicitly overridden
            op_cfg = {"*": op_cfg, **self._overrides}

        # ---- Create _QuantSession ----
        calibrator = _make_calibrator(self._config.calibrator)
        self._quant_session = _QuantSession(
            model,
            op_cfg,
            calibrator=calibrator,
            observers=[],
            keep_fp32=self._keep_fp32,
            quantize_nonlinear=self._config.quantize_nonlinear,
            fp32_ref=fp32_ref,
        )

        # ---- Patch per-layer SmoothQuant transforms into module configs ----
        if self._sq_transforms:
            for name, sq_t in self._sq_transforms.items():
                module = dict(self._quant_session.qmodel.named_modules()).get(name)
                if module is None or not hasattr(module, "cfg"):
                    continue
                old: OpQuantConfig = module.cfg
                new_kwargs: Dict[str, Any] = {}
                for f_name in old.__dataclass_fields__:
                    scheme = getattr(old, f_name)
                    if scheme is not None and f_name in _SMOOTH_INPUT_ROLES:
                        new_kwargs[f_name] = QuantScheme(
                            format=scheme.format,
                            granularity=scheme.granularity,
                            transform=sq_t,
                            round_mode=scheme.round_mode,
                            scale_storage=scheme.scale_storage,
                        )
                    else:
                        new_kwargs[f_name] = scheme
                module.cfg = OpQuantConfig(**new_kwargs)

        # ---- GPTQ: Hessian-based weight-only quantization ----
        if self._config.gptq:
            if calib_data is None:
                raise ValueError("calib_data is required for GPTQ")
            from src.calibration.gptq_optimizer import GPTQOptimizer

            gptq_opt = GPTQOptimizer(
                block_size=self._config.gptq_block_size,
                damp_percent=self._config.gptq_damp,
                act_order=self._config.gptq_act_order,
            )
            gptq_opt.optimize(
                self._quant_session.qmodel,
                calib_data,
                eval_fn=eval_fn,
            )

        # ---- Prescale: init pre_scales + optional LSQ ----
        if self._config.transform == "prescale":
            if calib_data is None:
                raise ValueError(
                    "calib_data is required for prescale transform"
                )
            self._quant_session.initialize_pre_scales(
                calib_data,
                init=self._config.prescale_init,
                pot=self._config.prescale_pot,
                granularity=self._config.prescale_granularity,
                eval_fn=eval_fn,
            )
            if self._config.lsq_steps > 0:
                from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

                opt = LayerwiseScaleOptimizer(
                    num_steps=self._config.lsq_steps,
                    lr=self._config.lsq_lr,
                )
                self._quant_session.optimize_scales(opt, calib_data, eval_fn=eval_fn)

        return self

    # ------------------------------------------------------------------
    # Step 2: Calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        calib_data,
        *,
        eval_fn: Optional[Callable] = None,
    ) -> "Session":
        """Run calibration to compute quantization scales.

        When ``config.transform == "adaptive"``, one forward pass is run
        first to select the best per-layer transform (none / hadamard /
        smoothquant) via QSNR estimation.  The choice is cached in
        ``session._adaptive_selection``.

        For MX per_block formats scale calibration is a **no-op** (scales
        are computed dynamically), but adaptive selection still runs.

        Args:
            calib_data: Calibration data (list of tensors or single tensor).
            eval_fn: Optional ``(model, data) -> Any`` for custom model
                interaction during calibration.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")

        # Adaptive transform selection: hook → forward → estimate QSNR
        # per candidate → patch cfgs + fuse SQ weights.
        if self._config.transform == "adaptive" and not self._adaptive_done:
            self._adaptive_done = True
            self._adaptive_selection = _adaptive_transform_selection(
                self._quant_session, self._config, calib_data, eval_fn,
            )

        if not _needs_calibration(self._quant_session.cfg):
            return self  # MX per_block: scales computed dynamically

        with self._quant_session.calibrate(
            track_input=self._config.static_input_scale,
        ):
            _run_model(self._quant_session, calib_data, eval_fn)

        return self

    # ------------------------------------------------------------------
    # Step 3: Analyze
    # ------------------------------------------------------------------

    def analyze(
        self,
        calib_data,
        *,
        outputs: Union[str, List[str]] = "default",
        eval_fn: Optional[Callable] = None,
    ) -> "Session":
        """Run error analysis on the quantized model.

        When the ``"qsnr"`` observer is active (default), both local (observer)
        and accumulated (hook) QSNR are collected in a single pass. Results go
        to separate fields:

        - ``result.qsnr_per_layer`` — local QSNR from observer (pre-quant vs
          post-quant per layer)
        - ``result.accum_qsnr_per_layer`` — accumulated QSNR from hooks
          (quant output vs fp32 reference, reflecting upstream propagation)

        Args:
            calib_data: Data to run through the model for analysis.
            outputs: Output keys — ``"default"``, ``"all"``, or a list of
                specific keys (``"qsnr"``, ``"mse"``, ``"histogram"``, ...).
            eval_fn: Optional ``(model, data) -> Any`` for custom model
                interaction.  When provided it is **always** used for every
                forward pass — ``model(batch)`` is never called directly.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")

        import math
        from collections import defaultdict

        from src.report._spec import resolve_outputs as _resolve_outputs, PRESETS

        observer_keys, _needs_eval, _needs_cost = _resolve_outputs(outputs)
        observer_keys_set = set(observer_keys)
        # Resolve the original output spec keys (not observer keys) to check
        # for pseudo-outputs like "smoothquant_distrib" that have no observers.
        if outputs == "default":
            _output_keys = set(PRESETS["default"])
        elif outputs == "all":
            _output_keys = set(PRESETS["all"])
        else:
            _output_keys = set(outputs)
        _wants_sq_distrib = "smoothquant_distrib" in _output_keys

        can_hook = (
            "qsnr" in observer_keys_set
            and self._keep_fp32
            and self._quant_session.fp32_model is not None
        )

        if not observer_keys_set and not can_hook and not _wants_sq_distrib:
            return self

        observers = [
            _OBSERVER_MAP[k]() for k in sorted(observer_keys_set)
            if k in _OBSERVER_MAP
        ] if observer_keys_set else []

        _obs_names = [type(o).__name__ for o in observers]
        if _obs_names:
            print(f"    observers: {_obs_names}, hook_path={can_hook}")

        # ── Hook + observer path (combined) ──────────────────────────
        if can_hook:
            qmodel = self._quant_session.qmodel
            fp32_model = self._quant_session.fp32_model

            # Only include modules whose config actually triggers
            # quantization.  Modules with an empty / all-None config
            # produce bit-exact fp32 output and would inflate QSNR.
            quant_names = [
                name for name, mod in qmodel.named_modules()
                if hasattr(mod, "cfg") and cfg_causes_quantization(mod.cfg)
            ]

            fp32_name_to_mod = dict(fp32_model.named_modules())
            qname_to_mod = dict(qmodel.named_modules())

            multi_batch = (
                eval_fn is None
                and isinstance(calib_data, (list, tuple))
            )
            batches = list(calib_data) if multi_batch else [calib_data]
            n_batches = len(batches)

            if multi_batch and n_batches > 1:
                print(f"    analyze: {n_batches} batches (hook+observer path)")

            accum_signal: dict = defaultdict(float)
            accum_error: dict = defaultdict(float)
            accum_count: dict = defaultdict(int)

            # Observers ride on the quant forward pass(es). Wrap the
            # whole loop so observer aggregates see all batches.
            obs_ctx = (
                self._quant_session.analyze(observers=observers)
                if observers else None
            )
            if obs_ctx is not None:
                obs_ctx.__enter__()

            try:
                for batch_idx, batch in enumerate(batches):
                    if multi_batch and n_batches > 1:
                        print(f"      batch {batch_idx + 1}/{n_batches}")

                    # -- fp32 reference forward -----------------------
                    fp32_refs: Dict[str, torch.Tensor] = {}
                    fp32_hooks = []
                    for name in quant_names:
                        mod = fp32_name_to_mod.get(name)
                        if mod is None:
                            continue
                        def _fp32_hook(_m, _inp, out, n=name):
                            fp32_refs[n] = out.detach()
                        fp32_hooks.append(
                            mod.register_forward_hook(_fp32_hook))

                    with torch.no_grad():
                        if eval_fn is not None:
                            eval_fn(fp32_model, batch)
                        else:
                            fp32_model(batch)
                    for h in fp32_hooks:
                        h.remove()

                    # -- quant forward --------------------------------
                    quant_outs: Dict[str, torch.Tensor] = {}
                    quant_hooks = []
                    for name in quant_names:
                        mod = qname_to_mod.get(name)
                        if mod is None:
                            continue
                        def _quant_hook(_m, _inp, out, n=name):
                            quant_outs[n] = out.detach()
                        quant_hooks.append(
                            mod.register_forward_hook(_quant_hook))

                    with torch.no_grad():
                        if eval_fn is not None:
                            eval_fn(self._quant_session, batch)
                        else:
                            self._quant_session(batch)
                    for h in quant_hooks:
                        h.remove()

                    # -- accumulate ----------------------------------
                    for name, fp in fp32_refs.items():
                        q = quant_outs.get(name)
                        if q is None or fp.shape != q.shape:
                            continue
                        accum_signal[name] += fp.pow(2).sum().item()
                        accum_error[name] += (
                            (fp - q).pow(2).sum().item()
                        )
                        accum_count[name] += fp.numel()
            finally:
                if obs_ctx is not None:
                    obs_ctx.__exit__(None, None, None)

            # Collect observer report (if any)
            if obs_ctx is not None:
                self._observers_data = obs_ctx.report()._raw

            # Accumulated QSNR / MSE from hook data
            self._accum_qsnr_per_layer = {}
            self._accum_mse_per_layer = {}
            for name in sorted(accum_signal):
                if accum_count[name] == 0:
                    continue
                mean_signal = accum_signal[name] / accum_count[name]
                mean_error = accum_error[name] / accum_count[name]
                if mean_error > 1e-30:
                    self._accum_qsnr_per_layer[name] = 10.0 * math.log10(
                        max(mean_signal, 1e-12) / mean_error
                    )
                self._accum_mse_per_layer[name] = mean_error

            # Local QSNR / MSE from observer data (all roles in one pass)
            if self._observers_data:
                qsnr_by_role, mse_by_role = _extract_all_roles_qsnr_mse(
                    self._observers_data
                )
                self._qsnr_by_role = qsnr_by_role
                self._mse_by_role = mse_by_role
                # Backward-compat: populate single-role fields from "output"
                self._qsnr_per_layer = qsnr_by_role.get("output", {})
                self._mse_per_layer = mse_by_role.get("output", {})
            else:
                self._qsnr_by_role = {}
                self._mse_by_role = {}
                self._qsnr_per_layer = {}
                self._mse_per_layer = {}
        else:
            # ── Observer-only path ───────────────────────────────────
            if observers:
                print(f"    analyze: observer-only path, {len(observers)} observer(s)")
                with self._quant_session.analyze(observers=observers) as ctx:
                    _run_model(self._quant_session, calib_data, eval_fn)
                self._observers_data = ctx.report()._raw
                qsnr_by_role, mse_by_role = _extract_all_roles_qsnr_mse(
                    self._observers_data
                )
                self._qsnr_by_role = qsnr_by_role
                self._mse_by_role = mse_by_role
                self._qsnr_per_layer = qsnr_by_role.get("output", {})
                self._mse_per_layer = mse_by_role.get("output", {})

        # SmoothQuant distribution comparison (auto-triggered, no-op
        # when transform is not "smoothquant" — the user asked for the
        # output but this config doesn't support it).
        if _wants_sq_distrib and self._config.transform == "smoothquant":
            self.compare_smoothquant_distributions(
                calib_data,
                eval_fn=eval_fn,
            )

        return self

    # ------------------------------------------------------------------
    # Step 3b: SmoothQuant distribution comparison
    # ------------------------------------------------------------------

    def compare_smoothquant_distributions(
        self,
        calib_data,
        *,
        eval_fn: Optional[Callable] = None,
        layers: Optional[List[str]] = None,
    ) -> "Session":
        """Compare activation/weight distributions before and after SmoothQuant.

        Runs ONE forward pass on the fused model (with hooks) to capture raw
        activations, then computes smoothed distributions as ``X / scale``
        using the per-layer SmoothQuant scales cached during ``.quantize()``.

        Only meaningful when ``transform="smoothquant"``.  Must be called
        after ``.quantize()``.  Stores the result in
        ``result.sq_distrib_comparison``.

        Args:
            calib_data: Calibration data (single tensor or batch list).
            eval_fn: Optional ``(model, data) -> Any``.  When provided,
                invoked once per forward pass — does NOT iterate over
                batch dimensions.
            layers: If given, only compare these layers.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called or
                the config transform is not ``"smoothquant"``.

        Returns:
            Self for chaining.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        if self._config.transform != "smoothquant":
            raise RuntimeError(
                "compare_smoothquant_distributions() requires "
                "transform='smoothquant'"
            )
        if self._sq_transforms is None or not self._sq_transforms:
            return self

        from src.analysis._smoothquant_distrib import (
            compare_smoothquant_distributions as _compare,
        )

        self._sq_distrib_comparison = _compare(
            fp32_model=self._model,
            fused_model=self._quant_session.qmodel,
            sq_transforms=self._sq_transforms,
            calib_data=calib_data,
            eval_fn=eval_fn,
            layers=layers,
        )
        return self

    # ------------------------------------------------------------------
    # Step 4: Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        eval_data,
        eval_fn: Optional[Callable] = None,
    ) -> "Session":
        """Evaluate fp32 vs quantized model accuracy.

        Args:
            eval_data: Evaluation data passed to ``eval_fn``.
            eval_fn: ``(model, data) -> Dict[str, float]``. Called on both
                fp32 and quantized models. The returned dict keys are used
                to compute per-metric deltas.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")

        if eval_fn is None:
            return self

        self._fp32_metrics = None
        self._quant_metrics = None
        self._delta = None

        if self._keep_fp32 and self._quant_session.fp32_model is not None:
            self._fp32_metrics = eval_fn(self._quant_session.fp32_model, eval_data)

        self._quant_metrics = eval_fn(self._quant_session, eval_data)

        if self._fp32_metrics is not None:
            self._delta = {
                k: self._quant_metrics[k] - self._fp32_metrics[k]
                for k in self._fp32_metrics
            }

        return self

    # ------------------------------------------------------------------
    # Step 5: Cost
    # ------------------------------------------------------------------

    def cost(self) -> "Session":
        """Estimate latency and memory for both fp32 and quantized models.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")

        self._cost = self._quant_session.estimate_cost(fp32=False)
        self._cost_fp32 = (
            self._quant_session.estimate_cost(fp32=True)
            if self._keep_fp32
            else None
        )

        return self

    # ------------------------------------------------------------------
    # Inference delegation
    # ------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Forward pass through the quantized model (delegates to _QuantSession)."""
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        return self._quant_session(*args, **kwargs)

    def use_fp32(self) -> "Session":
        """Switch to fp32 mode — ``session(x)`` calls the original model."""
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        self._quant_session.use_fp32()
        return self

    def use_quant(self) -> "Session":
        """Switch to quantized mode — ``session(x)`` calls the quantized model."""
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        self._quant_session.use_quant()
        return self

    @property
    def mode(self) -> str:
        """Current inference mode: ``"fp32"`` or ``"quant"``."""
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        return self._quant_session.mode

    # ------------------------------------------------------------------
    # ONNX Export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        output_path: str,
        dummy_input: Optional[Any] = None,  # Tensor | tuple | list | dict
        opset_version: int = 17,
    ) -> None:
        """Export quantized model to ONNX.

        Delegates to ``_QuantSession.export_onnx()`` which in turn calls
        ``model.export_onnx()``.  If *dummy_input* is not provided, uses
        the input from the most recent ``session(x)`` call.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")
        self._quant_session.export_onnx(
            output_path, dummy_input=dummy_input, opset_version=opset_version,
        )

    # ------------------------------------------------------------------
    # Full pipeline (backward compat)
    # ------------------------------------------------------------------

    def run(
        self,
        calib_data,
        *,
        eval_data=None,
        eval_fn: Optional[Callable] = None,
        outputs: Union[str, List[str]] = "default",
    ) -> SessionResult:
        """Run the full quantization workflow. Backward-compatible shortcut.

        Equivalent to calling ``.quantize() → .calibrate() → .analyze() →
        .evaluate() → .cost() → .result`` in sequence, with conditional
        evaluation and cost estimation based on *outputs*.

        When ``"qsnr"`` is in outputs (the default), both local QSNR (observer)
        and accumulated QSNR (hook path, comparing quant vs fp32 output) are
        collected and stored in separate fields on the result.
        """
        import time as _time
        from src.report._spec import resolve_outputs as _resolve_outputs, PRESETS

        observer_keys, needs_eval, needs_cost = _resolve_outputs(outputs)

        _t0 = _time.perf_counter()
        self.quantize(calib_data=calib_data, eval_fn=eval_fn)
        print(f"    quantize: {_time.perf_counter() - _t0:.1f}s")

        _t0 = _time.perf_counter()
        self.calibrate(calib_data, eval_fn=eval_fn)
        print(f"    calibrate: {_time.perf_counter() - _t0:.1f}s")

        _t0 = _time.perf_counter()
        self.analyze(calib_data, outputs=outputs, eval_fn=eval_fn)
        print(f"    analyze: {_time.perf_counter() - _t0:.1f}s")

        if needs_eval and eval_fn is not None:
            if eval_data is None:
                _logger.warning(
                    "eval_data not provided — falling back to calib_data for "
                    "evaluation. This often produces empty accuracy metrics "
                    "because calib_data is typically a list of tensors. "
                    "Pass eval_data=<your test loader> to Study.run() or "
                    "Session.run() to get accuracy results."
                )
            _t0 = _time.perf_counter()
            self.evaluate(
                eval_data if eval_data is not None else calib_data,
                eval_fn,
            )
            print(f"    evaluate: {_time.perf_counter() - _t0:.1f}s")

        if needs_cost:
            _t0 = _time.perf_counter()
            self.cost()
            print(f"    cost: {_time.perf_counter() - _t0:.1f}s")

        return self.result
