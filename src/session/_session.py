"""Session execution unit and SessionResult dataclass.

Session is the atomic execution unit: one QuantConfig → one SessionResult.
It wraps QuantSession and orchestrates calibrate → analyze → evaluate → cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.analysis.observers import (
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
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.session._config import QuantConfig
from src.session._quant import QuantSession
from src.transform.smooth_quant import SmoothQuantTransform, fuse_smoothquant_weights


# ---------------------------------------------------------------------------
# Observer key -> observer class mapping
# ---------------------------------------------------------------------------

_OBSERVER_MAP: Dict[str, type] = {
    "qsnr": QSNRObserver,
    "mse": MSEObserver,
    "histogram": HistogramObserver,
    "distribution": DistributionObserver,
}

# Roles that receive SmoothQuantTransform when patching per-layer configs.
# Only input-side roles are patched because the output side has a different
# channel count (out_features vs in_features) and the scale computed by
# from_model_calibration matches the input channel dimension.
# Weight must keep IdentityTransform since it was already fused.
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
    """Run *model* on *data*, forwarding via *eval_fn* when provided.

    When *eval_fn* is ``None``, falls back to calling the model directly:
    iterating over a list/tuple of batches, or calling once for a single
    tensor.  This guarantees forward hooks fire regardless of data format.
    """
    if eval_fn is not None:
        eval_fn(model, data)
    elif isinstance(data, (list, tuple)):
        for batch in data:
            model(batch)
    else:
        model(data)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """Atomic execution unit: one QuantConfig -> one SessionResult.

    Wraps QuantSession and orchestrates:
      1. Output resolution (observer_keys, needs_eval, needs_cost)
      2. [if smoothquant] SQ calibration + weight fusion
      3. to_op_config() translation
      4. [if prescale] two-step prescale init + optional LSQ optimization
      5. calibrate -> analyze -> evaluate -> cost

    Args:
        model: Original fp32 PyTorch model.
        config: User-facing ``QuantConfig``.
        keep_fp32: Keep a deep copy of the fp32 model for comparison
            (default: ``True``).
    """

    def __init__(
        self,
        model: nn.Module,
        config: QuantConfig,
        *,
        keep_fp32: bool = True,
    ):
        self._model = model
        self._config = config
        self._keep_fp32 = keep_fp32

    def run(
        self,
        calib_data,
        *,
        eval_data=None,
        eval_fn: Optional[Callable] = None,
        outputs: Union[str, List[str]] = "default",
    ) -> SessionResult:
        """Run the full quantization workflow for this config.

        Args:
            calib_data: Calibration data.  When ``eval_fn`` is ``None``,
                must be a sequence of tensors (iterated directly) or a
                single tensor.
            eval_data: Optional separate evaluation data.  Defaults to
                ``calib_data`` when ``None``.
            eval_fn: ``(model, data) -> Dict[str, float]``.  When provided,
                called for calibration forward passes (return value ignored)
                and for evaluation (return value used as metrics).  When
                ``None``, direct ``model(batch)`` calls are used for
                calibration and analysis, and evaluation is skipped.
            outputs: Output key or list of keys.  ``"default"`` resolves
                to ``["accuracy", "qsnr"]``; ``"all"`` resolves to every
                registered output.

        Returns:
            ``SessionResult`` containing all computed metrics, observer
            data, cost estimates, and the cached ``sq_transforms``.
        """
        # ------------------------------------------------------------------
        # 1. Resolve outputs -> observer_keys / needs_eval / needs_cost
        # ------------------------------------------------------------------
        from src.report._spec import resolve_outputs as _resolve_outputs

        observer_keys: set
        needs_eval: bool
        needs_cost: bool
        observer_keys, needs_eval, needs_cost = _resolve_outputs(outputs)
        observer_keys_set: set = set(observer_keys)

        # ------------------------------------------------------------------
        # 2. Map observer keys -> observer instances
        # ------------------------------------------------------------------
        observers = [_OBSERVER_MAP[k]() for k in sorted(observer_keys_set)]

        # ------------------------------------------------------------------
        # 3. SmoothQuant: compute per-channel scales, fuse weights
        # ------------------------------------------------------------------
        if self._config.transform == "smoothquant":
            sq_transforms = SmoothQuantTransform.from_model_calibration(
                self._model,
                calib_data,
                alpha=self._config.sq_alpha,
                eval_fn=eval_fn,
            )
            model = fuse_smoothquant_weights(self._model, sq_transforms)
        else:
            model = self._model
            sq_transforms = None

        # ------------------------------------------------------------------
        # 4. Build base OpQuantConfig
        # ------------------------------------------------------------------
        op_cfg = self._config.to_op_config()

        # ------------------------------------------------------------------
        # 5. Create QuantSession
        # ------------------------------------------------------------------
        calibrator = _make_calibrator(self._config.calibrator)
        qs = QuantSession(
            model,
            op_cfg,
            calibrator=calibrator,
            observers=observers,
            keep_fp32=self._keep_fp32,
        )

        # ------------------------------------------------------------------
        # 6. Patch per-layer SmoothQuant transforms into module configs
        #
        # ``config.to_op_config()`` creates a dummy SmoothQuantTransform
        # with scale=1.0.  Here we substitute the *real* per-layer scale
        # from ``from_model_calibration`` so that activation smoothing
        # matches the weight fusion.
        # ------------------------------------------------------------------
        if sq_transforms:
            for name, sq_t in sq_transforms.items():
                module = dict(qs.qmodel.named_modules()).get(name)
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
                            scale_format=scheme.scale_format,
                        )
                    else:
                        new_kwargs[f_name] = scheme
                module.cfg = OpQuantConfig(**new_kwargs)

        # ------------------------------------------------------------------
        # 7. Prescale: two-step translation Step 2
        # ------------------------------------------------------------------
        if self._config.transform == "prescale":
            qs.initialize_pre_scales(
                calib_data,
                init=self._config.prescale_init,
                pot=self._config.prescale_pot,
                granularity=self._config.prescale_granularity,
            )
            if self._config.lsq_steps > 0:
                from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

                opt = LayerwiseScaleOptimizer(
                    num_steps=self._config.lsq_steps,
                    lr=self._config.lsq_lr,
                )
                qs.optimize_scales(opt, calib_data, eval_fn=eval_fn)

        # ------------------------------------------------------------------
        # 8. Calibrate
        # ------------------------------------------------------------------
        with qs.calibrate():
            _run_model(qs, calib_data, eval_fn)

        # ------------------------------------------------------------------
        # 9. Analyze (only when observers are needed)
        # ------------------------------------------------------------------
        qsnr_per_layer: Dict[str, float] = {}
        mse_per_layer: Dict[str, float] = {}
        observers_data: Dict[str, Any] = {}

        if observers:
            with qs.analyze(observers=observers) as ctx:
                _run_model(qs, calib_data, eval_fn)
            report = ctx.report()
            observers_data = report._raw
            # Extract per-layer qsnr and mse from the nested report structure
            for layer, roles in observers_data.items():
                for _role, stages in roles.items():
                    for _stage, slices in stages.items():
                        for _slice_key, metrics in slices.items():
                            if "qsnr_db" in metrics:
                                qsnr_per_layer[layer] = max(
                                    qsnr_per_layer.get(layer, 0.0),
                                    metrics["qsnr_db"],
                                )
                            if "mse" in metrics:
                                mse_per_layer[layer] = max(
                                    mse_per_layer.get(layer, 0.0),
                                    metrics["mse"],
                                )

        # ------------------------------------------------------------------
        # 10. Evaluate (only when needed by the requested outputs)
        # ------------------------------------------------------------------
        fp32_metrics: Optional[Dict[str, float]] = None
        quant_metrics: Optional[Dict[str, float]] = None
        delta: Optional[Dict[str, float]] = None

        if needs_eval and eval_fn is not None:
            if eval_data is None:
                eval_data = calib_data
            if self._keep_fp32 and qs.fp32_model is not None:
                fp32_metrics = eval_fn(qs.fp32_model, eval_data)
            quant_metrics = eval_fn(qs, eval_data)
            if fp32_metrics is not None:
                delta = {
                    k: fp32_metrics[k] - quant_metrics[k]
                    for k in fp32_metrics
                }

        # ------------------------------------------------------------------
        # 11. Cost (only when needed by the requested outputs)
        # ------------------------------------------------------------------
        cost: Any = None
        cost_fp32: Any = None
        if needs_cost:
            cost = qs.estimate_cost(fp32=False)
            if self._keep_fp32:
                cost_fp32 = qs.estimate_cost(fp32=True)

        # ------------------------------------------------------------------
        # 12. Return result
        # ------------------------------------------------------------------
        return SessionResult(
            name=self._config.name,
            config=self._config,
            fp32_metrics=fp32_metrics,
            quant_metrics=quant_metrics,
            delta=delta,
            qsnr_per_layer=qsnr_per_layer,
            mse_per_layer=mse_per_layer,
            observers_data=observers_data,
            cost=cost,
            cost_fp32=cost_fp32,
            sq_transforms=sq_transforms,
        )


# ---------------------------------------------------------------------------
# SessionResult
# ---------------------------------------------------------------------------


@dataclass
class SessionResult:
    """Result of running a single Session (one QuantConfig).

    This replaces pipeline/runner.py:ExperimentResult with the addition
    of the config field and sq_transforms cache (fixing C1).
    """

    name: str
    config: QuantConfig
    fp32_metrics: Optional[Dict[str, float]] = None
    quant_metrics: Optional[Dict[str, float]] = None
    delta: Optional[Dict[str, float]] = None
    qsnr_per_layer: Dict[str, float] = field(default_factory=dict)
    mse_per_layer: Dict[str, float] = field(default_factory=dict)
    observers_data: Dict[str, Any] = field(default_factory=dict)
    cost: Any = None
    cost_fp32: Any = None
    sq_transforms: Optional[Dict[str, Any]] = None
