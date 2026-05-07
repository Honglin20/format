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
from src.session._quant import _QuantSession
from src.session._result import SessionResult
from src.transform.smooth_quant import SmoothQuantTransform, fuse_smoothquant_weights


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_qsnr_mse(observers_data: dict):
    """Extract per-layer QSNR and MSE from a nested observer report raw dict.

    Returns:
        (qsnr_per_layer, mse_per_layer) — each is ``Dict[str, float]``.
    """
    qsnr_per_layer: Dict[str, float] = {}
    mse_per_layer: Dict[str, float] = {}
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
    return qsnr_per_layer, mse_per_layer


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
    ):
        self._model = model
        self._config = config
        self._keep_fp32 = keep_fp32

        # Lazily initialized by .quantize()
        self._quant_session: Optional[_QuantSession] = None
        self._sq_transforms: Optional[Dict[str, Any]] = None

        # Collected results (populated by each step)
        self._qsnr_per_layer: Dict[str, float] = {}
        self._mse_per_layer: Dict[str, float] = {}
        self._observers_data: Dict[str, Any] = {}
        self._fp32_metrics: Optional[Dict[str, float]] = None
        self._quant_metrics: Optional[Dict[str, float]] = None
        self._delta: Optional[Dict[str, float]] = None
        self._cost: Any = None
        self._cost_fp32: Any = None

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
            observers_data=self._observers_data,
            cost=self._cost,
            cost_fp32=self._cost_fp32,
            sq_transforms=self._sq_transforms,
        )

    # ------------------------------------------------------------------
    # Step 1: Quantize
    # ------------------------------------------------------------------

    def quantize(self, *, calib_data=None) -> "Session":
        """Build the quantized model. Must be called first.

        After this method returns, ``session.qmodel`` is available for
        manual inference and ``session(x)`` delegates to the quantized model.

        Args:
            calib_data: Calibration data. Required when ``transform`` is
                ``"smoothquant"`` or ``"prescale"``. Not needed for other
                transforms or MX per_block formats.

        Raises:
            ValueError: If ``calib_data`` is ``None`` but the transform
                requires it.
        """
        # ---- SmoothQuant: compute per-channel scales + fuse weights ----
        if self._config.transform == "smoothquant":
            if calib_data is None:
                raise ValueError(
                    "calib_data is required for smoothquant transform"
                )
            self._sq_transforms = SmoothQuantTransform.from_model_calibration(
                self._model,
                calib_data,
                alpha=self._config.sq_alpha,
            )
            model = fuse_smoothquant_weights(self._model, self._sq_transforms)
        else:
            model = self._model
            self._sq_transforms = None

        # ---- Build OpQuantConfig ----
        op_cfg = self._config.to_op_config()

        # ---- Create _QuantSession ----
        calibrator = _make_calibrator(self._config.calibrator)
        self._quant_session = _QuantSession(
            model,
            op_cfg,
            calibrator=calibrator,
            observers=[],
            keep_fp32=self._keep_fp32,
            quantize_nonlinear=self._config.quantize_nonlinear,
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
            )
            if self._config.lsq_steps > 0:
                from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

                opt = LayerwiseScaleOptimizer(
                    num_steps=self._config.lsq_steps,
                    lr=self._config.lsq_lr,
                )
                self._quant_session.optimize_scales(opt, calib_data)

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

        For MX per_block formats this is a **no-op** — scales are computed
        dynamically during inference by :func:`quantize_mx`.

        Args:
            calib_data: Calibration data (list of tensors or single tensor).
            eval_fn: Optional ``(model, data) -> Any`` for custom model
                interaction during calibration.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")

        if not _needs_calibration(self._quant_session.cfg):
            return self  # MX per_block: scales computed dynamically

        with self._quant_session.calibrate():
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
        """Run error analysis with observers on the quantized model.

        Args:
            calib_data: Data to run through the model for analysis.
            outputs: Output keys — ``"default"``, ``"all"``, or a list of
                specific keys (``"qsnr"``, ``"mse"``, ``"histogram"``, ...).
            eval_fn: Optional ``(model, data) -> Any`` for custom model
                interaction.

        Raises:
            RuntimeError: If ``.quantize()`` has not been called yet.
        """
        if self._quant_session is None:
            raise RuntimeError("Call .quantize() first")

        from src.report._spec import resolve_outputs as _resolve_outputs

        observer_keys, _needs_eval, _needs_cost = _resolve_outputs(outputs)
        observer_keys_set = set(observer_keys)

        if not observer_keys_set:
            return self

        observers = [_OBSERVER_MAP[k]() for k in sorted(observer_keys_set)]

        with self._quant_session.analyze(observers=observers) as ctx:
            _run_model(self._quant_session, calib_data, eval_fn)

        report = ctx.report()
        self._observers_data = report._raw
        self._qsnr_per_layer, self._mse_per_layer = _extract_qsnr_mse(self._observers_data)

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
                k: self._fp32_metrics[k] - self._quant_metrics[k]
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
        """
        from src.report._spec import resolve_outputs as _resolve_outputs

        observer_keys, needs_eval, needs_cost = _resolve_outputs(outputs)

        self.quantize(calib_data=calib_data)
        self.calibrate(calib_data, eval_fn=eval_fn)
        self.analyze(calib_data, outputs=outputs, eval_fn=eval_fn)

        if needs_eval and eval_fn is not None:
            self.evaluate(
                eval_data if eval_data is not None else calib_data,
                eval_fn,
            )

        if needs_cost:
            self.cost()

        return self.result
