"""
_QuantSession: unified high-level API for model quantization workflow.

Wraps quantize_model, calibration, analysis, comparison, and ONNX export
into a single session object.  Designed as a thin layer on top of existing
APIs — no breaking changes to the underlying infrastructure.

Usage::

    session = _QuantSession(model, cfg)

    # Calibrate (scales auto-assigned on exit)
    with session.calibrate():
        eval_fn(session, calib_data)

    # Analyze
    with session.analyze() as ctx:
        eval_fn(session, data)
    report = ctx.report()

    # Evaluate
    fp32_metrics = eval_fn(session.fp32_model, eval_loader)
    quant_metrics = eval_fn(session, eval_loader)

    # Export
    session.export_onnx("model.onnx")
"""
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.calibration.pipeline import CalibrationSession
from src.calibration.strategies import MaxScaleStrategy, ScaleStrategy
from src.analysis.context import AnalysisContext
from src.analysis.e2e import Comparator, compare_models, _default_accuracy
from src.analysis.observers import QSNRObserver
from src.session._model import quantize_model
from src.scheme.op_config import OpQuantConfig


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


class _QuantSession:
    """Unified high-level API for model quantization workflow.

    Wraps quantize_model, calibration, analysis, comparison, and export
    into a single session object.

    Args:
        model: Original fp32 PyTorch model.
        cfg: ``OpQuantConfig`` or ``dict[name → OpQuantConfig]`` for
            per-layer configs.
        calibrator: ``ScaleStrategy`` for calibration (default:
            ``MaxScaleStrategy()``).
        observers: List of Observer instances for analysis (default:
            ``[QSNRObserver()]``).
        keep_fp32: Keep a deep copy of the original fp32 model for
            comparison (default: True).  Set False to save memory.
        op_cfgs: Optional per-op-type overrides for inline ops
            (``{"matmul": cfg, "add": cfg, ...}``).

    Example::

        session = _QuantSession(model, cfg)

        with session.calibrate():
            eval_fn(session, calib_data)

        with session.analyze() as ctx:
            eval_fn(session, data)
        report = ctx.report()

        fp32_metrics = eval_fn(session.fp32_model, eval_loader)
        quant_metrics = eval_fn(session, eval_loader)
        session.export_onnx("model.onnx")
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig]],
        *,
        calibrator: Optional[ScaleStrategy] = None,
        observers: Optional[List] = None,
        keep_fp32: bool = True,
        op_cfgs: Optional[Dict[str, OpQuantConfig]] = None,
        quantize_nonlinear: bool = True,
    ):
        self.cfg = cfg
        self.calibrator = calibrator if calibrator is not None else MaxScaleStrategy()
        self.observers = observers if observers is not None else [QSNRObserver()]
        self.op_cfgs = op_cfgs
        self._mode: str = "quant"
        self._last_input: Any = None

        if keep_fp32:
            self.fp32_model = copy.deepcopy(model)
        else:
            self.fp32_model = None

        # quantize a deepcopy so the user's original model is never mutated.
        # this also allows the same model object to be passed to multiple
        # Session / _QuantSession instances without cross-contamination.
        self.qmodel = quantize_model(
            copy.deepcopy(model),
            cfg=cfg,
            op_cfgs=op_cfgs,
            quantize_nonlinear=quantize_nonlinear,
        )

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def use_fp32(self) -> "_QuantSession":
        """Switch to fp32 mode — ``session(x)`` calls the original model."""
        if self.fp32_model is None:
            raise RuntimeError("fp32_model not available (keep_fp32=False)")
        self._mode = "fp32"
        return self

    def use_quant(self) -> "_QuantSession":
        """Switch to quantized mode — ``session(x)`` calls the quantized model."""
        self._mode = "quant"
        return self

    @property
    def mode(self) -> str:
        """Current inference mode: ``"fp32"`` or ``"quant"``."""
        return self._mode

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Forward pass through the active model (fp32 or quantized).

        In quantized mode, records the first positional argument as
        ``_last_input`` for automatic ONNX export.
        """
        if self._mode == "fp32":
            return self.fp32_model(*args, **kwargs)

        if args and self._last_input is None:
            self._last_input = args[0] if len(args) == 1 else args
        return self.qmodel(*args, **kwargs)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, strategy: Optional[ScaleStrategy] = None) -> CalibrationSession:
        """Return a ``CalibrationSession`` context manager.

        Scales are auto-assigned on context exit. The user runs forward
        passes inside the ``with`` block::

            with session.calibrate():
                eval_fn(session, calib_data)
        """
        strat = strategy if strategy is not None else self.calibrator
        return CalibrationSession(self.qmodel, strat)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, observers: Optional[List] = None) -> AnalysisContext:
        """Return an ``AnalysisContext`` context manager.

        Observers are attached on enter and detached on exit::

            with session.analyze() as ctx:
                eval_fn(session, data)
            report = ctx.report()
        """
        obs = observers if observers is not None else self.observers
        return AnalysisContext(self.qmodel, obs)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def comparator(self) -> Comparator:
        """Return a ``Comparator`` for manual end-to-end comparison.

        Usage::

            cmp = session.comparator()
            with cmp:
                for inputs, labels in data:
                    session.use_fp32()
                    fp32_out = session(inputs)
                    session.use_quant()
                    q_out = session(inputs)
                    cmp.record(fp32_out, q_out, labels)
            result = cmp.evaluate(my_eval_fn)
        """
        return Comparator()

    def compare(
        self,
        eval_dataloader,
        eval_fn: Callable[..., Dict[str, float]] = _default_accuracy,
        directions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Auto-mode: run fp32 and quantized models on *eval_dataloader*.

        Args:
            eval_dataloader: DataLoader yielding ``(inputs, labels)``.
            eval_fn: ``(logits, labels) -> dict[str, float]``.
            directions: Optional ``{"metric": "higher"|"lower"}`` hints.

        Returns:
            ``{"fp32": {...}, "quant": {...}, "delta": {...}}``
        """
        if self.fp32_model is None:
            raise RuntimeError("fp32_model not available (keep_fp32=False)")
        return compare_models(
            self.fp32_model, self.qmodel, eval_dataloader,
            eval_fn=eval_fn, directions=directions,
        )

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

        If *dummy_input* is not provided, uses the input from the most
        recent ``session(x)`` call.
        """
        inp = dummy_input if dummy_input is not None else self._last_input
        if inp is None:
            raise ValueError(
                "No dummy_input provided and no prior inference recorded. "
                "Call session(x) first or pass dummy_input explicitly."
            )
        self.qmodel.export_onnx(inp, output_path, opset_version=opset_version)

    # ------------------------------------------------------------------
    # Cost estimation (P6)
    # ------------------------------------------------------------------

    def estimate_cost(self, fp32: bool = False) -> "CostReport":
        """Estimate latency and memory for the current model.

        Does not require a forward pass — inspects model graph structure
        and quantization configs directly.

        Args:
            fp32: If True, estimate the fp32 baseline; otherwise the
                quantized model.

        Returns:
            CostReport with per-layer and total estimates.

        Raises:
            RuntimeError: If fp32=True but keep_fp32=False was set.
        """
        from src.cost.model_cost import analyze_model_cost

        if fp32:
            if self.fp32_model is None:
                raise RuntimeError(
                    "fp32_model not available (keep_fp32=False). "
                    "Cannot estimate fp32 cost."
                )
            model = self.fp32_model
        else:
            model = self.qmodel
        return analyze_model_cost(model)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_scales(self) -> List[str]:
        """Remove all ``_output_scale`` buffers from the quantized model."""
        cs = CalibrationSession(self.qmodel, self.calibrator, assign=False)
        return cs.clear_scales()

    # ------------------------------------------------------------------
    # Pre-scale (P5: learnable pre-scale via Transform)
    # ------------------------------------------------------------------

    def initialize_pre_scales(
        self,
        calib_data: Any,
        *,
        init: str = "ones",
        pot: bool = False,
        granularity: str = "per_tensor",
        trainable: bool = False,
        channel_axis: int = -1,
        eval_fn: Optional[Callable] = None,
    ) -> int:
        """Initialize ``_pre_scale`` tensors on all quantized modules.

        Args:
            calib_data: Calibration data passed through to ``eval_fn`` (only
                used when ``init`` is ``"amax"`` or ``"pot_amax"``).
                When ``eval_fn`` is None, must be ``List[Tensor]``.
                When ``eval_fn`` is provided, can be any type the user's
                function accepts.
            init: Initialization method:

                - ``"ones"`` — identity (all ones), for LSQ optimisation.
                - ``"amax"`` — ``1 / max(|x|)`` from calibration data.
                - ``"pot_amax"`` — ``2 ** round(log2(1/amax))`` (PoT).

            pot: If True, create a PreScaleTransform that projects the scale
                to nearest power-of-two at every forward pass.
            granularity: ``"per_tensor"`` (single scalar per module) or
                ``"per_channel"`` (one scalar per output channel).
                Per-channel only replaces activation-role transforms
                (input/output/grad_*) — weight/bias keep their originals.
            trainable: If True, register as ``nn.Parameter`` (gradient-based
                optimisation). If False, register as a buffer.
            channel_axis: Which dimension is the channel axis for per-channel
                broadcasting (default -1, the last dim).
            eval_fn: ``(model, data) -> Any``. Controls how the model is
                called during amax collection (when ``init`` is ``"amax"``
                or ``"pot_amax"``). When None, falls back to direct
                ``model(batch)`` calls.

        Returns:
            Number of modules that received pre-scale tensors.
        """
        from src.transform.pre_scale import PreScaleTransform
        from src.session._model import _get_quantized_modules
        from src.calibration.lsq_optimizer import (
            _replace_transform,
            _replace_transform_activation_only, _INPUT_ACTIVATION_ROLES,
        )

        if init not in ("ones", "amax", "pot_amax"):
            raise ValueError(f"Unknown init method: {init!r}")
        if granularity not in ("per_tensor", "per_channel"):
            raise ValueError(
                f"Unknown granularity: {granularity!r}"
            )

        # Collect per-module input amax when init requires calibration data
        amax_map = None
        if init in ("amax", "pot_amax"):
            amax_map = _QuantSession._collect_input_amax(
                calib_data, self.qmodel,
                channel_axis=channel_axis,
                granularity=granularity,
                eval_fn=eval_fn,
            )

        count = 0
        for name, module in _get_quantized_modules(self.qmodel):
            device = _module_device(module)

            # Compute initial pre-scale
            if init == "ones":
                if granularity == "per_channel":
                    out_channels = self._infer_out_channels(module)
                    if out_channels is None:
                        continue
                    init_scale = torch.ones(out_channels, device=device)
                else:
                    init_scale = torch.ones(1, device=device)
            else:  # "amax" or "pot_amax"
                amax = amax_map.get(name)
                if amax is None:
                    continue
                amax = amax.to(device).clamp(min=1e-12)
                init_scale = torch.tensor(1.0, device=device) / amax
                if init == "pot_amax":
                    init_scale = 2 ** torch.round(torch.log2(init_scale))

            # Register as buffer or Parameter
            if trainable:
                module.register_parameter("_pre_scale", nn.Parameter(init_scale))
            else:
                module.register_buffer("_pre_scale", init_scale)

            # Update cfg to route through PreScaleTransform
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
        self,
        optimizer: "LayerwiseScaleOptimizer",
        calib_data: Any,
        *,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run layer-wise LSQ optimization on pre-scale parameters.

        Args:
            optimizer: Configured ``LayerwiseScaleOptimizer`` instance.
            calib_data: Calibration data passed through to ``eval_fn``.
                When ``eval_fn`` is None, must be ``List[Tensor]``.
                When ``eval_fn`` is provided, can be any type.
            eval_fn: ``(model, data) -> Any``. Controls model interaction
                during LSQ. When None, falls back to iterating
                ``model(batch) for batch in calib_data``.

        Returns:
            Dict mapping module name → optimized pre_scale tensor.

        Raises:
            RuntimeError: If ``fp32_model`` is not available
                (``keep_fp32=False`` was passed to _QuantSession).
        """
        if self.fp32_model is None:
            raise RuntimeError(
                "optimize_scales requires fp32_model (keep_fp32=True)"
            )

        return optimizer.optimize(self.qmodel, self.fp32_model, calib_data, eval_fn=eval_fn)

    @staticmethod
    def _infer_out_channels(module) -> int:
        """Infer output channel count for a module."""
        if hasattr(module, "out_features"):
            return module.out_features
        if hasattr(module, "out_channels"):
            return module.out_channels
        if hasattr(module, "num_features"):
            return module.num_features
        return None

    @staticmethod
    def _collect_input_amax(
        calib_data: Any,
        model: nn.Module,
        *,
        channel_axis: int = -1,
        granularity: str = "per_tensor",
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Collect per-module input activation max-abs via forward hooks.

        Runs a forward pass with *calib_data* on *model*, hooking into
        every quantized module's input to compute amax.

        Args:
            calib_data: Calibration data passed through to ``eval_fn``.
                When ``eval_fn`` is None, must be ``List[Tensor]``.
                When ``eval_fn`` is provided, can be any type.
            model: Quantized model.
            channel_axis: Which dim is the channel (default -1, last dim).
            granularity: ``"per_tensor"`` — scalar amax per module;
                         ``"per_channel"`` — 1D amax along channel_axis.
            eval_fn: ``(model, data) -> Any``. Controls how the model is
                called during activation capture. When None, falls back to
                ``for batch in calib_data: model(batch)``.

        Returns:
            Dict mapping module name → amax tensor (shape ``()`` for
            per_tensor, ``(C,)`` for per_channel).
        """
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

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "_QuantSession":
        """Set training mode on both fp32 and quantized models."""
        self.qmodel.train(mode)
        if self.fp32_model is not None:
            self.fp32_model.train(mode)
        return self

    def eval(self) -> "_QuantSession":
        """Set evaluation mode on both fp32 and quantized models."""
        return self.train(False)

    def parameters(self):
        """Return an iterator over the quantized model's parameters."""
        return self.qmodel.parameters()

    def state_dict(self):
        """Return the quantized model's state_dict."""
        return self.qmodel.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a state_dict into the quantized model."""
        return self.qmodel.load_state_dict(state_dict, strict=strict)
