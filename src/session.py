"""
QuantSession: unified high-level API for model quantization workflow.

Wraps quantize_model, calibration, analysis, comparison, and ONNX export
into a single session object.  Designed as a thin layer on top of existing
APIs — no breaking changes to the underlying infrastructure.

Usage::

    session = QuantSession(model, cfg)

    # Calibrate (scales auto-assigned on exit)
    with session.calibrate():
        for batch in calib_data:
            session(batch)

    # Analyze
    with session.analyze() as ctx:
        for batch in data:
            session(batch)
    report = ctx.report()

    # Compare against fp32 baseline
    result = session.compare(eval_loader, my_eval_fn)

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
from src.mapping.quantize_model import quantize_model
from src.scheme.op_config import OpQuantConfig


class QuantSession:
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

        session = QuantSession(model, cfg)

        with session.calibrate():
            for batch in calib_data:
                session(batch)

        with session.analyze() as ctx:
            for batch in data:
                session(batch)
        report = ctx.report()

        result = session.compare(eval_loader, my_eval_fn)
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

        self.qmodel = quantize_model(
            model,
            cfg=cfg,
            op_cfgs=op_cfgs,
        )

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def use_fp32(self) -> "QuantSession":
        """Switch to fp32 mode — ``session(x)`` calls the original model."""
        if self.fp32_model is None:
            raise RuntimeError("fp32_model not available (keep_fp32=False)")
        self._mode = "fp32"
        return self

    def use_quant(self) -> "QuantSession":
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
            self._last_input = args[0]
        return self.qmodel(*args, **kwargs)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, strategy: Optional[ScaleStrategy] = None) -> CalibrationSession:
        """Return a ``CalibrationSession`` context manager.

        Scales are auto-assigned on context exit. The user runs forward
        passes inside the ``with`` block::

            with session.calibrate():
                for batch in calib_data:
                    session(batch)
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
                for batch in data:
                    session(batch)
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
        dummy_input: Optional[torch.Tensor] = None,
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
    # Utilities
    # ------------------------------------------------------------------

    def clear_scales(self) -> List[str]:
        """Remove all ``_output_scale`` buffers from the quantized model."""
        cs = CalibrationSession(self.qmodel, self.calibrator, assign=False)
        return cs.clear_scales()

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "QuantSession":
        """Set training mode on both fp32 and quantized models."""
        self.qmodel.train(mode)
        if self.fp32_model is not None:
            self.fp32_model.train(mode)
        return self

    def eval(self) -> "QuantSession":
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
