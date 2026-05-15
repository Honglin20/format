"""Thin backward-compat Session wrapper around run_quantization().

Provides the old stateful Session API (calibrate / analyze / compare / etc.)
while delegating to the standalone run_quantization() + helpers internally.

Key fix: ``__getattr__`` proxies to ``self.qmodel`` so custom model attributes
(num_heads, etc.) are accessible through the session object.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import run_quantization
from src.session._helpers import clear_scales as _clear_scales
from src.scheme.op_config import OpQuantConfig


class Session:
    """Backward-compat wrapper providing the old stateful Session API.

    Internally delegates to :func:`run_quantization` on construction.
    ``__getattr__`` proxies unrecognized attributes to ``self.qmodel``.
    """

    def __init__(
        self,
        model: nn.Module,
        config=None,
        *,
        keep_fp32: bool = True,
        calibrator=None,
        observers: Optional[List] = None,
        op_cfgs: Optional[Dict[str, OpQuantConfig]] = None,
        quantize_nonlinear: bool = True,
        **kwargs,
    ):
        from src.calibration.strategies import MaxScaleStrategy
        from src.analysis.observers import QSNRObserver

        # Build QuantConfig from kwargs when config is not QuantConfig
        if isinstance(config, (OpQuantConfig, dict)):
            cfg = QuantConfig()
            self._op_cfg = config
        elif isinstance(config, QuantConfig):
            cfg = config
            self._op_cfg = None
        elif config is None:
            cfg = QuantConfig()
            self._op_cfg = None
        else:
            cfg = config
            self._op_cfg = None

        self._config = cfg
        self._calibrator = calibrator if calibrator is not None else MaxScaleStrategy()
        self._observers = observers if observers is not None else [QSNRObserver()]
        self._mode = "quant"
        self._last_input = None

        # Defer full run — user can call .quantize() + .calibrate() + .analyze()
        # or just .run() for the full pipeline
        self._model = model
        self._keep_fp32 = keep_fp32
        self._op_cfgs = op_cfgs
        self._qnl = quantize_nonlinear
        self._kwargs = kwargs

        self.qmodel: Optional[nn.Module] = None
        self.fp32_model: Optional[nn.Module] = None
        self.result: Any = None

    # ------------------------------------------------------------------
    # Attribute proxying (the key fix)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in ("qmodel", "fp32_model", "result"):
            raise AttributeError(name)
        qm = self.__dict__.get("qmodel")
        if qm is not None:
            return getattr(qm, name)
        raise AttributeError(
            f"Session has no attribute {name!r} and qmodel is not initialized"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Chainable pipeline
    # ------------------------------------------------------------------

    def quantize(self, calib_data=None):
        """Quantize the model (replaces nn.Modules with Quantized equivalents)."""
        if calib_data is None:
            calib_data = [torch.randn(2, 4)]  # minimal default

        if self._op_cfg is not None:
            self.qmodel, self.fp32_model, self.result = run_quantization(
                self._model, self._op_cfg, calib_data,
                keep_fp32=self._keep_fp32,
                calibrator=self._calibrator,
                observers=self._observers,
                op_cfgs=self._op_cfgs,
                quantize_nonlinear=self._qnl,
            )
        else:
            self.qmodel, self.fp32_model, self.result = run_quantization(
                self._model, self._config, calib_data,
                keep_fp32=self._keep_fp32,
                calibrator=self._calibrator,
                observers=self._observers,
                op_cfgs=self._op_cfgs,
                quantize_nonlinear=self._qnl,
            )
        return self

    def calibrate(self, calib_data=None, strategy=None):
        """Return a CalibrationSession context manager for manual calibration."""
        from src.calibration.pipeline import CalibrationSession

        if self.qmodel is None:
            self.quantize(calib_data)
        else:
            # Clear existing scales for re-calibration
            _clear_scales(self.qmodel, None)

        strat = strategy if strategy is not None else self._calibrator
        return CalibrationSession(self.qmodel, strat, assign=True)

    def analyze(self, calib_data=None, outputs=None, observers=None, eval_fn=None):
        """Return an AnalysisContext or run full analysis."""
        if outputs or eval_fn:
            # Full re-run with new parameters
            if calib_data is None:
                calib_data = [torch.randn(2, 4)]
            obs = observers if observers is not None else self._observers
            self.qmodel, self.fp32_model, self.result = run_quantization(
                self._model, self._config, calib_data,
                keep_fp32=self._keep_fp32,
                calibrator=self._calibrator,
                observers=obs,
                op_cfgs=self._op_cfgs,
                quantize_nonlinear=self._qnl,
                outputs=outputs or ["qsnr"],
                eval_fn=eval_fn,
            )
            # Populate old-style internal attributes
            self._observers_data = self.result.observers_data
            self._accum_qsnr_per_layer = self.result.accum_qsnr_per_layer
            self._accum_mse_per_layer = self.result.accum_mse_per_layer
            self._qsnr_per_layer = self.result.qsnr_per_layer
            self._mse_per_layer = self.result.mse_per_layer
            return self

        from src.analysis.context import AnalysisContext
        if self.qmodel is None:
            self.quantize(calib_data)
        return AnalysisContext(self.qmodel, self._observers)

    def evaluate(self, eval_data, eval_fn=None):
        """Run evaluation on the quantized model."""
        if eval_fn is None:
            return {}
        return eval_fn(self.qmodel, eval_data)

    def run(self, calib_data, *, eval_data=None, eval_fn=None,
            outputs="default", keep_fp32=True, overrides=None):
        """Full pipeline: quantize → calibrate → analyze → evaluate."""
        self.qmodel, self.fp32_model, self.result = run_quantization(
            self._model, self._config, calib_data,
            eval_data=eval_data, eval_fn=eval_fn,
            outputs=outputs, keep_fp32=keep_fp32,
            overrides=overrides,
            calibrator=self._calibrator,
            observers=self._observers,
            op_cfgs=self._op_cfgs,
            quantize_nonlinear=self._qnl,
        )
        return self.result

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def use_fp32(self):
        if self.fp32_model is None:
            raise RuntimeError("fp32_model not available (keep_fp32=False)")
        self._mode = "fp32"

    def use_quant(self):
        self._mode = "quant"

    def __call__(self, *args, **kwargs):
        """Forward pass — delegates to fp32_model or qmodel based on mode."""
        if self._mode == "fp32" and self.fp32_model is not None:
            out = self.fp32_model(*args, **kwargs)
        else:
            out = self.qmodel(*args, **kwargs)
        if args:
            self._last_input = args[0] if len(args) == 1 else args
        return out

    def train(self, mode: bool = True):
        if self.qmodel is not None:
            self.qmodel.train(mode)
        return self

    def eval(self):
        if self.qmodel is not None:
            self.qmodel.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.qmodel.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.qmodel.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.qmodel.load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, eval_dataloader, eval_fn=None, directions=None):
        """Auto-mode: compare fp32 vs quant on a DataLoader."""
        from src.analysis.e2e import Comparator

        if self.fp32_model is None:
            raise RuntimeError("fp32_model not available (keep_fp32=False)")

        def _default_eval(logits, labels):
            return {"accuracy": (logits.argmax(-1) == labels).float().mean().item()}

        ef = eval_fn if eval_fn is not None else _default_eval
        cmp = Comparator()
        with cmp, torch.no_grad():
            for batch in eval_dataloader:
                inputs, labels = batch[0], batch[1]
                fp32_out = self.fp32_model(inputs)
                q_out = self.qmodel(inputs)
                cmp.record(fp32_out, q_out, labels)

        result = cmp.evaluate(ef, directions=directions)
        return result

    def comparator(self):
        """Return a standalone Comparator for manual collection."""
        from src.analysis.e2e import Comparator
        return Comparator()

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(self, output_path: str, dummy_input=None):
        """Export the quantized model to ONNX."""
        from src.onnx.export import export_quantized_model

        if dummy_input is None:
            if self._last_input is None:
                raise ValueError("No dummy_input and no _last_input recorded")
            dummy_input = self._last_input

        export_quantized_model(self.qmodel, dummy_input, output_path)

    # ------------------------------------------------------------------
    # Clear scales
    # ------------------------------------------------------------------

    def clear_scales(self):
        """Remove all _output_scale buffers."""
        return _clear_scales(self.qmodel, self._calibrator)

    # ------------------------------------------------------------------
    # Pre-scale
    # ------------------------------------------------------------------

    def initialize_pre_scales(self, calib_data, **kwargs):
        """Initialize _pre_scale buffers."""
        from src.session._helpers import initialize_pre_scales as _init_ps

        if self.qmodel is None:
            self.quantize(calib_data)
        return _init_ps(self.qmodel, calib_data, **kwargs)

    def optimize_scales(self, optimizer, calib_data, **kwargs):
        """Run LSQ optimization on pre-scale parameters."""
        from src.session._helpers import optimize_scales as _opt_scales

        if self.fp32_model is None:
            raise RuntimeError("optimize_scales requires keep_fp32=True")
        return _opt_scales(self.qmodel, self.fp32_model, optimizer, calib_data, **kwargs)
