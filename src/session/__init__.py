"""Integration layer: model quantization lifecycle.

QuantConfig      — user-facing configuration entry point, translates to OpQuantConfig
SessionResult    — structured result from a quantization run
Study            — aggregate multiple quantization config comparisons
QuantizeContext  — patches torch/F for inline-op quantization interception
quantize_model   — maps nn.Modules to QuantizedXxx equivalents
per_layer_optimal — post-hoc per-layer optimal transform selection
STUDY_CONFIG     — legacy study configuration dict (from study_config.py)

Usage::

    qmodel = quantize_model(copy.deepcopy(model), cfg.to_op_config())

    with CalibrationSession(qmodel, MaxScaleStrategy()):
        for batch in calib_data:
            qmodel(batch)

    with AnalysisContext(qmodel, [QSNRObserver()]) as ctx:
        for batch in calib_data:
            qmodel(batch)
"""
from ._config import QuantConfig, resolve_config
from ._result import SessionResult
from ._study import Study
from ._per_layer_opt import per_layer_optimal
from ._context import QuantizeContext, install_stack_hooks, remove_stack_hooks
from ._model import quantize_model
from ._compat import Session
from .study_config import STUDY_CONFIG

__all__ = [
    "QuantConfig",
    "SessionResult",
    "Study",
    "Session",
    "per_layer_optimal",
    "QuantizeContext",
    "quantize_model",
]
