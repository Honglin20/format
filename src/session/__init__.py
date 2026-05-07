"""Integration layer: model quantization lifecycle.

Session       — public API: .quantize() → .calibrate() → .analyze() → .evaluate() → .cost()
QuantConfig   — user-facing configuration entry point, translates to OpQuantConfig
SessionResult — result of running a single QuantConfig through Session
QuantizeContext — patches torch/F for inline-op quantization interception
quantize_model — maps nn.Modules to QuantizedXxx equivalents
STUDY_CONFIG  — legacy study configuration dict (from study_config.py)
"""
from ._config import QuantConfig, resolve_config
from ._quant import _QuantSession
from ._result import SessionResult
from ._session import Session
from ._study import Study
from ._per_layer_opt import per_layer_optimal
from ._context import QuantizeContext, install_stack_hooks, remove_stack_hooks
from ._model import quantize_model
from .study_config import STUDY_CONFIG

# Public alias for the low-level session API.
QuantSession = _QuantSession

__all__ = [
    "QuantConfig",
    "resolve_config",
    "Session",
    "SessionResult",
    "Study",
    "per_layer_optimal",
    "QuantSession",
    "QuantizeContext",
    "quantize_model",
    "install_stack_hooks",
    "remove_stack_hooks",
    "STUDY_CONFIG",
]
