"""Integration layer: model quantization lifecycle.

QuantSession  — orchestrates calibrate → analyze → evaluate → export
QuantConfig   — user-facing configuration entry point, translates to OpQuantConfig
SessionResult — result of running a single QuantConfig through Session
QuantizeContext — patches torch/F for inline-op quantization interception
quantize_model — maps nn.Modules to QuantizedXxx equivalents
STUDY_CONFIG  — legacy study configuration dict (from study_config.py)
"""
from ._config import QuantConfig, resolve_config
from ._quant import QuantSession
from ._session import Session, SessionResult
from ._study import Study
from ._per_layer_opt import per_layer_optimal
from ._context import QuantizeContext, install_stack_hooks, remove_stack_hooks
from ._model import quantize_model
from .study_config import STUDY_CONFIG

__all__ = [
    "QuantConfig",
    "resolve_config",
    "QuantSession",
    "Session",
    "SessionResult",
    "Study",
    "per_layer_optimal",
    "QuantizeContext",
    "quantize_model",
    "install_stack_hooks",
    "remove_stack_hooks",
    "STUDY_CONFIG",
]
