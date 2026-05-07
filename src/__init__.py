"""microxcaling — incremental quantization library built on microsoft/microxcaling.

Public API
----------
- ``Session`` / ``Study`` — high-level quantization workflow
- ``QuantConfig`` — user-facing configuration entry point
- ``SessionResult`` / ``SessionReport`` / ``StudyReport`` — results and reporting
- ``QuantizeContext`` — torch/F patching for inline-op quantization interception
- ``quantize_model`` — maps nn.Modules to QuantizedXxx equivalents
- ``per_layer_optimal`` — per-layer optimal transform selection
"""
from src.session import (
    Session,
    Study,
    QuantConfig,
    SessionResult,
    per_layer_optimal,
    QuantizeContext,
    quantize_model,
)
from src.report import SessionReport, StudyReport

__all__ = [
    "Session",
    "Study",
    "QuantConfig",
    "SessionResult",
    "SessionReport",
    "StudyReport",
    "per_layer_optimal",
    "QuantizeContext",
    "quantize_model",
]
