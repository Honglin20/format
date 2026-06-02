"""microxcaling — incremental quantization library built on microsoft/microxcaling.

Public API
----------
- ``quantize_model`` — maps nn.Modules to QuantizedXxx equivalents
- ``QuantConfig`` — user-facing configuration entry point
- ``Study`` — aggregate multiple quantization config comparisons
- ``SessionResult`` / ``SessionReport`` / ``StudyReport`` — results and reporting
- ``QuantizeContext`` — torch/F patching for inline-op quantization interception
- ``per_layer_optimal`` — per-layer optimal transform selection
"""
from src.session import (
    Study,
    QuantConfig,
    SessionResult,
    per_layer_optimal,
    QuantizeContext,
    quantize_model,
)
from src.report import SessionReport, StudyReport

__all__ = [
    "Study",
    "QuantConfig",
    "SessionResult",
    "SessionReport",
    "StudyReport",
    "per_layer_optimal",
    "QuantizeContext",
    "quantize_model",
]
