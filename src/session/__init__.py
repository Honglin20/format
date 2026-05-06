"""Integration layer: model quantization lifecycle.

QuantSession  — orchestrates calibrate → analyze → evaluate → export
QuantizeContext — patches torch/F for inline-op quantization interception
quantize_model — maps nn.Modules to QuantizedXxx equivalents
"""
from ._session import QuantSession
from ._context import QuantizeContext, install_stack_hooks, remove_stack_hooks
from ._model import quantize_model

__all__ = [
    "QuantSession",
    "QuantizeContext",
    "quantize_model",
    "install_stack_hooks",
    "remove_stack_hooks",
]
