"""
QuantEvent: event emitted by quantized operators for analysis.

Carries pre/post quantization tensors plus the QuantScheme used,
so observers can compute per-slice metrics without touching operator code.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from src.scheme.quant_scheme import QuantScheme


@dataclass(frozen=True)
class QuantEvent:
    """Event emitted at each quantization point inside an operator.

    Not designed for value equality — Tensor fields make __eq__ ambiguous.
    Use individual field access instead of == comparison.

    Attributes:
        layer_name: Module path in the model (e.g. "model.layer1.linear").
        role: Tensor role: "input" / "weight" / "output" / "grad_*".
        pipeline_index: Step index in the scheme pipeline (0 = first scheme, etc.).
        stage: Event point name (e.g. "input_pre_quant", "output_post_quant").
        fp32_tensor: Input tensor before quantization (detached, read-only).
        quant_tensor: Output tensor after quantization (detached, read-only).
        scheme: QuantScheme used for this quantization step.
        group_map: Optional group_id tensor for dynamic-grouping quantization.
    """
    layer_name: str
    role: str
    pipeline_index: int
    stage: str
    fp32_tensor: Tensor
    quant_tensor: Tensor
    scheme: QuantScheme
    group_map: Optional[Tensor] = None

    def __post_init__(self):
        if not isinstance(self.layer_name, str) or not self.layer_name:
            raise TypeError(
                f"layer_name must be a non-empty str, got {self.layer_name!r}"
            )
        if not isinstance(self.role, str) or not self.role:
            raise TypeError(
                f"role must be a non-empty str, got {self.role!r}"
            )
        if not isinstance(self.pipeline_index, int) or self.pipeline_index < 0:
            raise ValueError(
                f"pipeline_index must be a non-negative int, got {self.pipeline_index!r}"
            )
        if not isinstance(self.stage, str) or not self.stage:
            raise TypeError(
                f"stage must be a non-empty str, got {self.stage!r}"
            )
        if not isinstance(self.fp32_tensor, Tensor):
            raise TypeError(
                f"fp32_tensor must be a Tensor, got {type(self.fp32_tensor).__name__}"
            )
        if not isinstance(self.quant_tensor, Tensor):
            raise TypeError(
                f"quant_tensor must be a Tensor, got {type(self.quant_tensor).__name__}"
            )
        if not isinstance(self.scheme, QuantScheme):
            raise TypeError(
                f"scheme must be a QuantScheme, got {type(self.scheme).__name__}"
            )
        if self.group_map is not None and not isinstance(self.group_map, Tensor):
            raise TypeError(
                f"group_map must be a Tensor or None, got {type(self.group_map).__name__}"
            )
