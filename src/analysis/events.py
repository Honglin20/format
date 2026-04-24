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
