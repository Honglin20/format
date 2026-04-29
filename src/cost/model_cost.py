"""Model-level cost analysis: walk nn.Module tree and aggregate per-layer costs."""
from __future__ import annotations
from typing import Optional

import torch.nn as nn

from .device import DeviceSpec
from .op_cost import op_cost, OpCost
from .report import CostReport


def analyze_model_cost(
    model: nn.Module,
    shapes: Optional[dict] = None,
    device: Optional[DeviceSpec] = None,
    model_name: str = "",
) -> CostReport:
    """Walk model and compute per-layer + total cost.

    Args:
        model: PyTorch model (quantized or FP32).
        shapes: Dict with shape hints (batch, in_features, etc.).
                Missing keys are inferred from module attributes.
        device: Target DeviceSpec. Default: DeviceSpec.a100().
        model_name: Label for the report.

    Returns:
        CostReport with per-layer OpCost entries and aggregate totals.
    """
    if shapes is None:
        shapes = {}
    if device is None:
        device = DeviceSpec.a100()

    layers = []
    for name, module in model.named_modules():
        cost = _try_cost(module, name, shapes, device)
        if cost is not None:
            layers.append(cost)

    return CostReport(layers=layers, model_name=model_name)


def _try_cost(module: nn.Module, name: str, shapes: dict, device: DeviceSpec) -> Optional[OpCost]:
    """Return OpCost for module if it's a recognized leaf op, else None."""
    # Skip container modules (they are just structural, not compute ops)
    if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
        return None
    # Try to compute cost; skip unrecognized modules (e.g. custom containers)
    cost = op_cost(module, shapes, device)
    if cost.op_type == "unknown":
        return None
    # Use the module path name; for root module, use the class name
    cost.op_name = name if name else type(module).__name__.lower()
    return cost
