from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class CalibrationBuffers:
    """All calibration-time buffers for a quantized op.

    Passed as a single argument to LinearFunction / ConvFunction so that
    adding a new buffer type only requires changing this dataclass, not
    every function signature, apply() call, backward() return tuple, and
    symbolic() signature.
    """
    output_scale: Optional[torch.Tensor] = None
    input_scale: Optional[torch.Tensor] = None
    output_mask: Optional[torch.Tensor] = None
    output_scale_o: Optional[torch.Tensor] = None
    input_mask: Optional[torch.Tensor] = None
    input_scale_o: Optional[torch.Tensor] = None
    weight_importance: Optional[torch.Tensor] = None
    weight_scale: Optional[torch.Tensor] = None
    input_sq_activation_mask: Optional[torch.Tensor] = None
    output_group_mask: Optional[torch.Tensor] = None
    input_group_mask: Optional[torch.Tensor] = None
