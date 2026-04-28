"""PreScaleTransform: learnable per-channel pre-scale via the Transform slot."""
import torch
from torch import Tensor
from src.scheme.transform import TransformBase


class PreScaleTransform(TransformBase):
    """Pre-scale transform: x -> x * scale, x_q -> x_q / scale.

    Holds a **reference** (not a copy) to an externally-owned scale tensor.
    This allows the tensor to be an ``nn.Parameter`` updated by an optimizer,
    or a buffer written by calibration — the transform automatically sees
    the latest values on the next forward pass.

    invertible = True.
    """

    invertible = True

    def __init__(self, scale: Tensor):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"scale must be a torch.Tensor, got {type(scale).__name__}")
        object.__setattr__(self, "scale", scale)

    def forward(self, x: Tensor) -> Tensor:
        shape = self.scale.shape + (1,) * (x.ndim - self.scale.ndim)
        return x * self.scale.view(shape)

    def inverse(self, x_q: Tensor) -> Tensor:
        shape = self.scale.shape + (1,) * (x_q.ndim - self.scale.ndim)
        return x_q / self.scale.view(shape)

    def __eq__(self, other):
        if not isinstance(other, PreScaleTransform):
            return NotImplemented
        return self.scale is other.scale

    def __hash__(self):
        return hash(("PreScaleTransform", id(self.scale)))

    def __repr__(self):
        return f"PreScaleTransform(shape={tuple(self.scale.shape)})"
