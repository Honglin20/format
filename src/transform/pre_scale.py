"""PreScaleTransform: learnable per-channel pre-scale via the Transform slot."""
import torch
from torch import Tensor
from src.scheme.transform import TransformBase


def _pot_scale(scale: Tensor) -> Tensor:
    """Project *scale* to the nearest power-of-two: ``2 ** round(log2(scale))``."""
    return 2 ** torch.round(torch.log2(scale))


class PreScaleTransform(TransformBase):
    """Pre-scale transform: x -> x * scale, x_q -> x_q / scale.

    Holds a **reference** (not a copy) to an externally-owned scale tensor.
    This allows the tensor to be an ``nn.Parameter`` updated by an optimizer,
    or a buffer written by calibration — the transform automatically sees
    the latest values on the next forward pass.

    When *pot* is True, the scale is projected to the nearest power-of-two
    before use, making multiplication a bit-shift (hardware-friendly).

    invertible = True.
    """

    invertible = True

    def __init__(self, scale: Tensor, pot: bool = False):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"scale must be a torch.Tensor, got {type(scale).__name__}")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "pot", pot)

    def _effective_scale(self, x: Tensor) -> Tensor:
        """Return the scale tensor broadcastable to *x*, optionally PoT-projected."""
        s = self.scale
        if self.pot:
            s = _pot_scale(s)
        shape = s.shape + (1,) * (x.ndim - s.ndim)
        return s.view(shape)

    def forward(self, x: Tensor) -> Tensor:
        return x * self._effective_scale(x)

    def inverse(self, x_q: Tensor) -> Tensor:
        return x_q / self._effective_scale(x_q)

    def __eq__(self, other):
        if not isinstance(other, PreScaleTransform):
            return NotImplemented
        return self.scale is other.scale and self.pot == other.pot

    def __hash__(self):
        return hash(("PreScaleTransform", id(self.scale), self.pot))

    def __repr__(self):
        pot_str = ", pot=True" if self.pot else ""
        return f"PreScaleTransform(shape={tuple(self.scale.shape)}{pot_str})"
