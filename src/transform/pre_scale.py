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

    *channel_axis* controls which dimension the scale broadcasts along.
    The scale is reshaped to ``[1, ..., C, ..., 1]`` with C at channel_axis
    before element-wise multiply/divide.

    invertible = True.
    """

    invertible = True

    def __init__(self, scale: Tensor, pot: bool = False, channel_axis: int = -1):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"scale must be a torch.Tensor, got {type(scale).__name__}")
        if not isinstance(pot, bool):
            raise TypeError(f"pot must be a bool, got {type(pot).__name__}")
        if not isinstance(channel_axis, int):
            raise TypeError(
                f"channel_axis must be an int, got {type(channel_axis).__name__}"
            )
        self.scale = scale
        self.pot = pot
        self.channel_axis = channel_axis

    def _effective_scale(self, x: Tensor) -> Tensor:
        """Return the scale tensor broadcastable to *x*, optionally PoT-projected."""
        ax = self.channel_axis
        if ax < 0:
            ax = x.ndim + ax
        if not (0 <= ax < x.ndim):
            raise ValueError(
                f"channel_axis={self.channel_axis} is out of bounds for "
                f"tensor with ndim={x.ndim}"
            )
        s = self.scale
        if s.device != x.device:
            s = s.to(device=x.device)
        if self.pot:
            s = _pot_scale(s)
        shape = [1] * x.ndim
        shape[ax] = -1
        return s.view(*shape)

    def forward(self, x: Tensor) -> Tensor:
        return x * self._effective_scale(x)

    def inverse(self, x_q: Tensor) -> Tensor:
        return x_q / self._effective_scale(x_q)

    def __eq__(self, other):
        if not isinstance(other, PreScaleTransform):
            return NotImplemented
        return (self.scale is other.scale
                and self.pot == other.pot
                and self.channel_axis == other.channel_axis)

    def __hash__(self):
        return hash(("PreScaleTransform", id(self.scale), self.pot, self.channel_axis))

    def __repr__(self):
        pot_str = ", pot=True" if self.pot else ""
        ax_str = f", channel_axis={self.channel_axis}" if self.channel_axis != -1 else ""
        return f"PreScaleTransform(shape={tuple(self.scale.shape)}{pot_str}{ax_str})"
