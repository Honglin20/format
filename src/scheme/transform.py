"""
TransformBase: optional pre/post-quantization transform.

Part of the three-axis QuantScheme design (format + granularity + transform).
Transforms allow orthogonal rotations, absmax rescaling, etc. that reduce
quantization error without changing the format or granularity logic.
"""
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TransformBase(ABC):
    """Abstract base for quantization transforms.

    Subclasses must implement forward(). If invertible, also override inverse()
    and set invertible = True.
    """

    invertible: bool = False

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Pre-quantization transform."""

    def inverse(self, x_q: Tensor) -> Tensor:
        """Post-quantization inverse. Default: identity (non-invertible transforms)."""
        return x_q


class IdentityTransform(TransformBase):
    """No-op transform: forward and inverse are both identity."""

    invertible = True

    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, x_q: Tensor) -> Tensor:
        return x_q

    def __eq__(self, other):
        return isinstance(other, IdentityTransform)

    def __hash__(self):
        return hash("IdentityTransform")
