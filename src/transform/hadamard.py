"""
HadamardTransform: pre-quantization orthogonal rotation via Fast Walsh-Hadamard Transform.

The Walsh-Hadamard transform is a self-inverse orthogonal transform that can be
computed in O(n log n) using the butterfly algorithm. It rotates the tensor along
the last dimension, spreading information across elements to reduce quantization
error when followed by element-wise quantization.

Normalization: 1/sqrt(d) ensures the transform is orthogonal (self-inverse).
For non-power-of-2 dimensions, the tensor is silently padded to the next power of 2.
"""
import math

import torch
from torch import Tensor

from ..scheme.transform import TransformBase


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to ``n``.

    Args:
        n: A non-negative integer.

    Returns:
        The smallest power of 2 >= n. 0 if n == 0.
    """
    if n <= 0:
        return 0
    return 2 ** ((n - 1).bit_length())


# ---------------------------------------------------------------------------
# FWHT implementation
# ---------------------------------------------------------------------------

def hadamard(x: Tensor) -> Tensor:
    """Fast Walsh-Hadamard Transform along the last dimension.

    Uses the in-place iterative butterfly algorithm (O(n log n) where n is the
    size of the last dimension). The transform is normalized by 1/sqrt(d) so
    that ``hadamard(hadamard(x)) == x`` (self-inverse, up to floating-point
    precision).

    For tensors whose last dimension is not a power of 2, the implementation
    silently pads to the next power of 2, applies the transform, then truncates
    back to the original size. In this case the self-inverse property holds
    only approximately.

    Args:
        x: Input tensor of any shape. The transform is applied along the last
            dimension.

    Returns:
        Transformed tensor with the same shape as ``x``.
    """
    # Clone to avoid modifying the input
    x = x.clone()

    d = x.shape[-1]
    n = _next_power_of_2(d)

    # Pad to next power of 2 along last dim if needed
    if n != d:
        x = torch.nn.functional.pad(x, (0, n - d))

    # Store shape for later reshape
    orig_shape = x.shape

    # Flatten all dims except the last for vectorized butterfly processing
    x_2d = x.reshape(-1, n)

    # FWHT: in-place butterfly
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = x_2d[:, i : i + h]          # view into left  half of pair
            b = x_2d[:, i + h : i + 2 * h]  # view into right half of pair
            # Compute both results BEFORE any writes — `a` and `b` are views into
            # x_2d, so modifying x_2d corrupts the view values.
            sum_ab = a + b
            diff_ab = a - b
            x_2d[:, i : i + h] = sum_ab
            x_2d[:, i + h : i + 2 * h] = diff_ab
        h *= 2

    # Restore original shape
    x = x_2d.reshape(orig_shape)

    # Orthogonal normalization
    x = x / math.sqrt(n)

    # Truncate back to original size if padded
    if n != d:
        x = x[..., :d]

    return x


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------

class HadamardTransform(TransformBase):
    """Pre-quantization Hadamard rotation.

    Applies a Fast Walsh-Hadamard Transform along the last dimension before
    quantization, and its inverse (same operation, due to orthonormal
    normalization) after quantization. This spreads quantization error across
    all elements, which can improve accuracy for certain data distributions.

    The transform is self-inverse and requires no state beyond the type itself.
    It is hashable and usable as a drop-in transform in ``QuantScheme``.
    """

    invertible = True

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Hadamard transform before quantization.

        Args:
            x: Input tensor.

        Returns:
            Hadamard-transformed tensor (same shape as input).
        """
        return hadamard(x)

    def inverse(self, x_q: Tensor) -> Tensor:
        """Apply the inverse Hadamard transform after quantization.

        Since the transform is self-inverse (orthogonal with 1/sqrt(d)
        normalization), this is the same as :meth:`forward`.

        Args:
            x_q: Quantized tensor.

        Returns:
            Inverse-transformed tensor (same shape as input).
        """
        return hadamard(x_q)

    def __eq__(self, other) -> bool:
        """Two HadamardTransform instances are always equal (stateless)."""
        return isinstance(other, HadamardTransform)

    def __hash__(self) -> int:
        """Hash based on the type name (all instances are equal)."""
        return hash("HadamardTransform")
