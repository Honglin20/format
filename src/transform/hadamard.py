"""
HadamardTransform: pre-quantization orthogonal rotation via Fast Walsh-Hadamard Transform.

The Walsh-Hadamard transform is a self-inverse orthogonal transform that can be
computed in O(n log n) using the butterfly algorithm. It rotates the tensor along
the last dimension, spreading information across elements to reduce quantization
error when followed by element-wise quantization.

Normalization: 1/sqrt(d) ensures each power-of-2 block is orthogonal (self-inverse).
For non-power-of-2 dimensions, the tensor is decomposed into power-of-2 chunks and
each chunk is transformed independently.  This preserves the self-inverse property
for arbitrary dimensions.
"""
import math

import torch
from torch import Tensor

from ..scheme.transform import TransformBase


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _largest_power_of_2_le(n: int) -> int:
    """Return the largest power of 2 less than or equal to ``n``.

    Args:
        n: A positive integer.

    Returns:
        The largest power of 2 <= n.
    """
    return 1 << (n.bit_length() - 1)


def _decompose_pow2(d: int):
    """Yield power-of-2 chunk sizes that sum to ``d``, in descending order."""
    remaining = d
    while remaining > 0:
        sz = _largest_power_of_2_le(remaining)
        yield sz
        remaining -= sz


# ---------------------------------------------------------------------------
# FWHT implementation (power-of-2 only, no padding needed)
# ---------------------------------------------------------------------------

def _hadamard_pow2(x_2d: Tensor, n: int) -> Tensor:
    """In-place FWHT on the last dimension of a 2D tensor.

    ``x_2d`` must have shape ``(M, n)`` where ``n`` is a power of 2.
    ``x_2d`` is a view/slice of the original tensor; modifications are in-place.

    Normalizes by ``1/sqrt(n)`` so the transform is self-inverse.
    """
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = x_2d[:, i: i + h]
            b = x_2d[:, i + h: i + 2 * h]
            sum_ab = a + b
            diff_ab = a - b
            x_2d[:, i: i + h] = sum_ab
            x_2d[:, i + h: i + 2 * h] = diff_ab
        h *= 2
    x_2d.div_(math.sqrt(n))


def hadamard(x: Tensor) -> Tensor:
    """Fast Walsh-Hadamard Transform along the last dimension.

    For power-of-2 dimensions the classical butterfly algorithm is used.
    For non-power-of-2 dimensions the tensor is decomposed into power-of-2
    chunks along the last dimension; each chunk is independently transformed.
    This preserves the self-inverse property for arbitrary dimensions.

    The transform is normalized so that ``hadamard(hadamard(x)) == x``
    for all dimension sizes.

    Args:
        x: Input tensor of any shape. The transform is applied along the last
            dimension.

    Returns:
        Transformed tensor with the same shape as ``x``.
    """
    d = x.shape[-1]

    # Fast path: power-of-2 dimension
    if d & (d - 1) == 0:
        x = x.clone()
        orig_shape = x.shape
        x_2d = x.reshape(-1, d)
        _hadamard_pow2(x_2d, d)
        return x_2d.reshape(orig_shape)

    # Non-power-of-2: decompose into power-of-2 chunks
    x = x.clone()
    orig_shape = x.shape
    x_2d = x.reshape(-1, d)

    offset = 0
    for chunk_size in _decompose_pow2(d):
        chunk = x_2d[:, offset: offset + chunk_size]
        _hadamard_pow2(chunk, chunk_size)
        offset += chunk_size

    return x_2d.reshape(orig_shape)


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
