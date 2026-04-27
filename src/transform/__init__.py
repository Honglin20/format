"""
Transform package for pre/post-quantization transforms.

Provides:
- ``HadamardTransform`` — orthogonal rotation via Fast Walsh-Hadamard Transform.
- ``hadamard()`` — the raw FWHT function.
- ``TransformBase`` / ``IdentityTransform`` — re-exported from ``src.scheme.transform``
  for convenience.
"""
from .hadamard import HadamardTransform, hadamard
from src.scheme.transform import TransformBase, IdentityTransform

__all__ = [
    "HadamardTransform",
    "hadamard",
    "TransformBase",
    "IdentityTransform",
]
