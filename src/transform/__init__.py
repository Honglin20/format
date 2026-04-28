"""
Transform package for pre/post-quantization transforms.

Provides:
- ``HadamardTransform`` — orthogonal rotation via Fast Walsh-Hadamard Transform.
- ``hadamard()`` — the raw FWHT function.
- ``SmoothQuantTransform`` — per-channel activation smoothing (SmoothQuant).
- ``compute_smoothquant_scale()`` — compute SmoothQuant scale factors.
- ``TransformBase`` / ``IdentityTransform`` — re-exported from ``src.scheme.transform``
  for convenience.
"""
from .hadamard import HadamardTransform, hadamard
from .pre_scale import PreScaleTransform
from .smooth_quant import SmoothQuantTransform, compute_smoothquant_scale
from src.scheme.transform import TransformBase, IdentityTransform

__all__ = [
    "HadamardTransform",
    "hadamard",
    "PreScaleTransform",
    "SmoothQuantTransform",
    "compute_smoothquant_scale",
    "TransformBase",
    "IdentityTransform",
]
