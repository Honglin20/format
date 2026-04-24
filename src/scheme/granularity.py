"""
Granularity enum for quantization scaling.
"""
from enum import Enum


class Granularity(Enum):
    """Quantization granularity: how the shared scale is computed."""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"
