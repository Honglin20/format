"""
QuantScheme: format + granularity + round — the tensor-level quantization strategy.
"""
from dataclasses import dataclass
from typing import Optional

from .granularity import Granularity


@dataclass(frozen=True)
class QuantScheme:
    """Tensor-level quantization scheme.

    Combines:
    - format: element format string (e.g., "fp8_e4m3", "int8")
    - granularity: how the scale is shared (per_tensor, per_channel, per_block)
    - block_size: block size for per_block granularity (0 otherwise)
    - round: rounding mode ("nearest", "floor", "even")
    """
    format: str
    granularity: Granularity = Granularity.PER_BLOCK
    block_size: int = 0
    round: str = "nearest"

    @staticmethod
    def per_tensor(format: str, round: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format, granularity=Granularity.PER_TENSOR,
                          block_size=0, round=round)

    @staticmethod
    def per_channel(format: str, round: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format, granularity=Granularity.PER_CHANNEL,
                          block_size=0, round=round)

    @staticmethod
    def mxfp(format: str, block_size: int = 32, round: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format, granularity=Granularity.PER_BLOCK,
                          block_size=block_size, round=round)

    @property
    def is_mx(self) -> bool:
        """True if this is a block-level (MX) quantization scheme."""
        return self.granularity == Granularity.PER_BLOCK
