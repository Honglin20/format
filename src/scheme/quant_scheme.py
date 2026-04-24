"""
QuantScheme: format + granularity + round_mode — the tensor-level quantization strategy.
"""
from dataclasses import dataclass

from .granularity import Granularity

# Old code's RoundingMode enum had nearest/floor/even. "dither" was implemented
# in old _round_mantissa() (mx/elemwise_ops.py) but not in the enum.
# Including it here as valid is an intentional extension of the old public API
# to match the actual runtime behavior.
_VALID_ROUND_MODES = {"nearest", "floor", "even", "dither"}


@dataclass(frozen=True)
class QuantScheme:
    """Tensor-level quantization scheme.

    Combines:
    - format: element format string (e.g., "fp8_e4m3", "int8")
    - granularity: how the scale is shared (per_tensor, per_channel, per_block)
    - block_size: block size for per_block granularity (0 otherwise)
    - round_mode: rounding mode ("nearest", "floor", "even", "dither")
    """
    format: str
    granularity: Granularity = Granularity.PER_BLOCK
    block_size: int = 0
    round_mode: str = "nearest"

    def __post_init__(self):
        # Validate format string resolves in registry
        from ..formats.registry import get_format
        get_format(self.format)  # raises ValueError if unknown

        # Validate round_mode
        if self.round_mode not in _VALID_ROUND_MODES:
            raise ValueError(
                f"Invalid round_mode {self.round_mode!r}. Must be one of {_VALID_ROUND_MODES}"
            )

        # Validate block_size consistency with granularity
        if self.granularity == Granularity.PER_BLOCK and self.block_size <= 0:
            raise ValueError(
                f"PER_BLOCK granularity requires block_size > 0, got {self.block_size}"
            )
        if self.granularity in (Granularity.PER_TENSOR, Granularity.PER_CHANNEL):
            if self.block_size != 0:
                raise ValueError(
                    f"{self.granularity} granularity requires block_size=0, got {self.block_size}"
                )

    @staticmethod
    def per_tensor(format: str, round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format, granularity=Granularity.PER_TENSOR,
                          block_size=0, round_mode=round_mode)

    @staticmethod
    def per_channel(format: str, round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format, granularity=Granularity.PER_CHANNEL,
                          block_size=0, round_mode=round_mode)

    @staticmethod
    def mxfp(format: str, block_size: int = 32, round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format, granularity=Granularity.PER_BLOCK,
                          block_size=block_size, round_mode=round_mode)

    @property
    def is_mx(self) -> bool:
        """True if this is a block-level (MX) quantization scheme."""
        return self.granularity == Granularity.PER_BLOCK
