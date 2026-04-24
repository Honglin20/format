"""
QuantScheme: format + granularity + round_mode — the tensor-level quantization strategy.
"""
from dataclasses import dataclass, field

from .granularity import GranularitySpec

_VALID_ROUND_MODES = {"nearest", "floor", "even", "dither"}


@dataclass(frozen=True)
class QuantScheme:
    """Tensor-level quantization scheme.

    Combines:
    - format: element format string (e.g., "fp8_e4m3", "int8")
    - granularity: GranularitySpec (mode + block_size + channel_axis)
    - round_mode: rounding mode ("nearest", "floor", "even", "dither")
    """
    format: str
    granularity: GranularitySpec = field(default_factory=GranularitySpec.per_tensor)
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

    @staticmethod
    def per_tensor(format: str, round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format,
                          granularity=GranularitySpec.per_tensor(),
                          round_mode=round_mode)

    @staticmethod
    def per_channel(format: str, axis: int = 0, round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format,
                          granularity=GranularitySpec.per_channel(axis),
                          round_mode=round_mode)

    @staticmethod
    def mxfp(format: str, block_size: int = 32, round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=format,
                          granularity=GranularitySpec.per_block(block_size),
                          round_mode=round_mode)

    @property
    def is_mx(self) -> bool:
        return self.granularity.is_mx

    @property
    def block_size(self) -> int:
        return self.granularity.block_size
