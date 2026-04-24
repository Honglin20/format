"""
QuantScheme: format + granularity + transform + round_mode — the three-axis
tensor-level quantization strategy.
"""
from dataclasses import dataclass, field
from typing import Union

from ..formats.base import FormatBase
from .granularity import GranularitySpec
from .transform import TransformBase, IdentityTransform

_VALID_ROUND_MODES = {"nearest", "floor", "even", "dither"}


def _resolve_format(fmt: Union[str, FormatBase]) -> FormatBase:
    """Convert string to FormatBase via registry, or pass through FormatBase."""
    if isinstance(fmt, FormatBase):
        return fmt
    if isinstance(fmt, str):
        from ..formats.registry import get_format
        return get_format(fmt)
    raise TypeError(f"format must be str or FormatBase, got {type(fmt).__name__}")


@dataclass(frozen=True)
class QuantScheme:
    """Tensor-level quantization scheme (three-axis design).

    Combines:
    - format: FormatBase instance (e.g., FP8E4M3Format(), Int8Format())
    - granularity: GranularitySpec (mode + block_size + channel_axis)
    - transform: TransformBase (default: IdentityTransform)
    - round_mode: rounding mode ("nearest", "floor", "even", "dither")

    Default: INT8, per_tensor, IdentityTransform, round_mode="nearest".
    """
    format: FormatBase = field(default_factory=lambda: _resolve_format("int8"))
    granularity: GranularitySpec = field(default_factory=GranularitySpec.per_tensor)
    transform: TransformBase = field(default_factory=IdentityTransform)
    round_mode: str = "nearest"

    def __post_init__(self):
        # Coerce string format to FormatBase (supports factory methods accepting str)
        if isinstance(self.format, str):
            # frozen dataclass standard pattern: use object.__setattr__ inside __post_init__
            object.__setattr__(self, "format", _resolve_format(self.format))

        if not isinstance(self.format, FormatBase):
            raise TypeError(
                f"format must be FormatBase, got {type(self.format).__name__}"
            )

        if not isinstance(self.transform, TransformBase):
            raise TypeError(
                f"transform must be TransformBase, got {type(self.transform).__name__}"
            )

        # Validate round_mode
        if self.round_mode not in _VALID_ROUND_MODES:
            raise ValueError(
                f"Invalid round_mode {self.round_mode!r}. Must be one of {_VALID_ROUND_MODES}"
            )

    @staticmethod
    def per_tensor(format: Union[str, FormatBase], round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=_resolve_format(format),
                          granularity=GranularitySpec.per_tensor(),
                          round_mode=round_mode)

    @staticmethod
    def per_channel(format: Union[str, FormatBase], axis: int = 0,
                    round_mode: str = "nearest") -> "QuantScheme":
        if isinstance(axis, str):
            raise TypeError(
                f"axis must be int, not str. "
                f"Did you mean: per_channel({format!r}, round_mode={axis!r})? "
                f"The API changed: axis was inserted before round_mode."
            )
        return QuantScheme(format=_resolve_format(format),
                          granularity=GranularitySpec.per_channel(axis),
                          round_mode=round_mode)

    @staticmethod
    def mxfp(format: Union[str, FormatBase], block_size: int = 32,
             round_mode: str = "nearest") -> "QuantScheme":
        return QuantScheme(format=_resolve_format(format),
                          granularity=GranularitySpec.per_block(block_size),
                          round_mode=round_mode)

    @property
    def is_mx(self) -> bool:
        return self.granularity.is_mx

    @property
    def block_size(self) -> int:
        return self.granularity.block_size

    @property
    def format_name(self) -> str:
        return self.format.name
