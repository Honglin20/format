"""
GranularitySpec: how the shared scale is computed during quantization.

Replaces the old Granularity enum with a composable dataclass that carries
mode + parameters (block_size, channel_axis, block_axis) together.
"""
from dataclasses import dataclass
from enum import Enum


class GranularityMode(Enum):
    """Quantization granularity mode."""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"
    DYNAMIC_GROUP = "dynamic_group"


@dataclass(frozen=True)
class GranularitySpec:
    """Quantization granularity specification.

    Combines mode with its parameters so that block_size / channel_axis /
    block_axis are always colocated with the mode that gives them meaning.

    channel_axis supports NumPy-style negative indexing (e.g. -1 = last dimension).
    Since GranularitySpec does not hold a tensor shape, out-of-bounds checking is
    deferred to FormatBase.quantize() at quantization time.

    block_axis: axis along which to split into blocks (PER_BLOCK only).
    Default: -1 (last dimension). Supports negative indexing.
    Out-of-bounds checking is deferred to quantization time (same as channel_axis).
    """
    mode: GranularityMode = GranularityMode.PER_TENSOR
    block_size: int = 0
    channel_axis: int = 0
    block_axis: int = -1  # Default: last dimension (MX forward convention)

    def __post_init__(self):
        if self.mode == GranularityMode.PER_BLOCK and self.block_size <= 0:
            raise ValueError(
                f"PER_BLOCK requires block_size > 0, got {self.block_size}"
            )
        if self.mode == GranularityMode.PER_TENSOR and self.block_size != 0:
            raise ValueError(
                f"PER_TENSOR requires block_size=0, got {self.block_size}"
            )
        if self.mode == GranularityMode.PER_CHANNEL and self.block_size != 0:
            raise ValueError(
                f"PER_CHANNEL requires block_size=0, got {self.block_size}"
            )
        if self.mode not in (GranularityMode.PER_CHANNEL, GranularityMode.DYNAMIC_GROUP) and self.channel_axis != 0:
            raise ValueError(
                f"{self.mode.name} requires channel_axis=0, got {self.channel_axis}"
            )
        if self.mode == GranularityMode.DYNAMIC_GROUP and self.block_size != 0:
            raise ValueError(
                f"DYNAMIC_GROUP requires block_size=0, got {self.block_size}"
            )
        if self.mode not in (GranularityMode.PER_BLOCK,) and self.block_axis != -1:
            raise ValueError(
                f"{self.mode.name} requires block_axis=-1, got {self.block_axis}"
            )

    @staticmethod
    def per_tensor() -> "GranularitySpec":
        return GranularitySpec(mode=GranularityMode.PER_TENSOR)

    @staticmethod
    def per_channel(axis: int = 0) -> "GranularitySpec":
        return GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=axis)

    @staticmethod
    def per_block(size: int, axis: int = -1) -> "GranularitySpec":
        return GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=size, block_axis=axis)

    @property
    def is_mx(self) -> bool:
        return self.mode == GranularityMode.PER_BLOCK
