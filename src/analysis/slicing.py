"""
iter_slices: granularity-aware tensor slicing for analysis.

Single entry point — new granularity modes only need a branch here.
"""
from typing import Iterator, Optional, Tuple

from torch import Tensor

from src.scheme.granularity import GranularityMode, GranularitySpec

SliceKey = Tuple[str, ...]


def iter_slices(
    fp32: Tensor,
    quant: Tensor,
    granularity: GranularitySpec,
    group_map: Optional[Tensor] = None,
) -> Iterator[Tuple[SliceKey, Tensor, Tensor]]:
    """Yield (key, fp32_slice, quant_slice) for each granularity unit.

    Args:
        fp32: Pre-quantization tensor.
        quant: Post-quantization tensor.
        granularity: GranularitySpec controlling how to slice.
        group_map: Optional group_id tensor for dynamic-grouping quantization.

    Yields:
        (slice_key, fp32_slice, quant_slice) tuples.
    """
    mode = granularity.mode

    if mode == GranularityMode.PER_TENSOR:
        yield ("tensor",), fp32, quant

    elif mode == GranularityMode.PER_CHANNEL:
        axis = granularity.channel_axis
        if axis < 0:
            axis = fp32.ndim + axis
        for i in range(fp32.shape[axis]):
            yield ("channel", i), fp32.select(axis, i), quant.select(axis, i)

    elif mode == GranularityMode.PER_BLOCK:
        bs = granularity.block_size
        last_dim = fp32.shape[-1]
        n_blocks = (last_dim + bs - 1) // bs
        for b in range(n_blocks):
            sl = slice(b * bs, min((b + 1) * bs, last_dim))
            yield ("block", b), fp32[..., sl], quant[..., sl]

    elif mode == GranularityMode.DYNAMIC_GROUP:
        if group_map is None:
            raise ValueError(
                "DYNAMIC_GROUP granularity requires group_map to be set. "
                "Make sure the format's quantize() returns the group_map tensor."
            )
        for gid in group_map.unique().tolist():
            mask = (group_map == gid)
            yield ("group", gid), fp32[mask], quant[mask]

    else:
        raise ValueError(f"Unknown granularity mode: {mode}")
