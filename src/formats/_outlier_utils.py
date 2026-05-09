"""Per-bank outlier/normal split quantization — internal helper.

Used by FormatBase._quantize_per_block() when granularity.outlier_ratio > 0.
"""
import torch

from ._block_utils import (
    _reshape_to_blocks,
    _undo_reshape_to_blocks,
    _shared_exponents,
)


def _quantize_outlier_bank(format_self, x, granularity, round_mode,
                          scale_storage="pot"):
    """PER_BLOCK quantization with per-bank outlier/normal split.

    Within each bank, the top-k elements by magnitude (outliers) and the
    remaining elements (normals) each get their own MX-style shared exponent.
    Both groups are quantized with the SAME format — only the scale differs.

    Degeneracy: when k >= block_size, all elements are in one group,
    equivalent to standard PER_BLOCK.  This case returns early.
    """
    block_size = granularity.block_size
    outlier_ratio = granularity.outlier_ratio
    axes = [granularity.block_axis]

    # Normalize axes to non-negative
    axes = [a + x.ndim if a < 0 else a for a in axes]

    # Tile into hardware-vector-sized blocks (same reshape as PER_BLOCK)
    A, axes, orig_shape, padded_shape = _reshape_to_blocks(x, axes, block_size)

    # Block dimension after reshape
    block_dim = axes[-1] + 1

    # Number of outliers per bank
    k = max(1, int(block_size * outlier_ratio))

    if k >= block_size:
        # All elements are outliers — degenerate to single-group PER_BLOCK.
        # Don't call _quantize_per_block() (would re-dispatch here).
        # Single shared exponent for the whole block, same as normal path.
        shared_exp = _shared_exponents(A, method="max", axes=[block_dim], ebits=0)
        shared_exp = shared_exp - format_self.emax
        scale_emax = 2 ** (8 - 1) - 1
        shared_exp[shared_exp > scale_emax] = float("NaN")
        shared_exp[shared_exp < -scale_emax] = -scale_emax
        A = A / (2 ** shared_exp)
        A = format_self.quantize_elemwise(
            A, round_mode=round_mode, allow_denorm=True, saturate_normals=True)
        A = A * (2 ** shared_exp)
        return _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    # Select top-k by magnitude in each bank
    _, top_indices = torch.topk(torch.abs(A), k, dim=block_dim)
    mask = torch.zeros_like(A, dtype=torch.bool)
    mask.scatter_(block_dim, top_indices, True)

    # Compute shared exponents per group.
    # Mask the OTHER group to zero so it doesn't affect max reduction.
    exp_o = _shared_exponents(
        A * mask.float(), method="max", axes=[block_dim], ebits=0)
    exp_n = _shared_exponents(
        A * (~mask).float(), method="max", axes=[block_dim], ebits=0)

    # Offset by format's max representable exponent
    exp_o = exp_o - format_self.emax
    exp_n = exp_n - format_self.emax

    # Clamp shared exponents to int8 range (same as normal PER_BLOCK)
    scale_emax = 2 ** (8 - 1) - 1
    exp_o[exp_o > scale_emax] = float("NaN")
    exp_o[exp_o < -scale_emax] = -scale_emax
    exp_n[exp_n > scale_emax] = float("NaN")
    exp_n[exp_n < -scale_emax] = -scale_emax

    # Normalize → elemwise quantize → rescale for each group
    A_o = A / (2 ** exp_o)
    A_o = format_self.quantize_elemwise(
        A_o, round_mode=round_mode, allow_denorm=True, saturate_normals=True)
    A_o = A_o * (2 ** exp_o)

    A_n = A / (2 ** exp_n)
    A_n = format_self.quantize_elemwise(
        A_n, round_mode=round_mode, allow_denorm=True, saturate_normals=True)
    A_n = A_n * (2 ** exp_n)

    # Merge the two groups
    A_q = torch.where(mask, A_o, A_n)

    # Undo block tiling
    A_q = _undo_reshape_to_blocks(A_q, padded_shape, orig_shape, axes)
    return A_q
