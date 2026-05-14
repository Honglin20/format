"""compute_sparse_mask: per-sample top-k + cross-sample voting.

Independent function that computes a fixed boolean mask from calibration
data. The mask identifies which element positions should be quantized as
"outliers" with their own scale, based on cross-sample voting.
"""
import torch
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.formats.base import FormatBase


def compute_sparse_mask(
    x_calib: torch.Tensor,
    fmt: FormatBase,
    granularity: GranularitySpec,
    outlier_ratio: float,
) -> torch.Tensor:
    """Compute a fixed sparse mask from calibration data.

    Step 1 — Per-sample mask: within each granularity group, top-k
    elements by magnitude → mask_s (same shape as a single sample).
    Step 2 — Cross-sample voting: average all mask_s → mask_avg.
    Step 3 — Final mask: global top-k of mask_avg.

    Args:
        x_calib: Calibration samples stacked along dim 0 (S, D1, D2, ...).
        fmt: Target format (reserved for future format-specific logic).
        granularity: Granularity spec determining group boundaries for
                     per-sample top-k.
        outlier_ratio: Fraction of elements to mark as outliers.

    Returns:
        Boolean mask with shape (D1, D2, ...), True for outlier positions.
    """
    if x_calib.dim() < 2:
        raise ValueError(
            f"x_calib must have batch dim + tensor dims, got shape {x_calib.shape}"
        )
    if not (0.0 < outlier_ratio < 1.0):
        raise ValueError(
            f"outlier_ratio must be in (0, 1), got {outlier_ratio}"
        )

    S = x_calib.shape[0]
    sample_shape = x_calib.shape[1:]
    N_sample = _numel(sample_shape)
    k_total = max(1, int(N_sample * outlier_ratio))

    mask_accum = torch.zeros(sample_shape, dtype=torch.float32, device=x_calib.device)

    for s in range(S):
        x_s = x_calib[s]
        mask_s = _per_sample_mask(x_s, granularity, outlier_ratio)
        mask_accum += mask_s.float()

    mask_avg = mask_accum / S
    mask_flat = mask_avg.flatten()

    _, top_indices = torch.topk(mask_flat, k_total)
    final_flat = torch.zeros(N_sample, dtype=torch.bool, device=x_calib.device)
    final_flat.scatter_(0, top_indices, True)
    return final_flat.reshape(sample_shape)


# ---------------------------------------------------------------------------
# Per-sample mask helpers
# ---------------------------------------------------------------------------


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _per_sample_mask(x: torch.Tensor, granularity: GranularitySpec,
                     outlier_ratio: float) -> torch.Tensor:
    """Compute per-element outlier mask for a single sample.

    Within each granularity group, selects top-k elements by magnitude.
    """
    mode = granularity.mode
    if mode == GranularityMode.PER_TENSOR:
        return _mask_per_tensor(x, outlier_ratio)
    elif mode == GranularityMode.PER_CHANNEL:
        return _mask_per_channel(x, granularity, outlier_ratio)
    elif mode == GranularityMode.PER_BLOCK:
        return _mask_per_block(x, granularity, outlier_ratio)
    elif mode == GranularityMode.BANK:
        return _mask_per_bank(x, granularity, outlier_ratio)
    else:
        raise ValueError(f"Unsupported granularity mode: {mode}")


def _mask_per_tensor(x: torch.Tensor, outlier_ratio: float) -> torch.Tensor:
    N = x.numel()
    k = max(1, int(N * outlier_ratio))
    if k >= N:
        return torch.ones_like(x, dtype=torch.bool)
    _, top_indices = torch.topk(torch.abs(x).flatten(), k)
    mask = torch.zeros(N, dtype=torch.bool, device=x.device)
    mask.scatter_(0, top_indices, True)
    return mask.reshape(x.shape)


def _mask_per_channel(x: torch.Tensor, granularity: GranularitySpec,
                      outlier_ratio: float) -> torch.Tensor:
    axis = granularity.channel_axis
    if axis < 0:
        axis = x.ndim + axis
    C = x.shape[axis]
    x_t = x.transpose(0, axis)
    N_per_channel = x_t[0].numel()
    k = max(1, int(N_per_channel * outlier_ratio))
    if k >= N_per_channel:
        # Every element in channel is "outlier" — degenerate to all-True.
        return torch.ones_like(x, dtype=torch.bool)
    shape_t = x_t.shape  # (C, ...)
    x_flat = x_t.reshape(C, N_per_channel)
    _, top_indices = torch.topk(torch.abs(x_flat), k, dim=1)
    mask_flat = torch.zeros(C, N_per_channel, dtype=torch.bool, device=x.device)
    mask_flat.scatter_(1, top_indices, True)
    mask_t = mask_flat.reshape(shape_t)
    return mask_t.transpose(0, axis).reshape(x.shape)


def _mask_per_block(x: torch.Tensor, granularity: GranularitySpec,
                    outlier_ratio: float) -> torch.Tensor:
    block_size = granularity.block_size
    axes = [granularity.block_axis]
    axes = [a + x.ndim if a < 0 else a for a in axes]
    from src.formats._block_utils import _reshape_to_blocks, _undo_reshape_to_blocks
    A, axes, orig_shape, padded_shape = _reshape_to_blocks(x, axes, block_size)
    block_dim = axes[-1] + 1  # inner block dimension
    k = max(1, int(block_size * outlier_ratio))
    if k >= block_size:
        mask = torch.ones_like(A, dtype=torch.bool)
        return _undo_reshape_to_blocks(mask, padded_shape, orig_shape, axes)
    _, top_indices = torch.topk(torch.abs(A), k, dim=block_dim)
    mask = torch.zeros_like(A, dtype=torch.bool)
    mask.scatter_(block_dim, top_indices, True)
    return _undo_reshape_to_blocks(mask, padded_shape, orig_shape, axes)


def _mask_per_bank(x: torch.Tensor, granularity: GranularitySpec,
                   outlier_ratio: float) -> torch.Tensor:
    axis = granularity.bank_axis
    if axis < 0:
        axis = x.ndim + axis
    bank_size = granularity.bank_size
    N_along = x.shape[axis]
    num_banks = N_along // bank_size

    # Reshape: (..., D_a, ...) → (..., num_banks, bank_size, ...)
    new_shape = list(x.shape)
    new_shape[axis] = num_banks
    new_shape.insert(axis + 1, bank_size)
    x_r = x.reshape(new_shape)
    # x_r shape: (..., num_banks, bank_size, ...)
    # bank dim at position `axis`, inner bank at `axis+1`

    # Move bank dim to front for per-group processing
    # Transpose to (num_banks, ..., bank_size, ...)
    ndim_r = x_r.ndim
    perm = list(range(ndim_r))
    perm.pop(axis)
    perm = [axis] + perm
    x_b = x_r.permute(perm)
    # x_b shape: (num_banks, ..., bank_size, ...)

    group_size = x_b[0].numel()
    k = max(1, int(group_size * outlier_ratio))
    num_banks_actual = x_b.shape[0]

    if k >= group_size:
        mask_b = torch.ones_like(x_b, dtype=torch.bool)
    else:
        x_flat = x_b.reshape(num_banks_actual, group_size)
        _, top_indices = torch.topk(torch.abs(x_flat), k, dim=1)
        mask_flat = torch.zeros(num_banks_actual, group_size, dtype=torch.bool,
                                device=x.device)
        mask_flat.scatter_(1, top_indices, True)
        mask_b = mask_flat.reshape(x_b.shape)

    # Undo permutation: back to (..., num_banks, bank_size, ...)
    inv_perm = [0] * ndim_r
    for i, p in enumerate(perm):
        inv_perm[p] = i
    mask_r = mask_b.permute(inv_perm)

    return mask_r.reshape(x.shape)
