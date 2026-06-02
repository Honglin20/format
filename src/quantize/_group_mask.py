"""compute_group_mask: per-group amax → cross-sample max → top-k groups.

Independent function that computes a fixed per-group boolean mask from
calibration data.  The mask identifies which granularity groups (channels,
blocks, banks) should be quantized with the high-precision ``group_format``.

Contrast with ``_sparse_mask.py`` (ADR-012): that mask is per-element;
this mask is per-granularity-group — one bool per channel/block/bank.
"""
import torch
from src.scheme.granularity import GranularityMode, GranularitySpec


def compute_group_mask(
    x_calib: torch.Tensor,
    granularity: GranularitySpec,
    group_ratio: float,
) -> torch.Tensor:
    """Compute a fixed per-group boolean mask from calibration data.

    Step 1 — Per-sample per-group amax: for each calibration sample,
             compute the max absolute value within each granularity group.
    Step 2 — Cross-sample aggregation: element-wise max over all samples.
    Step 3 — Top-k groups: select the k = group_ratio * G groups with
             highest aggregated amax.

    Args:
        x_calib: Calibration samples stacked along dim 0 (S, D1, D2, ...).
        granularity: Granularity spec determining group boundaries.
        group_ratio: Fraction of groups to mark as H (high precision).
                     Must be in (0, 1].

    Returns:
        Boolean per-group mask. Shape depends on mode:
        - PER_TENSOR:  ()  scalar True
        - PER_CHANNEL: (C,)  one bool per channel
        - PER_BLOCK:   block-group shape (e.g. (M, N//blk))
        - BANK:        (num_banks,)  one bool per bank
    """
    if x_calib.dim() < 2:
        raise ValueError(
            f"x_calib must have batch dim + tensor dims, got shape {x_calib.shape}"
        )
    if not (0.0 < group_ratio <= 1.0):
        raise ValueError(
            f"group_ratio must be in (0, 1], got {group_ratio}"
        )

    mode = granularity.mode
    if mode == GranularityMode.PER_TENSOR:
        return _group_mask_per_tensor(x_calib, group_ratio)
    elif mode == GranularityMode.PER_CHANNEL:
        return _group_mask_per_channel(x_calib, granularity, group_ratio)
    elif mode == GranularityMode.PER_BLOCK:
        return _group_mask_per_block(x_calib, granularity, group_ratio)
    elif mode == GranularityMode.BANK:
        return _group_mask_per_bank(x_calib, granularity, group_ratio)
    else:
        raise ValueError(f"Unsupported granularity mode: {mode}")


# ---------------------------------------------------------------------------
# Per-mode helpers
# ---------------------------------------------------------------------------


def _group_mask_per_tensor(x_calib: torch.Tensor, group_ratio: float) -> torch.Tensor:
    """PER_TENSOR: one group → always True."""
    return torch.tensor(True, device=x_calib.device)


def _group_mask_per_channel(x_calib: torch.Tensor, gran: GranularitySpec,
                             group_ratio: float) -> torch.Tensor:
    S = x_calib.shape[0]
    sample_shape = x_calib.shape[1:]
    n = len(sample_shape)

    axis = gran.channel_axis
    if axis < 0:
        axis = n + axis
    if not (0 <= axis < n):
        raise ValueError(
            f"channel_axis={gran.channel_axis} out of range "
            f"for sample tensor with ndim={n}"
        )

    C = sample_shape[axis]
    dims_to_reduce = tuple(i for i in range(n) if i != axis)

    scores = None
    for s in range(S):
        x_s = x_calib[s]
        amax_s = torch.amax(torch.abs(x_s), dim=dims_to_reduce)  # (C,)
        scores = torch.max(scores, amax_s) if scores is not None else amax_s

    return _topk_mask(scores, group_ratio, x_calib.device)


def _group_mask_per_block(x_calib: torch.Tensor, gran: GranularitySpec,
                           group_ratio: float) -> torch.Tensor:
    from src.formats._block_utils import _reshape_to_blocks

    S = x_calib.shape[0]
    block_size = gran.block_size
    axes = [gran.block_axis]

    scores = None
    for s in range(S):
        x_s = x_calib[s]
        A, _axes, _orig_shape, _padded_shape = _reshape_to_blocks(
            x_s, axes, block_size,
        )
        # A exposes blocks; the innermost group dimension is the block_size.
        # Reduce the last dim (block_size elements) to get per-block amax.
        amax_s = torch.amax(torch.abs(A), dim=-1)
        scores = torch.max(scores, amax_s) if scores is not None else amax_s

    return _topk_mask(scores, group_ratio, x_calib.device)


def _group_mask_per_bank(x_calib: torch.Tensor, gran: GranularitySpec,
                          group_ratio: float) -> torch.Tensor:
    S = x_calib.shape[0]
    sample_shape = x_calib.shape[1:]
    n = len(sample_shape)

    axis = gran.bank_axis
    if axis < 0:
        axis = n + axis
    if not (0 <= axis < n):
        raise ValueError(
            f"bank_axis={gran.bank_axis} out of range "
            f"for sample tensor with ndim={n}"
        )

    bank_size = gran.bank_size
    N_along = sample_shape[axis]
    if N_along % bank_size != 0:
        raise ValueError(
            f"Bank axis {axis} size {N_along} not divisible "
            f"by bank_size {bank_size}"
        )

    num_banks = N_along // bank_size

    scores = None
    for s in range(S):
        x_s = x_calib[s]
        # Reshape: split bank_axis into (num_banks, bank_size)
        new_shape = list(x_s.shape)
        new_shape[axis] = num_banks
        new_shape.insert(axis + 1, bank_size)
        x_r = x_s.reshape(new_shape)  # (..., num_banks, bank_size, ...)

        # Reduce all dims except the bank dim (at `axis`)
        dims_to_reduce = tuple(i for i in range(x_r.ndim) if i != axis)
        amax_s = torch.amax(torch.abs(x_r), dim=dims_to_reduce)  # (num_banks,)
        scores = torch.max(scores, amax_s) if scores is not None else amax_s

    return _topk_mask(scores, group_ratio, x_calib.device)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _topk_mask(scores: torch.Tensor, group_ratio: float,
               device: torch.device) -> torch.Tensor:
    """Select top-k groups by score, return boolean mask of same shape."""
    G = scores.numel()
    k = max(1, int(G * group_ratio))
    if k >= G:
        return torch.ones_like(scores, dtype=torch.bool)

    scores_flat = scores.flatten()
    _, top_indices = torch.topk(scores_flat, k)
    mask_flat = torch.zeros(G, dtype=torch.bool, device=device)
    mask_flat.scatter_(0, top_indices, True)
    return mask_flat.reshape(scores.shape)
