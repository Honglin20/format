"""
Equivalence testing utilities for verifying bit-identical behavior
between old mx/ code and new src/ code.
"""
import torch
import math
from typing import Optional


def assert_bit_identical(
    old: torch.Tensor,
    new: torch.Tensor,
    name: str = "",
) -> None:
    """Assert two tensors are bit-for-bit identical.

    Handles NaN/Inf correctly: NaN == NaN, Inf == Inf, -Inf == -Inf.
    """
    assert old.shape == new.shape, (
        f"[{name}] Shape mismatch: {old.shape} vs {new.shape}"
    )
    assert old.dtype == new.dtype, (
        f"[{name}] Dtype mismatch: {old.dtype} vs {new.dtype}"
    )

    # torch.equal handles NaN correctly (NaN != NaN in ==, but torch.equal
    # requires bitwise identity for all elements)
    if not torch.equal(old, new):
        diff_mask = old != new
        # Handle NaN: NaN != NaN is True even for identical NaNs
        both_nan = torch.isnan(old) & torch.isnan(new)
        real_diff = diff_mask & ~both_nan

        if real_diff.any():
            n_diff = real_diff.sum().item()
            n_total = old.numel()
            max_abs_diff = (old - new).abs().max().item()

            # Find first diff location
            flat_idx = real_diff.reshape(-1).argmax().item()
            old_val = old.reshape(-1)[flat_idx].item()
            new_val = new.reshape(-1)[flat_idx].item()

            raise AssertionError(
                f"[{name}] Not bit-identical: {n_diff}/{n_total} elements differ, "
                f"max_abs_diff={max_abs_diff}, "
                f"first_diff: old={old_val}, new={new_val} at flat_idx={flat_idx}"
            )


def assert_close(
    ref: torch.Tensor,
    actual: torch.Tensor,
    name: str = "",
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Assert two tensors are close within tolerance.

    Use this for cases where bit-identity cannot be guaranteed (e.g., different
    computation orders). Prefer assert_bit_identical for same-algorithm tests.
    """
    assert ref.shape == actual.shape, (
        f"[{name}] Shape mismatch: {ref.shape} vs {actual.shape}"
    )

    if not torch.allclose(ref, actual, atol=atol, rtol=rtol, equal_nan=True):
        diff = (ref - actual).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"[{name}] Not close (atol={atol}, rtol={rtol}): "
            f"max_diff={max_diff}, mean_diff={mean_diff}"
        )


def assert_grads_equal(
    old_grads: list[Optional[torch.Tensor]],
    new_grads: list[Optional[torch.Tensor]],
    name: str = "",
) -> None:
    """Assert lists of gradients are bit-identical, handling None entries."""
    assert len(old_grads) == len(new_grads), (
        f"[{name}] Gradient list length mismatch: {len(old_grads)} vs {len(new_grads)}"
    )
    for i, (og, ng) in enumerate(zip(old_grads, new_grads)):
        if og is None and ng is None:
            continue
        assert og is not None and ng is not None, (
            f"[{name}] Gradient {i}: one is None, other is not"
        )
        assert_bit_identical(og, ng, name=f"{name}.grad[{i}]")
