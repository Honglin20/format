"""
Tests for iter_slices — P3.1-b.

Covers: PER_TENSOR, PER_CHANNEL (positive + negative axis), PER_BLOCK,
DYNAMIC_GROUP (stub), and unknown mode error.
"""
import pytest
import torch

from src.analysis.slicing import iter_slices, SliceKey
from src.scheme.granularity import GranularitySpec, GranularityMode


# ---------------------------------------------------------------------------
# PER_TENSOR
# ---------------------------------------------------------------------------

def test_per_tensor_yields_single_slice():
    fp32 = torch.randn(3, 4)
    quant = torch.randn(3, 4)
    g = GranularitySpec.per_tensor()
    results = list(iter_slices(fp32, quant, g))
    assert len(results) == 1
    key, f, q = results[0]
    assert key == ("tensor",)
    assert torch.equal(f, fp32)
    assert torch.equal(q, quant)


# ---------------------------------------------------------------------------
# PER_CHANNEL
# ---------------------------------------------------------------------------

def test_per_channel_yields_per_channel_slices():
    fp32 = torch.randn(3, 5)
    quant = torch.randn(3, 5)
    g = GranularitySpec.per_channel(axis=0)
    results = list(iter_slices(fp32, quant, g))
    assert len(results) == 3
    for i, (key, f, q) in enumerate(results):
        assert key == ("channel", i)
        assert torch.equal(f, fp32.select(0, i))
        assert torch.equal(q, quant.select(0, i))


def test_per_channel_negative_axis():
    fp32 = torch.randn(3, 5)
    quant = torch.randn(3, 5)
    g_neg = GranularitySpec.per_channel(axis=-1)
    g_pos = GranularitySpec.per_channel(axis=1)
    results_neg = list(iter_slices(fp32, quant, g_neg))
    results_pos = list(iter_slices(fp32, quant, g_pos))
    assert len(results_neg) == 5
    assert len(results_pos) == 5
    for i in range(5):
        assert results_neg[i][0] == ("channel", i)
        assert results_pos[i][0] == ("channel", i)
        assert torch.equal(results_neg[i][1], results_pos[i][1])


def test_per_channel_axis1():
    """axis=1 on (3,4,5) shape should yield 4 slices."""
    fp32 = torch.randn(3, 4, 5)
    quant = torch.randn(3, 4, 5)
    g = GranularitySpec.per_channel(axis=1)
    results = list(iter_slices(fp32, quant, g))
    assert len(results) == 4
    assert results[0][0] == ("channel", 0)


# ---------------------------------------------------------------------------
# PER_BLOCK
# ---------------------------------------------------------------------------

def test_per_block_exact_division():
    fp32 = torch.randn(2, 8)
    quant = torch.randn(2, 8)
    g = GranularitySpec.per_block(4)
    results = list(iter_slices(fp32, quant, g))
    assert len(results) == 2
    assert results[0][0] == ("block", 0)
    assert results[1][0] == ("block", 1)
    assert results[0][1].shape == (2, 4)
    assert results[1][1].shape == (2, 4)


def test_per_block_non_exact_division():
    """Last block may be smaller than block_size."""
    fp32 = torch.randn(2, 10)
    quant = torch.randn(2, 10)
    g = GranularitySpec.per_block(4)
    results = list(iter_slices(fp32, quant, g))
    # 10 / 4 = 3 full blocks (12) rounded up = 3, last block = 2 elements
    assert len(results) == 3
    assert results[0][1].shape == (2, 4)  # block 0: [0:4]
    assert results[1][1].shape == (2, 4)  # block 1: [4:8]
    assert results[2][1].shape == (2, 2)  # block 2: [8:10]


def test_per_block_3d_tensor():
    """Block slicing always along last dim."""
    fp32 = torch.randn(2, 3, 16)
    quant = torch.randn(2, 3, 16)
    g = GranularitySpec.per_block(8)
    results = list(iter_slices(fp32, quant, g))
    assert len(results) == 2
    assert results[0][1].shape == (2, 3, 8)


# ---------------------------------------------------------------------------
# DYNAMIC_GROUP (stub — requires group_map)
# ---------------------------------------------------------------------------

def test_dynamic_group_with_group_map():
    fp32 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    quant = torch.tensor([1.1, 2.1, 3.1, 4.1])
    group_map = torch.tensor([0, 0, 1, 1])
    # Need to construct GranularitySpec with DYNAMIC_GROUP mode
    g = GranularitySpec(mode=GranularityMode.DYNAMIC_GROUP)
    results = list(iter_slices(fp32, quant, g, group_map=group_map))
    assert len(results) == 2
    assert results[0][0] == ("group", 0)
    assert results[1][0] == ("group", 1)
    assert torch.equal(results[0][1], torch.tensor([1.0, 2.0]))
    assert torch.equal(results[1][1], torch.tensor([3.0, 4.0]))


def test_dynamic_group_without_group_map_raises():
    fp32 = torch.randn(4)
    quant = torch.randn(4)
    g = GranularitySpec(mode=GranularityMode.DYNAMIC_GROUP)
    with pytest.raises(ValueError, match="group_map"):
        list(iter_slices(fp32, quant, g))


# ---------------------------------------------------------------------------
# Unknown mode
# ---------------------------------------------------------------------------

def test_unknown_granularity_mode_raises():
    fp32 = torch.randn(3)
    quant = torch.randn(3)
    # Create a GranularitySpec with an invalid mode by bypassing __post_init__
    # Instead, we test iter_slices directly by constructing a mock
    from unittest.mock import MagicMock
    g = MagicMock()
    g.mode = "invalid_mode"
    with pytest.raises(ValueError, match="Unknown granularity mode"):
        list(iter_slices(fp32, quant, g))


# ---------------------------------------------------------------------------
# SliceKey type
# ---------------------------------------------------------------------------

def test_slice_key_is_tuple_of_strings():
    """SliceKey — first element is always a string tag."""
    fp32 = torch.randn(3, 4)
    quant = torch.randn(3, 4)
    g = GranularitySpec.per_channel(axis=0)
    for key, _, _ in iter_slices(fp32, quant, g):
        assert isinstance(key, tuple)
        assert isinstance(key[0], str)


# ---------------------------------------------------------------------------
# PER_CHANNEL out-of-bounds (M1 fix)
# ---------------------------------------------------------------------------

def test_per_channel_out_of_bounds_positive_raises():
    fp32 = torch.randn(3, 4)
    quant = torch.randn(3, 4)
    g = GranularitySpec.per_channel(axis=5)
    with pytest.raises(ValueError, match="out of range"):
        list(iter_slices(fp32, quant, g))


def test_per_channel_out_of_bounds_negative_raises():
    fp32 = torch.randn(3, 4)
    quant = torch.randn(3, 4)
    g = GranularitySpec.per_channel(axis=-100)
    with pytest.raises(ValueError, match="out of range"):
        list(iter_slices(fp32, quant, g))


# ---------------------------------------------------------------------------
# DYNAMIC_GROUP block_size validation (M3 fix)
# ---------------------------------------------------------------------------

def test_dynamic_group_rejects_nonzero_block_size():
    with pytest.raises(ValueError, match="DYNAMIC_GROUP requires block_size=0"):
        GranularitySpec(mode=GranularityMode.DYNAMIC_GROUP, block_size=5)
