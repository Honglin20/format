"""Tests for compute_sparse_mask — per-sample top-k + cross-sample voting."""
import pytest
import torch
from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec, GranularityMode


_FMT = FormatBase.from_str("int8")


class TestSparseMaskShapeAndBasics:
    """Shape correctness and basic invariants of compute_sparse_mask."""

    def test_mask_shape_matches_tensor(self):
        """Mask has same shape as a single sample (not including batch dim)."""
        from src.quantize._sparse_mask import compute_sparse_mask

        x_calib = torch.randn(3, 4, 8)  # S=3, shape=(4,8)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.25)
        assert mask.shape == (4, 8)
        assert mask.dtype == torch.bool

    def test_mask_outlier_count_matches_ratio(self):
        """Number of True values = outlier_ratio * numel."""
        from src.quantize._sparse_mask import compute_sparse_mask

        x_calib = torch.randn(5, 3, 6)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.2)
        k_expected = max(1, int(3 * 6 * 0.2))
        assert mask.sum().item() == k_expected

    def test_single_sample_returns_self_mask(self):
        """With S=1, mask_avg = mask_0, final mask is top-k of that mask."""
        from src.quantize._sparse_mask import compute_sparse_mask

        # Construct x so top-3 positions are uniquely determined
        x_calib = torch.tensor([[[10.0, 1.0, 2.0],
                                  [3.0, 8.0, 4.0]]])  # S=1, 2x3
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        # k = 6 * 0.5 = 3
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.5)
        # Top-3 by magnitude: 10.0 at (0,0), 8.0 at (1,1), 4.0 at (1,2)
        expected = torch.tensor([[True, False, False],
                                 [False, True, True]])
        assert torch.equal(mask, expected)

    def test_per_channel_groups_independent(self):
        """PER_CHANNEL: each channel does top-k independently, then global vote.

        3 samples to ensure no ties in mask_avg values.
        """
        from src.quantize._sparse_mask import compute_sparse_mask

        # 3 samples, 2 channels (rows), 4 columns, ratio=0.5
        # k_per_ch=2, k_total=4 (8*0.5). Design for unique avg scores.
        # Ch0: S0/S2 pick cols 0,2; S1 picks cols 0,1 → avg: 1.0, 1/3, 2/3, 0
        # Ch1: S0/S2 pick cols 1,3; S1 picks cols 1,2 → avg: 0, 1.0, 1/3, 2/3
        x_calib = torch.tensor([
            [[10.0, 1.0, 8.0, 2.0],
             [1.0, 10.0, 2.0, 8.0]],   # S0
            [[10.0, 8.0, 1.0, 2.0],
             [1.0, 10.0, 8.0, 2.0]],   # S1
            [[10.0, 1.0, 8.0, 2.0],
             [1.0, 10.0, 2.0, 8.0]],   # S2 (= S0)
        ])  # S=3, 2x4

        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.5)

        # mask_avg scores:
        # (0,0)=1.0, (0,1)=1/3, (0,2)=2/3, (0,3)=0,
        # (1,0)=0,   (1,1)=1.0, (1,2)=1/3, (1,3)=2/3
        # Global top-4 unique: (0,0)=1.0, (1,1)=1.0, (0,2)=2/3, (1,3)=2/3
        expected = torch.tensor([[True, False, True, False],
                                 [False, True, False, True]])
        assert torch.equal(mask, expected)


class TestSparseMaskVoting:
    """Cross-sample voting correctness with designed data."""

    def test_voting_consensus(self):
        """Position that is outlier in ALL samples gets selected over partial."""
        from src.quantize._sparse_mask import compute_sparse_mask

        # 3 samples, 2x2, PER_TENSOR, ratio=0.25 → k_global=1
        # Position (0,0) is always the largest magnitude → avg=1.0
        # Other positions vary
        x_calib = torch.tensor([
            [[10.0, 1.0], [1.0, 1.0]],
            [[10.0, 1.0], [1.0, 1.0]],
            [[10.0, 1.0], [1.0, 1.0]],
        ])  # S=3
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.25)

        # k_per_sample=1: each sample picks (0,0). Avg: (0,0)=1.0, rest=0.
        # k_total=1 → picks (0,0)
        expected = torch.tensor([[True, False],
                                 [False, False]])
        assert torch.equal(mask, expected)

    def test_voting_partial_consensus(self):
        """Position that is outlier in majority wins over minority."""
        from src.quantize._sparse_mask import compute_sparse_mask

        # 3 samples, 2x3, PER_TENSOR, ratio=1/3 → k_per_sample=2, k_total=2
        # Sample 0: [10, 1, 2, 3, 1, 4] → top-2: (0,0)=10, (1,2)=4
        # Sample 1: [10, 1, 1, 9, 1, 1] → top-2: (0,0)=10, (1,0)=9
        # Sample 2: [10, 1, 1, 1, 8, 1] → top-2: (0,0)=10, (1,1)=8
        x_calib = torch.tensor([
            [[10.0, 1.0, 2.0], [3.0, 1.0, 4.0]],
            [[10.0, 1.0, 1.0], [9.0, 1.0, 1.0]],
            [[10.0, 1.0, 1.0], [1.0, 8.0, 1.0]],
        ])
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=1.0 / 3.0)

        # mask_avg: (0,0)=1.0, (1,0)=1/3, (1,1)=1/3, (1,2)=1/3, rest=0
        # top-2: (0,0) and first of the 1/3 positions
        assert mask[0, 0].item() is True

    def test_all_identical_samples(self):
        """All samples identical → result equals single-sample top-k."""
        from src.quantize._sparse_mask import compute_sparse_mask

        x_single = torch.tensor([[10.0, 1.0, 2.0],
                                 [3.0, 8.0, 4.0]])  # 2x3
        x_calib = x_single.unsqueeze(0).repeat(4, 1, 1)  # S=4
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)

        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.5)
        # All 4 samples same: top-3 positions: (0,0)=10, (1,1)=8, (1,2)=4
        # Avg = mask_0. Top-3 of the exact same mask → those 3 positions.
        assert mask[0, 0].item() is True
        assert mask[1, 1].item() is True
        assert mask[1, 2].item() is True
        assert mask.sum().item() == 3


class TestSparseMaskBankGranularity:
    """BANK granularity mode with compute_sparse_mask."""

    def test_bank_groups_independent(self):
        """BANK mode: each bank does top-k independently per sample.

        group_size for a bank = all elements in that bank segment
        = M * bank_size (for 2D). k = group_size * outlier_ratio.
        """
        from src.quantize._sparse_mask import compute_sparse_mask

        # 3 samples, 2x4 tensor, bank_axis=-1, bank_size=2 → 2 banks
        # Each bank: group_size = 2*2 = 4, ratio=0.25 → k_per_bank=1
        # All 3 samples identical:
        # S0 Bank 0: [[10,1],[1,1]] → top-1: (0,0)=10
        # S0 Bank 1: [[3,2],[2,8]] → top-1: (1,3)=8
        single = [[10.0, 1.0, 3.0, 2.0],
                  [1.0, 1.0, 2.0, 8.0]]
        x_calib = torch.tensor([single, single, single])  # S=3, 2x4

        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.25)

        # Bank 0 avg: (0,0)=1.0, others=0 → top-1: (0,0)
        # Bank 1 avg: (1,3)=1.0, others=0 → top-1: (1,3)
        expected = torch.tensor([[True, False, False, False],
                                 [False, False, False, True]])
        assert torch.equal(mask, expected)


class TestSparseMaskBlockGranularity:
    """PER_BLOCK granularity mode with compute_sparse_mask."""

    def test_per_block_groups(self):
        """PER_BLOCK: each block does top-k independently per sample."""
        from src.quantize._sparse_mask import compute_sparse_mask

        # 3 identical samples, 2x4, block_axis=-1, block_size=2 → 4 blocks
        single = [[10.0, 1.0, 3.0, 2.0],
                  [1.0, 1.0, 2.0, 8.0]]
        x_calib = torch.tensor([single, single, single])
        g = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=2,
                            block_axis=-1)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.5)

        # Block (0,0-1): [10,1]→col0; (0,2-3): [3,2]→col2; (1,0-1): [1,1]→col0; (1,2-3): [2,8]→col3
        assert mask[0, 0].item() is True
        assert mask[0, 2].item() is True
        assert mask[1, 3].item() is True
        assert mask.sum().item() == 4


class TestSparseMaskEdgeCases:
    """Input validation and edge cases."""

    def test_ratio_zero_raises(self):
        """outlier_ratio=0 raises — use non-sparse path."""
        from src.quantize._sparse_mask import compute_sparse_mask
        x_calib = torch.randn(3, 4, 8)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        with pytest.raises(ValueError, match="outlier_ratio"):
            compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.0)

    def test_ratio_one_raises(self):
        """outlier_ratio=1 raises — all elements outliers, degenerate."""
        from src.quantize._sparse_mask import compute_sparse_mask
        x_calib = torch.randn(3, 4, 8)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        with pytest.raises(ValueError, match="outlier_ratio"):
            compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=1.0)

    def test_no_batch_dim_raises(self):
        """1D tensor (no batch dim) raises."""
        from src.quantize._sparse_mask import compute_sparse_mask
        x_calib = torch.randn(8)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR)
        with pytest.raises(ValueError, match="batch dim"):
            compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.1)

    def test_per_block_full_group_competition(self):
        """PER_BLOCK: all rows within a block compete for the k outlier spots.

        With 4 rows and block_size=2, a block tile has 8 elements.
        If one row has a dominant element, it should claim more outlier
        spots than a row with uniformly small values.
        """
        from src.quantize._sparse_mask import compute_sparse_mask

        # 4 identical samples, 4x4 tensor, block_axis=-1, block_size=2
        # Each block is a 4×2 tile (8 elements per block, 2 blocks total)
        # Block 0 (cols 0-1): row 0 has values [100, 0], rows 1-3 have [0.1, 0.1]
        #   → row 0 should dominate top-k for block 0
        single = [
            [100.0, 0.0, 1.0, 1.0],   # row 0: dominant in block 0
            [0.1, 0.1, 1.0, 1.0],     # row 1: small values in block 0
            [0.1, 0.1, 1.0, 1.0],     # row 2: small values in block 0
            [0.1, 0.1, 1.0, 1.0],     # row 3: small values in block 0
        ]
        x_calib = torch.tensor([single, single, single])
        g = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=2,
                            block_axis=-1)
        mask = compute_sparse_mask(x_calib, _FMT, g, outlier_ratio=0.25)

        # Block 0 (cols 0-1): group_size=8, k=2. Row 0's 100.0 is clearly top.
        # Position (0,0) with value 100.0 must be an outlier.
        assert mask[0, 0].item() is True

        # Block 1 (cols 2-3): all values are 1.0 (all rows tie).
        # With group_size=8, k=2. Positions are determined by tie-breaking.

        # Verify shape and total outlier count
        assert mask.shape == (4, 4)
        # 2 blocks × k=2 per block = 4 outliers
        assert mask.sum().item() == 4
