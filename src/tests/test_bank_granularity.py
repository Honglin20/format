"""Tests for BANK granularity mode."""
import pytest
import torch
from src.scheme.granularity import GranularitySpec, GranularityMode


class TestBankGranularitySpec:
    """GranularitySpec construction and validation for BANK mode."""

    def test_bank_mode_constructs(self):
        """BANK mode with default bank_size constructs."""
        g = GranularitySpec(mode=GranularityMode.BANK)
        assert g.mode == GranularityMode.BANK
        assert g.bank_size == 16
        assert g.bank_axis == -1

    def test_bank_with_custom_size(self):
        """BANK with custom bank_size."""
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=8, bank_axis=0)
        assert g.bank_size == 8
        assert g.bank_axis == 0

    def test_bank_requires_positive_size(self):
        """BANK with bank_size <= 0 raises."""
        with pytest.raises(ValueError, match="bank_size"):
            GranularitySpec(mode=GranularityMode.BANK, bank_size=0)

    def test_bank_rejects_block_size(self):
        """BANK with block_size != 0 raises."""
        with pytest.raises(ValueError, match="block_size"):
            GranularitySpec(mode=GranularityMode.BANK, block_size=32)

    def test_bank_rejects_channel_axis(self):
        """BANK with channel_axis != 0 raises."""
        with pytest.raises(ValueError, match="channel_axis"):
            GranularitySpec(mode=GranularityMode.BANK, channel_axis=1)

    def test_non_bank_rejects_bank_axis(self):
        """PER_TENSOR with bank_axis != -1 raises."""
        with pytest.raises(ValueError, match="bank_axis"):
            GranularitySpec(mode=GranularityMode.PER_TENSOR, bank_axis=0)

    def test_bank_per_tensor_factory_unchanged(self):
        """Existing factories still work."""
        g = GranularitySpec.per_tensor()
        assert g.mode == GranularityMode.PER_TENSOR
        assert g.bank_size == 16  # default, not used for PER_TENSOR

    def test_bank_outlier_ratio_stores(self):
        """BANK stores outlier_ratio for future sparse use."""
        g = GranularitySpec(mode=GranularityMode.BANK, outlier_ratio=0.1)
        assert g.outlier_ratio == 0.1
