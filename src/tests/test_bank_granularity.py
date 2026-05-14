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


from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.quantize.elemwise import quantize


class TestBankQuantizeBitExact:
    """Bit-exact verification of BANK quantization with int4 format."""

    @pytest.fixture
    def x(self):
        # 2x4 tensor: rows = [1,2,3,4], [5,6,7,8]
        return torch.tensor([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0]])

    @pytest.fixture
    def fmt(self):
        return FormatBase.from_str("int4")

    def test_bank_quantize_pot_bit_exact(self, x, fmt):
        """BANK int4 pot: hand-derived expected values."""
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)

        # Hand derivation (see verification doc):
        # Bank 0 (cols 0-1): amax=max(1,2,5,6)=6.0 → pot=8.0
        #   → [2.0, 2.0] row0, [6.0, 6.0] row1
        # Bank 1 (cols 2-3): amax=max(3,4,7,8)=8.0 → pot=8.0
        #   → [4.0, 4.0] row0, [8.0, 8.0] row1
        expected = torch.tensor([[2.0, 2.0, 4.0, 4.0],
                                 [6.0, 6.0, 8.0, 8.0]])
        assert torch.equal(result, expected), \
            f"BANK pot mismatch:\n got {result}\n expected {expected}"

    def test_bank_quantize_fp32_bit_exact(self, x, fmt):
        """BANK int4 fp32: hand-derived expected values."""
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="fp32")
        result = quantize(x, scheme)

        # Bank 0: amax=6.0 (no pot rounding)
        #   [1,2]/6→[0.25,0.25]×6=[1.5,1.5], [5,6]/6→[0.75,1.0]×6=[4.5,6.0]
        # Bank 1: amax=8.0 (no pot rounding)
        #   [3,4]/8→[0.5,0.5]×8=[4.0,4.0], [7,8]/8→[1.0,1.0]×8=[8.0,8.0]
        expected = torch.tensor([[1.5, 1.5, 4.0, 4.0],
                                 [4.5, 6.0, 8.0, 8.0]])
        assert torch.equal(result, expected), \
            f"BANK fp32 mismatch:\n got {result}\n expected {expected}"

    def test_bank_axis_0(self, fmt):
        """BANK with axis=0 splits along rows."""
        x = torch.tensor([[1.0, 2.0],
                          [5.0, 6.0],
                          [3.0, 4.0],
                          [7.0, 8.0]])  # 4x2
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=0)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)

        # Bank 0 (rows 0-1): amax=max(1,2,5,6)=6.0→pot=8.0
        # Bank 1 (rows 2-3): amax=max(3,4,7,8)=8.0→pot=8.0
        expected = torch.tensor([[2.0, 2.0],
                                 [6.0, 6.0],
                                 [4.0, 4.0],
                                 [8.0, 8.0]])
        assert torch.equal(result, expected), \
            f"BANK axis=0 mismatch:\n got {result}\n expected {expected}"
