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


class TestBankEdgeCases:
    """Shape preservation, boundary, and error cases for BANK quantization."""

    @pytest.mark.parametrize("shape,axis", [
        ((4, 8), -1),
        ((2, 3, 8), -1),
        ((2, 4, 6, 8), 1),
    ])
    def test_shape_preserved(self, shape, axis):
        """Output shape matches input for various ranks."""
        torch.manual_seed(42)
        x = torch.randn(*shape)
        fmt = FormatBase.from_str("int8")
        dim_size = shape[axis] if axis >= 0 else shape[axis]
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=dim_size // 2, bank_axis=axis)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        assert result.shape == x.shape

    def test_non_divisible_raises(self):
        """Dimension not divisible by bank_size raises."""
        x = torch.randn(4, 7)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=3, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        with pytest.raises(ValueError, match="not divisible"):
            quantize(x, scheme)

    def test_bank_axis_out_of_range_raises(self):
        """bank_axis out of range raises."""
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=5)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        with pytest.raises(ValueError, match="bank_axis"):
            quantize(x, scheme)

    def test_zero_input(self):
        """Zero input → zero output."""
        x = torch.zeros(4, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        assert (result == 0).all()

    def test_static_scale_smoke(self):
        """Static scale with BANK produces finite output."""
        x = torch.randn(2, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        # Pre-computed scale: 2 banks → shape (1, 2, 1) after reshape
        scale = torch.tensor([[[2.0], [3.0]]])  # broadcastable
        result = quantize(x, scheme, scale=scale)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_per_tensor_vs_bank_full_tensor(self):
        """BANK with bank_size covering entire axis = per_tensor behavior."""
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("int8")

        g_pt = GranularitySpec.per_tensor()
        scheme_pt = QuantScheme(format=fmt, granularity=g_pt, scale_storage="pot")
        result_pt = quantize(x, scheme_pt)

        g_bank = GranularitySpec(mode=GranularityMode.BANK, bank_size=8, bank_axis=-1)
        scheme_bank = QuantScheme(format=fmt, granularity=g_bank, scale_storage="pot")
        result_bank = quantize(x, scheme_bank)

        assert torch.equal(result_pt, result_bank), \
            "BANK covering full axis should match per_tensor"


class TestBankQuantConfig:
    """QuantConfig → OpQuantConfig resolution for BANK mode."""

    def test_quantconfig_bank_resolves(self):
        """QuantConfig with bank granularity produces BANK GranularitySpec."""
        from src.session._config import QuantConfig

        cfg = QuantConfig(
            w_format="int8",
            w_granularity="bank",
            w_block_size=16,
            w_axis=0,
        )
        op_cfg = cfg.to_op_config()
        g = op_cfg.weight.granularity
        assert g.mode == GranularityMode.BANK
        assert g.bank_size == 16
        assert g.bank_axis == 0

    def test_quantconfig_bank_requires_size(self):
        """BANK without block_size raises."""
        from src.session._config import QuantConfig
        with pytest.raises(ValueError, match="bank_size"):
            QuantConfig(w_granularity="bank")

    def test_quantconfig_bank_outlier_ratio(self):
        """BANK + outlier_ratio stores correctly for future sparse."""
        from src.session._config import QuantConfig
        cfg = QuantConfig(
            w_format="int8",
            w_granularity="bank",
            w_block_size=8,
            outlier_ratio=0.1,
        )
        op_cfg = cfg.to_op_config()
        assert op_cfg.weight.granularity.outlier_ratio == 0.1


class TestBankSessionIntegration:
    """Session-level tests: QuantConfig with bank → qmodel end-to-end."""

    @pytest.fixture
    def simple_model(self):
        import torch.nn as nn

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.fc2(x)
                return x

        torch.manual_seed(42)
        return TinyMLP()

    @pytest.mark.xfail(reason="Observer does not support BANK mode yet (P3)")
    def test_bank_smoke(self, simple_model):
        """QuantConfig bank mode produces valid output."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        cfg = QuantConfig(
            name="test-bank",
            w_format="int8",
            w_granularity="bank",
            w_block_size=2,
            w_axis=0,
            a_granularity="bank",
            a_block_size=2,
            a_axis=-1,
            weight_only=False,
        ).to_op_config()

        x = torch.randn(3, 4)
        qmodel, fp32_model, result = run_quantization(
            simple_model, cfg, calib_data=[x], keep_fp32=False)
        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (3, 2)
        assert torch.isfinite(out).all()
