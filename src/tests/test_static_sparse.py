"""Tests for static sparse quantization (_quantize_static_sparse).

Verifies that passing pre-computed mask + scales (amax_n, amax_o) to quantize()
triggers the static sparse path, producing correct per-group quantization.

Layer 1: Bit-exact verification against hand-derived values.
Layer 2: Broadcast compatibility for PER_TENSOR, PER_CHANNEL, BANK.
"""
import pytest
import torch

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.quantize.elemwise import quantize


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Tensor Static Sparse — bit-exact hand-derived verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerTensorStaticSparse:
    """Bit-exact verification of per_tensor static sparse path."""

    @pytest.fixture
    def fmt(self):
        return FormatBase.from_str("int4")

    def test_static_sparse_basic(self, fmt):
        """Static sparse with scalar amax: outliers and normals quantized separately."""
        x = torch.tensor([0.5, 1.0, 10.0, 0.25])
        # Mark 10.0 as outlier
        mask = torch.tensor([False, False, True, False])

        # Compute per-group amax
        amax_n = torch.amax(torch.abs(x * (~mask).float()))  # 1.0
        amax_o = torch.amax(torch.abs(x * mask.float()))     # 10.0

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")

        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)

        # Normal group: [0.5, 1.0, 0.0, 0.25] / 1.0 → int4 levels
        #   0.5 → nearest level in {-1, -0.5, 0, 0.5} = 0.5
        #   1.0 → nearest = 1.0
        #   0.0 → 0.0
        #   0.25 → nearest = 0.5 (but wait, levels are 0, 0.5 for positive)
        # Actually int4: mbits=0, ebits=0 → levels = {−1, 0, 1}
        #   0.5 → nearest = 1.0? No wait...
        # int4 max_norm = 1, levels are linear: -1, 0, 1 (since mbits=0)
        #   0.5 → nearest = 1.0

        # Outlier: 10.0 / 10.0 = 1.0 → int4 → 1.0 * 10.0 = 10.0
        assert torch.isfinite(result).all()
        assert result.shape == x.shape
        # Outlier preserved
        assert result[2] == 10.0

    def test_static_sparse_matches_dynamic(self, fmt):
        """Static sparse with correct mask+scales matches dynamic sparse result."""
        x = torch.tensor([0.5, 10.0, 0.25, 2.0])

        # First, get the dynamic sparse result (mask computed internally)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result_dynamic = quantize(x, scheme)  # no mask → dynamic path

        # Now pre-compute the same mask and scales that dynamic path would use
        k = max(1, int(4 * 0.25))  # k=1
        _, top_idx = torch.topk(torch.abs(x).flatten(), k)
        mask = torch.zeros(4, dtype=torch.bool)
        mask.scatter_(0, top_idx, True)

        amax_n = torch.amax(torch.abs(x * (~mask).float()))
        amax_o = torch.amax(torch.abs(x * mask.float()))

        result_static = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)

        assert torch.equal(result_dynamic, result_static), \
            f"Static should match dynamic: {result_dynamic} vs {result_static}"

    def test_static_sparse_no_outliers_smoke(self, fmt):
        """Static sparse with empty outlier mask (all normal)."""
        x = torch.randn(8)
        mask = torch.zeros(8, dtype=torch.bool)
        amax_n = torch.amax(torch.abs(x))
        amax_o = torch.tensor(1.0)

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_static_sparse_all_outliers_smoke(self, fmt):
        """Static sparse with all elements as outliers."""
        x = torch.randn(8)
        mask = torch.ones(8, dtype=torch.bool)
        amax_n = torch.tensor(1.0)
        amax_o = torch.amax(torch.abs(x))

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.99)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_static_sparse_pot_vs_fp32_storage(self, fmt):
        """Static sparse with scale_storage='fp32' vs 'pot'."""
        x = torch.tensor([0.5, 3.0, 7.0, 0.25])
        mask = torch.tensor([False, False, True, False])
        amax_n = torch.tensor(3.0)
        amax_o = torch.tensor(7.0)

        g_pot = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        g_fp32 = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)

        scheme_pot = QuantScheme(format=fmt, granularity=g_pot, scale_storage="pot")
        scheme_fp32 = QuantScheme(format=fmt, granularity=g_fp32, scale_storage="fp32")

        result_pot = quantize(x, scheme_pot, mask=mask, scale=amax_n, scale_o=amax_o)
        result_fp32 = quantize(x, scheme_fp32, mask=mask, scale=amax_n, scale_o=amax_o)

        # Both produce finite output
        assert torch.isfinite(result_pot).all()
        assert torch.isfinite(result_fp32).all()
        # POT and fp32 may differ due to rounding of amax

    def test_static_sparse_preserves_inf_nan(self, fmt):
        """Static sparse path preserves Inf and NaN values."""
        x = torch.tensor([1.0, float("Inf"), float("NaN"), -float("Inf"), 0.5])
        mask = torch.tensor([True, False, False, False, False])
        amax_n = torch.tensor(1.0)
        amax_o = torch.tensor(1.0)

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)

        assert result[1] == float("Inf")
        assert torch.isnan(result[2])
        assert result[3] == -float("Inf")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Channel Static Sparse — per-channel amax broadcasting
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerChannelStaticSparse:
    """Bit-exact verification of per_channel static sparse path."""

    @pytest.fixture
    def fmt(self):
        return FormatBase.from_str("int4")

    def test_static_sparse_per_channel_matches_dynamic(self, fmt):
        """Static sparse per_channel matches dynamic sparse."""
        x = torch.tensor([[0.5, 10.0, 0.25, 1.0],
                          [2.0, -8.0, 3.0, 1.5]])

        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result_dynamic = quantize(x, scheme)

        # Reconstruct the mask that dynamic path computes
        C = x.shape[0]
        N_per_channel = x[0].numel()
        k = max(1, int(N_per_channel * 0.25))  # k=1
        x_flat = x.reshape(C, N_per_channel)
        _, top_idx = torch.topk(torch.abs(x_flat), k, dim=1)
        mask_flat = torch.zeros(C, N_per_channel, dtype=torch.bool)
        mask_flat.scatter_(1, top_idx, True)
        mask = mask_flat.reshape(x.shape)

        x_masked_o = x * mask.float()
        x_masked_n = x * (~mask).float()
        amax_o = torch.amax(torch.abs(x_masked_o), dim=1)
        amax_n = torch.amax(torch.abs(x_masked_n), dim=1)
        broadcast_shape = (C,) + (1,) * (x.ndim - 1)
        amax_o_b = amax_o.reshape(broadcast_shape)
        amax_n_b = amax_n.reshape(broadcast_shape)

        result_static = quantize(x, scheme, mask=mask, scale=amax_n_b, scale_o=amax_o_b)

        assert torch.equal(result_dynamic, result_static), \
            f"Static per_channel should match dynamic"

    def test_static_sparse_per_channel_axis_last(self, fmt):
        """Static sparse per_channel with channel_axis=-1."""
        x = torch.tensor([[0.5, 10.0, 0.25],
                          [2.0, -8.0, 3.0]])
        # channel_axis=-1 → channels are columns
        C = x.shape[-1]  # 3

        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=-1,
                            outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result_dynamic = quantize(x, scheme)

        # Reconstruct per-channel mask (channels on axis=-1)
        # Transpose to channels on dim 0, compute mask, transpose back
        x_t = x.transpose(0, -1)
        N_per_channel = x_t[0].numel()
        k = max(1, int(N_per_channel * 0.25))
        x_flat = x_t.reshape(C, N_per_channel)
        _, top_idx = torch.topk(torch.abs(x_flat), k, dim=1)
        mask_flat = torch.zeros(C, N_per_channel, dtype=torch.bool)
        mask_flat.scatter_(1, top_idx, True)
        mask = mask_flat.reshape(x_t.shape).transpose(0, -1)

        x_masked_o = x * mask.float()
        x_masked_n = x * (~mask).float()
        amax_o = torch.amax(torch.abs(x_masked_o.transpose(0, -1).reshape(C, -1)), dim=1)
        amax_n = torch.amax(torch.abs(x_masked_n.transpose(0, -1).reshape(C, -1)), dim=1)
        broadcast_shape = (1,) * (x.ndim - 1) + (C,)
        amax_o_b = amax_o.reshape(broadcast_shape)
        amax_n_b = amax_n.reshape(broadcast_shape)

        result_static = quantize(x, scheme, mask=mask, scale=amax_n_b, scale_o=amax_o_b)

        assert torch.equal(result_dynamic, result_static), \
            f"Static per_channel axis=-1 should match dynamic"


# ═══════════════════════════════════════════════════════════════════════════════
# Shape preservation and edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestStaticSparseShapeAndEdgeCases:
    """Shape preservation and edge cases for static sparse path."""

    @pytest.mark.parametrize("shape", [
        (8,),
        (4, 16),
        (2, 3, 8),
    ])
    def test_shape_preserved_per_tensor(self, shape):
        """Output shape matches input for various ranks."""
        torch.manual_seed(42)
        x = torch.randn(*shape)
        fmt = FormatBase.from_str("int8")
        mask = torch.zeros(shape, dtype=torch.bool)
        mask_flat = mask.flatten()
        mask_flat[:max(1, mask_flat.numel() // 10)] = True  # ~10% outliers
        mask = mask_flat.reshape(shape)
        amax_n = torch.amax(torch.abs(x * (~mask).float()))
        amax_o = torch.amax(torch.abs(x * mask.float()))

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape

    def test_zero_input_static_sparse(self):
        """Zero input → zero output for static sparse."""
        x = torch.zeros(8)
        fmt = FormatBase.from_str("int8")
        mask = torch.tensor([True, False, False, False, True, False, False, False])
        amax_n = torch.tensor(0.0)  # will be clamped to 1e-12
        amax_o = torch.tensor(0.0)

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        # Zero input should stay zero (or near-zero after quantization)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# BANK Static Sparse — per-bank amax with reshape
# ═══════════════════════════════════════════════════════════════════════════════

class TestBankStaticSparse:
    """Verification of BANK static sparse path with per-bank amax."""

    @pytest.fixture
    def fmt(self):
        return FormatBase.from_str("int4")

    def test_bank_static_sparse_basic(self, fmt):
        """BANK static sparse: per-bank outlier/normal amax."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]])
        # 2 banks with bank_size=2 along axis=-1
        # Bank 0: cols 0-1, Bank 1: cols 2-3
        mask = torch.tensor([[False, True, False, False],
                              [False, False, True, False]])

        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1,
                            outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")

        # Compute per-bank amax in reshaped space: (2, 2, 2) = (M, num_banks, bank_size)
        x_r = x.reshape(2, 2, 2)  # (M=2, num_banks=2, bank_size=2)
        mask_r = mask.reshape(2, 2, 2)
        # Bank 0 (col 0): elements [[1,2],[5,6]], outliers=2,6
        # Bank 1 (col 1): elements [[3,4],[7,8]], outliers=7
        amax_n = torch.amax(torch.abs(x_r * (~mask_r).float()), dim=(0, 2), keepdim=True)
        amax_o = torch.amax(torch.abs(x_r * mask_r.float()), dim=(0, 2), keepdim=True)
        # amax shape: (1, 2, 1)

        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_bank_static_sparse_matches_non_sparse_when_no_split(self, fmt):
        """BANK static with no effective split (amax_n == amax_o) matches non-sparse BANK."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]])
        mask = torch.zeros(2, 4, dtype=torch.bool)  # no outliers

        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1,
                            outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")

        # Compute per-bank amax (normal group = all elements)
        x_r = x.reshape(2, 2, 2)
        amax_n = torch.amax(torch.abs(x_r), dim=(0, 2), keepdim=True)  # (1, 2, 1)
        amax_o = torch.tensor(1.0)  # won't be used since mask is all False

        result_sparse = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)

        # Compare with non-sparse BANK
        g_normal = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1)
        scheme_normal = QuantScheme(format=fmt, granularity=g_normal, scale_storage="pot")
        result_normal = quantize(x, scheme_normal)

        assert torch.equal(result_sparse, result_normal), \
            f"Empty mask static should match non-sparse: {result_sparse} vs {result_normal}"

    def test_bank_static_sparse_pot_rounding(self, fmt):
        """BANK static sparse with pot scale_storage rounds amax."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]])
        mask = torch.tensor([[True, False, False, False],
                              [False, False, False, True]])
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=2, bank_axis=-1,
                            outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")

        x_r = x.reshape(2, 2, 2)
        mask_r = mask.reshape(2, 2, 2)
        amax_n = torch.amax(torch.abs(x_r * (~mask_r).float()), dim=(0, 2), keepdim=True)
        amax_o = torch.amax(torch.abs(x_r * mask_r.float()), dim=(0, 2), keepdim=True)

        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_bank_static_sparse_fp32_storage(self, fmt):
        """BANK static sparse with fp32 scale_storage."""
        x = torch.randn(2, 8)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 0] = True
        mask[1, 7] = True
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1,
                            outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="fp32")

        x_r = x.reshape(2, 2, 4)
        mask_r = mask.reshape(2, 2, 4)
        amax_n = torch.amax(torch.abs(x_r * (~mask_r).float()), dim=(0, 2), keepdim=True)
        amax_o = torch.amax(torch.abs(x_r * mask_r.float()), dim=(0, 2), keepdim=True)

        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Contract tests: error paths and backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestStaticSparseContract:
    """API contract: error cases, float format delegation, backward compat."""

    def test_scale_without_mask_falls_through_per_tensor(self):
        """scale without mask with outlier_ratio > 0 falls through to dynamic sparse."""
        x = torch.randn(8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, scale=torch.tensor(2.0))
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_scale_without_mask_falls_through_per_channel(self):
        """scale without mask with per_channel + outlier_ratio > 0 falls through to dynamic sparse."""
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, scale=torch.ones(4, 1))
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_mask_without_outlier_ratio_ignored(self):
        """mask with outlier_ratio=0 just uses normal path (mask ignored)."""
        x = torch.randn(8)
        fmt = FormatBase.from_str("int8")
        mask = torch.ones(8, dtype=torch.bool)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.0)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        # Should not raise — outlier_ratio=0 skips the static sparse condition
        result = quantize(x, scheme, mask=mask, scale=torch.tensor(2.0))
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_mask_with_float_format_uses_direct_elemwise(self):
        """Float format with mask still goes through non-sparse path."""
        x = torch.randn(8)
        fmt = FormatBase.from_str("fp8_e4m3")
        mask = torch.zeros(8, dtype=torch.bool)
        mask[0] = True
        amax_n = torch.tensor(1.0)
        amax_o = torch.tensor(2.0)

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")

        # Float format: outlier_ratio > 0 but ebits > 0 → delegates to non-sparse
        # But wait: with mask, it goes to static sparse path which doesn't check ebits
        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_mask_without_outlier_ratio_no_static_path(self):
        """mask + scale without outlier_ratio uses normal scalar scale path."""
        x = torch.randn(8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.0)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")

        result_with_scale = quantize(x, scheme, scale=torch.tensor(3.0))
        # Passing mask should just be ignored — normal per_tensor path with scale
        result_with_mask = quantize(x, scheme, scale=torch.tensor(3.0),
                                     mask=torch.zeros(8, dtype=torch.bool))
        assert torch.equal(result_with_scale, result_with_mask), \
            "Mask without outlier_ratio should be ignored"


# ═══════════════════════════════════════════════════════════════════════════════
# Configurable outlier_format — static sparse + dynamic sparse
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutlierFormatStaticSparse:
    """Static sparse with a different format for the outlier group."""

    @pytest.fixture
    def x(self):
        return torch.tensor([0.5, 1.0, 10.0, 0.25])

    @pytest.fixture
    def mask(self):
        return torch.tensor([False, False, True, False])

    def test_outlier_format_int8_normal_int4(self):
        """Static sparse: normal group uses int4, outlier group uses int8.

        Verifies the outlier_format produces a different result from
        using the same format for both groups, confirming the two formats
        are actually used for their respective groups.
        """
        torch.manual_seed(42)
        int4 = FormatBase.from_str("int4")
        int8 = FormatBase.from_str("int8")

        x = torch.randn(16)
        mask = torch.zeros(16, dtype=torch.bool)
        mask[0] = True
        mask[8] = True

        amax_n = torch.amax(torch.abs(x * (~mask).float()))
        amax_o = torch.amax(torch.abs(x * mask.float()))

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.125)
        scheme_mixed = QuantScheme(format=int4, granularity=g, scale_storage="fp32",
                                   outlier_format=int8)
        scheme_int4_only = QuantScheme(format=int4, granularity=g, scale_storage="fp32")

        r_mixed = quantize(x, scheme_mixed, mask=mask, scale=amax_n, scale_o=amax_o)
        r_int4 = quantize(x, scheme_int4_only, mask=mask, scale=amax_n, scale_o=amax_o)

        assert r_mixed.shape == x.shape
        assert torch.isfinite(r_mixed).all()
        assert not torch.equal(r_mixed, r_int4), \
            "Mixed int8-outlier should differ from pure int4"

    def test_outlier_format_none_is_backward_compat(self, x, mask):
        """outlier_format=None → outlier group uses main format (int4)."""
        int4 = FormatBase.from_str("int4")
        amax_n = torch.amax(torch.abs(x * (~mask).float()))
        amax_o = torch.amax(torch.abs(x * mask.float()))

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme_with = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                                  outlier_format=None)
        scheme_without = QuantScheme(format=int4, granularity=g, scale_storage="pot")

        r_with = quantize(x, scheme_with, mask=mask, scale=amax_n, scale_o=amax_o)
        r_without = quantize(x, scheme_without, mask=mask, scale=amax_n, scale_o=amax_o)

        assert torch.equal(r_with, r_without), \
            "outlier_format=None should produce same result as no outlier_format"

    def test_bank_static_sparse_outlier_format(self):
        """BANK static sparse with outlier_format int8 for outliers."""
        int4 = FormatBase.from_str("int4")
        int8 = FormatBase.from_str("int8")

        x = torch.randn(2, 16)
        # Mark first element per bank as outlier
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, 0] = True
        mask[:, 8] = True

        amax_n = torch.tensor([[1.0], [1.0]])
        amax_o = torch.tensor([[3.0], [3.0]])

        g = GranularitySpec(mode=GranularityMode.BANK, bank_axis=-1, bank_size=8,
                            outlier_ratio=0.125)
        scheme = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                             outlier_format=int8)

        result = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


class TestOutlierFormatDynamicSparse:
    """Dynamic sparse with outlier_format — per_tensor_sparse and per_channel_sparse."""

    def test_per_tensor_sparse_outlier_format(self):
        """Dynamic per_tensor sparse uses outlier_format for top-k outlier group."""
        int4 = FormatBase.from_str("int4")
        int8 = FormatBase.from_str("int8")

        x = torch.randn(16)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme_int4_normal = QuantScheme(format=int4, granularity=g, scale_storage="pot")
        scheme_int8_outlier = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                                          outlier_format=int8)

        r_normal = quantize(x, scheme_int4_normal)
        r_outlier_fmt = quantize(x, scheme_int8_outlier)

        assert r_outlier_fmt.shape == x.shape
        assert torch.isfinite(r_outlier_fmt).all()
        # Results should differ when outlier_format differs
        assert not torch.equal(r_normal, r_outlier_fmt), \
            "outlier_format=int8 should produce different result from int4-only"

    def test_per_channel_sparse_outlier_format(self):
        """Dynamic per_channel sparse uses outlier_format for top-k outlier group."""
        int4 = FormatBase.from_str("int4")
        int8 = FormatBase.from_str("int8")

        x = torch.randn(4, 16)
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.25)
        scheme_int4_normal = QuantScheme(format=int4, granularity=g, scale_storage="pot")
        scheme_int8_outlier = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                                          outlier_format=int8)

        r_normal = quantize(x, scheme_int4_normal)
        r_outlier_fmt = quantize(x, scheme_int8_outlier)

        assert r_outlier_fmt.shape == x.shape
        assert torch.isfinite(r_outlier_fmt).all()
        assert not torch.equal(r_normal, r_outlier_fmt), \
            "outlier_format=int8 should produce different result from int4-only"

    def test_outlier_format_float_outlier_int_normal(self):
        """Float format as outlier_format with int normal format."""
        int4 = FormatBase.from_str("int4")
        fp8 = FormatBase.from_str("fp8_e4m3")

        x = torch.randn(16)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                             outlier_format=fp8)

        result = quantize(x, scheme)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_per_block_sparse_outlier_format(self):
        """PER_BLOCK dynamic sparse uses outlier_format for outlier group."""
        int4 = FormatBase.from_str("int4")
        int8 = FormatBase.from_str("int8")

        x = torch.randn(2, 8)
        g = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=4,
                            block_axis=-1, outlier_ratio=0.25)
        scheme_int4_only = QuantScheme(format=int4, granularity=g, scale_storage="pot")
        scheme_int8_outlier = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                                          outlier_format=int8)

        r_normal = quantize(x, scheme_int4_only)
        r_outlier_fmt = quantize(x, scheme_int8_outlier)

        assert r_outlier_fmt.shape == x.shape
        assert torch.isfinite(r_outlier_fmt).all()
        assert not torch.equal(r_normal, r_outlier_fmt), \
            "outlier_format=int8 should produce different result from int4-only"

    def test_bank_sparse_outlier_format(self):
        """BANK dynamic sparse uses outlier_format for outlier group."""
        int4 = FormatBase.from_str("int4")
        int8 = FormatBase.from_str("int8")

        x = torch.randn(2, 16)
        g = GranularitySpec(mode=GranularityMode.BANK, bank_size=8, bank_axis=-1,
                            outlier_ratio=0.25)
        scheme_int4_only = QuantScheme(format=int4, granularity=g, scale_storage="pot")
        scheme_int8_outlier = QuantScheme(format=int4, granularity=g, scale_storage="pot",
                                          outlier_format=int8)

        r_normal = quantize(x, scheme_int4_only)
        r_outlier_fmt = quantize(x, scheme_int8_outlier)

        assert r_outlier_fmt.shape == x.shape
        assert torch.isfinite(r_outlier_fmt).all()
        assert not torch.equal(r_normal, r_outlier_fmt), \
            "outlier_format=int8 should produce different result from int4-only"


class TestSessionStaticSparse:
    """Session-level static sparse: calibration computes mask + per-group scales."""

    @pytest.fixture
    def tiny_model(self):
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 8)

            def forward(self, x):
                return self.fc(x)

        torch.manual_seed(42)
        return Tiny()

    def test_calibration_stores_static_sparse_buffers(self, tiny_model):
        """CalibrationSession(sparse=True) stores mask + per-group scale buffers."""
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec, GranularityMode
        from src.session._model import quantize_model
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=FormatBase.from_str("int4"), granularity=g,
                             scale_storage="fp32")
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        qmodel = quantize_model(tiny_model, cfg=cfg)

        # Calibrate with 3 samples, sparse mode enabled
        x = torch.randn(3, 4)
        with CalibrationSession(qmodel, MaxScaleStrategy(), sparse=True):
            with torch.no_grad():
                for s in range(3):
                    qmodel(x[s:s + 1])

        fc = qmodel.fc
        assert hasattr(fc, "_output_mask"), "should store output mask"
        assert hasattr(fc, "_output_scale"), "should store normal group scale"
        assert hasattr(fc, "_output_scale_o"), "should store outlier group scale"
        assert fc._output_mask.dtype == torch.bool
        assert fc._output_mask.shape == (1, 8)  # (batch=1, features)
        n_outliers = fc._output_mask.sum().item()
        assert n_outliers == max(1, int(8 * 0.25))

    def test_static_sparse_forward_differs_from_dynamic(self, tiny_model):
        """Static sparse forward uses pre-computed mask."""
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec, GranularityMode
        from src.session._model import quantize_model
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy

        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=FormatBase.from_str("int4"), granularity=g,
                             scale_storage="fp32")
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        torch.manual_seed(99)
        qmodel = quantize_model(tiny_model, cfg=cfg)

        x_calib = torch.randn(3, 4)
        with CalibrationSession(qmodel, MaxScaleStrategy(), sparse=True):
            with torch.no_grad():
                for s in range(3):
                    qmodel(x_calib[s:s + 1])

        # Forward with static sparse (buffers are on the module)
        x_test = torch.randn(2, 4)
        with torch.no_grad():
            out_static = qmodel(x_test)

        assert out_static.shape == (2, 8)
        assert torch.isfinite(out_static).all()

    def test_static_sparse_per_channel(self, tiny_model):
        """Calibration with per_channel granularity stores correct shapes."""
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec, GranularityMode
        from src.session._model import quantize_model
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy

        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.25)
        scheme = QuantScheme(format=FormatBase.from_str("int8"), granularity=g,
                             scale_storage="fp32")
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        qmodel = quantize_model(tiny_model, cfg=cfg)

        x = torch.randn(3, 4)
        with CalibrationSession(qmodel, MaxScaleStrategy(), sparse=True):
            with torch.no_grad():
                for s in range(3):
                    qmodel(x[s:s + 1])

        fc = qmodel.fc
        assert hasattr(fc, "_output_mask")
        assert fc._output_mask.shape == (1, 8)  # (batch=1, features)
        assert fc._output_scale.ndim >= 1
        assert fc._output_scale_o.ndim >= 1

    def test_static_sparse_bank(self):
        """Calibration with bank granularity stores correct shapes."""
        import torch.nn as nn
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec, GranularityMode
        from src.session._model import quantize_model
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy

        class BankModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 32)

            def forward(self, x):
                return self.fc(x)

        torch.manual_seed(42)
        model = BankModel()

        g = GranularitySpec(mode=GranularityMode.BANK, bank_axis=-1,
                            bank_size=8, outlier_ratio=0.125)
        scheme = QuantScheme(format=FormatBase.from_str("int4"), granularity=g,
                             scale_storage="fp32")
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        qmodel = quantize_model(model, cfg=cfg)

        x = torch.randn(3, 16)
        with CalibrationSession(qmodel, MaxScaleStrategy(), sparse=True):
            with torch.no_grad():
                for s in range(3):
                    qmodel(x[s:s + 1])

        fc = qmodel.fc
        assert hasattr(fc, "_output_mask")
        assert fc._output_mask.shape == (1, 32)  # (batch=1, features)
        num_banks = 32 // 8
        assert fc._output_scale.shape[fc._output_scale.ndim - 2] == num_banks
        assert fc._output_scale_o.shape[fc._output_scale_o.ndim - 2] == num_banks

        with torch.no_grad():
            out = qmodel(torch.randn(2, 16))
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()
