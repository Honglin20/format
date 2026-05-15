"""
Sparse (outlier_ratio) generalization tests for per_tensor and per_channel.

Layer 1: Bit-exact verification of format-level quantize() with outlier_ratio > 0.
Layer 4: Session/QuantConfig integration tests for per_tensor + per_channel sparse.

Verification docs: docs/verification/018-sparse-per-tensor.md
                   docs/verification/019-sparse-per-channel.md
"""
import pytest
import torch

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.quantize.elemwise import quantize


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: GranularitySpec construction — validate outlier_ratio is no longer
#          restricted to PER_BLOCK only
# ═══════════════════════════════════════════════════════════════════════════════

class TestGranularitySpecOutlierRatio:
    """outlier_ratio > 0 should be valid for ALL granularity modes."""

    def test_per_tensor_with_outlier_ratio_constructs(self):
        """PER_TENSOR + outlier_ratio > 0 should construct without error."""
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        assert g.outlier_ratio == 0.1
        assert g.mode == GranularityMode.PER_TENSOR

    def test_per_channel_with_outlier_ratio_constructs(self):
        """PER_CHANNEL + outlier_ratio > 0 should construct without error."""
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.05)
        assert g.outlier_ratio == 0.05
        assert g.mode == GranularityMode.PER_CHANNEL

    def test_per_block_with_outlier_ratio_still_works(self):
        """PER_BLOCK + outlier_ratio > 0 should still work."""
        g = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=32,
                            outlier_ratio=0.1)
        assert g.outlier_ratio == 0.1
        assert g.mode == GranularityMode.PER_BLOCK

    def test_outlier_ratio_zero_all_modes(self):
        """outlier_ratio=0 should work for all modes (no-op sparse)."""
        for mode in [GranularityMode.PER_TENSOR, GranularityMode.PER_CHANNEL]:
            g = GranularitySpec(mode=mode, block_size=32 if mode == GranularityMode.PER_BLOCK else 0,
                                channel_axis=0, block_axis=-1,
                                outlier_ratio=0.0)
            assert g.outlier_ratio == 0.0

    def test_outlier_ratio_range_validation(self):
        """outlier_ratio must be in [0, 1]."""
        with pytest.raises(ValueError, match="outlier_ratio must be in"):
            GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=1.5)
        with pytest.raises(ValueError, match="outlier_ratio must be in"):
            GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=-0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Per-Tensor Sparse — bit-exact verification against hand-derived values
#          See docs/verification/018-sparse-per-tensor.md
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerTensorSparseBitExact:
    """Bit-exact verification of per_tensor quantization with outlier_ratio > 0.

    Expected values are hand-derived in docs/verification/018-sparse-per-tensor.md.
    """

    @pytest.fixture
    def x(self):
        return torch.tensor([0.5, 1.0, 10.0, 0.25])

    @pytest.fixture
    def fmt(self):
        return FormatBase.from_str("int4")

    def test_per_tensor_no_sparse_baseline(self, x, fmt):
        """Baseline: per_tensor int8 without sparse."""
        g = GranularitySpec.per_tensor()
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        expected = torch.tensor([0.0, 2.0, 10.0, 0.0])
        assert torch.equal(result, expected), \
            f"Baseline mismatch: got {result}, expected {expected}"

    def test_per_tensor_sparse_bit_exact(self, x, fmt):
        """Per-tensor int8 + outlier_ratio=0.25 produces hand-derived values."""
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        expected = torch.tensor([0.5, 1.0, 10.0, 0.25])
        assert torch.equal(result, expected), \
            f"Sparse mismatch: got {result}, expected {expected}"

    def test_per_tensor_sparse_improves_small_values(self, x, fmt):
        """Sparse should preserve small values that would be crushed without it."""
        g_no_sparse = GranularitySpec.per_tensor()
        g_sparse = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)

        scheme_no = QuantScheme(format=fmt, granularity=g_no_sparse, scale_storage="pot")
        scheme_sparse = QuantScheme(format=fmt, granularity=g_sparse, scale_storage="pot")

        result_no = quantize(x, scheme_no)
        result_sparse = quantize(x, scheme_sparse)

        # Small values (0.5, 0.25) are crushed to 0 without sparse, preserved with sparse
        assert result_no[0] == 0.0
        assert result_sparse[0] == 0.5
        assert result_no[3] == 0.0
        assert result_sparse[3] == 0.25

    def test_per_tensor_sparse_degenerates_when_k_exceeds_numel(self, x, fmt):
        """When k >= numel, sparse degenerates to standard per_tensor."""
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=1.0)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        # k = max(1, int(4 * 1.0)) = 4 >= 4 → degenerate to single-group
        # Should match non-sparse result
        g_normal = GranularitySpec.per_tensor()
        scheme_normal = QuantScheme(format=fmt, granularity=g_normal, scale_storage="pot")
        expected = quantize(x, scheme_normal)
        assert torch.equal(result, expected), \
            f"Degenerate sparse should match non-sparse: got {result}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Per-Channel Sparse — bit-exact verification against hand-derived values
#          See docs/verification/019-sparse-per-channel.md
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerChannelSparseBitExact:
    """Bit-exact verification of per_channel quantization with outlier_ratio > 0.

    Expected values are hand-derived in docs/verification/019-sparse-per-channel.md.
    """

    @pytest.fixture
    def x(self):
        return torch.tensor([[0.5, 10.0, 0.25, 1.0],
                             [2.0, -8.0, 3.0, 1.5]])

    @pytest.fixture
    def fmt(self):
        return FormatBase.from_str("int4")

    def test_per_channel_no_sparse_baseline(self, x, fmt):
        """Baseline: per_channel int8 without sparse."""
        g = GranularitySpec.per_channel(axis=0)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        expected = torch.tensor([[0.0, 10.0, 0.0, 2.0],
                                 [2.0, -8.0, 4.0, 2.0]])
        assert torch.equal(result, expected), \
            f"Baseline mismatch: got {result}, expected {expected}"

    def test_per_channel_sparse_bit_exact(self, x, fmt):
        """Per-channel int8 + outlier_ratio=0.25 produces hand-derived values."""
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        expected = torch.tensor([[0.5, 10.0, 0.25, 1.0],
                                 [2.0, -8.0, 3.0, 2.0]])
        assert torch.equal(result, expected), \
            f"Sparse mismatch: got {result}, expected {expected}"

    def test_per_channel_sparse_improves_per_channel(self, x, fmt):
        """Each channel independently benefits from outlier isolation."""
        g_no = GranularitySpec.per_channel(axis=0)
        g_sparse = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                                   outlier_ratio=0.25)

        scheme_no = QuantScheme(format=fmt, granularity=g_no, scale_storage="pot")
        scheme_sparse = QuantScheme(format=fmt, granularity=g_sparse, scale_storage="pot")

        result_no = quantize(x, scheme_no)
        result_sparse = quantize(x, scheme_sparse)

        # Channel 0: small values were crushed
        assert result_no[0, 0] == 0.0
        assert result_sparse[0, 0] == 0.5
        assert result_no[0, 2] == 0.0
        assert result_sparse[0, 2] == 0.25

        # Channel 1: 3.0 was pushed to 4.0 (over-estimated), now closer
        assert result_no[1, 2] == 4.0
        assert result_sparse[1, 2] == 3.0

    def test_per_channel_sparse_different_axis(self, fmt):
        """Per-channel sparse works with channel_axis=-1 (last dim)."""
        x = torch.tensor([[0.5, 10.0], [0.25, 1.0]])  # 2 channels on axis=-1
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=-1,
                            outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        # Should not error and produce finite output
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_per_channel_sparse_degenerates_when_k_exceeds(self, x, fmt):
        """When k >= elements per channel, sparse degenerates to per_channel."""
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=1.0)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)

        g_normal = GranularitySpec.per_channel(axis=0)
        scheme_normal = QuantScheme(format=fmt, granularity=g_normal, scale_storage="pot")
        expected = quantize(x, scheme_normal)

        assert torch.equal(result, expected), \
            f"Degenerate sparse should match non-sparse"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Shape preservation and edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestSparseShapeAndEdgeCases:
    """Shape preservation and edge case handling for sparse quantize."""

    @pytest.mark.parametrize("shape", [
        (8,),           # 1D per_tensor
        (4, 16),        # 2D per_tensor
        (2, 3, 8),      # 3D
    ])
    def test_shape_preserved_per_tensor_sparse(self, shape):
        """Output shape matches input shape for various ranks."""
        torch.manual_seed(42)
        x = torch.randn(*shape)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        assert result.shape == x.shape

    @pytest.mark.parametrize("shape", [
        (4, 16),        # 2D per_channel axis=0
        (2, 3, 8),      # 3D per_channel
    ])
    def test_shape_preserved_per_channel_sparse(self, shape):
        """Output shape matches input shape for per_channel sparse."""
        torch.manual_seed(42)
        x = torch.randn(*shape)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.05)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        assert result.shape == x.shape

    def test_zero_input_sparse(self):
        """Zero input → zero output for all sparse variants."""
        x = torch.zeros(4, 8)
        fmt = FormatBase.from_str("int8")

        for g in [
            GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1),
            GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0, outlier_ratio=0.1),
        ]:
            scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
            result = quantize(x, scheme)
            assert (result == 0).all(), f"Zero input not preserved for {g}"

    def test_all_identical_values_sparse(self):
        """When all values are identical, top-k is arbitrary but result is correct."""
        x = torch.ones(2, 8)
        fmt = FormatBase.from_str("int8")

        for g in [
            GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25),
            GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0, outlier_ratio=0.25),
        ]:
            scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
            result = quantize(x, scheme)
            assert torch.isfinite(result).all()
            # All values should be equal (uniform quantization)
            assert (result == result.flatten()[0]).all(), \
                f"All-one input should produce uniform output for {g}"

    @pytest.mark.parametrize("fmt_name", ["int8", "int4", "fp8_e4m3"])
    def test_sparse_with_different_formats(self, fmt_name):
        """Sparse per_tensor works with various formats."""
        torch.manual_seed(42)
        x = torch.randn(16)
        fmt = FormatBase.from_str(fmt_name)
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.2)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: API contract — scale / allow_denorm / float format / scale_storage
# ═══════════════════════════════════════════════════════════════════════════════

class TestSparseContract:
    """API contract tests for sparse paths: error cases, float formats, fp32 storage."""

    def test_scale_with_sparse_per_tensor_falls_through(self):
        """Passing scale to quantize() with outlier_ratio > 0 falls through to dynamic sparse.

        Calibrated modules pass _output_scale without a static mask; the scale is
        ignored and the dynamic sparse path recomputes amax via topk.
        """
        x = torch.randn(8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, scale=torch.tensor(1.0))
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_scale_with_sparse_per_channel_falls_through(self):
        """Passing scale to quantize() with per_channel + outlier_ratio > 0 falls through to dynamic sparse."""
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, scale=torch.ones(4, 1))
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_scale_none_with_sparse_ok(self):
        """scale=None (dynamic) with sparse should work fine."""
        x = torch.randn(8)
        fmt = FormatBase.from_str("int4")
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, scale=None)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_scale_storage_fp32_sparse_per_tensor(self):
        """scale_storage='fp32' with sparse per_tensor produces finite correct-shaped output."""
        torch.manual_seed(42)
        x = torch.randn(16)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.2)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="fp32")
        result = quantize(x, scheme)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
        # With fp32 storage, amax is not POT-rounded — values may differ from pot
        # but the result should still be a valid quantization.

    def test_scale_storage_fp32_sparse_per_channel(self):
        """scale_storage='fp32' with sparse per_channel produces finite correct-shaped output."""
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("int8")
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                            outlier_ratio=0.1)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="fp32")
        result = quantize(x, scheme)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_float_format_sparse_degenerates_to_non_sparse(self):
        """Float formats (ebits > 0) with sparse should delegate to non-sparse path."""
        torch.manual_seed(42)
        x = torch.randn(16)
        fmt = FormatBase.from_str("fp8_e4m3")
        g_sparse = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        g_normal = GranularitySpec.per_tensor()

        scheme_sparse = QuantScheme(format=fmt, granularity=g_sparse, scale_storage="pot")
        scheme_normal = QuantScheme(format=fmt, granularity=g_normal, scale_storage="pot")

        result_sparse = quantize(x, scheme_sparse)
        result_normal = quantize(x, scheme_normal)
        assert torch.equal(result_sparse, result_normal), \
            f"Float format sparse should match non-sparse"

    def test_float_format_sparse_per_channel_degenerates(self):
        """Float format per_channel sparse should delegate to non-sparse per_channel."""
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        fmt = FormatBase.from_str("fp8_e4m3")
        g_sparse = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                                    outlier_ratio=0.25)
        g_normal = GranularitySpec.per_channel(axis=0)

        scheme_sparse = QuantScheme(format=fmt, granularity=g_sparse, scale_storage="pot")
        scheme_normal = QuantScheme(format=fmt, granularity=g_normal, scale_storage="pot")

        result_sparse = quantize(x, scheme_sparse)
        result_normal = quantize(x, scheme_normal)
        assert torch.equal(result_sparse, result_normal), \
            f"Float format per_channel sparse should match non-sparse"

    def test_allow_denorm_false_with_sparse(self):
        """allow_denorm=False threads through sparse path correctly."""
        x = torch.randn(8)
        fmt = FormatBase.from_str("int4")
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.25)
        scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")
        result = quantize(x, scheme, allow_denorm=False)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: Session integration — QuantConfig → Session resolves sparse correctly
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionSparseIntegration:
    """Session-level tests: QuantConfig with outlier_ratio → Session → quantize."""

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

    def test_quantconfig_per_tensor_sparse_roundtrips(self, simple_model):
        """QuantConfig with per_tensor + outlier_ratio produces valid qmodel."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        cfg = QuantConfig(
            name="test-sparse-pt",
            w_format="int8",
            w_granularity="per_tensor",
            a_granularity="per_tensor",
            outlier_ratio=0.1,
            scale_storage="pot",
            weight_only=False,
            quantize_nonlinear=True,
        ).to_op_config()

        x = torch.randn(3, 4)
        qmodel, fp32, result = run_quantization(
            simple_model, cfg, [x], keep_fp32=False,
        )
        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (3, 2)
        assert torch.isfinite(out).all()

    def test_quantconfig_per_channel_sparse_roundtrips(self, simple_model):
        """QuantConfig with per_channel + outlier_ratio produces valid qmodel."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        cfg = QuantConfig(
            name="test-sparse-pc",
            w_format="int8",
            w_granularity="per_channel",
            a_granularity="per_channel",
            outlier_ratio=0.05,
            scale_storage="pot",
            weight_only=False,
            quantize_nonlinear=True,
        ).to_op_config()

        x = torch.randn(3, 4)
        qmodel, fp32, result = run_quantization(
            simple_model, cfg, [x], keep_fp32=False,
        )
        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (3, 2)
        assert torch.isfinite(out).all()

    def test_to_op_config_passes_outlier_ratio(self):
        """QuantConfig.to_op_config() threads outlier_ratio to GranularitySpec."""
        from src.session._config import QuantConfig

        cfg = QuantConfig(
            w_format="int8",
            w_granularity="per_tensor",
            a_granularity="per_channel",
            outlier_ratio=0.15,
        )
        op_cfg = cfg.to_op_config()

        assert op_cfg.weight.granularity.outlier_ratio == 0.15
        assert op_cfg.input.granularity.outlier_ratio == 0.15

    def test_resolve_config_passes_outlier_ratio(self):
        """resolve_config() with outlier_ratio in descriptor works."""
        from src.session._config import resolve_config

        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "outlier_ratio": 0.2,
            "scale_format": "pot",
        }
        op_cfg = resolve_config(desc)
        assert op_cfg.weight.granularity.outlier_ratio == 0.2
