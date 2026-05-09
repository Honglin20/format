"""
NF4 golden verification tests.

Since mx has no NF4 implementation, we verify NF4Format against independent
golden references. The golden functions are cleanroom implementations — they
do NOT import from src/ and serve as the ground truth.

The NF4 levels are the standard values from the QLoRA paper (Dettmers et al.,
2023). See: https://arxiv.org/abs/2305.14314
"""
import pytest
import torch

from src.scheme.granularity import GranularitySpec

# ============================================================================
# QLoRA NF4 levels — the golden standard
# ============================================================================

# These 16 values are from the QLoRA paper and verified against bitsandbytes.
QLORA_NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)

# ============================================================================
# Helper
# ============================================================================


def _assert_tensor_equal(actual, expected, label="tensor"):
    """Bit-exact comparison handling NaN (torch.equal fails on NaN != NaN)."""
    nan_mask = torch.isnan(expected)
    if nan_mask.any():
        assert torch.isnan(actual[nan_mask]).all(), f"{label}: expected NaN but got finite"
        non_nan = ~nan_mask
        assert torch.equal(actual[non_nan], expected[non_nan]), (
            f"{label}: finite values differ\n"
            f"first diff idx: {(actual[non_nan] != expected[non_nan]).nonzero(as_tuple=True)[0][:5]}"
        )
    else:
        assert torch.equal(actual, expected), f"{label}: not bit-exact"

# ============================================================================
# Golden reference functions (NO imports from src/)
# ============================================================================


def golden_lut_quantize(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Independent nearest-neighbor LUT quantization.

    Values outside [-1, 1] are clamped. NaN inputs produce NaN outputs.
    Inf inputs are clamped to ±1.

    Args:
        x: Input tensor of any shape.
        levels: Sorted 1D tensor of quantization levels (must be in [-1, 1]).

    Returns:
        Quantized tensor with same shape and dtype as x.
    """
    levels = levels.to(dtype=x.dtype, device=x.device)
    max_norm = levels.abs().max()

    nan_mask = torch.isnan(x)
    x_safe = torch.where(nan_mask, torch.zeros_like(x), x)
    x_safe = torch.clamp(x_safe, -max_norm, max_norm)

    # Nearest-neighbor via argmin over last dim
    x_expanded = x_safe.unsqueeze(-1)       # (*, 1)
    levels_expanded = levels.view(
        *([1] * x_safe.ndim), -1
    )  # (1, ..., 1, L)
    distances = torch.abs(x_expanded - levels_expanded)
    indices = torch.argmin(distances, dim=-1)
    result = levels[indices]

    if nan_mask.any():
        result = result.clone()
        result[nan_mask] = float("nan")

    return result


def golden_per_channel_quantize(x: torch.Tensor, levels: torch.Tensor,
                                 axis: int) -> torch.Tensor:
    """Independent per-channel NF4 quantization.

    Computes per-channel amax via max(abs(x)), normalizes each channel,
    quantizes with LUT, and rescales.

    Args:
        x: Input tensor.
        levels: Quantization levels.
        axis: Channel axis.

    Returns:
        Quantized tensor with same shape as x.
    """
    levels = levels.to(dtype=x.dtype, device=x.device)
    axis = axis if axis >= 0 else x.ndim + axis

    # Per-channel amax
    dims = [i for i in range(x.ndim) if i != axis]
    amax = torch.amax(torch.abs(x), dim=tuple(dims), keepdim=True)
    amax = amax.clamp(min=1e-12)

    # Normalize → quantize → rescale
    x_norm = x / amax
    x_q_norm = golden_lut_quantize(x_norm, levels)
    return x_q_norm * amax


# ============================================================================
# Tests: Levels verification
# ============================================================================


class TestNF4LevelsGolden:
    """Verify NF4 levels match the QLoRA paper definition."""

    def test_levels_match_qlora_paper(self):
        """The 16 hardcoded NF4 levels must match the known QLoRA paper values."""
        from src.formats.lookup_formats import NF4Format

        nf4 = NF4Format()
        src_levels = nf4.levels.to(torch.float32)

        _assert_tensor_equal(src_levels, QLORA_NF4_LEVELS,
                            label="NF4 levels vs QLoRA paper")

    def test_level_count(self):
        from src.formats.lookup_formats import NF4Format
        assert NF4Format().levels.numel() == 16

    def test_level_asymmetry(self):
        """Must have 7 negative + 1 zero + 8 positive = 16 levels."""
        from src.formats.lookup_formats import NF4Format
        lv = NF4Format().levels
        assert (lv < 0).sum().item() == 7
        assert (lv == 0).sum().item() == 1
        assert (lv > 0).sum().item() == 8

    def test_levels_sorted(self):
        from src.formats.lookup_formats import NF4Format
        lv = NF4Format().levels
        assert (lv[1:] - lv[:-1] >= 0).all()

    def test_levels_bounds(self):
        from src.formats.lookup_formats import NF4Format
        lv = NF4Format().levels
        assert lv[0].item() == -1.0
        assert lv[-1].item() == 1.0


# ============================================================================
# Tests: quantize_elemwise golden
# ============================================================================


class TestNF4QuantizeElemwiseGolden:
    """Verify NF4Format.quantize_elemwise() against independent golden."""

    def test_matches_golden_random_tensors(self):
        """20 random tensors with NaN/Inf must match golden bit-exact."""
        from src.formats.lookup_formats import NF4Format
        nf4 = NF4Format()

        torch.manual_seed(42)
        for i in range(20):
            x = torch.randn(500, dtype=torch.float32) * 2.0
            # Inject NaN and Inf at deterministic positions
            x[i * 3 % 500] = float("nan")
            x[(i * 7 + 1) % 500] = float("inf")
            x[(i * 11 + 2) % 500] = float("-inf")

            src_result = nf4.quantize_elemwise(x)
            golden_result = golden_lut_quantize(x, nf4.levels)

            _assert_tensor_equal(src_result, golden_result,
                               label=f"quantize_elemwise seed={i}")

    def test_exact_levels_map_to_themselves(self):
        """Each NF4 level, when quantized, must map to itself."""
        from src.formats.lookup_formats import NF4Format
        nf4 = NF4Format()
        x = nf4.levels.clone()
        result = nf4.quantize_elemwise(x)
        _assert_tensor_equal(result, x, label="vs zero")

    def test_all_results_are_valid_levels(self):
        """Every quantized value must be one of the NF4 levels."""
        from src.formats.lookup_formats import NF4Format
        nf4 = NF4Format()
        x = torch.randn(1000) * 3.0
        result = nf4.quantize_elemwise(x)
        levels = nf4.levels
        for i, val in enumerate(result):
            d = (val - levels).abs().min()
            assert d < 1e-7, f"result[{i}]={val:.10f} not in levels (min dist={d:.2e})"

    def test_nan_preserved(self):
        from src.formats.lookup_formats import NF4Format
        nf4 = NF4Format()
        x = torch.tensor([0.5, float("nan"), -0.3])
        result = nf4.quantize_elemwise(x)
        assert not torch.isnan(result[0])
        assert torch.isnan(result[1])
        assert not torch.isnan(result[2])

    def test_inf_clamped(self):
        from src.formats.lookup_formats import NF4Format
        nf4 = NF4Format()
        x = torch.tensor([float("inf"), float("-inf")])
        result = nf4.quantize_elemwise(x)
        assert result[0].item() == 1.0
        assert result[1].item() == -1.0

    def test_out_of_range_clamped(self):
        """Values outside [-1, 1] should be clamped."""
        from src.formats.lookup_formats import NF4Format
        nf4 = NF4Format()
        x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.5])
        result = nf4.quantize_elemwise(x)
        golden = golden_lut_quantize(torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.5]),
                                      nf4.levels)
        assert torch.equal(result, golden)


# ============================================================================
# Tests: per_tensor golden
# ============================================================================


class TestNF4PerTensorGolden:
    """Verify full quantize() path with per_tensor granularity."""

    def test_matches_golden(self):
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize
        from src.formats.lookup_formats import NF4Format

        nf4 = NF4Format()
        torch.manual_seed(42)
        x = torch.randn(4, 8) * 0.5
        scheme = QuantScheme.per_tensor("nf4")

        src_result = quantize(x, scheme)
        golden_result = golden_lut_quantize(x, nf4.levels)
        assert torch.equal(src_result, golden_result)


# ============================================================================
# Tests: per_channel golden
# ============================================================================


class TestNF4PerChannelGolden:
    """Verify full quantize() path with per_channel granularity."""

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_matches_golden(self, axis):
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize
        from src.formats.lookup_formats import NF4Format

        nf4 = NF4Format()
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        scheme = QuantScheme(format="nf4",
                             granularity=GranularitySpec.per_channel(axis=axis),
                             scale_storage="fp32")

        src_result = quantize(x, scheme)
        golden_result = golden_per_channel_quantize(x, nf4.levels, axis)
        assert torch.equal(src_result, golden_result), (
            f"per_channel axis={axis}: src vs golden mismatch"
        )

    def test_differs_from_per_tensor(self):
        """Per-channel quantization should differ from per-tensor for 2D input."""
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        torch.manual_seed(42)
        x = torch.randn(4, 8)
        r_pt = quantize(x, QuantScheme.per_tensor("nf4"))
        r_pc = quantize(x, QuantScheme.per_channel("nf4", axis=1))
        assert not torch.equal(r_pt, r_pc)

    def test_scale_kwarg(self):
        """Per-channel with pre-computed scale matches auto-computed."""
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        torch.manual_seed(42)
        x = torch.randn(4, 8)
        scheme = QuantScheme.per_channel("nf4", axis=-1)
        dims = tuple(i for i in range(x.ndim) if i != (x.ndim - 1))
        amax = torch.amax(torch.abs(x), dim=dims, keepdim=True).clamp(min=1e-12)

        r_auto = quantize(x, scheme)
        r_scaled = quantize(x, scheme, scale=amax)
        assert torch.equal(r_auto, r_scaled)


# ============================================================================
# Tests: per_block structural validation
# ============================================================================


class TestNF4PerBlockGolden:
    """Structural validation of per_block NF4 quantization."""

    @pytest.mark.parametrize("block_size", [8, 16, 32])
    def test_shape_and_finite(self, block_size):
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        torch.manual_seed(42)
        x = torch.randn(4, 64)
        scheme = QuantScheme.mxfp("nf4", block_size=block_size)

        result = quantize(x, scheme)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_blocks_differ(self):
        """Per-block quantization should produce per-block structure."""
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        torch.manual_seed(42)
        x = torch.randn(2, 32)
        scheme = QuantScheme.mxfp("nf4", block_size=8)
        result = quantize(x, scheme)

        # Different blocks should have different scales (shared exponents),
        # so the result should differ from per-tensor quantization.
        r_pt = quantize(x, QuantScheme.per_tensor("nf4"))
        assert not torch.equal(result, r_pt)


# ============================================================================
# Tests: Edge case propagation
# ============================================================================


class TestNF4EdgeCases:
    """NaN and extreme-value propagation through the quantization pipeline."""

    def test_per_tensor_zero_input(self):
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        x = torch.zeros(4, 8)
        result = quantize(x, QuantScheme.per_tensor("nf4"))
        _assert_tensor_equal(result, x, label="vs zero")

    def test_nan_propagates_per_channel(self):
        """NaN in a channel makes entire channel NaN (per-channel amax = NaN)."""
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        torch.manual_seed(42)
        x = torch.randn(2, 5)
        x[1, 2] = float("nan")
        scheme = QuantScheme.per_channel("nf4", axis=1)

        result = quantize(x, scheme)
        # Channel 2's amax is NaN → all rows in channel 2 are NaN
        assert torch.isfinite(result[0, [0, 1, 3, 4]]).all()
        assert torch.isnan(result[0, 2])
        assert torch.isnan(result[1, 2])

    def test_nan_propagates_per_block(self):
        """NaN in a block propagates to entire block."""
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        torch.manual_seed(42)
        x = torch.randn(2, 32)
        x[0, 5] = float("nan")
        scheme = QuantScheme.mxfp("nf4", block_size=8)

        result = quantize(x, scheme)
        # Block 0 (positions 0-7) gets NaN from shared_exp
        assert torch.isnan(result[0, :8]).all()
        # Other blocks clean
        assert torch.isfinite(result[0, 8:]).all()
        assert torch.isfinite(result[1, :]).all()
