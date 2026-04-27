"""
Tests for P3: LookupFormat and NF4Format — lookup-table-based quantization.

NF4 from QLoRA (Dettmers et al., 2023): 16 asymmetric levels optimized
for normal distributions, quantized by nearest-neighbor LUT search.
"""
import pytest
import torch

from src.formats.base import FormatBase
from src.formats.lookup_formats import LookupFormat, NF4Format
from src.formats.int_formats import IntFormat
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularityMode


# ---------------------------------------------------------------------------
# 1. NF4Format construction and properties
# ---------------------------------------------------------------------------

def test_nf4_constructor_name():
    fmt = NF4Format()
    assert fmt.name == "nf4"


def test_nf4_constructor_ebits_mbits():
    fmt = NF4Format()
    assert fmt.ebits == 0
    assert fmt.mbits == 4


def test_nf4_constructor_emax_min_norm():
    fmt = NF4Format()
    assert fmt.emax == 0
    assert fmt.min_norm == 0.0


def test_nf4_max_norm_is_one():
    fmt = NF4Format()
    assert fmt.max_norm == 1.0


def test_nf4_is_integer():
    fmt = NF4Format()
    assert fmt.is_integer is True


def test_nf4_levels_count():
    fmt = NF4Format()
    assert fmt.levels.numel() == 16


def test_nf4_levels_sorted():
    fmt = NF4Format()
    diffs = fmt.levels[1:] - fmt.levels[:-1]
    assert (diffs >= 0).all(), "levels must be non-decreasing"


def test_nf4_levels_bounds():
    fmt = NF4Format()
    assert fmt.levels[0].item() == -1.0
    assert fmt.levels[-1].item() == 1.0


def test_nf4_levels_zero_included():
    fmt = NF4Format()
    assert (fmt.levels == 0.0).any(), "zero must be a quantization level"


# ---------------------------------------------------------------------------
# 2. Generic LookupFormat construction
# ---------------------------------------------------------------------------

def test_lookup_format_constructor():
    levels = [-0.5, 0.0, 0.5, 1.0]
    fmt = LookupFormat("test_lut", levels=levels)
    assert fmt.name == "test_lut"
    assert fmt.ebits == 0
    # 4 levels → 3 representable intervals, log2(3)=2 bits
    assert fmt.mbits == 2
    assert fmt.max_norm == 1.0
    assert torch.equal(fmt.levels, torch.tensor(levels, dtype=torch.float32))


def test_lookup_format_with_tensor_levels():
    levels = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
    fmt = LookupFormat("tensor_lut", levels=levels)
    assert torch.equal(fmt.levels, levels)


# ---------------------------------------------------------------------------
# 3. quantize_elemwise correctness
# ---------------------------------------------------------------------------

@pytest.fixture
def nf4():
    return NF4Format()


def test_nf4_elemwise_level_maps_to_itself(nf4):
    """Each NF4 level quantized maps to itself."""
    x = nf4.levels.clone()
    result = nf4.quantize_elemwise(x)
    assert torch.equal(result, x)


def test_nf4_elemwise_midpoints(nf4):
    """Values halfway between levels snap to the nearest."""
    levels = nf4.levels
    for i in range(len(levels) - 1):
        mid = (levels[i] + levels[i + 1]) / 2.0
        # Midpoint is equidistant → argmin returns first minimum (lower index)
        # Either level is acceptable; we test that the result IS one of the two
        result = nf4.quantize_elemwise(mid.unsqueeze(0))
        assert result.item() in (levels[i].item(), levels[i + 1].item()), \
            f"midpoint {mid:.6f} between [{levels[i]:.6f}, {levels[i+1]:.6f}] mapped to {result.item():.6f}"


def test_nf4_elemwise_close_to_level(nf4):
    """Value very close to a level maps to that level."""
    target = nf4.levels[5]
    eps = 1e-7
    result = nf4.quantize_elemwise((target + eps).unsqueeze(0))
    assert result.item() == target.item()


def test_nf4_elemwise_out_of_range_positive(nf4):
    """Values > 1.0 map to 1.0."""
    x = torch.tensor([1.5, 2.0, 100.0])
    result = nf4.quantize_elemwise(x)
    assert torch.equal(result, torch.ones(3))


def test_nf4_elemwise_out_of_range_negative(nf4):
    """Values < -1.0 map to -1.0."""
    x = torch.tensor([-1.5, -2.0, -100.0])
    result = nf4.quantize_elemwise(x)
    assert torch.equal(result, torch.full((3,), -1.0))


def test_nf4_elemwise_nan_preserved(nf4):
    """NaN input produces NaN output."""
    x = torch.tensor([0.5, float('nan'), -0.3])
    result = nf4.quantize_elemwise(x)
    assert result[0] != float('nan') and not torch.isnan(result[0])
    assert torch.isnan(result[1])
    assert not torch.isnan(result[2])


def test_nf4_elemwise_inf_saturated(nf4):
    """+Inf → 1.0, -Inf → -1.0."""
    x = torch.tensor([float('inf'), float('-inf')])
    result = nf4.quantize_elemwise(x)
    assert result[0].item() == 1.0
    assert result[1].item() == -1.0


def test_nf4_elemwise_preserves_shape(nf4):
    """Multi-dimensional input preserves shape."""
    x = torch.randn(3, 5, 7)
    result = nf4.quantize_elemwise(x)
    assert result.shape == (3, 5, 7)
    assert result.dtype == torch.float32


def test_nf4_elemwise_all_results_are_levels(nf4):
    """Every quantized value is one of the NF4 levels."""
    x = torch.randn(100)
    result = nf4.quantize_elemwise(x)
    levels = nf4.levels
    for i in range(100):
        assert (result[i].item() - levels).abs().min().item() < 1e-6, \
            f"result[{i}] = {result[i]:.6f} not in levels"


# ---------------------------------------------------------------------------
# 4. round_mode validation
# ---------------------------------------------------------------------------

def test_nf4_elemwise_rejects_floor(nf4):
    with pytest.raises(ValueError, match="round_mode"):
        nf4.quantize_elemwise(torch.tensor([0.5]), round_mode="floor")


def test_nf4_elemwise_rejects_even(nf4):
    with pytest.raises(ValueError, match="round_mode"):
        nf4.quantize_elemwise(torch.tensor([0.5]), round_mode="even")


def test_nf4_elemwise_rejects_dither(nf4):
    with pytest.raises(ValueError, match="round_mode"):
        nf4.quantize_elemwise(torch.tensor([0.5]), round_mode="dither")


def test_nf4_elemwise_accepts_nearest(nf4):
    """nearest is accepted (no error)."""
    result = nf4.quantize_elemwise(torch.tensor([0.5]), round_mode="nearest")
    assert result.numel() == 1


# ---------------------------------------------------------------------------
# 5. Full quantize() with granularity modes (via QuantScheme)
# ---------------------------------------------------------------------------

def test_nf4_per_tensor_quantize(nf4):
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_tensor("nf4")
    y = quantize_via_scheme(x, scheme)
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()


def test_nf4_per_channel_quantize_axis_0(nf4):
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_channel("nf4", axis=0)
    y = quantize_via_scheme(x, scheme)
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()


def test_nf4_per_channel_quantize_axis_1(nf4):
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_channel("nf4", axis=1)
    y = quantize_via_scheme(x, scheme)
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()


def test_nf4_per_channel_axis_0_differs_from_axis_1(nf4):
    """Different quantization axis produces different results."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    y0 = quantize_via_scheme(x, QuantScheme.per_channel("nf4", axis=0))
    y1 = quantize_via_scheme(x, QuantScheme.per_channel("nf4", axis=1))
    assert not torch.equal(y0, y1)


def test_nf4_per_block_quantize(nf4):
    torch.manual_seed(42)
    x = torch.randn(4, 32)
    scheme = QuantScheme.mxfp("nf4", block_size=8)
    y = quantize_via_scheme(x, scheme)
    assert y.shape == (4, 32)
    assert torch.isfinite(y).all()


def test_nf4_per_block_various_sizes(nf4):
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    for bs in [16, 32]:
        scheme = QuantScheme.mxfp("nf4", block_size=bs)
        y = quantize_via_scheme(x, scheme)
        assert y.shape == (4, 64)
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# 6. Registry
# ---------------------------------------------------------------------------

def test_nf4_in_registry():
    fmt = FormatBase.from_str("nf4")
    assert isinstance(fmt, NF4Format)
    assert fmt.name == "nf4"


def test_nf4_case_insensitive():
    fmt = FormatBase.from_str("NF4")
    assert isinstance(fmt, NF4Format)


# ---------------------------------------------------------------------------
# 7. Equality / Hashing
# ---------------------------------------------------------------------------

def test_nf4_equality():
    a = NF4Format()
    b = NF4Format()
    assert a == b
    assert hash(a) == hash(b)


def test_nf4_not_equal_to_int4():
    nf4 = NF4Format()
    int4 = IntFormat(4)
    assert nf4 != int4
    assert hash(nf4) != hash(int4)


def test_nf4_not_equal_to_lookup_format():
    nf4 = NF4Format()
    lut = LookupFormat("nf4", levels=NF4Format.NF4_LEVELS)
    # NF4Format uses isinstance check, LookupFormat does not
    assert nf4 != lut
    assert hash(nf4) != hash(lut)


def test_lookup_format_equality():
    a = LookupFormat("test", levels=[-0.5, 0.0, 0.5])
    b = LookupFormat("test", levels=[-0.5, 0.0, 0.5])
    assert a == b
    assert hash(a) == hash(b)


def test_lookup_format_different_levels_not_equal():
    a = LookupFormat("test", levels=[-0.5, 0.0, 0.5])
    b = LookupFormat("test", levels=[-0.5, 0.0, 1.0])
    assert a != b


def test_lookup_format_different_name_not_equal():
    a = LookupFormat("a", levels=[-0.5, 0.0, 0.5])
    b = LookupFormat("b", levels=[-0.5, 0.0, 0.5])
    assert a != b


# ---------------------------------------------------------------------------
# 8. Immutability
# ---------------------------------------------------------------------------

def test_nf4_frozen_name(nf4):
    with pytest.raises(AttributeError):
        nf4.name = "other"


def test_nf4_frozen_levels(nf4):
    with pytest.raises(AttributeError):
        nf4.levels = torch.zeros(8)


def test_nf4_no_dict(nf4):
    """NF4Format should not have a per-instance __dict__ (uses __slots__)."""
    assert not hasattr(nf4, '__dict__')


# ---------------------------------------------------------------------------
# 9. Edge cases through full quantize path
# ---------------------------------------------------------------------------

def test_nf4_zero_input_per_tensor(nf4):
    x = torch.zeros(4, 8)
    scheme = QuantScheme.per_tensor("nf4")
    y = quantize_via_scheme(x, scheme)
    assert torch.equal(y, x)  # zero maps to zero


def test_nf4_inf_nan_per_tensor(nf4):
    x = torch.tensor([[1.0, float('inf')], [float('-inf'), float('nan')]])
    scheme = QuantScheme.per_tensor("nf4")
    y = quantize_via_scheme(x, scheme)
    assert y[0, 0] != float('nan')
    assert y[0, 1].item() == 1.0  # +Inf → 1.0
    assert y[1, 0].item() == -1.0  # -Inf → -1.0
    assert torch.isnan(y[1, 1])  # NaN preserved


def test_nf4_inf_nan_per_channel(nf4):
    """NaN in channel makes entire channel NaN (amax becomes NaN)."""
    torch.manual_seed(42)
    x = torch.randn(2, 5)
    x[1, 2] = float('nan')
    scheme = QuantScheme.per_channel("nf4", axis=1)
    y = quantize_via_scheme(x, scheme)
    assert y.shape == (2, 5)
    # Row 0 has no NaN → all finite
    assert torch.isfinite(y[0, :]).all()
    # Row 1 has NaN at position 2, which corrupts amax → all NaN in row 1
    assert torch.isnan(y[1, :]).all()


def test_nf4_large_random_values_are_finite(nf4):
    """Quantization of large random values should produce finite output."""
    torch.manual_seed(123)
    x = torch.randn(10, 20) * 5.0  # values well outside [-1, 1]
    y = quantize_via_scheme(x, QuantScheme.per_tensor("nf4"))
    assert torch.isfinite(y).all()
    # All values should be nf4 levels (in [-1, 1])
    assert y.min() >= -1.0
    assert y.max() <= 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def quantize_via_scheme(x, scheme):
    """Helper to quantize via scheme.format.quantize() for granularity tests."""
    from src.quantize.elemwise import quantize
    return quantize(x, scheme)
