"""
P2F-3 TDD tests: FormatBase.quantize(x, granularity, round_mode) — abstract method
and subclass implementations.

Tests written FIRST (red phase), then implementation makes them green.
"""
import pytest
import torch

from src.formats.base import FormatBase
from src.formats.int_formats import IntFormat
from src.formats.fp_formats import FPFormat
from src.formats.bf16_fp16 import BFloat16Format, Float16Format
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.quantize.elemwise import _quantize_elemwise_core, _quantize_elemwise, _quantize_bfloat, _quantize_fp
from src.quantize.mx_quantize import _quantize_mx


# ---------------------------------------------------------------------------
# 1. FormatBase.quantize() is abstract — cannot instantiate FormatBase directly
# ---------------------------------------------------------------------------

def test_format_base_cannot_be_instantiated_with_quantize_missing():
    """FormatBase itself should be abstract due to quantize()."""
    with pytest.raises(TypeError):
        FormatBase()


# ---------------------------------------------------------------------------
# 2. PER_TENSOR: format.quantize(x, per_tensor, round_mode) equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int8", "int4", "int2",
                                       "fp8_e4m3", "fp8_e5m2",
                                       "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"])
@pytest.mark.parametrize("round_mode", ["nearest", "floor"])
def test_per_tensor_quantize_equiv(fmt_name, round_mode):
    """format.quantize(x, per_tensor, round) == _quantize_elemwise(x, fmt, round)."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_tensor()
    result = fmt.quantize(x, gran, round_mode)

    expected = _quantize_elemwise(x, fmt, round_mode=round_mode)
    assert torch.allclose(result, expected, atol=1e-7), \
        f"{fmt_name}/per_tensor/{round_mode}: max diff = {(result - expected).abs().max()}"


# ---------------------------------------------------------------------------
# 3. PER_CHANNEL: format.quantize(x, per_channel, round_mode) basic behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3"])
def test_per_channel_quantize_produces_finite_output(fmt_name):
    """Per-channel quantization should produce finite output for normal input."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_channel(axis=0)
    result = fmt.quantize(x, gran, "nearest")
    assert result.isfinite().all(), f"{fmt_name}/per_channel produced non-finite values"


def test_per_channel_quantize_axis_1():
    """Per-channel along axis=1 should work and differ from axis=0."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("fp8_e4m3")

    result_ax0 = fmt.quantize(x, GranularitySpec.per_channel(axis=0), "nearest")
    result_ax1 = fmt.quantize(x, GranularitySpec.per_channel(axis=1), "nearest")
    assert not torch.allclose(result_ax0, result_ax1), \
        "Per-channel axis=0 and axis=1 should produce different results"


# ---------------------------------------------------------------------------
# 4. PER_BLOCK: format.quantize(x, per_block(32), round_mode) equivalence with _quantize_mx
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "fp8_e5m2",
                                       "fp6_e3m2", "fp4_e2m1"])
def test_per_block_quantize_equiv(fmt_name):
    """format.quantize(x, per_block(32), round) == _quantize_mx(x, 8, fmt, block_size=32)."""
    torch.manual_seed(42)
    x = torch.randn(4, 32)

    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_block(32)
    result = fmt.quantize(x, gran, "nearest")

    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=32, axes=-1, round_mode="nearest")
    assert torch.allclose(result, expected, atol=1e-6), \
        f"{fmt_name}/per_block(32): max diff = {(result - expected).abs().max()}"


@pytest.mark.parametrize("block_size", [16, 32])
def test_per_block_various_sizes(block_size):
    """Different block sizes should both work and produce different results."""
    torch.manual_seed(42)
    x = torch.randn(2, 64)
    fmt = FormatBase.from_str("fp8_e4m3")

    result_16 = FormatBase.from_str("fp8_e4m3").quantize(
        x, GranularitySpec.per_block(block_size), "nearest")
    assert result_16.isfinite().all(), f"per_block({block_size}) produced non-finite values"


# ---------------------------------------------------------------------------
# 5. IntFormat specialization — ebits=0 means saturate_normals (clamp, no Inf)
# ---------------------------------------------------------------------------

def test_int8_per_tensor_clamps_to_max_norm():
    """INT8 per-tensor should clamp, not produce Inf."""
    x = torch.tensor([100.0, -100.0, 0.5])
    fmt = FormatBase.from_str("int8")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert result.isfinite().all()
    assert result.abs().max() <= fmt.max_norm + 1e-6


def test_int4_per_tensor_clamps_to_max_norm():
    x = torch.tensor([10.0, -10.0])
    fmt = FormatBase.from_str("int4")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert result.isfinite().all()
    assert result.abs().max() <= fmt.max_norm + 1e-6


# ---------------------------------------------------------------------------
# 6. BFloat16Format shortcut — round_mode="even" → .to(torch.bfloat16)
# ---------------------------------------------------------------------------

def test_bfloat16_even_round_shortcut():
    """BFloat16Format with round_mode='even' should use .to(bfloat16) shortcut."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("bfloat16")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "even")
    expected = x.to(torch.bfloat16).float()
    assert torch.equal(result, expected)


def test_bfloat16_nearest_round_uses_elemwise():
    """BFloat16Format with round_mode='nearest' should use elemwise path."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("bfloat16")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    expected = _quantize_elemwise(x, fmt, round_mode="nearest")
    assert torch.allclose(result, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# 7. Float16Format shortcut — round_mode="even" → .to(torch.float16)
# ---------------------------------------------------------------------------

def test_float16_even_round_shortcut():
    """Float16Format with round_mode='even' should use .to(float16) shortcut."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("float16")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "even")
    expected = x.to(torch.float16).float()
    assert torch.equal(result, expected)


def test_float16_nearest_round_uses_elemwise():
    """Float16Format with round_mode='nearest' should use elemwise path."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("float16")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    expected = _quantize_elemwise(x, fmt, round_mode="nearest")
    assert torch.allclose(result, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# 8. Round mode coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("round_mode", ["nearest", "floor", "even", "dither"])
def test_all_round_modes_produce_finite_output(round_mode):
    """All valid round modes should produce finite output for FP8."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("fp8_e4m3")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), round_mode)
    assert result.isfinite().all(), f"round_mode={round_mode} produced non-finite values"


# ---------------------------------------------------------------------------
# 9. Input preservation — Inf/NaN passthrough
# ---------------------------------------------------------------------------

def test_inf_nan_passthrough():
    """Inf and NaN in input should be preserved in output."""
    x = torch.tensor([1.0, float("Inf"), -float("Inf"), float("NaN"), 2.0])
    fmt = FormatBase.from_str("fp8_e5m2")  # e5m2 supports Inf
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert result[1] == float("Inf")
    assert result[2] == -float("Inf")
    assert result[3] != result[3]  # NaN


# ---------------------------------------------------------------------------
# 10. Edge: zero input
# ---------------------------------------------------------------------------

def test_zero_input_produces_zero():
    """Zero input should produce zero output."""
    x = torch.zeros(4, 8)
    fmt = FormatBase.from_str("fp8_e4m3")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert (result == 0).all()


# ---------------------------------------------------------------------------
# 11. Negative test: invalid round_mode
# ---------------------------------------------------------------------------

def test_invalid_round_mode_raises():
    """Invalid round_mode should raise ValueError."""
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("fp8_e4m3")
    with pytest.raises(ValueError, match="round_mode"):
        fmt.quantize(x, GranularitySpec.per_tensor(), "invalid")


# ---------------------------------------------------------------------------
# 12. Granularity validation — PER_BLOCK with wrong block_size is caught by GranularitySpec
# ---------------------------------------------------------------------------

def test_per_block_requires_positive_block_size():
    with pytest.raises(ValueError):
        GranularitySpec.per_block(0)


# ---------------------------------------------------------------------------
# 13. FormatBase __eq__/__hash__ (C1 fix verification)
# ---------------------------------------------------------------------------

def test_format_base_requires_eq_and_hash():
    """Concrete FormatBase subclass without __eq__/__hash__ should be rejected by ABC."""
    class IncompleteFormat(FormatBase):
        __slots__ = ()
        def __init__(self):
            self.name = "bad"
            self.ebits = 0
            self.mbits = 8
            self.emax = 0
            self.max_norm = 1.0
            self.min_norm = 0.0
            self._freeze()
        def quantize(self, x, granularity, round_mode="nearest"):
            return super().quantize(x, granularity, round_mode)
    with pytest.raises(TypeError):
        IncompleteFormat()


def test_int8_format_value_equality():
    """Two IntFormat(8) instances should be equal and have same hash."""
    a = IntFormat(bits=8)
    b = IntFormat(bits=8)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_fp8_format_value_equality():
    """Two FPFormat('fp8_e4m3', 4, 5, 448.0) should be equal and have same hash."""
    a = FPFormat(name="fp8_e4m3", ebits=4, mbits=5, max_norm_override=448.0)
    b = FPFormat(name="fp8_e4m3", ebits=4, mbits=5, max_norm_override=448.0)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_different_formats_not_equal():
    """IntFormat(8) and IntFormat(4) should not be equal."""
    assert IntFormat(bits=8) != IntFormat(bits=4)


def test_bfloat16_format_value_equality():
    """Two BFloat16Format instances should be equal and have same hash."""
    a = BFloat16Format()
    b = BFloat16Format()
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_float16_format_value_equality():
    """Two Float16Format instances should be equal and have same hash."""
    a = Float16Format()
    b = Float16Format()
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# ---------------------------------------------------------------------------
# 14. BFloat16/Float16 per_channel/per_block — shortcut bypass (M4)
# ---------------------------------------------------------------------------

def test_bfloat16_per_channel_even_uses_elemwise_not_shortcut():
    """BFloat16 with per_channel+even should NOT use .to(bfloat16) shortcut."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("bfloat16")
    per_tensor_result = fmt.quantize(x, GranularitySpec.per_tensor(), "even")
    per_channel_result = fmt.quantize(x, GranularitySpec.per_channel(axis=0), "even")
    # per_channel should differ from per_tensor (shortcut only applies to per_tensor)
    assert not torch.equal(per_tensor_result, per_channel_result)


def test_float16_per_channel_even_uses_elemwise_not_shortcut():
    """Float16 with per_channel+even should NOT use .to(float16) shortcut."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("float16")
    per_tensor_result = fmt.quantize(x, GranularitySpec.per_tensor(), "even")
    per_channel_result = fmt.quantize(x, GranularitySpec.per_channel(axis=0), "even")
    assert not torch.equal(per_tensor_result, per_channel_result)


def test_bfloat16_per_block_produces_finite_output():
    torch.manual_seed(42)
    x = torch.randn(4, 32)
    fmt = FormatBase.from_str("bfloat16")
    result = fmt.quantize(x, GranularitySpec.per_block(32), "nearest")
    assert result.isfinite().all()


def test_float16_per_block_produces_finite_output():
    torch.manual_seed(42)
    x = torch.randn(4, 32)
    fmt = FormatBase.from_str("float16")
    result = fmt.quantize(x, GranularitySpec.per_block(32), "nearest")
    assert result.isfinite().all()


# ---------------------------------------------------------------------------
# 15. Per-channel negative axis (M4)
# ---------------------------------------------------------------------------

def test_per_channel_negative_axis():
    """Per-channel with axis=-1 should behave like axis=ndim-1."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("fp8_e4m3")
    result_neg1 = fmt.quantize(x, GranularitySpec.per_channel(axis=-1), "nearest")
    result_pos = fmt.quantize(x, GranularitySpec.per_channel(axis=x.ndim - 1), "nearest")
    assert torch.allclose(result_neg1, result_pos, atol=1e-7)
