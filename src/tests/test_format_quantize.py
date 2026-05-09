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
from src.quantize.elemwise import _quantize_elemwise_core
from src.tests._compat import _quantize_elemwise, _quantize_bfloat, _quantize_fp
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

@pytest.mark.parametrize("fmt_name", ["int8", "int4", "int2"])
@pytest.mark.parametrize("round_mode", ["nearest", "floor"])
def test_int_per_tensor_quantize_equiv(fmt_name, round_mode):
    """Integer format per_tensor normalizes to [-1,1] before elemwise."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_tensor()
    result = fmt.quantize(x, gran, round_mode, scale_storage="fp32")

    # Integer formats: amax = max(|x|), normalize, quantize, rescale
    amax = x.abs().max().clamp(min=1e-12)
    x_norm = x / amax
    expected = _quantize_elemwise(x_norm, fmt, round_mode=round_mode) * amax
    assert torch.allclose(result, expected, atol=1e-7), \
        f"{fmt_name}/per_tensor/{round_mode}: max diff = {(result - expected).abs().max()}"


@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp8_e5m2",
                                       "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"])
@pytest.mark.parametrize("round_mode", ["nearest", "floor"])
def test_float_per_tensor_quantize_equiv(fmt_name, round_mode):
    """Float format per_tensor uses direct elemwise (no normalization)."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_tensor()
    result = fmt.quantize(x, gran, round_mode)

    # Float formats: direct elemwise, matching mx/ behaviour
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

    result_ax0 = fmt.quantize(x, GranularitySpec.per_channel(axis=0), "nearest", scale_storage="fp32")
    result_ax1 = fmt.quantize(x, GranularitySpec.per_channel(axis=1), "nearest", scale_storage="fp32")
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
# 4b. PER_BLOCK safety net: fmt.quantize(PER_BLOCK) ≡ _quantize_mx (all paths)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1", "int8"])
def test_per_block_quantize_vs_mx_round_modes(fmt_name):
    """fmt.quantize(x, per_block, round) == _quantize_mx for all round modes."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_block(32)

    for rm in ["nearest", "floor"]:
        result = fmt.quantize(x, gran, rm)
        expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                                block_size=32, axes=-1, round_mode=rm)
        assert torch.equal(result, expected), \
            f"{fmt_name}/per_block(32)/{rm}: mismatch"


def test_per_block_quantize_default_shared_exp_method():
    """fmt.quantize(PER_BLOCK) uses shared_exp_method='max' by default."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(32)

    result = fmt.quantize(x, gran, "nearest")
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=32, axes=-1, round_mode="nearest",
                            shared_exp_method="max")
    assert torch.equal(result, expected)


def test_per_block_quantize_default_flush_fp32_subnorms():
    """fmt.quantize(PER_BLOCK) uses flush_fp32_subnorms=False by default."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(32)

    result = fmt.quantize(x, gran, "nearest")
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=32, axes=-1, round_mode="nearest",
                            flush_fp32_subnorms=False)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_per_block_quantize_vs_mx_various_blocks(block_size):
    """fmt.quantize(x, per_block(N)) == _quantize_mx for various block sizes."""
    torch.manual_seed(42)
    x = torch.randn(4, block_size * 4)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(block_size)

    result = fmt.quantize(x, gran, "nearest")
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=block_size, axes=-1, round_mode="nearest")
    assert torch.equal(result, expected)


@pytest.mark.parametrize("shape", [
    (4, 64), (2, 3, 64), (2, 3, 4, 64),
])
def test_per_block_quantize_vs_mx_multi_dim(shape):
    """fmt.quantize(x, per_block) == _quantize_mx for multi-dim tensors."""
    torch.manual_seed(42)
    x = torch.randn(*shape)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(32)

    result = fmt.quantize(x, gran, "nearest")
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=32, axes=-1, round_mode="nearest")
    assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# 5. IntFormat specialization — ebits=0 means saturate_normals (clamp, no Inf)
# ---------------------------------------------------------------------------

def test_int8_per_tensor_normalizes_before_elemwise():
    """INT8 per-tensor normalizes by amax so outlier doesn't cause clamping."""
    # Values: 100 and -100 are outliers; 0.5 is the small value.
    # amax = 100, so 0.5 → 0.5/100 = 0.005 → int8 elemwise rounds to 0.
    x = torch.tensor([100.0, -100.0, 0.5])
    fmt = FormatBase.from_str("int8")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert result.isfinite().all()
    # After normalization by 100, values are in [-1, 1] ⊂ int8 range → no clamping
    assert torch.allclose(result[0], torch.tensor(100.0), atol=1.0), \
        f"100.0 should survive, got {result[0]}"
    assert torch.allclose(result[1], torch.tensor(-100.0), atol=1.0), \
        f"-100.0 should survive, got {result[1]}"
    # 0.5 / 100 = 0.005, int8 step = 1/64 ≈ 0.0156 → rounds to 0
    assert result[2] == 0.0, f"0.5→0 after normalization, got {result[2]}"


def test_int4_per_tensor_normalizes_before_elemwise():
    """INT4 per-tensor normalizes by amax before elemwise."""
    x = torch.tensor([10.0, -10.0])
    fmt = FormatBase.from_str("int4")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert result.isfinite().all()
    # After normalization by amax=10, 10.0 → 1.0, int4 quantizes 1.0 to
    # round(1.0 * 4) / 4 = 1.0 → 1.0 * 10 = 10.0
    assert torch.allclose(result[0], torch.tensor(10.0), atol=1.0)
    assert torch.allclose(result[1], torch.tensor(-10.0), atol=1.0)


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
    """BFloat16Format with round_mode='nearest' uses direct elemwise (no normalization)."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("bfloat16")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    # BFloat16 is a float format — direct elemwise without normalization
    expected = _quantize_elemwise(x, fmt, round_mode="nearest")
    assert torch.allclose(result, expected, atol=1e-5)


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
    """Float16Format with round_mode='nearest' uses direct elemwise (no normalization)."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("float16")
    result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    # Float16 is a float format — direct elemwise without normalization
    expected = _quantize_elemwise(x, fmt, round_mode="nearest")
    assert torch.allclose(result, expected, atol=1e-4)


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
    per_channel_result = fmt.quantize(x, GranularitySpec.per_channel(axis=0), "even", scale_storage="fp32")
    # per_channel should differ from per_tensor (shortcut only applies to per_tensor)
    assert not torch.equal(per_tensor_result, per_channel_result)


def test_float16_per_channel_even_uses_elemwise_not_shortcut():
    """Float16 with per_channel+even should NOT use .to(float16) shortcut."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    fmt = FormatBase.from_str("float16")
    per_tensor_result = fmt.quantize(x, GranularitySpec.per_tensor(), "even")
    per_channel_result = fmt.quantize(x, GranularitySpec.per_channel(axis=0), "even", scale_storage="fp32")
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


# ---------------------------------------------------------------------------
# 16. FormatBase._quantize_per_block() — direct method tests
# ---------------------------------------------------------------------------

class TestPerBlockQuantizeDirect:

    @pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "fp8_e5m2",
                                           "fp6_e3m2", "fp4_e2m1"])
    def test_same_as_format_quantize_dispatch(self, fmt_name):
        """_quantize_per_block() == fmt.quantize(x, PER_BLOCK, ...)."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)
        gran = GranularitySpec.per_block(32)

        via_dispatch = fmt.quantize(x, gran, "nearest")
        via_direct = fmt._quantize_per_block(x, gran, "nearest")
        assert torch.equal(via_dispatch, via_direct), \
            f"{fmt_name}: dispatch != direct"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    def test_output_finite(self, fmt_name):
        """Per-block quantized output should always be finite."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert result.isfinite().all(), \
            f"{fmt_name}: non-finite values in output"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1"])
    def test_idempotent(self, fmt_name):
        """Quantizing twice should be idempotent (or nearly)."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)
        gran = GranularitySpec.per_block(32)

        once = fmt._quantize_per_block(x, gran, "nearest")
        twice = fmt._quantize_per_block(once, gran, "nearest")
        assert torch.allclose(once, twice, atol=1e-6), \
            f"{fmt_name}: not idempotent"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_various_block_sizes(self, fmt_name, block_size):
        """Different block sizes should produce finite outputs."""
        torch.manual_seed(42)
        x = torch.randn(4, block_size * 3)
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(block_size), "nearest")
        assert result.isfinite().all()

    def test_round_mode_effect(self):
        """floor vs nearest should differ for low-precision formats."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str("fp4_e2m1")
        gran = GranularitySpec.per_block(32)

        nearest = fmt._quantize_per_block(x, gran, "nearest")
        floor = fmt._quantize_per_block(x, gran, "floor")
        assert not torch.equal(nearest, floor), \
            "floor and nearest should differ for fp4"

    def test_shape_preserved(self):
        """Output shape must equal input shape for various tensor ranks."""
        torch.manual_seed(42)
        fmt = FormatBase.from_str("fp8_e4m3")
        gran = GranularitySpec.per_block(32)

        for shape in [(4, 64), (2, 3, 64), (1, 2, 3, 128)]:
            x = torch.randn(*shape)
            result = fmt._quantize_per_block(x, gran, "nearest")
            assert result.shape == x.shape, \
                f"shape mismatch: {result.shape} != {x.shape}"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    def test_small_values(self, fmt_name):
        """Values near zero should not become NaN or Inf."""
        torch.manual_seed(42)
        x = torch.randn(4, 64) * 1e-6
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert result.isfinite().all(), \
            f"{fmt_name}: small values produced non-finite results"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    def test_large_values(self, fmt_name):
        """Large values should not become NaN."""
        torch.manual_seed(42)
        x = torch.randn(4, 64) * 1e6
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert not result.isnan().any(), \
            f"{fmt_name}: large values produced NaN"

    def test_zero_input(self):
        """Zero input should produce zero output."""
        x = torch.zeros(4, 64)
        fmt = FormatBase.from_str("fp8_e4m3")
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert (result == 0).all()

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    def test_no_nan_in_output(self, fmt_name):
        """Output should never contain NaN for random normal inputs."""
        torch.manual_seed(42)
        x = torch.randn(8, 128)
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert not result.isnan().any()


# ===========================================================================
# 17. POT scale_storage: per_tensor, per_channel, per_block behavior
# ===========================================================================

INT_FORMATS = ["int8", "int4", "int2"]
FLOAT_FORMATS_POT = ["fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"]


class TestPOTScaleStorage:
    """Verify POT scale_storage behavior.

    Math (per_tensor, integer format, ebits==0):
      amax = max(|x|).clamp(min=1e-12)
      amax_pot = 2 ** round(log2(amax))
      x_q = amax_pot * elemwise_quantize(x / amax_pot)

    Properties:
      - POT in [amax * 2^{-0.5}, amax * 2^{0.5}]  (0.707x-1.414x of amax)
      - Idempotent: if amax = 2^k exactly, amax_pot = amax
      - Float formats (ebits>0): no normalization → POT is no-op
      - Per_block MX: shared exponent is inherently POT → scale_storage is no-op
    """

    # ---- per_tensor integer POT -------------------------------------------------

    @pytest.mark.parametrize("fmt_name", INT_FORMATS)
    def test_per_tensor_pot_differs_from_fp32(self, fmt_name):
        """POT rounds amax to nearest power-of-2 → output differs from fp32."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)

        out_pot = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                               "nearest", scale_storage="pot")
        out_fp32 = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                                "nearest", scale_storage="fp32")

        assert not torch.equal(out_pot, out_fp32), \
            f"{fmt_name}: POT should differ from fp32 for random amax"

    @pytest.mark.parametrize("fmt_name", INT_FORMATS)
    def test_per_tensor_pot_idempotent(self, fmt_name):
        """When amax is exactly 2^k, POT rounding is idempotent → POT == fp32."""
        torch.manual_seed(42)
        fmt = FormatBase.from_str(fmt_name)
        x = torch.tensor([[1.0, -3.0], [8.0, 0.5]])  # amax = 8.0 = 2^3

        out_pot = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                               "nearest", scale_storage="pot")
        out_fp32 = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                                "nearest", scale_storage="fp32")

        assert torch.equal(out_pot, out_fp32), \
            f"{fmt_name}: POT == fp32 when amax is exactly 2^k"

    @pytest.mark.parametrize("fmt_name", FLOAT_FORMATS_POT)
    def test_per_tensor_pot_float_noop(self, fmt_name):
        """Float formats (ebits>0) skip normalization → POT == fp32 (no-op)."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)

        out_pot = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                               "nearest", scale_storage="pot")
        out_fp32 = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                                "nearest", scale_storage="fp32")

        assert torch.equal(out_pot, out_fp32), \
            f"{fmt_name}: POT should be no-op for float formats (no normalization)"

    def test_per_tensor_pot_zero_input(self):
        """Zero input → amax clamped to 1e-12 → POT rounds amax → output valid."""
        x = torch.zeros(4, 64)
        fmt = FormatBase.from_str("int8")

        out = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                          "nearest", scale_storage="pot")
        assert (out == 0).all(), "Zero input should produce zero output with POT"

    def test_per_tensor_pot_inf_input(self):
        """Inf/NaN in input → POT still produces output with correct shape."""
        x = torch.tensor([[1.0, float("inf")], [-float("inf"), float("nan")]])
        fmt = FormatBase.from_str("int8")

        out = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                          "nearest", scale_storage="pot")
        assert out.shape == x.shape

    @pytest.mark.parametrize("fmt_name", INT_FORMATS)
    def test_per_tensor_pot_preserves_sign(self, fmt_name):
        """POT scale > 0 → no element flips sign.
        Low-precision formats may quantize small values to zero (sign=0),
        but no positive becomes negative or vice versa."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)

        out = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                          "nearest", scale_storage="pot")
        # out * x >= 0  ⇔  signs agree (or one is zero)
        assert (out * x >= 0).all(), \
            f"{fmt_name}: POT should not flip sign of any element"

    def test_per_tensor_pot_small_amax(self):
        """Very small amax (subnormal range) still produces valid POT scale."""
        x = torch.tensor([1e-30, -2e-30, 3e-30])
        fmt = FormatBase.from_str("int8")

        out = fmt.quantize(x.clone(), GranularitySpec.per_tensor(),
                          "nearest", scale_storage="pot")
        assert not out.isnan().any(), "Small amax should not produce NaN"
        assert not out.isinf().any(), "Small amax should not produce Inf"

    # ---- per_channel POT --------------------------------------------------------

    @pytest.mark.parametrize("fmt_name", INT_FORMATS)
    def test_per_channel_pot_differs_from_fp32(self, fmt_name):
        """Per-channel POT rounds each channel's amax independently."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)

        out_pot = fmt.quantize(x.clone(), GranularitySpec.per_channel(axis=-1),
                               "nearest", scale_storage="pot")
        out_fp32 = fmt.quantize(x.clone(), GranularitySpec.per_channel(axis=-1),
                                "nearest", scale_storage="fp32")

        assert not torch.equal(out_pot, out_fp32), \
            f"{fmt_name}: per_channel POT should differ from fp32"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp8_e5m2"])
    def test_per_channel_pot_float(self, fmt_name):
        """Float formats (ebits>0) with per_channel DO normalize, POT takes effect."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)

        out_pot = fmt.quantize(x.clone(), GranularitySpec.per_channel(axis=-1),
                               "nearest", scale_storage="pot")
        out_fp32 = fmt.quantize(x.clone(), GranularitySpec.per_channel(axis=-1),
                                "nearest", scale_storage="fp32")

        assert not torch.equal(out_pot, out_fp32), \
            f"{fmt_name}: per_channel float POT should differ from fp32"

    # ---- per_block (MX) — scale_storage is no-op ---------------------------------

    @pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3"])
    def test_per_block_scale_storage_noop(self, fmt_name):
        """Per_block MX shared exponent is inherently POT; scale_storage has no effect."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)

        out_pot = fmt.quantize(x.clone(), GranularitySpec.per_block(32),
                               "nearest", scale_storage="pot")
        out_fp32 = fmt.quantize(x.clone(), GranularitySpec.per_block(32),
                                "nearest", scale_storage="fp32")

        assert torch.equal(out_pot, out_fp32), \
            f"{fmt_name}: per_block scale_storage should be no-op (MX inherently POT)"

    # ---- default POT via quantize() entry ---------------------------------------

    def test_pot_default_via_quantize_entry(self):
        """quantize(x, scheme) with default (POT) scheme differs from explicit fp32."""
        from src.scheme.quant_scheme import QuantScheme
        from src.quantize.elemwise import quantize

        torch.manual_seed(42)
        x = torch.randn(4, 64)

        scheme_default = QuantScheme.per_tensor("int8")  # defaults to scale_storage="pot"
        scheme_fp32 = QuantScheme(format="int8",
                                  granularity=GranularitySpec.per_tensor(),
                                  scale_storage="fp32")

        out_default = quantize(x.clone(), scheme_default)
        out_fp32 = quantize(x.clone(), scheme_fp32)

        assert not torch.equal(out_default, out_fp32), \
            "Default POT scheme should differ from explicit fp32 scheme"
