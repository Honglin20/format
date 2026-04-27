"""
Tests for parameterized format registration (P4).

Verifies:
- register_float_format / register_int_format one-liner factories
- Auto-parsing from naming convention (FormatBase.from_str)
- from_str fallback: registry → aliases → auto-parse
- Equivalence: auto-registered format quantization matches manual FPFormat
- Validation: invalid names, bit constraints
"""
import pytest
import torch

from src.formats.base import FormatBase
from src.formats.registry import (
    FORMAT_REGISTRY,
    register_format,
    register_float_format,
    register_int_format,
    get_format,
)
from src.formats.fp_formats import FPFormat
from src.formats.int_formats import IntFormat
from src.scheme.granularity import GranularitySpec


# ---------------------------------------------------------------------------
# 1. register_float_format — one-liner factory
# ---------------------------------------------------------------------------

def test_register_float_format_creates_fpformat():
    """register_float_format returns an FPFormat instance."""
    fmt = register_float_format("fp5_e3m1", ebits=3, mbits=3, overwrite=True)
    assert isinstance(fmt, FPFormat)
    assert fmt.name == "fp5_e3m1"
    assert fmt.ebits == 3
    assert fmt.mbits == 3


def test_register_float_format_registers_in_registry():
    """After registration, get_format() returns the same instance."""
    fmt = register_float_format("fp7_e4m2", ebits=4, mbits=2, overwrite=True)
    assert get_format("fp7_e4m2") is fmt


def test_register_float_format_case_insensitive_lookup():
    """get_format is case-insensitive."""
    fmt = register_float_format("fp9_e5m3", ebits=5, mbits=3, overwrite=True)
    assert get_format("FP9_E5M3") is fmt
    assert get_format("Fp9_e5M3") is fmt


def test_register_float_format_with_alias():
    """Aliases are registered alongside canonical name."""
    fmt = register_float_format("fp3_e1m1", ebits=1, mbits=1, aliases=["fp3"], overwrite=True)
    assert get_format("fp3") is fmt
    assert get_format("fp3_e1m1") is fmt


def test_register_float_format_with_max_norm_override():
    """Custom max_norm_override is stored on the format."""
    fmt = register_float_format("fp8_custom", ebits=4, mbits=3,
                                max_norm_override=100.0, overwrite=True)
    assert fmt.max_norm == 100.0


def test_register_float_format_rejects_duplicate():
    """Duplicate registration raises ValueError without overwrite=True."""
    register_float_format("fp6_test", ebits=3, mbits=2, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_float_format("fp6_test", ebits=3, mbits=2)


def test_register_float_format_overwrite_silent():
    """overwrite=True silently replaces existing entry."""
    fmt1 = register_float_format("fp6_ow", ebits=3, mbits=2, overwrite=True)
    fmt2 = register_float_format("fp6_ow", ebits=3, mbits=2, overwrite=True)
    assert get_format("fp6_ow") is fmt2  # replaced


def test_register_float_format_negative_params_raise():
    """Negative ebits or mbits raises ValueError."""
    with pytest.raises(ValueError, match="must be non-negative"):
        register_float_format("bad_fmt", ebits=-1, mbits=1, overwrite=True)
    with pytest.raises(ValueError, match="must be non-negative"):
        register_float_format("bad_fmt", ebits=1, mbits=-1, overwrite=True)


# ---------------------------------------------------------------------------
# 2. register_int_format — one-liner factory
# ---------------------------------------------------------------------------

def test_register_int_format_creates_intformat():
    """register_int_format returns an IntFormat instance."""
    fmt = register_int_format("int16", bits=16, overwrite=True)
    assert isinstance(fmt, IntFormat)
    assert fmt.name == "int16"
    assert fmt.mbits == 16
    assert fmt.ebits == 0


def test_register_int_format_registers_in_registry():
    """After registration, get_format() returns the same instance."""
    fmt = register_int_format("int12", bits=12, overwrite=True)
    assert get_format("int12") is fmt


def test_register_int_format_case_insensitive_lookup():
    """get_format is case-insensitive for int formats."""
    fmt = register_int_format("int10", bits=10, overwrite=True)
    assert get_format("INT10") is fmt


def test_register_int_format_with_alias():
    """Int format aliases work."""
    fmt = register_int_format("int3", bits=3, aliases=["i3"], overwrite=True)
    assert get_format("i3") is fmt


def test_register_int_format_rejects_bit_too_small():
    """bits < 2 raises ValueError."""
    with pytest.raises(ValueError, match="bits must be >= 2"):
        register_int_format("int1", bits=1)


def test_register_int_format_rejects_duplicate():
    """Duplicate int registration raises without overwrite=True."""
    register_int_format("int6_test", bits=6, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_int_format("int6_test", bits=6)


# ---------------------------------------------------------------------------
# 3. Auto-parse from naming convention (FormatBase.from_str / get_format)
# ---------------------------------------------------------------------------

def test_from_str_auto_parses_fp_naming_convention():
    """FormatBase.from_str('fp5_e3m1') auto-creates FPFormat(ebits=3, mbits=1)."""
    fmt = FormatBase.from_str("fp5_e3m1")
    assert isinstance(fmt, FPFormat)
    assert fmt.ebits == 3
    assert fmt.mbits == 3  # sign(1) + implicit(1) + actual(1)
    assert fmt.name == "fp5_e3m1"


def test_from_str_auto_parses_int_naming_convention():
    """FormatBase.from_str('int16') auto-creates IntFormat(bits=16)."""
    fmt = FormatBase.from_str("int16")
    assert isinstance(fmt, IntFormat)
    assert fmt.mbits == 16
    assert fmt.name == "int16"


def test_from_str_auto_parsed_format_is_cached():
    """After auto-parse, subsequent calls return the same instance (from registry)."""
    fmt1 = FormatBase.from_str("fp11_e6m4")
    fmt2 = FormatBase.from_str("fp11_e6m4")
    assert fmt1 is fmt2


def test_from_str_still_works_for_builtin_formats():
    """Built-in formats still resolve correctly."""
    assert FormatBase.from_str("fp8_e4m3").ebits == 4
    assert FormatBase.from_str("int8").mbits == 8
    assert FormatBase.from_str("bfloat16").name == "bfloat16"


def test_from_str_bad_naming_convention_raises():
    """Unparseable + unregistered name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown format"):
        FormatBase.from_str("fp_bad_name")


def test_from_str_mismatched_bit_count_raises():
    """fp10_e3m2: 10 != 3+2+1 → ValueError."""
    with pytest.raises(ValueError, match="Cannot parse"):
        FormatBase.from_str("fp10_e3m2")  # 10 != 6


# ---------------------------------------------------------------------------
# 4. Quantization equivalence: auto-registered vs manual FPFormat
# ---------------------------------------------------------------------------

def test_auto_registered_float_quantization_matches_manual():
    """Quantization of fp5_e3m1 via from_str == manual FPFormat construction."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    auto_fmt = FormatBase.from_str("fp5_e3m1")
    # Naming convention fp5_e3m1: actual mantissa=1 → code mbits = 1+2 = 3
    manual_fmt = FPFormat(name="fp5_e3m1", ebits=3, mbits=3)

    result_auto = auto_fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    result_manual = manual_fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")

    assert torch.equal(result_auto, result_manual)


def test_auto_registered_int_quantization_matches_manual():
    """Quantization of int6 via from_str == manual IntFormat construction."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    auto_fmt = FormatBase.from_str("int6")
    manual_fmt = IntFormat(bits=6, name="int6")

    result_auto = auto_fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
    result_manual = manual_fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")

    assert torch.equal(result_auto, result_manual)


def test_auto_registered_format_per_channel():
    """Auto-registered format works with per_channel granularity."""
    torch.manual_seed(42)
    x = torch.randn(4, 16)

    fmt = FormatBase.from_str("fp5_e3m1")
    result = fmt.quantize(x, GranularitySpec.per_channel(axis=-1), "nearest")
    assert result.shape == x.shape
    assert result.isfinite().all()


def test_auto_registered_format_per_block():
    """Auto-registered format works with per_block granularity."""
    torch.manual_seed(42)
    x = torch.randn(4, 32)

    fmt = FormatBase.from_str("fp5_e3m1")
    result = fmt.quantize(x, GranularitySpec.per_block(8), "nearest")
    assert result.shape == x.shape
    assert result.isfinite().all()


# ---------------------------------------------------------------------------
# 5. register_format raw — direct registration
# ---------------------------------------------------------------------------

def test_register_format_raw():
    """register_format() accepts any FormatBase instance directly."""
    fmt = FPFormat(name="custom_fp", ebits=2, mbits=2)
    register_format("custom_fp", fmt, overwrite=True)
    assert get_format("custom_fp") is fmt


def test_register_format_with_aliases_list():
    """register_format aliases argument creates alias entries."""
    fmt = FPFormat(name="aliased_fp", ebits=3, mbits=2)
    register_format("aliased_fp", fmt, aliases=["afp", "a.fp"], overwrite=True)
    assert get_format("afp") is fmt
    assert get_format("a.fp") is fmt


def test_register_format_rejects_duplicate():
    """register_format raises without overwrite on existing name."""
    fmt2 = FPFormat(name="dup_fp", ebits=2, mbits=3)
    register_format("dup_fp", fmt2, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_format("dup_fp", fmt2)


# ---------------------------------------------------------------------------
# 6. Integration: auto-registered formats in QuantScheme
# ---------------------------------------------------------------------------

def test_from_str_format_in_quant_scheme():
    """Auto-registered format works inside QuantScheme."""
    from src.scheme.quant_scheme import QuantScheme

    scheme = QuantScheme(format=FormatBase.from_str("fp5_e3m1"))
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    result = scheme.format.quantize(x, scheme.granularity, scheme.round_mode)
    assert result.shape == x.shape
    assert result.isfinite().all()


# ---------------------------------------------------------------------------
# 7. get_format with aliases for auto-registered formats
# ---------------------------------------------------------------------------

def test_auto_registered_format_supports_alias():
    """register_float_format with alias makes alias work with get_format."""
    fmt = register_float_format("fp4_test_p4", ebits=2, mbits=1,
                                aliases=["f4t"], overwrite=True)
    assert get_format("f4t") is fmt


def test_auto_registered_int_format_supports_alias():
    """register_int_format with alias makes alias work with get_format."""
    fmt = register_int_format("int5_test", bits=5, aliases=["i5t"], overwrite=True)
    assert get_format("i5t") is fmt
