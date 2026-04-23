"""
Format equivalence tests (TDD — tests written FIRST).

Verifies that the new FormatBase system produces identical format parameters
to the old _get_format_params() function for every supported format.
"""
import pytest
import torch
import math

# Old code imports
from mx.formats import ElemFormat, _get_format_params, _get_min_norm, _get_max_norm

# New code imports
from src.formats.base import FormatBase, compute_min_norm, compute_max_norm
from src.formats.registry import FORMAT_REGISTRY, register_format, get_format
from src.scheme.granularity import Granularity
from src.scheme.quant_scheme import QuantScheme


# ---------------------------------------------------------------------------
# 1. Format parameter equivalence: old _get_format_params vs new FormatBase
# ---------------------------------------------------------------------------

ALL_FORMAT_NAMES = [
    "int8", "int4", "int2",
    "fp8_e5m2", "fp8_e4m3",
    "fp6_e3m2", "fp6_e2m3",
    "fp4", "fp4_e2m1",
    "float16", "fp16",
    "bfloat16", "bf16",
]


@pytest.mark.parametrize("format_name", ALL_FORMAT_NAMES)
def test_format_params_equivalence(format_name):
    """New FormatBase must produce identical (ebits, mbits, emax, max_norm, min_norm)
    as old _get_format_params()."""
    old_fmt = ElemFormat.from_str(format_name)
    old_ebits, old_mbits, old_emax, old_max_norm, old_min_norm = _get_format_params(old_fmt)

    new_fmt = FormatBase.from_str(format_name)
    assert new_fmt.ebits == old_ebits, f"{format_name}: ebits {new_fmt.ebits} != {old_ebits}"
    assert new_fmt.mbits == old_mbits, f"{format_name}: mbits {new_fmt.mbits} != {old_mbits}"
    assert new_fmt.emax == old_emax, f"{format_name}: emax {new_fmt.emax} != {old_emax}"
    assert new_fmt.max_norm == old_max_norm, f"{format_name}: max_norm {new_fmt.max_norm} != {old_max_norm}"
    assert new_fmt.min_norm == old_min_norm, f"{format_name}: min_norm {new_fmt.min_norm} != {old_min_norm}"


# ---------------------------------------------------------------------------
# 2. Registry tests
# ---------------------------------------------------------------------------

def test_registry_has_all_formats():
    for name in ALL_FORMAT_NAMES:
        fmt = FormatBase.from_str(name)
        assert fmt is not None, f"Format {name} not found in registry"


def test_registry_aliases():
    assert FormatBase.from_str("fp4").name == FormatBase.from_str("fp4_e2m1").name
    assert FormatBase.from_str("fp16").name == FormatBase.from_str("float16").name
    assert FormatBase.from_str("bf16").name == FormatBase.from_str("bfloat16").name


def test_registry_unknown_format_raises():
    with pytest.raises(ValueError):
        FormatBase.from_str("nonexistent_format")


def test_register_custom_format():
    from src.formats.int_formats import IntFormat
    custom = IntFormat(bits=3, name="int3_custom")
    register_format("int3_custom", custom, overwrite=True)
    assert FormatBase.from_str("int3_custom") is custom
    # Clean up
    del FORMAT_REGISTRY["int3_custom"]


def test_register_format_overwrite_protection():
    """register_format should raise if name already exists and overwrite=False."""
    with pytest.raises(ValueError, match="already registered"):
        register_format("int8", FormatBase.from_str("int4"))


def test_register_format_overwrite_allowed():
    """register_format with overwrite=True should succeed."""
    original = FormatBase.from_str("int8")
    from src.formats.int_formats import IntFormat
    replacement = IntFormat(bits=8, name="int8")
    register_format("int8", replacement, overwrite=True)
    # Restore original
    register_format("int8", original, overwrite=True)


def test_case_insensitive_lookup():
    """Old ElemFormat.from_str() lowercases input; new code should too."""
    assert FormatBase.from_str("FP8_E4M3").name == "fp8_e4m3"
    assert FormatBase.from_str("Int8").name == "int8"
    assert FormatBase.from_str("BF16").name == "bfloat16"


# ---------------------------------------------------------------------------
# 3. Immutability tests
# ---------------------------------------------------------------------------

def test_format_instance_immutable():
    """Format instances should be frozen after construction."""
    fmt = FormatBase.from_str("fp8_e4m3")
    with pytest.raises(AttributeError, match="immutable"):
        fmt.emax = 999


def test_format_registry_returns_same_immutable_instance():
    """Registry returns singleton that can't be mutated."""
    fmt = FormatBase.from_str("fp8_e4m3")
    original_emax = fmt.emax
    with pytest.raises(AttributeError):
        fmt.emax = 999
    # Verify original value unchanged
    assert FormatBase.from_str("fp8_e4m3").emax == original_emax


# ---------------------------------------------------------------------------
# 4. Shared helper function equivalence tests
# ---------------------------------------------------------------------------

def test_compute_min_norm_matches_old():
    for ebits in [0, 2, 3, 4, 5, 8]:
        assert compute_min_norm(ebits) == _get_min_norm(ebits), \
            f"compute_min_norm({ebits}) != _get_min_norm({ebits})"


def test_compute_max_norm_matches_old():
    for ebits, mbits in [(5, 12), (5, 4), (8, 9)]:
        assert compute_max_norm(ebits, mbits) == _get_max_norm(ebits, mbits), \
            f"compute_max_norm({ebits},{mbits}) != _get_max_norm({ebits},{mbits})"


# ---------------------------------------------------------------------------
# 5. Granularity enum tests
# ---------------------------------------------------------------------------

def test_granularity_values():
    assert Granularity.PER_TENSOR.value == "per_tensor"
    assert Granularity.PER_CHANNEL.value == "per_channel"
    assert Granularity.PER_BLOCK.value == "per_block"


# ---------------------------------------------------------------------------
# 6. QuantScheme tests
# ---------------------------------------------------------------------------

def test_quant_scheme_basic():
    scheme = QuantScheme(
        format="fp8_e4m3",
        granularity=Granularity.PER_BLOCK,
        block_size=32,
        round="nearest",
    )
    assert scheme.format == "fp8_e4m3"
    assert scheme.granularity == Granularity.PER_BLOCK
    assert scheme.block_size == 32
    assert scheme.round == "nearest"


def test_quant_scheme_per_tensor():
    scheme = QuantScheme.per_tensor("fp8_e4m3")
    assert scheme.granularity == Granularity.PER_TENSOR
    assert scheme.block_size == 0


def test_quant_scheme_per_channel():
    scheme = QuantScheme.per_channel("fp8_e4m3")
    assert scheme.granularity == Granularity.PER_CHANNEL
    assert scheme.block_size == 0


def test_quant_scheme_mxfp():
    scheme = QuantScheme.mxfp("fp6_e3m2", block_size=32)
    assert scheme.granularity == Granularity.PER_BLOCK
    assert scheme.block_size == 32
    assert scheme.format == "fp6_e3m2"


def test_quant_scheme_immutability():
    scheme = QuantScheme(format="fp8_e4m3", granularity=Granularity.PER_BLOCK, block_size=32)
    scheme_dict = {scheme: "test"}
    assert scheme_dict[scheme] == "test"


# ---------------------------------------------------------------------------
# 7. Format classification tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int2", "int4", "int8"])
def test_format_is_integer(fmt_name):
    assert FormatBase.from_str(fmt_name).is_integer


@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp6_e3m2", "fp4_e2m1", "float16", "bfloat16"])
def test_format_is_float(fmt_name):
    assert not FormatBase.from_str(fmt_name).is_integer


# ---------------------------------------------------------------------------
# 8. Specific format value tests (known good values from old code)
# ---------------------------------------------------------------------------

def test_fp8_e4m3_params():
    fmt = FormatBase.from_str("fp8_e4m3")
    assert fmt.ebits == 4
    assert fmt.mbits == 5
    assert fmt.emax == 8
    assert fmt.max_norm == 448.0


def test_fp8_e5m2_params():
    fmt = FormatBase.from_str("fp8_e5m2")
    assert fmt.ebits == 5
    assert fmt.mbits == 4
    assert fmt.emax == 15


def test_fp4_e2m1_params():
    fmt = FormatBase.from_str("fp4_e2m1")
    assert fmt.ebits == 2
    assert fmt.mbits == 3
    assert fmt.emax == 2


def test_int8_params():
    fmt = FormatBase.from_str("int8")
    assert fmt.ebits == 0
    assert fmt.mbits == 8
    assert fmt.emax == 0


# ---------------------------------------------------------------------------
# 9. Repr test
# ---------------------------------------------------------------------------

def test_format_repr():
    fmt = FormatBase.from_str("fp8_e4m3")
    r = repr(fmt)
    assert "FPFormat" in r
    assert "fp8_e4m3" in r
