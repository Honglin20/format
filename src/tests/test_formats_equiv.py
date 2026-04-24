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
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.scheme.transform import TransformBase, IdentityTransform
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
# 5. GranularityMode + GranularitySpec tests
# ---------------------------------------------------------------------------

def test_granularity_mode_values():
    assert GranularityMode.PER_TENSOR.value == "per_tensor"
    assert GranularityMode.PER_CHANNEL.value == "per_channel"
    assert GranularityMode.PER_BLOCK.value == "per_block"


def test_granularity_spec_per_tensor():
    g = GranularitySpec.per_tensor()
    assert g.mode == GranularityMode.PER_TENSOR
    assert g.block_size == 0
    assert not g.is_mx


def test_granularity_spec_per_channel():
    g = GranularitySpec.per_channel(axis=1)
    assert g.mode == GranularityMode.PER_CHANNEL
    assert g.channel_axis == 1
    assert g.block_size == 0
    assert not g.is_mx


def test_granularity_spec_per_block():
    g = GranularitySpec.per_block(32)
    assert g.mode == GranularityMode.PER_BLOCK
    assert g.block_size == 32
    assert g.is_mx


def test_granularity_spec_per_block_requires_positive_size():
    with pytest.raises(ValueError, match="PER_BLOCK requires block_size > 0"):
        GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=0)


def test_granularity_spec_per_tensor_rejects_block_size():
    with pytest.raises(ValueError, match="PER_TENSOR requires block_size=0"):
        GranularitySpec(mode=GranularityMode.PER_TENSOR, block_size=32)


def test_granularity_spec_per_channel_rejects_block_size():
    with pytest.raises(ValueError, match="PER_CHANNEL requires block_size=0"):
        GranularitySpec(mode=GranularityMode.PER_CHANNEL, block_size=8)


def test_granularity_spec_default_values():
    g = GranularitySpec()
    assert g.mode == GranularityMode.PER_TENSOR
    assert g.block_size == 0
    assert g.channel_axis == 0


def test_granularity_spec_equality():
    assert GranularitySpec.per_tensor() == GranularitySpec.per_tensor()
    assert GranularitySpec.per_block(32) == GranularitySpec.per_block(32)
    assert GranularitySpec.per_channel(axis=1) == GranularitySpec.per_channel(axis=1)


def test_granularity_spec_inequality():
    assert GranularitySpec.per_tensor() != GranularitySpec.per_block(32)
    assert GranularitySpec.per_block(32) != GranularitySpec.per_block(16)
    assert GranularitySpec.per_channel(axis=0) != GranularitySpec.per_channel(axis=1)


def test_granularity_spec_hashing():
    s = {GranularitySpec.per_tensor(), GranularitySpec.per_block(32), GranularitySpec.per_block(32)}
    assert len(s) == 2  # per_block(32) deduplicated


def test_granularity_spec_frozen():
    g = GranularitySpec.per_block(32)
    with pytest.raises(AttributeError):
        g.block_size = 64


def test_granularity_spec_per_block_negative_size():
    with pytest.raises(ValueError, match="PER_BLOCK requires block_size > 0"):
        GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=-1)


def test_granularity_spec_per_channel_default_axis():
    g = GranularitySpec.per_channel()
    assert g.channel_axis == 0


def test_granularity_spec_per_tensor_rejects_channel_axis():
    with pytest.raises(ValueError, match="PER_TENSOR requires channel_axis=0"):
        GranularitySpec(mode=GranularityMode.PER_TENSOR, channel_axis=5)


def test_granularity_spec_per_block_rejects_channel_axis():
    with pytest.raises(ValueError, match="PER_BLOCK requires channel_axis=0"):
        GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=32, channel_axis=1)


# ---------------------------------------------------------------------------
# 6. QuantScheme tests
# ---------------------------------------------------------------------------

def test_quant_scheme_basic():
    scheme = QuantScheme(
        format="fp8_e4m3",
        granularity=GranularitySpec.per_block(32),
        round_mode="nearest",
    )
    assert scheme.format_name == "fp8_e4m3"
    assert isinstance(scheme.format, FormatBase)
    assert scheme.granularity == GranularitySpec.per_block(32)
    assert scheme.block_size == 32
    assert scheme.round_mode == "nearest"


def test_quant_scheme_per_tensor():
    scheme = QuantScheme.per_tensor("fp8_e4m3")
    assert scheme.granularity == GranularitySpec.per_tensor()
    assert scheme.block_size == 0


def test_quant_scheme_per_channel():
    scheme = QuantScheme.per_channel("fp8_e4m3")
    assert scheme.granularity == GranularitySpec.per_channel()
    assert scheme.block_size == 0


def test_quant_scheme_mxfp():
    scheme = QuantScheme.mxfp("fp6_e3m2", block_size=32)
    assert scheme.granularity == GranularitySpec.per_block(32)
    assert scheme.block_size == 32
    assert scheme.format_name == "fp6_e3m2"


def test_quant_scheme_hashable():
    scheme = QuantScheme(format="fp8_e4m3", granularity=GranularitySpec.per_block(32))
    scheme_dict = {scheme: "test"}
    assert scheme_dict[scheme] == "test"


def test_quant_scheme_invalid_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        QuantScheme(format="nonexistent", granularity=GranularitySpec.per_block(32))


def test_quant_scheme_invalid_round_mode_raises():
    with pytest.raises(ValueError, match="Invalid round_mode"):
        QuantScheme(format="fp8_e4m3", granularity=GranularitySpec.per_block(32),
                    round_mode="invalid")


def test_quant_scheme_per_block_requires_block_size():
    with pytest.raises(ValueError, match="PER_BLOCK requires block_size > 0"):
        QuantScheme(format="fp8_e4m3", granularity=GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=0))


def test_quant_scheme_per_tensor_rejects_block_size():
    with pytest.raises(ValueError, match="PER_TENSOR requires block_size=0"):
        QuantScheme(format="fp8_e4m3", granularity=GranularitySpec(mode=GranularityMode.PER_TENSOR, block_size=32))


def test_quant_scheme_per_channel_rejects_block_size():
    with pytest.raises(ValueError, match="PER_CHANNEL requires block_size=0"):
        QuantScheme(format="fp8_e4m3", granularity=GranularitySpec(mode=GranularityMode.PER_CHANNEL, block_size=8))


def test_quant_scheme_is_mx_property():
    mx = QuantScheme.mxfp("fp8_e4m3", block_size=32)
    pt = QuantScheme.per_tensor("fp8_e4m3")
    assert mx.is_mx
    assert not pt.is_mx


def test_quant_scheme_dither_round_mode():
    """dither is supported in old _round_mantissa() but not in RoundingMode enum."""
    scheme = QuantScheme(format="fp8_e4m3", granularity=GranularitySpec.per_block(32),
                        round_mode="dither")
    assert scheme.round_mode == "dither"


def test_quant_scheme_default_granularity_is_per_tensor():
    scheme = QuantScheme(format="fp8_e4m3")
    assert scheme.granularity == GranularitySpec.per_tensor()
    assert scheme.block_size == 0
    assert not scheme.is_mx


def test_quant_scheme_default_round_mode():
    scheme = QuantScheme(format="fp8_e4m3")
    assert scheme.round_mode == "nearest"


def test_quant_scheme_equality():
    a = QuantScheme.mxfp("fp8_e4m3", block_size=32)
    b = QuantScheme.mxfp("fp8_e4m3", block_size=32)
    assert a == b


def test_quant_scheme_inequality():
    a = QuantScheme.mxfp("fp8_e4m3", block_size=32)
    b = QuantScheme.mxfp("fp8_e4m3", block_size=16)
    assert a != b
    c = QuantScheme.per_tensor("fp8_e4m3")
    assert a != c


def test_quant_scheme_frozen():
    scheme = QuantScheme(format="fp8_e4m3")
    with pytest.raises(AttributeError):
        scheme.round_mode = "floor"


def test_quant_scheme_per_channel_with_axis():
    scheme = QuantScheme.per_channel("fp8_e4m3", axis=1)
    assert scheme.granularity.channel_axis == 1


def test_quant_scheme_mxfp_default_block_size():
    scheme = QuantScheme.mxfp("fp8_e4m3")
    assert scheme.block_size == 32


def test_quant_scheme_empty_format_raises():
    with pytest.raises(ValueError):
        QuantScheme(format="")


def test_quant_scheme_round_mode_case_sensitive():
    with pytest.raises(ValueError, match="Invalid round_mode"):
        QuantScheme(format="fp8_e4m3", granularity=GranularitySpec.per_block(32),
                    round_mode="Nearest")


# --- Transform tests ---

def test_identity_transform_forward():
    t = IdentityTransform()
    x = torch.randn(4, 8)
    assert torch.equal(t.forward(x), x)


def test_identity_transform_inverse():
    t = IdentityTransform()
    x = torch.randn(4, 8)
    assert torch.equal(t.inverse(x), x)


def test_identity_transform_invertible():
    assert IdentityTransform().invertible


def test_identity_transform_equality():
    assert IdentityTransform() == IdentityTransform()


def test_transform_base_not_invertible_by_default():
    # TransformBase itself is abstract, but the default invertible flag is False
    assert TransformBase.invertible is False


# --- QuantScheme transform field ---

def test_quant_scheme_default_transform():
    scheme = QuantScheme(format="fp8_e4m3")
    assert isinstance(scheme.transform, IdentityTransform)


def test_quant_scheme_explicit_transform():
    scheme = QuantScheme(format="fp8_e4m3", transform=IdentityTransform())
    assert isinstance(scheme.transform, IdentityTransform)


# --- C1: TransformBase requires __eq__/__hash__ ---

def test_transform_base_requires_eq_and_hash():
    """Concrete TransformBase without __eq__/__hash__ should be rejected by ABC."""
    class IncompleteTransform(TransformBase):
        def forward(self, x):
            return x
    with pytest.raises(TypeError):
        IncompleteTransform()


# --- C2: transform type validation ---

def test_quant_scheme_invalid_transform_type_raises():
    with pytest.raises(TypeError, match="transform must be TransformBase"):
        QuantScheme(format="fp8_e4m3", transform="invalid")


def test_quant_scheme_invalid_transform_none_raises():
    with pytest.raises(TypeError, match="transform must be TransformBase"):
        QuantScheme(format="fp8_e4m3", transform=None)


# --- C3: per_channel string axis guard ---

def test_per_channel_rejects_string_axis():
    with pytest.raises(TypeError, match="axis must be int"):
        QuantScheme.per_channel("fp8_e4m3", "floor")


# --- M2: IdentityTransform hash consistency ---

def test_quant_scheme_identity_transform_hash_stable():
    s1 = QuantScheme(format="fp8_e4m3", transform=IdentityTransform())
    s2 = QuantScheme(format="fp8_e4m3", transform=IdentityTransform())
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


# --- M4: Custom Transform equality in QuantScheme ---

def test_quant_scheme_custom_transform_equality():
    class MyTransform(TransformBase):
        def forward(self, x):
            return x
        def __eq__(self, other):
            return isinstance(other, MyTransform)
        def __hash__(self):
            return hash("MyTransform")

    s1 = QuantScheme(format="fp8_e4m3", transform=MyTransform())
    s2 = QuantScheme(format="fp8_e4m3", transform=MyTransform())
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


# --- QuantScheme format as FormatBase ---

def test_quant_scheme_format_auto_coercion_from_str():
    scheme = QuantScheme(format="fp8_e4m3")
    assert isinstance(scheme.format, FormatBase)
    assert scheme.format.name == "fp8_e4m3"


def test_quant_scheme_format_from_format_base():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt)
    assert scheme.format is fmt


def test_quant_scheme_format_name_property():
    scheme = QuantScheme.mxfp("fp8_e4m3", block_size=32)
    assert scheme.format_name == "fp8_e4m3"


def test_quant_scheme_invalid_format_type_raises():
    with pytest.raises(TypeError, match="format must be FormatBase"):
        QuantScheme(format=123)


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


# ---------------------------------------------------------------------------
# 10. __slots__ verification — subclasses should not have __dict__
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "float16", "bfloat16"])
def test_format_no_dict(fmt_name):
    """Subclasses with __slots__ = () should not have per-instance __dict__."""
    fmt = FormatBase.from_str(fmt_name)
    assert not hasattr(fmt, "__dict__"), f"{fmt_name} has __dict__ (missing __slots__ = ())"


# ---------------------------------------------------------------------------
# 11. src.scheme module export tests
# ---------------------------------------------------------------------------

def test_scheme_module_exports_granularity_mode():
    from src.scheme import GranularityMode
    assert GranularityMode.PER_TENSOR is not None


def test_scheme_module_exports_granularity_spec():
    from src.scheme import GranularitySpec
    assert GranularitySpec.per_block(32).block_size == 32


def test_scheme_module_exports_quant_scheme():
    from src.scheme import QuantScheme
    assert QuantScheme.per_tensor("int8").format_name == "int8"


# ---------------------------------------------------------------------------
# P2F-7: Granularity type guard + channel_axis negative / out-of-bounds
# ---------------------------------------------------------------------------

# C1: QuantScheme rejects invalid granularity types
def test_quant_scheme_invalid_granularity_type_raises():
    with pytest.raises(TypeError, match="granularity must be GranularitySpec"):
        QuantScheme(format="fp8_e4m3", granularity="per_tensor")


def test_quant_scheme_invalid_granularity_none_raises():
    with pytest.raises(TypeError, match="granularity must be GranularitySpec"):
        QuantScheme(format="fp8_e4m3", granularity=None)


def test_quant_scheme_invalid_granularity_int_raises():
    with pytest.raises(TypeError, match="granularity must be GranularitySpec"):
        QuantScheme(format="fp8_e4m3", granularity=123)


# C2: channel_axis negative indexing + out-of-bounds
def test_granularity_spec_per_channel_negative_axis_allowed():
    """GranularitySpec accepts negative axis (not validated at construction time)."""
    g = GranularitySpec.per_channel(axis=-1)
    assert g.channel_axis == -1


def test_format_quantize_per_channel_negative_axis_normalization():
    """_quantize_per_channel normalizes -1 to ndim-1, producing same result as positive axis."""
    fmt = FormatBase.from_str("fp8_e4m3")
    x = torch.randn(3, 4, 5)
    g_pos = GranularitySpec.per_channel(axis=2)
    g_neg = GranularitySpec.per_channel(axis=-1)
    assert torch.equal(
        fmt.quantize(x.clone(), g_pos, "nearest"),
        fmt.quantize(x.clone(), g_neg, "nearest"),
    )


def test_format_quantize_per_channel_out_of_bounds_axis_raises():
    """Out-of-bounds axis raises ValueError at quantization time."""
    fmt = FormatBase.from_str("fp8_e4m3")
    x = torch.randn(3, 4)
    g = GranularitySpec.per_channel(axis=-100)
    with pytest.raises(ValueError, match="out of range"):
        fmt.quantize(x, g, "nearest")

