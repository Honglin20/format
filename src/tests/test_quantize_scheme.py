"""
P2F-4 TDD tests: quantize(x, scheme) unified entry point and
quantize_elemwise_op rewritten as QuantScheme-driven compat wrapper.

Tests written FIRST (red phase), then implementation makes them green.
"""
import pytest
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.transform import IdentityTransform
from src.formats.base import FormatBase
from src.quantize.elemwise import quantize, quantize_elemwise_op, _format_from_mx_specs


# ---------------------------------------------------------------------------
# 1. quantize(x, scheme) — unified entry point
# ---------------------------------------------------------------------------

def test_quantize_with_per_tensor_int8():
    """quantize(x, per_tensor/int8) should match FormatBase.quantize()."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_tensor("int8")
    result = quantize(x, scheme)
    expected = FormatBase.from_str("int8").quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert torch.allclose(result, expected, atol=1e-7)


def test_quantize_with_per_tensor_fp8_e4m3():
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_tensor("fp8_e4m3")
    result = quantize(x, scheme)
    expected = FormatBase.from_str("fp8_e4m3").quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert torch.allclose(result, expected, atol=1e-7)


def test_quantize_with_per_channel():
    """quantize(x, per_channel) should match FormatBase.quantize()."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_channel("fp8_e4m3", axis=0)
    result = quantize(x, scheme)
    expected = FormatBase.from_str("fp8_e4m3").quantize(x, GranularitySpec.per_channel(axis=0), "nearest")
    assert torch.allclose(result, expected, atol=1e-6)


def test_quantize_with_per_block():
    """quantize(x, per_block) should match FormatBase.quantize()."""
    torch.manual_seed(42)
    x = torch.randn(4, 32)
    scheme = QuantScheme.mxfp("fp8_e4m3", block_size=32)
    result = quantize(x, scheme)
    expected = FormatBase.from_str("fp8_e4m3").quantize(x, GranularitySpec.per_block(32), "nearest")
    assert torch.allclose(result, expected, atol=1e-6)


def test_quantize_with_identity_transform():
    """quantize with IdentityTransform should produce same result as without."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme(format="fp8_e4m3",
                         granularity=GranularitySpec.per_tensor(),
                         transform=IdentityTransform())
    result = quantize(x, scheme)
    expected = FormatBase.from_str("fp8_e4m3").quantize(x, GranularitySpec.per_tensor(), "nearest")
    assert torch.allclose(result, expected, atol=1e-7)


def test_quantize_with_floor_round_mode():
    """quantize should respect round_mode from scheme."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    scheme = QuantScheme.per_tensor("fp8_e4m3", round_mode="floor")
    result = quantize(x, scheme)
    expected = FormatBase.from_str("fp8_e4m3").quantize(x, GranularitySpec.per_tensor(), "floor")
    assert torch.allclose(result, expected, atol=1e-7)


def test_quantize_allow_denorm_false():
    """quantize with allow_denorm=False should flush subnormals to zero."""
    x = torch.tensor([1e-38, 1e-37, 1.0, 0.5])
    scheme = QuantScheme.per_tensor("bfloat16")
    result_denorm = quantize(x.clone(), scheme, allow_denorm=True)
    result_no_denorm = quantize(x.clone(), scheme, allow_denorm=False)
    # Subnormal values should be zeroed when allow_denorm=False
    assert result_no_denorm[0] == 0.0 or result_no_denorm[0].abs() < result_denorm[0].abs()


# ---------------------------------------------------------------------------
# 2. quantize_elemwise_op compat wrapper — produces same output as before
# ---------------------------------------------------------------------------

def test_quantize_elemwise_op_bfloat16():
    """Compat wrapper with bfloat=16 should match old behavior."""
    from src.specs.specs import finalize_mx_specs
    from mx.elemwise_ops import quantize_elemwise_op as old_op
    from mx.specs import finalize_mx_specs as old_finalize

    torch.manual_seed(42)
    x = torch.randn(4, 32)
    old_specs = old_finalize({"bfloat": 16})
    new_specs = finalize_mx_specs({"bfloat": 16})
    old_out = old_op(x.clone(), mx_specs=old_specs)
    new_out = quantize_elemwise_op(x.clone(), mx_specs=new_specs)
    assert torch.allclose(old_out, new_out, atol=1e-7), \
        f"max diff = {(old_out - new_out).abs().max()}"


def test_quantize_elemwise_op_bfloat12():
    """Compat wrapper with bfloat=12 should match old behavior."""
    from src.specs.specs import finalize_mx_specs
    from mx.elemwise_ops import quantize_elemwise_op as old_op
    from mx.specs import finalize_mx_specs as old_finalize

    torch.manual_seed(42)
    x = torch.randn(4, 32)
    old_specs = old_finalize({"bfloat": 12})
    new_specs = finalize_mx_specs({"bfloat": 12})
    old_out = old_op(x.clone(), mx_specs=old_specs)
    new_out = quantize_elemwise_op(x.clone(), mx_specs=new_specs)
    assert torch.allclose(old_out, new_out, atol=1e-7)


def test_quantize_elemwise_op_none():
    """Compat wrapper with mx_specs=None should pass through."""
    x = torch.randn(4, 32)
    result = quantize_elemwise_op(x.clone(), mx_specs=None)
    assert torch.equal(result, x)


def test_quantize_elemwise_op_bfloat_subnorms_false():
    """Compat wrapper with bfloat_subnorms=False should match old behavior."""
    from src.specs.specs import finalize_mx_specs
    from mx.elemwise_ops import quantize_elemwise_op as old_op
    from mx.specs import finalize_mx_specs as old_finalize

    torch.manual_seed(42)
    # Mix of normal and subnormal-range values
    x = torch.randn(4, 32) * 0.01
    old_specs = old_finalize({"bfloat": 16, "bfloat_subnorms": False})
    new_specs = finalize_mx_specs({"bfloat": 16, "bfloat_subnorms": False})
    old_out = old_op(x.clone(), mx_specs=old_specs)
    new_out = quantize_elemwise_op(x.clone(), mx_specs=new_specs)
    assert torch.allclose(old_out, new_out, atol=1e-7), \
        f"max diff = {(old_out - new_out).abs().max()}"


# ---------------------------------------------------------------------------
# 3. quantize() equivalence with old quantize_elemwise_op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
                                       "bfloat16", "float16"])
def test_quantize_equiv_old_elemwise_op(fmt_name):
    """quantize(x, per_tensor/fmt) should match old quantize_elemwise_op path."""
    from src.quantize.elemwise import _quantize_elemwise

    torch.manual_seed(42)
    x = torch.randn(4, 32)
    scheme = QuantScheme.per_tensor(fmt_name)
    result = quantize(x, scheme)
    expected = _quantize_elemwise(x, fmt_name, round_mode="nearest")
    assert torch.allclose(result, expected, atol=1e-7), \
        f"{fmt_name}: max diff = {(result - expected).abs().max()}"


# ---------------------------------------------------------------------------
# 4. _format_from_mx_specs helper
# ---------------------------------------------------------------------------

def test_format_from_mx_specs_bfloat16():
    """bfloat=16 in mx_specs should produce BFloat16Format."""
    from src.specs.specs import finalize_mx_specs
    from src.formats.bf16_fp16 import BFloat16Format
    specs = finalize_mx_specs({"bfloat": 16})
    fmt = _format_from_mx_specs(specs)
    assert fmt is not None
    assert isinstance(fmt, BFloat16Format)
    assert fmt.ebits == 8
    assert fmt.mbits == 9


def test_format_from_mx_specs_bfloat12():
    """bfloat=12 should produce ebits=8, mbits=5."""
    from src.specs.specs import finalize_mx_specs
    specs = finalize_mx_specs({"bfloat": 12})
    fmt = _format_from_mx_specs(specs)
    assert fmt is not None
    assert fmt.ebits == 8
    assert fmt.mbits == 5


def test_format_from_mx_specs_no_format():
    """bfloat=0, fp=0 should return None (finalize_mx_specs returns None)."""
    # When bfloat=0, finalize_mx_specs returns None (no quantization active)
    fmt = _format_from_mx_specs(None)
    assert fmt is None


def test_format_from_mx_specs_both_bfloat_and_fp_raises():
    """Setting both bfloat>0 and fp>0 should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot set both"):
        _format_from_mx_specs({"bfloat": 16, "fp": 8})


def test_format_from_mx_specs_bfloat_too_small_raises():
    """bfloat <= 9 should raise ValueError."""
    with pytest.raises(ValueError, match="bfloat"):
        _format_from_mx_specs({"bfloat": 8})


def test_format_from_mx_specs_fp_too_small_raises():
    """fp <= 6 should raise ValueError."""
    with pytest.raises(ValueError, match="fp"):
        _format_from_mx_specs({"fp": 5})
