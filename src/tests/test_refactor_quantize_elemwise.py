"""
Verify quantize_elemwise refactoring: FormatBase.quantize_elemwise() produces
bit-identical output to old mx/ _quantize_elemwise_core for every format,
round mode, and pipeline path (per-tensor, per-channel, per-block MX).

Targeted coverage of gaps identified during 2026-04-26 refactoring review.
"""
import pytest
import torch
from mx.elemwise_ops import _quantize_elemwise_core as old_core
from mx.mx_ops import _quantize_mx as old_quantize_mx, quantize_mx_op as old_qmx_op
from mx.formats import ElemFormat
from mx.specs import finalize_mx_specs as old_finalize
from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec
from src.scheme.quant_scheme import QuantScheme
from src.tests.equivalence import assert_bit_identical


# ===========================================================================
# 1. FormatBase.quantize_elemwise() — direct equivalence vs old core
# ===========================================================================

ALL_FORMATS = ["int8", "int4", "int2",
               "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1",
               "float16", "bfloat16"]

ROUND_MODES = ["nearest", "floor", "even"]

FLOAT_FORMATS = ["fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1",
                 "float16", "bfloat16"]


@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
@pytest.mark.parametrize("round_mode", ROUND_MODES)
def test_quantize_elemwise_vs_old_core(fmt_name, round_mode):
    """FormatBase.quantize_elemwise() == old _quantize_elemwise_core() for all formats."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round=round_mode)
    new_out = fmt.quantize_elemwise(A.clone(), round_mode=round_mode)

    assert torch.equal(old_out, new_out), \
        f"quantize_elemwise mismatch: {fmt_name}/{round_mode}"


@pytest.mark.parametrize("fmt_name", FLOAT_FORMATS)
def test_quantize_elemwise_no_denorm_vs_old(fmt_name):
    """quantize_elemwise(allow_denorm=False) == old core(allow_denorm=False)."""
    torch.manual_seed(42)
    A = torch.randn(4, 64) * 0.01  # small values to hit denorm range
    fmt = FormatBase.from_str(fmt_name)

    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round="nearest", allow_denorm=False)
    new_out = fmt.quantize_elemwise(A.clone(), round_mode="nearest",
                                    allow_denorm=False)

    assert torch.equal(old_out, new_out), \
        f"quantize_elemwise no_denorm mismatch: {fmt_name}"


@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
def test_quantize_elemwise_saturate_normals_true(fmt_name):
    """quantize_elemwise(saturate_normals=True) == old core(saturate_normals=True)."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round="nearest", saturate_normals=True)
    new_out = fmt.quantize_elemwise(A.clone(), round_mode="nearest",
                                    saturate_normals=True)

    assert torch.equal(old_out, new_out), \
        f"quantize_elemwise saturate mismatch: {fmt_name}"


@pytest.mark.parametrize("fmt_name", FLOAT_FORMATS)
def test_quantize_elemwise_saturate_normals_false(fmt_name):
    """quantize_elemwise(saturate_normals=False) == old core(saturate_normals=False)
    for float formats (can produce Inf)."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round="nearest", saturate_normals=False)
    new_out = fmt.quantize_elemwise(A.clone(), round_mode="nearest",
                                    saturate_normals=False)

    assert torch.equal(old_out, new_out), \
        f"quantize_elemwise no-saturate mismatch: {fmt_name}"


# ===========================================================================
# 2. Integer format default saturate_normals (ebits==0 → saturate_normals=True)
# ===========================================================================

@pytest.mark.parametrize("fmt_name", ["int8", "int4", "int2"])
def test_integer_format_defaults_to_saturate(fmt_name):
    """Integer formats default saturate_normals=True via quantize_elemwise."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    # Default (saturate_normals=None → True for ints)
    result_default = fmt.quantize_elemwise(A.clone())
    result_explicit = fmt.quantize_elemwise(A.clone(), saturate_normals=True)

    assert torch.equal(result_default, result_explicit), \
        f"Integer {fmt_name} should default saturate_normals=True"


@pytest.mark.parametrize("fmt_name", FLOAT_FORMATS)
def test_float_format_defaults_to_no_saturate(fmt_name):
    """Float formats default saturate_normals=False via quantize_elemwise."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    result_default = fmt.quantize_elemwise(A.clone())
    result_explicit = fmt.quantize_elemwise(A.clone(), saturate_normals=False)

    assert torch.equal(result_default, result_explicit), \
        f"Float {fmt_name} should default saturate_normals=False"


# ===========================================================================
# 3. PER_TENSOR path: FormatBase.quantize(x, per_tensor) == old _quantize_elemwise_core
# ===========================================================================

@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
@pytest.mark.parametrize("round_mode", ROUND_MODES)
def test_per_tensor_via_quantize_vs_old_core(fmt_name, round_mode):
    """format.quantize(x, per_tensor, round) == old _quantize_elemwise_core()."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    new_out = fmt.quantize(A.clone(), GranularitySpec.per_tensor(), round_mode)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round=round_mode)

    assert torch.equal(old_out, new_out), \
        f"per_tensor quantize mismatch: {fmt_name}/{round_mode}"


# ===========================================================================
# 4. PER_BLOCK (MX) path: FormatBase.quantize(x, per_block) == old _quantize_mx
# ===========================================================================

@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
def test_per_block_via_quantize_vs_old_mx(fmt_name):
    """format.quantize(x, per_block, round) == old _quantize_mx()."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    new_out = fmt.quantize(A.clone(), GranularitySpec.per_block(32), "nearest")
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt_name),
                              block_size=32, axes=[-1], round="nearest")

    assert torch.equal(old_out, new_out), \
        f"per_block quantize mismatch: {fmt_name}"


@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8", "fp4_e2m1"])
@pytest.mark.parametrize("round_mode", ["nearest", "floor", "even"])
def test_per_block_via_quantize_all_round_modes(fmt_name, round_mode):
    """MX path through format.quantize() works for all round modes."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    new_out = fmt.quantize(A.clone(), GranularitySpec.per_block(32), round_mode)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt_name),
                              block_size=32, axes=[-1], round=round_mode)

    assert torch.equal(old_out, new_out), \
        f"per_block/{round_mode} mismatch: {fmt_name}"


@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
def test_per_block_dither_round(fmt_name):
    """MX path with dither round mode — same seed, same output."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    torch.manual_seed(99)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt_name),
                              block_size=32, axes=[-1], round="dither")
    torch.manual_seed(99)
    new_out = fmt.quantize(A.clone(), GranularitySpec.per_block(32), "dither")

    assert torch.equal(old_out, new_out), \
        f"per_block/dither mismatch: {fmt_name}"


def test_per_block_block64():
    """MX with block_size=64."""
    torch.manual_seed(42)
    A = torch.randn(4, 128)
    fmt = FormatBase.from_str("fp8_e4m3")

    new_out = fmt.quantize(A.clone(), GranularitySpec.per_block(64), "nearest")
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str("fp8_e4m3"),
                              block_size=64, axes=[-1], round="nearest")

    assert torch.equal(old_out, new_out)


def test_per_block_negative_axis():
    """MX with negative block_axis."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str("fp8_e4m3")

    # axis=-1 (last dim) via GranularitySpec
    new_neg = fmt.quantize(A.clone(),
                           GranularitySpec.per_block(32, axis=-1), "nearest")
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str("fp8_e4m3"),
                              block_size=32, axes=[-1], round="nearest")

    assert torch.equal(old_out, new_neg)


def test_per_block_shared_exp_none():
    """MX with shared_exp_method='none'."""
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str("fp8_e4m3")

    new_out = _quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                           block_size=32, axes=[-1], round_mode="nearest",
                           shared_exp_method="none")
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str("fp8_e4m3"),
                              block_size=32, axes=[-1], round="nearest",
                              shared_exp_method="none")

    assert torch.equal(old_out, new_out)


def test_per_block_flush_fp32_subnorms():
    """MX with flush_fp32_subnorms=True."""
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str("fp8_e4m3")

    new_out = _quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                           block_size=32, axes=[-1], round_mode="nearest",
                           flush_fp32_subnorms=True)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str("fp8_e4m3"),
                              block_size=32, axes=[-1], round="nearest",
                              flush_fp32_subnorms=True)

    assert torch.equal(old_out, new_out)


# ===========================================================================
# 5. Full pipeline: quantize(x, scheme) vs old paths
# ===========================================================================

@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
def test_full_pipeline_per_tensor_vs_old(fmt_name):
    """quantize(x, QuantScheme.per_tensor(fmt)) == old _quantize_elemwise_core."""
    from src.quantize.elemwise import quantize
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    scheme = QuantScheme.per_tensor(fmt_name)
    fmt = FormatBase.from_str(fmt_name)

    new_out = quantize(A.clone(), scheme)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round="nearest")

    assert torch.equal(old_out, new_out), \
        f"full pipeline per_tensor mismatch: {fmt_name}"


@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
def test_full_pipeline_mx_vs_old(fmt_name):
    """quantize(x, QuantScheme.mxfp(fmt, 32)) == old _quantize_mx."""
    from src.quantize.elemwise import quantize
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    scheme = QuantScheme.mxfp(fmt_name, block_size=32)

    new_out = quantize(A.clone(), scheme)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt_name),
                              block_size=32, axes=[-1], round="nearest")

    assert torch.equal(old_out, new_out), \
        f"full pipeline mx mismatch: {fmt_name}"


# ===========================================================================
# 6. Edge cases: zeros, Inf, NaN, sparse through quantize_elemwise
# ===========================================================================

@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
def test_quantize_elemwise_zeros(fmt_name):
    """Zero input → zero output through quantize_elemwise."""
    A = torch.zeros(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    result = fmt.quantize_elemwise(A)
    assert (result == 0).all(), f"Zeros not preserved for {fmt_name}"


def test_quantize_elemwise_inf_nan_passthrough():
    """Inf/NaN passthrough through quantize_elemwise."""
    A = torch.tensor([1.0, float("inf"), -float("inf"), float("nan"), 0.0])
    fmt = FormatBase.from_str("fp8_e5m2")

    result = fmt.quantize_elemwise(A)
    assert result[1] == float("inf")
    assert result[2] == -float("inf")
    assert torch.isnan(result[3])


# ===========================================================================
# 7. MX parameter propagation: _quantize_mx via FormatBase.quantize
#    verifies that round_mode flows correctly through the refactored path
# ===========================================================================

@pytest.mark.parametrize("fmt_name", ALL_FORMATS)
@pytest.mark.parametrize("round_mode", ROUND_MODES)
def test_mx_round_mode_propagation(fmt_name, round_mode):
    """round_mode propagates: format.quantize(PER_BLOCK, round_mode) → _quantize_mx
    → fmt.quantize_elemwise(round_mode) → _quantize_elemwise_core(round_mode)."""
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)

    # Through FormatBase.quantize (per-block dispatch)
    via_quantize = fmt.quantize(A.clone(), GranularitySpec.per_block(32),
                                round_mode)
    # Direct _quantize_mx call
    from src.quantize.mx_quantize import _quantize_mx
    via_mx_direct = _quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                                 block_size=32, axes=[-1],
                                 round_mode=round_mode)

    assert torch.equal(via_quantize, via_mx_direct), \
        f"MX round_mode propagation mismatch: {fmt_name}/{round_mode}"


# ===========================================================================
# 8. Custom format: verify quantize_elemwise override works in MX path
# ===========================================================================

def test_custom_format_override_in_mx_path():
    """A format that overrides quantize_elemwise has its override called
    during MX quantization, not the base implementation."""
    from src.formats.int_formats import IntFormat

    call_log = []

    class TrackingIntFormat(IntFormat):
        """IntFormat that tracks whether quantize_elemwise was called."""
        def quantize_elemwise(self, x, round_mode="nearest", allow_denorm=True,
                              saturate_normals=None):
            call_log.append("quantize_elemwise")
            return super().quantize_elemwise(x, round_mode, allow_denorm,
                                             saturate_normals)

        def quantize(self, x, granularity, round_mode="nearest",
                     allow_denorm=True):
            return super().quantize(x, granularity, round_mode, allow_denorm)

    fmt = TrackingIntFormat(bits=8)
    x = torch.randn(4, 32)

    # Per-tensor calls quantize_elemwise once
    call_log.clear()
    fmt.quantize(x.clone(), GranularitySpec.per_tensor(), "nearest")
    assert len(call_log) == 1, f"per_tensor: expected 1 call, got {len(call_log)}"

    # Per-block (MX) should also call quantize_elemwise
    call_log.clear()
    fmt.quantize(x.clone(), GranularitySpec.per_block(32), "nearest")
    assert len(call_log) >= 1, \
        f"per_block (MX): expected >=1 quantize_elemwise call, got {len(call_log)}"

    # Per-channel calls quantize_elemwise once on the normalized tensor
    call_log.clear()
    fmt.quantize(x.clone(), GranularitySpec.per_channel(axis=0), "nearest")
    assert len(call_log) == 1, \
        f"per_channel: expected 1 call (normalizes then elemwise once), got {len(call_log)}"
