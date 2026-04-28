"""
Element-wise quantization equivalence tests: verify new src/quantize/elemwise.py
produces bit-identical output to old mx/elemwise_ops.py for every function.
"""
import pytest
import torch
from mx.elemwise_ops import (
    _round_mantissa as old_round_mantissa,
    _safe_lshift as old_lshift,
    _safe_rshift as old_rshift,
    _quantize_elemwise_core as old_core,
    _quantize_elemwise as old_quantize_elemwise,
    _quantize_bfloat as old_qbf,
    _quantize_fp as old_qfp,
    quantize_elemwise_op as old_quantize_op,
)
from mx.formats import ElemFormat
from mx.specs import finalize_mx_specs as old_finalize
from src.tests.equivalence import assert_bit_identical
from src.scheme.quant_scheme import QuantScheme
from src.formats.fp_formats import FPFormat


# ---------------------------------------------------------------------------
# 1. round_mantissa
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("round_mode", ["nearest", "floor", "even"])
def test_round_mantissa(round_mode):
    from src.quantize.elemwise import _round_mantissa
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_out = old_round_mantissa(A.clone(), bits=4, round=round_mode)
    new_out = _round_mantissa(A.clone(), bits=4, round_mode=round_mode)
    assert torch.equal(old_out, new_out), f"round_mantissa mismatch for {round_mode}"


def test_round_mantissa_dither():
    from src.quantize.elemwise import _round_mantissa
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    # Dither uses random noise, so set same seed before each call
    torch.manual_seed(99)
    old_out = old_round_mantissa(A.clone(), bits=4, round="dither")
    torch.manual_seed(99)
    new_out = _round_mantissa(A.clone(), bits=4, round_mode="dither")
    assert torch.equal(old_out, new_out), "round_mantissa mismatch for dither"


def test_round_mantissa_clamp():
    from src.quantize.elemwise import _round_mantissa
    A = torch.tensor([10.0, -10.0, 0.0, 3.5])
    old_out = old_round_mantissa(A.clone(), bits=4, round="nearest", clamp=True)
    new_out = _round_mantissa(A.clone(), bits=4, round_mode="nearest", clamp=True)
    assert torch.equal(old_out, new_out)


def test_round_mantissa_tie_breaking():
    from src.quantize.elemwise import _round_mantissa
    A = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5])
    old_out = old_round_mantissa(A.clone(), bits=4, round="even")
    new_out = _round_mantissa(A.clone(), bits=4, round_mode="even")
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 2. safe_lshift / safe_rshift
# ---------------------------------------------------------------------------

def test_safe_lshift():
    from src.quantize.elemwise import _safe_lshift
    A = torch.tensor([1.0, 2.0, 0.5, 0.0])
    exp = torch.tensor([2.0, 1.0, 0.0, -1.0])
    old_out = old_lshift(A.clone(), bits=3, exp=exp)
    new_out = _safe_lshift(A.clone(), bits=3, exp=exp)
    assert torch.equal(old_out, new_out)


def test_safe_rshift():
    from src.quantize.elemwise import _safe_rshift
    A = torch.tensor([8.0, 4.0, 2.0, 0.0])
    exp = torch.tensor([2.0, 1.0, 0.0, -1.0])
    old_out = old_rshift(A.clone(), bits=3, exp=exp)
    new_out = _safe_rshift(A.clone(), bits=3, exp=exp)
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 3. quantize_elemwise_core
# ---------------------------------------------------------------------------

ALL_ELEM_FORMATS = ["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3",
                    "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "float16", "bfloat16"]


@pytest.mark.parametrize("fmt_name", ALL_ELEM_FORMATS)
def test_quantize_elemwise_core_normal(fmt_name):
    from src.quantize.elemwise import _quantize_elemwise_core
    from src.formats.base import FormatBase
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round="nearest")
    new_out = _quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round_mode="nearest")
    assert torch.equal(old_out, new_out), f"core mismatch for {fmt_name}"


@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1", "int8"])
def test_quantize_elemwise_core_no_denorm(fmt_name):
    from src.quantize.elemwise import _quantize_elemwise_core
    from src.formats.base import FormatBase
    torch.manual_seed(42)
    A = torch.randn(4, 32) * 0.01
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round="nearest", allow_denorm=False)
    new_out = _quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                                     round_mode="nearest", allow_denorm=False)
    assert torch.equal(old_out, new_out)


@pytest.mark.parametrize("round_mode", ["nearest", "floor", "even"])
@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8", "fp4_e2m1"])
def test_quantize_elemwise_core_round_modes(fmt_name, round_mode):
    from src.quantize.elemwise import _quantize_elemwise_core
    from src.formats.base import FormatBase
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round=round_mode)
    new_out = _quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round_mode=round_mode)
    assert torch.equal(old_out, new_out), f"core round mismatch for {fmt_name}/{round_mode}"


def test_quantize_elemwise_core_nan_inf():
    from src.quantize.elemwise import _quantize_elemwise_core
    A = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, 1.0])
    old_out = old_core(A.clone(), 5, 4, 448.0, round="nearest")
    new_out = _quantize_elemwise_core(A.clone(), 5, 4, 448.0, round_mode="nearest")
    assert_bit_identical(old_out, new_out, name="core_nan_inf")


def test_quantize_elemwise_core_zeros():
    from src.quantize.elemwise import _quantize_elemwise_core
    A = torch.zeros(4, 32)
    old_out = old_core(A.clone(), 5, 4, 448.0, round="nearest")
    new_out = _quantize_elemwise_core(A.clone(), 5, 4, 448.0, round_mode="nearest")
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 4. quantize_elemwise_op (public API)
# ---------------------------------------------------------------------------

MX_CONFIGS = [
    ("bfloat16", QuantScheme.per_tensor("bfloat16"), {"bfloat": 16}),
    ("bfloat12", QuantScheme(format=FPFormat(name="bfloat12", ebits=8, mbits=5), round_mode="nearest"), {"bfloat": 12}),
]


@pytest.mark.parametrize("name,scheme,old_config", MX_CONFIGS, ids=[c[0] for c in MX_CONFIGS])
def test_quantize_elemwise_op(name, scheme, old_config):
    from src.quantize.elemwise import quantize
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_specs = old_finalize(old_config.copy())
    old_out = old_quantize_op(A.clone(), mx_specs=old_specs)
    new_out = quantize(A.clone(), scheme)
    assert torch.equal(old_out, new_out), f"elemwise_op mismatch for {name}"


def test_quantize_elemwise_op_none():
    from src.quantize.elemwise import quantize
    A = torch.randn(4, 32)
    old_out = old_quantize_op(A.clone(), mx_specs=None)
    new_out = quantize(A.clone(), scheme=None)
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 5. _quantize_elemwise (format-name wrapper)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1", "int8", "fp6_e3m2"])
def test_quantize_elemwise(fmt_name):
    from src.quantize.elemwise import _quantize_elemwise
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_out = old_quantize_elemwise(A.clone(), elem_format=ElemFormat.from_str(fmt_name),
                                    round="nearest")
    new_out = _quantize_elemwise(A.clone(), elem_format=fmt_name, round_mode="nearest")
    assert torch.equal(old_out, new_out), f"_quantize_elemwise mismatch for {fmt_name}"


def test_quantize_elemwise_none():
    from src.quantize.elemwise import _quantize_elemwise
    A = torch.randn(4, 32)
    old_out = old_quantize_elemwise(A.clone(), elem_format=None)
    new_out = _quantize_elemwise(A.clone(), elem_format=None)
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 6. _quantize_bfloat (direct)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bfloat_bits", [16, 12])
def test_quantize_bfloat_direct(bfloat_bits):
    from src.quantize.elemwise import _quantize_bfloat
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_out = old_qbf(A.clone(), bfloat=bfloat_bits, round="nearest")
    new_out = _quantize_bfloat(A.clone(), bfloat=bfloat_bits, round_mode="nearest")
    assert torch.equal(old_out, new_out), f"_quantize_bfloat mismatch for bfloat{bfloat_bits}"


# ---------------------------------------------------------------------------
# 7. _quantize_fp (direct)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_bits,mantissa_bits", [(5, 1), (5, 3), (5, 5)])
def test_quantize_fp_direct(exp_bits, mantissa_bits):
    from src.quantize.elemwise import _quantize_fp
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_out = old_qfp(A.clone(), exp_bits=exp_bits, mantissa_bits=mantissa_bits,
                      round="nearest")
    new_out = _quantize_fp(A.clone(), exp_bits=exp_bits, mantissa_bits=mantissa_bits,
                           round_mode="nearest")
    assert torch.equal(old_out, new_out), \
        f"_quantize_fp mismatch for exp={exp_bits}, mantissa={mantissa_bits}"


def test_quantize_fp_none():
    from src.quantize.elemwise import _quantize_fp
    A = torch.randn(4, 32)
    old_out = old_qfp(A.clone(), exp_bits=None, mantissa_bits=None)
    new_out = _quantize_fp(A.clone(), exp_bits=None, mantissa_bits=None)
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 8. Sparse tensor (bug fix verification)
# ---------------------------------------------------------------------------

def test_quantize_elemwise_core_sparse():
    from src.quantize.elemwise import _quantize_elemwise_core
    indices = torch.tensor([[0, 0, 1, 1], [0, 3, 1, 2]])
    values = torch.tensor([1.5, -0.5, 3.0, -2.0])
    A = torch.sparse_coo_tensor(indices, values, size=(2, 4))
    # Old code has UnboundLocalError on sparse path, so only test new code
    out = _quantize_elemwise_core(A, bits=5, exp_bits=4, max_norm=448.0,
                                  round_mode="nearest")
    assert out.is_sparse, "Output should be sparse"
    assert out.shape == A.shape
