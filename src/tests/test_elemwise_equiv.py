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
    _quantize_bfloat as old_qbf,
    _quantize_fp as old_qfp,
    quantize_elemwise_op as old_quantize_op,
)
from mx.specs import finalize_mx_specs as old_finalize
from src.tests.equivalence import assert_bit_identical


# ---------------------------------------------------------------------------
# 1. round_mantissa
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("round_mode", ["nearest", "floor", "even"])
def test_round_mantissa(round_mode):
    from src.quantize.elemwise import round_mantissa
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_out = old_round_mantissa(A.clone(), bits=4, round=round_mode)
    new_out = round_mantissa(A.clone(), bits=4, round_mode=round_mode)
    assert torch.equal(old_out, new_out), f"round_mantissa mismatch for {round_mode}"


def test_round_mantissa_dither():
    from src.quantize.elemwise import round_mantissa
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    # Dither uses random noise, so set same seed before each call
    torch.manual_seed(99)
    old_out = old_round_mantissa(A.clone(), bits=4, round="dither")
    torch.manual_seed(99)
    new_out = round_mantissa(A.clone(), bits=4, round_mode="dither")
    assert torch.equal(old_out, new_out), "round_mantissa mismatch for dither"


def test_round_mantissa_clamp():
    from src.quantize.elemwise import round_mantissa
    A = torch.tensor([10.0, -10.0, 0.0, 3.5])
    old_out = old_round_mantissa(A.clone(), bits=4, round="nearest", clamp=True)
    new_out = round_mantissa(A.clone(), bits=4, round_mode="nearest", clamp=True)
    assert torch.equal(old_out, new_out)


def test_round_mantissa_tie_breaking():
    from src.quantize.elemwise import round_mantissa
    A = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5])
    old_out = old_round_mantissa(A.clone(), bits=4, round="even")
    new_out = round_mantissa(A.clone(), bits=4, round_mode="even")
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 2. safe_lshift / safe_rshift
# ---------------------------------------------------------------------------

def test_safe_lshift():
    from src.quantize.elemwise import safe_lshift
    A = torch.tensor([1.0, 2.0, 0.5, 0.0])
    exp = torch.tensor([2.0, 1.0, 0.0, -1.0])
    old_out = old_lshift(A.clone(), bits=3, exp=exp)
    new_out = safe_lshift(A.clone(), bits=3, exp=exp)
    assert torch.equal(old_out, new_out)


def test_safe_rshift():
    from src.quantize.elemwise import safe_rshift
    A = torch.tensor([8.0, 4.0, 2.0, 0.0])
    exp = torch.tensor([2.0, 1.0, 0.0, -1.0])
    old_out = old_rshift(A.clone(), bits=3, exp=exp)
    new_out = safe_rshift(A.clone(), bits=3, exp=exp)
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 3. quantize_elemwise_core
# ---------------------------------------------------------------------------

ALL_ELEM_FORMATS = ["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3",
                    "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "float16", "bfloat16"]


@pytest.mark.parametrize("fmt_name", ALL_ELEM_FORMATS)
def test_quantize_elemwise_core_normal(fmt_name):
    from src.quantize.elemwise import quantize_elemwise_core
    from src.formats.base import FormatBase
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round="nearest")
    new_out = quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round_mode="nearest")
    assert torch.equal(old_out, new_out), f"core mismatch for {fmt_name}"


@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1", "int8"])
def test_quantize_elemwise_core_no_denorm(fmt_name):
    from src.quantize.elemwise import quantize_elemwise_core
    from src.formats.base import FormatBase
    torch.manual_seed(42)
    A = torch.randn(4, 32) * 0.01
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                       round="nearest", allow_denorm=False)
    new_out = quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm,
                                     round_mode="nearest", allow_denorm=False)
    assert torch.equal(old_out, new_out)


@pytest.mark.parametrize("round_mode", ["nearest", "floor", "even"])
@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8", "fp4_e2m1"])
def test_quantize_elemwise_core_round_modes(fmt_name, round_mode):
    from src.quantize.elemwise import quantize_elemwise_core
    from src.formats.base import FormatBase
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    fmt = FormatBase.from_str(fmt_name)
    old_out = old_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round=round_mode)
    new_out = quantize_elemwise_core(A.clone(), fmt.mbits, fmt.ebits, fmt.max_norm, round_mode=round_mode)
    assert torch.equal(old_out, new_out), f"core round mismatch for {fmt_name}/{round_mode}"


def test_quantize_elemwise_core_nan_inf():
    from src.quantize.elemwise import quantize_elemwise_core
    A = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, 1.0])
    old_out = old_core(A.clone(), 5, 4, 448.0, round="nearest")
    new_out = quantize_elemwise_core(A.clone(), 5, 4, 448.0, round_mode="nearest")
    assert_bit_identical(old_out, new_out, name="core_nan_inf")


def test_quantize_elemwise_core_zeros():
    from src.quantize.elemwise import quantize_elemwise_core
    A = torch.zeros(4, 32)
    old_out = old_core(A.clone(), 5, 4, 448.0, round="nearest")
    new_out = quantize_elemwise_core(A.clone(), 5, 4, 448.0, round_mode="nearest")
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 4. quantize_elemwise_op (public API)
# ---------------------------------------------------------------------------

MX_CONFIGS = [
    {"bfloat": 16},
    {"bfloat": 12},
    {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16},
]


@pytest.mark.parametrize("config", MX_CONFIGS)
def test_quantize_elemwise_op(config):
    from src.quantize.elemwise import quantize_elemwise_op
    from src.specs.specs import finalize_mx_specs as new_finalize
    torch.manual_seed(42)
    A = torch.randn(4, 32)
    old_specs = old_finalize(config.copy())
    new_specs = new_finalize(config.copy())
    old_out = old_quantize_op(A.clone(), mx_specs=old_specs)
    new_out = quantize_elemwise_op(A.clone(), mx_specs=new_specs)
    assert torch.equal(old_out, new_out), f"elemwise_op mismatch for {config}"


def test_quantize_elemwise_op_none():
    from src.quantize.elemwise import quantize_elemwise_op
    A = torch.randn(4, 32)
    old_out = old_quantize_op(A.clone(), mx_specs=None)
    new_out = quantize_elemwise_op(A.clone(), mx_specs=None)
    assert torch.equal(old_out, new_out)
