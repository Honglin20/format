"""
MX block quantization equivalence tests: verify new src/quantize/mx_quantize.py
produces bit-identical output to old mx/mx_ops.py for every function.
"""
import pytest
import torch
from mx.mx_ops import (
    _shared_exponents as old_shared_exp,
    _reshape_to_blocks as old_reshape,
    _undo_reshape_to_blocks as old_undo,
    _quantize_mx as old_quantize_mx,
    quantize_mx_op as old_qmx_op,
)
from mx.formats import ElemFormat
from mx.specs import finalize_mx_specs as old_finalize
from src.tests.equivalence import assert_bit_identical
from src.scheme.quant_scheme import QuantScheme


# ---------------------------------------------------------------------------
# 1. _shared_exponents
# ---------------------------------------------------------------------------

def test_shared_exponents_max():
    from src.quantize.mx_quantize import _shared_exponents
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_shared_exp(A.clone(), method="max", axes=[-1])
    new_out = _shared_exponents(A.clone(), method="max", axes=[-1])
    assert torch.equal(old_out, new_out)


def test_shared_exponents_none():
    from src.quantize.mx_quantize import _shared_exponents
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_shared_exp(A.clone(), method="none")
    new_out = _shared_exponents(A.clone(), method="none")
    assert torch.equal(old_out, new_out)


def test_shared_exponents_multi_axes():
    from src.quantize.mx_quantize import _shared_exponents
    torch.manual_seed(42)
    A = torch.randn(2, 4, 64)
    old_out = old_shared_exp(A.clone(), method="max", axes=[-2, -1])
    new_out = _shared_exponents(A.clone(), method="max", axes=[-2, -1])
    assert torch.equal(old_out, new_out)


def test_shared_exponents_with_ebits():
    from src.quantize.mx_quantize import _shared_exponents
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_shared_exp(A.clone(), method="max", axes=[-1], ebits=4)
    new_out = _shared_exponents(A.clone(), method="max", axes=[-1], ebits=4)
    assert torch.equal(old_out, new_out)


def test_shared_exponents_zeros():
    from src.quantize.mx_quantize import _shared_exponents
    A = torch.zeros(4, 64)
    old_out = old_shared_exp(A.clone(), method="max", axes=[-1])
    new_out = _shared_exponents(A.clone(), method="max", axes=[-1])
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 2. _reshape_to_blocks / _undo_reshape_to_blocks
# ---------------------------------------------------------------------------

def test_reshape_undo_blocks_no_padding():
    from src.quantize.mx_quantize import _reshape_to_blocks
    A = torch.randn(2, 64)
    old_A, old_axes, old_orig, old_padded = old_reshape(A.clone(), axes=[-1], block_size=32)
    new_A, new_axes, new_orig, new_padded = _reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    assert torch.equal(old_A, new_A)


def test_reshape_undo_blocks_with_padding():
    from src.quantize.mx_quantize import _reshape_to_blocks
    A = torch.randn(2, 48)
    old_A, old_axes, old_orig, old_padded = old_reshape(A.clone(), axes=[-1], block_size=32)
    new_A, new_axes, new_orig, new_padded = _reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    assert torch.equal(old_A, new_A)


def test_reshape_undo_roundtrip_no_padding():
    from src.quantize.mx_quantize import _reshape_to_blocks, _undo_reshape_to_blocks
    A = torch.randn(2, 64)
    reshaped, axes, orig_shape, padded_shape = _reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    recovered = _undo_reshape_to_blocks(reshaped, padded_shape, orig_shape, axes)
    assert torch.equal(A, recovered)


def test_reshape_undo_roundtrip_with_padding():
    from src.quantize.mx_quantize import _reshape_to_blocks, _undo_reshape_to_blocks
    A = torch.randn(2, 48)
    reshaped, axes, orig_shape, padded_shape = _reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    recovered = _undo_reshape_to_blocks(reshaped, padded_shape, orig_shape, axes)
    assert torch.equal(A, recovered)


def test_reshape_undo_blocks_3d():
    from src.quantize.mx_quantize import _reshape_to_blocks
    A = torch.randn(2, 4, 64)
    old_A, old_axes, old_orig, old_padded = old_reshape(A.clone(), axes=[-1], block_size=32)
    new_A, new_axes, new_orig, new_padded = _reshape_to_blocks(A.clone(), axes=[-1], block_size=32)
    assert torch.equal(old_A, new_A)


# ---------------------------------------------------------------------------
# 3. _quantize_mx (core)
# ---------------------------------------------------------------------------

MX_FORMATS = ["fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3",
              "fp4_e2m1", "int8", "int4", "int2"]


@pytest.mark.parametrize("fmt", MX_FORMATS)
def test_quantize_mx(fmt):
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt),
                              block_size=32, axes=[-1], round="nearest")
    new_out = _quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                           block_size=32, axes=[-1], round_mode="nearest")
    assert torch.equal(old_out, new_out), f"mx quantize mismatch for {fmt}"


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "int8"])
def test_quantize_mx_block64(fmt):
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    A = torch.randn(4, 128)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt),
                              block_size=64, axes=[-1], round="nearest")
    new_out = _quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                           block_size=64, axes=[-1], round_mode="nearest")
    assert torch.equal(old_out, new_out)


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp4_e2m1"])
def test_quantize_mx_no_block(fmt):
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt),
                              block_size=0, axes=[-1], round="nearest")
    new_out = _quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                           block_size=0, axes=[-1], round_mode="nearest")
    assert torch.equal(old_out, new_out)


def test_quantize_mx_none_format():
    from src.quantize.mx_quantize import _quantize_mx
    A = torch.randn(4, 64)
    old_out = old_quantize_mx(A.clone(), scale_bits=8, elem_format=None,
                              block_size=32, axes=[-1])
    new_out = _quantize_mx(A.clone(), scale_bits=8, elem_format=None,
                           block_size=32, axes=[-1])
    assert torch.equal(old_out, new_out)


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "int8"])
def test_quantize_mx_floor_round(fmt):
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    old_out = old_quantize_mx(A.clone(), scale_bits=8,
                              elem_format=ElemFormat.from_str(fmt),
                              block_size=32, axes=[-1], round="floor")
    new_out = _quantize_mx(A.clone(), scale_bits=8, elem_format=fmt,
                           block_size=32, axes=[-1], round_mode="floor")
    assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# 4. quantize_mx(scheme) equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", MX_FORMATS)
def test_quantize_mx_scheme(fmt):
    from src.quantize.mx_quantize import quantize_mx
    torch.manual_seed(42)
    A = torch.randn(4, 64)
    config = {"w_elem_format": fmt, "a_elem_format": fmt, "block_size": 32, "bfloat": 16}
    old_specs = old_finalize(config.copy())
    scheme = QuantScheme.mxfp(fmt, block_size=32)
    old_out = old_qmx_op(A.clone(), mx_specs=old_specs, elem_format=fmt, axes=[-1])
    new_out = quantize_mx(A.clone(), scheme=scheme, axes=[-1])
    assert torch.equal(old_out, new_out), f"mx scheme mismatch for {fmt}"
