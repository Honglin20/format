"""
Equivalence tests for src/quantize/vector.py vs mx/vector_ops.py.

TDD: Every function tested against old code — bit-identical output required.
"""
import pytest
import torch

from mx import vector_ops as old_vec
from src.quantize import vector
from mx.specs import finalize_mx_specs as old_finalize
from src.specs.specs import finalize_mx_specs as new_finalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BFLOAT16_CONFIG = {"bfloat": 16}


def _old_specs():
    return old_finalize(BFLOAT16_CONFIG.copy())


def _new_specs():
    return new_finalize(BFLOAT16_CONFIG.copy())


# ---------------------------------------------------------------------------
# vec_quantize
# ---------------------------------------------------------------------------

class TestVecQuantize:

    def test_vec_quantize(self):
        torch.manual_seed(42)
        A = torch.randn(4, 32)
        old_out = old_vec.vec_quantize(A.clone(), mx_specs=_old_specs())
        new_out = vector.vec_quantize(A.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_quantize_none_specs(self):
        torch.manual_seed(42)
        A = torch.randn(4, 32)
        old_out = old_vec.vec_quantize(A.clone(), mx_specs=None)
        new_out = vector.vec_quantize(A.clone(), mx_specs=None)
        assert torch.equal(old_out, new_out)

    def test_vec_quantize_explicit_round(self):
        torch.manual_seed(42)
        A = torch.randn(4, 32)
        old_out = old_vec.vec_quantize(A.clone(), mx_specs=_old_specs(), round="floor")
        new_out = vector.vec_quantize(A.clone(), mx_specs=_new_specs(), round_mode="floor")
        assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

class TestVecArithmetic:

    def test_vec_add(self):
        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        old_out = old_vec.vec_add(a.clone(), b.clone(), mx_specs=_old_specs())
        new_out = vector.vec_add(a.clone(), b.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_sub(self):
        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        old_out = old_vec.vec_sub(a.clone(), b.clone(), mx_specs=_old_specs())
        new_out = vector.vec_sub(a.clone(), b.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_mul(self):
        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        old_out = old_vec.vec_mul(a.clone(), b.clone(), mx_specs=_old_specs())
        new_out = vector.vec_mul(a.clone(), b.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_div(self):
        torch.manual_seed(42)
        a, b = torch.randn(4, 8) + 2.0, torch.randn(4, 8) + 2.0  # avoid div-by-zero
        old_out = old_vec.vec_div(a.clone(), b.clone(), mx_specs=_old_specs())
        new_out = vector.vec_div(a.clone(), b.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_div_with_recip(self):
        """vec_use_recip=True changes division path."""
        torch.manual_seed(42)
        a, b = torch.randn(4, 8) + 2.0, torch.randn(4, 8) + 2.0
        old_specs = old_finalize({"bfloat": 16, "vec_use_recip": True})
        new_specs = new_finalize({"bfloat": 16, "vec_use_recip": True})
        old_out = old_vec.vec_div(a.clone(), b.clone(), mx_specs=old_specs)
        new_out = vector.vec_div(a.clone(), b.clone(), mx_specs=new_specs)
        assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# Special math ops
# ---------------------------------------------------------------------------

class TestVecSpecial:

    def test_vec_exp(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_exp(A.clone(), mx_specs=_old_specs())
        new_out = vector.vec_exp(A.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_exp_with_exp2(self):
        """vec_use_exp2=True changes exp path."""
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_specs = old_finalize({"bfloat": 16, "vec_use_exp2": True})
        new_specs = new_finalize({"bfloat": 16, "vec_use_exp2": True})
        old_out = old_vec.vec_exp(A.clone(), mx_specs=old_specs)
        new_out = vector.vec_exp(A.clone(), mx_specs=new_specs)
        assert torch.equal(old_out, new_out)

    def test_vec_exp2(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_exp2(A.clone(), mx_specs=_old_specs())
        new_out = vector.vec_exp2(A.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_recip(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8) + 2.0  # avoid div-by-zero
        old_out = old_vec.vec_recip(A.clone(), mx_specs=_old_specs())
        new_out = vector.vec_recip(A.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_sqrt(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8).abs() + 0.1  # positive only
        old_out = old_vec.vec_sqrt(A.clone(), mx_specs=_old_specs())
        new_out = vector.vec_sqrt(A.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_tanh(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_tanh(A.clone(), mx_specs=_old_specs())
        new_out = vector.vec_tanh(A.clone(), mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# Reduce ops
# ---------------------------------------------------------------------------

class TestVecReduce:

    def test_vec_reduce_sum(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_sum(A.clone(), dim=-1, mx_specs=_old_specs())
        new_out = vector.vec_reduce_sum(A.clone(), dim=-1, mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_reduce_sum_keepdim(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_sum(A.clone(), dim=-1, keepdim=True, mx_specs=_old_specs())
        new_out = vector.vec_reduce_sum(A.clone(), dim=-1, keepdim=True, mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_reduce_mean(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_mean(A.clone(), dim=-1, mx_specs=_old_specs())
        new_out = vector.vec_reduce_mean(A.clone(), dim=-1, mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)

    def test_vec_reduce_mean_keepdim(self):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_mean(A.clone(), dim=-1, keepdim=True, mx_specs=_old_specs())
        new_out = vector.vec_reduce_mean(A.clone(), dim=-1, keepdim=True, mx_specs=_new_specs())
        assert torch.equal(old_out, new_out)
