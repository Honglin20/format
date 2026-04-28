"""
Equivalence tests for src/quantize/vector.py vs mx/vector_ops.py.

TDD: Every function tested against old code — bit-identical output required.
"""
import pytest
import torch

from mx import vector_ops as old_vec
from src.quantize import vector
from src.scheme.quant_scheme import QuantScheme


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _scheme():
    return QuantScheme.per_tensor("bfloat16")


# ---------------------------------------------------------------------------
# vec_quantize
# ---------------------------------------------------------------------------

class TestVecQuantize:

    def test_vec_quantize(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 32)
        old_out = old_vec.vec_quantize(A.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_quantize(A.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_quantize_none_scheme(self):
        torch.manual_seed(42)
        A = torch.randn(4, 32)
        old_out = old_vec.vec_quantize(A.clone(), mx_specs=None)
        new_out = vector.vec_quantize(A.clone(), scheme=None)
        assert torch.equal(old_out, new_out)

    def test_vec_quantize_explicit_round(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 32)
        old_out = old_vec.vec_quantize(A.clone(), mx_specs=old_finalize({"bfloat": 16}), round="floor")
        scheme_floor = QuantScheme.per_tensor("bfloat16", round_mode="floor")
        new_out = vector.vec_quantize(A.clone(), scheme=scheme_floor)
        assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

class TestVecArithmetic:

    def test_vec_add(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        old_out = old_vec.vec_add(a.clone(), b.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_add(a.clone(), b.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_sub(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        old_out = old_vec.vec_sub(a.clone(), b.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_sub(a.clone(), b.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_mul(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        old_out = old_vec.vec_mul(a.clone(), b.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_mul(a.clone(), b.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_div(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        a, b = torch.randn(4, 8) + 2.0, torch.randn(4, 8) + 2.0  # avoid div-by-zero
        old_out = old_vec.vec_div(a.clone(), b.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_div(a.clone(), b.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_div_with_recip(self):
        """vec_use_recip=True changes division path."""
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        a, b = torch.randn(4, 8) + 2.0, torch.randn(4, 8) + 2.0
        old_specs = old_finalize({"bfloat": 16, "vec_use_recip": True})
        old_out = old_vec.vec_div(a.clone(), b.clone(), mx_specs=old_specs)
        new_out = vector.vec_div(a.clone(), b.clone(), scheme=_scheme(), use_recip=True)
        assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# Special math ops
# ---------------------------------------------------------------------------

class TestVecSpecial:

    def test_vec_exp(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_exp(A.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_exp(A.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_exp_with_exp2(self):
        """vec_use_exp2=True changes exp path."""
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_specs = old_finalize({"bfloat": 16, "vec_use_exp2": True})
        old_out = old_vec.vec_exp(A.clone(), mx_specs=old_specs)
        new_out = vector.vec_exp(A.clone(), scheme=_scheme(), use_exp2=True)
        assert torch.equal(old_out, new_out)

    def test_vec_exp2(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_exp2(A.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_exp2(A.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_recip(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8) + 2.0  # avoid div-by-zero
        old_out = old_vec.vec_recip(A.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_recip(A.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_sqrt(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8).abs() + 0.1  # positive only
        old_out = old_vec.vec_sqrt(A.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_sqrt(A.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_tanh(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_tanh(A.clone(), mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_tanh(A.clone(), scheme=_scheme())
        assert torch.equal(old_out, new_out)


# ---------------------------------------------------------------------------
# Reduce ops
# ---------------------------------------------------------------------------

class TestVecReduce:

    def test_vec_reduce_sum(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_sum(A.clone(), dim=-1, mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_reduce_sum(A.clone(), dim=-1, scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_reduce_sum_keepdim(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_sum(A.clone(), dim=-1, keepdim=True, mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_reduce_sum(A.clone(), dim=-1, keepdim=True, scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_reduce_mean(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_mean(A.clone(), dim=-1, mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_reduce_mean(A.clone(), dim=-1, scheme=_scheme())
        assert torch.equal(old_out, new_out)

    def test_vec_reduce_mean_keepdim(self):
        from mx.specs import finalize_mx_specs as old_finalize
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        old_out = old_vec.vec_reduce_mean(A.clone(), dim=-1, keepdim=True, mx_specs=old_finalize({"bfloat": 16}))
        new_out = vector.vec_reduce_mean(A.clone(), dim=-1, keepdim=True, scheme=_scheme())
        assert torch.equal(old_out, new_out)
