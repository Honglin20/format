"""
Bit-exact equivalence tests: src/ops/elemwise.py vs mx/simd_ops.py — P3.5.

All comparisons use torch.equal (bit-exact, no atol/rtol).
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.elemwise import (
    simd_add, simd_sub, simd_mul, simd_div,
    simd_split, simd_square, simd_sqrt,
    simd_exp, simd_log,
    simd_reduce_sum, simd_reduce_mean, simd_norm,
)
from src.tests._compat import simd_config_from_mx_specs


def _assert_bit_exact(mx_out, src_out, label="output"):
    if mx_out is None and src_out is None:
        return
    assert mx_out is not None and src_out is not None, f"{label}: one is None"
    if torch.equal(mx_out, src_out):
        return
    # Handle NaN — NaN != NaN by IEEE 754, but identical NaN positions are bit-exact
    mx_nan = torch.isnan(mx_out)
    src_nan = torch.isnan(src_out)
    assert torch.equal(mx_nan, src_nan), (
        f"{label}: NaN mismatch — positions differ"
    )
    # Compare non-NaN elements
    mx_valid = mx_out[~mx_nan]
    src_valid = src_out[~src_nan]
    assert torch.equal(mx_valid, src_valid), (
        f"{label}: valid elements not bit-exact "
        f"(max diff={torch.max(torch.abs(mx_valid - src_valid))})"
    )


MX_SPECS = [
    pytest.param("bfloat16", {"bfloat": 16}, id="bf16"),
    pytest.param("bfloat10", {"bfloat": 10}, id="bf10"),
]

MX_SPECS_STE = [
    pytest.param("bf16-ste", {"bfloat": 16, "quantize_backprop": False}, id="bf16-ste"),
]


def _make_input(shape=(4, 8), seed=42):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32)


def _make_pair(shape=(4, 8), seed1=42, seed2=99):
    torch.manual_seed(seed1)
    a = torch.randn(*shape, dtype=torch.float32)
    torch.manual_seed(seed2)
    b = torch.randn(*shape, dtype=torch.float32)
    return a, b


# ---------------------------------------------------------------------------
# SIMD Add
# ---------------------------------------------------------------------------

class TestSIMDAdd:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        a, b = _make_pair()
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_add(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_add(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"simd_add-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        a, b = _make_pair()
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_add(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_add(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"simd_add-grad-a ({name})")
        _assert_bit_exact(mx_b.grad, src_b.grad, label=f"simd_add-grad-b ({name})")

    def test_forward_const(self):
        """simd_add with constant in2 (not a Tensor)."""
        mx_specs = apply_mx_specs({"bfloat": 16})
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        a = _make_input()

        mx_out = mx.simd_add(a, 2.0, mx_specs=mx_specs)
        src_out = simd_add(a.clone(), 2.0, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label="simd_add-const")


# ---------------------------------------------------------------------------
# SIMD Sub
# ---------------------------------------------------------------------------

class TestSIMDSub:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        a, b = _make_pair(seed1=42, seed2=99)
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_sub(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_sub(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"simd_sub-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        a, b = _make_pair(seed1=42, seed2=99)
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_sub(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_sub(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"simd_sub-grad-a ({name})")
        _assert_bit_exact(mx_b.grad, src_b.grad, label=f"simd_sub-grad-b ({name})")


# ---------------------------------------------------------------------------
# SIMD Mul
# ---------------------------------------------------------------------------

class TestSIMDMul:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        a, b = _make_pair(seed1=42, seed2=99)
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_mul(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_mul(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"simd_mul-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        a, b = _make_pair(seed1=42, seed2=99)
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_mul(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_mul(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"simd_mul-grad-a ({name})")
        _assert_bit_exact(mx_b.grad, src_b.grad, label=f"simd_mul-grad-b ({name})")


# ---------------------------------------------------------------------------
# SIMD Div
# ---------------------------------------------------------------------------

class TestSIMDDiv:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        a, b = _make_pair(seed1=42, seed2=99)
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_div(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_div(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"simd_div-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        a, b = _make_pair(seed1=42, seed2=99)
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_div(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_div(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"simd_div-grad-a ({name})")
        _assert_bit_exact(mx_b.grad, src_b.grad, label=f"simd_div-grad-b ({name})")


# ---------------------------------------------------------------------------
# SIMD Square, Sqrt, Exp, Log
# ---------------------------------------------------------------------------

class TestSIMDUnary:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    @pytest.mark.parametrize("op_name", ["square", "sqrt", "exp", "log"])
    def test_forward(self, name, mx_specs, op_name):
        a = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_fn = getattr(mx, f"simd_{op_name}")
        src_fn = globals()[f"simd_{op_name}"]
        inner, qbp = simd_config_from_mx_specs(mx_specs)

        mx_out = mx_fn(mx_a, mx_specs=mx_specs)
        src_out = src_fn(src_a, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"simd_{op_name}-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    @pytest.mark.parametrize("op_name", ["square", "sqrt", "exp", "log"])
    def test_backward(self, name, mx_specs, op_name):
        a = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_fn = getattr(mx, f"simd_{op_name}")
        src_fn = globals()[f"simd_{op_name}"]
        inner, qbp = simd_config_from_mx_specs(mx_specs)

        mx_out = mx_fn(mx_a, mx_specs=mx_specs)
        src_out = src_fn(src_a, inner_scheme=inner, quantize_backprop=qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"simd_{op_name}-grad ({name})")


# ---------------------------------------------------------------------------
# SIMD Split
# ---------------------------------------------------------------------------

class TestSIMDSplit:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        a = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_o1, mx_o2 = mx.simd_split(mx_a, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_o1, src_o2 = simd_split(src_a, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_o1.detach(), src_o1.detach(), label=f"simd_split-o1 ({name})")
        _assert_bit_exact(mx_o2.detach(), src_o2.detach(), label=f"simd_split-o2 ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        a = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_o1, mx_o2 = mx.simd_split(mx_a, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_o1, src_o2 = simd_split(src_a, inner_scheme=inner, quantize_backprop=qbp)

        (mx_o1.sum() + mx_o2.sum()).backward()
        (src_o1.sum() + src_o2.sum()).backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"simd_split-grad ({name})")


# ---------------------------------------------------------------------------
# SIMD ReduceSum / ReduceMean / Norm
# ---------------------------------------------------------------------------

class TestSIMDReduce:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_reduce_sum_forward(self, name, mx_specs):
        a = _make_input(shape=(4, 8))
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_out = mx.simd_reduce_sum(mx_a, dim=-1, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_reduce_sum(src_a, dim=-1, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"reduce_sum-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_reduce_sum_backward(self, name, mx_specs):
        a = _make_input(shape=(4, 8))
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_out = mx.simd_reduce_sum(mx_a, dim=-1, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_reduce_sum(src_a, dim=-1, inner_scheme=inner, quantize_backprop=qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"reduce_sum-grad ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_reduce_mean_forward(self, name, mx_specs):
        a = _make_input(shape=(4, 8))
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_out = mx.simd_reduce_mean(mx_a, dim=-1, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_reduce_mean(src_a, dim=-1, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"reduce_mean-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_norm_forward(self, name, mx_specs):
        a = _make_input(shape=(4, 8))
        mx_specs = apply_mx_specs(mx_specs)
        mx_a = a.clone().requires_grad_(True)
        src_a = a.clone().requires_grad_(True)

        mx_out = mx.simd_norm(mx_a, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_norm(src_a, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"simd_norm-fwd ({name})")


# ---------------------------------------------------------------------------
# STE (quantize_backprop=False) regression
# ---------------------------------------------------------------------------

class TestSIMDSTE:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_STE)
    def test_mul_ste(self, name, mx_specs):
        a, b = _make_pair(seed1=42, seed2=99)
        mx_specs = apply_mx_specs(mx_specs)
        mx_a, mx_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        src_a, src_b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)

        mx_out = mx.simd_mul(mx_a, mx_b, mx_specs=mx_specs)
        inner, qbp = simd_config_from_mx_specs(mx_specs)
        src_out = simd_mul(src_a, src_b, inner_scheme=inner, quantize_backprop=qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"simd_mul-ste-fwd ({name})")
        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_a.grad, src_a.grad, label=f"simd_mul-ste-grad-a ({name})")
