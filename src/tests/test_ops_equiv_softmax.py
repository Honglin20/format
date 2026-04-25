"""
Bit-exact equivalence tests: src/ops/softmax.py vs mx/softmax.py — P3.4.

All comparisons use torch.equal (bit-exact, no atol/rtol).
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.softmax import SoftmaxFunction, QuantizedSoftmax
from src.tests._compat import softmax_config_from_mx_specs


def _assert_bit_exact(mx_out, src_out, label="output"):
    if mx_out is None and src_out is None:
        return
    assert mx_out is not None and src_out is not None, f"{label}: one is None"
    assert torch.equal(mx_out, src_out), (
        f"{label}: not bit-exact (max diff={torch.max(torch.abs(mx_out - src_out))})"
    )


MX_SPECS = [
    pytest.param("bfloat16", {"bfloat": 16}, id="bf16"),
    pytest.param("bfloat10", {"bfloat": 10}, id="bf10"),
]

MX_SPECS_EXP2 = [
    pytest.param("bfloat16-exp2", {"bfloat": 16, "softmax_exp2": True}, id="bf16-exp2"),
]

MX_SPECS_STE = [
    pytest.param("bf16-ste", {"bfloat": 16, "quantize_backprop": False}, id="bf16-ste"),
]


def _make_input(shape=(4, 8), seed=42):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32)


class TestSoftmax:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.softmax(mx_x, dim=-1, mx_specs=mx_specs)
        cfg, exp2 = softmax_config_from_mx_specs(mx_specs)
        inner = cfg.input[0] if cfg.input else None
        qbp = bool(cfg.grad_input)
        src_out = SoftmaxFunction.apply(src_x, -1, inner, exp2, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"softmax-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.softmax(mx_x, dim=-1, mx_specs=mx_specs)
        cfg, exp2 = softmax_config_from_mx_specs(mx_specs)
        inner = cfg.input[0] if cfg.input else None
        qbp = bool(cfg.grad_input)
        src_out = SoftmaxFunction.apply(src_x, -1, inner, exp2, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"softmax-grad ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward_dim0(self, name, mx_specs):
        x = _make_input(shape=(4, 8))
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.softmax(mx_x, dim=0, mx_specs=mx_specs)
        cfg, exp2 = softmax_config_from_mx_specs(mx_specs)
        inner = cfg.input[0] if cfg.input else None
        qbp = bool(cfg.grad_input)
        src_out = SoftmaxFunction.apply(src_x, 0, inner, exp2, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"softmax-fwd-dim0 ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_EXP2)
    def test_forward_exp2(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.softmax(mx_x, dim=-1, mx_specs=mx_specs)
        cfg, exp2 = softmax_config_from_mx_specs(mx_specs)
        inner = cfg.input[0] if cfg.input else None
        qbp = bool(cfg.grad_input)
        src_out = SoftmaxFunction.apply(src_x, -1, inner, exp2, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"softmax-fwd-exp2 ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_EXP2)
    def test_backward_exp2(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.softmax(mx_x, dim=-1, mx_specs=mx_specs)
        cfg, exp2 = softmax_config_from_mx_specs(mx_specs)
        inner = cfg.input[0] if cfg.input else None
        qbp = bool(cfg.grad_input)
        src_out = SoftmaxFunction.apply(src_x, -1, inner, exp2, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"softmax-grad-exp2 ({name})")


class TestSoftmaxSTE:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_STE)
    def test_ste_forward_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.softmax(mx_x, dim=-1, mx_specs=mx_specs)
        cfg, exp2 = softmax_config_from_mx_specs(mx_specs)
        inner = cfg.input[0] if cfg.input else None
        qbp = bool(cfg.grad_input)
        src_out = SoftmaxFunction.apply(src_x, -1, inner, exp2, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"softmax-ste-fwd ({name})")
        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"softmax-ste-grad ({name})")
