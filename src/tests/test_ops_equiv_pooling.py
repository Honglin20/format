"""
Bit-exact equivalence tests: src/ops/pooling.py vs mx/adaptive_avg_pooling.py — P3.4.

All comparisons use torch.equal (bit-exact, no atol/rtol).
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.pooling import AdaptiveAvgPool2dFunction, QuantizedAdaptiveAvgPool2d
from src.tests._compat import pool_config_from_mx_specs


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

MX_SPECS_STE = [
    pytest.param("bf16-ste", {"bfloat": 16, "quantize_backprop": False}, id="bf16-ste"),
]


def _make_input(batch=2, channels=3, h=8, w=8, seed=42):
    torch.manual_seed(seed)
    return torch.randn(batch, channels, h, w, dtype=torch.float32)


class TestAdaptiveAvgPool2d:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        output_size = (4, 4)

        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.adaptive_avg_pool2d(mx_x, output_size, mx_specs=mx_specs)
        cfg = pool_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = AdaptiveAvgPool2dFunction.apply(src_x, output_size, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"pool-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        output_size = (4, 4)

        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.adaptive_avg_pool2d(mx_x, output_size, mx_specs=mx_specs)
        cfg = pool_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = AdaptiveAvgPool2dFunction.apply(src_x, output_size, inner, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"pool-grad ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward_int_output_size(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        output_size = 3

        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.adaptive_avg_pool2d(mx_x, output_size, mx_specs=mx_specs)
        cfg = pool_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = AdaptiveAvgPool2dFunction.apply(src_x, output_size, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"pool-fwd-int ({name})")


class TestAdaptiveAvgPool2dSTE:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_STE)
    def test_ste_forward_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        output_size = (4, 4)

        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.adaptive_avg_pool2d(mx_x, output_size, mx_specs=mx_specs)
        cfg = pool_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = AdaptiveAvgPool2dFunction.apply(src_x, output_size, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"pool-ste-fwd ({name})")
        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"pool-ste-grad ({name})")
