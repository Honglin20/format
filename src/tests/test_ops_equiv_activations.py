"""
Bit-exact equivalence tests: src/ops/activations.py vs mx/activations.py — P3.4.

All comparisons use torch.equal (bit-exact, no atol/rtol).
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.activations import (
    SigmoidFunction, TanhFunction, ReLUFunction, ReLU6Function,
    LeakyReLUFunction, SiLUFunction, GELUFunction,
    QuantizedSigmoid, QuantizedTanh, QuantizedReLU, QuantizedReLU6,
    QuantizedLeakyReLU, QuantizedSiLU, QuantizedGELU,
)
from src.tests._compat import activation_config_from_mx_specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_input(shape=(4, 8), seed=42):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.sigmoid(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = SigmoidFunction.apply(src_x, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"sigmoid-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.sigmoid(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = SigmoidFunction.apply(src_x, inner, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"sigmoid-grad ({name})")


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------

class TestTanh:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.tanh(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = TanhFunction.apply(src_x, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"tanh-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.tanh(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = TanhFunction.apply(src_x, inner, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"tanh-grad ({name})")


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

class TestReLU:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.relu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = ReLUFunction.apply(src_x, False, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"relu-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.relu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = ReLUFunction.apply(src_x, False, inner, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"relu-grad ({name})")


# ---------------------------------------------------------------------------
# ReLU6
# ---------------------------------------------------------------------------

class TestReLU6:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.relu6(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = ReLU6Function.apply(src_x, False, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"relu6-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.relu6(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = ReLU6Function.apply(src_x, False, inner, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"relu6-grad ({name})")


# ---------------------------------------------------------------------------
# LeakyReLU
# ---------------------------------------------------------------------------

class TestLeakyReLU:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.leaky_relu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = LeakyReLUFunction.apply(src_x, 0.01, False, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"leaky_relu-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.leaky_relu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = LeakyReLUFunction.apply(src_x, 0.01, False, inner, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"leaky_relu-grad ({name})")


# ---------------------------------------------------------------------------
# SiLU
# ---------------------------------------------------------------------------

class TestSiLU:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.silu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = SiLUFunction.apply(src_x, False, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"silu-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.silu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = SiLUFunction.apply(src_x, False, inner, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"silu-grad ({name})")


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------

class TestGELU:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward_detailed(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.gelu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = GELUFunction.apply(src_x, inner, False, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"gelu-fwd-detailed ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward_detailed(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.gelu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = GELUFunction.apply(src_x, inner, False, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"gelu-grad-detailed ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward_first_order(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.gelu(mx_x, mx_specs=mx_specs, first_order_gelu=True)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = GELUFunction.apply(src_x, inner, True, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"gelu-fwd-1st ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward_first_order(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.gelu(mx_x, mx_specs=mx_specs, first_order_gelu=True)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = GELUFunction.apply(src_x, inner, True, qbp)

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"gelu-grad-1st ({name})")


# ---------------------------------------------------------------------------
# STE (quantize_backprop=False) tests
# ---------------------------------------------------------------------------

class TestActivationSTE:
    """Test that quantize_backprop=False produces passthrough backward (bit-exact vs mx)."""

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_STE)
    def test_sigmoid_ste(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.sigmoid(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = SigmoidFunction.apply(src_x, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"sigmoid-ste-fwd ({name})")
        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"sigmoid-ste-grad ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_STE)
    def test_relu_ste(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.relu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = ReLUFunction.apply(src_x, False, inner, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"relu-ste-fwd ({name})")
        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"relu-ste-grad ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_STE)
    def test_gelu_ste(self, name, mx_specs):
        x = _make_input()
        mx_specs = apply_mx_specs(mx_specs)
        mx_x = x.clone().requires_grad_(True)
        src_x = x.clone().requires_grad_(True)

        mx_out = mx.gelu(mx_x, mx_specs=mx_specs)
        cfg = activation_config_from_mx_specs(mx_specs)
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = GELUFunction.apply(src_x, inner, False, qbp)

        _assert_bit_exact(mx_out.detach(), src_out.detach(), label=f"gelu-ste-fwd ({name})")
        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label=f"gelu-ste-grad ({name})")
