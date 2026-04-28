"""
Bit-exact equivalence tests: src/ops/conv.py vs mx/convolution.py — P3.2.

All comparisons use torch.equal (bit-exact, no atol/rtol).
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.conv import ConvFunction, QuantizedConv2d
from src.scheme.op_config import OpQuantConfig
from src.tests._compat import op_config_from_mx_specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conv_tensors(batch=2, in_c=4, out_c=8, h=8, w=8, k=3, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(batch, in_c, h, w, dtype=torch.float32)
    weight = torch.randn(out_c, in_c, k, k, dtype=torch.float32)
    bias = torch.randn(out_c, dtype=torch.float32)
    return x, weight, bias


def _run_mx_conv(x, weight, bias, mx_specs, stride=1, padding=0):
    mx_specs = apply_mx_specs(mx_specs)
    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    bias = bias.clone().requires_grad_(True) if bias is not None else None
    out = mx.conv2d(x, weight, bias, mx_specs=mx_specs, stride=stride, padding=padding)
    out.sum().backward()
    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        weight.grad.detach() if weight.grad is not None else None,
        bias.grad.detach() if bias is not None and bias.grad is not None else None,
    )


def _run_src_conv(x, weight, bias, cfg, stride=1, padding=0):
    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    bias = bias.clone().requires_grad_(True) if bias is not None else None
    out = ConvFunction.apply(x, weight, bias, stride, padding, 1, 1, cfg)
    out.sum().backward()
    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        weight.grad.detach() if weight.grad is not None else None,
        bias.grad.detach() if bias is not None and bias.grad is not None else None,
    )


def _assert_bit_exact(mx_out, src_out, label="output"):
    if mx_out is None and src_out is None:
        return
    assert mx_out is not None and src_out is not None, f"{label}: one is None"
    assert torch.equal(mx_out, src_out), (
        f"{label}: not bit-exact (max diff={torch.max(torch.abs(mx_out - src_out))})"
    )


MX_SPECS_CONV = [
    pytest.param("bfloat16", {"bfloat": 16}, id="bf16"),
    pytest.param("bf16+mxfp8e4m3", {
        "bfloat": 16, "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3", "block_size": 32,
    }, id="bf16+mxfp8e4m3"),
    pytest.param("bf16+mxfp8e5m2", {
        "bfloat": 16, "a_elem_format": "fp8_e5m2",
        "w_elem_format": "fp8_e5m2", "block_size": 32,
    }, id="bf16+mxfp8e5m2"),
    pytest.param("mxfp8e4m3-no-bf", {
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3", "block_size": 32,
    }, id="mxfp8e4m3-no-bf"),
]


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

class TestConv2dForward:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV)
    def test_forward_with_bias(self, name, mx_specs):
        x, w, b = _make_conv_tensors()
        mx_out, _, _, _ = _run_mx_conv(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")
        src_out, _, _, _ = _run_src_conv(x, w, b, cfg)
        _assert_bit_exact(mx_out, src_out, label=f"conv-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV)
    def test_forward_no_bias(self, name, mx_specs):
        x, w, b = _make_conv_tensors()
        mx_out, _, _, _ = _run_mx_conv(x, w, None, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")
        src_out, _, _, _ = _run_src_conv(x, w, None, cfg)
        _assert_bit_exact(mx_out, src_out, label=f"conv-fwd-nobias ({name})")

    def test_forward_passthrough(self):
        x, w, b = _make_conv_tensors()
        mx_out = torch.nn.functional.conv2d(x, w, b)
        cfg = OpQuantConfig()
        src_out, _, _, _ = _run_src_conv(x, w, b, cfg)
        _assert_bit_exact(mx_out, src_out, label="conv-passthrough")


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

class TestConv2dBackward:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV)
    def test_backward_grad_input(self, name, mx_specs):
        x, w, b = _make_conv_tensors()
        _, mx_gi, _, _ = _run_mx_conv(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")
        _, src_gi, _, _ = _run_src_conv(x, w, b, cfg)
        _assert_bit_exact(mx_gi, src_gi, label=f"conv-grad_input ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV)
    def test_backward_grad_weight(self, name, mx_specs):
        x, w, b = _make_conv_tensors()
        _, _, mx_gw, _ = _run_mx_conv(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")
        _, _, src_gw, _ = _run_src_conv(x, w, b, cfg)
        _assert_bit_exact(mx_gw, src_gw, label=f"conv-grad_weight ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV)
    def test_backward_grad_bias(self, name, mx_specs):
        x, w, b = _make_conv_tensors()
        _, _, _, mx_gb = _run_mx_conv(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")
        _, _, _, src_gb = _run_src_conv(x, w, b, cfg)
        _assert_bit_exact(mx_gb, src_gb, label=f"conv-grad_bias ({name})")


# ---------------------------------------------------------------------------
# QuantizedConv2d module test
# ---------------------------------------------------------------------------

class TestQuantizedConv2dModule:
    def test_module_forward_matches_functional(self):
        mx_specs = {"bfloat": 16, "a_elem_format": "fp8_e4m3",
                     "w_elem_format": "fp8_e4m3", "block_size": 32}
        x, w, b = _make_conv_tensors()
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")

        # Functional
        x_f = x.clone().requires_grad_(True)
        out_f = ConvFunction.apply(x_f, w, b, 1, 0, 1, 1, cfg)
        out_f.sum().backward()

        # Module
        qc = QuantizedConv2d(4, 8, 3, bias=True, cfg=cfg)
        qc.weight.data.copy_(w)
        qc.bias.data.copy_(b)
        x_m = x.clone().requires_grad_(True)
        out_m = qc(x_m)
        out_m.sum().backward()

        assert torch.equal(out_f.detach(), out_m.detach()), "Module forward != functional"
        assert torch.equal(x_f.grad, x_m.grad), "Module grad_input != functional"
