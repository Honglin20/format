"""
Bit-exact equivalence tests: src/ops/conv.py ConvTranspose vs
mx/transpose_convolution.py — P3.2.

All comparisons use torch.equal (bit-exact, no atol/rtol).
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.conv import ConvTransposeFunction, QuantizedConvTranspose2d
from src.scheme.op_config import OpQuantConfig
from src.tests._compat import op_config_from_mx_specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conv_transpose_tensors(batch=2, in_c=4, out_c=8, h=6, w=6, k=3,
                                 seed=42):
    torch.manual_seed(seed)
    # ConvTranspose2d weight shape: (in_channels, out_channels/groups, kH, kW)
    x = torch.randn(batch, in_c, h, w, dtype=torch.float32)
    weight = torch.randn(in_c, out_c, k, k, dtype=torch.float32)
    bias = torch.randn(out_c, dtype=torch.float32)
    return x, weight, bias


def _run_mx_conv_transpose(x, weight, bias, mx_specs, stride=1, padding=0,
                           output_padding=0):
    mx_specs = apply_mx_specs(mx_specs)
    # Use mx.ConvTranspose2d module
    mx_mod = mx.ConvTranspose2d(
        x.shape[1], weight.shape[1], weight.shape[2],
        stride=stride, padding=padding, bias=bias is not None,
        mx_specs=mx_specs,
    )
    mx_mod.weight.data.copy_(weight)
    if bias is not None:
        mx_mod.bias.data.copy_(bias)
    mx_mod = mx_mod.to(x.dtype)

    x = x.clone().requires_grad_(True)
    mx_mod.weight.requires_grad_(True)
    if bias is not None:
        mx_mod.bias.requires_grad_(True)

    out = mx_mod(x)
    out.sum().backward()

    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        mx_mod.weight.grad.detach() if mx_mod.weight.grad is not None else None,
        mx_mod.bias.grad.detach() if bias is not None and mx_mod.bias.grad is not None else None,
    )


def _run_src_conv_transpose(x, weight, bias, cfg, stride=1, padding=0,
                            output_padding=0):
    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    bias = bias.clone().requires_grad_(True) if bias is not None else None

    out = ConvTransposeFunction.apply(
        x, weight, bias, stride, padding, output_padding, 1, 1, cfg,
    )
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


MX_SPECS_CONV_TRANSPOSE = [
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

class TestConvTranspose2dForward:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV_TRANSPOSE)
    def test_forward_with_bias(self, name, mx_specs):
        x, w, b = _make_conv_transpose_tensors()
        mx_out, _, _, _ = _run_mx_conv_transpose(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv_transpose")
        src_out, _, _, _ = _run_src_conv_transpose(x, w, b, cfg)
        _assert_bit_exact(mx_out, src_out, label=f"convT-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV_TRANSPOSE)
    def test_forward_no_bias(self, name, mx_specs):
        x, w, b = _make_conv_transpose_tensors()
        mx_out, _, _, _ = _run_mx_conv_transpose(x, w, None, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv_transpose")
        src_out, _, _, _ = _run_src_conv_transpose(x, w, None, cfg)
        _assert_bit_exact(mx_out, src_out, label=f"convT-fwd-nobias ({name})")

    def test_forward_passthrough(self):
        x, w, b = _make_conv_transpose_tensors()
        mx_out = torch.nn.functional.conv_transpose2d(x, w, b)
        cfg = OpQuantConfig()
        src_out, _, _, _ = _run_src_conv_transpose(x, w, b, cfg)
        _assert_bit_exact(mx_out, src_out, label="convT-passthrough")


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

class TestConvTranspose2dBackward:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV_TRANSPOSE)
    def test_backward_grad_input(self, name, mx_specs):
        x, w, b = _make_conv_transpose_tensors()
        _, mx_gi, _, _ = _run_mx_conv_transpose(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv_transpose")
        _, src_gi, _, _ = _run_src_conv_transpose(x, w, b, cfg)
        _assert_bit_exact(mx_gi, src_gi, label=f"convT-grad_input ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV_TRANSPOSE)
    def test_backward_grad_weight(self, name, mx_specs):
        x, w, b = _make_conv_transpose_tensors()
        _, _, mx_gw, _ = _run_mx_conv_transpose(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv_transpose")
        _, _, src_gw, _ = _run_src_conv_transpose(x, w, b, cfg)
        _assert_bit_exact(mx_gw, src_gw, label=f"convT-grad_weight ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONV_TRANSPOSE)
    def test_backward_grad_bias(self, name, mx_specs):
        x, w, b = _make_conv_transpose_tensors()
        _, _, _, mx_gb = _run_mx_conv_transpose(x, w, b, mx_specs)
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv_transpose")
        _, _, _, src_gb = _run_src_conv_transpose(x, w, b, cfg)
        _assert_bit_exact(mx_gb, src_gb, label=f"convT-grad_bias ({name})")


# ---------------------------------------------------------------------------
# QuantizedConvTranspose2d module test
# ---------------------------------------------------------------------------

class TestQuantizedConvTranspose2dModule:
    def test_module_forward_matches_functional(self):
        mx_specs = {"bfloat": 16, "a_elem_format": "fp8_e4m3",
                     "w_elem_format": "fp8_e4m3", "block_size": 32}
        x, w, b = _make_conv_transpose_tensors()
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv_transpose")

        # Functional
        x_f = x.clone().requires_grad_(True)
        out_f = ConvTransposeFunction.apply(x_f, w, b, 1, 0, 0, 1, 1, cfg)
        out_f.sum().backward()

        # Module
        qc = QuantizedConvTranspose2d(4, 8, 3, bias=True, cfg=cfg)
        qc.weight.data.copy_(w)
        qc.bias.data.copy_(b)
        x_m = x.clone().requires_grad_(True)
        out_m = qc(x_m)
        out_m.sum().backward()

        assert torch.equal(out_f.detach(), out_m.detach()), "Module forward != functional"
        assert torch.equal(x_f.grad, x_m.grad), "Module grad_input != functional"
