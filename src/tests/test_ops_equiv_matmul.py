"""
Bit-exact equivalence tests: src/ops/linear.py vs mx/linear.py — P3.1-d.

All comparisons use torch.equal (bit-exact, no atol/rtol).
Dither mode tests fix seed for determinism.
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.linear import QuantizedLinear, LinearFunction
from src.scheme.op_config import OpQuantConfig
from src.tests._compat import op_config_from_mx_specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensors(batch=2, seq=4, in_f=16, out_f=8, seed=42):
    """Create deterministic input/weight/bias tensors."""
    torch.manual_seed(seed)
    x = torch.randn(batch, seq, in_f, dtype=torch.float32)
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    return x, w, b


def _run_mx_linear(x, w, b, mx_specs, with_bias=True):
    """Run mx.Linear forward + backward, return (output, grad_input, grad_weight, grad_bias)."""
    mx_specs = apply_mx_specs(mx_specs)
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True) if with_bias else None

    out = mx.linear(x, w, b, mx_specs=mx_specs)
    out.sum().backward()

    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        w.grad.detach() if w.grad is not None else None,
        b.grad.detach() if b is not None and b.grad is not None else None,
    )


def _run_src_linear(x, w, b, cfg, with_bias=True):
    """Run src LinearFunction forward + backward, return (output, grad_input, grad_weight, grad_bias)."""
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True) if with_bias else None

    out = LinearFunction.apply(x, w, b, cfg)
    out.sum().backward()

    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        w.grad.detach() if w.grad is not None else None,
        b.grad.detach() if b is not None and b.grad is not None else None,
    )


def _assert_bit_exact(mx_out, src_out, label="output"):
    if mx_out is None and src_out is None:
        return
    assert mx_out is not None and src_out is not None, f"{label}: one is None"
    assert torch.equal(mx_out, src_out), f"{label}: not bit-exact (max diff={torch.max(torch.abs(mx_out - src_out))})"


# ---------------------------------------------------------------------------
# Test configurations: (name, mx_specs dict)
# ---------------------------------------------------------------------------

MX_SPECS_CONFIGS = [
    # Bfloat16 only
    pytest.param("bfloat16", {"bfloat": 16}, id="bfloat16"),
    # Bfloat16 + MX FP8E4M3
    pytest.param("bf16+mxfp8e4m3", {
        "bfloat": 16, "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3", "block_size": 32,
    }, id="bf16+mxfp8e4m3"),
    # Bfloat16 + MX FP8E5M2
    pytest.param("bf16+mxfp8e5m2", {
        "bfloat": 16, "a_elem_format": "fp8_e5m2",
        "w_elem_format": "fp8_e5m2", "block_size": 32,
    }, id="bf16+mxfp8e5m2"),
    # Bfloat16 + MX INT8
    pytest.param("bf16+mxint8", {
        "bfloat": 16, "a_elem_format": "int8",
        "w_elem_format": "int8", "block_size": 32,
    }, id="bf16+mxint8"),
    # MX FP8 only (no bfloat)
    pytest.param("mxfp8e4m3", {
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3", "block_size": 32,
    }, id="mxfp8e4m3-no-bf"),
    # Bfloat16 + MX FP4
    pytest.param("bf16+mxfp4", {
        "bfloat": 16, "a_elem_format": "fp4_e2m1",
        "w_elem_format": "fp4_e2m1", "block_size": 32,
    }, id="bf16+mxfp4"),
]


# ---------------------------------------------------------------------------
# Forward bit-exact tests
# ---------------------------------------------------------------------------

class TestLinearForward:
    """Forward bit-exact: src LinearFunction vs mx.linear."""

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONFIGS)
    def test_forward_with_bias(self, name, mx_specs):
        x, w, b = _make_tensors()
        mx_out, _, _, _ = _run_mx_linear(x, w, b, mx_specs, with_bias=True)
        cfg = op_config_from_mx_specs(mx_specs)
        src_out, _, _, _ = _run_src_linear(x, w, b, cfg, with_bias=True)
        _assert_bit_exact(mx_out, src_out, label=f"forward ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONFIGS)
    def test_forward_no_bias(self, name, mx_specs):
        x, w, b = _make_tensors()
        mx_out, _, _, _ = _run_mx_linear(x, w, None, mx_specs, with_bias=False)
        cfg = op_config_from_mx_specs(mx_specs)
        src_out, _, _, _ = _run_src_linear(x, w, None, cfg, with_bias=False)
        _assert_bit_exact(mx_out, src_out, label=f"forward-no-bias ({name})")

    def test_forward_passthrough_no_quant(self):
        """Empty OpQuantConfig → passthrough (no quantization)."""
        x, w, b = _make_tensors()
        mx_out = torch.nn.functional.linear(x, w, b)
        cfg = OpQuantConfig()
        src_out, _, _, _ = _run_src_linear(x, w, b, cfg, with_bias=True)
        _assert_bit_exact(mx_out, src_out, label="passthrough")


# ---------------------------------------------------------------------------
# Backward bit-exact tests
# ---------------------------------------------------------------------------

class TestLinearBackward:
    """Backward bit-exact: src LinearFunction vs mx.linear."""

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONFIGS)
    def test_backward_grad_input(self, name, mx_specs):
        x, w, b = _make_tensors()
        _, mx_gi, _, _ = _run_mx_linear(x, w, b, mx_specs, with_bias=True)
        cfg = op_config_from_mx_specs(mx_specs)
        _, src_gi, _, _ = _run_src_linear(x, w, b, cfg, with_bias=True)
        _assert_bit_exact(mx_gi, src_gi, label=f"grad_input ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONFIGS)
    def test_backward_grad_weight(self, name, mx_specs):
        x, w, b = _make_tensors()
        _, _, mx_gw, _ = _run_mx_linear(x, w, b, mx_specs, with_bias=True)
        cfg = op_config_from_mx_specs(mx_specs)
        _, _, src_gw, _ = _run_src_linear(x, w, b, cfg, with_bias=True)
        _assert_bit_exact(mx_gw, src_gw, label=f"grad_weight ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONFIGS)
    def test_backward_grad_bias(self, name, mx_specs):
        x, w, b = _make_tensors()
        _, _, _, mx_gb = _run_mx_linear(x, w, b, mx_specs, with_bias=True)
        cfg = op_config_from_mx_specs(mx_specs)
        _, _, _, src_gb = _run_src_linear(x, w, b, cfg, with_bias=True)
        _assert_bit_exact(mx_gb, src_gb, label=f"grad_bias ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONFIGS)
    def test_backward_no_bias_grad(self, name, mx_specs):
        """No bias → grad_bias should be None."""
        x, w, b = _make_tensors()
        _, _, _, mx_gb = _run_mx_linear(x, w, None, mx_specs, with_bias=False)
        cfg = op_config_from_mx_specs(mx_specs)
        _, _, _, src_gb = _run_src_linear(x, w, None, cfg, with_bias=False)
        assert mx_gb is None and src_gb is None


# ---------------------------------------------------------------------------
# QuantizedLinear module test
# ---------------------------------------------------------------------------

class TestQuantizedLinearModule:
    """QuantizedLinear as nn.Module replacement."""

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_CONFIGS[:3])  # subset for speed
    def test_module_forward_matches_functional(self, name, mx_specs):
        x, w, b = _make_tensors()
        cfg = op_config_from_mx_specs(mx_specs)

        # Functional path
        x_f = x.clone().requires_grad_(True)
        w_f = w.clone().requires_grad_(True)
        b_f = b.clone().requires_grad_(True)
        out_f = LinearFunction.apply(x_f, w_f, b_f, cfg)
        out_f.sum().backward()

        # Module path
        ql = QuantizedLinear(w.shape[1], w.shape[0], bias=True, cfg=cfg)
        ql.weight.data.copy_(w)
        ql.bias.data.copy_(b)
        x_m = x.clone().requires_grad_(True)
        out_m = ql(x_m)
        out_m.sum().backward()

        assert torch.equal(out_f.detach(), out_m.detach()), "Module forward != functional"
        assert torch.equal(x_f.grad, x_m.grad), "Module grad_input != functional"


# ---------------------------------------------------------------------------
# quantize_backprop=False test
# ---------------------------------------------------------------------------

class TestLinearNoBackprop:
    """quantize_backprop=False → STE backward (no quantization in backward)."""

    def test_no_backprop_forward_same(self):
        mx_specs = {"bfloat": 16, "a_elem_format": "fp8_e4m3",
                    "w_elem_format": "fp8_e4m3", "block_size": 32,
                    "quantize_backprop": False}
        x, w, b = _make_tensors()
        mx_out, _, _, _ = _run_mx_linear(x, w, b, mx_specs, with_bias=True)
        cfg = op_config_from_mx_specs(mx_specs)
        src_out, _, _, _ = _run_src_linear(x, w, b, cfg, with_bias=True)
        _assert_bit_exact(mx_out, src_out, label="forward no-backprop")

    def test_no_backprop_is_training_false(self):
        mx_specs = {"bfloat": 16, "quantize_backprop": False}
        cfg = op_config_from_mx_specs(mx_specs)
        assert cfg.is_training is False
