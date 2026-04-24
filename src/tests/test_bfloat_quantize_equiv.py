"""
Equivalence tests for src/quantize/bfloat_quantize.py vs mx/quantize.py.

TDD: Every function tested against old code — bit-identical output required.
"""
import pytest
import torch

from mx.quantize import quantize_bfloat as old_quantize_bfloat
from src.quantize.bfloat_quantize import quantize_bfloat
from mx.specs import finalize_mx_specs as old_finalize
from src.specs.specs import finalize_mx_specs as new_finalize


class TestQuantizeBfloatForward:
    """Forward pass: new quantize_bfloat vs old quantize_bfloat."""

    @pytest.mark.parametrize("bfloat_bits", [16, 12])
    def test_bfloat_forward(self, bfloat_bits):
        torch.manual_seed(42)
        x = torch.randn(4, 32)
        old_specs = old_finalize({"bfloat": bfloat_bits})
        new_specs = new_finalize({"bfloat": bfloat_bits})
        old_out = old_quantize_bfloat(x.clone(), mx_specs=old_specs)
        new_out = quantize_bfloat(x.clone(), mx_specs=new_specs)
        assert torch.equal(old_out, new_out), f"bfloat{bfloat_bits} forward mismatch"

    def test_bfloat_forward_none_specs(self):
        """When mx_specs is None, input should pass through unchanged."""
        x = torch.randn(4, 32)
        old_out = old_quantize_bfloat(x.clone(), mx_specs=None)
        new_out = quantize_bfloat(x.clone(), mx_specs=None)
        assert torch.equal(old_out, new_out)

    def test_bfloat_forward_explicit_round(self):
        """Pass explicit round_mode override."""
        torch.manual_seed(42)
        x = torch.randn(4, 32)
        old_specs = old_finalize({"bfloat": 16, "round": "nearest"})
        new_specs = new_finalize({"bfloat": 16, "round": "nearest"})
        old_out = old_quantize_bfloat(x.clone(), mx_specs=old_specs, round="floor")
        new_out = quantize_bfloat(x.clone(), mx_specs=new_specs, round_mode="floor")
        assert torch.equal(old_out, new_out)


class TestQuantizeBfloatBackward:
    """Backward pass: gradient equivalence with quantize_backprop."""

    @pytest.mark.parametrize("quantize_bp", [True, False])
    def test_bfloat_backward(self, quantize_bp):
        torch.manual_seed(42)
        x1 = torch.randn(4, 32, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        old_specs = old_finalize({"bfloat": 16, "quantize_backprop": quantize_bp})
        old_out = old_quantize_bfloat(x1, mx_specs=old_specs)
        old_out.sum().backward()

        new_specs = new_finalize({"bfloat": 16, "quantize_backprop": quantize_bp})
        new_out = quantize_bfloat(x2, mx_specs=new_specs)
        new_out.sum().backward()

        assert torch.equal(x1.grad, x2.grad), \
            f"backward mismatch with quantize_backprop={quantize_bp}"

    def test_bfloat_backward_no_bp_zeros_grad(self):
        """When quantize_backprop=False, gradient should be identity (no quantization)."""
        torch.manual_seed(42)
        x1 = torch.randn(4, 32, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        old_specs = old_finalize({"bfloat": 16, "quantize_backprop": False})
        old_out = old_quantize_bfloat(x1, mx_specs=old_specs)
        old_out.sum().backward()

        new_specs = new_finalize({"bfloat": 16, "quantize_backprop": False})
        new_out = quantize_bfloat(x2, mx_specs=new_specs)
        new_out.sum().backward()

        # With quantize_backprop=False, gradient is unquantized (identity)
        assert torch.equal(x1.grad, x2.grad)
