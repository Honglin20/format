"""
Tests for QuantScheme-driven APIs added in P2F-5.

Verify that new APIs (quantize_mx, quantize_bfloat, vec_* with scheme)
produce bit-identical output to their compat wrappers.
"""
import pytest
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.transform import IdentityTransform


# ---------------------------------------------------------------------------
# quantize_mx(A, scheme)
# ---------------------------------------------------------------------------

class TestQuantizeMxScheme:

    @pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp4_e2m1", "int8"])
    def test_quantize_mx_matches_compat(self, fmt):
        """quantize_mx(scheme) should match quantize_mx_op(mx_specs)."""
        from src.quantize.mx_quantize import quantize_mx, quantize_mx_op
        from src.specs.specs import finalize_mx_specs

        torch.manual_seed(42)
        A = torch.randn(4, 64)
        config = {"w_elem_format": fmt, "a_elem_format": fmt,
                  "block_size": 32, "bfloat": 16}
        mx_specs = finalize_mx_specs(config.copy())

        scheme = QuantScheme.mxfp(fmt, block_size=32)

        compat_out = quantize_mx_op(A.clone(), mx_specs=mx_specs,
                                    elem_format=fmt, axes=[-1])
        scheme_out = quantize_mx(A.clone(), scheme=scheme, axes=[-1])
        assert torch.equal(compat_out, scheme_out), f"mismatch for {fmt}"

    def test_quantize_mx_none_scheme(self):
        """scheme=None should pass through unchanged."""
        from src.quantize.mx_quantize import quantize_mx
        A = torch.randn(4, 64)
        out = quantize_mx(A.clone(), scheme=None)
        assert torch.equal(A, out)

    def test_quantize_mx_per_channel_raises(self):
        """quantize_mx should reject PER_CHANNEL granularity."""
        from src.quantize.mx_quantize import quantize_mx
        A = torch.randn(4, 64)
        scheme = QuantScheme.per_channel("fp8_e4m3", axis=0)
        with pytest.raises(ValueError, match="PER_CHANNEL"):
            quantize_mx(A, scheme=scheme)

    def test_quantize_mx_no_block(self):
        """quantize_mx with PER_TENSOR (block_size=0) should work."""
        from src.quantize.mx_quantize import quantize_mx, quantize_mx_op
        from src.specs.specs import finalize_mx_specs

        torch.manual_seed(42)
        A = torch.randn(4, 64)
        config = {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3",
                  "block_size": 0, "bfloat": 16}
        mx_specs = finalize_mx_specs(config.copy())

        scheme = QuantScheme(format="fp8_e4m3",
                             granularity=GranularitySpec.per_tensor())
        compat_out = quantize_mx_op(A.clone(), mx_specs=mx_specs,
                                    elem_format="fp8_e4m3", axes=[-1])
        scheme_out = quantize_mx(A.clone(), scheme=scheme, axes=[-1])
        assert torch.equal(compat_out, scheme_out)


# ---------------------------------------------------------------------------
# quantize_bfloat(x, scheme)
# ---------------------------------------------------------------------------

class TestQuantizeBfloatScheme:

    def test_quantize_bfloat_matches_compat(self):
        """quantize_bfloat(scheme) should match quantize_bfloat_from_specs(mx_specs)."""
        from src.quantize.bfloat_quantize import quantize_bfloat, quantize_bfloat_from_specs
        from src.specs.specs import finalize_mx_specs

        torch.manual_seed(42)
        x = torch.randn(4, 32)
        mx_specs = finalize_mx_specs({"bfloat": 16})

        scheme = QuantScheme.per_tensor("bfloat16")
        compat_out = quantize_bfloat_from_specs(x.clone(), mx_specs=mx_specs)
        scheme_out = quantize_bfloat(x.clone(), scheme=scheme)
        assert torch.equal(compat_out, scheme_out)

    def test_quantize_bfloat_none_scheme(self):
        """scheme=None should pass through unchanged."""
        from src.quantize.bfloat_quantize import quantize_bfloat
        x = torch.randn(4, 32)
        out = quantize_bfloat(x.clone(), scheme=None)
        assert torch.equal(x, out)

    def test_quantize_bfloat_backward_matches_compat(self):
        """quantize_bfloat(scheme) backward should match compat wrapper backward."""
        from src.quantize.bfloat_quantize import quantize_bfloat, quantize_bfloat_from_specs
        from src.specs.specs import finalize_mx_specs

        torch.manual_seed(42)
        x1 = torch.randn(4, 32, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        scheme = QuantScheme.per_tensor("bfloat16")
        mx_specs = finalize_mx_specs({"bfloat": 16})

        out1 = quantize_bfloat(x1, scheme=scheme)
        out1.sum().backward()

        out2 = quantize_bfloat_from_specs(x2, mx_specs=mx_specs)
        out2.sum().backward()

        assert torch.equal(x1.grad, x2.grad), "backward mismatch vs compat"

    def test_quantize_bfloat_backward_no_bp_matches_compat(self):
        """quantize_bfloat with backwards_scheme=None should match compat with quantize_backprop=False."""
        from src.quantize.bfloat_quantize import quantize_bfloat, quantize_bfloat_from_specs
        from src.specs.specs import finalize_mx_specs

        torch.manual_seed(42)
        x1 = torch.randn(4, 32, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        scheme = QuantScheme.per_tensor("bfloat16")
        mx_specs = finalize_mx_specs({"bfloat": 16, "quantize_backprop": False})

        out1 = quantize_bfloat(x1, scheme=scheme, backwards_scheme=None)
        out1.sum().backward()

        out2 = quantize_bfloat_from_specs(x2, mx_specs=mx_specs)
        out2.sum().backward()

        assert torch.equal(x1.grad, x2.grad), "backward mismatch vs compat (no bp)"

    def test_quantize_bfloat_backward_no_bp(self):
        """Gradient should be identity when backwards_scheme=None."""
        from src.quantize.bfloat_quantize import quantize_bfloat

        torch.manual_seed(42)
        x = torch.randn(4, 32, requires_grad=True)
        scheme = QuantScheme.per_tensor("bfloat16")
        out = quantize_bfloat(x, scheme=scheme, backwards_scheme=None)
        out.sum().backward()
        # With no backwards_scheme, gradient is identity
        assert torch.equal(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# vec_* with scheme
# ---------------------------------------------------------------------------

class TestVecScheme:

    def test_vec_quantize_scheme(self):
        """vec_quantize with scheme should match compat wrapper."""
        from src.quantize.vector import vec_quantize
        from src.quantize.elemwise import quantize_elemwise_op

        torch.manual_seed(42)
        A = torch.randn(4, 32)
        scheme = QuantScheme.per_tensor("bfloat16")
        scheme_out = vec_quantize(A.clone(), scheme=scheme)

        from src.specs.specs import finalize_mx_specs
        mx_specs = finalize_mx_specs({"bfloat": 16})
        compat_out = quantize_elemwise_op(A.clone(), mx_specs=mx_specs)
        assert torch.equal(scheme_out, compat_out)

    def test_vec_add_scheme(self):
        """vec_add with scheme should produce quantized output."""
        from src.quantize.vector import vec_add

        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        scheme = QuantScheme.per_tensor("bfloat16")
        out = vec_add(a.clone(), b.clone(), scheme=scheme)
        assert out.shape == a.shape

    def test_vec_exp_with_use_exp2_scheme(self):
        """vec_exp with scheme and use_exp2=True should work."""
        from src.quantize.vector import vec_exp

        torch.manual_seed(42)
        A = torch.randn(4, 8)
        scheme = QuantScheme.per_tensor("bfloat16")
        out = vec_exp(A.clone(), scheme=scheme, use_exp2=True)
        assert out.shape == A.shape

    def test_vec_div_with_use_recip_scheme(self):
        """vec_div with scheme and use_recip=True should work."""
        from src.quantize.vector import vec_div

        torch.manual_seed(42)
        a = torch.randn(4, 8) + 2.0
        b = torch.randn(4, 8) + 2.0
        scheme = QuantScheme.per_tensor("bfloat16")
        out = vec_div(a.clone(), b.clone(), scheme=scheme, use_recip=True)
        assert out.shape == a.shape

    def test_dispatch_rejects_both_scheme_and_mx_specs(self):
        """Providing both scheme and mx_specs should raise TypeError."""
        from src.quantize.vector import _dispatch_quantize
        from src.specs.specs import finalize_mx_specs

        A = torch.randn(4, 8)
        scheme = QuantScheme.per_tensor("bfloat16")
        mx_specs = finalize_mx_specs({"bfloat": 16})
        with pytest.raises(TypeError, match="Cannot specify both"):
            _dispatch_quantize(A, scheme=scheme, mx_specs=mx_specs)

    def test_dispatch_rejects_scheme_with_round_mode(self):
        """Providing scheme with round_mode should raise TypeError."""
        from src.quantize.vector import _dispatch_quantize

        A = torch.randn(4, 8)
        scheme = QuantScheme.per_tensor("bfloat16")
        with pytest.raises(TypeError, match="round_mode is ignored"):
            _dispatch_quantize(A, scheme=scheme, round_mode="floor")


# ---------------------------------------------------------------------------
# expand_and_reshape raises
# ---------------------------------------------------------------------------

class TestQuantizeMxOpCompat:

    def test_expand_and_reshape_raises(self):
        """expand_and_reshape=True should raise NotImplementedError."""
        from src.quantize.mx_quantize import quantize_mx_op
        from src.specs.specs import finalize_mx_specs

        A = torch.randn(4, 64)
        mx_specs = finalize_mx_specs(
            {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3",
             "block_size": 32, "bfloat": 16})
        with pytest.raises(NotImplementedError, match="expand_and_reshape"):
            quantize_mx_op(A, mx_specs=mx_specs, elem_format="fp8_e4m3",
                           expand_and_reshape=True)
