"""
Tests for QuantScheme-driven APIs (quantize, quantize_bfloat, vec_* with scheme).

Verify bit-identical output to old mx/ code (direct comparison, no MxSpecs).
"""
import pytest
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.transform import IdentityTransform


# ---------------------------------------------------------------------------
# quantize(x, scheme) — canonical entry point for per_block / per_tensor
# ---------------------------------------------------------------------------

class TestQuantizePerBlock:

    @pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp4_e2m1", "int8"])
    def test_per_block_matches_old(self, fmt):
        """quantize(x, QuantScheme.mxfp(fmt, 32)) matches old mx quantize_mx_op."""
        from src.quantize.elemwise import quantize
        from mx.mx_ops import quantize_mx_op as old_qmx_op
        from mx.specs import finalize_mx_specs as old_finalize

        torch.manual_seed(42)
        A = torch.randn(4, 64)
        config = {"w_elem_format": fmt, "a_elem_format": fmt,
                  "block_size": 32, "bfloat": 16}
        old_specs = old_finalize(config.copy())

        scheme = QuantScheme.mxfp(fmt, block_size=32)

        old_out = old_qmx_op(A.clone(), mx_specs=old_specs,
                             elem_format=fmt, axes=[-1])
        scheme_out = quantize(A.clone(), scheme)
        assert torch.equal(old_out, scheme_out), f"mismatch for {fmt}"

    def test_none_scheme_passthrough(self):
        """scheme=None should pass through unchanged."""
        from src.quantize.elemwise import quantize
        A = torch.randn(4, 64)
        out = quantize(A.clone(), None)
        assert torch.equal(A, out)

    def test_per_channel_supported(self):
        """quantize() supports PER_CHANNEL granularity."""
        from src.quantize.elemwise import quantize
        A = torch.randn(4, 64)
        scheme = QuantScheme.per_channel("fp8_e4m3", axis=0)
        result = quantize(A, scheme)
        assert result.shape == A.shape
        assert result.isfinite().all()

    def test_per_tensor_mx_shared_exp(self):
        """quantize_mx per_tensor (shared exponents) matches old mx quantize_mx_op."""
        from src.quantize.mx_quantize import quantize_mx
        from mx.mx_ops import quantize_mx_op as old_qmx_op
        from mx.specs import finalize_mx_specs as old_finalize

        torch.manual_seed(42)
        A = torch.randn(4, 64)
        config = {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3",
                  "block_size": 0, "bfloat": 16}
        old_specs = old_finalize(config.copy())

        scheme = QuantScheme(format="fp8_e4m3",
                             granularity=GranularitySpec.per_tensor())
        old_out = old_qmx_op(A.clone(), mx_specs=old_specs,
                             elem_format="fp8_e4m3", axes=[-1])
        scheme_out = quantize_mx(A.clone(), scheme=scheme, axes=[-1])
        assert torch.equal(old_out, scheme_out)

    def test_transform_applied(self):
        """quantize() correctly applies transforms during quantization."""
        from src.quantize.elemwise import quantize
        from src.transform.pre_scale import PreScaleTransform

        torch.manual_seed(42)
        A = torch.randn(4, 64)
        scale = torch.ones(1) * 2.0
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_block(32),
            transform=PreScaleTransform(scale=scale),
        )
        out = quantize(A.clone(), scheme)
        assert out.shape == A.shape
        assert out.isfinite().all()


# ---------------------------------------------------------------------------
# quantize_bfloat(x, scheme)
# ---------------------------------------------------------------------------

class TestQuantizeBfloatScheme:

    def test_quantize_bfloat_matches_old(self):
        """quantize_bfloat(scheme) should match old quantize_bfloat."""
        from src.quantize.bfloat_quantize import quantize_bfloat
        from mx.quantize import quantize_bfloat as old_qbf
        from mx.specs import finalize_mx_specs as old_finalize

        torch.manual_seed(42)
        x = torch.randn(4, 32)
        old_specs = old_finalize({"bfloat": 16})

        scheme = QuantScheme.per_tensor("bfloat16")
        old_out = old_qbf(x.clone(), mx_specs=old_specs)
        scheme_out = quantize_bfloat(x.clone(), scheme=scheme)
        assert torch.equal(old_out, scheme_out)

    def test_quantize_bfloat_none_scheme(self):
        """scheme=None should pass through unchanged."""
        from src.quantize.bfloat_quantize import quantize_bfloat
        x = torch.randn(4, 32)
        out = quantize_bfloat(x.clone(), scheme=None)
        assert torch.equal(x, out)

    def test_quantize_bfloat_backward_matches_old(self):
        """quantize_bfloat(scheme) backward should match old code backward."""
        from src.quantize.bfloat_quantize import quantize_bfloat
        from mx.quantize import quantize_bfloat as old_qbf
        from mx.specs import finalize_mx_specs as old_finalize

        torch.manual_seed(42)
        x1 = torch.randn(4, 32, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        scheme = QuantScheme.per_tensor("bfloat16")
        old_specs = old_finalize({"bfloat": 16})

        out1 = quantize_bfloat(x1, scheme=scheme)
        out1.sum().backward()

        out2 = old_qbf(x2, mx_specs=old_specs)
        out2.sum().backward()

        assert torch.equal(x1.grad, x2.grad), "backward mismatch vs old"

    def test_quantize_bfloat_backward_no_bp_matches_old(self):
        """quantize_bfloat with backwards_scheme=None should match old with quantize_backprop=False."""
        from src.quantize.bfloat_quantize import quantize_bfloat
        from mx.quantize import quantize_bfloat as old_qbf
        from mx.specs import finalize_mx_specs as old_finalize

        torch.manual_seed(42)
        x1 = torch.randn(4, 32, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        scheme = QuantScheme.per_tensor("bfloat16")
        old_specs = old_finalize({"bfloat": 16, "quantize_backprop": False})

        out1 = quantize_bfloat(x1, scheme=scheme, backwards_scheme=None)
        out1.sum().backward()

        out2 = old_qbf(x2, mx_specs=old_specs)
        out2.sum().backward()

        assert torch.equal(x1.grad, x2.grad), "backward mismatch vs old (no bp)"

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
        """vec_quantize with scheme should match old code."""
        from src.ops.vec_ops import vec_quantize
        from mx import vector_ops as old_vec
        from mx.specs import finalize_mx_specs as old_finalize

        torch.manual_seed(42)
        A = torch.randn(4, 32)
        scheme = QuantScheme.per_tensor("bfloat16")
        scheme_out = vec_quantize(A.clone(), scheme=scheme)

        old_specs = old_finalize({"bfloat": 16})
        old_out = old_vec.vec_quantize(A.clone(), mx_specs=old_specs)
        assert torch.equal(scheme_out, old_out)

    def test_vec_add_scheme(self):
        """vec_add with scheme should produce quantized output."""
        from src.ops.vec_ops import vec_add

        torch.manual_seed(42)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        scheme = QuantScheme.per_tensor("bfloat16")
        out = vec_add(a.clone(), b.clone(), scheme=scheme)
        assert out.shape == a.shape

    def test_vec_exp_with_use_exp2_scheme(self):
        """vec_exp with scheme and use_exp2=True should work."""
        from src.ops.vec_ops import vec_exp

        torch.manual_seed(42)
        A = torch.randn(4, 8)
        scheme = QuantScheme.per_tensor("bfloat16")
        out = vec_exp(A.clone(), scheme=scheme, use_exp2=True)
        assert out.shape == A.shape

    def test_vec_div_with_use_recip_scheme(self):
        """vec_div with scheme and use_recip=True should work."""
        from src.ops.vec_ops import vec_div

        torch.manual_seed(42)
        a = torch.randn(4, 8) + 2.0
        b = torch.randn(4, 8) + 2.0
        scheme = QuantScheme.per_tensor("bfloat16")
        out = vec_div(a.clone(), b.clone(), scheme=scheme, use_recip=True)
        assert out.shape == a.shape
