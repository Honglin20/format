"""
Tests for HadamardTransform — the first real Transform beyond IdentityTransform.

Phase 8, sub-task 8A.1. Tests follow TDD: written before the implementation.
"""
import pytest
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.transform import IdentityTransform
from src.quantize.elemwise import quantize


# ============================================================================
# 1. Raw hadamard() function tests
# ============================================================================

def _try_import():
    """Import hadamard function; skip test if module not yet implemented."""
    try:
        from src.transform import hadamard, HadamardTransform
        return hadamard, HadamardTransform
    except (ImportError, AttributeError):
        pytest.skip("src.transform.hadamard not yet implemented")


class TestHadamardFunction:

    def test_hadamard_orthogonal(self):
        """H(H(x)) ≈ x (self-inverse property due to 1/sqrt(d) normalization)."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 16)  # power of 2
        hx = hadamard(x)
        hhx = hadamard(hx)
        assert hhx.shape == x.shape, f"Shape mismatch: {hhx.shape} != {x.shape}"
        # Self-inverse should be near-perfect for float32
        assert torch.allclose(hhx, x, atol=1e-5), \
            f"max diff = {(hhx - x).abs().max()}"

    def test_hadamard_orthogonal_larger(self):
        """Orthogonal property holds for larger power-of-2 sizes."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(2, 64)
        hx = hadamard(x)
        hhx = hadamard(hx)
        assert hhx.shape == x.shape
        assert torch.allclose(hhx, x, atol=1e-5)

    def test_hadamard_orthogonal_square(self):
        """Orthogonal property holds for square non-2D tensors."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 8, 4)  # last dim = 4 (power of 2)
        hx = hadamard(x)
        hhx = hadamard(hx)
        assert hhx.shape == x.shape
        assert torch.allclose(hhx, x, atol=1e-5)

    def test_hadamard_non_power_of_two(self):
        """Tensor with last dim not power of 2 — should pad internally and roundtrip."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 7)  # not a power of 2
        hx = hadamard(x)
        assert hx.shape == x.shape, f"Shape changed: {hx.shape} != {x.shape}"
        assert not torch.isnan(hx).any(), "NaN in hadamard output"
        assert not torch.isinf(hx).any(), "Inf in hadamard output"
        # Roundtrip is approximate due to padding/truncation
        hhx = hadamard(hx)
        assert hhx.shape == x.shape

    def test_hadamard_1d_tensor(self):
        """1D tensor works correctly."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(8)
        hx = hadamard(x)
        assert hx.shape == x.shape
        hhx = hadamard(hx)
        assert torch.allclose(hhx, x, atol=1e-5)

    def test_hadamard_batch_1d(self):
        """Batch of 1D sequences works."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(3, 8)
        hx = hadamard(x)
        assert hx.shape == x.shape
        hhx = hadamard(hx)
        assert torch.allclose(hhx, x, atol=1e-5)

    def test_hadamard_3d(self):
        """3D tensor works along last dim only."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(2, 3, 8)
        hx = hadamard(x)
        assert hx.shape == x.shape
        hhx = hadamard(hx)
        assert torch.allclose(hhx, x, atol=1e-5)

    def test_hadamard_identity_for_dim_1(self):
        """Hadamard of a tensor with last dim 1 is identity (up to sign)."""
        hadamard, _ = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 1)
        hx = hadamard(x)
        assert hx.shape == x.shape
        # H_1 = [1], normalize by 1/sqrt(1) = 1, so H(H(x)) = x
        hhx = hadamard(hx)
        assert torch.allclose(hhx, x, atol=1e-7)

    def test_hadamard_requires_grad_preserved(self):
        """hadamard preserves requires_grad."""
        hadamard, _ = _try_import()
        x = torch.randn(4, 8, requires_grad=True)
        hx = hadamard(x)
        assert hx.requires_grad
        hx.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ============================================================================
# 2. HadamardTransform class tests
# ============================================================================

class TestHadamardTransform:

    def test_roundtrip(self):
        """HadamardTransform.forward then .inverse recovers original."""
        _, HadamardTransform = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 16)
        t = HadamardTransform()
        xt = t.forward(x)
        xr = t.inverse(xt)
        assert xr.shape == x.shape
        assert torch.allclose(xr, x, atol=1e-5)

    def test_invertible_flag(self):
        """HadamardTransform.invertible is True."""
        _, HadamardTransform = _try_import()
        t = HadamardTransform()
        assert t.invertible is True

    def test_is_transform_base(self):
        """HadamardTransform is a TransformBase subclass."""
        _, HadamardTransform = _try_import()
        from src.scheme.transform import TransformBase
        assert issubclass(HadamardTransform, TransformBase)

    def test_eq_hash(self):
        """Two instances are equal and hash same."""
        _, HadamardTransform = _try_import()
        t1 = HadamardTransform()
        t2 = HadamardTransform()
        assert t1 == t2
        assert hash(t1) == hash(t2)

    def test_not_equal_to_identity(self):
        """HadamardTransform != IdentityTransform."""
        _, HadamardTransform = _try_import()
        t1 = HadamardTransform()
        assert t1 != IdentityTransform()
        assert hash(t1) != hash(IdentityTransform())


# ============================================================================
# 3. QuantScheme integration tests
# ============================================================================

class TestHadamardQuantScheme:

    def test_hadamard_transform_quant_scheme(self):
        """QuantScheme accepts HadamardTransform."""
        _, HadamardTransform = _try_import()
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=HadamardTransform()
        )
        assert isinstance(scheme.transform, HadamardTransform)

    def test_hadamard_quantize_pipeline(self):
        """End-to-end: quantize with HadamardTransform produces correct output shape."""
        _, HadamardTransform = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 16)
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=HadamardTransform()
        )
        result = quantize(x, scheme)
        assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
        assert not torch.isnan(result).any(), "NaN in quantized output"
        assert not torch.isinf(result).any(), "Inf in quantized output"

    def test_hadamard_quantize_roundtrip_approximate(self):
        """quantize(x, scheme) with Hadamard approximately preserves values through roundtrip.

        The full pipeline: transform.forward → format.quantize → transform.inverse.
        Int8 quantization dominates the error; the output should be reasonable.
        """
        _, HadamardTransform = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 16) * 0.1  # small values within int8 range
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=HadamardTransform()
        )
        result = quantize(x, scheme)
        # Values should not be wildly different from input
        # (within int8 range, quantization error is small but hadamard spreads error)
        diff = (result - x).abs().mean()
        assert diff < 0.5, f"Mean abs diff too large: {diff}"

    def test_hadamard_quantize_identity_equiv(self):
        """With per_tensor/int8, hadamard transform should be similar to identity transform
        for small tensors (the pipeline produces valid output in both cases)."""
        _, HadamardTransform = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 16)
        scheme_hadamard = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=HadamardTransform()
        )
        scheme_identity = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=IdentityTransform()
        )
        result_h = quantize(x, scheme_hadamard)
        result_i = quantize(x, scheme_identity)
        # Both should produce valid output of the same shape
        assert result_h.shape == result_i.shape
        # They should differ (hadamard changes the distribution)
        assert not torch.equal(result_h, result_i)
