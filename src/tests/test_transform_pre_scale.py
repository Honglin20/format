"""Tests for PreScaleTransform — learnable per-channel pre-scale via Transform slot."""
import pytest
import torch
from src.scheme.transform import TransformBase, IdentityTransform
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.quantize.elemwise import quantize


class TestPreScaleTransform:
    """Unit tests for PreScaleTransform."""

    def test_forward_multiplies(self):
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([2.0, 0.5, 1.0])
        x = torch.ones(3, 4)
        t = PreScaleTransform(scale=scale)
        out = t.forward(x)
        assert torch.allclose(out[0], torch.full((4,), 2.0))
        assert torch.allclose(out[1], torch.full((4,), 0.5))
        assert torch.allclose(out[2], torch.full((4,), 1.0))

    def test_inverse_divides(self):
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([2.0, 0.5, 1.0])
        x_q = torch.ones(3, 4) * 3.0
        t = PreScaleTransform(scale=scale)
        out = t.inverse(x_q)
        assert torch.allclose(out[0], torch.full((4,), 1.5))
        assert torch.allclose(out[1], torch.full((4,), 6.0))

    def test_roundtrip_identity(self):
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.rand(8)
        x = torch.randn(8, 16)
        t = PreScaleTransform(scale=scale)
        assert torch.allclose(t.inverse(t.forward(x)), x)

    def test_invertible_flag(self):
        from src.transform.pre_scale import PreScaleTransform
        t = PreScaleTransform(scale=torch.ones(3))
        assert t.invertible is True

    def test_eq_same_reference(self):
        from src.transform.pre_scale import PreScaleTransform
        s = torch.tensor([2.0, 3.0])
        t1 = PreScaleTransform(scale=s)
        t2 = PreScaleTransform(scale=s)
        assert t1 == t2

    def test_eq_different_reference(self):
        from src.transform.pre_scale import PreScaleTransform
        t1 = PreScaleTransform(scale=torch.tensor([2.0]))
        t2 = PreScaleTransform(scale=torch.tensor([2.0]))
        assert t1 != t2  # different objects -> not equal

    def test_eq_cross_type(self):
        from src.transform.pre_scale import PreScaleTransform
        t = PreScaleTransform(scale=torch.ones(3))
        assert t != IdentityTransform()
        assert t != "not a transform"

    def test_hash_stable(self):
        from src.transform.pre_scale import PreScaleTransform
        s = torch.tensor([1.0, 2.0])
        t = PreScaleTransform(scale=s)
        h1 = hash(t)
        h2 = hash(t)
        assert h1 == h2

    def test_hash_different_objects(self):
        from src.transform.pre_scale import PreScaleTransform
        t1 = PreScaleTransform(scale=torch.tensor([1.0]))
        t2 = PreScaleTransform(scale=torch.tensor([2.0]))
        assert hash(t1) != hash(t2)

    def test_reference_mutation_visible(self):
        """Changing the underlying tensor changes transform behavior."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([2.0, 2.0])
        t = PreScaleTransform(scale=scale)
        x = torch.ones(2, 3)
        out1 = t.forward(x)
        scale[0] = 10.0
        out2 = t.forward(x)
        assert torch.allclose(out2[0], torch.full((3,), 10.0))
        assert not torch.allclose(out1, out2)

    def test_quant_scheme_integration(self):
        """QuantScheme with PreScaleTransform works in quantize()."""
        from src.transform.pre_scale import PreScaleTransform
        from src.formats.base import FormatBase
        fmt = FormatBase.from_str("int8")
        scale = torch.tensor([0.5])
        scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec.per_tensor(),
            transform=PreScaleTransform(scale=scale),
        )
        x = torch.tensor([0.3, -0.6, 0.9])
        out = quantize(x, scheme)
        assert out.shape == x.shape

    def test_per_channel_broadcast(self):
        """Pre-scale broadcasts along last dim for per-channel scaling."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([1.0, 2.0, 0.5])  # shape (3,)
        x = torch.ones(3, 4)  # shape (3, 4)
        t = PreScaleTransform(scale=scale)
        out = t.forward(x)
        assert out.shape == (3, 4)
        assert torch.allclose(out[:, 0], scale)

    def test_rejects_non_tensor(self):
        from src.transform.pre_scale import PreScaleTransform
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            PreScaleTransform(scale=[1.0, 2.0])
