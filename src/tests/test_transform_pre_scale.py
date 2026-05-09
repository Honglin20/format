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
        x = torch.ones(4, 3)
        t = PreScaleTransform(scale=scale)
        out = t.forward(x)
        assert torch.allclose(out[:, 0], torch.full((4,), 2.0))
        assert torch.allclose(out[:, 1], torch.full((4,), 0.5))
        assert torch.allclose(out[:, 2], torch.full((4,), 1.0))

    def test_inverse_divides(self):
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([2.0, 0.5, 1.0])
        x_q = torch.ones(4, 3) * 3.0
        t = PreScaleTransform(scale=scale)
        out = t.inverse(x_q)
        assert torch.allclose(out[:, 0], torch.full((4,), 1.5))
        assert torch.allclose(out[:, 1], torch.full((4,), 6.0))

    def test_roundtrip_identity(self):
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.rand(8)
        x = torch.randn(16, 8)
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
        x = torch.ones(3, 2)
        out1 = t.forward(x)
        scale[0] = 10.0
        out2 = t.forward(x)
        assert torch.allclose(out2[:, 0], torch.full((3,), 10.0))
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
        x = torch.ones(4, 3)  # shape (4, 3)
        t = PreScaleTransform(scale=scale)
        out = t.forward(x)
        assert out.shape == (4, 3)
        assert torch.allclose(out[0, :], scale)

    def test_rejects_non_tensor(self):
        from src.transform.pre_scale import PreScaleTransform
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            PreScaleTransform(scale=[1.0, 2.0])

    # ---- PoT (power-of-two) tests ----

    def test_pot_forward_is_power_of_two(self):
        """With pot=True, scale is projected to nearest PoT before use."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([3.0, 0.3, 7.0])  # → 4, 0.25, 8
        x = torch.ones(4, 3)
        t = PreScaleTransform(scale=scale, pot=True)
        out = t.forward(x)
        assert torch.allclose(out[:, 0], torch.full((4,), 4.0))
        assert torch.allclose(out[:, 1], torch.full((4,), 0.25))
        assert torch.allclose(out[:, 2], torch.full((4,), 8.0))

    def test_pot_inverse_divides_by_pot(self):
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([3.0, 0.3])  # → 4, 0.25
        x_q = torch.ones(3, 2) * 4.0
        t = PreScaleTransform(scale=scale, pot=True)
        out = t.inverse(x_q)
        assert torch.allclose(out[:, 0], torch.ones(3))
        assert torch.allclose(out[:, 1], torch.full((3,), 16.0))  # 4/0.25=16

    def test_pot_scale_values_are_exact_powers_of_two(self):
        from src.transform.pre_scale import PreScaleTransform, _pot_scale
        scale = torch.tensor([1.7, 3.14, 0.4, 15.0])
        pot = _pot_scale(scale)
        expected = torch.tensor([2.0, 4.0, 0.5, 16.0])
        assert torch.equal(pot, expected)
        # Verify they are exact powers of two
        log2 = torch.log2(pot)
        assert torch.equal(log2, torch.round(log2))

    def test_pot_eq_differs_from_non_pot(self):
        from src.transform.pre_scale import PreScaleTransform
        s = torch.tensor([2.0])
        t1 = PreScaleTransform(scale=s, pot=True)
        t2 = PreScaleTransform(scale=s, pot=False)
        assert t1 != t2

    def test_pot_hash_differs_from_non_pot(self):
        from src.transform.pre_scale import PreScaleTransform
        s = torch.tensor([2.0])
        t1 = PreScaleTransform(scale=s, pot=True)
        t2 = PreScaleTransform(scale=s, pot=False)
        assert hash(t1) != hash(t2)

    def test_pot_roundtrip_preserves_shape(self):
        """With pot=True, roundtrip values are approximate but shape is preserved."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.rand(4)
        x = torch.randn(8, 4)
        t = PreScaleTransform(scale=scale, pot=True)
        out = t.inverse(t.forward(x))
        assert out.shape == x.shape

    def test_pot_integration_with_quant_scheme(self):
        """QuantScheme with PoT PreScaleTransform works in quantize()."""
        from src.transform.pre_scale import PreScaleTransform
        from src.formats.base import FormatBase
        fmt = FormatBase.from_str("int8")
        scale = torch.tensor([3.0])  # → PoT 4
        scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec.per_tensor(),
            transform=PreScaleTransform(scale=scale, pot=True),
        )
        x = torch.tensor([0.1, -0.2, 0.3])
        out = quantize(x, scheme)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_pot_repr(self):
        from src.transform.pre_scale import PreScaleTransform
        t = PreScaleTransform(scale=torch.ones(3), pot=True)
        assert "pot=True" in repr(t)

    def test_pot_rejects_non_bool(self):
        from src.transform.pre_scale import PreScaleTransform
        with pytest.raises(TypeError, match="pot must be a bool"):
            PreScaleTransform(scale=torch.ones(1), pot="yes")

    # ---- channel_axis tests ----

    def test_channel_axis_0(self):
        """Explicit channel_axis=0 broadcasts along first dim."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([2.0, 0.5, 1.0])
        x = torch.ones(3, 4)
        t = PreScaleTransform(scale=scale, channel_axis=0)
        out = t.forward(x)
        assert torch.allclose(out[0], torch.full((4,), 2.0))
        assert torch.allclose(out[1], torch.full((4,), 0.5))
        assert torch.allclose(out[2], torch.full((4,), 1.0))

    def test_channel_axis_1_conv2d(self):
        """Simulate Conv2d with channel_axis=1 and (N, C, H, W) input."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([2.0, 0.5])
        x = torch.ones(3, 2, 5, 5)
        t = PreScaleTransform(scale=scale, channel_axis=1)
        out = t.forward(x)
        assert out.shape == (3, 2, 5, 5)
        assert torch.allclose(out[:, 0, :, :], torch.full((3, 5, 5), 2.0))
        assert torch.allclose(out[:, 1, :, :], torch.full((3, 5, 5), 0.5))
        # Roundtrip
        assert torch.allclose(t.inverse(out), x)

    def test_channel_axis_default_is_last_dim(self):
        """channel_axis=-1 default broadcasts along last dim."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([3.0, 7.0])
        x = torch.ones(4, 2)
        t = PreScaleTransform(scale=scale)
        out = t.forward(x)
        assert torch.allclose(out[:, 0], torch.full((4,), 3.0))
        assert torch.allclose(out[:, 1], torch.full((4,), 7.0))

    def test_channel_axis_out_of_bounds(self):
        """Out-of-bounds channel_axis raises ValueError at forward time."""
        from src.transform.pre_scale import PreScaleTransform
        scale = torch.tensor([2.0, 3.0])
        t = PreScaleTransform(scale=scale, channel_axis=5)
        x = torch.ones(3, 2)
        with pytest.raises(ValueError, match="channel_axis=5"):
            t.forward(x)

    def test_channel_axis_in_eq_hash(self):
        """PreScaleTransforms with different channel_axis are not equal."""
        from src.transform.pre_scale import PreScaleTransform
        s = torch.tensor([2.0, 3.0])
        t1 = PreScaleTransform(scale=s, channel_axis=0)
        t2 = PreScaleTransform(scale=s, channel_axis=1)
        assert t1 != t2
        assert hash(t1) != hash(t2)

    def test_channel_axis_in_repr(self):
        """repr includes channel_axis when non-default."""
        from src.transform.pre_scale import PreScaleTransform
        t = PreScaleTransform(scale=torch.ones(3), channel_axis=1)
        assert "channel_axis=1" in repr(t)
        # Default -1 is omitted
        t2 = PreScaleTransform(scale=torch.ones(3))
        assert "channel_axis" not in repr(t2)

    def test_channel_axis_rejects_non_int(self):
        from src.transform.pre_scale import PreScaleTransform
        with pytest.raises(TypeError, match="channel_axis must be an int"):
            PreScaleTransform(scale=torch.ones(1), channel_axis="last")
