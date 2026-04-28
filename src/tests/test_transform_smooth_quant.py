"""
Tests for SmoothQuantTransform — per-channel activation smoothing.

Phase 8, sub-task 8A.2. Tests follow TDD: written before the implementation.
"""
import pytest
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.transform import IdentityTransform
from src.quantize.elemwise import quantize


# ============================================================================
# Helpers
# ============================================================================

def _try_import():
    """Import SmoothQuant symbols; skip test if module not yet implemented."""
    try:
        from src.transform.smooth_quant import (
            compute_smoothquant_scale,
            SmoothQuantTransform,
        )
        return compute_smoothquant_scale, SmoothQuantTransform
    except (ImportError, AttributeError):
        pytest.skip("src.transform.smooth_quant not yet implemented")


def _try_import_hadamard():
    """Import HadamardTransform for cross-type inequality tests."""
    try:
        from src.transform import HadamardTransform
        return HadamardTransform
    except (ImportError, AttributeError):
        pytest.skip("src.transform.hadamard not yet implemented")


# ============================================================================
# 1. compute_smoothquant_scale function tests
# ============================================================================

class TestComputeSmoothQuantScale:

    def test_compute_scale_alpha_half(self):
        """alpha=0.5: s_j uses both activation and weight."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        # Activation: per-channel max values [C]
        act_max = torch.tensor([2.0, 4.0, 8.0])
        # Weight: [OC, IC] for linear
        W = torch.tensor([[1.0, 3.0, 5.0],
                          [2.0, 4.0, 6.0],
                          [3.0, 5.0, 7.0]])  # [3, 3] -> OC=3, IC=3
        # max(|W|) along dims except OC (dim 0): [3.0, 5.0, 7.0]  (max of each row)
        # s = max(|X|)^0.5 / max(|W|)^0.5
        #   = [2.0^0.5 / 3.0^0.5, 4.0^0.5 / 5.0^0.5, 8.0^0.5 / 7.0^0.5]
        #   = [sqrt(2/3), sqrt(4/5), sqrt(8/7)]
        expected = torch.sqrt(act_max / torch.tensor([3.0, 5.0, 7.0]))
        scale = compute_smoothquant_scale(act_max, W, alpha=0.5)
        assert scale.shape == (3,), f"Expected shape (3,), got {scale.shape}"
        assert torch.allclose(scale, expected, atol=1e-6), \
            f"Scale mismatch: {scale} != {expected}"

    def test_compute_scale_alpha_zero(self):
        """alpha=0: s_j only depends on weight (1/max(|W|))."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        act_max = torch.tensor([2.0, 4.0, 8.0])
        W = torch.tensor([[1.0, 3.0, 5.0],
                          [2.0, 4.0, 6.0],
                          [3.0, 5.0, 7.0]])  # [3, 3]
        # s_j = max(|X|)^0 / max(|W|)^1 = 1 / max(|W|)
        # max(|W|) along dims except OC: [3.0, 5.0, 7.0]
        # s = [1/3, 1/5, 1/7]
        expected = 1.0 / torch.tensor([3.0, 5.0, 7.0])
        scale = compute_smoothquant_scale(act_max, W, alpha=0.0)
        assert torch.allclose(scale, expected, atol=1e-6), \
            f"Scale mismatch: {scale} != {expected}"

    def test_compute_scale_alpha_one(self):
        """alpha=1: s_j only depends on activation (max(|X|))."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        act_max = torch.tensor([2.0, 4.0, 8.0])
        W = torch.tensor([[1.0, 3.0, 5.0],
                          [2.0, 4.0, 6.0],
                          [3.0, 5.0, 7.0]])  # [3, 3]
        # s_j = max(|X|)^1 / max(|W|)^0 = max(|X|)
        # s = [2.0, 4.0, 8.0]
        expected = torch.tensor([2.0, 4.0, 8.0])
        scale = compute_smoothquant_scale(act_max, W, alpha=1.0)
        assert torch.allclose(scale, expected, atol=1e-6), \
            f"Scale mismatch: {scale} != {expected}"

    def test_compute_scale_shape(self):
        """Output is 1D [C] matching input channel count (activation's last dim)."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        # Activation last dim = C must equal Weight input channel dim (dim 1)
        # X [N, C=16], W [OC=8, IC=16] → scale [16]
        X = torch.randn(4, 16)
        W = torch.randn(8, 16)  # [OC=8, IC=16]
        scale = compute_smoothquant_scale(X, W, alpha=0.5)
        assert scale.ndim == 1, f"Expected 1D, got {scale.ndim}D"
        assert scale.shape[0] == 16, f"Expected 16 channels, got {scale.shape[0]}"

    def test_compute_scale_with_1d_activation_stats(self):
        """X_act is pre-computed 1D per-channel stats."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        act_stats = torch.randn(12).abs()  # [C=12] pre-computed max per channel
        W = torch.randn(6, 12)  # [OC=6, IC=12]
        scale = compute_smoothquant_scale(act_stats, W, alpha=0.5)
        assert scale.ndim == 1
        assert scale.shape[0] == 12

    def test_compute_scale_conv_weight(self):
        """Works with Conv weight shape [OC, IC, H, W] using pre-computed act stats."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        # Use pre-computed 1D activation stats (most common pattern for Conv)
        act_stats = torch.randn(4).abs()  # [IC=4] per-channel max
        W = torch.randn(16, 4, 3, 3)  # [OC=16, IC=4, H=3, W=3]
        scale = compute_smoothquant_scale(act_stats, W, alpha=0.5)
        assert scale.ndim == 1, f"Expected 1D, got {scale.ndim}D"
        assert scale.shape[0] == 4, f"Expected 4 input channels, got {scale.shape[0]}"

    def test_compute_scale_all_positive(self):
        """All scale values are strictly positive."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        X = torch.randn(4, 16)  # [N, C=16]
        W = torch.randn(8, 16)  # [OC=8, IC=16]
        scale = compute_smoothquant_scale(X, W, alpha=0.5)
        assert (scale > 0).all(), f"Some scale values are not positive: {scale}"

    def test_compute_scale_numerical_stability(self):
        """Handles zero max values gracefully (clamp prevents division by zero)."""
        compute_smoothquant_scale, _ = _try_import()
        # Activation with zero max channel
        X = torch.zeros(4, 8)  # [N, C=8]
        W = torch.randn(8, 8)  # [OC=8, IC=8]
        scale = compute_smoothquant_scale(X, W, alpha=0.5)
        assert not torch.isnan(scale).any(), "NaN in scale"
        assert not torch.isinf(scale).any(), "Inf in scale"
        assert (scale >= 0).all(), "All values should be non-negative"
        # Weight with zero max channel
        X = torch.randn(4, 8)
        W = torch.zeros(8, 8)  # [OC=8, IC=8]
        scale = compute_smoothquant_scale(X, W, alpha=0.5)
        assert not torch.isnan(scale).any(), "NaN in scale"
        assert not torch.isinf(scale).any(), "Inf in scale"


    def test_act_channel_axis_conv2d(self):
        """act_channel_axis=1 reduces over all dims except dim=1 for Conv2d-style activation."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        # Conv2d activation: (N=2, C=3, H=4, W=5) — channel at dim=1
        X = torch.randn(2, 3, 4, 5)
        W = torch.randn(8, 3, 3, 3)  # Conv weight (OC=8, IC=3, KH=3, KW=3)

        scale = compute_smoothquant_scale(X, W, alpha=0.5,
                                           act_channel_axis=1, w_channel_axis=1)

        assert scale.shape == (3,), f"Expected (3,), got {scale.shape}"

        # Verify act_channel_axis=1 gives same result as manually computing per-channel
        act_amax_manual = torch.amax(torch.abs(X), dim=(0, 2, 3))  # reduce all except dim 1
        w_amax_manual = torch.amax(torch.abs(W), dim=(0, 2, 3))    # reduce all except dim 1
        expected = (act_amax_manual.clamp(min=1e-12).pow(0.5)
                    / w_amax_manual.clamp(min=1e-12).pow(0.5))
        assert torch.allclose(scale, expected, atol=1e-6)

    def test_act_channel_axis_default(self):
        """Default act_channel_axis=-1 preserves backward compatibility."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        X = torch.randn(4, 16)
        W = torch.randn(8, 16)
        scale_new = compute_smoothquant_scale(X, W, act_channel_axis=-1, w_channel_axis=1)
        scale_old = compute_smoothquant_scale(X, W)
        assert torch.equal(scale_new, scale_old)

    def test_w_channel_axis_zero(self):
        """w_channel_axis=0: reduce all except dim 0 (unusual weight layout)."""
        compute_smoothquant_scale, _ = _try_import()
        torch.manual_seed(42)
        # Unusual weight: [IC=4, OC=8] (transposed)
        W = torch.randn(4, 8)  # [IC, OC] — input channel at dim 0
        act_stats = torch.randn(4).abs()
        scale = compute_smoothquant_scale(act_stats, W, w_channel_axis=0)
        assert scale.shape == (4,)

    def test_from_calibration_passes_act_channel_axis(self):
        """from_calibration passes act_channel_axis through to the transform."""
        _, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        X = torch.randn(2, 3, 4, 5)  # Conv2d-style: channel at dim=1
        W = torch.randn(8, 3, 3, 3)
        t = SmoothQuantTransform.from_calibration(
            X, W, act_channel_axis=1
        )
        assert t.channel_axis == 1
        assert t.scale.shape == (3,)


# ============================================================================
# 2. SmoothQuantTransform class tests
# ============================================================================

class TestSmoothQuantTransform:

    def test_roundtrip(self):
        """SmoothQuantTransform.forward then .inverse recovers original exactly."""
        _, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        scale = torch.tensor([0.5, 2.0, 1.0])
        t = SmoothQuantTransform(scale)
        x = torch.randn(4, 3)
        xt = t.forward(x)
        xr = t.inverse(xt)
        assert xr.shape == x.shape, f"Shape mismatch: {xr.shape} != {x.shape}"
        assert torch.equal(xr, x), \
            f"Roundtrip failed: max diff = {(xr - x).abs().max()}"

    def test_roundtrip_higher_rank(self):
        """Roundtrip works for 3D and 4D tensors.

        Uses only power-of-2 scale values so that division and multiplication
        are exact (no floating-point rounding for normal-range values).
        """
        _, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        scale = torch.tensor([0.5, 2.0, 1.0, 4.0])
        t = SmoothQuantTransform(scale)
        # 3D: [batch, seq, channels]
        x3 = torch.randn(2, 4, 4)
        xr3 = t.inverse(t.forward(x3))
        assert torch.equal(xr3, x3), "3D roundtrip failed"
        # 4D: [batch, seq_left, seq_right, channels]
        x4 = torch.randn(2, 3, 4, 4)
        xr4 = t.inverse(t.forward(x4))
        assert torch.equal(xr4, x4), "4D roundtrip failed"

    def test_forward_applies_division(self):
        """forward(x) = x / scale, verified numerically."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantTransform(scale)
        x = torch.tensor([[10.0, 20.0],
                          [30.0, 40.0],
                          [50.0, 60.0]])
        expected = torch.tensor([[5.0, 5.0],
                                 [15.0, 10.0],
                                 [25.0, 15.0]])
        result = t.forward(x)
        assert torch.equal(result, expected), f"Forward mismatch: {result} != {expected}"

    def test_inverse_applies_multiplication(self):
        """inverse(x_q) = x_q * scale, verified numerically."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantTransform(scale)
        x_q = torch.tensor([[5.0, 5.0],
                            [15.0, 10.0],
                            [25.0, 15.0]])
        expected = torch.tensor([[10.0, 20.0],
                                 [30.0, 40.0],
                                 [50.0, 60.0]])
        result = t.inverse(x_q)
        assert torch.equal(result, expected), f"Inverse mismatch: {result} != {expected}"

    def test_invertible_flag(self):
        """SmoothQuantTransform.invertible is True."""
        _, SmoothQuantTransform = _try_import()
        t = SmoothQuantTransform(torch.tensor([1.0, 2.0]))
        assert t.invertible is True

    def test_is_transform_base(self):
        """SmoothQuantTransform is a TransformBase subclass."""
        _, SmoothQuantTransform = _try_import()
        from src.scheme.transform import TransformBase
        assert issubclass(SmoothQuantTransform, TransformBase)

    def test_scale_stored_as_clone(self):
        """Scale is stored as a clone; modifying original doesn't affect transform.

        Uses power-of-2 scale values so that the roundtrip is bit-exact.
        """
        _, SmoothQuantTransform = _try_import()
        original = torch.tensor([1.0, 2.0, 4.0])
        t = SmoothQuantTransform(original)
        original[0] = 999.0  # Modify original
        x = torch.randn(4, 3)
        xr = t.inverse(t.forward(x))
        assert torch.equal(xr, x), "Scale was not cloned properly"

    def test_scale_positive(self):
        """All scale values are strictly positive."""
        _, SmoothQuantTransform = _try_import()
        t = SmoothQuantTransform(torch.tensor([0.1, 2.5, 1.0]))
        assert (t.scale > 0).all(), \
            f"Scale should be positive, got {t.scale}"

    def test_from_calibration(self):
        """Factory method creates valid transform."""
        compute_smoothquant_scale, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        # Activation last dim (C=16) must match weight IC (dim 1 = 16)
        X = torch.randn(4, 16)
        W = torch.randn(8, 16)
        t = SmoothQuantTransform.from_calibration(X, W, alpha=0.5)
        assert isinstance(t, SmoothQuantTransform)
        assert t.invertible is True
        # Roundtrip should recover original (scale is not power-of-2, so use allclose)
        x = torch.randn(4, 16)
        xr = t.inverse(t.forward(x))
        assert torch.allclose(xr, x, atol=1e-6), \
            f"from_calibration roundtrip failed: max diff = {(xr - x).abs().max()}"

    def test_from_calibration_default_alpha(self):
        """from_calibration uses default alpha=0.5."""
        compute_smoothquant_scale, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        X = torch.randn(4, 16)
        W = torch.randn(8, 16)
        t1 = SmoothQuantTransform.from_calibration(X, W, alpha=0.5)
        t2 = SmoothQuantTransform.from_calibration(X, W)  # default
        assert t1 == t2, "Default alpha should be 0.5"

    def test_eq_same_scale(self):
        """Two instances with same scale are equal."""
        _, SmoothQuantTransform = _try_import()
        t1 = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        assert t1 == t2

    def test_eq_different_scale(self):
        """Two instances with different scale are not equal."""
        _, SmoothQuantTransform = _try_import()
        t1 = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantTransform(torch.tensor([0.5, 3.0, 1.0]))
        assert t1 != t2

    def test_hash_consistent(self):
        """Same scale produces same hash."""
        _, SmoothQuantTransform = _try_import()
        t1 = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        assert hash(t1) == hash(t2)

    def test_hash_different_scale(self):
        """Different scale produces different hash."""
        _, SmoothQuantTransform = _try_import()
        t1 = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantTransform(torch.tensor([0.5, 3.0, 1.0]))
        # Very unlikely to collide with different float values
        assert hash(t1) != hash(t2)

    def test_different_from_identity(self):
        """SmoothQuantTransform != IdentityTransform."""
        _, SmoothQuantTransform = _try_import()
        t = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        assert t != IdentityTransform()
        assert hash(t) != hash(IdentityTransform())

    def test_not_equal_to_hadamard(self):
        """SmoothQuantTransform != HadamardTransform."""
        _, SmoothQuantTransform = _try_import()
        HadamardTransform = _try_import_hadamard()
        t_sq = SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        t_h = HadamardTransform()
        assert t_sq != t_h
        assert hash(t_sq) != hash(t_h)

    def test_channel_axis_conv2d_activation(self):
        """channel_axis=1: scale broadcasts to dim=1 for Conv2d-style input (N, C, H, W)."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([2.0, 4.0, 1.0])  # 3 channels
        t = SmoothQuantTransform(scale, channel_axis=1)
        # Conv2d-style input: (N=2, C=3, H=2, W=2)
        x = torch.full((2, 3, 2, 2), 1.0)  # all ones so division result is just 1/s
        result = t.forward(x)
        # Channel 0 divided by 2.0, channel 1 by 4.0, channel 2 by 1.0
        assert torch.equal(result[:, 0:1], x[:, 0:1] / 2.0)
        assert torch.equal(result[:, 1:2], x[:, 1:2] / 4.0)
        assert torch.equal(result[:, 2:3], x[:, 2:3] / 1.0)

    def test_channel_axis_default_is_minus_one(self):
        """Default channel_axis=-1 preserves backward compatibility."""
        _, SmoothQuantTransform = _try_import()
        t1 = SmoothQuantTransform(torch.tensor([2.0, 4.0]))
        t2 = SmoothQuantTransform(torch.tensor([2.0, 4.0]), channel_axis=-1)
        assert t1 == t2

    def test_channel_axis_roundtrip_conv2d(self):
        """Roundtrip with channel_axis=1 recovers original (Conv2d layout)."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([0.5, 2.0, 1.0, 4.0])
        t = SmoothQuantTransform(scale, channel_axis=1)
        x = torch.randn(2, 4, 4, 4)  # (N=2, C=4, H=4, W=4)
        xr = t.inverse(t.forward(x))
        assert torch.equal(xr, x), f"Roundtrip failed with channel_axis=1"

    def test_eq_different_channel_axis(self):
        """Transforms with same scale but different channel_axis are not equal."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([0.5, 2.0, 1.0])
        t1 = SmoothQuantTransform(scale, channel_axis=-1)
        t2 = SmoothQuantTransform(scale, channel_axis=1)
        assert t1 != t2

    def test_hash_different_channel_axis(self):
        """Transforms with same scale but different channel_axis have different hash."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([0.5, 2.0, 1.0])
        t1 = SmoothQuantTransform(scale, channel_axis=-1)
        t2 = SmoothQuantTransform(scale, channel_axis=1)
        assert hash(t1) != hash(t2)

    def test_channel_axis_out_of_bounds_raises(self):
        """Out-of-bounds channel_axis raises ValueError with clear message."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantTransform(scale, channel_axis=5)  # 5 > ndim=2
        x = torch.randn(4, 2)
        with pytest.raises(ValueError, match="out of bounds"):
            t.forward(x)


class TestSmoothQuantWeightTransform:

    @staticmethod
    def _try_import_wt():
        try:
            from src.transform.smooth_quant import SmoothQuantWeightTransform
            return SmoothQuantWeightTransform
        except (ImportError, AttributeError):
            pytest.skip("SmoothQuantWeightTransform not yet implemented")

    def test_roundtrip_linear_weight(self):
        """Roundtrip for Linear weight (OC, IC)."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([0.5, 2.0, 1.0])  # 3 input channels
        t = SmoothQuantWeightTransform(scale)  # default channel_axis=1
        W = torch.randn(4, 3)  # (OC=4, IC=3)
        Wr = t.inverse(t.forward(W))
        assert torch.equal(Wr, W)

    def test_roundtrip_conv_weight(self):
        """Roundtrip for Conv2d weight (OC, IC, KH, KW)."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([0.5, 2.0])
        t = SmoothQuantWeightTransform(scale, channel_axis=1)
        W = torch.randn(8, 2, 3, 3)  # (OC=8, IC=2, KH=3, KW=3)
        Wr = t.inverse(t.forward(W))
        assert torch.equal(Wr, W)

    def test_forward_linear_weight(self):
        """forward(W) = W * scale along dim=1."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantWeightTransform(scale)
        W = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        expected = torch.tensor([[2.0, 8.0],
                                 [6.0, 16.0]])
        assert torch.equal(t.forward(W), expected)

    def test_forward_conv_weight(self):
        """forward(W) = W * scale along dim=1 for Conv weight."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0])
        t = SmoothQuantWeightTransform(scale, channel_axis=1)
        W = torch.ones(4, 1, 2, 2)  # (OC=4, IC=1, H=2, W=2)
        result = t.forward(W)
        assert torch.equal(result, W * 2.0)

    def test_inverse_linear_weight(self):
        """inverse(W_q) = W_q / scale along dim=1."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantWeightTransform(scale)
        W_q = torch.tensor([[2.0, 8.0],
                            [6.0, 16.0]])
        expected = torch.tensor([[1.0, 2.0],
                                 [3.0, 4.0]])
        assert torch.equal(t.inverse(W_q), expected)

    def test_invertible_flag(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t = SmoothQuantWeightTransform(torch.tensor([1.0, 2.0]))
        assert t.invertible is True

    def test_is_transform_base(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        from src.scheme.transform import TransformBase
        assert issubclass(SmoothQuantWeightTransform, TransformBase)

    def test_scale_stored_as_clone(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        original = torch.tensor([1.0, 2.0, 4.0])
        t = SmoothQuantWeightTransform(original)
        original[0] = 999.0
        W = torch.randn(4, 3)
        Wr = t.inverse(t.forward(W))
        assert torch.equal(Wr, W)

    def test_channel_axis_default_is_one(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t1 = SmoothQuantWeightTransform(torch.tensor([2.0, 4.0]))
        t2 = SmoothQuantWeightTransform(torch.tensor([2.0, 4.0]), channel_axis=1)
        assert t1 == t2

    def test_eq_different_channel_axis(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t1 = SmoothQuantWeightTransform(scale, channel_axis=0)
        t2 = SmoothQuantWeightTransform(scale, channel_axis=1)
        assert t1 != t2

    def test_hash_different_channel_axis(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t1 = SmoothQuantWeightTransform(scale, channel_axis=0)
        t2 = SmoothQuantWeightTransform(scale, channel_axis=1)
        assert hash(t1) != hash(t2)

    def test_eq_same_scale_and_axis(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t1 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        assert t1 == t2

    def test_hash_consistent(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t1 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        assert hash(t1) == hash(t2)

    def test_channel_axis_out_of_bounds_raises(self):
        """Out-of-bounds channel_axis raises ValueError."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantWeightTransform(scale, channel_axis=5)  # 5 > ndim=2
        W = torch.randn(4, 2)
        with pytest.raises(ValueError, match="out of bounds"):
            t.forward(W)

    def test_not_equal_to_identity(self):
        """SmoothQuantWeightTransform != IdentityTransform."""
        SmoothQuantWeightTransform = self._try_import_wt()
        t = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0]))
        assert t != IdentityTransform()

    def test_not_equal_to_activation_transform(self):
        """SmoothQuantWeightTransform != SmoothQuantTransform with same scale."""
        SmoothQuantWeightTransform = self._try_import_wt()
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([0.5, 2.0, 1.0])
        t_w = SmoothQuantWeightTransform(scale)
        t_a = SmoothQuantTransform(scale)
        assert t_w != t_a


# ============================================================================
# 3. QuantScheme integration tests
# ============================================================================

class TestSmoothQuantQuantScheme:

    def test_smooth_quant_quant_scheme(self):
        """QuantScheme accepts SmoothQuantTransform."""
        _, SmoothQuantTransform = _try_import()
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=SmoothQuantTransform(torch.tensor([0.5, 2.0, 1.0]))
        )
        assert isinstance(scheme.transform, SmoothQuantTransform)

    def test_smooth_quant_quantize_pipeline(self):
        """End-to-end: quantize with SmoothQuantTransform produces correct output shape."""
        _, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=SmoothQuantTransform(torch.tensor([0.5] * 8))
        )
        result = quantize(x, scheme)
        assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
        assert not torch.isnan(result).any(), "NaN in quantized output"
        assert not torch.isinf(result).any(), "Inf in quantized output"

    def test_smooth_quant_quantize_meaningful(self):
        """SmoothQuant quantize produces different output from identity quantize."""
        _, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        scheme_sq = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=SmoothQuantTransform(torch.tensor([2.0] * 8))
        )
        scheme_id = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=IdentityTransform()
        )
        result_sq = quantize(x, scheme_sq)
        result_id = quantize(x, scheme_id)
        assert result_sq.shape == result_id.shape
        # SmoothQuant changes values, so results should differ
        assert not torch.equal(result_sq, result_id), \
            "SmoothQuant should produce different results from identity"

    def test_smooth_quant_quantize_with_per_channel(self):
        """SmoothQuant + per_channel granularity works."""
        _, SmoothQuantTransform = _try_import()
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_channel(axis=-1),
            transform=SmoothQuantTransform(torch.tensor([2.0] * 8))
        )
        result = quantize(x, scheme)
        assert result.shape == x.shape

    def test_smooth_quant_scale_property(self):
        """Scale is accessible via .scale property."""
        _, SmoothQuantTransform = _try_import()
        scale = torch.tensor([0.5, 2.0, 1.0])
        t = SmoothQuantTransform(scale)
        assert hasattr(t, 'scale'), "SmoothQuantTransform should expose .scale"
        assert torch.equal(t.scale, scale), f"scale property mismatch"
