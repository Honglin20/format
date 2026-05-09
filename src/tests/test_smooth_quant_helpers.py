"""
Tests for SmoothQuant helper functions migrated from pipeline/format_study.py.

Tests:
1. from_model_calibration() on 2-layer Linear model -- correct layer names
2. from_model_calibration() on Conv2d model -- correct channel axes
3. Each returned value is a SmoothQuantTransform instance
4. from_model_calibration() with custom alpha -- scale differs
4b. from_model_calibration() with single-tensor calib_data -- works correctly
5. fuse_smoothquant_weights() does not mutate original model
6. fuse_smoothquant_weights() -- weights are modified
7. Fused model forward pass matches original within fp32 tolerance
7b. Conv2d fused forward pass equivalence
8. fuse_smoothquant_weights() with layer_names filter
9. from_model_calibration() on empty model --> empty dict
10. fuse_smoothquant_weights() with empty sq_transforms
11. from_model_calibration() with eval_fn -- eval_fn is called
12. Integration: calibrate --> fuse --> forward pass equivalence
13-17. compute_smoothquant_scale unit tests (1D input, alpha bounds, shape, errors)
"""
import copy
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

# Import scheme first to resolve circular dependency between src.transform and
# src.scheme (scheme/op_config.py imports from transform/hadamard.py).
# This follows the same initialization pattern as existing tests.
import src.scheme.transform  # noqa: F401
from src.transform.smooth_quant import (
    SmoothQuantTransform,
    compute_smoothquant_scale,
    fuse_smoothquant_weights,
)


# ============================================================================
# Model definitions
# ============================================================================

class TwoLayerLinear(nn.Module):
    """Simple 2-layer Linear model for SmoothQuant tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class ConvModel(nn.Module):
    """Simple 2-layer Conv2d model for SmoothQuant tests."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class EmptyModel(nn.Module):
    """Model with no Linear or Conv2d layers."""

    def forward(self, x):
        return x


# ============================================================================
# 1-4. from_model_calibration tests
# ============================================================================

class TestFromModelCalibration:

    def test_linear_model_layer_names(self):
        """from_model_calibration on a 2-layer Linear model returns correct layer names."""
        model = TwoLayerLinear()
        calib_data = [torch.randn(4, 16)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        assert set(result.keys()) == {"fc1", "fc2"}, \
            f"Expected {{'fc1', 'fc2'}}, got {set(result.keys())}"
        assert len(result) == 2

    def test_conv_model_channel_axes(self):
        """from_model_calibration on Conv2d model returns transforms with channel_axis=1."""
        model = ConvModel()
        calib_data = [torch.randn(2, 3, 8, 8)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        assert set(result.keys()) == {"conv1", "conv2"}, \
            f"Expected {{'conv1', 'conv2'}}, got {set(result.keys())}"
        for name, t in result.items():
            assert t.channel_axis == 1, \
                f"{name}: expected channel_axis=1 for Conv2d, got {t.channel_axis}"

    def test_returns_smooth_quant_transform_instances(self):
        """Each returned value is a SmoothQuantTransform instance."""
        model = TwoLayerLinear()
        calib_data = [torch.randn(4, 16)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        for name, t in result.items():
            assert isinstance(t, SmoothQuantTransform), \
                f"{name}: expected SmoothQuantTransform, got {type(t)}"

    def test_custom_alpha_changes_scale(self):
        """from_model_calibration with alpha=0.7 produces different scales from alpha=0.5."""
        model = TwoLayerLinear()
        calib_data = [torch.randn(4, 16)]
        result_05 = SmoothQuantTransform.from_model_calibration(
            model, calib_data, alpha=0.5,
        )
        # Same model, same calib_data -- only alpha differs
        result_07 = SmoothQuantTransform.from_model_calibration(
            model, calib_data, alpha=0.7,
        )
        for name in result_05:
            assert name in result_07, f"Layer {name} missing from alpha=0.7 run"
            # Scales should differ for different alpha values
            assert not torch.equal(result_05[name].scale, result_07[name].scale), \
                f"Layer {name}: scales should differ for alpha=0.5 vs alpha=0.7"

    def test_single_tensor_calib_data(self):
        """from_model_calibration works with a raw Tensor (not wrapped in a list)."""
        model = TwoLayerLinear()
        calib_data = torch.randn(4, 16)  # raw tensor, not [tensor]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        assert set(result.keys()) == {"fc1", "fc2"}, \
            f"Expected {{'fc1', 'fc2'}}, got {set(result.keys())}"
        for t in result.values():
            assert isinstance(t, SmoothQuantTransform)


# ============================================================================
# 5-8. fuse_smoothquant_weights tests
# ============================================================================

class TestFuseSmoothQuantWeights:

    def test_does_not_mutate_original_model(self):
        """fuse_smoothquant_weights does not modify the original model."""
        model = TwoLayerLinear()
        calib_data = [torch.randn(4, 16)]
        original_state = copy.deepcopy(model.state_dict())
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        _ = fuse_smoothquant_weights(model, result)
        # Original model should be completely unchanged
        for key in original_state:
            assert torch.equal(model.state_dict()[key], original_state[key]), \
                f"{key}: original model was mutated"

    def test_fused_weights_modified(self):
        """fuse_smoothquant_weights modifies layer weights but not biases."""
        model = TwoLayerLinear()
        calib_data = [torch.randn(4, 16)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        fused = fuse_smoothquant_weights(model, result)
        # fc1.weight should be modified
        assert not torch.equal(fused.fc1.weight, model.fc1.weight), \
            "fc1.weight should be modified by SmoothQuant fusion"
        # fc2.weight should be modified
        assert not torch.equal(fused.fc2.weight, model.fc2.weight), \
            "fc2.weight should be modified by SmoothQuant fusion"
        # biases should NOT be modified
        assert torch.equal(fused.fc1.bias, model.fc1.bias), \
            "fc1.bias should not be modified"
        assert torch.equal(fused.fc2.bias, model.fc2.bias), \
            "fc2.bias should not be modified"

    def test_fused_forward_pass_equivalence(self):
        """Activation smoothing + fused weights produces same output as original.

        Mathematical identity: (x / s) @ (W * s) = x @ W.
        The test applies activation smoothing (x / s) before each layer
        of the fused (W * s) model and compares with the original.
        """
        model = TwoLayerLinear()
        model.eval()
        calib_data = [torch.randn(4, 16)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        fused = fuse_smoothquant_weights(model, result)
        fused.eval()

        with torch.no_grad():
            x = calib_data[0]
            out_orig = model(x)

            # Apply SmoothQuant pipeline: activation smoothing before each
            # layer, then forward through the fused-weight model.
            # Layer 1: (x / s1) @ (W1 * s1) = x @ W1
            h = result["fc1"].forward(x)
            h = fused.fc1(h)
            # Layer 2: (h / s2) @ (W2 * s2) = h @ W2
            h = result["fc2"].forward(h)
            out_fused = fused.fc2(h)

        assert torch.allclose(out_orig, out_fused, atol=1e-5), \
            f"Forward pass mismatch: max diff = {(out_orig - out_fused).abs().max().item()}"

    def test_fuse_with_layer_names_filter(self):
        """fuse_smoothquant_weights with layer_names filters only specified layers."""
        model = TwoLayerLinear()
        calib_data = [torch.randn(4, 16)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        fused = fuse_smoothquant_weights(model, result, layer_names=["fc1"])
        # fc1.weight should be modified
        assert not torch.equal(fused.fc1.weight, model.fc1.weight), \
            "fc1.weight should be modified when included in layer_names"
        # fc2.weight should NOT be modified
        assert torch.equal(fused.fc2.weight, model.fc2.weight), \
            "fc2.weight should not be modified when excluded from layer_names"

    def test_fused_forward_pass_equivalence_conv2d(self):
        """Conv2d: activation smoothing + fused weights produces same output as original.

        Mathematical identity for Conv2d: conv(x / s, W * s) = conv(x, W)
        where the per-channel scale s cancels element-wise along the input
        channel dimension inside the convolution.
        """
        model = ConvModel()
        model.eval()
        calib_data = [torch.randn(2, 3, 8, 8)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        fused = fuse_smoothquant_weights(model, result)
        fused.eval()

        with torch.no_grad():
            x = calib_data[0]
            out_orig = model(x)

            # Apply SmoothQuant pipeline: activation smoothing before each
            # conv layer, then forward through the fused-weight model.
            # Layer 1: conv(x / s1, W1 * s1) = conv(x, W1)
            h = result["conv1"].forward(x)
            h = fused.conv1(h)
            # Layer 2: conv(h / s2, W2 * s2) = conv(h, W2)
            h = result["conv2"].forward(h)
            out_fused = fused.conv2(h)

        assert torch.allclose(out_orig, out_fused, atol=1e-5), \
            f"Conv2d forward pass mismatch: max diff = " \
            f"{(out_orig - out_fused).abs().max().item()}"


# ============================================================================
# 9-10. Edge case tests
# ============================================================================

class TestSmoothQuantHelpersEdgeCases:

    def test_empty_model_returns_empty_dict(self):
        """from_model_calibration on model with no Linear/Conv2d returns empty dict."""
        model = EmptyModel()
        calib_data = [torch.randn(4, 16)]
        result = SmoothQuantTransform.from_model_calibration(model, calib_data)
        assert result == {}, f"Expected empty dict, got {result}"

    def test_fuse_empty_transforms_returns_deep_copy(self):
        """fuse_smoothquant_weights with empty sq_transforms returns unchanged deep copy."""
        model = TwoLayerLinear()
        original_state = copy.deepcopy(model.state_dict())
        fused = fuse_smoothquant_weights(model, {})
        assert id(fused) != id(model), \
            "Result should be a deep copy (different object identity)"
        for key in original_state:
            assert torch.equal(fused.state_dict()[key], original_state[key]), \
                f"{key}: weights should be unchanged"


# ============================================================================
# 11. eval_fn mock test
# ============================================================================

class TestEvalFn:

    def test_eval_fn_is_called(self):
        """from_model_calibration with eval_fn calls eval_fn exactly once."""
        model = TwoLayerLinear()
        calib_data = [torch.randn(4, 16)]
        mock_fn = Mock()
        result = SmoothQuantTransform.from_model_calibration(
            model, calib_data, eval_fn=mock_fn,
        )
        mock_fn.assert_called_once()
        # Verify arguments
        mock_fn.assert_called_once_with(model, calib_data)


# ============================================================================
# 12. Integration test
# ============================================================================

class TestIntegration:

    def test_calibrate_fuse_forward_equivalence(self):
        """Full integration: from_model_calibration -> fuse_smoothquant_weights
        produces a model whose forward output matches the original."""
        model = TwoLayerLinear()
        model.eval()
        calib_data = [torch.randn(4, 16)]

        # Step 1: Calibrate
        sq_transforms = SmoothQuantTransform.from_model_calibration(
            model, calib_data, alpha=0.5,
        )
        assert set(sq_transforms.keys()) == {"fc1", "fc2"}
        for t in sq_transforms.values():
            assert isinstance(t, SmoothQuantTransform)

        # Step 2: Fuse weights
        fused = fuse_smoothquant_weights(model, sq_transforms)
        fused.eval()
        assert id(fused) != id(model)

        # Step 3: Verify forward pass equivalence
        # Mathematical identity: (x / s) @ (W * s) = x @ W
        with torch.no_grad():
            x = calib_data[0]
            out_orig = model(x)

            # Apply activation smoothing before each layer, then forward
            # through fused-weight model.
            h = sq_transforms["fc1"].forward(x)
            h = fused.fc1(h)
            h = sq_transforms["fc2"].forward(h)
            out_fused = fused.fc2(h)

        assert torch.allclose(out_orig, out_fused, atol=1e-5), \
            f"Integration: forward pass mismatch, max diff = " \
            f"{(out_orig - out_fused).abs().max().item()}"


# ============================================================================
# 13-17. compute_smoothquant_scale unit tests
# ============================================================================

class TestComputeSmoothQuantScale:
    """Direct unit tests for compute_smoothquant_scale()."""

    def test_1d_input_path(self):
        """Pre-computed per-channel statistics passed as 1D tensor."""
        X_act = torch.tensor([2.0, 4.0, 6.0])  # already per-channel max
        W = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        scale = compute_smoothquant_scale(X_act, W, alpha=0.5)
        # act_amax = [2, 4, 6], w_amax = [4, 5, 6]
        # s_j = max(|X_j|)^0.5 / max(|W_j|)^0.5
        expected = torch.tensor([
            (2.0 / 4.0) ** 0.5,
            (4.0 / 5.0) ** 0.5,
            (6.0 / 6.0) ** 0.5,
        ])
        assert scale.shape == (3,), f"Expected shape (3,), got {scale.shape}"
        assert torch.allclose(scale, expected)

    def test_alpha_zero_boundary(self):
        """alpha=0: scale = 1 / max(|W|) per channel."""
        X_act = torch.randn(4, 3)  # last dim=3 matches W's IC dim
        W = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        scale = compute_smoothquant_scale(X_act, W, alpha=0.0)
        # w_amax = [4, 5, 6]
        # s = max(|X|)^0 / max(|W|)^1 = 1 / [4, 5, 6]
        expected = torch.tensor([1.0/4.0, 1.0/5.0, 1.0/6.0])
        assert torch.allclose(scale, expected)

    def test_alpha_one_boundary(self):
        """alpha=1: scale = max(|X|) per channel."""
        X_act = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        W = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # value ignored
        scale = compute_smoothquant_scale(X_act, W, alpha=1.0)
        # act_amax = [1, 2, 3]
        # s = max(|X|)^1 / max(|W|)^0 = [1, 2, 3]
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(scale, expected)

    def test_alpha_out_of_range_raises(self):
        """alpha outside [0, 1] raises ValueError."""
        X_act = torch.randn(3, 8)
        W = torch.randn(5, 8)
        with pytest.raises(ValueError, match="alpha must be in"):
            compute_smoothquant_scale(X_act, W, alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            compute_smoothquant_scale(X_act, W, alpha=-0.5)

    def test_output_shape_is_1d(self):
        """Output is a 1D tensor with C elements (C = number of channels)."""
        X_act = torch.randn(4, 16)
        W = torch.randn(32, 16)  # OC=32, IC=16
        scale = compute_smoothquant_scale(X_act, W)
        assert isinstance(scale, torch.Tensor), \
            f"Expected Tensor, got {type(scale)}"
        assert scale.ndim == 1, \
            f"Expected 1D tensor, got {scale.ndim}D"
        assert scale.shape[0] == 16, \
            f"Expected 16 channels, got {scale.shape[0]}"
