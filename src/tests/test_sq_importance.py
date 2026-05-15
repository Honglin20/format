# src/tests/test_sq_importance.py
import torch
import pytest
from src.formats._sq_importance import (
    compute_hessian_importance,
    compute_hessian_diag_from_inputs,
    compute_activation_channel_importance,
)


class TestHessianImportance:
    def test_shape_preserved(self):
        w = torch.randn(8, 16)
        h = torch.ones(16)
        imp = compute_hessian_importance(w, h)
        assert imp.shape == w.shape

    def test_hessian_scaling(self):
        """Weights in high-Hessian directions get higher importance."""
        w = torch.tensor([[1.0, 1.0], [3.0, 3.0]])  # same weight per row so ratio isolates H-scaling
        h = torch.tensor([10.0, 1.0])  # dim 0 is 100x more important
        imp = compute_hessian_importance(w, h)
        # Column 0 should have 100x (10²/1²) higher importance than column 1
        ratio = imp[:, 0] / imp[:, 1]
        expected_ratio = (h[0] ** 2) / (h[1] ** 2)  # = 100
        assert torch.allclose(ratio, torch.full_like(ratio, expected_ratio))

    def test_diag_from_inputs(self):
        inputs = [torch.ones(4, 8), torch.full((4, 8), 2.0)]
        h = compute_hessian_diag_from_inputs(inputs)
        # mean(x²): for ones: 1, for twos: 4. Average over 2 samples = 2.5
        expected = torch.full((8,), 2.5)
        assert torch.allclose(h, expected)


class TestActivationChannelImportance:
    def test_shape(self):
        act_avg = torch.ones(4)
        w = torch.randn(4, 8)
        imp = compute_activation_channel_importance(act_avg, w)
        assert imp.shape == (4,)

    def test_zero_activation_zero_importance(self):
        act_avg = torch.tensor([0.0, 1.0, 2.0])
        w = torch.ones(3, 5)
        imp = compute_activation_channel_importance(act_avg, w)
        assert imp[0] == 0.0
        assert imp[1] > 0.0
        assert imp[2] > imp[1]  # larger activation → more important

    def test_zero_weight_zero_importance(self):
        act_avg = torch.ones(3)
        w = torch.zeros(3, 5)
        imp = compute_activation_channel_importance(act_avg, w)
        assert torch.all(imp == 0.0)
