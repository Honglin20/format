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


class TestActivationChannelImportancePaperFormula:
    """Verify I_j = |A_j * Σ_i W_{j,i}| — abs outside the sum."""

    def test_mixed_sign_weights_sum_differently(self):
        """|sum(W)| != sum(|W|) when weights have mixed signs."""
        act_avg = torch.ones(3)
        # Row 0: large positive and negative weights cancel → low |sum(W)|
        # Row 1: all positive → sum(W) = sum(|W|)
        # Row 2: all positive, same magnitude as row 1
        w = torch.tensor([
            [5.0, -5.0, 0.0, 0.0],  # sum=0, abs_sum=10 → paper: low, old: high
            [1.0, 1.0, 1.0, 1.0],   # sum=4, abs_sum=4
            [2.0, 2.0, 0.0, 0.0],   # sum=4, abs_sum=4
        ])
        imp = compute_activation_channel_importance(act_avg, w)
        # Paper formula: I_0 = |1 * 0| = 0, I_1 = |1 * 4| = 4, I_2 = |1 * 4| = 4
        # Channel 0 should be LEAST important (cancelling weights)
        assert imp[0] < imp[1]
        assert imp[0] < imp[2]
        # Channel 0 should be near-zero importance
        assert imp[0] < 0.01

    def test_all_positive_same_as_before(self):
        """When all weights positive, |sum(W)| = sum(|W|)."""
        act_avg = torch.ones(3)
        w = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        imp = compute_activation_channel_importance(act_avg, w)
        # With all-positive weights, result should match old formula
        expected = act_avg * torch.sum(torch.abs(w), dim=-1)
        assert torch.allclose(imp, expected)
