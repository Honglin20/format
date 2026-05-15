"""SQ-format importance scoring functions.

ADR-014: Hessian-based importance for weight masks (Algorithm 1)
and A·W product importance for activation channel masks (Algorithm 2).
"""
import torch


def compute_hessian_importance(weight: torch.Tensor, hessian_diag: torch.Tensor) -> torch.Tensor:
    """I_{r,i} = W_{r,i}² · H_i² — per-element weight importance.

    Higher values mean the weight is more sensitive to quantization error.
    """
    return (weight ** 2) * (hessian_diag ** 2)


def compute_hessian_diag_from_inputs(inputs: list) -> torch.Tensor:
    """Approximate Hessian diagonal from calibration inputs.

    For a linear layer y = x @ W^T, the Hessian diagonal w.r.t. W
    is H_i ≈ mean(x²)_i over the calibration set.

    Args:
        inputs: list of activation tensors, each shape (..., N)
    Returns:
        hessian_diag shape (N,) — mean squared input per feature
    """
    all_inputs = []
    for x in inputs:
        # Flatten all dims except the last (feature dim)
        x_flat = x.reshape(-1, x.shape[-1])
        all_inputs.append(x_flat)
    stacked = torch.cat(all_inputs, dim=0)
    return (stacked ** 2).mean(dim=0)


def compute_activation_channel_importance(
    act_avg: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """I_j = |Ā_j · Σ_i |W_{j,i}|| — per-channel activation importance.

    Measures the contribution of input channel j to the dot product A · W.

    Args:
        act_avg: per-channel average activation, shape (K,)
        weight: weight matrix, shape (K, N) where K=input channels
    Returns:
        per-channel importance scores, shape (K,)
    """
    weight_sum = torch.sum(torch.abs(weight), dim=-1)  # shape (K,)
    return torch.abs(act_avg.to(weight.device)) * weight_sum
