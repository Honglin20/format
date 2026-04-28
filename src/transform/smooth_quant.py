"""
SmoothQuantTransform: pre-quantization activation smoothing via per-channel scaling.

SmoothQuant (Xiao et al., 2023) smooths activation outliers by dividing each
channel of the activation by a per-channel scale factor, then compensating by
multiplying the corresponding input channel of the weight by the same factor.
This migrates quantization difficulty from activations to weights.

This module implements the ACTIVATION side: forward(x) = x / scale.
The weight side (W * s) is applied separately by the caller.

Design: immutable scale set at construction time. The ``from_calibration()``
factory computes scale from activation statistics and weight tensor.
"""
import torch
from torch import Tensor

from ..scheme.transform import TransformBase


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------

def compute_smoothquant_scale(
    X_act: Tensor,
    W: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """Compute per-channel SmoothQuant smoothing factor.

    The scale for each channel j is::

        s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)

    where ``X_j`` refers to the j-th channel of the activation and ``W_j``
    refers to the j-th input channel of the weight.

    Args:
        X_act: Per-channel max absolute activation values, or raw activation
               tensor. If 1D, treated as per-channel statistics directly. If
               2D+, the channel dimension is the last dim (dim -1); max is
               computed along all other dims.
        W: Weight tensor. The input channel dimension is dim 1 (for both
           Linear [OC, IC] and Conv [OC, IC, H, W]). Per-channel max is
           computed along all dims except dim 1.
        alpha: Smoothing strength. 0 = all weight, 1 = all activation.
               Default 0.5.

    Returns:
        Scale tensor of shape ``[C]`` where ``C`` is the input channel count
        (matching both the activation's last dim and the weight's dim 1).

    Raises:
        ValueError: If ``alpha`` is outside [0, 1].
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(
            f"alpha must be in [0, 1], got {alpha}"
        )

    # --- Activation per-channel max ---
    if X_act.ndim == 1:
        # Pre-computed per-channel statistics
        act_amax = X_act.abs()
    else:
        # Reduce all dims except the last (channel dim)
        act_amax = torch.amax(
            torch.abs(X_act), dim=tuple(range(X_act.ndim - 1))
        )

    # --- Weight per-input-channel max ---
    # Input channel dimension is dim 1 for both Linear and Conv weights.
    # Reduce all dims except dim 1: dim 0 (output channels) and dims 2+
    # (spatial dims for Conv).
    w_reduce_dims = tuple(d for d in range(W.ndim) if d != 1)
    w_amax = torch.amax(torch.abs(W), dim=w_reduce_dims)

    # --- SmoothQuant scale ---
    s = (
        act_amax.clamp(min=1e-12).pow(alpha)
        / w_amax.clamp(min=1e-12).pow(1.0 - alpha)
    )
    return s


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------

class SmoothQuantTransform(TransformBase):
    """Pre-quantization SmoothQuant activation smoothing (immutable scale).

    Applies per-channel scaling to activations before quantization::

        forward(x) = x / scale
        inverse(x_q) = x_q * scale

    The scale is immutable after construction. Use :meth:`from_calibration`
    to create from activation statistics and a weight tensor; this avoids
    the "uncalibrated" illegal state that a mutable design would allow.

    The activation's channel dimension is always the **last** dim (-1).
    """

    invertible = True

    def __init__(self, scale: Tensor):
        """Create SmoothQuantTransform with a pre-computed per-channel scale.

        Args:
            scale: 1D tensor of per-channel smoothing factors. Must be
                   strictly positive (values <= 0 will produce NaNs in
                   forward/inverse).

        The scale is cloned internally to prevent external mutation.
        Effectively immutable: no public setter is provided.
        """
        object.__setattr__(self, "_scale", scale.detach().clone())

    # Convenience read-only access — no setter
    @property
    def scale(self) -> Tensor:
        """The per-channel smoothing factor (read-only)."""
        return self._scale

    def _broadcast_scale(self, x: Tensor) -> Tensor:
        """Reshape ``self._scale`` to broadcast against ``x``.

        ``x`` has channel dim at the last position: ``[..., C]``.
        ``scale`` is ``[C]``, so the view is ``[1, ..., 1, C]``.
        """
        shape = [1] * (x.ndim - 1) + [-1]
        return self._scale.view(*shape)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SmoothQuant: ``x / scale``.

        Args:
            x: Input activation tensor. The channel dimension must be the
               last dim: ``[..., C]``.

        Returns:
            Smoothed activation (same shape as ``x``).
        """
        return x / self._broadcast_scale(x)

    def inverse(self, x_q: Tensor) -> Tensor:
        """Reverse SmoothQuant: ``x_q * scale``.

        Args:
            x_q: Quantized (or post-quantization) tensor with channel dim
                 as the last dim.

        Returns:
            Scale-restored tensor (same shape as ``x_q``).
        """
        return x_q * self._broadcast_scale(x_q)

    @staticmethod
    def from_calibration(
        X_act: Tensor, W: Tensor, alpha: float = 0.5
    ) -> "SmoothQuantTransform":
        """Factory: compute scale from activation statistics and weight.

        Convenience wrapper around :func:`compute_smoothquant_scale` that
        returns a fully-formed transform.

        Args:
            X_act: Activation statistics or raw activation tensor. See
                   :func:`compute_smoothquant_scale` for details.
            W: Weight tensor.
            alpha: Smoothing strength (default 0.5).

        Returns:
            A ``SmoothQuantTransform`` with the computed scale.
        """
        scale = compute_smoothquant_scale(X_act, W, alpha)
        return SmoothQuantTransform(scale)

    def __eq__(self, other) -> bool:
        """Two transforms are equal iff they have the same scale tensor."""
        if not isinstance(other, SmoothQuantTransform):
            return False
        return torch.equal(self._scale, other._scale)

    def __hash__(self) -> int:
        """Hash based on the scale tensor values."""
        return hash(tuple(self._scale.flatten().tolist()))
