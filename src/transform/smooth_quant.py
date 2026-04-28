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

    The activation's channel dimension is configurable via ``channel_axis``
    (default ``-1``, the last dim). For ``nn.Linear`` activations with shape
    ``(N, C)`` or ``(B, S, C)`` this is ``-1``. For ``nn.Conv2d`` activations
    with shape ``(N, C, H, W)`` use ``channel_axis=1``.
    """

    invertible = True

    def __init__(self, scale: Tensor, channel_axis: int = -1):
        """Create SmoothQuantTransform with a pre-computed per-channel scale.

        Args:
            scale: 1D tensor of per-channel smoothing factors. Must be
                   strictly positive (values <= 0 will produce NaNs in
                   forward/inverse).
            channel_axis: The dimension along which channels are arranged in
                          the input tensor. Default ``-1`` (last dim), which
                          is correct for ``nn.Linear`` activations
                          ``(..., C)``. Use ``1`` for ``nn.Conv2d``
                          activations ``(N, C, H, W)``.

        The scale is cloned internally to prevent external mutation.
        Effectively immutable: no public setter is provided.
        """
        object.__setattr__(self, "_scale", scale.detach().clone())
        object.__setattr__(self, "_channel_axis", channel_axis)

    # Convenience read-only access — no setter
    @property
    def scale(self) -> Tensor:
        """The per-channel smoothing factor (read-only)."""
        return self._scale

    @property
    def channel_axis(self) -> int:
        """The channel axis for broadcasting (read-only)."""
        return self._channel_axis

    def _broadcast_scale(self, x: Tensor) -> Tensor:
        """Reshape ``self._scale`` to broadcast against ``x``.

        ``x`` has channel dim at ``self._channel_axis``.
        ``scale`` is ``[C]``, so the view is ``[1, ..., C, ..., 1]`` with
        ``C`` placed at the channel axis position.

        Raises:
            ValueError: If ``self._channel_axis`` is out of bounds for ``x``.
        """
        if not (-x.ndim <= self._channel_axis < x.ndim):
            raise ValueError(
                f"channel_axis={self._channel_axis} is out of bounds for "
                f"tensor with {x.ndim} dimensions"
            )
        shape = [1] * x.ndim
        shape[self._channel_axis] = -1
        return self._scale.view(*shape)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SmoothQuant: ``x / scale``.

        Args:
            x: Input activation tensor. The channel dimension is
               ``self.channel_axis``.

        Returns:
            Smoothed activation (same shape as ``x``).
        """
        return x / self._broadcast_scale(x)

    def inverse(self, x_q: Tensor) -> Tensor:
        """Reverse SmoothQuant: ``x_q * scale``.

        Args:
            x_q: Quantized (or post-quantization) tensor with channel dim
                 at ``self.channel_axis``.

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
        """Two transforms are equal iff they have the same scale and channel_axis."""
        if not isinstance(other, SmoothQuantTransform):
            return False
        return (
            self._channel_axis == other._channel_axis
            and torch.equal(self._scale, other._scale)
        )

    def __hash__(self) -> int:
        """Hash based on the scale tensor values and channel_axis."""
        return hash((self._channel_axis, tuple(self._scale.flatten().tolist())))


class SmoothQuantWeightTransform(TransformBase):
    """Weight-side SmoothQuant compensation: ``forward(W) = W * scale``.

    Companion to :class:`SmoothQuantTransform`.  While the activation side
    divides by the per-channel scale (``x / s``), the weight side multiplies
    by the same scale (``W * s``) to maintain mathematical equivalence::

        (X / s) @ (W * s) = X @ W

    The scale is the same tensor used in the activation-side transform,
    obtained via ``sq_transform.scale``.

    The weight's input-channel dimension defaults to **dim 1** (``[OC, IC]``
    for Linear, ``[OC, IC, H, W]`` for Conv).  The scale broadcasts as
    ``[1, IC, 1, ...]``.

    Args:
        scale: 1D tensor of per-channel compensation factors. Cloned internally.
        channel_axis: Dimension of ``W`` that is the input-channel axis.
               Default ``1`` (matches both Linear [OC, IC] and
               Conv [OC, IC, H, W] weight layouts).
    """

    invertible = True

    def __init__(self, scale: Tensor, channel_axis: int = 1):
        object.__setattr__(self, "_scale", scale.detach().clone())
        object.__setattr__(self, "_channel_axis", channel_axis)

    @property
    def scale(self) -> Tensor:
        """The per-channel compensation factor (read-only)."""
        return self._scale

    @property
    def channel_axis(self) -> int:
        """The weight input-channel dimension index."""
        return self._channel_axis

    def _broadcast_scale(self, W: Tensor) -> Tensor:
        """Reshape ``self._scale`` to broadcast against ``W``.

        Places scale at ``self._channel_axis`` and size-1 everywhere else.

        Raises:
            ValueError: If ``self._channel_axis`` is out of bounds for ``W``.
        """
        if not (-W.ndim <= self._channel_axis < W.ndim):
            raise ValueError(
                f"channel_axis={self._channel_axis} is out of bounds for "
                f"tensor with {W.ndim} dimensions"
            )
        shape = [1] * W.ndim
        shape[self._channel_axis] = -1
        return self._scale.view(*shape)

    def forward(self, W: Tensor) -> Tensor:
        return W * self._broadcast_scale(W)

    def inverse(self, W_q: Tensor) -> Tensor:
        return W_q / self._broadcast_scale(W_q)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SmoothQuantWeightTransform):
            return False
        return (self._channel_axis == other._channel_axis
                and torch.equal(self._scale, other._scale))

    def __hash__(self) -> int:
        return hash((self._channel_axis,
                     tuple(self._scale.flatten().tolist())))
