"""
Pluggable scale strategies for quantization scale computation.

Each strategy implements ``compute(x, axis) -> Tensor`` that returns
a scale tensor suitable for normalizing ``x`` before element-wise
quantization.

Usage::

    strategy = MaxScaleStrategy()
    scale = strategy.compute(x, axis=1)
    x_norm = x / scale          # normalize to [-1, 1]
    x_q = quantize_elemwise(x_norm) * scale  # quantize + rescale
"""
from abc import ABC, abstractmethod

import torch


def _simple_quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Simple int8-like quantize/dequantize for scale search.

    Normalizes ``x`` by ``scale``, rounds to 127 discrete levels (int8-like),
    then rescales. Used internally by MSE and KL strategies to evaluate
    candidate scales without depending on a specific format.
    """
    x_norm = x / scale
    x_q = torch.round(x_norm * 127.0)
    x_q = x_q.clamp(-127.0, 127.0)
    x_q = x_q / 127.0 * scale
    return x_q


class ScaleStrategy(ABC):
    """Abstract base for scale computation strategies.

    Subclasses must implement ``compute(x, axis)``, ``__eq__``, and ``__hash__``.
    Strategies are stateless — ``compute`` is a pure function of the input.

    ``__eq__`` and ``__hash__`` are required because strategy instances may
    be stored as fields of frozen dataclasses (e.g. calibration config),
    where value-based equality is required to prevent id-based hash bugs.
    """

    @abstractmethod
    def compute(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        """Compute scale factors for quantization along the given axis.

        Args:
            x: Input tensor (float).
            axis: Dimension along which to compute per-slice scales.
                  Supports NumPy-style negative indexing.

        Returns:
            Scale tensor with the same number of dimensions as ``x``,
            with size 1 along ``axis`` (ready for broadcasting back
            into ``x / scale`` or similar).
        """
        ...

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Value-based equality (required for frozen-dataclass field use)."""
        ...

    @abstractmethod
    def __hash__(self) -> int:
        """Hash based on strategy parameters (required for frozen-dataclass field use)."""
        ...


class MaxScaleStrategy(ScaleStrategy):
    """Absmax scale — current default behavior.

    Returns ``max(|x|)`` along ``axis``, clamped to a minimum of 1e-12.
    This is bit-exact with ``FormatBase._quantize_per_channel()``.
    """

    def compute(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        amax = torch.amax(torch.abs(x), dim=axis, keepdim=True)
        return amax.clamp(min=1e-12)

    def __eq__(self, other):
        return isinstance(other, MaxScaleStrategy)

    def __hash__(self):
        return hash("MaxScaleStrategy")


class PercentileScaleStrategy(ScaleStrategy):
    """N-th percentile of ``|x|`` as scale.

    Excludes outliers above the given percentile. For example,
    ``q=99`` uses the 99th percentile, ignoring the top 1% of values.

    Args:
        q: Percentile in range [0, 100].  ``q=100`` is equivalent to
           ``MaxScaleStrategy``; ``q=0`` uses the minimum absolute value.
    """

    def __init__(self, q: float = 99.0):
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"Percentile q must be in [0, 100], got {q}")
        self.q = q

    def compute(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        # torch.quantile expects q in [0, 1]
        p = torch.quantile(torch.abs(x), self.q / 100.0, dim=axis, keepdim=True)
        return p.clamp(min=1e-12)

    def __eq__(self, other):
        if not isinstance(other, PercentileScaleStrategy):
            return NotImplemented
        return self.q == other.q

    def __hash__(self):
        return hash(("PercentileScaleStrategy", self.q))


class MSEScaleStrategy(ScaleStrategy):
    """Grid-search for the scale that minimizes MSE.

    Searches ``n_steps`` candidates between ``0.5 * amax`` and
    ``2.0 * amax`` (where ``amax`` is the per-slice absmax), and
    returns the candidate with the lowest mean-squared error after
    quantize-dequantize.

    Args:
        n_steps: Number of candidate scale factors to evaluate.
    """

    def __init__(self, n_steps: int = 20):
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")
        self.n_steps = n_steps

    def compute(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        amax = torch.amax(torch.abs(x), dim=axis, keepdim=True).clamp(min=1e-12)

        best_scale = amax.clone()
        best_mse = torch.full_like(amax, float("inf"))

        candidates = torch.linspace(0.5, 2.0, self.n_steps, device=x.device)

        for factor in candidates:
            scale = amax * factor
            x_q = _simple_quantize(x, scale)
            # MSE per slice along axis
            mse = (x - x_q).pow(2).mean(dim=axis, keepdim=True)
            # Update best where this candidate is better
            mask = mse < best_mse
            best_mse = torch.where(mask, mse, best_mse)
            best_scale = torch.where(mask, scale, best_scale)

        return best_scale

    def __eq__(self, other):
        if not isinstance(other, MSEScaleStrategy):
            return NotImplemented
        return self.n_steps == other.n_steps

    def __hash__(self):
        return hash(("MSEScaleStrategy", self.n_steps))


class KLScaleStrategy(ScaleStrategy):
    """Grid-search for the scale that minimizes KL divergence.

    Computes histograms of ``|x|`` and the quantized-dequantized
    reconstruction, then searches ``n_steps`` candidates to minimize
    KL divergence between the two distributions.

    Args:
        n_bins: Number of bins for the histogram.
        n_steps: Number of candidate scale factors to evaluate.
    """

    def __init__(self, n_bins: int = 256, n_steps: int = 20):
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")
        self.n_bins = n_bins
        self.n_steps = n_steps

    def compute(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        # Per-slice KL uses one scale per slice (not per-position).
        # Reshape so the axis is first, flatten the rest, and compute
        # per-slice amax over all values in each slice.
        n_dims = x.dim()
        axis_pos = axis if axis >= 0 else n_dims + axis
        n_slices = x.shape[axis_pos]

        perm = [axis_pos] + [i for i in range(n_dims) if i != axis_pos]
        x_perm = x.permute(*perm).reshape(n_slices, -1)
        amax_per_slice = torch.amax(torch.abs(x_perm), dim=1).clamp(min=1e-12)

        # Reshape to broadcastable shape: axis dim = n_slices, others = 1
        broadcast_shape = [1] * n_dims
        broadcast_shape[axis_pos] = n_slices
        amax = amax_per_slice.reshape(broadcast_shape)

        best_scale = amax.clone()
        best_kl = torch.full_like(amax, float("inf"))

        x_abs = torch.abs(x)

        candidates = torch.linspace(0.5, 2.0, self.n_steps, device=x.device)

        for factor in candidates:
            scale = amax * factor
            x_q = _simple_quantize(x, scale)
            x_q_abs = torch.abs(x_q)

            kl_per_slice = _compute_kl_divergence(x_abs, x_q_abs, axis, self.n_bins)
            mask = kl_per_slice < best_kl
            best_kl = torch.where(mask, kl_per_slice, best_kl)
            best_scale = torch.where(mask, scale, best_scale)

        return best_scale

    def __eq__(self, other):
        if not isinstance(other, KLScaleStrategy):
            return NotImplemented
        return self.n_bins == other.n_bins and self.n_steps == other.n_steps

    def __hash__(self):
        return hash(("KLScaleStrategy", self.n_bins, self.n_steps))


def _compute_kl_divergence(
    p_vals: torch.Tensor,
    q_vals: torch.Tensor,
    axis: int,
    n_bins: int,
) -> torch.Tensor:
    """Compute KL(P || Q) per slice along ``axis``.

    Each slice along ``axis`` gets its own histogram (pooling all positions
    within that slice), and KL divergence is computed per slice.  This
    matches the standard TensorRT calibration approach: one histogram per
    channel, not one histogram per spatial position.

    Returns:
        KL tensor with same number of dimensions as input, with size
        ``n_slices`` along ``axis`` and 1 elsewhere.
    """
    n_dims = p_vals.dim()
    axis_pos = axis if axis >= 0 else n_dims + axis
    n_slices = p_vals.shape[axis_pos]

    # Permute so axis is first, flatten remaining dims → (n_slices, -1)
    perm = [axis_pos] + [i for i in range(n_dims) if i != axis_pos]
    p_perm = p_vals.permute(*perm)
    q_perm = q_vals.permute(*perm)

    # Per-slice max for normalization (max over all values in each slice)
    p_max = torch.amax(p_perm.reshape(n_slices, -1), dim=1, keepdim=True).clamp(min=1e-12)

    p_flat = p_perm.reshape(n_slices, -1) / p_max  # (n_slices, M), M = product of other dims
    q_flat = q_perm.reshape(n_slices, -1) / p_max

    kl_list = []
    for i in range(n_slices):
        hist_p = torch.histc(p_flat[i], bins=n_bins, min=0.0, max=1.0) + 1e-12
        hist_p = hist_p / hist_p.sum()
        hist_q = torch.histc(q_flat[i], bins=n_bins, min=0.0, max=1.0) + 1e-12
        hist_q = hist_q / hist_q.sum()
        kl = (hist_p * (hist_p / hist_q).log()).sum()
        kl_list.append(kl)

    # Reshape to (n_slices, 1, ..., 1) with axis dim = n_slices
    out_shape = [1] * n_dims
    out_shape[axis_pos] = n_slices
    kl_tensor = torch.tensor(kl_list, device=p_vals.device, dtype=p_vals.dtype)
    return kl_tensor.reshape(out_shape)
