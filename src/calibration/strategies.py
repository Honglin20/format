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

    Subclasses must implement ``compute(x, axis)``.
    Strategies are stateless — ``compute`` is a pure function of the input.
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


class MaxScaleStrategy(ScaleStrategy):
    """Absmax scale — current default behavior.

    Returns ``max(|x|)`` along ``axis``, clamped to a minimum of 1e-12.
    This is bit-exact with ``FormatBase._quantize_per_channel()``.
    """

    def compute(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        amax = torch.amax(torch.abs(x), dim=axis, keepdim=True)
        return amax.clamp(min=1e-12)


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
        amax = torch.amax(torch.abs(x), dim=axis, keepdim=True).clamp(min=1e-12)

        best_scale = amax.clone()
        best_kl = torch.full_like(amax, float("inf"))

        # Pre-compute histograms of |x| per slice
        x_abs = torch.abs(x)

        candidates = torch.linspace(0.5, 2.0, self.n_steps, device=x.device)

        for factor in candidates:
            scale = amax * factor
            x_q = _simple_quantize(x, scale)
            x_q_abs = torch.abs(x_q)

            # Compute KL divergence per slice
            # Strategy: bin both |x| and |x_q| into the same bins, then
            # compute KL(orig || quantized)
            kl_per_slice = _compute_kl_divergence(x_abs, x_q_abs, axis, self.n_bins)
            mask = kl_per_slice < best_kl
            best_kl = torch.where(mask, kl_per_slice, best_kl)
            best_scale = torch.where(mask, scale, best_scale)

        return best_scale


def _compute_kl_divergence(
    p_vals: torch.Tensor,
    q_vals: torch.Tensor,
    axis: int,
    n_bins: int,
) -> torch.Tensor:
    """Compute KL(P || Q) per slice along ``axis``.

    Both ``p_vals`` and ``q_vals`` are positive-valued tensors
    (absolute values). The function bins each slice into ``n_bins``
    equal-width bins in [0, 1] (after normalizing by per-slice max)
    and returns KL divergence per slice as a tensor with size 1
    along ``axis``.
    """
    # Normalize both by per-slice max of p so bins are consistent per slice
    max_vals = torch.amax(p_vals, dim=axis, keepdim=True).clamp(min=1e-12)
    p_norm = p_vals / max_vals  # [0, 1] per slice
    q_norm = q_vals / max_vals  # [0, 1] per slice, using p's max for fair comparison

    n_dims = p_vals.dim()
    axis_pos = axis if axis >= 0 else n_dims + axis

    # Permute so the quantized axis is last, then flatten the rest
    perm = list(range(n_dims))
    perm.remove(axis_pos)
    perm.append(axis_pos)

    p_flat = p_norm.permute(*perm).reshape(-1, p_norm.shape[axis_pos])
    q_flat = q_norm.permute(*perm).reshape(-1, q_norm.shape[axis_pos])

    n_slices = max_vals.numel()
    kl_list = []

    for i in range(n_slices):
        # Histogram of p (original) for this slice
        hist_p = torch.histc(p_flat[i], bins=n_bins, min=0.0, max=1.0) + 1e-12
        hist_p = hist_p / hist_p.sum()

        # Histogram of q (quantized) for this slice
        hist_q = torch.histc(q_flat[i], bins=n_bins, min=0.0, max=1.0) + 1e-12
        hist_q = hist_q / hist_q.sum()

        # KL(P || Q)
        kl = (hist_p * (hist_p / hist_q).log()).sum()
        kl_list.append(kl)

    # Reshape back to match max_vals (amax) shape
    kl_tensor = torch.tensor(kl_list, device=p_vals.device, dtype=p_vals.dtype)
    kl_tensor = kl_tensor.reshape(max_vals.shape)
    return kl_tensor
