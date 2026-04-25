import torch
from src.analysis.observer import SliceAwareObserver


class DistributionObserver(SliceAwareObserver):
    """Per-slice fp32 statistical fingerprint for distribution taxonomy."""

    def __init__(self, sparse_eps: float = 1e-8, outlier_sigma: float = 3.0,
                 hist_bins: int = 64):
        super().__init__()
        self.sparse_eps = sparse_eps
        self.outlier_sigma = outlier_sigma
        self.hist_bins = hist_bins

    def _measure(self, key, fp32, quant):
        f = fp32
        f_abs = f.abs()
        n = f.numel()
        non_zero_mask = f_abs > self.sparse_eps
        min_nonzero = f_abs[non_zero_mask].min().item() if non_zero_mask.any() else self.sparse_eps

        # Central moments
        mean = f.mean()
        delta = f - mean
        var = delta.pow(2).mean()
        std = var.sqrt()
        m3 = delta.pow(3).mean()
        m4 = delta.pow(4).mean()
        skew = (m3 / (var * std + 1e-30)).item()
        kurt = (m4 / (var.pow(2) + 1e-30)).item()
        excess_kurt = kurt - 3.0

        # Sarle's bimodality coefficient
        bc_denom = excess_kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3) + 1e-30)
        bimodality = (skew**2 + 1) / (bc_denom + 1e-30)

        # Normalized Shannon entropy from histogram
        hist = torch.histc(f, bins=self.hist_bins)
        probs = hist.float() / (n + 1e-30)
        probs_pos = probs[probs > 0]
        entropy_raw = -(probs_pos * torch.log2(probs_pos + 1e-30)).sum().item()
        max_entropy = torch.log2(torch.tensor(self.hist_bins, dtype=torch.float32)).item()
        norm_entropy = entropy_raw / (max_entropy + 1e-30)

        return {
            "min": f.min().item(),
            "max": f.max().item(),
            "mean": mean.item(),
            "std": std.item(),
            "skewness": skew,
            "kurtosis": kurt,
            "excess_kurtosis": excess_kurt,
            "bimodality_coefficient": bimodality,
            "sparse_ratio": (f_abs < self.sparse_eps).float().mean().item(),
            "dynamic_range_bits": (torch.log2(f_abs.max() / min_nonzero)).item() if non_zero_mask.any() else 0.0,
            "outlier_ratio": (f_abs > self.outlier_sigma * std).float().mean().item(),
            "norm_entropy": norm_entropy,
        }


class QSNRObserver(SliceAwareObserver):
    """QSNR = 10 * log10(||fp32||^2 / ||fp32 - quant||^2), unit dB."""

    def _measure(self, key, fp32, quant):
        err = fp32 - quant
        num = fp32.pow(2).mean()
        den = err.pow(2).mean().clamp_min(1e-30)
        return {"qsnr_db": (10 * torch.log10(num / den)).item()}


class MSEObserver(SliceAwareObserver):
    """Mean squared error per slice."""

    def _measure(self, key, fp32, quant):
        return {"mse": (fp32 - quant).pow(2).mean().item()}
