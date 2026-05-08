from typing import List

import torch

from src.observer import SliceAwareObserver


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

        rms = f.pow(2).mean().sqrt()
        peak = f_abs.max()

        return {
            "min": f.min().item(),
            "max": f.max().item(),
            "mean": mean.item(),
            "std": std.item(),
            "peak": peak.item(),
            "rms": rms.item(),
            "crest_factor": (peak / (rms + 1e-30)).item(),
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

    def _measure_per_unit(self, fp32_2d, quant_2d):
        """Vectorized: one kernel per metric over [N, D]."""
        err = fp32_2d - quant_2d
        num = fp32_2d.pow(2).mean(dim=1)
        den = err.pow(2).mean(dim=1).clamp_min(1e-30)
        qsnr = 10 * torch.log10(num / den)
        return [{"qsnr_db": v} for v in qsnr.tolist()]

    def _measure_batch(self, fp32_2d, quant_2d, valid_counts=None):
        """Vectorized per-block aggregate: mean/std/min/max of per-block QSNR.

        QSNR ratio is invariant to the divisor (sum(f²)/k / sum(err²)/k =
        sum(f²)/sum(err²)), so .mean(dim=1) is correct even for partial blocks.
        """
        err = fp32_2d - quant_2d
        num = fp32_2d.pow(2).mean(dim=1)
        den = err.pow(2).mean(dim=1).clamp_min(1e-30)
        qsnr = 10 * torch.log10(num / den)
        return {
            "qsnr_db": qsnr.mean().item(),
            "qsnr_db_std": qsnr.std(unbiased=False).item() if qsnr.numel() > 1 else 0.0,
            "qsnr_db_min": qsnr.min().item(),
            "qsnr_db_max": qsnr.max().item(),
        }


class MSEObserver(SliceAwareObserver):
    """Mean squared error per slice."""

    def _measure(self, key, fp32, quant):
        return {"mse": (fp32 - quant).pow(2).mean().item()}

    def _measure_per_unit(self, fp32_2d, quant_2d):
        """Vectorized: one kernel over [N, D]."""
        mse = (fp32_2d - quant_2d).pow(2).mean(dim=1)
        return [{"mse": v} for v in mse.tolist()]

    def _measure_batch(self, fp32_2d, quant_2d, valid_counts=None):
        """Vectorized per-block aggregate: mean/std/min/max of per-block MSE.

        Uses valid_counts for correct partial-block measurement when
        dim_size % block_size != 0.
        """
        err_sq = (fp32_2d - quant_2d).pow(2)
        if valid_counts is not None:
            mse = err_sq.sum(dim=1) / valid_counts.clamp_min(1)
        else:
            mse = err_sq.mean(dim=1)
        return {
            "mse": mse.mean().item(),
            "mse_std": mse.std(unbiased=False).item() if mse.numel() > 1 else 0.0,
            "mse_min": mse.min().item(),
            "mse_max": mse.max().item(),
        }


class HistogramObserver(SliceAwareObserver):
    """fp32 / quant / error three-channel histogram."""

    def __init__(self, n_bins: int = 128):
        super().__init__()
        self.n_bins = n_bins

    def _measure(self, key, fp32, quant):
        return {
            "fp32_hist": torch.histc(fp32, bins=self.n_bins).cpu(),
            "quant_hist": torch.histc(quant, bins=self.n_bins).cpu(),
            "err_hist": torch.histc(fp32 - quant, bins=self.n_bins).cpu(),
        }
