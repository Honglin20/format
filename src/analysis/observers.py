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

    # ------------------------------------------------------------------
    # Per-slice (PER_TENSOR)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Vectorized per-channel (the default Python loop was the #1 bottleneck)
    # ------------------------------------------------------------------

    def _measure_per_unit(self, fp32_2d, quant_2d):
        """Vectorized per-channel stats. Only histc loops per channel."""
        f = fp32_2d  # [N_ch, D]
        n = f.shape[1]
        f_abs = f.abs()
        eps = self.sparse_eps

        # -- all reduction-based stats, vectorized over dim=1 ----------
        f_min = f.min(dim=1).values
        f_max = f.max(dim=1).values
        mean = f.mean(dim=1)
        delta = f - mean.unsqueeze(1)
        var = delta.pow(2).mean(dim=1)
        std = var.sqrt()
        m3 = delta.pow(3).mean(dim=1)
        m4 = delta.pow(4).mean(dim=1)
        skew = m3 / (var * std + 1e-30)
        kurt = m4 / (var.pow(2) + 1e-30)
        excess_kurt = kurt - 3.0
        bc_denom = excess_kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3) + 1e-30)
        bimodality = (skew.pow(2) + 1) / (bc_denom + 1e-30)

        rms = f.pow(2).mean(dim=1).sqrt()
        peak = f_abs.max(dim=1).values
        crest_factor = peak / (rms + 1e-30)
        sparse_ratio = (f_abs < eps).float().mean(dim=1)

        non_zero_mask = f_abs > eps
        min_nonzero = torch.where(
            non_zero_mask, f_abs,
            torch.tensor(float('inf'), device=f.device, dtype=f.dtype)
        ).min(dim=1).values
        has_nonzero = non_zero_mask.any(dim=1)
        dynamic_range_bits = torch.where(
            has_nonzero,
            torch.log2(f_max / min_nonzero.clamp_min(eps)),
            torch.zeros_like(f_max)
        )
        outlier_ratio = (f_abs > self.outlier_sigma * std.unsqueeze(1)).float().mean(dim=1)

        # -- pull to Python once ---------------------------
        f_min_l = f_min.tolist()
        f_max_l = f_max.tolist()
        mean_l = mean.tolist()
        std_l = std.tolist()
        peak_l = peak.tolist()
        rms_l = rms.tolist()
        crest_l = crest_factor.tolist()
        skew_l = skew.tolist()
        kurt_l = kurt.tolist()
        excess_l = excess_kurt.tolist()
        bimod_l = bimodality.tolist()
        sparse_l = sparse_ratio.tolist()
        drange_l = dynamic_range_bits.tolist()
        outlier_l = outlier_ratio.tolist()

        # -- per-channel histogram + entropy (torch.histc is single-threaded CPU) --
        results = []
        max_entropy = torch.log2(torch.tensor(self.hist_bins, dtype=torch.float32)).item()
        for i in range(f.shape[0]):
            hist = torch.histc(f[i], bins=self.hist_bins)
            probs = hist.float() / (n + 1e-30)
            probs_pos = probs[probs > 0]
            entropy_raw = -(probs_pos * torch.log2(probs_pos + 1e-30)).sum().item()
            norm_entropy = entropy_raw / (max_entropy + 1e-30)

            results.append({
                "min": f_min_l[i],
                "max": f_max_l[i],
                "mean": mean_l[i],
                "std": std_l[i],
                "peak": peak_l[i],
                "rms": rms_l[i],
                "crest_factor": crest_l[i],
                "skewness": skew_l[i],
                "kurtosis": kurt_l[i],
                "excess_kurtosis": excess_l[i],
                "bimodality_coefficient": bimod_l[i],
                "sparse_ratio": sparse_l[i],
                "dynamic_range_bits": drange_l[i],
                "outlier_ratio": outlier_l[i],
                "norm_entropy": norm_entropy,
            })

        return results

    # ------------------------------------------------------------------
    # Vectorized per-block aggregate
    # ------------------------------------------------------------------

    def _measure_batch(self, fp32_2d, quant_2d, valid_counts=None):
        """Per-block aggregate: mean/std/min/max of key distribution stats."""
        err_sq = (fp32_2d - quant_2d).pow(2)
        if valid_counts is not None:
            n_valid = valid_counts.clamp_min(1)
            mse = err_sq.sum(dim=1) / n_valid
        else:
            mse = err_sq.mean(dim=1)
        return {
            "mse": mse.mean().item(),
            "mse_std": mse.std(unbiased=False).item() if mse.numel() > 1 else 0.0,
            "mse_min": mse.min().item(),
            "mse_max": mse.max().item(),
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
            "fp32_min": fp32.min().item(),
            "fp32_max": fp32.max().item(),
        }


# ---------------------------------------------------------------------------
# DistributionFitObserver — scipy-based parametric distribution fitting
# ---------------------------------------------------------------------------

try:
    import scipy.stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class DistributionFitObserver(SliceAwareObserver):
    """Fit fp32 tensor distribution to parametric distributions via MLE + KS.

    For each slice, fits all applicable candidate distributions (norm,
    laplace, cauchy, uniform on any data; lognorm, expon, gamma on
    non-negative data), ranks by Kolmogorov-Smirnov statistic, and reports
    the best fit with its parameters.

    Requires ``scipy``. Install with ``pip install scipy``.
    """

    _ALL_DISTS = ["norm", "laplace", "cauchy", "uniform"]
    _POSITIVE_DISTS = ["lognorm", "expon", "gamma"]

    def __init__(self, candidates=None):
        super().__init__()
        if not _HAS_SCIPY:
            raise ImportError(
                "DistributionFitObserver requires scipy. "
                "Install with: pip install scipy"
            )
        self.candidates = candidates or self._ALL_DISTS + self._POSITIVE_DISTS

    def _measure(self, key, fp32, quant):
        x = fp32.detach().cpu().numpy().ravel()

        dists = list(self._ALL_DISTS)
        if x.min() >= 0:
            dists.extend(self._POSITIVE_DISTS)
        dists = [d for d in dists if d in self.candidates]

        results = []
        for name in dists:
            dist = getattr(_scipy_stats, name)
            try:
                params = dist.fit(x)
                ks_stat, _ = _scipy_stats.kstest(x, dist.cdf, args=params)
                results.append((name, params, ks_stat))
            except Exception:
                continue

        if not results:
            return {"best_fit": "unknown", "best_fit_ks": float("inf")}

        results.sort(key=lambda r: r[2])
        best = results[0]
        return {
            "best_fit": best[0],
            "best_fit_params": tuple(float(p) for p in best[1]),
            "best_fit_ks": float(best[2]),
            "fit_ranking": [(r[0], float(r[2])) for r in results],
        }
