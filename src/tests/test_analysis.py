import torch
import pytest
from src.analysis.observers import DistributionObserver


class TestDistributionObserver:
    """Unit tests for DistributionObserver._measure()."""

    def test_gaussian_distribution(self):
        """Synthetic Gaussian: skew≈0, kurt≈3, sparse≈0."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.randn(1000)
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["mean"] == pytest.approx(0.0, abs=0.1)
        assert metrics["std"] == pytest.approx(1.0, abs=0.1)
        assert metrics["skewness"] == pytest.approx(0.0, abs=0.3)
        assert metrics["excess_kurtosis"] == pytest.approx(0.0, abs=0.5)
        assert metrics["sparse_ratio"] < 0.05
        assert metrics["norm_entropy"] > 0.5

    def test_positive_skewed(self):
        """ReLU-like: right-skewed, high sparsity."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.randn(1000).clamp(min=0)
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["skewness"] > 0.5
        assert metrics["sparse_ratio"] == pytest.approx(0.5, abs=0.1)

    def test_bimodal_distribution(self):
        """Two separated Gaussians: bimodality_coefficient > 0.555."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.cat([torch.randn(500) - 2.0, torch.randn(500) + 2.0])
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["bimodality_coefficient"] > 0.555
        assert abs(metrics["skewness"]) < 0.5

    def test_all_zeros(self):
        """Edge case: all zeros → sparse_ratio=1, dynamic_range_bits=0."""
        obs = DistributionObserver()
        f = torch.zeros(100)
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["sparse_ratio"] == 1.0
        assert metrics["dynamic_range_bits"] == 0.0
        assert metrics["std"] == 0.0

    def test_heavy_tailed(self):
        """Cauchy-like: excess kurtosis >> 0."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.distributions.Cauchy(0, 1).sample((2000,))
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["excess_kurtosis"] > 3.0

    def test_uniform_distribution(self):
        """Uniform: high normalized entropy, low skew."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.rand(1000)
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["norm_entropy"] > 0.85
        assert abs(metrics["skewness"]) < 0.5

    def test_dynamic_range_bits(self):
        """Known range: [1e-6, 1.0] → dynamic_range_bits ≈ log2(1e6) ≈ 20."""
        obs = DistributionObserver()
        f = torch.tensor([1e-6, 1.0, 0.5, 2e-6, 0.0, -0.3])
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["dynamic_range_bits"] == pytest.approx(19.93, abs=0.1)

    def test_outlier_detection(self):
        """One extreme outlier: outlier_ratio ≈ 1/N."""
        obs = DistributionObserver(outlier_sigma=3.0)
        torch.manual_seed(42)
        f = torch.randn(500)
        f[0] = 100.0
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)
        assert metrics["outlier_ratio"] > 0.0
        assert metrics["outlier_ratio"] < 0.01


from src.analysis.observers import QSNRObserver, MSEObserver


class TestQSNRObserver:
    def test_perfect_quantization_high_qsnr(self):
        """fp32 == quant → error=0 → QSNR should be very high."""
        obs = QSNRObserver()
        f = torch.randn(100)
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)
        assert metrics["qsnr_db"] > 100

    def test_known_error(self):
        """fp32=[1,2,3], 10% error → QSNR = 20 dB."""
        obs = QSNRObserver()
        f = torch.tensor([1.0, 2.0, 3.0])
        q = torch.tensor([0.9, 1.8, 2.7])

        metrics = obs._measure(("tensor",), f, q)
        # num = (1+4+9)/3 = 14/3, den = (0.01+0.04+0.09)/3 = 0.14/3
        # QSNR = 10*log10(100) = 20 dB
        assert metrics["qsnr_db"] == pytest.approx(20.0, abs=0.01)


class TestMSEObserver:
    def test_perfect_quantization_zero_mse(self):
        obs = MSEObserver()
        f = torch.randn(100)
        q = f.clone()
        metrics = obs._measure(("tensor",), f, q)
        assert metrics["mse"] == 0.0

    def test_known_error(self):
        obs = MSEObserver()
        f = torch.tensor([1.0, 2.0, 3.0])
        q = torch.tensor([0.9, 1.8, 2.7])
        metrics = obs._measure(("tensor",), f, q)
        # MSE = (0.01+0.04+0.09)/3
        assert metrics["mse"] == pytest.approx(0.04667, abs=1e-5)


from src.analysis.observers import HistogramObserver


class TestHistogramObserver:
    def test_bin_count(self):
        obs = HistogramObserver(n_bins=64)
        f = torch.randn(500)
        q = f + 0.01 * torch.randn(500)

        metrics = obs._measure(("tensor",), f, q)
        assert metrics["fp32_hist"].numel() == 64
        assert metrics["quant_hist"].numel() == 64
        assert metrics["err_hist"].numel() == 64

    def test_counts_sum_to_total(self):
        obs = HistogramObserver(n_bins=32)
        f = torch.randn(300)
        q = f + 0.01 * torch.randn(300)

        metrics = obs._measure(("tensor",), f, q)
        assert metrics["fp32_hist"].sum().item() == 300
        assert metrics["quant_hist"].sum().item() == 300
        assert metrics["err_hist"].sum().item() == 300
