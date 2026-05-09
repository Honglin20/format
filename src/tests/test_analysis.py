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

    def test_crest_factor_fields(self):
        """Verify peak, rms, and crest_factor are computed correctly."""
        obs = DistributionObserver()
        # Known values: [2.0, 3.0]
        # peak = max(abs([2, 3])) = 3.0
        # rms = sqrt(mean([4, 9])) = sqrt(6.5) ≈ 2.5495
        # crest_factor = 3.0 / 2.5495 ≈ 1.1767
        f = torch.tensor([2.0, 3.0])
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert "peak" in metrics
        assert "rms" in metrics
        assert "crest_factor" in metrics
        assert metrics["peak"] == pytest.approx(3.0)
        assert metrics["rms"] == pytest.approx(2.5495, abs=1e-4)
        assert metrics["crest_factor"] == pytest.approx(1.1767, abs=1e-4)

    def test_crest_factor_negative_values(self):
        """peak = max(|x|), crest should work for signed tensors."""
        obs = DistributionObserver()
        # [-5.0, 2.0] → peak = 5.0, rms = sqrt((25+4)/2) = sqrt(14.5) ≈ 3.8079
        f = torch.tensor([-5.0, 2.0])
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["peak"] == pytest.approx(5.0)
        assert metrics["crest_factor"] == pytest.approx(5.0 / 3.8079, abs=1e-4)

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


import torch.nn as nn
from src.analysis.context import AnalysisContext


class TestAnalysisContext:
    def test_context_attaches_and_detaches_observers(self):
        from src.observer.mixin import ObservableMixin

        class DummyLayer(ObservableMixin, nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(DummyLayer(), DummyLayer())
        from src.analysis.observers import QSNRObserver

        with AnalysisContext(model, [QSNRObserver()]) as ctx:
            for m in model.modules():
                if isinstance(m, ObservableMixin):
                    assert len(m._observers) == 1

        for m in model.modules():
            if isinstance(m, ObservableMixin):
                assert len(m._observers) == 0

    def test_report_returns_report_object(self):
        from src.observer.mixin import ObservableMixin

        class NoOpLayer(ObservableMixin, nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(NoOpLayer())
        from src.analysis.observers import QSNRObserver

        with AnalysisContext(model, [QSNRObserver()]) as ctx:
            model(torch.randn(3, 4))

        from src.analysis.report import Report
        report = ctx.report()
        assert isinstance(report, Report)

    def test_warmup_batches_reset_observer(self):
        from src.observer.mixin import ObservableMixin

        class SimpleLayer(ObservableMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10))

            def forward(self, x):
                return x

        model = SimpleLayer()
        from src.analysis.observers import MSEObserver
        obs = MSEObserver()

        with AnalysisContext(model, [obs], warmup_batches=2) as ctx:
            model(torch.randn(5, 10))
            ctx.step()
            # After warmup, no data should be accumulated (emit_fn not wired here)
            assert len(obs.report()) == 0


from src.quantize import quantize
from src.scheme.quant_scheme import QuantScheme
from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.transform import IdentityTransform


class TestEndToEnd:
    """End-to-end: quantized model + AnalysisContext → report."""

    def test_two_layer_linear_e2e(self):
        from src.ops.linear import QuantizedLinear
        from src.scheme.op_config import OpQuantConfig

        fmt = FormatBase.from_str("fp8_e4m3")
        scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec(mode=GranularityMode.PER_TENSOR),
            transform=IdentityTransform(),
        )
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = QuantizedLinear(8, 4, bias=False, cfg=cfg, name="layer0")
                self.layer1 = QuantizedLinear(4, 2, bias=False, cfg=cfg, name="layer1")

            def forward(self, x):
                return self.layer1(self.layer0(x))

        model = TwoLayer()
        observers = [DistributionObserver(), QSNRObserver(), MSEObserver()]
        x = torch.randn(3, 8)

        with AnalysisContext(model, observers) as ctx:
            model(x)

        report = ctx.report()
        assert len(report.keys()) >= 2
        for name in report.keys():
            assert "layer" in name.lower()

        df = report.to_dataframe()
        assert len(df) > 0
        assert "qsnr_db" in df.columns
        assert "dynamic_range_bits" in df.columns

    def test_empty_observers_no_crash(self):
        from src.ops.linear import QuantizedLinear
        from src.scheme.op_config import OpQuantConfig

        fmt = FormatBase.from_str("fp8_e4m3")
        scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec(mode=GranularityMode.PER_TENSOR),
            transform=IdentityTransform(),
        )
        cfg = OpQuantConfig(input=scheme, weight=scheme)

        model = QuantizedLinear(8, 4, bias=False, cfg=cfg, name="test_layer")
        x = torch.randn(2, 8)
        y = model(x)
        assert y.shape == (2, 4)


from src.analysis.correlation import (
    DistributionProfile, DistributionTaxonomy,
)


class TestDistributionProfile:
    def make_dist_report(self):
        raw = {
            "layer1.linear": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "mean": 0.1, "std": 1.2, "skewness": 1.5,
                            "kurtosis": 5.0, "sparse_ratio": 0.3,
                            "dynamic_range_bits": 4.5, "outlier_ratio": 0.02,
                        },
                    }
                },
                "weight": {
                    "weight_pre_quant[0]": {
                        ("tensor",): {
                            "mean": -0.01, "std": 0.8, "skewness": -0.1,
                            "kurtosis": 3.2, "sparse_ratio": 0.01,
                            "dynamic_range_bits": 3.0, "outlier_ratio": 0.0,
                        },
                    }
                },
            },
            "layer2.conv": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "mean": 0.5, "std": 0.9, "skewness": 0.8,
                            "kurtosis": 4.1, "sparse_ratio": 0.05,
                            "dynamic_range_bits": 5.2, "outlier_ratio": 0.01,
                        },
                    }
                },
            },
        }
        from src.analysis.report import Report
        return Report(raw)

    def test_by_role_aggregates_correctly(self):
        report = self.make_dist_report()
        profile = DistributionProfile.from_report(report)

        ip = profile.by_role("input")
        assert ip["sample_count"] == 2
        assert ip["dynamic_range_bits"]["p50"] == pytest.approx(4.85, abs=0.1)

    def test_all_roles(self):
        report = self.make_dist_report()
        profile = DistributionProfile.from_report(report)
        all_roles = profile.all_roles()
        assert "input" in all_roles
        assert "weight" in all_roles

    def test_empty_report(self):
        from src.analysis.report import Report
        report = Report({})
        profile = DistributionProfile.from_report(report)
        assert profile.by_role("input")["sample_count"] == 0

    def test_print_profile_does_not_crash(self):
        report = self.make_dist_report()
        profile = DistributionProfile.from_report(report)
        profile.print_profile()


class TestDistributionTaxonomy:
    def make_taxonomy_report(self):
        raw = {}
        archetypes = [
            ("l_gauss", "weight", 0.1, 3.1, 0.4, 0.02, 0.6),
            ("l_pos", "input", 1.2, 4.0, 0.3, 0.25, 0.5),
            ("l_neg", "input", -0.8, 3.5, 0.4, 0.05, 0.5),
            ("l_heavy", "output", 0.2, 8.0, 0.3, 0.05, 0.4),
            ("l_bi", "weight", 0.1, 2.5, 0.6, 0.05, 0.55),
            ("l_unif", "input", 0.1, 2.0, 0.4, 0.05, 0.9),
            ("l_zero", "input", 0.2, 3.0, 0.3, 0.5, 0.3),
            ("l_logn", "output", 1.5, 5.0, 0.3, 0.1, 0.6),
        ]
        for layer, role, sk, ku, bi, sp, ent in archetypes:
            raw[layer] = {
                role: {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "skewness": sk, "kurtosis": ku,
                            "bimodality_coefficient": bi,
                            "sparse_ratio": sp, "norm_entropy": ent,
                        }
                    }
                }
            }
        from src.analysis.report import Report
        return Report(raw)

    def test_classify_all_eight_types(self):
        report = self.make_taxonomy_report()
        taxonomy = DistributionTaxonomy.from_report(report)
        result = taxonomy.classify()

        cluster_names = set(result.keys())
        expected = {"zero-centered-gaussian", "positive-skewed", "negative-skewed",
                    "heavy-tailed", "bimodal", "uniform-like",
                    "zero-inflated", "log-normal-like"}
        assert len(cluster_names & expected) >= 6

        for name, cluster in result.items():
            assert "count" in cluster
            assert "percentage" in cluster
            assert "representative_layers" in cluster
            assert cluster["count"] > 0

    def test_unclassified_fallback(self):
        raw = {
            "weird_layer": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "skewness": 0.3, "kurtosis": 4.5,
                            "bimodality_coefficient": 0.4,
                            "sparse_ratio": 0.15, "norm_entropy": 0.6,
                        }
                    }
                }
            }
        }
        from src.analysis.report import Report
        report = Report(raw)
        taxonomy = DistributionTaxonomy.from_report(report)
        result = taxonomy.classify()
        assert "unclassified" in result

    def test_get_exemplars_returns_structure(self):
        report = self.make_taxonomy_report()
        taxonomy = DistributionTaxonomy.from_report(report)
        exemplars = taxonomy.get_exemplars("positive-skewed", n=1)
        assert len(exemplars) >= 1
        assert "layer" in exemplars[0]
        assert "role" in exemplars[0]

    def test_print_taxonomy_no_crash(self):
        report = self.make_taxonomy_report()
        taxonomy = DistributionTaxonomy.from_report(report)
        taxonomy.print_taxonomy()
        taxonomy.print_taxonomy(ascii_plots=True)


from src.analysis.correlation import ErrorByDistribution, LayerSensitivity


class TestErrorByDistribution:
    def make_error_report(self):
        import math
        raw = {}
        for i, (dr, qsnr) in enumerate([
            (3.0, 45.0), (4.5, 35.0), (7.0, 15.0),
            (9.0, 6.0), (3.5, 42.0), (5.0, 28.0),
        ]):
            raw[f"layer{i}"] = {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "qsnr_db": qsnr,
                            "mse": 10 ** (-qsnr / 10),
                            "dynamic_range_bits": dr,
                        },
                    }
                }
            }
        from src.analysis.report import Report
        return Report(raw)

    def test_rank_layers(self):
        report = self.make_error_report()
        eb = ErrorByDistribution(report)

        ranked = eb.rank_layers(by="qsnr_db", k=3, ascending=True)
        assert len(ranked) == 3
        assert ranked[0][2] == pytest.approx(6.0, abs=0.1)
        assert ranked[0][0] == "layer3"

    def test_group_by_range(self):
        report = self.make_error_report()
        eb = ErrorByDistribution(report)

        groups = eb.group_by_range(role="input", bins=[0, 4, 7, 999])
        assert "0-4 bits" in groups
        assert groups["0-4 bits"]["avg_qsnr"] == pytest.approx(43.5, abs=0.1)
        assert groups["0-4 bits"]["verdict"] == "excellent"
        assert "7-999 bits" in groups
        assert groups["7-999 bits"]["avg_qsnr"] == pytest.approx(10.5, abs=0.1)

    def test_print_correlation_no_crash(self):
        report = self.make_error_report()
        eb = ErrorByDistribution(report)
        eb.print_correlation()


class TestLayerSensitivity:
    def make_sensitivity_report(self):
        raw = {}
        for i, (ltype, mse) in enumerate([
            ("Linear", 1e-3), ("Linear", 5e-4), ("Conv", 2e-3),
            ("Linear", 1e-5), ("Conv", 8e-4),
        ]):
            raw[f"layer{i}.{ltype}"] = {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {"mse": mse, "qsnr_db": 30.0},
                    }
                }
            }
        from src.analysis.report import Report
        return Report(raw)

    def test_topk(self):
        report = self.make_sensitivity_report()
        sens = LayerSensitivity(report)

        top2 = sens.topk(k=2, metric="mse")
        assert len(top2) == 2
        assert top2[0][2] == 0.002

    def test_by_layer_type(self):
        report = self.make_sensitivity_report()
        sens = LayerSensitivity(report)

        by_type = sens.by_layer_type()
        assert "Linear" in by_type
        assert "Conv" in by_type

    def test_above_threshold(self):
        report = self.make_sensitivity_report()
        sens = LayerSensitivity(report)

        above = sens.above_threshold(metric="mse", threshold=1e-3)
        assert len(above) >= 1


# ---------------------------------------------------------------------------
# DistributionFitObserver + DistributionFitTaxonomy
# ---------------------------------------------------------------------------

try:
    import scipy.stats  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_fit_skip = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")


@_fit_skip
class TestDistributionFitObserver:
    """Unit tests for DistributionFitObserver._measure()."""

    @staticmethod
    def _measure(f):
        from src.analysis.observers import DistributionFitObserver
        obs = DistributionFitObserver()
        q = f.clone()
        return obs._measure(("tensor",), f, q)

    def test_gaussian_identified(self):
        torch.manual_seed(42)
        f = torch.randn(1000)
        m = self._measure(f)
        assert m["best_fit"] == "norm"
        assert m["best_fit_ks"] < 0.1

    def test_laplace_identified(self):
        torch.manual_seed(42)
        d = torch.distributions.Laplace(0, 1)
        f = d.sample((1000,))
        m = self._measure(f)
        assert m["best_fit"] == "laplace"
        assert m["best_fit_ks"] < 0.1

    def test_lognormal_positive_data(self):
        torch.manual_seed(42)
        f = torch.distributions.LogNormal(0, 0.5).sample((1000,))
        m = self._measure(f)
        assert m["best_fit"] in ("lognorm", "gamma")

    def test_cauchy_heavy_tailed(self):
        torch.manual_seed(42)
        f = torch.distributions.Cauchy(0, 1).sample((2000,))
        m = self._measure(f)
        assert m["best_fit"] == "cauchy"

    def test_uniform_identified(self):
        torch.manual_seed(42)
        f = torch.rand(1000) * 10 - 5  # Uniform [-5, 5]
        m = self._measure(f)
        assert m["best_fit"] in ("uniform", "norm")  # uniform hard to distinguish

    def test_all_zeros_still_produces_result(self):
        """All-zeros (constant) data still gets fit; verify result structure."""
        f = torch.zeros(100)
        m = self._measure(f)
        assert isinstance(m["best_fit"], str)
        assert isinstance(m["best_fit_ks"], float)
        assert isinstance(m["fit_ranking"], list)

    def test_params_are_floats(self):
        torch.manual_seed(42)
        f = torch.randn(1000)
        m = self._measure(f)
        assert isinstance(m["best_fit_params"], tuple)
        for p in m["best_fit_params"]:
            assert isinstance(p, float)

    def test_fit_ranking_included(self):
        torch.manual_seed(42)
        f = torch.randn(1000)
        m = self._measure(f)
        assert isinstance(m["fit_ranking"], list)
        assert len(m["fit_ranking"]) >= 4  # at least the 4 all-data dists
        for name, ks in m["fit_ranking"]:
            assert isinstance(name, str)
            assert isinstance(ks, float)

    def test_negative_data_excludes_positive_dists(self):
        """Data with negative values should skip lognorm/expon/gamma."""
        torch.manual_seed(42)
        f = torch.randn(1000) - 2.0  # mean ≈ -2, plenty negative
        m = self._measure(f)
        dist_names = [r[0] for r in m["fit_ranking"]]
        assert "lognorm" not in dist_names
        assert "expon" not in dist_names
        assert "gamma" not in dist_names


@_fit_skip
class TestDistributionFitTaxonomy:
    """Tests for DistributionFitTaxonomy."""

    @staticmethod
    def _make_report(samples):
        """Build a synthetic report from (layer, role, best_fit, best_fit_ks) tuples."""
        raw = {}
        for layer, role, best_fit, ks in samples:
            raw[layer] = {
                role: {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "best_fit": best_fit,
                            "best_fit_ks": ks,
                            "best_fit_params": (),
                        }
                    }
                }
            }
        from src.analysis.report import Report
        return Report(raw)

    def test_classify_distribution_types(self):
        from src.analysis.correlation import DistributionFitTaxonomy

        report = self._make_report([
            ("l1", "input", "norm", 0.02),
            ("l2", "input", "norm", 0.03),
            ("l3", "weight", "norm", 0.01),
            ("l4", "input", "laplace", 0.04),
            ("l5", "input", "laplace", 0.05),
            ("l6", "output", "cauchy", 0.06),
        ])
        taxonomy = DistributionFitTaxonomy.from_report(report)
        result = taxonomy.classify()

        assert result["norm"]["count"] == 3
        assert result["laplace"]["count"] == 2
        assert result["cauchy"]["count"] == 1

    def test_percentage_sum_to_100(self):
        from src.analysis.correlation import DistributionFitTaxonomy

        report = self._make_report([
            ("l1", "input", "norm", 0.01),
            ("l2", "input", "norm", 0.02),
            ("l3", "weight", "laplace", 0.03),
        ])
        taxonomy = DistributionFitTaxonomy.from_report(report)
        result = taxonomy.classify()

        assert result["norm"]["percentage"] == "67%"
        assert result["laplace"]["percentage"] == "33%"

    def test_get_exemplars(self):
        from src.analysis.correlation import DistributionFitTaxonomy

        report = self._make_report([
            ("l1", "input", "norm", 0.01),
            ("l2", "weight", "norm", 0.02),
        ])
        taxonomy = DistributionFitTaxonomy.from_report(report)
        exemplars = taxonomy.get_exemplars("norm", n=1)
        assert len(exemplars) == 1
        assert exemplars[0]["layer"] == "l1"
        assert exemplars[0]["role"] == "input"

    def test_get_exemplars_missing_cluster(self):
        from src.analysis.correlation import DistributionFitTaxonomy

        report = self._make_report([("l1", "input", "norm", 0.01)])
        taxonomy = DistributionFitTaxonomy.from_report(report)
        assert taxonomy.get_exemplars("nonexistent") == []

    def test_no_fit_data_returns_empty(self):
        from src.analysis.report import Report
        from src.analysis.correlation import DistributionFitTaxonomy

        raw = {
            "l1": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {"mse": 0.001},  # no best_fit key
                    }
                }
            }
        }
        report = Report(raw)
        taxonomy = DistributionFitTaxonomy.from_report(report)
        assert taxonomy.classify() == {}

    def test_print_taxonomy_no_crash(self):
        from src.analysis.correlation import DistributionFitTaxonomy

        report = self._make_report([
            ("l1", "input", "norm", 0.02),
            ("l2", "weight", "laplace", 0.04),
        ])
        taxonomy = DistributionFitTaxonomy.from_report(report)
        taxonomy.print_taxonomy()


class TestTaxonomyAccessor:
    """Tests for the merged TaxonomyAccessor (fit-first, rules fallback)."""

    @staticmethod
    def _report(samples):
        """Build a synthetic AnalysisReport.

        Each sample: (layer, role, metrics_dict)
        """
        raw = {}
        for layer, role, metrics in samples:
            raw[layer] = {
                role: {
                    "pre[0]": {("tensor",): metrics},
                }
            }
        from src.analysis.report import Report
        return Report(raw)

    def test_fit_first_when_best_fit_present(self):
        report = self._report([
            ("l1", "input", {"best_fit": "norm", "best_fit_ks": 0.02,
             "best_fit_params": (0.0, 1.0), "skewness": 1.5, "kurtosis": 5.0}),
        ])
        result = report.taxonomy.classify()
        assert "norm" in result
        assert result["norm"]["fit_count"] == 1
        assert result["norm"]["rules_count"] == 0

    def test_rules_fallback_when_no_best_fit(self):
        report = self._report([
            ("l1", "weight", {"skewness": 0.1, "kurtosis": 3.1,
             "bimodality_coefficient": 0.4, "sparse_ratio": 0.02,
             "norm_entropy": 0.6}),
        ])
        result = report.taxonomy.classify()
        assert "zero-centered-gaussian" in result
        assert result["zero-centered-gaussian"]["rules_count"] == 1
        assert result["zero-centered-gaussian"]["fit_count"] == 0

    def test_mixed_fit_and_rules(self):
        report = self._report([
            ("l1", "weight", {"best_fit": "norm", "best_fit_ks": 0.02}),
            ("l2", "input", {"best_fit": "norm", "best_fit_ks": 0.03}),
            ("l3", "output", {"skewness": 0.8, "kurtosis": 4.0,
             "bimodality_coefficient": 0.3, "sparse_ratio": 0.05,
             "norm_entropy": 0.5}),
        ])
        result = report.taxonomy.classify()
        assert result["norm"]["count"] == 2
        assert result["norm"]["fit_count"] == 2
        assert result["positive-skewed"]["count"] == 1
        assert result["positive-skewed"]["rules_count"] == 1

    def test_empty_report(self):
        report = self._report([])
        result = report.taxonomy.classify()
        assert result == {}

    def test_no_distribution_data(self):
        report = self._report([
            ("l1", "input", {"mse": 0.001, "qsnr_db": 40.0}),
        ])
        result = report.taxonomy.classify()
        assert result == {}

    def test_exemplars(self):
        report = self._report([
            ("l1", "weight", {"best_fit": "norm", "best_fit_ks": 0.01}),
            ("l2", "input", {"best_fit": "norm", "best_fit_ks": 0.02}),
        ])
        ex = report.taxonomy.exemplars("norm", n=1)
        assert len(ex) == 1
        assert ex[0]["layer"] == "l1"

    def test_exemplars_missing_cluster(self):
        report = self._report([
            ("l1", "weight", {"best_fit": "norm", "best_fit_ks": 0.01}),
        ])
        assert report.taxonomy.exemplars("nonexistent") == []

    def test_print_no_crash(self):
        report = self._report([
            ("l1", "weight", {"best_fit": "norm", "best_fit_ks": 0.02}),
            ("l2", "input", {"skewness": 1.2, "kurtosis": 5.0,
             "bimodality_coefficient": 0.3, "sparse_ratio": 0.1,
             "norm_entropy": 0.5}),
        ])
        report.taxonomy.print()
