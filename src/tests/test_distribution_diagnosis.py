"""Tests for src/analysis/_distribution_diagnosis.py — DistributionDiagnosis + classify_distribution."""

import pytest

from src.analysis._distribution_diagnosis import (
    DistributionDiagnosis,
    classify_distribution,
    _RULES,
)
from src.session._config import QuantConfig
from src.session._result import SessionResult


# ---------------------------------------------------------------------------
# classify_distribution — standalone function
# ---------------------------------------------------------------------------

class TestClassifyDistribution:
    def test_outlier_dominated(self):
        metrics = {"outlier_ratio": 0.05, "crest_factor": 15.0}
        label, desc, suggestion = classify_distribution(metrics)
        assert label == "outlier_dominated"
        assert "Outlier-dominated" in desc
        assert "per_channel" in suggestion.lower()

    def test_high_dynamic_range(self):
        metrics = {"dynamic_range_bits": 10.0, "outlier_ratio": 0.01}
        label, desc, suggestion = classify_distribution(metrics)
        assert label == "high_dynamic_range"
        assert "High dynamic range" in desc

    def test_heavy_tailed(self):
        metrics = {"excess_kurtosis": 5.0, "outlier_ratio": 0.03}
        label, desc, suggestion = classify_distribution(metrics)
        assert label == "heavy_tailed"
        assert "Heavy-tailed" in desc

    def test_bimodal(self):
        metrics = {"bimodality_coefficient": 0.85}
        label, desc, suggestion = classify_distribution(metrics)
        assert label == "bimodal"
        assert "Bimodal" in desc

    def test_low_entropy(self):
        metrics = {"norm_entropy": 0.1}
        label, desc, suggestion = classify_distribution(metrics)
        assert label == "low_entropy"
        assert "Low entropy" in desc

    def test_benign_fallback(self):
        """When no specific rule matches, benign should be returned."""
        metrics = {"outlier_ratio": 0.01, "crest_factor": 5.0,
                   "dynamic_range_bits": 4.0, "excess_kurtosis": 1.0,
                   "bimodality_coefficient": 0.3, "norm_entropy": 0.5}
        label, desc, suggestion = classify_distribution(metrics)
        assert label == "benign"
        assert "Benign" in desc

    def test_empty_dict_returns_benign(self):
        label, desc, suggestion = classify_distribution({})
        assert label == "benign"

    def test_partial_keys_missing_triggers_next_rule(self):
        """Missing keys should cause KeyError → fall through to next rule."""
        metrics = {"outlier_ratio": 0.05}  # crest_factor missing → KeyError → skip
        label, desc, suggestion = classify_distribution(metrics)
        # Without crest_factor, outlier_dominated can't match.
        # Next rules: check high_dynamic_range (needs dynamic_range_bits → skip),
        # heavy_tailed (needs excess_kurtosis → skip), bimodal (needs bimodality_coefficient → skip),
        # low_entropy (needs norm_entropy → skip), benign (always matches)
        assert label == "benign"

    def test_returns_three_tuple(self):
        result = classify_distribution({})
        assert len(result) == 3
        assert all(isinstance(x, str) for x in result)

    def test_all_rules_have_expected_structure(self):
        """Verify each rule has (label, description, suggestion, callable)."""
        for rule in _RULES:
            assert len(rule) == 4
            label, desc, suggestion, condition = rule
            assert isinstance(label, str)
            assert isinstance(desc, str)
            assert isinstance(suggestion, str)
            assert callable(condition)


# ---------------------------------------------------------------------------
# Helpers for DistributionDiagnosis
# ---------------------------------------------------------------------------

def _make_dist_observers_data(**overrides) -> dict:
    """Build observers_data with distribution metrics for testing."""
    defaults = {
        "layer0.linear": {
            "weight": {
                "pre_quant[0]": {
                    ("tensor",): {
                        "crest_factor": 12.0,
                        "outlier_ratio": 0.03,
                        "dynamic_range_bits": 6.0,
                        "excess_kurtosis": 2.0,
                        "bimodality_coefficient": 0.3,
                        "norm_entropy": 0.5,
                        "skewness": 0.1,
                        "sparse_ratio": 0.05,
                    }
                }
            },
            "input": {
                "pre_quant[0]": {
                    ("tensor",): {
                        "crest_factor": 3.0,
                        "outlier_ratio": 0.01,
                    }
                }
            },
        },
    }
    defaults.update(overrides)
    return defaults


def _make_diag_result(**overrides) -> SessionResult:
    defaults = dict(
        name="test",
        config=QuantConfig(name="int8", w_format="int8", w_granularity="per_tensor"),
        qsnr_by_role={
            "input": {"layer0.linear": 25.0, "layer1.conv": 30.0},
            "weight": {"layer0.linear": 20.0, "layer1.conv": 35.0},
            "output": {},
        },
        mse_by_role={
            "input": {"layer0.linear": 0.001, "layer1.conv": 0.0005},
            "weight": {"layer0.linear": 0.01, "layer1.conv": 0.0002},
            "output": {},
        },
        observers_data=_make_dist_observers_data(),
    )
    defaults.update(overrides)
    return SessionResult(**defaults)


# ---------------------------------------------------------------------------
# DistributionDiagnosis — classify
# ---------------------------------------------------------------------------

class TestDistributionDiagnosisClassify:
    def test_returns_label_for_layer_with_dist_data(self):
        result = _make_diag_result()
        diag = DistributionDiagnosis(result)
        label = diag.classify("layer0.linear", role="weight")
        assert label == "outlier_dominated"

    def test_returns_no_data_when_layer_missing(self):
        result = _make_diag_result()
        diag = DistributionDiagnosis(result)
        assert diag.classify("nonexistent", role="weight") == "no_data"

    def test_returns_no_data_when_role_missing(self):
        result = _make_diag_result()
        diag = DistributionDiagnosis(result)
        assert diag.classify("layer0.linear", role="output") == "no_data"


# ---------------------------------------------------------------------------
# DistributionDiagnosis — profile
# ---------------------------------------------------------------------------

class TestDistributionDiagnosisProfile:
    def test_profile_with_distribution_data(self):
        result = _make_diag_result()
        diag = DistributionDiagnosis(result)
        text = diag.profile("layer0.linear", role="weight")
        assert "layer0.linear (weight)" in text
        assert "QSNR: 20.0 dB" in text
        assert "MSE:" in text
        assert "Crest factor" in text
        assert "Outlier ratio" in text
        assert "Diagnosis: outlier_dominated" in text

    def test_profile_without_distribution_data(self):
        result = _make_diag_result(observers_data={})
        diag = DistributionDiagnosis(result)
        text = diag.profile("layer0.linear", role="weight")
        assert "(no distributionobserver data)" in text.lower()

    def test_profile_shows_best_fit_when_present(self):
        data = {
            "layer0.linear": {
                "weight": {
                    "pre_quant[0]": {
                        ("tensor",): {
                            "best_fit": "gaussian",
                            "best_fit_ks": 0.05,
                            "best_fit_params": {"loc": 0.0, "scale": 1.0},
                        }
                    }
                }
            }
        }
        result = _make_diag_result(observers_data=data)
        diag = DistributionDiagnosis(result)
        text = diag.profile("layer0.linear", role="weight")
        assert "gaussian" in text
        assert "KS=0.05" in text


# ---------------------------------------------------------------------------
# DistributionDiagnosis — causal_analysis
# ---------------------------------------------------------------------------

class TestDistributionDiagnosisCausalAnalysis:
    def test_returns_formatted_table(self):
        result = _make_diag_result()
        diag = DistributionDiagnosis(result)
        text = diag.causal_analysis()
        assert "Layer" in text
        assert "QSNR" in text
        assert "Crest" in text
        assert "Classification" in text

    def test_no_qsnr_data(self):
        result = _make_diag_result(qsnr_by_role={})
        diag = DistributionDiagnosis(result)
        text = diag.causal_analysis()
        assert "No per-role QSNR" in text

    def test_no_observer_data_shows_warning(self):
        result = _make_diag_result(observers_data={})
        diag = DistributionDiagnosis(result)
        text = diag.causal_analysis()
        # Should still show the rows with "no_data" classification
        assert "no_data" in text

    def test_rows_sorted_by_qsnr_ascending(self):
        result = _make_diag_result()
        diag = DistributionDiagnosis(result)
        text = diag.causal_analysis()
        lines = text.split("\n")
        data_lines = [l for l in lines if l.startswith("layer")]
        # First data line should be worst QSNR = layer0.linear weight 20.0
        assert "20.0" in data_lines[0]
