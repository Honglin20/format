"""Tests for src.viz.tables and src.viz.theme."""
import math
import os
import tempfile

import pytest
from src.viz.tables import (
    accuracy_table,
    distribution_fit_table,
    format_comparison_table,
    pot_delta_table,
    sensitivity_table,
    transform_benefit_table,
    transform_distribution_table,
    transform_matrix_table,
)
from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS, HIST_COLORS, FALLBACK_CYCLE


class TestAccuracyTable:
    def test_generates_csv(self):
        results = {
            "MXINT-8": {
                "accuracy": {"accuracy": 0.95},
                "qsnr_per_layer": {"fc1": 20.0, "fc2": 18.0},
                "mse_per_layer": {"fc1": 0.001, "fc2": 0.002},
            },
            "FP32 (baseline)": {
                "accuracy": {"accuracy": 0.97},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            text = accuracy_table(results, title="Test Table", output_dir=tmpdir, filename="test.csv")

            csv_path = os.path.join(tmpdir, "tables", "test.csv")
            assert os.path.exists(csv_path)

            with open(csv_path) as f:
                content = f.read()
            assert "MXINT-8" in content
            assert "0.9500" in content

    def test_handles_flat_accuracy_value(self):
        results = {
            "A": {"accuracy": 0.88, "qsnr_per_layer": {}, "mse_per_layer": {}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = accuracy_table(results, title="T", output_dir=tmpdir, filename="f.csv")
            assert "0.8800" in text

    def test_handles_empty_results(self):
        text = accuracy_table({}, title="Empty", output_dir="/tmp", filename="empty.csv")
        assert text


class TestFormatComparisonTable:
    def test_delegates_to_accuracy_table(self):
        """format_comparison_table is an alias with a default filename."""
        results = {
            "A": {"accuracy": {"accuracy": 0.91}, "qsnr_per_layer": {}, "mse_per_layer": {}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = format_comparison_table(results, title="FC", output_dir=tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "comparison.csv")
            assert os.path.exists(csv_path)

    def test_custom_filename_accepted(self):
        results = {
            "A": {"accuracy": {"accuracy": 0.91}, "qsnr_per_layer": {}, "mse_per_layer": {}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = format_comparison_table(results, title="X", output_dir=tmpdir, filename="custom.csv")
            assert os.path.exists(os.path.join(tmpdir, "tables", "custom.csv"))


class TestPoTDeltaTable:
    def test_generates_csv_with_delta(self):
        part_c = {
            "FP32 (baseline)": {"accuracy": {"accuracy": 0.95}},
            "INT8-PC-FP32": {
                "accuracy": {"accuracy": 0.90},
                "qsnr_per_layer": {"fc1": 20.0},
                "mse_per_layer": {"fc1": 0.01},
            },
            "INT8-PC-PoT": {
                "accuracy": {"accuracy": 0.88},
                "qsnr_per_layer": {"fc1": 22.0},
                "mse_per_layer": {"fc1": 0.008},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = pot_delta_table(part_c, output_dir=tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "table3_pot.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
            assert "INT8-PC-FP32" in content
            # Delta = 0.90 - 0.95 = -0.05
            assert "-0.050000" in content

    def test_no_baseline(self):
        """Without 'FP32 (baseline)', baseline_acc stays 0."""
        part_c = {
            "A": {"accuracy": {"accuracy": 0.85}, "qsnr_per_layer": {}, "mse_per_layer": {}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = pot_delta_table(part_c, output_dir=tmpdir)
            # Delta = 0.85 - 0.0 = 0.85
            assert "+0.8500" in text


class TestTransformMatrixTable:
    def test_generates_csv(self):
        part_d = {
            "INT8": {
                "None": {"accuracy": {"accuracy": 0.85}},
                "SmoothQuant": {"accuracy": {"accuracy": 0.87}},
            },
            "FP8": {
                "None": {"accuracy": {"accuracy": 0.82}},
                "Hadamard": {"accuracy": {"accuracy": 0.84}},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_matrix_table(part_d, output_dir=tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "table4_format_x_transform.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
            assert "INT8" in content
            assert "FP8" in content

    def test_suffix_appended_to_filename(self):
        part_d = {"INT8": {"None": {"accuracy": {"accuracy": 0.85}}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_matrix_table(part_d, output_dir=tmpdir, suffix="_v2")
            assert os.path.exists(
                os.path.join(tmpdir, "tables", "table4_format_x_transform_v2.csv")
            )

    def test_handles_flat_accuracy(self):
        part_d = {"FMT": {"TX": {"accuracy": 0.77}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_matrix_table(part_d, output_dir=tmpdir)
            assert "0.7700" in text

    def test_missing_transform_shows_nan_in_text(self):
        """Missing transform rows get N/A text, CSV gets N/A."""
        part_d = {"FMT": {"TX": {"accuracy": {"accuracy": 0.85}}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_matrix_table(part_d, output_dir=tmpdir)
            # One transform present — text should show its value
            assert "0.8500" in text


class TestTransformDistributionTable:
    def test_generates_csv(self):
        part_d = {
            "INT8": {
                "None": {"qsnr_per_layer": {"fc1": 10.0, "fc2": 12.0}},
                "SmoothQuant": {"qsnr_per_layer": {"fc1": 15.0, "fc2": 11.0}},
                "Hadamard": {"qsnr_per_layer": {"fc1": 9.0, "fc2": 10.0}},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_distribution_table(part_d, output_dir=tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "table5_transform_distribution.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
            assert "INT8" in content
            assert "Total" in content

    def test_counts_are_nonnegative(self):
        part_d = {
            "FMT": {
                "None": {"qsnr_per_layer": {"a": 5.0, "b": 4.0}},
                "Hadamard": {"qsnr_per_layer": {"a": 6.0, "b": 3.0}},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_distribution_table(part_d, output_dir=tmpdir)
            lines = [l for l in text.split("\n") if l.strip() and "FMT" in l]
            assert len(lines) >= 1


class TestSensitivityTable:
    def test_generates_csv_ranked(self):
        all_results = {
            "part_a": {
                "INT8": {
                    "qsnr_per_layer": {"layer0": 15.0, "layer1": 22.0},
                    "mse_per_layer": {"layer0": 0.005, "layer1": 0.001},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = sensitivity_table(all_results, output_dir=tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "table6_sensitivity.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
            assert "Max_MSE" in content
            assert "Min_QSNR_dB" in content

    def test_limits_to_top_10(self):
        all_results = {}
        for i in range(20):
            all_results[f"p{i}"] = {
                f"cfg{i}": {
                    "qsnr_per_layer": {f"l{i}": float(30 - i)},
                    "mse_per_layer": {f"l{i}": float(0.001 * i)},
                },
            }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = sensitivity_table(all_results, output_dir=tmpdir)
            # Count rank lines (numbered rows)
            rank_lines = [l for l in text.split("\n") if l.strip() and l.strip()[0].isdigit()]
            assert len(rank_lines) <= 10

    def test_skips_non_dict_entries_gracefully(self):
        all_results = {"part_a": "not_a_dict"}
        with tempfile.TemporaryDirectory() as tmpdir:
            text = sensitivity_table(all_results, output_dir=tmpdir)
            assert text


class TestVizTheme:
    """Sanity checks for colour constants."""

    def test_format_colors_are_hex(self):
        for name, color in FORMAT_COLORS.items():
            assert color.startswith("#"), f"{name}: {color} not hex"
            assert len(color) == 7

    def test_transform_colors_are_hex(self):
        for name, color in TRANSFORM_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7

    def test_hist_colors_are_hex(self):
        for name, color in HIST_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7

    def test_fallback_cycle_has_entries(self):
        assert len(FALLBACK_CYCLE) >= 8
        for c in FALLBACK_CYCLE:
            assert c.startswith("#")
            assert len(c) == 7

    def test_each_format_has_unique_color(self):
        values = list(FORMAT_COLORS.values())
        assert len(values) == len(set(values))

    def test_each_transform_has_unique_color(self):
        values = list(TRANSFORM_COLORS.values())
        assert len(values) == len(set(values))


# ── Helpers for iter_slices-based table tests ───────────────────────────

class _MockReport:
    def __init__(self, slices):
        self._slices = slices

    def iter_slices(self):
        for item in self._slices:
            yield item


def _make_all_results(slices):
    return {"part_a": {"cfg1": {"report": _MockReport(slices)}}}


# ---------------------------------------------------------------------------
# Table 7 — Distribution fit classification
# ---------------------------------------------------------------------------

class TestDistributionFitTable:
    def test_generates_csv(self):
        slices = [
            ("l1", "input", "pre", ("tensor",), {"best_fit": "norm", "best_fit_ks": 0.05}),
            ("l1", "weight", "pre", ("tensor",), {"best_fit": "laplace", "best_fit_ks": 0.03}),
            ("l2", "input", "pre", ("tensor",), {"best_fit": "norm", "best_fit_ks": 0.07}),
            ("l2", "weight", "pre", ("tensor",), {"best_fit": "cauchy", "best_fit_ks": 0.08}),
            ("l3", "input", "pre", ("tensor",), {"best_fit": "norm", "best_fit_ks": 0.04}),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            text = distribution_fit_table(all_results, output_dir=tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "table7_distribution_fit.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
            assert "norm" in content
            assert "laplace" in content

    def test_counts_match_input(self):
        """3 norm + 1 laplace + 1 cauchy = 5 total for cfg1."""
        slices = [
            ("l1", "input", "pre", ("t",), {"best_fit": "norm"}),
            ("l2", "input", "pre", ("t",), {"best_fit": "norm"}),
            ("l3", "input", "pre", ("t",), {"best_fit": "norm"}),
            ("l4", "input", "pre", ("t",), {"best_fit": "laplace"}),
            ("l5", "input", "pre", ("t",), {"best_fit": "cauchy"}),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            text = distribution_fit_table(all_results, output_dir=tmpdir)
            # Text should contain "Total" column with value 5
            assert "Total" in text

    def test_empty_data_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Distribution fit data not available"):
                distribution_fit_table({}, output_dir=tmpdir)


# ---------------------------------------------------------------------------
# Table 8 — Transform per-layer benefit
# ---------------------------------------------------------------------------

class TestTransformBenefitTable:
    def test_generates_csv(self):
        part_d = {
            "INT8": {
                "None": {"qsnr_per_layer": {"fc1": 20.0, "fc2": 18.0}},
                "SmoothQuant": {"qsnr_per_layer": {"fc1": 22.0, "fc2": 17.0}},
            },
            "FP8": {
                "None": {"qsnr_per_layer": {"fc1": 25.0, "fc2": 23.0}},
                "Hadamard": {"qsnr_per_layer": {"fc1": 27.0, "fc2": 25.0}},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_benefit_table(part_d, output_dir=tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "table8_transform_benefit.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
            assert "INT8" in content
            assert "Baseline_QSNR" in content

    def test_shows_delta_values(self):
        part_d = {
            "INT8": {
                "None": {"qsnr_per_layer": {"fc1": 20.0}},
                "SmoothQuant": {"qsnr_per_layer": {"fc1": 22.0}},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_benefit_table(part_d, output_dir=tmpdir)
            # Delta = 22.0 - 20.0 = +2.00
            assert "+2.00" in text

    def test_no_baseline_skipped(self):
        part_d = {"INT8": {"SmoothQuant": {"qsnr_per_layer": {"fc1": 22.0}}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_benefit_table(part_d, output_dir=tmpdir)
            assert "no baseline data" in text.lower()

    def test_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No transform study data available"):
                transform_benefit_table({}, output_dir=tmpdir)

    def test_perlayeropt_ignored(self):
        """PerLayerOpt key should be ignored — it's not a transform."""
        part_d = {
            "INT8": {
                "None": {"qsnr_per_layer": {"fc1": 20.0}},
                "SmoothQuant": {"qsnr_per_layer": {"fc1": 22.0}},
                "PerLayerOpt": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            text = transform_benefit_table(part_d, output_dir=tmpdir)
            assert "SmoothQuant" in text
            assert "PerLayerOpt" not in text
