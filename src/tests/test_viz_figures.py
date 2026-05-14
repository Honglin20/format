"""Tests for src/viz/figures.py."""
import tempfile
import pytest
import matplotlib
matplotlib.use("Agg")

from src.viz._helpers import _compute_best_transform_per_layer
from src.viz.figures import (
    qsnr_line_chart,
    mse_box_plot,
    pot_delta_bar,
    histogram_overlay,
    transform_heatmap,
    transform_pie,
    transform_delta,
    error_vs_distribution,
    layer_type_qsnr,
    block_sweep_line_chart,
    hierarchical_delta_bar,
    outlier_analysis,
    per_block_qsnr,
    correlation_heatmap,
    role_distribution_comparison,
    kurtosis_analysis,
    _get_acc_val,
)


class TestHelpers:
    def test_get_acc_val_dict(self):
        data = {"accuracy": {"accuracy": 0.85}}
        assert _get_acc_val(data) == 0.85

    def test_get_acc_val_float(self):
        data = {"accuracy": 0.75}
        assert _get_acc_val(data) == 0.75

    def test_get_acc_val_empty(self):
        import math
        assert math.isnan(_get_acc_val({}))
        assert math.isnan(_get_acc_val(None))

    def test_compute_best_transform_per_layer(self):
        variant_qsnr = {
            "None": {"fc1": 10.0, "fc2": 12.0},
            "SmoothQuant": {"fc1": 15.0, "fc2": 11.0},
        }
        result = _compute_best_transform_per_layer(variant_qsnr)
        assert result["fc1"] == "SmoothQuant"
        assert result["fc2"] == "None"


class TestQSNRBarChart:
    def test_renders_without_error(self):
        results = {
            "MXINT-8": {"qsnr_per_layer": {"fc1": 20.0, "fc2": 18.0}},
            "MXFP-8":  {"qsnr_per_layer": {"fc1": 22.0, "fc2": 19.0}},
        }
        colors = {"MXINT-8": "#0072B2", "MXFP-8": "#D55E00"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = qsnr_line_chart(results, title="Test QSNR", colors=colors, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0

    def test_skips_baseline(self):
        results = {
            "baseline": {"qsnr_per_layer": {"fc1": 30.0}},
            "MXINT-8": {"qsnr_per_layer": {"fc1": 20.0}},
        }
        colors = {"MXINT-8": "#0072B2"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = qsnr_line_chart(results, title="Skip Baseline", colors=colors, output_dir=tmpdir)
            assert fig is not None
            # Should only have one line (baseline skipped)

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No QSNR data available"):
                qsnr_line_chart({}, title="Empty", colors={}, output_dir=tmpdir)

    def test_qsnr_aligns_by_shared_layer_names(self):
        """Different configs should use the same x position for the same layer."""
        results = {
            "A": {"qsnr_per_layer": {"fc1": 10.0, "fc2": 12.0}},
            "B": {"qsnr_per_layer": {"fc2": 15.0, "fc1": 8.0}},  # different order
        }
        colors = {"A": "#0072B2", "B": "#D55E00"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = qsnr_line_chart(results, title="Test", colors=colors, output_dir=tmpdir)
            assert fig is not None
            # Both configs should have the same number of x points (2 shared layers)
            ax = fig.axes[0]
            # Each line has exactly 2 points
            assert len(ax.lines) == 2
            for line in ax.lines:
                assert len(line.get_xdata()) == 2
                assert len(line.get_ydata()) == 2

    def test_line_values_reflect_input(self):
        """Line data points must match the QSNR values from the input dict."""
        results = {
            "A": {"qsnr_per_layer": {"fc1": 10.0, "fc2": 12.0}},
        }
        colors = {"A": "#0072B2"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = qsnr_line_chart(results, title="Test", colors=colors, output_dir=tmpdir)
            assert fig is not None
            ax = fig.axes[0]
            ydata = list(ax.lines[0].get_ydata())
            # Both values present and non-NaN
            assert len(ydata) == 2
            for y in ydata:
                assert y == pytest.approx(10.0) or y == pytest.approx(12.0)


class TestMSEBoxPlot:
    def test_renders_without_error(self):
        results = {
            "MXINT-8": {"mse_per_layer": {"fc1": 0.01, "fc2": 0.02}},
            "MXFP-8":  {"mse_per_layer": {"fc1": 0.005, "fc2": 0.015}},
        }
        colors = {"MXINT-8": "#0072B2", "MXFP-8": "#D55E00"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = mse_box_plot(results, title="Test MSE", colors=colors, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0

    def test_renders_no_data(self):
        results = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No MSE data available"):
                mse_box_plot(results, title="Empty MSE", colors={}, output_dir=tmpdir)

    def test_skips_baseline(self):
        results = {
            "baseline": {"mse_per_layer": {"fc1": 0.001}},
            "MXINT-8": {"mse_per_layer": {"fc1": 0.01}},
        }
        colors = {"MXINT-8": "#0072B2"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = mse_box_plot(results, title="Skip Baseline MSE", colors=colors, output_dir=tmpdir)
            assert fig is not None


class TestPoTDeltaBar:
    def test_renders_without_error(self):
        part_c = {
            "INT8-PC-FP32": {"qsnr_per_layer": {"fc1": 20.0, "fc2": 18.0}},
            "INT8-PC-PoT": {"qsnr_per_layer": {"fc1": 22.0, "fc2": 17.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = pot_delta_bar(part_c, output_dir=tmpdir)
            assert fig is not None

    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No PoT scaling data"):
                pot_delta_bar({}, output_dir=tmpdir)


class TestHistogramOverlay:
    def test_renders_no_data_message(self):
        """When no histogram data exists, a ValueError is raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Histogram data not available"):
                histogram_overlay({}, output_dir=tmpdir)

    def test_renders_with_qsnr_ranking(self):
        """When no data, raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Histogram data not available"):
                histogram_overlay({}, output_dir=tmpdir)


class TestSessionPlotHistogramOverlay:
    """Tests for SessionPlotAccessor.histogram_overlay()."""

    @staticmethod
    def _make_result(observers_data=None, qsnr_by_role=None):
        from src.session._result import SessionResult

        return SessionResult(
            name="test",
            config=None,
            observers_data=observers_data or {},
            qsnr_by_role=qsnr_by_role or {},
        )

    @staticmethod
    def _make_hist_metrics(fp32_hist, quant_hist, err_hist=None):
        """Helper to construct HistogramObserver-style metrics dict."""
        import numpy as np

        d = {
            "fp32_hist": np.asarray(fp32_hist, dtype=np.float64),
            "quant_hist": np.asarray(quant_hist, dtype=np.float64),
        }
        if err_hist is not None:
            d["err_hist"] = np.asarray(err_hist, dtype=np.float64)
        return d

    def _make_observers_data(self, layers_roles_metrics):
        """Build observers_data from [(layer, role, metrics), ...] tuples."""
        obs = {}
        for layer, role, metrics in layers_roles_metrics:
            obs.setdefault(layer, {}).setdefault(role, {}).setdefault(
                "post", {}
            )["default"] = metrics
        return obs

    def test_empty_data_raises(self):
        """No histogram data → ValueError with actionable message."""
        result = self._make_result()
        with pytest.raises(ValueError, match="Histogram data not available"):
            result.plot.histogram_overlay()

    def test_renders_with_qsnr_ranking(self):
        """Histogram data + QSNR → renders top-k most sensitive."""
        import numpy as np

        hist1 = np.array([0, 10, 50, 30, 10, 0], dtype=np.float64)
        hist2 = np.array([0, 5, 40, 60, 20, 5], dtype=np.float64)
        err1 = np.abs(hist1 - np.array([0, 10, 50, 30, 10, 0])) * 0.1
        err2 = np.abs(hist2 - np.array([0, 5, 40, 60, 20, 5])) * 0.2

        observers_data = self._make_observers_data([
            ("layer_a", "input", self._make_hist_metrics(hist1, hist1 * 0.95, err1)),
            ("layer_b", "output", self._make_hist_metrics(hist2, hist2 * 0.85, err2)),
        ])

        # layer_a has better QSNR, layer_b has worse QSNR → layer_b should be first
        qsnr_by_role = {
            "input": {"layer_a": 25.0},
            "output": {"layer_b": 10.0},
        }

        result = self._make_result(observers_data, qsnr_by_role)
        fig = result.plot.histogram_overlay(top_k=2)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 2
        # Worst layer first (lowest QSNR = 10.0)
        assert "layer_b" in axes[0].get_title()
        assert "layer_a" in axes[1].get_title()

    def test_fallback_no_qsnr(self):
        """No QSNR data → ranks by fp32 histogram magnitude (largest first)."""
        import numpy as np

        hist_small = np.array([0, 1, 2, 1, 0], dtype=np.float64)
        hist_large = np.array([0, 100, 500, 300, 50], dtype=np.float64)

        observers_data = self._make_observers_data([
            ("small_layer", "input", self._make_hist_metrics(hist_small, hist_small * 0.9)),
            ("large_layer", "weight", self._make_hist_metrics(hist_large, hist_large * 0.88)),
        ])

        result = self._make_result(observers_data)
        fig = result.plot.histogram_overlay(top_k=2)

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 2
        # Largest activation first (hist_large.sum() > hist_small.sum())
        assert "large_layer" in axes[0].get_title()
        assert "small_layer" in axes[1].get_title()

    def test_respects_top_k(self):
        """top_k limits the number of subplots."""
        import numpy as np

        hist = np.array([0, 10, 50, 30, 10, 0], dtype=np.float64)
        layers = [
            ("L1", "input", self._make_hist_metrics(hist, hist * 0.92)),
            ("L2", "weight", self._make_hist_metrics(hist, hist * 0.88)),
            ("L3", "output", self._make_hist_metrics(hist, hist * 0.85)),
        ]
        observers_data = self._make_observers_data(layers)
        qsnr_by_role = {
            "input": {"L1": 30.0},
            "weight": {"L2": 20.0},
            "output": {"L3": 10.0},
        }

        result = self._make_result(observers_data, qsnr_by_role)
        fig = result.plot.histogram_overlay(top_k=2)

        axes = fig.get_axes()
        assert len(axes) == 2
        # L3 (QSNR=10.0) worst, L2 (QSNR=20.0) second worst
        assert "L3" in axes[0].get_title()
        assert "L2" in axes[1].get_title()

    def test_role_filter_single_role(self):
        """role='weight' only collects weight histograms."""
        import numpy as np

        hist = np.array([0, 10, 50, 30, 10, 0], dtype=np.float64)
        observers_data = self._make_observers_data([
            ("L1", "input", self._make_hist_metrics(hist, hist * 0.90)),
            ("L1", "weight", self._make_hist_metrics(hist, hist * 0.80)),
            ("L1", "output", self._make_hist_metrics(hist, hist * 0.70)),
        ])
        qsnr_by_role = {
            "input": {"L1": 30.0},
            "weight": {"L1": 20.0},
            "output": {"L1": 10.0},
        }

        result = self._make_result(observers_data, qsnr_by_role)
        fig = result.plot.histogram_overlay(role="weight")

        axes = fig.get_axes()
        assert len(axes) == 1
        assert "[weight]" in axes[0].get_title()

    def test_role_filter_no_matching_data(self):
        """role with no histogram data raises ValueError."""
        import numpy as np

        hist = np.array([0, 10, 50, 30, 10, 0], dtype=np.float64)
        observers_data = self._make_observers_data([
            ("L1", "input", self._make_hist_metrics(hist, hist * 0.90)),
        ])
        qsnr_by_role = {"input": {"L1": 30.0}}

        result = self._make_result(observers_data, qsnr_by_role)
        with pytest.raises(ValueError, match="Histogram data not available"):
            result.plot.histogram_overlay(role="weight")


class TestTransformHeatmap:
    def test_renders_without_error(self):
        part_d = {
            "MXINT-8": {
                "None":         {"accuracy": {"accuracy": 0.85}},
                "SmoothQuant":  {"accuracy": {"accuracy": 0.87}},
            },
            "MXFP-8": {
                "None":        {"accuracy": {"accuracy": 0.82}},
                "Hadamard":    {"accuracy": {"accuracy": 0.84}},
            },
        }
        colors = {"None": "#0072B2", "SmoothQuant": "#D55E00", "Hadamard": "#009E73"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = transform_heatmap(part_d, colors=colors, output_dir=tmpdir)
            assert fig is not None

    def test_partial_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No transform study data"):
                transform_heatmap({}, colors={}, output_dir=tmpdir)

    def test_cell_values_match_input(self):
        """Heatmap annotations should reflect the underlying accuracy data."""
        part_d = {
            "INT8": {
                "None":         {"accuracy": {"accuracy": 0.85}},
                "SmoothQuant":  {"accuracy": {"accuracy": 0.92}},
            },
        }
        colors = {"None": "#0072B2", "SmoothQuant": "#D55E00"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = transform_heatmap(part_d, colors=colors, output_dir=tmpdir)
            assert fig is not None
            ax = fig.axes[0]
            # Verify axes are labelled with correct format/transform names
            x_labels = [t.get_text() for t in ax.get_xticklabels()]
            y_labels = [t.get_text() for t in ax.get_yticklabels()]
            assert "None" in x_labels
            assert "SmoothQuant" in x_labels
            assert "INT8" in y_labels


class TestTransformPie:
    def test_renders_without_error(self):
        part_d = {
            "MXINT-8": {
                "PerLayerOpt": True,
                "None": {"qsnr_per_layer": {"fc1": 10.0, "fc2": 12.0}},
                "SmoothQuant": {"qsnr_per_layer": {"fc1": 15.0, "fc2": 11.0}},
            },
        }
        colors = {"None": "#0072B2", "SmoothQuant": "#D55E00", "Hadamard": "#009E73"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = transform_pie(part_d, colors=colors, output_dir=tmpdir)
            assert fig is not None

    def test_no_perlayeropt_data(self):
        part_d = {"MXINT-8": {"None": {"qsnr_per_layer": {"fc1": 10.0}}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = transform_pie(part_d, colors={}, output_dir=tmpdir)
            assert fig is not None

    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No format study data"):
                transform_pie({}, colors={}, output_dir=tmpdir)

    def test_percentages_are_positive(self):
        """All pie wedge percentages should be non-negative."""
        part_d = {
            "MXINT-8": {
                "PerLayerOpt": True,
                "None": {"qsnr_per_layer": {"fc1": 10.0}},
                "SmoothQuant": {"qsnr_per_layer": {"fc1": 15.0}},
                "Hadamard": {"qsnr_per_layer": {"fc1": 12.0}},
            },
        }
        colors = {"None": "#0072B2", "SmoothQuant": "#D55E00", "Hadamard": "#009E73"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = transform_pie(part_d, colors=colors, output_dir=tmpdir)
            assert fig is not None
            ax = fig.axes[0]
            # All wedge widths should be non-negative
            for wedge in ax.patches:
                assert wedge.theta2 - wedge.theta1 >= 0


class TestTransformDelta:
    def test_renders_without_error(self):
        part_d = {
            "MXINT-8": {
                "None": {"qsnr_per_layer": {"fc1": 10.0, "fc2": 12.0}},
                "SmoothQuant": {"qsnr_per_layer": {"fc1": 15.0, "fc2": 11.0}},
            },
        }
        colors = {"None": "#0072B2", "SmoothQuant": "#D55E00", "Hadamard": "#009E73"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = transform_delta(part_d, colors=colors, output_dir=tmpdir)
            assert fig is not None

    def test_no_baseline(self):
        part_d = {"MXINT-8": {"SmoothQuant": {"qsnr_per_layer": {"fc1": 15.0}}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = transform_delta(part_d, colors={}, output_dir=tmpdir)
            assert fig is not None

    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No transform delta data"):
                transform_delta({}, colors={}, output_dir=tmpdir)


class TestErrorVsDistribution:
    def test_renders_no_data_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Distribution data not available"):
                error_vs_distribution({}, output_dir=tmpdir)


class TestLayerTypeQSNR:
    def test_renders_no_data_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Layer type data not available"):
                layer_type_qsnr({}, output_dir=tmpdir)

    def test_single_layer_type_falls_back_to_per_layer_chart(self):
        """When all layers are the same type, fall back to qsnr_line_chart."""
        from unittest.mock import MagicMock, PropertyMock
        # Create a mock report where LayerSensitivity only sees Linear layers
        mock_report = MagicMock()
        type(mock_report)._raw = PropertyMock(return_value={
            "fc1": {"weight": {"weight_pre_quant": {"0": {"mse": 0.01, "qsnr_db": 20.0}}}},
            "fc2": {"weight": {"weight_pre_quant": {"0": {"mse": 0.02, "qsnr_db": 18.0}}}},
        })
        results = {
            "part_a": {"INT8-PT": {"report": mock_report}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = layer_type_qsnr(results, output_dir=tmpdir)
            assert fig is not None
            # Should produce a line chart (single axes), not boxplot pair
            assert len(fig.axes) == 1


class TestBlockSweepLineChart:
    def test_renders_without_error(self):
        block_sweep = {
            "int8-blk16": {"qsnr_per_layer": {"fc1": 20.0, "fc2": 18.0}},
            "int8-blk32": {"qsnr_per_layer": {"fc1": 22.0, "fc2": 19.0}},
            "int8-blk64": {"qsnr_per_layer": {"fc1": 23.0, "fc2": 20.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = block_sweep_line_chart(block_sweep, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0

    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No block sweep data"):
                block_sweep_line_chart({}, output_dir=tmpdir)

    def test_skips_baseline(self):
        block_sweep = {
            "FP32 (baseline)": {"qsnr_per_layer": {"fc1": 30.0}},
            "int8-blk32": {"qsnr_per_layer": {"fc1": 20.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = block_sweep_line_chart(block_sweep, output_dir=tmpdir)
            assert fig is not None


class TestHierarchicalDeltaBar:
    def test_renders_without_error(self):
        hierarchical = {
            "MXINT-8-HIER": {"qsnr_per_layer": {"fc1": 25.0, "fc2": 23.0}},
            "MXFP-8-HIER": {"qsnr_per_layer": {"fc1": 24.0, "fc2": 22.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = hierarchical_delta_bar(hierarchical, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0

    def test_skips_baseline(self):
        hierarchical = {
            "FP32 (baseline)": {"qsnr_per_layer": {"fc1": 30.0}},
            "MXINT-8-HIER": {"qsnr_per_layer": {"fc1": 25.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = hierarchical_delta_bar(hierarchical, output_dir=tmpdir)
            assert fig is not None

    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No hierarchical study data"):
                hierarchical_delta_bar({}, output_dir=tmpdir)


# ── Helpers for iter_slices-based figure tests ──────────────────────────

class _MockReport:
    """Minimal mock report with iter_slices for figure tests."""

    def __init__(self, slices):
        self._slices = slices

    def iter_slices(self):
        for item in self._slices:
            yield item


def _make_all_results(slices):
    """Wrap a single report in the all_results dict."""
    return {"part_a": {"cfg1": {"report": _MockReport(slices)}}}


# ── P0.1: Outlier Analysis ──────────────────────────────────────────────

class TestOutlierAnalysis:
    def test_renders_without_error(self):
        slices = [
            ("layer1", "input", "pre", ("tensor",), {"outlier_ratio": 0.03, "qsnr_db": 30.0}),
            ("layer1", "weight", "pre", ("tensor",), {"outlier_ratio": 0.01, "qsnr_db": 42.0}),
            ("layer2", "input", "pre", ("tensor",), {"outlier_ratio": 0.05, "qsnr_db": 25.0}),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = outlier_analysis(all_results, output_dir=tmpdir, roles=("input",))
            assert fig is not None

    def test_empty_data(self):
        slices = [("l", "weight", "pre", ("tensor",), {"qsnr_db": 30.0})]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Outlier ratio data not available"):
                outlier_analysis(all_results, output_dir=tmpdir, roles=("input",))

    def test_no_slices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Outlier ratio data not available"):
                outlier_analysis({}, output_dir=tmpdir)


# ── P0.2: Per-Block QSNR ────────────────────────────────────────────────

class TestPerBlockQSNR:
    def test_renders_without_error(self):
        slices = [
            ("layer1", "input", "pre", ("tensor",), {
                "qsnr_db": 30.0, "qsnr_db_std": 2.0,
                "qsnr_db_min": 25.0, "qsnr_db_max": 35.0,
            }),
            ("layer2", "input", "pre", ("tensor",), {
                "qsnr_db": 28.0, "qsnr_db_std": 1.5,
                "qsnr_db_min": 24.0, "qsnr_db_max": 32.0,
            }),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = per_block_qsnr(all_results, output_dir=tmpdir, roles=("input",))
            assert fig is not None

    def test_empty_data(self):
        slices = [("l", "input", "pre", ("tensor",), {"qsnr_db": 30.0})]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Per-block QSNR statistics not available"):
                per_block_qsnr(all_results, output_dir=tmpdir, roles=("input",))


# ── P1.5: Correlation Heatmap ───────────────────────────────────────────

class TestCorrelationHeatmap:
    def test_renders_without_error(self):
        slices = [
            ("l1", "input", "pre", ("tensor",), {
                "crest_factor": 2.0, "skewness": 0.3, "kurtosis": 3.2,
                "sparse_ratio": 0.05, "dynamic_range_bits": 6.0,
                "outlier_ratio": 0.02, "norm_entropy": 0.75, "qsnr_db": 30.0,
            }),
            ("l2", "input", "pre", ("tensor",), {
                "crest_factor": 3.0, "skewness": 0.8, "kurtosis": 4.5,
                "sparse_ratio": 0.12, "dynamic_range_bits": 4.5,
                "outlier_ratio": 0.06, "norm_entropy": 0.62, "qsnr_db": 25.0,
            }),
            ("l3", "input", "pre", ("tensor",), {
                "crest_factor": 1.5, "skewness": -0.1, "kurtosis": 2.9,
                "sparse_ratio": 0.01, "dynamic_range_bits": 7.0,
                "outlier_ratio": 0.01, "norm_entropy": 0.85, "qsnr_db": 35.0,
            }),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = correlation_heatmap(all_results, output_dir=tmpdir)
            assert fig is not None

    def test_insufficient_features(self):
        slices = [("l1", "input", "pre", ("tensor",), {"crest_factor": 2.0})]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Insufficient distribution feature"):
                correlation_heatmap(all_results, output_dir=tmpdir)

    def test_empty_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Insufficient distribution feature"):
                correlation_heatmap({}, output_dir=tmpdir)


# ── P1.7: Role Distribution Comparison ──────────────────────────────────

class TestRoleDistributionComparison:
    def test_renders_without_error(self):
        slices = [
            ("l1", "input", "pre", ("tensor",), {"skewness": 0.3, "kurtosis": 3.2, "norm_entropy": 0.75}),
            ("l1", "weight", "pre", ("tensor",), {"skewness": -0.2, "kurtosis": 2.8, "norm_entropy": 0.82}),
            ("l1", "output", "pre", ("tensor",), {"skewness": 0.1, "kurtosis": 3.1, "norm_entropy": 0.70}),
            ("l2", "input", "pre", ("tensor",), {"skewness": 0.8, "kurtosis": 4.2, "norm_entropy": 0.65}),
            ("l2", "weight", "pre", ("tensor",), {"skewness": 0.5, "kurtosis": 3.8, "norm_entropy": 0.72}),
            ("l2", "output", "pre", ("tensor",), {"skewness": 0.2, "kurtosis": 3.3, "norm_entropy": 0.68}),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = role_distribution_comparison(all_results, output_dir=tmpdir)
            assert fig is not None

    def test_custom_roles(self):
        slices = [
            ("l1", "input", "pre", ("tensor",), {"skewness": 0.3, "kurtosis": 3.2, "norm_entropy": 0.75}),
            ("l1", "weight", "pre", ("tensor",), {"skewness": -0.2, "kurtosis": 2.8, "norm_entropy": 0.82}),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = role_distribution_comparison(all_results, output_dir=tmpdir,
                                               roles=("input", "weight"))
            assert fig is not None

    def test_empty_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Distribution data not available"):
                role_distribution_comparison({}, output_dir=tmpdir)

    def test_no_matching_roles(self):
        slices = [("l1", "bias", "pre", ("tensor",), {"skewness": 0.0, "kurtosis": 3.0, "norm_entropy": 0.5})]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No data found for roles"):
                role_distribution_comparison(all_results, output_dir=tmpdir)


# ── Kurtosis Analysis ────────────────────────────────────────────────────

class TestKurtosisAnalysis:
    def test_renders_without_error(self):
        slices = [
            ("l1", "input", "pre", ("tensor",), {"kurtosis": 3.2, "excess_kurtosis": 0.2, "qsnr_db": 30.0}),
            ("l1", "weight", "pre", ("tensor",), {"kurtosis": 2.9, "excess_kurtosis": -0.1, "qsnr_db": 42.0}),
            ("l1", "output", "pre", ("tensor",), {"kurtosis": 4.1, "excess_kurtosis": 1.1, "qsnr_db": 28.0}),
            ("l2", "input", "pre", ("tensor",), {"kurtosis": 8.5, "excess_kurtosis": 5.5, "qsnr_db": 18.0}),
            ("l2", "weight", "pre", ("tensor",), {"kurtosis": 3.1, "excess_kurtosis": 0.1, "qsnr_db": 38.0}),
            ("l2", "output", "pre", ("tensor",), {"kurtosis": 5.2, "excess_kurtosis": 2.2, "qsnr_db": 22.0}),
            ("l3", "input", "pre", ("tensor",), {"kurtosis": 2.5, "excess_kurtosis": -0.5, "qsnr_db": 35.0}),
            ("l3", "weight", "pre", ("tensor",), {"kurtosis": 3.0, "excess_kurtosis": 0.0, "qsnr_db": 40.0}),
            ("l3", "output", "pre", ("tensor",), {"kurtosis": 12.0, "excess_kurtosis": 9.0, "qsnr_db": 12.0}),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = kurtosis_analysis(all_results, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) == 3

    def test_custom_roles(self):
        slices = [
            ("l1", "input", "pre", ("tensor",), {"kurtosis": 3.2, "qsnr_db": 30.0}),
            ("l1", "weight", "pre", ("tensor",), {"kurtosis": 2.9, "qsnr_db": 42.0}),
        ]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = kurtosis_analysis(all_results, output_dir=tmpdir,
                                    roles=("input",))
            assert fig is not None

    def test_empty_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Kurtosis data not available"):
                kurtosis_analysis({}, output_dir=tmpdir)

    def test_no_kurtosis(self):
        slices = [("l1", "input", "pre", ("tensor",), {"qsnr_db": 30.0})]
        all_results = _make_all_results(slices)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Kurtosis data not available"):
                kurtosis_analysis(all_results, output_dir=tmpdir)
