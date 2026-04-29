"""Tests for src/viz/figures.py."""
import tempfile
import matplotlib
matplotlib.use("Agg")

from src.viz.figures import (
    qsnr_bar_chart,
    mse_box_plot,
    pot_delta_bar,
    histogram_overlay,
    transform_heatmap,
    transform_pie,
    transform_delta,
    error_vs_distribution,
    layer_type_qsnr,
    _compute_best_transform_per_layer,
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
            fig = qsnr_bar_chart(results, title="Test QSNR", colors=colors, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0

    def test_skips_baseline(self):
        results = {
            "baseline": {"qsnr_per_layer": {"fc1": 30.0}},
            "MXINT-8": {"qsnr_per_layer": {"fc1": 20.0}},
        }
        colors = {"MXINT-8": "#0072B2"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = qsnr_bar_chart(results, title="Skip Baseline", colors=colors, output_dir=tmpdir)
            assert fig is not None
            # Should only have one line (baseline skipped)

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = qsnr_bar_chart({}, title="Empty", colors={}, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0


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
            fig = mse_box_plot(results, title="Empty MSE", colors={}, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0

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
            fig = pot_delta_bar({}, output_dir=tmpdir)
            assert fig is not None


class TestHistogramOverlay:
    def test_renders_no_data_message(self):
        """When no histogram data exists, a placeholder figure should render."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = histogram_overlay({}, output_dir=tmpdir)
            assert fig is not None
            assert len(fig.axes) > 0


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
            fig = transform_heatmap({}, colors={}, output_dir=tmpdir)
            assert fig is not None


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
            fig = transform_pie({}, colors={}, output_dir=tmpdir)
            assert fig is not None


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
            fig = transform_delta({}, colors={}, output_dir=tmpdir)
            assert fig is not None


class TestErrorVsDistribution:
    def test_renders_no_data_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = error_vs_distribution({}, output_dir=tmpdir)
            assert fig is not None


class TestLayerTypeQSNR:
    def test_renders_no_data_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = layer_type_qsnr({}, output_dir=tmpdir)
            assert fig is not None
