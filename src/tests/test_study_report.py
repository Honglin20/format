"""Tests for src.report — SessionReport, StudyReport, and plot accessor."""

import json
import os
import tempfile

import pytest

from src.report._session_report import SessionReport
from src.report._study_report import StudyReport
from src.session._config import QuantConfig
from src.session._session import SessionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_result_a() -> SessionResult:
    return SessionResult(
        name="int8",
        config=QuantConfig(name="int8", w_format="int8", a_format="int8",
                           w_granularity="per_tensor", a_granularity="per_tensor"),
        quant_metrics={"accuracy": 0.92, "f1": 0.90},
        fp32_metrics={"accuracy": 0.95, "f1": 0.93},
        delta={"accuracy": -0.03, "f1": -0.03},
        qsnr_per_layer={"layer_1": 28.5, "layer_2": 32.1, "layer_3": 25.8},
        mse_per_layer={"layer_1": 0.0012, "layer_2": 0.0008, "layer_3": 0.0021},
        observers_data={},
    )


@pytest.fixture
def mock_result_b() -> SessionResult:
    return SessionResult(
        name="int4",
        config=QuantConfig(name="int4", w_format="int4", a_format="int4",
                           w_granularity="per_tensor", a_granularity="per_tensor"),
        quant_metrics={"accuracy": 0.85, "f1": 0.82},
        fp32_metrics={"accuracy": 0.95, "f1": 0.93},
        delta={"accuracy": -0.10, "f1": -0.11},
        qsnr_per_layer={"layer_1": 22.0, "layer_2": 25.5, "layer_3": 20.1},
        mse_per_layer={"layer_1": 0.0050, "layer_2": 0.0030, "layer_3": 0.0080},
        observers_data={},
    )


@pytest.fixture
def mock_result_no_metrics() -> SessionResult:
    return SessionResult(
        name="empty",
        config=QuantConfig(name="empty"),
    )


@pytest.fixture
def mock_result_with_observers() -> SessionResult:
    """Mock result with merged observer data (QSNR + Distribution)."""
    return SessionResult(
        name="int8",
        config=QuantConfig(name="int8", w_format="int8", w_granularity="per_tensor"),
        observers_data={
            "layer_1": {
                "input": {
                    "pre_quant[0]": {
                        ("tensor",): {
                            "qsnr_db": 35.2,
                            "peak": 3.5, "rms": 1.8, "crest_factor": 1.94,
                            "mean": 0.1, "std": 1.2,
                        }
                    }
                },
                "weight": {
                    "pre_quant[0]": {
                        ("tensor",): {
                            "qsnr_db": 42.1,
                            "peak": 2.1, "rms": 0.9, "crest_factor": 2.33,
                        }
                    }
                },
            },
            "layer_2": {
                "input": {
                    "pre_quant[0]": {
                        ("tensor",): {
                            "qsnr_db": 30.5,
                            "peak": 5.2, "rms": 2.1, "crest_factor": 2.48,
                        }
                    }
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# Test SessionReport
# ---------------------------------------------------------------------------

class TestSessionReport:

    def test_construction(self, mock_result_a):
        report = SessionReport(mock_result_a)
        assert report.result.name == "int8"

    def test_metrics_table_with_quant_metrics(self, mock_result_a):
        report = SessionReport(mock_result_a)
        table = report.metrics_table()
        assert "int8" in table
        assert "0.92" in table
        assert "accuracy" in table

    def test_metrics_table_no_metrics(self, mock_result_no_metrics):
        report = SessionReport(mock_result_no_metrics)
        table = report.metrics_table()
        assert "empty" in table

    def test_to_dataframe(self, mock_result_a):
        report = SessionReport(mock_result_a)
        df = report.to_dataframe()
        assert isinstance(df, list)
        assert len(df) == 3  # 3 layers
        for row in df:
            assert "layer" in row
            assert "qsnr" in row
            assert "mse" in row
        layers = {r["layer"] for r in df}
        assert layers == {"layer_1", "layer_2", "layer_3"}

    def test_to_dataframe_no_data(self, mock_result_no_metrics):
        report = SessionReport(mock_result_no_metrics)
        df = report.to_dataframe()
        assert df == []

    def test_save_creates_file(self, mock_result_a):
        report = SessionReport(mock_result_a)
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "metrics.txt"))


# ---------------------------------------------------------------------------
# Test StudyReport
# ---------------------------------------------------------------------------

class TestStudyReport:

    def test_construction_empty(self):
        report = StudyReport({})
        assert report.parts == []
        assert report.total_experiments == 0

    def test_construction_single_part(self, mock_result_a):
        report = StudyReport({"part_1": [mock_result_a]})
        assert report.parts == ["part_1"]
        assert report.total_experiments == 1

    def test_construction_multiple_parts(self, mock_result_a, mock_result_b):
        report = StudyReport({
            "part_1": [mock_result_a],
            "part_2": [mock_result_b],
        })
        assert sorted(report.parts) == ["part_1", "part_2"]
        assert report.total_experiments == 2

    def test_print_summary_empty(self, capsys):
        report = StudyReport({})
        report.print_summary()
        captured = capsys.readouterr()
        assert "(no results)" not in captured.out

    def test_print_summary(self, mock_result_a, capsys):
        report = StudyReport({"part_1": [mock_result_a]})
        report.print_summary()
        captured = capsys.readouterr()
        assert "Part: part_1" in captured.out
        assert "int8" in captured.out
        # Verify computed averages from _avg_qsnr_mse
        # qsnr: (28.5 + 32.1 + 25.8) / 3 = 28.80
        # mse:   (0.0012 + 0.0008 + 0.0021) / 3 = 0.001367
        assert "28.80" in captured.out
        assert "0.001367" in captured.out

    def test_print_summary_with_delta(self, mock_result_a, capsys):
        report = StudyReport({"part_1": [mock_result_a]})
        report.print_summary()
        captured = capsys.readouterr()
        assert "accuracy=-0.0300" in captured.out

    def test_to_serializable(self, mock_result_a):
        report = StudyReport({"part_1": [mock_result_a]})
        data = report.to_serializable()
        assert "part_1" in data
        assert "int8" in data["part_1"]
        assert data["part_1"]["int8"]["accuracy"] == {"accuracy": 0.92, "f1": 0.90}

    def test_to_serializable_is_json_serializable(self, mock_result_a, mock_result_b):
        report = StudyReport({"part_1": [mock_result_a, mock_result_b]})
        data = report.to_serializable()
        json_str = json.dumps(data, indent=2, default=str)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["part_1"]["int8"]["accuracy"]["accuracy"] == 0.92

    def test_to_serializable_empty(self):
        report = StudyReport({})
        assert report.to_serializable() == {}

    def test_save_creates_directories(self, mock_result_a):
        report = StudyReport({"part_1": [mock_result_a]})
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            assert os.path.isdir(os.path.join(tmpdir, "tables"))
            assert os.path.isdir(os.path.join(tmpdir, "figures"))

    def test_save_creates_results_json(self, mock_result_a):
        report = StudyReport({"part_1": [mock_result_a]})
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            json_path = os.path.join(tmpdir, "results.json")
            assert os.path.isfile(json_path)
            with open(json_path) as f:
                data = json.load(f)
            assert "part_1" in data

    def test_save_creates_accuracy_csv_when_eval_present(self, mock_result_a):
        report = StudyReport({"part_1": [mock_result_a]})
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "accuracy.csv")
            assert os.path.isfile(csv_path)

    def test_save_skips_accuracy_csv_when_no_eval(self, mock_result_no_metrics):
        report = StudyReport({"part_1": [mock_result_no_metrics]})
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            csv_path = os.path.join(tmpdir, "tables", "accuracy.csv")
            assert not os.path.isfile(csv_path)

    def test_save_creates_figures_with_observers(self, mock_result_with_observers):
        report = StudyReport({"p1": [mock_result_with_observers]})
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            figures_dir = os.path.join(tmpdir, "figures")
            assert os.path.isdir(figures_dir)
            assert os.path.isfile(os.path.join(figures_dir, "qsnr_comparison.png"))
            assert os.path.isfile(os.path.join(figures_dir, "crest_vs_qsnr_input.png"))
            assert os.path.isfile(os.path.join(figures_dir, "crest_vs_qsnr_weight.png"))

    def test_save_empty_results(self):
        report = StudyReport({})
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "results.json"))

    def test_from_file_round_trip(self, mock_result_a, mock_result_b):
        original = StudyReport({"part_1": [mock_result_a, mock_result_b]})
        with tempfile.TemporaryDirectory() as tmpdir:
            original.save(tmpdir)
            reloaded = StudyReport.from_file(tmpdir)
            assert isinstance(reloaded, StudyReport)
            assert sorted(reloaded.parts) == ["part_1"]
            assert reloaded.to_serializable() == original.to_serializable()

    def test_from_file_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            StudyReport.from_file("/nonexistent/path")

    def test_properties(self, mock_result_a, mock_result_b):
        parts = {
            "a": [mock_result_a],
            "b": [mock_result_a, mock_result_b],
        }
        report = StudyReport(parts)
        assert sorted(report.parts) == ["a", "b"]
        assert report.total_experiments == 3

    # ── to_dataframe ──────────────────────────────────────────────────

    def test_to_dataframe_empty(self):
        report = StudyReport({})
        df = report.to_dataframe()
        assert df is not None
        assert df.empty

    def test_to_dataframe_no_observers(self, mock_result_a):
        report = StudyReport({"p1": [mock_result_a]})
        df = report.to_dataframe()
        # No observers_data → empty
        assert df is not None
        assert df.empty

    def test_to_dataframe_with_observers(self, mock_result_with_observers):
        report = StudyReport({"p1": [mock_result_with_observers]})
        df = report.to_dataframe()
        assert df is not None
        assert len(df) > 0
        assert "part" in df.columns
        assert "config" in df.columns
        assert "format" in df.columns
        assert "layer" in df.columns
        assert "role" in df.columns
        assert "qsnr_db" in df.columns
        assert "crest_factor" in df.columns
        # 3 rows: layer_1/input, layer_1/weight, layer_2/input
        assert len(df) == 3

    def test_to_dataframe_leading_columns(self, mock_result_with_observers):
        report = StudyReport({"p1": [mock_result_with_observers]})
        df = report.to_dataframe()
        # Leading columns are part/config/format/layer/role
        assert list(df.columns[:5]) == ["part", "config", "format", "layer", "role"]

    # ── plot accessor ─────────────────────────────────────────────────

    def test_plot_accessor_returns_accessor(self, mock_result_a):
        report = StudyReport({"p1": [mock_result_a]})
        from src.report._plot import StudyPlotAccessor
        assert isinstance(report.plot, StudyPlotAccessor)

    def test_qsnr_comparison_smoke(self, mock_result_with_observers):
        report = StudyReport({"p1": [mock_result_with_observers]})
        fig = report.plot.qsnr_comparison()
        assert fig is not None

    def test_qsnr_comparison_no_data(self, mock_result_a):
        report = StudyReport({"p1": [mock_result_a]})
        fig = report.plot.qsnr_comparison()
        assert fig is not None  # graceful fallback

    def test_crest_vs_qsnr_smoke(self, mock_result_with_observers):
        report = StudyReport({"p1": [mock_result_with_observers]})
        fig = report.plot.crest_vs_qsnr(role="input")
        assert fig is not None

    def test_crest_vs_qsnr_no_crest_data(self, mock_result_a):
        report = StudyReport({"p1": [mock_result_a]})
        fig = report.plot.crest_vs_qsnr(role="input")
        assert fig is not None  # graceful fallback

    def test_crest_vs_qsnr_missing_role(self, mock_result_with_observers):
        report = StudyReport({"p1": [mock_result_with_observers]})
        fig = report.plot.crest_vs_qsnr(role="output")
        assert fig is not None  # graceful fallback
