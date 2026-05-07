"""Tests for src.report — SessionReport, StudyReport, and converters."""

import json
import os
import tempfile

import pytest

from src.report._converters import (
    extract_metric_per_layer,
    results_to_combined_viz_dict,
    results_to_nested_viz_dict,
    results_to_viz_dict,
)
from src.report._session_report import SessionReport
from src.report._study_report import StudyReport
from src.session._config import QuantConfig
from src.session._session import SessionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_result_a() -> SessionResult:
    """A mock SessionResult with metric/observers/cost."""
    return SessionResult(
        name="int8",
        config=QuantConfig(name="int8", w_format="int8", a_format="int8",
                           w_granularity="per_tensor", a_granularity="per_tensor"),
        quant_metrics={"accuracy": 0.92, "f1": 0.90},
        fp32_metrics={"accuracy": 0.95, "f1": 0.93},
        delta={"accuracy": -0.03, "f1": -0.03},
        qsnr_per_layer={"layer_1": 28.5, "layer_2": 32.1, "layer_3": 25.8},
        mse_per_layer={"layer_1": 0.0012, "layer_2": 0.0008, "layer_3": 0.0021},
        observers_data={"histogram": {"layer_1": {"fp32_hist": [1]}}},
        cost=100.0,
        cost_fp32=200.0,
    )


@pytest.fixture
def mock_result_b() -> SessionResult:
    """A second mock SessionResult with heavier quantization."""
    return SessionResult(
        name="int4",
        config=QuantConfig(name="int4", w_format="int4", a_format="int4",
                           w_granularity="per_tensor", a_granularity="per_tensor"),
        quant_metrics={"accuracy": 0.85, "f1": 0.82},
        fp32_metrics={"accuracy": 0.95, "f1": 0.93},
        delta={"accuracy": -0.10, "f1": -0.11},
        qsnr_per_layer={"layer_1": 22.0, "layer_2": 25.5, "layer_3": 20.1},
        mse_per_layer={"layer_1": 0.0050, "layer_2": 0.0030, "layer_3": 0.0080},
    )


@pytest.fixture
def mock_result_no_metrics() -> SessionResult:
    """A SessionResult with minimal fields (no metrics)."""
    return SessionResult(
        name="empty",
        config=QuantConfig(name="empty"),
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
            # Should create metrics.txt
            assert os.path.exists(os.path.join(tmpdir, "metrics.txt"))


# ---------------------------------------------------------------------------
# Test converters
# ---------------------------------------------------------------------------

class TestResultsToVizDict:

    def test_single_result(self, mock_result_a):
        result = results_to_viz_dict([mock_result_a])
        assert "int8" in result
        assert result["int8"]["accuracy"] == {"accuracy": 0.92, "f1": 0.90}
        assert "qsnr_per_layer" in result["int8"]
        assert "mse_per_layer" in result["int8"]
        assert "delta" in result["int8"]
        assert result["int8"]["fp32_accuracy"] == {"accuracy": 0.95, "f1": 0.93}

    def test_multiple_results(self, mock_result_a, mock_result_b):
        result = results_to_viz_dict([mock_result_a, mock_result_b])
        assert len(result) == 2
        assert "int8" in result
        assert "int4" in result

    def test_result_no_metrics(self, mock_result_no_metrics):
        result = results_to_viz_dict([mock_result_no_metrics])
        assert "empty" in result
        assert result["empty"] == {}

    def test_empty_list(self):
        result = results_to_viz_dict([])
        assert result == {}


class TestResultsToNestedVizDict:

    def test_nested_structure(self, mock_result_a, mock_result_b):
        configs = [
            {"name": "int8", "transform": "none"},
            {"name": "int4", "transform": "none"},
        ]
        result = results_to_nested_viz_dict([mock_result_a, mock_result_b], configs)
        assert "int8" in result
        assert "int4" in result
        assert "None" in result["int8"]
        assert result["int8"]["None"]["accuracy"] == {"accuracy": 0.92, "f1": 0.90}

    def test_transform_labels(self):
        had_result = SessionResult(
            name="int8-Had",
            config=QuantConfig(name="int8-Had", transform="hadamard"),
            qsnr_per_layer={"l1": 30.0},
        )
        configs = [{"name": "int8-Had", "transform": "hadamard"}]
        result = results_to_nested_viz_dict([had_result], configs)
        assert "int8" in result
        assert "Hadamard" in result["int8"]

    def test_empty_results(self):
        result = results_to_nested_viz_dict([], [])
        assert result == {}


class TestResultsToCombinedVizDict:

    def test_single_part_single_result(self, mock_result_a):
        combined = results_to_combined_viz_dict({"part1": [mock_result_a]})
        assert "part1" in combined
        assert "int8" in combined["part1"]
        assert combined["part1"]["int8"]["accuracy"] == {"accuracy": 0.92, "f1": 0.90}

    def test_multiple_parts(self, mock_result_a, mock_result_b):
        combined = results_to_combined_viz_dict({
            "part1": [mock_result_a],
            "part2": [mock_result_b],
        })
        assert "part1" in combined
        assert "part2" in combined
        assert "int8" in combined["part1"]
        assert "int4" in combined["part2"]

    def test_empty_results(self):
        combined = results_to_combined_viz_dict({})
        assert combined == {}

    def test_empty_part_list(self):
        combined = results_to_combined_viz_dict({"part1": []})
        assert "part1" in combined
        assert combined["part1"] == {}


class TestExtractMetricPerLayer:

    def test_extract_qsnr(self, mock_result_a):
        report = SessionReport(mock_result_a)
        result = extract_metric_per_layer(report, "qsnr")
        assert result == {"layer_1": 28.5, "layer_2": 32.1, "layer_3": 25.8}

    def test_extract_mse(self, mock_result_a):
        report = SessionReport(mock_result_a)
        result = extract_metric_per_layer(report, "mse")
        assert result == {"layer_1": 0.0012, "layer_2": 0.0008, "layer_3": 0.0021}

    def test_extract_no_data(self, mock_result_no_metrics):
        report = SessionReport(mock_result_no_metrics)
        result = extract_metric_per_layer(report, "qsnr")
        assert result == {}


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
        assert "(no results)" not in captured.out  # no parts, nothing printed

    def test_print_summary(self, mock_result_a, capsys):
        report = StudyReport({"part_1": [mock_result_a]})
        report.print_summary()
        captured = capsys.readouterr()
        assert "Part: part_1" in captured.out
        assert "int8" in captured.out
        assert "28.80" in captured.out  # avg_qsnr = (28.5+32.1+25.8)/3
        assert "0.001367" in captured.out  # avg_mse = (0.0012+0.0008+0.0021)/3

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
        assert "qsnr_per_layer" in data["part_1"]["int8"]

    def test_to_serializable_is_json_serializable(self, mock_result_a, mock_result_b):
        report = StudyReport({"part_1": [mock_result_a, mock_result_b]})
        data = report.to_serializable()
        # Should not raise
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

    def test_save_with_config(self, mock_result_a):
        report = StudyReport({"part_1": [mock_result_a]})
        config = {
            "part_1": {
                "output": {
                    "tables": ["accuracy"],
                    "figures": ["qsnr"],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir, config=config)
            assert os.path.isfile(os.path.join(tmpdir, "results.json"))

    def test_save_empty_results(self):
        report = StudyReport({})
        with tempfile.TemporaryDirectory() as tmpdir:
            report.save(tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "results.json"))

    def test_from_file_round_trip(self, mock_result_a, mock_result_b):
        """StudyReport.from_file round-trips correctly."""
        original = StudyReport({"part_1": [mock_result_a, mock_result_b]})
        with tempfile.TemporaryDirectory() as tmpdir:
            original.save(tmpdir)
            # Reload from results.json
            reloaded = StudyReport.from_file(tmpdir)
            assert isinstance(reloaded, StudyReport)
            assert sorted(reloaded.parts) == ["part_1"]
            # Check serializable form matches
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

    def test_handles_observers_data(self, mock_result_a):
        """Observers data doesn't affect StudyReport methods."""
        report = StudyReport({"p1": [mock_result_a]})
        data = report.to_serializable()
        assert "p1" in data
