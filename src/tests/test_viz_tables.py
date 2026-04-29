"""Tests for src.viz.tables."""
import os
import tempfile
from src.viz.tables import accuracy_table


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
