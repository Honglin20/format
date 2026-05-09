"""Tests for src/analysis/eval_performance.py — PerformanceReport, evaluate_performance."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.eval_performance import PerformanceReport, evaluate_performance


# ===================================================================
# PerformanceReport
# ===================================================================

class TestPerformanceReport:
    def test_summary_includes_baseline(self):
        baseline = {"accuracy": 0.95, "loss": 0.2}
        quantized = {"q8": {"accuracy": 0.90, "loss": 0.25}}
        report = PerformanceReport(baseline, quantized)

        s = report.summary()
        assert "fp32_baseline" in s
        assert "q8" in s

    def test_delta_computed(self):
        baseline = {"accuracy": 0.95}
        quantized = {"q8": {"accuracy": 0.90}}
        report = PerformanceReport(baseline, quantized)

        s = report.summary()
        assert "delta_accuracy" in s["q8"]
        assert s["q8"]["delta_accuracy"] == pytest.approx(-0.05)

    def test_to_dataframe(self):
        baseline = {"accuracy": 0.95}
        quantized = {"q8": {"accuracy": 0.90}}
        report = PerformanceReport(baseline, quantized)

        df = report.to_dataframe()
        assert len(df) == 2  # baseline + q8
        if hasattr(df, "columns"):
            assert "model" in df.columns

    def test_print_summary_no_crash(self):
        baseline = {"accuracy": 0.95, "loss": 0.2}
        quantized = {"a": {"accuracy": 0.90, "loss": 0.3}, "b": {"accuracy": 0.88, "loss": 0.35}}
        report = PerformanceReport(baseline, quantized)
        report.print_summary()

    def test_empty_quantized(self):
        baseline = {"accuracy": 0.8}
        report = PerformanceReport(baseline, {})
        report.print_summary()


# ===================================================================
# evaluate_performance
# ===================================================================

class TestEvaluatePerformance:
    def test_evaluates_and_returns_report(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        fp32 = TinyModel()
        q = TinyModel()
        q.load_state_dict(fp32.state_dict())

        data = torch.randn(8, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)

        def eval_fn(model, dl):
            correct = 0
            total = 0
            for x, y in dl:
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.numel()
            return {"accuracy": correct / total}

        report = evaluate_performance(fp32, {"quant": q}, loader, eval_fn, device="cpu")
        assert isinstance(report, PerformanceReport)
        s = report.summary()
        assert "fp32_baseline" in s
        assert "quant" in s

    def test_requires_at_least_one_quantized_model(self):
        with pytest.raises(ValueError, match="at least one"):
            evaluate_performance(
                nn.Linear(2, 2), {}, DataLoader(TensorDataset(torch.randn(4, 2))),
                lambda m, dl: {"x": 0.0},
            )
