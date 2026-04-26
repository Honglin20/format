import torch
import torch.nn as nn
import pytest

from src.analysis.eval_performance import evaluate_performance, PerformanceReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """A trivial 2-layer model for eval testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _dummy_dataloader(n_batches=3, batch_size=4):
    """Returns a list of (x, y) batches simulating a DataLoader."""
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, 4)
        y = torch.randint(0, 2, (batch_size,))
        batches.append((x, y))
    return batches


def _cls_eval_fn(model, dataloader):
    """Simple classification eval: returns accuracy and dummy loss."""
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0, "loss": 0.5}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvaluatePerformance:
    def test_basic_evaluation(self):
        fp32 = _TinyModel()
        quantized = {"fp8": _TinyModel()}
        loader = _dummy_dataloader()

        report = evaluate_performance(fp32, quantized, loader, _cls_eval_fn)

        assert isinstance(report, PerformanceReport)
        assert "accuracy" in report.baseline
        assert "loss" in report.baseline
        assert "fp8" in report.quantized
        assert "accuracy" in report.quantized["fp8"]

    def test_summary_includes_deltas(self):
        fp32 = _TinyModel()
        quantized = {"fp8": _TinyModel()}
        loader = _dummy_dataloader()

        report = evaluate_performance(fp32, quantized, loader, _cls_eval_fn)
        summary = report.summary()

        assert "fp32_baseline" in summary
        assert "fp8" in summary
        assert "delta_accuracy" in summary["fp8"]
        assert "delta_loss" in summary["fp8"]
        # delta = quantized - baseline
        assert abs(summary["fp8"]["delta_accuracy"]
                   - (summary["fp8"]["accuracy"] - summary["fp32_baseline"]["accuracy"])) < 1e-9

    def test_to_dataframe(self):
        fp32 = _TinyModel()
        quantized = {"fp8": _TinyModel()}
        loader = _dummy_dataloader()

        report = evaluate_performance(fp32, quantized, loader, _cls_eval_fn)
        df = report.to_dataframe()

        assert len(df) == 2  # baseline + fp8
        assert "model" in df
        assert "accuracy" in df

    def test_print_summary_does_not_crash(self):
        fp32 = _TinyModel()
        quantized = {"fp8": _TinyModel()}
        loader = _dummy_dataloader()

        report = evaluate_performance(fp32, quantized, loader, _cls_eval_fn)
        report.print_summary()

    def test_multiple_quantized_models(self):
        fp32 = _TinyModel()
        quantized = {"fp8": _TinyModel(), "int8": _TinyModel()}
        loader = _dummy_dataloader()

        report = evaluate_performance(fp32, quantized, loader, _cls_eval_fn)

        assert len(report.quantized) == 2
        assert "fp8" in report.quantized
        assert "int8" in report.quantized

    def test_empty_quantized_raises(self):
        fp32 = _TinyModel()
        loader = _dummy_dataloader()

        with pytest.raises(ValueError, match="at least one"):
            evaluate_performance(fp32, {}, loader, _cls_eval_fn)

    def test_custom_metric_fn(self):
        fp32 = _TinyModel()
        quantized = {"q": _TinyModel()}
        loader = _dummy_dataloader()

        def custom_eval(model, dl):
            return {"f1_score": 0.87, "perplexity": 12.3}

        report = evaluate_performance(fp32, quantized, loader, custom_eval)
        assert "f1_score" in report.baseline
        assert "perplexity" in report.baseline
        assert "f1_score" in report.quantized["q"]
