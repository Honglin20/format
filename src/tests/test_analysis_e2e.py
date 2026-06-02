"""Tests for src/analysis/e2e.py — Comparator, compare_models, compare_sessions."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.e2e import Comparator, compare_models, compare_sessions, _default_accuracy


# ===================================================================
# _default_accuracy
# ===================================================================

class TestDefaultAccuracy:
    def test_perfect_prediction(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([0, 1])
        result = _default_accuracy(logits, labels)
        assert result["accuracy"] == 1.0

    def test_wrong_prediction(self):
        logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1])
        result = _default_accuracy(logits, labels)
        assert result["accuracy"] == 0.0

    def test_mixed(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1])
        result = _default_accuracy(logits, labels)
        assert result["accuracy"] == 0.5


# ===================================================================
# Comparator
# ===================================================================

class TestComparator:
    def test_context_manager_clears_state(self):
        cmp = Comparator()
        with cmp:
            cmp.record(torch.randn(2, 3), torch.randn(2, 3), torch.tensor([0, 1]))
        # After exit, entering again should start clean
        with cmp:
            assert cmp.num_samples == 0

    def test_record_and_num_samples(self):
        cmp = Comparator()
        with cmp:
            cmp.record(torch.randn(3, 5), torch.randn(3, 5), torch.tensor([1, 2, 3]))
            cmp.record(torch.randn(2, 5), torch.randn(2, 5), torch.tensor([0, 1]))
            assert cmp.num_samples == 5

    def test_evaluate_returns_structure(self):
        cmp = Comparator()
        with cmp:
            cmp.record(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                torch.tensor([0, 1]),
            )

        def acc_fn(logits, labels):
            return {"accuracy": (logits.argmax(-1) == labels).float().mean().item()}

        result = cmp.evaluate(acc_fn)
        assert "fp32" in result
        assert "quant" in result
        assert "delta" in result
        assert result["fp32"]["accuracy"] == 1.0

    def test_delta_computed_correctly(self):
        cmp = Comparator()
        with cmp:
            # fp32: logits[0] > logits[1] → class 0 → correct for label 0
            # quant: logits[1] > logits[0] → class 1 → wrong for label 0
            cmp.record(
                torch.tensor([[10.0, 0.0]]),
                torch.tensor([[0.0, 10.0]]),
                torch.tensor([0]),
            )

        def acc_fn(logits, labels):
            return {"accuracy": (logits.argmax(-1) == labels).float().mean().item()}

        result = cmp.evaluate(acc_fn)
        assert result["fp32"]["accuracy"] == 1.0
        assert result["quant"]["accuracy"] == 0.0
        assert result["delta"]["accuracy"] == -1.0

    def test_device_transfer(self):
        cmp = Comparator(device=torch.device("cpu"))
        with cmp:
            cmp.record(torch.randn(2, 3), torch.randn(2, 3), torch.tensor([0, 1]))

        def acc_fn(logits, labels):
            return {"accuracy": (logits.argmax(-1) == labels).float().mean().item()}

        result = cmp.evaluate(acc_fn)
        assert "fp32" in result


# ===================================================================
# compare_models
# ===================================================================

class TestCompareModels:
    def test_auto_mode_runs(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        fp32_model = SimpleModel()
        qmodel = SimpleModel()
        qmodel.load_state_dict(fp32_model.state_dict())

        data = torch.randn(8, 4)
        labels = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)

        result = compare_models(fp32_model, qmodel, loader)
        assert "fp32" in result
        assert "quant" in result
        assert "delta" in result

    def test_custom_eval_fn(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        data = torch.randn(4, 4)
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)

        def custom_eval(logits, labels):
            return {"my_metric": logits.sum().item()}

        result = compare_models(model, model, loader, eval_fn=custom_eval)
        assert "my_metric" in result["fp32"]
        assert result["delta"]["my_metric"] == 0.0


# ===================================================================
# compare_sessions
# ===================================================================

class TestCompareSessions:
    def test_returns_per_session_results(self):
        """compare_sessions runs one pass and returns per-session metrics."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        fp32_model = TinyModel()
        qmodel = TinyModel()
        qmodel.load_state_dict(fp32_model.state_dict())

        data = torch.randn(8, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)

        results = compare_sessions(fp32_model, {"a": qmodel, "b": qmodel}, loader)
        assert "fp32" in results
        assert "a" in results
        assert "b" in results

    def test_fp32_label_customizable(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        fp32_model = TinyModel()
        qmodel = TinyModel()
        qmodel.load_state_dict(fp32_model.state_dict())

        data = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)

        results = compare_sessions(fp32_model, {"x": qmodel}, loader, fp32_label="baseline")
        assert "baseline" in results
        assert "fp32" not in results
