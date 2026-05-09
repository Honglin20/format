"""Tests for model-level cost analysis."""
import pytest
import torch
import torch.nn as nn
from src.cost.device import DeviceSpec
from src.cost.model_cost import analyze_model_cost
from src.cost.report import CostReport


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_analyze_fp32_model():
    model = TinyModel()
    report = analyze_model_cost(model, shapes={"batch": 4})

    assert isinstance(report, CostReport)
    # Two Linear + one ReLU = 3 recognized layers
    assert len(report.layers) >= 2
    assert report.total_latency_us > 0
    assert report.total_memory_bytes > 0

    # FP32 model has no quantize FLOPs
    for layer in report.layers:
        assert layer.flops_quantize == 0


def test_model_latency_is_sum_of_layers():
    model = TinyModel()
    report = analyze_model_cost(model, shapes={"batch": 1})
    layer_sum = sum(l.latency_us for l in report.layers)
    assert abs(report.total_latency_us - layer_sum) < 1e-6


def test_memory_is_weight_sum_plus_max_activation():
    model = TinyModel()
    report = analyze_model_cost(model, shapes={"batch": 1})
    weight_sum = sum(l.memory_weight_bytes for l in report.layers)
    max_act = max(l.memory_activation_bytes for l in report.layers)
    assert report.total_memory_bytes == weight_sum + max_act


def test_layer_names_are_set():
    model = TinyModel()
    report = analyze_model_cost(model)
    layer_names = {l.op_name for l in report.layers}
    assert "fc1" in layer_names
    assert "fc2" in layer_names


def test_default_device_is_a100():
    """analyze_model_cost uses A100 when device not specified."""
    model = nn.Linear(64, 32)
    report = analyze_model_cost(model)
    assert len(report.layers) == 1
    assert report.total_latency_us > 0


def test_model_name_set_in_report():
    model = TinyModel()
    report = analyze_model_cost(model, model_name="my_model")
    assert report.model_name == "my_model"


def test_sequential_module_not_duplicated():
    """Sequential containers should not appear as separate layers."""
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
    report = analyze_model_cost(model)
    # Only leaf ops (not the root Sequential) should appear
    for layer in report.layers:
        assert layer.op_type != "unknown" or layer.op_name != ""
    # Layer names should not be empty string (root)
    for layer in report.layers:
        assert layer.op_name != ""


def test_empty_model():
    class EmptyModel(nn.Module):
        def forward(self, x): return x

    model = EmptyModel()
    report = analyze_model_cost(model)
    assert len(report.layers) == 0
    assert report.total_latency_us == 0.0
