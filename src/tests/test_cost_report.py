"""Tests for CostReport."""
import pytest
from src.cost.report import CostReport
from src.cost.op_cost import OpCost


def make_linear_cost(name="fc1", latency=10.0, mem_w=1000, mem_a=500):
    return OpCost(
        op_name=name, op_type="linear",
        flops_math=1000, latency_us=latency,
        memory_weight_bytes=mem_w, memory_activation_bytes=mem_a,
    )


def test_report_aggregates_latency():
    layers = [make_linear_cost("fc1", 10.0), make_linear_cost("fc2", 20.0)]
    report = CostReport(layers=layers)
    assert report.total_latency_us == 30.0


def test_report_aggregates_memory():
    layers = [
        make_linear_cost("fc1", mem_w=1000, mem_a=500),
        make_linear_cost("fc2", mem_w=2000, mem_a=800),
    ]
    report = CostReport(layers=layers)
    # weight sum + max activation
    assert report.total_memory_bytes == (1000 + 2000) + max(500, 800)


def test_report_to_dataframe():
    layers = [make_linear_cost("fc1"), make_linear_cost("fc2")]
    report = CostReport(layers=layers, model_name="test")
    df = report.to_dataframe()
    # Returns pandas DataFrame if available, else list of dicts
    try:
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df["op_name"].iloc[0] == "fc1"
    except ImportError:
        assert isinstance(df, list)
        assert len(df) == 2
        assert df[0]["op_name"] == "fc1"


def test_report_print_summary_runs():
    layers = [make_linear_cost("fc1")]
    report = CostReport(layers=layers, model_name="test")
    report.print_summary()  # should not raise


def test_print_comparison_runs():
    fp32 = CostReport([make_linear_cost("fc1", latency=5.0)], model_name="FP32")
    quant = CostReport([make_linear_cost("fc1", latency=8.0)], model_name="INT8")
    quant.print_comparison(fp32)  # should not raise


def test_report_empty_layers():
    report = CostReport(layers=[])
    assert report.total_latency_us == 0.0
    assert report.total_memory_bytes == 0


def test_summary_dict_keys():
    layers = [make_linear_cost("fc1")]
    report = CostReport(layers=layers, model_name="mymodel")
    s = report.summary()
    assert s["model"] == "mymodel"
    assert "total_latency_us" in s
    assert "total_memory_mb" in s
    assert "total_flops_math" in s
