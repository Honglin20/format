"""Integration tests for cost model with QuantSession and pipeline."""
import pytest
import torch
import torch.nn as nn
from src.session import QuantSession
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.formats.base import FormatBase


@pytest.fixture
def int8_cfg():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)


def test_session_estimate_cost_quantized(int8_cfg):
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    session = QuantSession(model, int8_cfg, keep_fp32=True)

    cost_q = session.estimate_cost()
    assert cost_q.total_latency_us > 0
    assert cost_q.total_memory_bytes > 0
    assert len(cost_q.layers) >= 2


def test_session_estimate_cost_fp32(int8_cfg):
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    session = QuantSession(model, int8_cfg, keep_fp32=True)

    cost_fp32 = session.estimate_cost(fp32=True)
    assert cost_fp32.total_latency_us > 0
    assert cost_fp32.total_memory_bytes > 0


def test_quantized_model_has_less_weight_memory(int8_cfg):
    """INT8 quantized weights should use less memory than FP32.

    quantize_model replaces named child modules, so we must wrap the
    Linear in a container to ensure replacement happens (root is skipped).
    """
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32)
        def forward(self, x): return self.fc(x)

    session = QuantSession(Wrapper(), int8_cfg, keep_fp32=True)

    cost_q = session.estimate_cost()
    cost_fp32 = session.estimate_cost(fp32=True)

    # INT8 weight memory (2048 B) < FP32 weight memory (8192 B)
    assert cost_q.total_memory_bytes < cost_fp32.total_memory_bytes


def test_session_estimate_cost_no_fp32_raises(int8_cfg):
    model = nn.Linear(64, 10)
    session = QuantSession(model, int8_cfg, keep_fp32=False)

    with pytest.raises(RuntimeError, match="fp32_model"):
        session.estimate_cost(fp32=True)


def test_session_estimate_cost_quantized_no_fp32_ok(int8_cfg):
    """estimate_cost() (quantized) works even when keep_fp32=False."""
    model = nn.Linear(64, 10)
    session = QuantSession(model, int8_cfg, keep_fp32=False)

    cost = session.estimate_cost()  # fp32=False (default)
    assert cost.total_latency_us > 0


# ── ExperimentRunner cost integration ────────────────────────────

def _dummy_eval_fn(model, data):
    return {"accuracy": 0.9}


def test_runner_attaches_cost_keys(int8_cfg):
    from src.pipeline.runner import ExperimentRunner

    search_space = {"test_part": {"configs": {"cfg1": int8_cfg}}}
    runner = ExperimentRunner(search_space)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

    results = runner.run(model, eval_fn=_dummy_eval_fn)
    entry = results["test_part/cfg1"]
    assert "cost" in entry
    assert "cost_fp32" in entry
    assert entry["cost"].total_latency_us > 0
    assert entry["cost_fp32"].total_latency_us > 0


def test_runner_cost_keys_present_even_without_analysis(int8_cfg):
    """cost and cost_fp32 are present even when analysis is skipped."""
    from src.pipeline.runner import ExperimentRunner

    search_space = {"test_part": {"configs": {"cfg1": int8_cfg}}}
    runner = ExperimentRunner(search_space)
    model = nn.Sequential(nn.Linear(64, 32))

    results = runner.run(model, eval_fn=_dummy_eval_fn)
    entry = results["test_part/cfg1"]
    assert "cost" in entry
    assert "cost_fp32" in entry
    assert entry["cost"].total_latency_us > 0
