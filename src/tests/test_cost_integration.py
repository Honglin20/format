"""Integration tests for cost model with quantize_model and run_quantization."""
import copy

import pytest
import torch
import torch.nn as nn

from src.session import QuantConfig
from src.session._model import quantize_model
from src.session._session import run_quantization
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.formats.base import FormatBase
from src.cost.model_cost import analyze_model_cost


@pytest.fixture
def int8_cfg():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)


def test_quantized_estimate_cost_quantized(int8_cfg):
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    qmodel = quantize_model(copy.deepcopy(model), int8_cfg)

    cost_q = analyze_model_cost(qmodel)
    assert cost_q.total_latency_us > 0
    assert cost_q.total_memory_bytes > 0
    assert len(cost_q.layers) >= 2


def test_quantized_estimate_cost_fp32(int8_cfg):
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    fp32_model = copy.deepcopy(model)

    cost_fp32 = analyze_model_cost(fp32_model)
    assert cost_fp32.total_latency_us > 0
    assert cost_fp32.total_memory_bytes > 0


def test_quantized_model_has_less_weight_memory(int8_cfg):
    """INT8 quantized weights should use less memory than FP32."""
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32)
        def forward(self, x): return self.fc(x)

    qmodel = quantize_model(Wrapper(), int8_cfg)
    fp32_model = Wrapper()

    cost_q = analyze_model_cost(qmodel)
    cost_fp32 = analyze_model_cost(fp32_model)

    assert cost_q.total_memory_bytes < cost_fp32.total_memory_bytes


def test_estimate_cost_no_fp32_model_ok(int8_cfg):
    """analyze_model_cost works on quantized model alone."""
    model = nn.Linear(64, 10)
    qmodel = quantize_model(model, int8_cfg)

    cost = analyze_model_cost(qmodel)
    assert cost.total_latency_us > 0


# ── run_quantization cost integration tests ──────────────────────────

def _dummy_eval_fn(model, data):
    model.eval()
    with torch.no_grad():
        if isinstance(data, (list, tuple)):
            for batch in data:
                model(batch)
        else:
            model(data)
    return {"accuracy": 0.9}


def test_run_quantization_attaches_cost_keys():
    """run_quantization() returns result with cost and cost_fp32."""
    cfg = QuantConfig(name="cfg1", w_format="int8", w_granularity="per_tensor")
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

    qmodel, fp32_model, result = run_quantization(
        model, cfg,
        calib_data=[torch.randn(2, 64)],
        eval_data=torch.randn(2, 64),
        eval_fn=_dummy_eval_fn,
        outputs="all",
    )
    assert result.cost is not None
    assert result.cost_fp32 is not None
    assert result.cost.total_latency_us > 0
    assert result.cost_fp32.total_latency_us > 0


def test_run_quantization_cost_present():
    """cost keys are present in result."""
    cfg = QuantConfig(name="cfg1", w_format="int8", w_granularity="per_tensor")
    model = nn.Sequential(nn.Linear(64, 32))

    qmodel, fp32_model, result = run_quantization(
        model, cfg,
        calib_data=[torch.randn(2, 64)],
        eval_data=torch.randn(2, 64),
        eval_fn=_dummy_eval_fn,
        outputs="all",
    )
    assert result.cost is not None
    assert result.cost_fp32 is not None
    assert result.cost.total_latency_us > 0
