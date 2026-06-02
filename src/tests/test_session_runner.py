"""Tests for the run_quantization replacement for Session results.

The old Session class has been replaced by run_quantization().
These tests verify the new API covers the same use cases.
"""
import copy

import pytest
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._result import SessionResult
from src.session._session import run_quantization


class TinyModel(nn.Sequential):
    """Single Linear layer wrapped in Sequential for quantize_model compatibility."""

    def __init__(self):
        super().__init__(nn.Linear(4, 3))


class TestRunQuantizationResults:
    def test_run_quantization_returns_session_result(self):
        """run_quantization() returns a SessionResult with expected keys."""
        model = nn.Sequential(nn.Linear(4, 3))
        model[0].weight.data.fill_(0.5)
        model[0].bias.data.fill_(0.0)

        cfg = QuantConfig(w_format="int8", w_granularity="per_channel", name="int8_pc")

        calib_data = [torch.randn(2, 4)]

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean_output": sum(results) / len(results)}
                out = m(data)
            return {"mean_output": out.mean().item()}

        qmodel, fp32_model, result = run_quantization(
            model, cfg, calib_data,
            eval_data=torch.randn(2, 4),
            eval_fn=_eval_fn,
        )

        assert isinstance(result, SessionResult)
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None
        assert result.delta is not None

    def test_original_model_not_mutated_when_copied(self):
        """When run_quantization is called, the original remains nn.Linear."""
        model = nn.Sequential(nn.Linear(4, 3))

        cfg = QuantConfig(w_format="int8", w_granularity="per_tensor")

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean_output": sum(results) / len(results)}
                out = m(data)
            return {"mean_output": out.mean().item()}

        qmodel, fp32_model, result = run_quantization(
            model, cfg, [torch.randn(2, 4)],
            eval_data=torch.randn(2, 4),
            eval_fn=_eval_fn,
        )

        # Original model should still be unquantized (nn.Linear, not QuantizedLinear)
        assert isinstance(model[0], nn.Linear), "Original model was mutated"
