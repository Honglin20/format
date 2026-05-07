"""Tests for the legacy pipeline runner replaced by Session/Study.

The old ExperimentRunner is replaced by ``Session`` + ``SessionResult``.
These tests verify that the new API covers the same use cases.
"""
import copy

import pytest
import torch
import torch.nn as nn

from src.session import QuantConfig, Session, SessionResult


class TinyModel(nn.Sequential):
    """Single Linear layer wrapped in Sequential for quantize_model compatibility."""

    def __init__(self):
        super().__init__(nn.Linear(4, 3))


class TestSessionResults:
    def test_session_returns_session_result(self):
        """Session.run() returns a SessionResult with expected keys."""
        model = nn.Sequential(nn.Linear(4, 3))
        model[0].weight.data.fill_(0.5)
        model[0].bias.data.fill_(0.0)

        cfg = QuantConfig(w_format="int8", w_granularity="per_channel", name="int8_pc")
        session = Session(model, cfg)

        calib_data = [torch.randn(2, 4)]

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean_output": sum(results) / len(results)}
                out = m(data)
            return {"mean_output": out.mean().item()}

        result = session.run(
            calib_data=calib_data,
            eval_data=torch.randn(2, 4),
            eval_fn=_eval_fn,
        )

        assert isinstance(result, SessionResult)
        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None
        assert result.delta is not None

    def test_original_model_not_mutated_when_copied(self):
        """When the caller passes a deep copy, the original remains nn.Linear."""
        model = nn.Sequential(nn.Linear(4, 3))

        cfg = QuantConfig(w_format="int8", w_granularity="per_tensor")
        model_copy = copy.deepcopy(model)
        session = Session(model_copy, cfg)

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean_output": sum(results) / len(results)}
                out = m(data)
            return {"mean_output": out.mean().item()}

        session.run(
            calib_data=[torch.randn(2, 4)],
            eval_data=torch.randn(2, 4),
            eval_fn=_eval_fn,
        )

        # Original model should still be unquantized (nn.Linear, not QuantizedLinear)
        assert isinstance(model[0], nn.Linear), "Original model was mutated"
