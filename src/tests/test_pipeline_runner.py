import pytest
import torch
import torch.nn as nn
from src.pipeline.runner import ExperimentRunner, ExperimentResult
from src.pipeline.config import resolve_config


class TinyModel(nn.Sequential):
    """Single Linear layer wrapped in Sequential for quantize_model compatibility."""

    def __init__(self):
        super().__init__(nn.Linear(4, 3))


def _make_tiny_study():
    return {
        "int8_test": {
            "description": "tiny test",
            "configs": [
                {"name": "int8_pc", "format": "int8", "granularity": "per_channel", "axis": 0},
            ],
        },
    }


class TestExperimentRunner:
    def test_runner_returns_expected_keys(self):
        model = nn.Sequential(nn.Linear(4, 3))
        model[0].weight.data.fill_(0.5)
        model[0].bias.data.fill_(0.0)

        study = _make_tiny_study()
        runner = ExperimentRunner(study)

        calib_data = [torch.randn(2, 4)]

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean_output": sum(results) / len(results)}
                out = m(data)
            return {"mean_output": out.mean().item()}

        results = runner.run(
            fp32_model=model,
            eval_fn=_eval_fn,
            calib_data=calib_data,
            eval_data=torch.randn(2, 4),
        )

        assert "int8_test" in results
        part_results = results["int8_test"]
        assert len(part_results) == 1
        r = part_results[0]
        assert isinstance(r, ExperimentResult)
        assert r.fp32_metrics is not None
        assert r.quant_metrics is not None
        assert r.delta is not None

    def test_runner_deepcopies_model(self):
        model = nn.Sequential(nn.Linear(4, 3))
        study = _make_tiny_study()
        runner = ExperimentRunner(study)

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean_output": sum(results) / len(results)}
                out = m(data)
            return {"mean_output": out.mean().item()}

        runner.run(
            fp32_model=model,
            eval_fn=_eval_fn,
            calib_data=[torch.randn(2, 4)],
            eval_data=torch.randn(2, 4),
        )

        # Original model should still be unquantized (nn.Linear, not QuantizedLinear)
        assert isinstance(model[0], nn.Linear), "Original model was mutated"
