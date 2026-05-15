"""Tests for Study aggregation layer.

Study is pure aggregation - zero quantization logic.
These tests mock run_quantization() to isolate Study behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import SessionResult
from src.session._study import Study


class _TwoLayerModel(nn.Module):
    """Minimal 2-layer Linear model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def model():
    return _TwoLayerModel()


@pytest.fixture
def configs():
    return [
        QuantConfig(name="int8", w_format="int8", a_format="int8"),
        QuantConfig(name="int4", w_format="int4", a_format="int4"),
    ]


@pytest.fixture
def mock_run_quantization():
    """Return a factory that creates a predictable SessionResult for a config."""

    def _make(config: QuantConfig) -> SessionResult:
        return SessionResult(
            name=config.name,
            config=config,
            qsnr_per_layer={"fc1": 20.0, "fc2": 15.0},
        )
    return _make


class TestStudyConstruction:
    """Study.__init__ stores configs and model."""

    def test_stores_configs_and_model(self, model, configs):
        study = Study(configs, model=model)
        assert study._configs == configs
        assert study._model is model

    def test_empty_configs(self, model):
        study = Study([], model=model)
        assert study._configs == []
        assert study._model is model


class TestStudyRun:
    """Study.run() delegates to run_quantization and returns StudyReport."""

    @patch("src.session._study.run_quantization")
    def test_single_config_returns_study_report(
        self, mock_rq, model, configs, mock_run_quantization,
    ):
        cfg = configs[0]
        mock_rq.return_value = (None, None, mock_run_quantization(cfg))

        study = Study([cfg], model=model)
        report = study.run(torch.randn(2, 4))

        assert report is not None
        assert report.total_experiments == 1
        assert report.parts == ["int8"]

    @patch("src.session._study.run_quantization")
    def test_multiple_configs_multiple_entries(
        self, mock_rq, model, configs, mock_run_quantization,
    ):
        mock_rq.side_effect = [
            (None, None, mock_run_quantization(cfg)) for cfg in configs
        ]

        study = Study(configs, model=model)
        report = study.run(torch.randn(2, 4))

        assert report.total_experiments == 2
        assert set(report.parts) == {"int8", "int4"}

    @patch("src.session._study.run_quantization")
    def test_model_factory_called_per_config(
        self, mock_rq, model, configs, mock_run_quantization,
    ):
        mock_rq.return_value = (None, None, mock_run_quantization(configs[0]))

        factory_calls = []

        def model_factory(cfg):
            factory_calls.append(cfg.name)
            return _TwoLayerModel()

        study = Study(configs, model=model)
        study.run(torch.randn(2, 4), model_factory=model_factory)

        assert len(factory_calls) == 2
        assert factory_calls == ["int8", "int4"]

    @patch("src.session._study.run_quantization")
    def test_model_factory_result_used_for_session(
        self, mock_rq, model, configs, mock_run_quantization,
    ):
        mock_rq.return_value = (None, None, mock_run_quantization(configs[0]))

        custom_model = _TwoLayerModel()

        def model_factory(cfg):
            return custom_model

        study = Study(configs, model=model)
        study.run(torch.randn(2, 4), model_factory=model_factory)

        # Verify run_quantization was called with the custom model
        model_arg = mock_rq.call_args[0][0]
        assert model_arg is custom_model

    @patch("src.session._study.run_quantization")
    def test_passes_outputs_to_session(
        self, mock_rq, model, configs, mock_run_quantization,
    ):
        cfg = configs[0]
        mock_rq.return_value = (None, None, mock_run_quantization(cfg))

        study = Study([cfg], model=model)
        study.run(torch.randn(2, 4), outputs=["accuracy", "qsnr"])

        call_kwargs = mock_rq.call_args
        assert call_kwargs[1]["outputs"] == ["accuracy", "qsnr"]

    @patch("src.session._study.run_quantization")
    def test_default_outputs_passed_to_session(
        self, mock_rq, model, configs, mock_run_quantization,
    ):
        cfg = configs[0]
        mock_rq.return_value = (None, None, mock_run_quantization(cfg))

        study = Study([cfg], model=model)
        study.run(torch.randn(2, 4))

        call_kwargs = mock_rq.call_args
        assert call_kwargs[1]["outputs"] == "default"

    @patch("src.session._study.run_quantization")
    def test_empty_configs_empty_report(self, mock_rq, model):
        study = Study([], model=model)
        report = study.run(torch.randn(2, 4))

        assert report.total_experiments == 0
        assert report.parts == []

    @patch("src.session._study.run_quantization")
    def test_creates_deep_copies_of_model(
        self, mock_rq, model,
    ):
        """Without model_factory, each run_quantization gets a deep copy."""
        mock_rq.return_value = (None, None, SessionResult(
            name="test", config=QuantConfig(name="test"),
        ))

        study = Study(
            [QuantConfig(name="a"), QuantConfig(name="b")],
            model=model,
        )
        study.run(torch.randn(2, 4))

        assert mock_rq.call_count == 2
        models_passed = [call.args[0] for call in mock_rq.call_args_list]
        for m in models_passed:
            assert m is not model  # Not the original
        assert models_passed[0] is not models_passed[1]  # Different copies

    @patch("src.session._study.run_quantization")
    def test_with_eval_fn_metric_populated(
        self, mock_rq, model, configs,
    ):
        cfg = configs[0]
        mock_rq.return_value = (None, None, SessionResult(
            name="int8",
            config=cfg,
            fp32_metrics={"acc": 0.95},
            quant_metrics={"acc": 0.93},
            delta={"acc": 0.02},
        ))

        eval_fn = MagicMock(return_value={"acc": 0.9})
        study = Study([cfg], model=model)
        report = study.run(
            torch.randn(2, 4), eval_data=torch.randn(2, 4),
            eval_fn=eval_fn,
        )

        assert report.total_experiments == 1
        entry = report._results["int8"][0]
        assert entry.fp32_metrics == {"acc": 0.95}
        assert entry.delta == {"acc": 0.02}

    @patch("src.session._study.run_quantization")
    def test_eval_data_passed_to_session(
        self, mock_rq, model, configs, mock_run_quantization,
    ):
        cfg = configs[0]
        mock_rq.return_value = (None, None, mock_run_quantization(cfg))

        eval_data = torch.randn(4, 4)
        eval_fn = MagicMock(return_value={"acc": 0.9})
        study = Study([cfg], model=model)
        study.run(
            torch.randn(2, 4), eval_data=eval_data, eval_fn=eval_fn,
        )

        call_kwargs = mock_rq.call_args
        assert call_kwargs[1]["eval_data"] is eval_data
        assert call_kwargs[1]["eval_fn"] is eval_fn
