"""Tests for per_layer_optimal function.

per_layer_optimal absorbs logic from pipeline/format_study.py L540-643.
These tests mock QuantSession to isolate the PerLayerOpt logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import SessionResult
from src.session._per_layer_opt import per_layer_optimal
from src.transform.smooth_quant import SmoothQuantTransform


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
    return _TwoLayerModel().eval()


@pytest.fixture
def calib_data():
    return [torch.randn(2, 4)]


@pytest.fixture
def eval_fn():
    def _eval(m, d):
        return {"acc": 0.9}
    return _eval


@pytest.fixture
def variant_results():
    """Create variant SessionResults with known QSNR values.

    Layer fc1: hadamard (25) > none (20) > smoothquant (10)
    Layer fc2: smoothquant (30) > none (15) > hadamard (5)

    So best per layer: {"fc1": "hadamard", "fc2": "smoothquant"}
    """
    return [
        SessionResult(
            name="test-none",
            config=QuantConfig(
                name="test", w_format="int8", a_format="int8",
                transform="none",
            ),
            qsnr_per_layer={"fc1": 20.0, "fc2": 15.0},
        ),
        SessionResult(
            name="test-hadamard",
            config=QuantConfig(
                name="test", w_format="int8", a_format="int8",
                transform="hadamard",
            ),
            qsnr_per_layer={"fc1": 25.0, "fc2": 5.0},
        ),
        SessionResult(
            name="test-smoothquant",
            config=QuantConfig(
                name="test", w_format="int8", a_format="int8",
                transform="smoothquant",
            ),
            qsnr_per_layer={"fc1": 10.0, "fc2": 30.0},
        ),
    ]


def _setup_mock_qs(mock_qs_class, model):
    """Configure a mocked QuantSession for per_layer_optimal tests."""
    mock_qs = mock_qs_class.return_value
    mock_qs.estimate_cost.return_value = None
    mock_qs.fp32_model = model

    # Calibrate context (no-op)
    mock_qs.calibrate.return_value.__enter__.return_value = None

    # Analyze context returns report with empty raw data
    mock_ctx = MagicMock()
    mock_ctx.report.return_value._raw = {}
    mock_qs.analyze.return_value.__enter__.return_value = mock_ctx

    return mock_qs


class TestPerLayerOptimal:
    """Tests for per_layer_optimal function."""

    @patch("src.session._per_layer_opt.QuantSession")
    def test_returns_session_result(
        self, MockQuantSession, model, variant_results, calib_data, eval_fn,
    ):
        _setup_mock_qs(MockQuantSession, model)

        result = per_layer_optimal(
            variant_results, calib_data, model, eval_fn,
            eval_data=calib_data,
        )

        assert isinstance(result, SessionResult)
        assert "PerLayerOpt" in result.name

    @patch("src.session._per_layer_opt.QuantSession")
    def test_name_contains_perlayeropt(
        self, MockQuantSession, model, variant_results, calib_data, eval_fn,
    ):
        _setup_mock_qs(MockQuantSession, model)

        result = per_layer_optimal(
            variant_results, calib_data, model, eval_fn,
            eval_data=calib_data,
        )

        assert "PerLayerOpt" in result.name

    @patch("src.session._per_layer_opt.QuantSession")
    def test_sq_transforms_cached_not_recomputed(
        self, MockQuantSession, model, variant_results, calib_data, eval_fn,
    ):
        _setup_mock_qs(MockQuantSession, model)

        sq_cache = {
            "fc2": SmoothQuantTransform(torch.tensor([1.0]), channel_axis=-1),
        }

        with patch(
            "src.session._per_layer_opt.SmoothQuantTransform"
            ".from_model_calibration",
        ) as mock_from_calib:
            per_layer_optimal(
                variant_results, calib_data, model, eval_fn,
                eval_data=calib_data, sq_transforms=sq_cache,
            )
            mock_from_calib.assert_not_called()

    @patch("src.session._per_layer_opt.QuantSession")
    def test_sq_transforms_returned_in_result(
        self, MockQuantSession, model, variant_results, calib_data, eval_fn,
    ):
        _setup_mock_qs(MockQuantSession, model)

        sq_cache = {
            "fc2": SmoothQuantTransform(torch.tensor([1.0]), channel_axis=-1),
        }

        result = per_layer_optimal(
            variant_results, calib_data, model, eval_fn,
            eval_data=calib_data, sq_transforms=sq_cache,
        )

        assert result.sq_transforms is sq_cache

    def test_empty_results_raises_error(self, model, calib_data, eval_fn):
        """Empty results can't be processed - IndexError from part_results[0]."""
        with pytest.raises(IndexError):
            per_layer_optimal([], calib_data, model, eval_fn)

    @patch("src.session._per_layer_opt.QuantSession")
    def test_all_identity_no_sq_computed(
        self, MockQuantSession, model, calib_data, eval_fn,
    ):
        """When all layers pick Identity/None, SQ is never computed."""
        _setup_mock_qs(MockQuantSession, model)

        cfg_none = QuantConfig(
            name="test", w_format="int8", a_format="int8", transform="none",
        )
        results = [
            SessionResult(
                name="test-none", config=cfg_none,
                qsnr_per_layer={"fc1": 10.0},
            ),
            SessionResult(
                name="test-hadamard",
                config=QuantConfig(
                    name="test", w_format="int8", a_format="int8",
                    transform="hadamard",
                ),
                qsnr_per_layer={"fc1": 5.0},
            ),
        ]

        with patch(
            "src.session._per_layer_opt.SmoothQuantTransform"
            ".from_model_calibration",
        ) as mock_from_calib:
            per_layer_optimal(results, calib_data, model, eval_fn)
            mock_from_calib.assert_not_called()

    @patch("src.session._per_layer_opt.QuantSession")
    def test_best_transform_selection_delegation(
        self, MockQuantSession, model, variant_results, calib_data, eval_fn,
    ):
        """per_layer_optimal calls _compute_best_transform_per_layer
        with the correct variant_qsnr."""
        _setup_mock_qs(MockQuantSession, model)

        with patch(
            "src.session._per_layer_opt._compute_best_transform_per_layer",
        ) as mock_compute:
            mock_compute.return_value = {"fc1": "none", "fc2": "hadamard"}

            per_layer_optimal(
                variant_results, calib_data, model, eval_fn,
                eval_data=calib_data,
            )

            call_qsnr = mock_compute.call_args[0][0]
            assert "none" in call_qsnr
            assert "hadamard" in call_qsnr
            assert "smoothquant" in call_qsnr
            assert call_qsnr["none"] == {"fc1": 20.0, "fc2": 15.0}
            assert call_qsnr["hadamard"] == {"fc1": 25.0, "fc2": 5.0}
            assert call_qsnr["smoothquant"] == {"fc1": 10.0, "fc2": 30.0}

    @patch("src.session._per_layer_opt.QuantSession")
    def test_passes_per_layer_cfg_to_quant_session(
        self, MockQuantSession, model, variant_results, calib_data, eval_fn,
    ):
        """QuantSession receives a dict of per-layer OpQuantConfig."""
        _setup_mock_qs(MockQuantSession, model)

        with patch(
            "src.session._per_layer_opt._compute_best_transform_per_layer",
        ) as mock_compute:
            mock_compute.return_value = {"fc1": "none", "fc2": "hadamard"}

            per_layer_optimal(
                variant_results, calib_data, model, eval_fn,
                eval_data=calib_data,
            )

            cfg_arg = MockQuantSession.call_args[0][1]
            assert isinstance(cfg_arg, dict)
            assert "fc1" in cfg_arg
            assert "fc2" in cfg_arg

    @patch("src.session._per_layer_opt.QuantSession")
    def test_sq_computed_when_not_cached(
        self, MockQuantSession, model, variant_results, calib_data, eval_fn,
    ):
        """When sq_transforms is not provided but SQ is the best transform,
        from_model_calibration IS called."""
        _setup_mock_qs(MockQuantSession, model)

        with patch(
            "src.session._per_layer_opt.SmoothQuantTransform"
            ".from_model_calibration",
        ) as mock_from_calib:
            mock_from_calib.return_value = {
                "fc2": SmoothQuantTransform(
                    torch.tensor([1.0]), channel_axis=-1,
                ),
            }

            result = per_layer_optimal(
                variant_results, calib_data, model, eval_fn,
                eval_data=calib_data,
            )

            mock_from_calib.assert_called_once()
            assert isinstance(result, SessionResult)
            assert "PerLayerOpt" in result.name
