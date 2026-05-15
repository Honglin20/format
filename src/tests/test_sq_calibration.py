# src/tests/test_sq_calibration.py
import torch
import pytest
import torch.nn as nn

from src.session._session import quantize_model
from src.session._config import QuantConfig
from src.calibration.pipeline import CalibrationSession
from src.calibration.strategies import MaxScaleStrategy
from src.scheme.granularity import GranularityMode


class TestSQCalibrationWeight:
    """Calibration pipeline produces SQ-format Hessian importance for weight SQ."""

    def test_calibration_session_sq_mode_accepted(self):
        model = nn.Sequential(nn.Linear(4, 8))
        session = CalibrationSession(model, MaxScaleStrategy(), sq_mode="weight")
        assert session._sq_mode == "weight"

    def test_calibration_collects_sq_inputs(self):
        """With sq_mode='weight', CalibrationSession collects input tensors
        for modules whose weight scheme has sq_importance=True."""
        cfg = QuantConfig(
            w_format="int4", w_granularity="bank", w_block_size=4,
            outlier_format="int8", sq_mode="weight",
        )
        op_cfg = cfg.to_op_config()
        model = nn.Sequential(nn.Linear(8, 4))
        qmodel = quantize_model(model, cfg=op_cfg)

        with CalibrationSession(qmodel, MaxScaleStrategy(), sq_mode="weight") as calib:
            x = torch.randn(2, 8)
            qmodel(x)

        # Should have collected inputs for the linear layer
        assert len(calib._sq_inputs) > 0

    def test_calibration_computes_sq_importance_buffer(self):
        """After calibration with sq_mode='weight', _sq_importance buffer exists."""
        cfg = QuantConfig(
            w_format="int4", w_granularity="bank", w_block_size=4,
            outlier_format="int8", sq_mode="weight",
        )
        op_cfg = cfg.to_op_config()
        model = nn.Sequential(nn.Linear(8, 4))
        qmodel = quantize_model(model, cfg=op_cfg)

        with CalibrationSession(qmodel, MaxScaleStrategy(), sq_mode="weight"):
            for _ in range(4):
                x = torch.randn(2, 8)
                qmodel(x)

        # _sq_importance buffer should be registered on the linear module
        linear = qmodel[0]
        assert hasattr(linear, "_sq_importance")
        imp = linear.get_buffer("_sq_importance")
        assert imp.shape == linear.weight.shape


class TestSQCalibrationActivationStatic:
    """Calibration pipeline produces per-channel mask for activation_static SQ."""

    def test_calibration_collects_sq_outputs(self):
        """With sq_mode='activation_static', CalibrationSession collects outputs."""
        cfg = QuantConfig(
            w_format="int8", w_granularity="per_tensor",
            a_format="int4", a_granularity="bank", a_block_size=4,
            outlier_format="int8", sq_mode="activation_static",
        )
        op_cfg = cfg.to_op_config()
        model = nn.Sequential(nn.Linear(8, 4))
        qmodel = quantize_model(model, cfg=op_cfg)

        with CalibrationSession(qmodel, MaxScaleStrategy(), sq_mode="activation_static") as calib:
            x = torch.randn(2, 8)
            qmodel(x)

        assert len(calib._sq_outputs) > 0

    def test_calibration_computes_sq_activation_mask(self):
        """After calibration with sq_mode='activation_static', _sq_activation_mask exists."""
        cfg = QuantConfig(
            w_format="int8", w_granularity="per_tensor",
            a_format="int4", a_granularity="bank", a_block_size=4,
            outlier_format="int8", sq_mode="activation_static",
        )
        op_cfg = cfg.to_op_config()
        model = nn.Sequential(nn.Linear(8, 4))
        qmodel = quantize_model(model, cfg=op_cfg)

        with CalibrationSession(qmodel, MaxScaleStrategy(), sq_mode="activation_static"):
            for _ in range(4):
                x = torch.randn(2, 8)
                qmodel(x)

        linear = qmodel[0]
        assert hasattr(linear, "_sq_activation_mask")
        mask = linear.get_buffer("_sq_activation_mask")
        assert mask.shape == (4,)  # K = input channels
        assert mask.dtype == torch.bool


class TestSQCalibrationNoop:
    """Without sq_mode, calibration behaves as before (no SQ artifacts)."""

    def test_no_sq_mode_no_sq_buffers(self):
        cfg = QuantConfig(
            w_format="int4", w_granularity="bank", w_block_size=4,
            outlier_format="int8",
        )
        op_cfg = cfg.to_op_config()
        model = nn.Sequential(nn.Linear(8, 4))
        qmodel = quantize_model(model, cfg=op_cfg)

        with CalibrationSession(qmodel, MaxScaleStrategy()):
            for _ in range(4):
                x = torch.randn(2, 8)
                qmodel(x)

        linear = qmodel[0]
        assert not hasattr(linear, "_sq_importance")
        assert not hasattr(linear, "_sq_activation_mask")
