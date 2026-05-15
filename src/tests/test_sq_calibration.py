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


class TestSQPerBankMaskSelection:
    """Mask selection is per-bank, not global."""

    def test_per_bank_selection(self):
        """Each bank gets (1-s)*bank_size high-precision channels."""
        # Simulate 2 banks of 4 channels each
        # Bank 0: channels [0,1,2,3], Bank 1: channels [4,5,6,7]
        act_avg = torch.tensor([1.0, 2.0, 3.0, 4.0,   # Bank 0
                                5.0, 5.0, 1.0, 1.0])  # Bank 1
        w = torch.ones(8, 4)  # uniform weight → importance ∝ |act_avg|

        mask = CalibrationSession._compute_activation_mask_per_bank(
            act_avg, w, bank_size=4, sq_sparsity=0.5
        )
        # s=0.5 → per bank: (1-0.5)*4 = 2 high-precision channels
        # Bank 0: channels 2,3 (highest act_avg in bank 0)
        # Bank 1: channels 4,5 (highest act_avg in bank 1)
        assert mask.sum().item() == 4  # 2 per bank * 2 banks
        assert mask[2].item() is True   # Bank 0, top-2
        assert mask[3].item() is True   # Bank 0, top-2
        assert mask[4].item() is True   # Bank 1, top-2
        assert mask[5].item() is True   # Bank 1, top-2
        assert mask[0].item() is False  # Bank 0, not top-2
        assert mask[1].item() is False  # Bank 0, not top-2
        assert mask[6].item() is False  # Bank 1, not top-2
        assert mask[7].item() is False  # Bank 1, not top-2

    def test_flat_mask_when_no_bank_info(self):
        """When bank_size=None or covers all channels, global selection."""
        act_avg = torch.tensor([1.0, 2.0, 3.0, 4.0])
        w = torch.ones(4, 4)

        mask = CalibrationSession._compute_activation_mask_per_bank(
            act_avg, w, bank_size=None, sq_sparsity=0.5
        )
        # Global top-50%: channels 2,3
        assert mask.sum().item() == 2
        assert mask[2].item() is True
        assert mask[3].item() is True

    def test_sq_sparsity_ratio_respected(self):
        """sq_sparsity controls fraction of low-precision per bank."""
        act_avg = torch.ones(8)  # all same → any selection works
        w = torch.ones(8, 4)

        # s=0.25 → (1-0.25)*4 = 3 high-precision per bank
        mask = CalibrationSession._compute_activation_mask_per_bank(
            act_avg, w, bank_size=4, sq_sparsity=0.25
        )
        assert mask.sum().item() == 6  # 3 per bank * 2 banks

        # s=0.75 → (1-0.75)*4 = 1 high-precision per bank
        mask = CalibrationSession._compute_activation_mask_per_bank(
            act_avg, w, bank_size=4, sq_sparsity=0.75
        )
        assert mask.sum().item() == 2  # 1 per bank * 2 banks
