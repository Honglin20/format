"""Tests for LayerwiseScaleOptimizer — gradient-based per-layer pre-scale optimization."""
import pytest
import torch
import torch.nn as nn
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec


class _TinyModel(nn.Module):
    """Single quantized layer for unit testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


class TestLayerwiseScaleOptimizer:
    """Unit tests for LayerwiseScaleOptimizer."""

    def test_construct(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        opt = LayerwiseScaleOptimizer(num_steps=50, num_batches=4)
        assert opt.num_steps == 50
        assert opt.num_batches == 4
        assert opt.lr == 1e-3
        assert opt.loss == "mse"

    def test_construct_defaults(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        opt = LayerwiseScaleOptimizer()
        assert opt.num_steps == 100
        assert opt.num_batches == 8

    def test_rejects_invalid_num_steps(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        with pytest.raises(ValueError, match="num_steps"):
            LayerwiseScaleOptimizer(num_steps=0)

    def test_rejects_invalid_num_batches(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        with pytest.raises(ValueError, match="num_batches"):
            LayerwiseScaleOptimizer(num_batches=0)

    def test_rejects_invalid_lr(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        with pytest.raises(ValueError, match="lr"):
            LayerwiseScaleOptimizer(lr=-0.1)

    def test_rejects_invalid_loss(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        with pytest.raises(ValueError, match="loss"):
            LayerwiseScaleOptimizer(loss="huber")

    def test_rejects_invalid_optimizer(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        with pytest.raises(ValueError, match="optimizer"):
            LayerwiseScaleOptimizer(optimizer="rmsprop")


class TestLayerwiseScaleOptimizerIntegration:
    """Integration: optimizer runs on a single quantized layer."""

    def test_optimizer_runs_and_produces_scales(self):
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        from src.mapping.quantize_model import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        fp32_model = _TinyModel()
        fp32_model.load_state_dict(model.state_dict())

        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
        )
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        qmodel = quantize_model(model, cfg)

        batches = [torch.randn(2, 4) for _ in range(4)]

        opt = LayerwiseScaleOptimizer(num_steps=30, num_batches=4, lr=0.01)
        scales = opt.optimize(qmodel, fp32_model, batches)

        assert len(scales) > 0

        # Check _pre_scale buffer was registered
        for _, mod in qmodel.named_modules():
            if hasattr(mod, "_pre_scale"):
                assert isinstance(mod._pre_scale, torch.Tensor)

        # Forward pass still works after optimization
        qmodel.eval()
        out = qmodel(torch.randn(2, 4))
        assert out.shape == (2, 3)
        assert not torch.isnan(out).any()
