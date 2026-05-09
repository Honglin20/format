"""Tests for LayerwiseScaleOptimizer — gradient-based per-layer pre-scale optimization."""
import pytest
import torch
import torch.nn as nn
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.transform import IdentityTransform
from src.transform.pre_scale import PreScaleTransform


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
        from src.session import quantize_model

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

    def test_pot_optimization_produces_power_of_two_scales(self):
        """With pot=True, optimized scales should be exact powers of two."""
        from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
        from src.session import quantize_model

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
        opt = LayerwiseScaleOptimizer(num_steps=30, num_batches=4, lr=0.01, pot=True)
        scales = opt.optimize(qmodel, fp32_model, batches)

        assert len(scales) > 0
        for scale in scales.values():
            log2 = torch.log2(scale)
            assert torch.equal(log2, torch.round(log2)), \
                f"scale {scale} is not power-of-two"


# ===================================================================
# _replace_transform / _replace_transform_activation_only
# ===================================================================

class TestReplaceTransform:
    def test_replaces_all_schemes(self):
        from src.calibration.lsq_optimizer import _replace_transform
        from src.formats.base import FormatBase

        fmt = FormatBase.from_str("int8")
        scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec.per_tensor(),
            transform=IdentityTransform(),
        )
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        new_transform = PreScaleTransform(scale=torch.tensor(2.0), pot=True)
        new_cfg = _replace_transform(cfg, new_transform)

        assert new_cfg.input.transform == new_transform
        assert new_cfg.weight.transform == new_transform
        assert new_cfg.output.transform == new_transform
        # format/granularity preserved
        assert new_cfg.input.format is fmt
        assert new_cfg.input.granularity.mode == scheme.granularity.mode

    def test_preserves_none_fields(self):
        from src.calibration.lsq_optimizer import _replace_transform

        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
        )
        cfg = OpQuantConfig(input=scheme, weight=None, output=scheme)

        new_transform = PreScaleTransform(scale=torch.tensor(2.0))
        new_cfg = _replace_transform(cfg, new_transform)

        assert new_cfg.input.transform == new_transform
        assert new_cfg.weight is None
        assert new_cfg.output.transform == new_transform

    def test_preserves_round_mode_and_scale_storage(self):
        from src.calibration.lsq_optimizer import _replace_transform

        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            round_mode="dither",
            scale_storage="pot",
        )
        cfg = OpQuantConfig(input=scheme)

        new_cfg = _replace_transform(cfg, PreScaleTransform(scale=torch.tensor(1.0)))
        assert new_cfg.input.round_mode == "dither"
        assert new_cfg.input.scale_storage == "pot"


class TestReplaceTransformActivationOnly:
    def test_replaces_only_activation_roles(self):
        from src.calibration.lsq_optimizer import _replace_transform_activation_only

        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=IdentityTransform(),
        )
        cfg = OpQuantConfig(
            input=scheme,
            weight=scheme,
            output=scheme,
            grad_input=scheme,
            grad_weight=scheme,
        )

        new_transform = PreScaleTransform(scale=torch.tensor(3.0))
        new_cfg = _replace_transform_activation_only(cfg, new_transform)

        # Activation roles replaced
        assert new_cfg.input.transform == new_transform
        assert new_cfg.output.transform == new_transform
        assert new_cfg.grad_input.transform == new_transform
        # Weight role preserved
        assert new_cfg.weight.transform == IdentityTransform()
        assert new_cfg.grad_weight.transform == IdentityTransform()

    def test_restricted_roles_parameter(self):
        from src.calibration.lsq_optimizer import (
            _replace_transform_activation_only,
            _INPUT_ACTIVATION_ROLES,
        )

        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
            transform=IdentityTransform(),
        )
        cfg = OpQuantConfig(input=scheme, output=scheme, weight=scheme)

        new_transform = PreScaleTransform(scale=torch.tensor(2.0))
        new_cfg = _replace_transform_activation_only(
            cfg, new_transform, roles=_INPUT_ACTIVATION_ROLES,
        )

        assert new_cfg.input.transform == new_transform
        # output is NOT in _INPUT_ACTIVATION_ROLES → preserved
        assert new_cfg.output.transform == IdentityTransform()
        assert new_cfg.weight.transform == IdentityTransform()

    def test_none_schemes_unchanged(self):
        from src.calibration.lsq_optimizer import _replace_transform_activation_only

        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_tensor(),
        )
        cfg = OpQuantConfig(input=scheme, output=None, weight=scheme)

        new_cfg = _replace_transform_activation_only(
            cfg, PreScaleTransform(scale=torch.tensor(1.0)),
        )
        assert new_cfg.output is None
        assert new_cfg.input.transform is not None
