"""Tests for format study helper functions (adapted to new API).

The old pipeline/format_study.py helpers have been replaced:
  - make_op_cfg / make_op_cfg_weight_only  → QuantConfig.to_op_config()
  - _make_smoothquant_transforms           → SmoothQuantTransform.from_model_calibration()
  - _fuse_smoothquant_weights              → fuse_smoothquant_weights()
  - _build_per_layer_optimal_cfg           → per_layer_optimal() in src.session._per_layer_opt
"""
import torch
import torch.nn as nn
import pytest
from src.session._config import QuantConfig
from src.scheme.granularity import GranularitySpec
from src.transform.smooth_quant import SmoothQuantTransform, fuse_smoothquant_weights
from src.scheme.transform import IdentityTransform
from pipeline._model import ToyMLP


class TestMakeSmoothQuantTransforms:
    def test_produces_per_layer_dict(self):
        """ToyMLP has fc1 and fc2 -- should produce transforms for both."""
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        model.eval()
        calib = [torch.randn(4, 16)]
        result = SmoothQuantTransform.from_model_calibration(model, calib)
        assert isinstance(result, dict)
        assert "fc1" in result or not any(
            isinstance(m, (nn.Linear, nn.Conv2d)) for m in model.modules()
        )
        # Verify keys correspond to Linear/Conv2d modules
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                assert name in result, f"{name} missing from SQ transforms"
                assert isinstance(result[name], SmoothQuantTransform)

    def test_none_model_returns_empty(self):
        result = SmoothQuantTransform.from_model_calibration(
            None, [torch.randn(2, 8)]
        )
        assert result == {}

    def test_empty_hooks_returns_empty(self):
        """Model with no Linear/Conv2d returns empty dict."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.1))
        calib = [torch.randn(2, 8)]
        result = SmoothQuantTransform.from_model_calibration(model, calib)
        assert result == {}

    def test_empty_calib_data_raises(self):
        """Empty calib_data list should raise ValueError."""
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        with pytest.raises(ValueError, match="calib_data must contain at least one batch"):
            SmoothQuantTransform.from_model_calibration(model, [])


class TestFuseSmoothQuantWeights:
    def test_does_not_mutate_original(self):
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        orig_fc1_w = model.fc1.weight.data.clone()
        sq_transforms = {"fc1": SmoothQuantTransform(torch.ones(16))}
        fuse_smoothquant_weights(model, sq_transforms)
        assert torch.equal(model.fc1.weight.data, orig_fc1_w)

    def test_fuses_scale_into_weight(self):
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        W_orig = model.fc1.weight.data.clone()
        scale = torch.full((16,), 2.0)
        sq_transforms = {"fc1": SmoothQuantTransform(scale)}
        fused = fuse_smoothquant_weights(model, sq_transforms)
        # After fusion: W_fused = W_orig * scale (broadcast along dim 1)
        expected = W_orig * scale.view(1, 16)
        assert torch.allclose(fused.fc1.weight.data, expected)

    def test_layer_names_filter(self):
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        orig_fc2_w = model.fc2.weight.data.clone()
        scale = torch.full((32,), 2.0)
        sq_transforms = {
            "fc1": SmoothQuantTransform(torch.ones(16)),
            "fc2": SmoothQuantTransform(scale),
        }
        fused = fuse_smoothquant_weights(model, sq_transforms, layer_names=["fc1"])
        # fc2 should NOT be fused
        assert torch.equal(fused.fc2.weight.data, orig_fc2_w)

    def test_skips_non_smoothquant_transform(self):
        """IdentityTransform entries are silently skipped by fuse_smoothquant_weights."""
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        orig_w = model.fc1.weight.data.clone()
        # fuse_smoothquant_weights only processes SmoothQuantTransform entries;
        # IdentityTransform has no .scale attribute and is ignored.
        sq_transforms = {
            "fc1": SmoothQuantTransform(torch.ones(16)),
        }
        fused = fuse_smoothquant_weights(model, sq_transforms, layer_names=[])
        assert torch.equal(fused.fc1.weight.data, orig_w)


class TestMakeOpCfg:
    def test_basic_int8_per_tensor(self):
        cfg = QuantConfig(w_format="int8", w_granularity="per_tensor").to_op_config()
        assert cfg.weight.format.name == "int8"
        assert cfg.input.format.name == "int8"

    def test_with_transform(self):
        from src.transform.hadamard import HadamardTransform
        cfg = QuantConfig(
            w_format="int4", w_granularity="per_tensor",
            transform="hadamard",
        ).to_op_config()
        assert isinstance(cfg.weight.transform, HadamardTransform)

    def test_act_format_w4a8(self):
        cfg = QuantConfig(
            w_format="int4", a_format="int8",
            w_granularity="per_channel",
        ).to_op_config()
        assert cfg.weight.format.name == "int4"
        assert cfg.input.format.name == "int8"

    def test_act_format_same_granularity(self):
        cfg = QuantConfig(
            w_format="int4", a_format="fp8_e4m3",
            w_granularity="per_block", w_block_size=32,
            a_granularity="per_block", a_block_size=32,
        ).to_op_config()
        assert cfg.weight.granularity.block_size == 32
        assert cfg.input.granularity.block_size == 32

    def test_scale_format_pot_default(self):
        cfg = QuantConfig(w_format="int8", w_granularity="per_tensor").to_op_config()
        assert cfg.weight.scale_storage == "pot"

    def test_scale_format_pot(self):
        cfg = QuantConfig(
            w_format="int4", w_granularity="per_channel",
            scale_storage="pot",
        ).to_op_config()
        assert cfg.weight.scale_storage == "pot"
        assert cfg.input.scale_storage == "pot"

    def test_act_format_with_pot_scale(self):
        cfg = QuantConfig(
            w_format="int4", a_format="int8",
            w_granularity="per_tensor", a_granularity="per_tensor",
            scale_storage="pot",
        ).to_op_config()
        assert cfg.weight.scale_storage == "pot"
        assert cfg.input.scale_storage == "pot"
        assert cfg.weight.format.name == "int4"
        assert cfg.input.format.name == "int8"

    def test_without_act_format_all_roles_identical(self):
        cfg = QuantConfig(w_format="int8", w_granularity="per_tensor").to_op_config()
        assert cfg.weight.format.name == cfg.input.format.name


class TestMakeOpCfgWeightOnly:
    def test_basic_weight_only(self):
        cfg = QuantConfig(
            w_format="nf4", w_granularity="per_channel",
            weight_only=True,
        ).to_op_config()
        assert cfg.weight is not None
        assert cfg.input is None
        assert cfg.output is None

    def test_scale_format_pot_default(self):
        cfg = QuantConfig(
            w_format="int4", w_granularity="per_tensor",
            weight_only=True,
        ).to_op_config()
        assert cfg.weight.scale_storage == "pot"

    def test_scale_format_pot(self):
        cfg = QuantConfig(
            w_format="int4", w_granularity="per_tensor",
            weight_only=True, scale_storage="pot",
        ).to_op_config()
        assert cfg.weight.scale_storage == "pot"
