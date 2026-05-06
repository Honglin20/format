"""Tests for format study helper functions."""
import torch
import torch.nn as nn
import pytest
from src.pipeline.format_study import (
    _make_smoothquant_transforms,
    _fuse_smoothquant_weights,
    _build_per_layer_optimal_cfg,
    make_op_cfg,
    make_op_cfg_weight_only,
)
from src.scheme.granularity import GranularitySpec
from src.transform.smooth_quant import SmoothQuantTransform
from src.scheme.transform import IdentityTransform
from pipeline._model import ToyMLP


class TestMakeSmoothQuantTransforms:
    def test_produces_per_layer_dict(self):
        """ToyMLP has fc1 and fc2 -- should produce transforms for both."""
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        model.eval()
        calib = [torch.randn(4, 16)]
        result = _make_smoothquant_transforms(model, calib)
        assert isinstance(result, dict)
        assert "fc1" in result
        assert "fc2" in result
        assert isinstance(result["fc1"], SmoothQuantTransform)
        assert isinstance(result["fc2"], SmoothQuantTransform)

    def test_none_model_returns_empty(self):
        result = _make_smoothquant_transforms(None, [torch.randn(2, 8)])
        assert result == {}

    def test_empty_hooks_returns_empty(self):
        """Model with no Linear/Conv2d returns empty dict."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.1))
        calib = [torch.randn(2, 8)]
        result = _make_smoothquant_transforms(model, calib)
        assert result == {}


    def test_empty_calib_data_raises(self):
        """Empty calib_data list should raise ValueError."""
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        with pytest.raises(ValueError, match="calib_data must contain at least one batch"):
            _make_smoothquant_transforms(model, [])


class TestFuseSmoothQuantWeights:
    def test_does_not_mutate_original(self):
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        orig_fc1_w = model.fc1.weight.data.clone()
        sq_transforms = {"fc1": SmoothQuantTransform(torch.ones(16))}
        _fuse_smoothquant_weights(model, sq_transforms)
        assert torch.equal(model.fc1.weight.data, orig_fc1_w)

    def test_fuses_scale_into_weight(self):
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        W_orig = model.fc1.weight.data.clone()
        scale = torch.full((16,), 2.0)
        sq_transforms = {"fc1": SmoothQuantTransform(scale)}
        fused = _fuse_smoothquant_weights(model, sq_transforms)
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
        fused = _fuse_smoothquant_weights(model, sq_transforms, layer_names={"fc1"})
        # fc2 should NOT be fused
        assert torch.equal(fused.fc2.weight.data, orig_fc2_w)

    def test_skips_non_smoothquant_transform(self):
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        orig_w = model.fc1.weight.data.clone()
        sq_transforms = {"fc1": IdentityTransform()}
        fused = _fuse_smoothquant_weights(model, sq_transforms)
        assert torch.equal(fused.fc1.weight.data, orig_w)


class TestBuildPerLayerOptimalCfg:
    def test_selects_best_transform(self):
        variant_results = {
            "None": {"qsnr_per_layer": {"fc1": 10.0, "fc2": 12.0}},
            "SmoothQuant": {"qsnr_per_layer": {"fc1": 15.0, "fc2": 11.0}},
        }
        sq_transforms = {"fc1": SmoothQuantTransform(torch.ones(16))}
        gran = GranularitySpec.per_tensor()
        result = _build_per_layer_optimal_cfg(
            variant_results, sq_transforms, "int8", gran, make_op_cfg, weight_only=False,
        )
        # fc1 -> SmoothQuant (15 > 10), fc2 -> None (12 > 11)
        assert "fc1" in result
        assert "fc2" in result
        assert result["fc1"].input is not None  # SmoothQuant applies to input

    def test_weight_only_mode(self):
        variant_results = {
            "None": {"qsnr_per_layer": {"fc1": 10.0}},
        }
        sq_transforms = {}
        gran = GranularitySpec.per_tensor()
        result = _build_per_layer_optimal_cfg(
            variant_results, sq_transforms, "nf4", gran, make_op_cfg_weight_only, weight_only=True,
        )
        cfg = result["fc1"]
        assert cfg.weight is not None
        assert cfg.input is None
        assert cfg.output is None

    def test_hadamard_selection_path(self):
        variant_results = {
            "None": {"qsnr_per_layer": {"fc1": 10.0}},
            "Hadamard": {"qsnr_per_layer": {"fc1": 15.0}},
        }
        sq_transforms = {}
        gran = GranularitySpec.per_tensor()
        result = _build_per_layer_optimal_cfg(
            variant_results, sq_transforms, "int8", gran, make_op_cfg, weight_only=False,
        )
        assert "fc1" in result
        assert result["fc1"].input is not None  # Hadamard applies to input

    def test_scale_format_threads_through(self):
        variant_results = {
            "None": {"qsnr_per_layer": {"fc1": 10.0}},
        }
        sq_transforms = {}
        gran = GranularitySpec.per_tensor()
        result = _build_per_layer_optimal_cfg(
            variant_results, sq_transforms, "int8", gran, make_op_cfg,
            weight_only=False, scale_format="pot",
        )
        assert result["fc1"].weight.scale_format == "pot"

    def test_scale_format_threads_through_smoothquant(self):
        variant_results = {
            "SmoothQuant": {"qsnr_per_layer": {"fc1": 10.0}},
        }
        sq_transforms = {"fc1": SmoothQuantTransform(torch.ones(16))}
        gran = GranularitySpec.per_tensor()
        result = _build_per_layer_optimal_cfg(
            variant_results, sq_transforms, "int8", gran, make_op_cfg,
            weight_only=False, scale_format="pot",
        )
        assert result["fc1"].weight.scale_format == "pot"
        assert result["fc1"].input.scale_format == "pot"


class TestMakeOpCfg:
    def test_basic_int8_per_tensor(self):
        cfg = make_op_cfg("int8", GranularitySpec.per_tensor())
        assert cfg.weight.format.name == "int8"
        assert cfg.input.format.name == "int8"
        assert cfg.output.format.name == "int8"

    def test_with_transform(self):
        from src.transform.hadamard import HadamardTransform
        cfg = make_op_cfg("int4", GranularitySpec.per_tensor(), transform=HadamardTransform())
        assert isinstance(cfg.weight.transform, HadamardTransform)

    def test_act_format_w4a8(self):
        cfg = make_op_cfg("int4", GranularitySpec.per_channel(axis=-1), act_format="int8")
        assert cfg.weight.format.name == "int4"
        assert cfg.input.format.name == "int8"
        assert cfg.output.format.name == "int8"

    def test_act_format_same_granularity(self):
        cfg = make_op_cfg("int4", GranularitySpec.per_block(size=32), act_format="fp8_e4m3")
        assert cfg.weight.granularity.mode == GranularitySpec.per_block(32).mode
        assert cfg.input.granularity.block_size == 32

    def test_scale_format_fp32_default(self):
        cfg = make_op_cfg("int8", GranularitySpec.per_tensor())
        assert cfg.weight.scale_format == "fp32"

    def test_scale_format_pot(self):
        cfg = make_op_cfg("int4", GranularitySpec.per_channel(axis=-1), scale_format="pot")
        assert cfg.weight.scale_format == "pot"
        assert cfg.input.scale_format == "pot"

    def test_act_format_with_pot_scale(self):
        cfg = make_op_cfg("int4", GranularitySpec.per_tensor(), act_format="int8", scale_format="pot")
        assert cfg.weight.scale_format == "pot"
        assert cfg.input.scale_format == "pot"
        assert cfg.weight.format.name == "int4"
        assert cfg.input.format.name == "int8"

    def test_without_act_format_all_roles_identical(self):
        cfg = make_op_cfg("int8", GranularitySpec.per_tensor())
        # All three roles share the same scheme object when no act_format
        assert cfg.weight.format.name == cfg.input.format.name == cfg.output.format.name


class TestMakeOpCfgWeightOnly:
    def test_basic_weight_only(self):
        cfg = make_op_cfg_weight_only("nf4", GranularitySpec.per_channel(axis=0))
        assert cfg.weight is not None
        assert cfg.input is None
        assert cfg.output is None

    def test_scale_format_default(self):
        cfg = make_op_cfg_weight_only("int4", GranularitySpec.per_tensor())
        assert cfg.weight.scale_format == "fp32"

    def test_scale_format_pot(self):
        cfg = make_op_cfg_weight_only("int4", GranularitySpec.per_tensor(), scale_format="pot")
        assert cfg.weight.scale_format == "pot"
