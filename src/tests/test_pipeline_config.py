import pytest

from src.session import resolve_config
from src.session._config import _resolve_granularity
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.transform.hadamard import HadamardTransform


class TestResolveGranularity:
    def test_per_tensor(self):
        spec = _resolve_granularity("per_tensor")
        assert spec.mode == GranularityMode.PER_TENSOR

    def test_per_channel_with_axis(self):
        spec = _resolve_granularity("per_channel", axis=0)
        assert spec.mode == GranularityMode.PER_CHANNEL
        assert spec.channel_axis == 0

    def test_per_channel_default_axis(self):
        spec = _resolve_granularity("per_channel")
        assert spec.mode == GranularityMode.PER_CHANNEL
        assert spec.channel_axis == -1

    def test_per_block_with_size_and_axis(self):
        spec = _resolve_granularity("per_block", block_size=32, axis=-1)
        assert spec.mode == GranularityMode.PER_BLOCK
        assert spec.block_size == 32
        assert spec.block_axis == -1

    def test_per_block_default_axis(self):
        spec = _resolve_granularity("per_block", block_size=64)
        assert spec.mode == GranularityMode.PER_BLOCK
        assert spec.block_size == 64
        assert spec.block_axis == -1

    def test_unknown_granularity_raises(self):
        with pytest.raises(ValueError, match="Unknown granularity"):
            _resolve_granularity("per_group")

    def test_per_block_missing_size_raises(self):
        with pytest.raises(ValueError, match="per_block granularity requires block_size"):
            _resolve_granularity("per_block")


class TestResolveConfig:
    def test_basic_int8_per_tensor(self):
        cfg = resolve_config({"format": "int8", "granularity": "per_tensor"})
        assert isinstance(cfg, OpQuantConfig)
        assert cfg.input is not None
        assert cfg.weight is not None
        assert cfg.output is None

    def test_weight_only(self):
        cfg = resolve_config({"format": "nf4", "granularity": "per_channel", "axis": 0, "weight_only": True})
        assert cfg.input is None
        assert cfg.weight is not None
        assert cfg.output is None

    def test_with_hadamard_transform(self):
        cfg = resolve_config({"format": "int4", "granularity": "per_tensor", "transform": "hadamard"})
        assert isinstance(cfg.input.transform, HadamardTransform)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            resolve_config({"format": "unknown_fmt", "granularity": "per_tensor"})

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Invalid transform"):
            resolve_config({"format": "int8", "granularity": "per_tensor", "transform": "no_such_transform"})

    def test_missing_format_raises(self):
        with pytest.raises(ValueError, match="must contain 'format' key"):
            resolve_config({"granularity": "per_tensor"})

    def test_missing_granularity_raises(self):
        with pytest.raises(ValueError, match="must contain 'granularity' key"):
            resolve_config({"format": "int8"})

    def test_format_must_be_string(self):
        with pytest.raises(TypeError, match="'format' must be a string"):
            resolve_config({"format": 42, "granularity": "per_tensor"})

    def test_granularity_must_be_string(self):
        with pytest.raises(TypeError, match="'granularity' must be a string"):
            resolve_config({"format": "int8", "granularity": 42})

    def test_axis_must_be_int(self):
        with pytest.raises(TypeError, match="'axis' must be an int"):
            resolve_config({"format": "int8", "granularity": "per_channel", "axis": "-1"})

    def test_block_size_must_be_int(self):
        with pytest.raises(TypeError, match="'block_size' must be an int"):
            resolve_config({"format": "int8", "granularity": "per_block", "block_size": "32"})

    def test_weight_only_must_be_bool(self):
        with pytest.raises(TypeError, match="'weight_only' must be a bool"):
            resolve_config({"format": "int8", "granularity": "per_tensor", "weight_only": "yes"})

    # ── scale_format ──────────────────────────────────────────────────

    def test_scale_format_default_fp32(self):
        cfg = resolve_config({"format": "int8", "granularity": "per_tensor"})
        assert cfg.weight.scale_storage == "fp32"
        assert cfg.input.scale_storage == "fp32"

    def test_scale_format_pot(self):
        cfg = resolve_config({"format": "int8", "granularity": "per_tensor", "scale_format": "pot"})
        assert cfg.weight.scale_storage == "pot"
        assert cfg.input.scale_storage == "pot"

    def test_scale_format_per_channel_pot(self):
        cfg = resolve_config({"format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "pot"})
        assert cfg.weight.scale_storage == "pot"
        assert cfg.weight.granularity.mode == GranularityMode.PER_CHANNEL

    def test_scale_format_must_be_string(self):
        with pytest.raises(TypeError, match="'scale_format' must be a string"):
            resolve_config({"format": "int8", "granularity": "per_tensor", "scale_format": 42})

    def test_scale_format_invalid_value_raises(self):
        with pytest.raises(ValueError, match="Invalid scale_format"):
            resolve_config({"format": "int8", "granularity": "per_tensor", "scale_format": "float16"})

    # ── act_format (mixed-precision wXaY) ─────────────────────────────

    def test_act_format_w4a8_weight_vs_activation(self):
        cfg = resolve_config({
            "format": "int4", "act_format": "int8",
            "granularity": "per_channel", "axis": -1,
        })
        assert cfg.weight.format.name == "int4"
        assert cfg.input.format.name == "int8"

    def test_act_format_with_fp_activation(self):
        cfg = resolve_config({
            "format": "int4", "act_format": "fp8_e4m3",
            "granularity": "per_block", "block_size": 32,
        })
        assert cfg.weight.format.name == "int4"
        assert cfg.input.format.name == "fp8_e4m3"

    def test_act_format_same_granularity_as_weight(self):
        cfg = resolve_config({
            "format": "int4", "act_format": "int8",
            "granularity": "per_channel", "axis": 0,
        })
        assert cfg.weight.granularity.channel_axis == 0
        assert cfg.input.granularity.channel_axis == 0

    def test_act_format_inherits_scale_format(self):
        cfg = resolve_config({
            "format": "int4", "act_format": "int8",
            "granularity": "per_tensor", "scale_format": "pot",
        })
        assert cfg.weight.scale_storage == "pot"
        assert cfg.input.scale_storage == "pot"

    def test_act_format_inherits_transform(self):
        cfg = resolve_config({
            "format": "int4", "act_format": "int8",
            "granularity": "per_tensor", "transform": "hadamard",
        })
        assert isinstance(cfg.weight.transform, HadamardTransform)
        assert isinstance(cfg.input.transform, HadamardTransform)

    def test_act_format_with_weight_only_raises(self):
        with pytest.raises(ValueError, match="'act_format' cannot be used with 'weight_only=True'"):
            resolve_config({
                "format": "int4", "act_format": "int8",
                "granularity": "per_channel", "axis": 0,
                "weight_only": True,
            })

    def test_act_format_must_be_string(self):
        with pytest.raises(TypeError, match="'act_format' must be a string"):
            resolve_config({
                "format": "int4", "act_format": 42,
                "granularity": "per_tensor",
            })

    def test_act_format_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            resolve_config({
                "format": "int4", "act_format": "no_such_fmt",
                "granularity": "per_tensor",
            })

    def test_without_act_format_all_roles_same(self):
        cfg = resolve_config({"format": "int8", "granularity": "per_tensor"})
        assert cfg.weight.format.name == "int8"
        assert cfg.input.format.name == "int8"

    # ── Transform type validation ─────────────────────────────────────

    def test_transform_type_error(self):
        """Non-string transform raises TypeError."""
        with pytest.raises(TypeError, match="'transform' must be a string"):
            resolve_config({"format": "int8", "granularity": "per_tensor", "transform": 42})
