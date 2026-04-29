import pytest

from src.pipeline.config import resolve_config, _resolve_granularity
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.transform.hadamard import HadamardTransform


class TestResolveGranularity:
    def test_per_tensor(self):
        spec = _resolve_granularity({"granularity": "per_tensor"})
        assert spec.mode.name == "PER_TENSOR"

    def test_per_channel_with_axis(self):
        spec = _resolve_granularity({"granularity": "per_channel", "axis": 0})
        assert spec.mode.name == "PER_CHANNEL"
        assert spec.channel_axis == 0

    def test_per_channel_default_axis(self):
        spec = _resolve_granularity({"granularity": "per_channel"})
        assert spec.mode.name == "PER_CHANNEL"
        assert spec.channel_axis == -1

    def test_per_block_with_size_and_axis(self):
        spec = _resolve_granularity({"granularity": "per_block", "block_size": 32, "axis": -1})
        assert spec.mode.name == "PER_BLOCK"
        assert spec.block_size == 32
        assert spec.block_axis == -1

    def test_per_block_default_axis(self):
        spec = _resolve_granularity({"granularity": "per_block", "block_size": 64})
        assert spec.mode.name == "PER_BLOCK"
        assert spec.block_size == 64
        assert spec.block_axis == -1

    def test_unknown_granularity_raises(self):
        with pytest.raises(ValueError, match="Unknown granularity"):
            _resolve_granularity({"granularity": "per_group"})


class TestResolveConfig:
    def test_basic_int8_per_tensor(self):
        cfg = resolve_config({"format": "int8", "granularity": "per_tensor"})
        assert isinstance(cfg, OpQuantConfig)
        assert cfg.input is not None
        assert cfg.weight is not None
        assert cfg.output is not None

    def test_weight_only(self):
        cfg = resolve_config({"format": "nf4", "granularity": "per_channel", "axis": 0, "weight_only": True})
        assert cfg.input is None
        assert cfg.weight is not None
        assert cfg.output is None

    def test_with_hadamard_transform(self):
        cfg = resolve_config({"format": "int4", "granularity": "per_tensor", "transform": "hadamard"})
        assert isinstance(cfg.input.transform, HadamardTransform)

    def test_unknown_format_raises(self):
        with pytest.raises((KeyError, ValueError)):
            resolve_config({"format": "unknown_fmt", "granularity": "per_tensor"})
