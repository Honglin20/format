"""Tests for QuantConfig dataclass and resolve_config() backward-compat path."""
import pytest
import torch

from src.formats.base import FormatBase
from src.formats.int_formats import IntFormat
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform
from src.session._config import QuantConfig, resolve_config


# ===========================================================================
# QuantConfig: field defaults
# ===========================================================================

class TestQuantConfigDefaults:
    def test_default_values(self):
        cfg = QuantConfig()
        assert cfg.name == ""
        assert cfg.w_format == "int8"
        assert cfg.w_granularity == "per_tensor"
        assert cfg.w_block_size is None
        assert cfg.w_axis == -1
        assert cfg.a_format is None
        assert cfg.a_granularity == "per_tensor"
        assert cfg.a_block_size is None
        assert cfg.a_axis == -1
        assert cfg.transform == "none"
        assert cfg.sq_alpha == 0.5
        assert cfg.prescale_init == "ones"
        assert cfg.prescale_pot is False
        assert cfg.prescale_granularity is None
        assert cfg.lsq_steps == 0
        assert cfg.lsq_lr == 1e-3
        assert cfg.scale_storage == "pot"
        assert cfg.calibrator == "mse"
        assert cfg.weight_only is False

    def test_custom_values(self):
        cfg = QuantConfig(
            name="test",
            w_format="nf4",
            w_granularity="per_channel",
            a_format="fp8_e4m3",
            transform="hadamard",
            calibrator="percentile",
            scale_storage="pot",
        )
        assert cfg.name == "test"
        assert cfg.w_format == "nf4"
        assert cfg.a_format == "fp8_e4m3"
        assert cfg.transform == "hadamard"
        assert cfg.calibrator == "percentile"
        assert cfg.scale_storage == "pot"


# ===========================================================================
# QuantConfig.to_op_config() — basic scheme resolution
# ===========================================================================

class TestToOpConfigBasic:
    def test_basic_int8_per_tensor(self):
        cfg = QuantConfig(w_format="int8")
        result = cfg.to_op_config()
        assert isinstance(result, OpQuantConfig)
        assert result.input is not None
        assert result.weight is not None
        assert result.output is None
        # Weight scheme
        assert result.weight.format.name == "int8"
        assert result.weight.granularity.mode == GranularityMode.PER_TENSOR
        # Input scheme (same as weight when a_format is None)
        assert result.input.format.name == "int8"
        assert result.input.granularity.mode == GranularityMode.PER_TENSOR

    def test_with_a_format_mixed_precision(self):
        """a_format set → activation uses different format from weight."""
        cfg = QuantConfig(w_format="int8", a_format="fp8_e4m3")
        result = cfg.to_op_config()
        assert result.weight.format.name == "int8"
        assert result.input.format.name == "fp8_e4m3"

    def test_a_format_none_uses_weight_format(self):
        """a_format=None → activation uses same format as weight."""
        cfg = QuantConfig(w_format="nf4", a_format=None)
        result = cfg.to_op_config()
        assert result.weight.format.name == "nf4"
        assert result.input.format.name == "nf4"

    def test_scale_storage_pot(self):
        """scale_storage='pot' → scale_storage='pot' on QuantScheme."""
        cfg = QuantConfig(w_format="int8", scale_storage="pot")
        result = cfg.to_op_config()
        assert result.weight.scale_storage == "pot"
        assert result.input.scale_storage == "pot"

    def test_with_block_size(self):
        """w_block_size=32 → QuantScheme has block_size=32."""
        cfg = QuantConfig(w_format="int8", w_granularity="per_block", w_block_size=32)
        result = cfg.to_op_config()
        assert result.weight.granularity.mode == GranularityMode.PER_BLOCK
        assert result.weight.granularity.block_size == 32

    def test_axis_forwarded_to_granularity(self):
        """w_axis=0, a_axis=1 → GranularitySpec uses those axes."""
        cfg = QuantConfig(
            w_format="int8", w_granularity="per_channel", w_axis=0,
            a_granularity="per_channel", a_axis=1,
        )
        result = cfg.to_op_config()
        assert result.weight.granularity.channel_axis == 0
        assert result.input.granularity.channel_axis == 1

    def test_axis_default_is_minus_one(self):
        """w_axis and a_axis default to -1."""
        cfg = QuantConfig(w_format="int8", w_granularity="per_channel",
                          a_granularity="per_channel")
        result = cfg.to_op_config()
        assert result.weight.granularity.channel_axis == -1
        assert result.input.granularity.channel_axis == -1


# ===========================================================================
# QuantConfig.to_op_config() — transform resolution
# ===========================================================================

class TestToOpConfigTransform:
    def test_transform_none(self):
        cfg = QuantConfig(transform="none")
        result = cfg.to_op_config()
        assert isinstance(result.weight.transform, IdentityTransform)
        assert isinstance(result.input.transform, IdentityTransform)

    def test_transform_hadamard(self):
        cfg = QuantConfig(transform="hadamard")
        result = cfg.to_op_config()
        assert isinstance(result.weight.transform, HadamardTransform)
        assert isinstance(result.input.transform, HadamardTransform)

    def test_transform_prescale(self):
        """prescale → IdentityTransform placeholder on both weight and activation."""
        cfg = QuantConfig(transform="prescale")
        result = cfg.to_op_config()
        assert isinstance(result.weight.transform, IdentityTransform)
        assert isinstance(result.input.transform, IdentityTransform)

    def test_transform_smoothquant(self):
        """smoothquant → IdentityTransform on weight, SmoothQuantTransform on activation."""
        cfg = QuantConfig(transform="smoothquant", sq_alpha=0.7)
        result = cfg.to_op_config()
        assert isinstance(result.weight.transform, IdentityTransform)
        assert isinstance(result.input.transform, SmoothQuantTransform)
        assert cfg.sq_alpha == 0.7


# ===========================================================================
# QuantConfig.to_op_config() — weight_only
# ===========================================================================

class TestToOpConfigWeightOnly:
    def test_weight_only(self):
        """weight_only=True → only weight scheme, input/output are None."""
        cfg = QuantConfig(w_format="nf4", w_granularity="per_channel", weight_only=True)
        result = cfg.to_op_config()
        assert result.input is None
        assert result.weight is not None
        assert result.output is None
        assert result.weight.format.name == "nf4"

    def test_weight_only_with_a_format_raises(self):
        """weight_only=True + a_format set → ValueError."""
        with pytest.raises(ValueError, match="a_format.*weight_only"):
            QuantConfig(w_format="int8", a_format="fp8_e4m3", weight_only=True)


# ===========================================================================
# QuantConfig: validation
# ===========================================================================

class TestQuantConfigValidation:
    def test_invalid_w_granularity(self):
        with pytest.raises(ValueError, match="Invalid w_granularity"):
            QuantConfig(w_granularity="per_group")

    def test_invalid_a_granularity(self):
        with pytest.raises(ValueError, match="Invalid a_granularity"):
            QuantConfig(a_granularity="per_group")

    def test_invalid_transform(self):
        with pytest.raises(ValueError, match="Invalid transform"):
            QuantConfig(transform="invalid_tx")

    def test_invalid_calibrator(self):
        with pytest.raises(ValueError, match="Invalid calibrator"):
            QuantConfig(calibrator="invalid_cal")

    def test_invalid_scale_storage(self):
        with pytest.raises(ValueError, match="Invalid scale_storage"):
            QuantConfig(scale_storage="int8")

    def test_lsq_steps_negative(self):
        with pytest.raises(ValueError, match="lsq_steps.*>= 0"):
            QuantConfig(lsq_steps=-1)

    def test_lsq_steps_without_prescale(self):
        """transform != 'prescale' + lsq_steps > 0 → ValueError."""
        for tx in ("none", "hadamard", "smoothquant"):
            with pytest.raises(ValueError, match="lsq_steps.*prescale"):
                QuantConfig(transform=tx, lsq_steps=10)

    def test_lsq_steps_with_prescale_succeeds(self):
        """transform='prescale' + lsq_steps > 0 → OK (no error)."""
        cfg = QuantConfig(transform="prescale", lsq_steps=10)
        assert cfg.lsq_steps == 10
        assert cfg.transform == "prescale"

    def test_w_per_block_without_block_size(self):
        with pytest.raises(ValueError, match="w_block_size.*required"):
            QuantConfig(w_granularity="per_block")

    def test_a_per_block_without_block_size(self):
        with pytest.raises(ValueError, match="a_block_size.*required"):
            QuantConfig(a_granularity="per_block")


# ===========================================================================
# QuantConfig: field storage (non-transform fields)
# ===========================================================================

class TestQuantConfigFieldStorage:
    def test_calibrator_percentile_stored(self):
        cfg = QuantConfig(calibrator="percentile")
        assert cfg.calibrator == "percentile"

    def test_prescale_fields_stored(self):
        cfg = QuantConfig(
            prescale_init="amax",
            prescale_pot=True,
            prescale_granularity="per_channel",
        )
        assert cfg.prescale_init == "amax"
        assert cfg.prescale_pot is True
        assert cfg.prescale_granularity == "per_channel"

    def test_activation_granularity_differs_from_weight(self):
        """w_granularity and a_granularity can differ."""
        cfg = QuantConfig(
            w_granularity="per_channel",
            a_granularity="per_tensor",
        )
        assert cfg.w_granularity == "per_channel"
        assert cfg.a_granularity == "per_tensor"

    def test_prescale_granularity_default_follows_a_granularity(self):
        """prescale_granularity defaults to a_granularity only when transform='prescale'."""
        cfg = QuantConfig(a_granularity="per_channel", transform="prescale")
        assert cfg.prescale_granularity == "per_channel"

    def test_prescale_granularity_none_when_not_prescale(self):
        """prescale_granularity stays None when transform is not 'prescale'."""
        cfg = QuantConfig(a_granularity="per_channel", transform="none")
        assert cfg.prescale_granularity is None


# ===========================================================================
# resolve_config() — backward-compat descriptor → OpQuantConfig
# ===========================================================================

class TestResolveConfig:
    def test_basic(self):
        desc = {"format": "int8", "granularity": "per_tensor"}
        result = resolve_config(desc)
        assert isinstance(result, OpQuantConfig)
        assert result.input is not None
        assert result.weight is not None
        assert result.output is None
        assert result.weight.format.name == "int8"
        assert result.weight.granularity.mode == GranularityMode.PER_TENSOR
        assert result.input.format.name == "int8"

    def test_with_act_format(self):
        desc = {
            "format": "int8",
            "act_format": "fp8_e4m3",
            "granularity": "per_tensor",
        }
        result = resolve_config(desc)
        assert result.weight.format.name == "int8"
        assert result.input.format.name == "fp8_e4m3"

    def test_with_scale_format_pot(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "scale_format": "pot",
        }
        result = resolve_config(desc)
        assert result.weight.scale_storage == "pot"
        assert result.input.scale_storage == "pot"

    def test_with_weight_only(self):
        desc = {
            "format": "int8",
            "granularity": "per_channel",
            "axis": 0,
            "weight_only": True,
        }
        result = resolve_config(desc)
        assert result.input is None
        assert result.weight is not None
        assert result.output is None
        assert result.weight.format.name == "int8"
        assert result.weight.granularity.mode == GranularityMode.PER_CHANNEL

    def test_with_hadamard_transform(self):
        desc = {
            "format": "int4",
            "granularity": "per_tensor",
            "transform": "hadamard",
        }
        result = resolve_config(desc)
        assert isinstance(result.weight.transform, HadamardTransform)
        assert isinstance(result.input.transform, HadamardTransform)

    def test_weight_only_with_act_format_raises(self):
        desc = {
            "format": "int8",
            "act_format": "fp8_e4m3",
            "granularity": "per_tensor",
            "weight_only": True,
        }
        with pytest.raises(ValueError, match="act_format.*weight_only"):
            resolve_config(desc)


# ===========================================================================
# resolve_config() — error paths
# ===========================================================================

class TestResolveConfigErrors:
    def test_missing_format(self):
        desc = {"granularity": "per_tensor"}
        with pytest.raises(ValueError, match="descriptor must contain 'format' key"):
            resolve_config(desc)

    def test_missing_granularity(self):
        desc = {"format": "int8"}
        with pytest.raises(ValueError, match="descriptor must contain 'granularity' key"):
            resolve_config(desc)

    def test_unknown_granularity_string(self):
        desc = {"format": "int8", "granularity": "invalid_gran"}
        with pytest.raises(ValueError, match="Unknown granularity"):
            resolve_config(desc)

    def test_non_string_format(self):
        desc = {"format": 42, "granularity": "per_tensor"}
        with pytest.raises(TypeError, match="'format' must be a string"):
            resolve_config(desc)

    def test_non_string_granularity(self):
        desc = {"format": "int8", "granularity": 123}
        with pytest.raises(TypeError, match="'granularity' must be a string"):
            resolve_config(desc)

    def test_non_int_axis(self):
        desc = {
            "format": "int8",
            "granularity": "per_channel",
            "axis": "not_an_int",
        }
        with pytest.raises(TypeError, match="'axis' must be an int"):
            resolve_config(desc)

    def test_non_int_block_size(self):
        desc = {
            "format": "int8",
            "granularity": "per_block",
            "block_size": "not_an_int",
        }
        with pytest.raises(TypeError, match="'block_size' must be an int"):
            resolve_config(desc)

    def test_non_bool_weight_only(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "weight_only": "yes",
        }
        with pytest.raises(TypeError, match="'weight_only' must be a bool"):
            resolve_config(desc)

    def test_unknown_transform_string(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "transform": "unknown_tx",
        }
        with pytest.raises(ValueError, match="Invalid transform"):
            resolve_config(desc)

    def test_invalid_transform_type(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "transform": 42,
        }
        with pytest.raises(TypeError, match="'transform' must be a string"):
            resolve_config(desc)

    def test_invalid_scale_format_value(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "scale_format": "int8",
        }
        with pytest.raises(ValueError, match="Invalid scale_format"):
            resolve_config(desc)

    def test_non_string_scale_format(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "scale_format": 123,
        }
        with pytest.raises(TypeError, match="'scale_format' must be a string"):
            resolve_config(desc)

    def test_non_string_act_format(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "act_format": 42,
        }
        with pytest.raises(TypeError, match="'act_format' must be a string"):
            resolve_config(desc)

    def test_unknown_act_format_string(self):
        desc = {
            "format": "int8",
            "granularity": "per_tensor",
            "act_format": "unknown_format",
        }
        with pytest.raises(ValueError, match="Unknown format"):
            resolve_config(desc)

    def test_per_block_without_block_size(self):
        desc = {"format": "int8", "granularity": "per_block"}
        with pytest.raises(ValueError, match="per_block granularity requires block_size"):
            resolve_config(desc)


# ===========================================================================
# QuantConfig: outlier_format / a_outlier_format fields
# ===========================================================================

class TestQuantConfigOutlierFormat:
    def test_outlier_format_default_none(self):
        cfg = QuantConfig()
        assert cfg.outlier_format is None
        assert cfg.a_outlier_format is None

    def test_outlier_format_stored(self):
        cfg = QuantConfig(outlier_format="int8")
        assert cfg.outlier_format == "int8"

    def test_invalid_outlier_format_raises(self):
        with pytest.raises(ValueError, match="Unknown outlier_format"):
            QuantConfig(outlier_format="no_such_format")

    def test_outlier_format_type_error(self):
        with pytest.raises(TypeError, match="outlier_format must be a string"):
            QuantConfig(outlier_format=42)  # type: ignore[arg-type]

    def test_a_outlier_format_stored(self):
        cfg = QuantConfig(a_outlier_format="fp8_e4m3")
        assert cfg.a_outlier_format == "fp8_e4m3"

    def test_a_outlier_format_weight_only_raises(self):
        with pytest.raises(ValueError, match="a_outlier_format.*weight_only"):
            QuantConfig(w_format="nf4", weight_only=True, a_outlier_format="int8")

    def test_outlier_format_without_sparse_ok(self):
        """outlier_format set but outlier_ratio=0 — valid, just unused at runtime."""
        cfg = QuantConfig(outlier_format="int8", outlier_ratio=0.0)
        assert cfg.outlier_format == "int8"
        assert cfg.outlier_ratio == 0.0

    def test_to_op_config_outlier_format_on_weight(self):
        """outlier_format → QuantScheme.outlier_format is set on weight and activation."""
        cfg = QuantConfig(w_format="int4", outlier_format="int8", outlier_ratio=0.1)
        result = cfg.to_op_config()
        assert result.weight.outlier_format is not None
        assert result.weight.outlier_format.name == "int8"
        assert result.input.outlier_format is not None
        assert result.input.outlier_format.name == "int8"

    def test_to_op_config_a_outlier_format_overrides(self):
        """a_outlier_format overrides outlier_format on activation scheme only."""
        cfg = QuantConfig(
            w_format="int4", a_format="int8",
            outlier_format="fp8_e4m3",
            a_outlier_format="nf4",
            outlier_ratio=0.1,
        )
        result = cfg.to_op_config()
        assert result.weight.outlier_format.name == "fp8_e4m3"
        assert result.input.outlier_format.name == "nf4"

    def test_to_op_config_outlier_format_none(self):
        """No outlier_format → QuantScheme.outlier_format is None."""
        cfg = QuantConfig(w_format="int4", outlier_ratio=0.1)
        result = cfg.to_op_config()
        assert result.weight.outlier_format is None
        assert result.input.outlier_format is None

    def test_a_outlier_format_falls_back_to_outlier_format(self):
        """a_outlier_format=None → activation follows outlier_format."""
        cfg = QuantConfig(w_format="int4", outlier_format="int8", outlier_ratio=0.1)
        result = cfg.to_op_config()
        assert result.input.outlier_format is not None
        assert result.input.outlier_format.name == "int8"


# ===========================================================================
# resolve_config() — outlier_format backward-compat
# ===========================================================================

class TestResolveConfigOutlierFormat:
    def test_outlier_format_in_descriptor(self):
        desc = {
            "format": "int4",
            "granularity": "per_tensor",
            "outlier_format": "int8",
            "outlier_ratio": 0.1,
        }
        result = resolve_config(desc)
        assert result.weight.outlier_format is not None
        assert result.weight.outlier_format.name == "int8"

    def test_a_outlier_format_in_descriptor(self):
        desc = {
            "format": "int4",
            "granularity": "per_tensor",
            "outlier_format": "fp8_e4m3",
            "a_outlier_format": "int8",
            "outlier_ratio": 0.1,
        }
        result = resolve_config(desc)
        assert result.weight.outlier_format.name == "fp8_e4m3"
        assert result.input.outlier_format.name == "int8"

    def test_outlier_format_non_string_raises(self):
        desc = {
            "format": "int4",
            "granularity": "per_tensor",
            "outlier_format": 42,
        }
        with pytest.raises(TypeError, match="'outlier_format' must be a string"):
            resolve_config(desc)

    def test_a_outlier_format_weight_only_raises(self):
        desc = {
            "format": "int4",
            "granularity": "per_tensor",
            "a_outlier_format": "int8",
            "weight_only": True,
        }
        with pytest.raises(ValueError, match="'a_outlier_format'.*weight_only"):
            resolve_config(desc)
