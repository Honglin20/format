"""
OpQuantConfig tests — P3.1-a.

Covers: construction, defaults, __post_init__ validation (every raise point),
is_training, frozen, hashing, equality, pipeline semantics.
"""
import pytest
from dataclasses import FrozenInstanceError

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s(fmt="fp8_e4m3", **kw):
    """Shorthand to build a QuantScheme for test fixtures."""
    return QuantScheme(format=fmt, **kw)


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

def test_op_config_default_construction():
    cfg = OpQuantConfig()
    for f in ("input", "weight", "bias", "output",
              "grad_output", "grad_input", "grad_weight", "grad_bias",
              "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi"):
        assert getattr(cfg, f) == (), f"Default {f} should be empty tuple"


def test_op_config_with_single_scheme():
    s = _s()
    cfg = OpQuantConfig(input=(s,))
    assert cfg.input == (s,)
    assert cfg.weight == ()


def test_op_config_with_pipeline():
    s1, s2 = _s("fp8_e4m3"), _s("int8")
    cfg = OpQuantConfig(input=(s1, s2))
    assert cfg.input == (s1, s2)


def test_op_config_all_forward_fields():
    si, sw, sb, so = _s(), _s(), _s(), _s()
    cfg = OpQuantConfig(input=(si,), weight=(sw,), bias=(sb,), output=(so,))
    assert cfg.input == (si,)
    assert cfg.weight == (sw,)
    assert cfg.bias == (sb,)
    assert cfg.output == (so,)


def test_op_config_all_backward_fields():
    s = _s()
    cfg = OpQuantConfig(
        grad_output=(s,), grad_input=(s,),
        grad_weight=(s,), grad_bias=(s,),
        input_gw=(s,), grad_output_gw=(s,),
        weight_gi=(s,), grad_output_gi=(s,),
    )
    assert all(getattr(cfg, f) == (s,) for f in (
        "grad_output", "grad_input", "grad_weight", "grad_bias",
        "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi",
    ))
    assert cfg.is_training is True


# ---------------------------------------------------------------------------
# is_training property
# ---------------------------------------------------------------------------

def test_is_training_false_when_no_backward():
    cfg = OpQuantConfig(input=(_s(),))
    assert cfg.is_training is False


def test_is_training_true_when_grad_output_set():
    cfg = OpQuantConfig(grad_output=(_s(),))
    assert cfg.is_training is True


def test_is_training_true_when_grad_input_set():
    cfg = OpQuantConfig(grad_input=(_s(),))
    assert cfg.is_training is True


def test_is_training_true_when_grad_weight_set():
    cfg = OpQuantConfig(grad_weight=(_s(),))
    assert cfg.is_training is True


def test_is_training_true_when_grad_bias_set():
    cfg = OpQuantConfig(grad_bias=(_s(),))
    assert cfg.is_training is True


def test_is_training_true_when_input_gw_set():
    cfg = OpQuantConfig(input_gw=(_s(),))
    assert cfg.is_training is True


def test_is_training_true_when_grad_output_gw_set():
    cfg = OpQuantConfig(grad_output_gw=(_s(),))
    assert cfg.is_training is True


def test_is_training_true_when_weight_gi_set():
    cfg = OpQuantConfig(weight_gi=(_s(),))
    assert cfg.is_training is True


def test_is_training_true_when_grad_output_gi_set():
    cfg = OpQuantConfig(grad_output_gi=(_s(),))
    assert cfg.is_training is True


def test_is_training_forward_only_config():
    s = _s()
    cfg = OpQuantConfig(input=(s,), weight=(s,), output=(s,))
    assert cfg.is_training is False


# ---------------------------------------------------------------------------
# __post_init__ validation — field type (each of 12 fields)
# ---------------------------------------------------------------------------

_FORWARD_FIELDS = ["input", "weight", "bias", "output"]
_BACKWARD_FIELDS = ["grad_output", "grad_input", "grad_weight", "grad_bias",
                    "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi"]
_ALL_FIELDS = _FORWARD_FIELDS + _BACKWARD_FIELDS


@pytest.mark.parametrize("field_name", _ALL_FIELDS)
def test_field_rejects_non_tuple(field_name):
    with pytest.raises(TypeError, match=f"OpQuantConfig.{field_name} must be tuple"):
        OpQuantConfig(**{field_name: "not_a_tuple"})


@pytest.mark.parametrize("field_name", _ALL_FIELDS)
def test_field_rejects_list(field_name):
    with pytest.raises(TypeError, match=f"OpQuantConfig.{field_name} must be tuple"):
        OpQuantConfig(**{field_name: [_s()]})


@pytest.mark.parametrize("field_name", _ALL_FIELDS)
def test_field_rejects_none(field_name):
    with pytest.raises(TypeError, match=f"OpQuantConfig.{field_name} must be tuple"):
        OpQuantConfig(**{field_name: None})


@pytest.mark.parametrize("field_name", _ALL_FIELDS)
def test_field_rejects_non_quant_scheme_element(field_name):
    with pytest.raises(TypeError, match=f"OpQuantConfig.{field_name}\\[0\\] must be QuantScheme"):
        OpQuantConfig(**{field_name: ("not_a_scheme",)})


@pytest.mark.parametrize("field_name", _ALL_FIELDS)
def test_field_rejects_int_element(field_name):
    with pytest.raises(TypeError, match=f"OpQuantConfig.{field_name}\\[0\\] must be QuantScheme"):
        OpQuantConfig(**{field_name: (123,)})


@pytest.mark.parametrize("field_name", _ALL_FIELDS)
def test_field_rejects_mixed_element_in_pipeline(field_name):
    s = _s()
    with pytest.raises(TypeError, match=f"OpQuantConfig.{field_name}\\[1\\] must be QuantScheme"):
        OpQuantConfig(**{field_name: (s, "bad")})


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

def test_op_config_frozen():
    cfg = OpQuantConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.input = (_s(),)


# ---------------------------------------------------------------------------
# Equality & hashing
# ---------------------------------------------------------------------------

def test_op_config_equality():
    s = _s()
    a = OpQuantConfig(input=(s,))
    b = OpQuantConfig(input=(s,))
    assert a == b


def test_op_config_inequality():
    a = OpQuantConfig(input=(_s("fp8_e4m3"),))
    b = OpQuantConfig(input=(_s("int8"),))
    assert a != b


def test_op_config_hash_equal():
    s = _s()
    a = OpQuantConfig(input=(s,))
    b = OpQuantConfig(input=(s,))
    assert hash(a) == hash(b)


def test_op_config_hash_usable_in_set():
    s = _s()
    a = OpQuantConfig(input=(s,))
    b = OpQuantConfig(weight=(s,))
    assert len({a, b}) == 2


def test_op_config_hash_usable_in_dict():
    s = _s()
    cfg = OpQuantConfig(input=(s,))
    d = {cfg: "value"}
    assert d[cfg] == "value"


# ---------------------------------------------------------------------------
# Pipeline ordering matters
# ---------------------------------------------------------------------------

def test_pipeline_order_matters_for_equality():
    s1 = _s("fp8_e4m3")
    s2 = _s("int8")
    a = OpQuantConfig(input=(s1, s2))
    b = OpQuantConfig(input=(s2, s1))
    assert a != b


# ---------------------------------------------------------------------------
# Empty pipeline is identity (no quantization)
# ---------------------------------------------------------------------------

def test_empty_pipeline_means_no_quantization():
    cfg = OpQuantConfig(input=(), weight=(), bias=(), output=())
    for s in cfg.input:
        pytest.fail("Should not iterate over empty tuple")
    assert cfg.is_training is False


# ---------------------------------------------------------------------------
# Activation module OpQuantConfig integration (M1 fix)
# ---------------------------------------------------------------------------


def test_activation_cfg_from_inner_scheme():
    """QuantizedSigmoid(inner_scheme=s) converts to cfg.input == (s,)."""
    from src.ops.activations import QuantizedSigmoid
    s = _s("fp8_e4m3")
    mod = QuantizedSigmoid(inner_scheme=s)
    assert isinstance(mod.cfg, OpQuantConfig)
    assert mod.cfg.input == (s,)
    assert mod.cfg.grad_input == (s,)  # quantize_backprop=True by default


def test_activation_cfg_from_opquantconfig():
    """QuantizedSigmoid(cfg=OpQuantConfig(input=(s,))) accepts cfg directly."""
    from src.ops.activations import QuantizedSigmoid
    s = _s("fp8_e4m3")
    cfg = OpQuantConfig(input=(s,))
    mod = QuantizedSigmoid(cfg=cfg)
    assert mod.cfg is cfg


def test_activation_cfg_both_raises():
    """Passing both cfg and inner_scheme raises ValueError."""
    from src.ops.activations import QuantizedSigmoid
    s = _s("fp8_e4m3")
    cfg = OpQuantConfig(input=(s,))
    with pytest.raises(ValueError, match="cfg.*inner_scheme"):
        QuantizedSigmoid(cfg=cfg, inner_scheme=s)


def test_activation_passthrough_empty_cfg():
    """QuantizedSigmoid(cfg=OpQuantConfig()) is passthrough = same as nn.Sigmoid."""
    import torch
    from src.ops.activations import QuantizedSigmoid
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    mod = QuantizedSigmoid(cfg=OpQuantConfig())
    out = mod(x)
    expected = torch.sigmoid(x)
    assert torch.equal(out, expected)


def test_activation_is_training_from_cfg():
    """cfg.grad_input non-empty → cfg.is_training == True."""
    s = _s("fp8_e4m3")
    cfg = OpQuantConfig(input=(s,), grad_input=(s,))
    assert cfg.is_training is True
    cfg_no_bw = OpQuantConfig(input=(s,))
    assert cfg_no_bw.is_training is False
