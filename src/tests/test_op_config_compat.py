"""
Tests for op_config_from_mx_specs adapter — P3.1-c.
"""
import pytest

from src.tests._compat import op_config_from_mx_specs
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode


def test_empty_mx_specs_returns_empty_config():
    """No quantization keys set → all pipelines empty."""
    cfg = op_config_from_mx_specs({})
    assert cfg == OpQuantConfig()
    assert cfg.is_training is False


def test_bfloat_only_forward():
    """bfloat=16 sets elemwise schemes on forward fields only."""
    mx_specs = {"bfloat": 16}
    cfg = op_config_from_mx_specs(mx_specs)
    assert len(cfg.input) == 1
    assert len(cfg.weight) == 1
    assert len(cfg.bias) == 1
    assert len(cfg.output) == 2  # two elemwise casts (post-matmul + post-bias)
    # Backward should also be populated (quantize_backprop=True by default)
    assert cfg.is_training is True


def test_mx_format_forward():
    """a_elem_format + w_elem_format + block_size sets MX block schemes."""
    mx_specs = {
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e5m2",
        "block_size": 32,
    }
    cfg = op_config_from_mx_specs(mx_specs)
    # input should have: no elemwise (bfloat=0) + MX block
    assert len(cfg.input) == 1
    assert cfg.input[0].granularity.mode == GranularityMode.PER_BLOCK
    assert cfg.input[0].granularity.block_size == 32
    assert cfg.input[0].format_name == "fp8_e4m3"
    # weight should have: no elemwise + MX block
    assert len(cfg.weight) == 1
    assert cfg.weight[0].format_name == "fp8_e5m2"


def test_bfloat_plus_mx_forward():
    """bfloat + MX formats → input pipeline has 2 schemes (elem + MX)."""
    mx_specs = {
        "bfloat": 16,
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3",
        "block_size": 32,
    }
    cfg = op_config_from_mx_specs(mx_specs)
    assert len(cfg.input) == 2  # elem + MX
    assert len(cfg.weight) == 2
    assert cfg.input[0].granularity.mode == GranularityMode.PER_TENSOR
    assert cfg.input[1].granularity.mode == GranularityMode.PER_BLOCK


def test_quantize_backprop_false_no_backward():
    """quantize_backprop=False → no backward pipelines."""
    mx_specs = {"bfloat": 16, "quantize_backprop": False}
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.is_training is False
    assert cfg.grad_output == ()
    assert cfg.grad_input == ()


def test_backward_with_mx_formats():
    """MX formats in backward: grad_weight/grad_input gemm re-quantization."""
    mx_specs = {
        "bfloat": 16,
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3",
        "block_size": 32,
        "quantize_backprop": True,
    }
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.is_training is True
    # input_gw and grad_output_gw should have MX schemes
    assert len(cfg.input_gw) >= 1
    assert len(cfg.grad_output_gw) >= 1
    # weight_gi and grad_output_gi should have MX schemes
    assert len(cfg.weight_gi) >= 1
    assert len(cfg.grad_output_gi) >= 1


def test_backward_format_fallback():
    """a_elem_format_bp defaults to a_elem_format if not set."""
    mx_specs = {
        "bfloat": 16,
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "int8",
        "block_size": 32,
    }
    cfg = op_config_from_mx_specs(mx_specs)
    # a_elem_format_bp not set → falls back to a_elem_format for input_gw
    assert len(cfg.input_gw) >= 1
    assert cfg.input_gw[-1].format_name == "fp8_e4m3"


def test_op_type_param_accepted():
    """op_type parameter is accepted (currently same logic for linear/matmul)."""
    mx_specs = {"bfloat": 16}
    cfg_linear = op_config_from_mx_specs(mx_specs, op_type="linear")
    cfg_matmul = op_config_from_mx_specs(mx_specs, op_type="matmul")
    # For bfloat-only configs, both should produce the same result
    assert cfg_linear == cfg_matmul
