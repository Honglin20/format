"""
Tests for op_config_from_mx_specs adapter — two-level model.
"""
import pytest

from src.tests._compat import op_config_from_mx_specs
from src.scheme.op_config import OpQuantConfig


def test_empty_mx_specs_returns_empty_config():
    """No quantization keys set → all fields None."""
    cfg = op_config_from_mx_specs({})
    assert cfg == OpQuantConfig()
    assert cfg.is_training is False


def test_bfloat_only_forward():
    """bfloat=16 sets storage + elemwise backward."""
    mx_specs = {"bfloat": 16}
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.storage is not None  # bf16 elemwise
    assert cfg.input is None        # no MX format
    assert cfg.weight is None
    assert cfg.bias is None
    assert cfg.output is None
    assert cfg.is_training is True


def test_mx_format_forward():
    """a_elem_format + w_elem_format + block_size sets MX compute schemes."""
    mx_specs = {
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e5m2",
        "block_size": 32,
    }
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.storage is None  # no bfloat/fp
    assert cfg.input is not None
    assert cfg.input.granularity.block_size == 32
    assert cfg.input.format_name == "fp8_e4m3"
    assert cfg.weight is not None
    assert cfg.weight.format_name == "fp8_e5m2"


def test_bfloat_plus_mx_forward():
    """bfloat + MX formats → storage + per-role compute."""
    mx_specs = {
        "bfloat": 16,
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3",
        "block_size": 32,
    }
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.storage is not None  # bf16 elemwise
    assert cfg.input is not None    # fp8 MX compute
    assert cfg.weight is not None


def test_quantize_backprop_false_no_backward():
    """quantize_backprop=False → no backward fields."""
    mx_specs = {"bfloat": 16, "quantize_backprop": False}
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.is_training is False
    assert cfg.grad_output is None
    assert cfg.grad_input is None


def test_backward_with_mx_formats():
    """MX formats in backward: grad_weight/grad_input gemm re-quantization."""
    mx_specs = {
        "bfloat": 16,
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3",
        "block_size": 32,
        "quantize_backprop": True,
        "a_elem_format_bp": "fp8_e4m3",
        "a_elem_format_bp_ex": "fp8_e4m3",
        "a_elem_format_bp_os": "fp8_e4m3",
        "w_elem_format_bp": "fp8_e4m3",
    }
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.is_training is True
    assert cfg.input_gw is not None
    assert cfg.grad_output_gw is not None
    assert cfg.weight_gi is not None
    assert cfg.grad_output_gi is not None


def test_backward_format_fallback():
    """a_elem_format_bp explicitly set → used for input_gw."""
    mx_specs = {
        "bfloat": 16,
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "int8",
        "block_size": 32,
        "a_elem_format_bp": "fp8_e4m3",
        "a_elem_format_bp_ex": "fp8_e4m3",
        "w_elem_format_bp": "int8",
    }
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.input_gw is not None
    assert cfg.input_gw.format_name == "fp8_e4m3"


def test_no_bp_keys_means_no_mx_backward():
    """Without _bp keys, backward gemm has no MX schemes."""
    mx_specs = {
        "bfloat": 16,
        "a_elem_format": "fp8_e4m3",
        "w_elem_format": "fp8_e4m3",
        "block_size": 32,
    }
    cfg = op_config_from_mx_specs(mx_specs)
    assert cfg.input_gw is None
    assert cfg.grad_output_gw is None
    assert cfg.weight_gi is None
    assert cfg.grad_output_gi is None
    assert cfg.is_training is True  # elemwise backward still present


def test_op_type_param_accepted():
    """op_type parameter is accepted."""
    mx_specs = {"bfloat": 16}
    cfg_linear = op_config_from_mx_specs(mx_specs, op_type="linear")
    cfg_matmul = op_config_from_mx_specs(mx_specs, op_type="matmul")
    assert cfg_linear == cfg_matmul


def test_matmul_grad_weight_round_key():
    """matmul grad_weight respects round_grad_weight, not round_grad_input."""
    mx_specs = {
        "bfloat": 16,
        "round_grad_weight": "floor",
        "round_grad_input": "even",
    }
    cfg = op_config_from_mx_specs(mx_specs, op_type="matmul")
    assert cfg.grad_weight is not None
    assert cfg.grad_weight.round_mode == "floor"
    assert cfg.grad_input is not None
    assert cfg.grad_input.round_mode == "even"
