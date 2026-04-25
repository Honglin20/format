"""
Bit-exact equivalence tests: src/ops/norm.py vs mx/{batchnorm,layernorm,groupnorm}.py — P3.3.

All comparisons use torch.equal (bit-exact, no atol/rtol).
"""
import pytest
import torch

import mx
from mx.specs import apply_mx_specs

from src.ops.norm import (BatchNormFunction, LayerNormFunction,
                          GroupNormFunction, RMSNormFunction,
                          QuantizedBatchNorm2d, QuantizedLayerNorm,
                          QuantizedGroupNorm, QuantizedRMSNorm)
from src.scheme.op_config import OpQuantConfig
from src.tests._compat import norm_config_from_mx_specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_bit_exact(mx_out, src_out, label="output"):
    if mx_out is None and src_out is None:
        return
    assert mx_out is not None and src_out is not None, f"{label}: one is None"
    assert torch.equal(mx_out, src_out), (
        f"{label}: not bit-exact (max diff={torch.max(torch.abs(mx_out - src_out))})"
    )


MX_SPECS_NORM = [
    pytest.param("bfloat16", {"bfloat": 16}, id="bf16"),
    pytest.param("bfloat10", {"bfloat": 10}, id="bf10"),
]


# ---------------------------------------------------------------------------
# BatchNorm tests
# ---------------------------------------------------------------------------

def _make_bn_tensors(batch=4, channels=8, h=6, w=6, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(batch, channels, h, w, dtype=torch.float32)
    weight = torch.randn(channels, dtype=torch.float32)
    bias = torch.randn(channels, dtype=torch.float32)
    return x, weight, bias


def _run_mx_batchnorm(x, weight, bias, mx_specs, training=True, eps=1e-5,
                      momentum=0.1, track_running_stats=True):
    mx_specs = apply_mx_specs(mx_specs)
    mx_mod = mx.BatchNorm2d(
        x.shape[1], eps=eps, momentum=momentum,
        track_running_stats=track_running_stats,
        mx_specs=mx_specs,
    )
    mx_mod.weight.data.copy_(weight)
    mx_mod.bias.data.copy_(bias)
    mx_mod = mx_mod.to(x.dtype)
    if training:
        mx_mod.train()
    else:
        mx_mod.eval()

    x = x.clone().requires_grad_(True)
    mx_mod.weight.requires_grad_(True)
    mx_mod.bias.requires_grad_(True)

    out = mx_mod(x)
    out.sum().backward()

    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        mx_mod.weight.grad.detach() if mx_mod.weight.grad is not None else None,
        mx_mod.bias.grad.detach() if mx_mod.bias.grad is not None else None,
    )


def _run_src_batchnorm(x, weight, bias, cfg, inner_scheme, training=True,
                       eps=1e-5, momentum=0.1, track_running_stats=True):
    running_mean = torch.zeros(x.shape[1]) if track_running_stats else None
    running_var = torch.ones(x.shape[1]) if track_running_stats else None

    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    bias = bias.clone().requires_grad_(True)

    out = BatchNormFunction.apply(
        x, running_mean, running_var, weight, bias,
        training, momentum, eps,
        cfg, inner_scheme,
    )
    out.sum().backward()
    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        weight.grad.detach() if weight.grad is not None else None,
        bias.grad.detach() if bias.grad is not None else None,
    )


class TestBatchNorm2d:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_forward_training(self, name, mx_specs):
        x, w, b = _make_bn_tensors()
        mx_out, _, _, _ = _run_mx_batchnorm(x, w, b, mx_specs, training=True)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="batch_norm")
        src_out, _, _, _ = _run_src_batchnorm(x, w, b, cfg, inner, training=True)
        _assert_bit_exact(mx_out, src_out, label=f"bn-fwd-train ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_input(self, name, mx_specs):
        x, w, b = _make_bn_tensors()
        _, mx_gi, _, _ = _run_mx_batchnorm(x, w, b, mx_specs, training=True)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="batch_norm")
        _, src_gi, _, _ = _run_src_batchnorm(x, w, b, cfg, inner, training=True)
        _assert_bit_exact(mx_gi, src_gi, label=f"bn-grad_input ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_weight(self, name, mx_specs):
        x, w, b = _make_bn_tensors()
        _, _, mx_gw, _ = _run_mx_batchnorm(x, w, b, mx_specs, training=True)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="batch_norm")
        _, _, src_gw, _ = _run_src_batchnorm(x, w, b, cfg, inner, training=True)
        _assert_bit_exact(mx_gw, src_gw, label=f"bn-grad_weight ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_bias(self, name, mx_specs):
        x, w, b = _make_bn_tensors()
        _, _, _, mx_gb = _run_mx_batchnorm(x, w, b, mx_specs, training=True)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="batch_norm")
        _, _, _, src_gb = _run_src_batchnorm(x, w, b, cfg, inner, training=True)
        _assert_bit_exact(mx_gb, src_gb, label=f"bn-grad_bias ({name})")


# ---------------------------------------------------------------------------
# LayerNorm tests
# ---------------------------------------------------------------------------

def _make_ln_tensors(batch=4, seq=8, hidden=16, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    weight = torch.randn(hidden, dtype=torch.float32)
    bias = torch.randn(hidden, dtype=torch.float32)
    return x, weight, bias


def _run_mx_layernorm(x, weight, bias, mx_specs, eps=1e-5):
    mx_specs = apply_mx_specs(mx_specs)
    mx_mod = mx.LayerNorm(x.shape[-1], eps=eps, mx_specs=mx_specs)
    mx_mod.weight.data.copy_(weight)
    mx_mod.bias.data.copy_(bias)

    x = x.clone().requires_grad_(True)
    mx_mod.weight.requires_grad_(True)
    mx_mod.bias.requires_grad_(True)

    out = mx_mod(x)
    out.sum().backward()

    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        mx_mod.weight.grad.detach() if mx_mod.weight.grad is not None else None,
        mx_mod.bias.grad.detach() if mx_mod.bias.grad is not None else None,
    )


def _run_src_layernorm(x, weight, bias, cfg, inner_scheme, eps=1e-5):
    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    bias = bias.clone().requires_grad_(True)

    out = LayerNormFunction.apply(x, weight, bias, eps, cfg, inner_scheme)
    out.sum().backward()
    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        weight.grad.detach() if weight.grad is not None else None,
        bias.grad.detach() if bias.grad is not None else None,
    )


class TestLayerNorm:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_forward(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        mx_out, _, _, _ = _run_mx_layernorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="layer_norm")
        src_out, _, _, _ = _run_src_layernorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_out, src_out, label=f"ln-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_input(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        _, mx_gi, _, _ = _run_mx_layernorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="layer_norm")
        _, src_gi, _, _ = _run_src_layernorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_gi, src_gi, label=f"ln-grad_input ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_weight(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        _, _, mx_gw, _ = _run_mx_layernorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="layer_norm")
        _, _, src_gw, _ = _run_src_layernorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_gw, src_gw, label=f"ln-grad_weight ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_bias(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        _, _, _, mx_gb = _run_mx_layernorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="layer_norm")
        _, _, _, src_gb = _run_src_layernorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_gb, src_gb, label=f"ln-grad_bias ({name})")


# ---------------------------------------------------------------------------
# GroupNorm tests
# ---------------------------------------------------------------------------

def _make_gn_tensors(batch=4, channels=8, h=6, w=6, num_groups=2, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(batch, channels, h, w, dtype=torch.float32)
    weight = torch.randn(channels, dtype=torch.float32)
    bias = torch.randn(channels, dtype=torch.float32)
    return x, weight, bias, num_groups


def _run_mx_groupnorm(x, weight, bias, num_groups, mx_specs, eps=1e-5):
    mx_specs = apply_mx_specs(mx_specs)
    mx_mod = mx.GroupNorm(num_groups, x.shape[1], eps=eps, mx_specs=mx_specs)
    mx_mod.weight.data.copy_(weight)
    mx_mod.bias.data.copy_(bias)

    x = x.clone().requires_grad_(True)
    mx_mod.weight.requires_grad_(True)
    mx_mod.bias.requires_grad_(True)

    out = mx_mod(x)
    out.sum().backward()

    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        mx_mod.weight.grad.detach() if mx_mod.weight.grad is not None else None,
        mx_mod.bias.grad.detach() if mx_mod.bias.grad is not None else None,
    )


def _run_src_groupnorm(x, weight, bias, num_groups, cfg, inner_scheme, eps=1e-5):
    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    bias = bias.clone().requires_grad_(True)

    out = GroupNormFunction.apply(
        x, num_groups, weight, bias, eps, cfg, inner_scheme,
    )
    out.sum().backward()
    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        weight.grad.detach() if weight.grad is not None else None,
        bias.grad.detach() if bias.grad is not None else None,
    )


class TestGroupNorm:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_forward(self, name, mx_specs):
        x, w, b, ng = _make_gn_tensors()
        mx_out, _, _, _ = _run_mx_groupnorm(x, w, b, ng, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="group_norm")
        src_out, _, _, _ = _run_src_groupnorm(x, w, b, ng, cfg, inner)
        _assert_bit_exact(mx_out, src_out, label=f"gn-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_input(self, name, mx_specs):
        x, w, b, ng = _make_gn_tensors()
        _, mx_gi, _, _ = _run_mx_groupnorm(x, w, b, ng, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="group_norm")
        _, src_gi, _, _ = _run_src_groupnorm(x, w, b, ng, cfg, inner)
        _assert_bit_exact(mx_gi, src_gi, label=f"gn-grad_input ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_weight(self, name, mx_specs):
        x, w, b, ng = _make_gn_tensors()
        _, _, mx_gw, _ = _run_mx_groupnorm(x, w, b, ng, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="group_norm")
        _, _, src_gw, _ = _run_src_groupnorm(x, w, b, ng, cfg, inner)
        _assert_bit_exact(mx_gw, src_gw, label=f"gn-grad_weight ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_bias(self, name, mx_specs):
        x, w, b, ng = _make_gn_tensors()
        _, _, _, mx_gb = _run_mx_groupnorm(x, w, b, ng, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="group_norm")
        _, _, _, src_gb = _run_src_groupnorm(x, w, b, ng, cfg, inner)
        _assert_bit_exact(mx_gb, src_gb, label=f"gn-grad_bias ({name})")


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------

def _run_mx_rmsnorm(x, weight, bias, mx_specs, eps=1e-5):
    mx_specs = apply_mx_specs(mx_specs)
    mx_mod = mx.RMSNorm(x.shape[-1], eps=eps, mx_specs=mx_specs)
    mx_mod.weight.data.copy_(weight)
    mx_mod.bias.data.copy_(bias)

    x = x.clone().requires_grad_(True)
    mx_mod.weight.requires_grad_(True)
    mx_mod.bias.requires_grad_(True)

    out = mx_mod(x)
    out.sum().backward()

    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        mx_mod.weight.grad.detach() if mx_mod.weight.grad is not None else None,
        mx_mod.bias.grad.detach() if mx_mod.bias.grad is not None else None,
    )


def _run_src_rmsnorm(x, weight, bias, cfg, inner_scheme, eps=1e-5):
    x = x.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    bias = bias.clone().requires_grad_(True)

    out = RMSNormFunction.apply(x, weight, bias, eps, cfg, inner_scheme)
    out.sum().backward()
    return (
        out.detach(),
        x.grad.detach() if x.grad is not None else None,
        weight.grad.detach() if weight.grad is not None else None,
        bias.grad.detach() if bias.grad is not None else None,
    )


class TestRMSNorm:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_forward(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        mx_out, _, _, _ = _run_mx_rmsnorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="rms_norm")
        src_out, _, _, _ = _run_src_rmsnorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_out, src_out, label=f"rms-fwd ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_input(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        _, mx_gi, _, _ = _run_mx_rmsnorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="rms_norm")
        _, src_gi, _, _ = _run_src_rmsnorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_gi, src_gi, label=f"rms-grad_input ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_weight(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        _, _, mx_gw, _ = _run_mx_rmsnorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="rms_norm")
        _, _, src_gw, _ = _run_src_rmsnorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_gw, src_gw, label=f"rms-grad_weight ({name})")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS_NORM)
    def test_backward_grad_bias(self, name, mx_specs):
        x, w, b = _make_ln_tensors()
        _, _, _, mx_gb = _run_mx_rmsnorm(x, w, b, mx_specs)
        cfg, inner = norm_config_from_mx_specs(mx_specs, op_type="rms_norm")
        _, _, _, src_gb = _run_src_rmsnorm(x, w, b, cfg, inner)
        _assert_bit_exact(mx_gb, src_gb, label=f"rms-grad_bias ({name})")
