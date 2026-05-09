"""
NF4 operator equivalence tests — activation ops.

Since mx has no NF4, equivalence is verified against independent golden
reference implementations that replicate the exact quantization chain of
each activation operator.

Every vec_quantize / vec_add / vec_exp / ... call in the src operator is
replaced by: raw torch op → golden_lut_quantize(result, levels).
"""
import pytest
import torch
import torch.nn.functional as F

from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.ops.activations import (
    SigmoidFunction, TanhFunction, ReLUFunction, ReLU6Function,
    LeakyReLUFunction, SiLUFunction, GELUFunction,
)
from src.ops.vec_ops import vec_quantize

# ============================================================================
# Standalone golden quantization (no src/ imports)
# ============================================================================

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


def _golden_q(x, levels=None):
    """Golden nearest-neighbor LUT quantization (no src/ dependency)."""
    if levels is None:
        levels = NF4_LEVELS
    levels = levels.to(dtype=x.dtype, device=x.device)

    nan_mask = torch.isnan(x)
    x_safe = torch.where(nan_mask, torch.zeros_like(x), x)
    x_safe = torch.clamp(x_safe, -1.0, 1.0)

    d = torch.abs(x_safe.unsqueeze(-1) - levels.view(*([1] * x_safe.ndim), -1))
    indices = torch.argmin(d, dim=-1)
    result = levels[indices]

    if nan_mask.any():
        result = result.clone()
        result[nan_mask] = float("nan")
    return result


# ============================================================================
# Golden activation reference chains
# ============================================================================


def golden_sigmoid_fwd(x, levels):
    """Reproduce SigmoidFunction.forward quantization chain."""
    q_in = _golden_q(x, levels)
    exp_nx_q = _golden_q(torch.exp(-q_in), levels)
    add_q = _golden_q(exp_nx_q + 1.0, levels)
    return _golden_q(1.0 / add_q, levels)


def golden_sigmoid_bwd(grad_output, output, levels):
    """Reproduce SigmoidFunction.backward quantization chain."""
    go_q = _golden_q(grad_output, levels)
    temp = _golden_q(1.0 - output, levels)
    gs = _golden_q(output * temp, levels)
    return _golden_q(gs * go_q, levels)


def golden_tanh_fwd(x, levels):
    """Reproduce TanhFunction.forward quantization chain."""
    q_in = _golden_q(x, levels)
    return _golden_q(torch.tanh(q_in), levels)


def golden_tanh_bwd(grad_output, output, levels):
    """Reproduce TanhFunction.backward quantization chain."""
    go_q = _golden_q(grad_output, levels)
    output2 = _golden_q(output * output, levels)
    grad_tanh = _golden_q(1.0 - output2, levels)
    return _golden_q(grad_tanh * go_q, levels)


def golden_relu_fwd(x, levels):
    """Reproduce ReLUFunction.forward quantization chain (non-inplace)."""
    fp_out = torch.relu(x)
    return _golden_q(fp_out, levels)


def golden_relu_bwd(grad_output, output, levels):
    """Reproduce ReLUFunction.backward quantization chain."""
    mask = output > 0
    zs = torch.zeros([1], dtype=grad_output.dtype, device=grad_output.device)
    grad_input = torch.where(mask, grad_output, zs)
    return _golden_q(grad_input, levels)


def golden_relu6_fwd(x, levels):
    """Reproduce ReLU6Function.forward quantization chain (non-inplace)."""
    fp_out = F.relu6(x)
    return _golden_q(fp_out, levels)


def golden_relu6_bwd(grad_output, output, levels):
    """Reproduce ReLU6Function.backward quantization chain."""
    mask = torch.logical_and(output > 0, output < 6)
    zs = torch.zeros([1], dtype=grad_output.dtype, device=grad_output.device)
    grad_input = torch.where(mask, grad_output, zs)
    return _golden_q(grad_input, levels)


def golden_leaky_relu_fwd(x, levels, negative_slope=0.01):
    """Reproduce LeakyReLUFunction.forward quantization chain."""
    q_in = _golden_q(x, levels)
    out = F.leaky_relu(q_in, negative_slope=negative_slope)
    return _golden_q(out, levels)


def golden_leaky_relu_bwd(grad_output, output, levels, negative_slope=0.01):
    """Reproduce LeakyReLUFunction.backward quantization chain."""
    go_q = _golden_q(grad_output, levels)
    grad_neg = _golden_q(go_q * negative_slope, levels)
    mask = output > 0
    return torch.where(mask, go_q, grad_neg)  # NOT quantized after where


def golden_silu_fwd(x, levels):
    """Reproduce SiLUFunction.forward quantization chain."""
    q_in = _golden_q(x, levels)
    exp_nx = _golden_q(torch.exp(-q_in), levels)
    add_one = _golden_q(exp_nx + 1.0, levels)
    sig_x = _golden_q(1.0 / add_one, levels)
    return _golden_q(q_in * sig_x, levels), sig_x  # output, sig_x (for backward)


def golden_silu_bwd(grad_output, output, sig_x, levels):
    """Reproduce SiLUFunction.backward quantization chain."""
    go_q = _golden_q(grad_output, levels)
    temp = _golden_q(1.0 - sig_x, levels)
    temp = _golden_q(output * temp, levels)
    grad_silu = _golden_q(sig_x + temp, levels)
    return _golden_q(grad_silu * go_q, levels)


def golden_gelu_fwd(x, levels):
    """Reproduce GELUFunction.forward quantization chain (detailed: not first-order)."""
    q_in = _golden_q(x, levels)
    # s1 = x^2
    s1 = _golden_q(q_in * q_in, levels)
    # s2 = x^3
    s2 = _golden_q(s1 * q_in, levels)
    # s3 = 0.044677734 * x^3
    s3 = _golden_q(0.044677734 * s2, levels)
    # s4 = x + s3
    s4 = _golden_q(q_in + s3, levels)
    # s5 = 1.59375 * s4
    s5 = _golden_q(1.59375 * s4, levels)
    # phi = sigmoid(s5)
    phi = _golden_q(torch.exp(-s5), levels)
    phi = _golden_q(phi + 1.0, levels)
    phi = _golden_q(1.0 / phi, levels)
    # output = x * phi
    output = _golden_q(q_in * phi, levels)
    return output, q_in, phi


def golden_gelu_bwd(grad_output, x_q, phi, levels):
    """Reproduce GELUFunction.backward quantization chain (detailed)."""
    go_q = _golden_q(grad_output, levels)
    # dphi = phi * (1 - phi)
    dphi = _golden_q(1.0 - phi, levels)
    dphi = _golden_q(phi * dphi, levels)
    # dy = 1.59375 + 0.21386719 * x^2
    dy = _golden_q(x_q * x_q, levels)
    dy = _golden_q(0.21386719 * dy, levels)
    dy = _golden_q(1.59375 + dy, levels)
    # dphi *= dy
    dphi = _golden_q(dy * dphi, levels)
    # x_dphi = x * dphi
    x_dphi = _golden_q(x_q * dphi, levels)
    # grad_gelu = phi + x_dphi
    grad_gelu = _golden_q(phi + x_dphi, levels)
    # grad_input = grad_gelu * go
    return _golden_q(grad_gelu * go_q, levels)


# ============================================================================
# Shared helpers
# ============================================================================


def _make_input(shape=(4, 8), seed=42):
    """Generate test input. Scale=0.5 ensures sigmoid/tanh are not saturating."""
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32) * 0.5


def _nf4_per_tensor_cfg(quantize_backprop=True):
    """Build OpQuantConfig with NF4 per_tensor inner scheme."""
    nf4_scheme = QuantScheme.per_tensor("nf4")
    bw = nf4_scheme if quantize_backprop else None
    return OpQuantConfig(input=nf4_scheme, grad_input=bw)


# ============================================================================
# Sigmoid
# ============================================================================


class TestNF4Sigmoid:
    def test_forward_matches_golden(self):
        x = _make_input(seed=1)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)

        src_out = SigmoidFunction.apply(x, inner, qbp)
        golden = golden_sigmoid_fwd(x, NF4_LEVELS)
        assert torch.equal(src_out, golden)

    def test_backward_matches_golden(self):
        x = _make_input(seed=2)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)

        src_x = x.clone().requires_grad_(True)
        src_out = SigmoidFunction.apply(src_x, inner, qbp)
        src_out.sum().backward()

        # Golden backward: replicate the exact backward chain manually
        gold_grad = golden_sigmoid_bwd(torch.ones_like(src_out),
                                       src_out.detach(), NF4_LEVELS)
        assert torch.equal(src_x.grad, gold_grad), (
            f"Sigmoid backward mismatch\n"
            f"max diff: {(src_x.grad - gold_grad).abs().max()}"
        )


# ============================================================================
# Tanh
# ============================================================================


class TestNF4Tanh:
    def test_forward_matches_golden(self):
        x = _make_input(seed=3)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = TanhFunction.apply(x, inner, qbp)
        assert torch.equal(src_out, golden_tanh_fwd(x, NF4_LEVELS))

    def test_backward_matches_golden(self):
        x = _make_input(seed=4)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_x = x.clone().requires_grad_(True)
        src_out = TanhFunction.apply(src_x, inner, qbp)
        src_out.sum().backward()
        gold_grad = golden_tanh_bwd(torch.ones_like(src_out),
                                    src_out.detach(), NF4_LEVELS)
        assert torch.equal(src_x.grad, gold_grad)


# ============================================================================
# ReLU
# ============================================================================


class TestNF4ReLU:
    def test_forward_matches_golden(self):
        x = _make_input(seed=5)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = ReLUFunction.apply(x, False, inner, qbp)
        assert torch.equal(src_out, golden_relu_fwd(x, NF4_LEVELS))

    def test_backward_matches_golden(self):
        x = _make_input(seed=6)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_x = x.clone().requires_grad_(True)
        src_out = ReLUFunction.apply(src_x, False, inner, qbp)
        src_out.sum().backward()
        gold_grad = golden_relu_bwd(torch.ones_like(src_out),
                                    src_out.detach(), NF4_LEVELS)
        assert torch.equal(src_x.grad, gold_grad)


# ============================================================================
# ReLU6
# ============================================================================


class TestNF4ReLU6:
    def test_forward_matches_golden(self):
        x = _make_input(seed=7)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = ReLU6Function.apply(x, False, inner, qbp)
        assert torch.equal(src_out, golden_relu6_fwd(x, NF4_LEVELS))

    def test_backward_matches_golden(self):
        x = _make_input(seed=8)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_x = x.clone().requires_grad_(True)
        src_out = ReLU6Function.apply(src_x, False, inner, qbp)
        src_out.sum().backward()
        gold_grad = golden_relu6_bwd(torch.ones_like(src_out),
                                     src_out.detach(), NF4_LEVELS)
        assert torch.equal(src_x.grad, gold_grad)


# ============================================================================
# LeakyReLU
# ============================================================================


class TestNF4LeakyReLU:
    def test_forward_matches_golden(self):
        x = _make_input(seed=9)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = LeakyReLUFunction.apply(x, 0.01, False, inner, qbp)
        assert torch.equal(src_out, golden_leaky_relu_fwd(x, NF4_LEVELS))

    def test_backward_matches_golden(self):
        x = _make_input(seed=10)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_x = x.clone().requires_grad_(True)
        src_out = LeakyReLUFunction.apply(src_x, 0.01, False, inner, qbp)
        src_out.sum().backward()
        gold_grad = golden_leaky_relu_bwd(torch.ones_like(src_out),
                                          src_out.detach(), NF4_LEVELS)
        assert torch.equal(src_x.grad, gold_grad)


# ============================================================================
# SiLU
# ============================================================================


class TestNF4SiLU:
    def test_forward_matches_golden(self):
        x = _make_input(seed=11)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = SiLUFunction.apply(x, False, inner, qbp)
        golden_out, _ = golden_silu_fwd(x, NF4_LEVELS)
        assert torch.equal(src_out, golden_out)

    def test_backward_matches_golden(self):
        x = _make_input(seed=12)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_x = x.clone().requires_grad_(True)
        src_out = SiLUFunction.apply(src_x, False, inner, qbp)
        src_out.sum().backward()
        # sig_x from golden forward is valid because forward is bit-exact
        _, gold_sig = golden_silu_fwd(x, NF4_LEVELS)
        gold_grad = golden_silu_bwd(torch.ones_like(src_out),
                                    src_out.detach(), gold_sig, NF4_LEVELS)
        assert torch.equal(src_x.grad, gold_grad)


# ============================================================================
# GELU
# ============================================================================


class TestNF4GELU:
    def test_forward_matches_golden(self):
        x = _make_input(seed=13)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_out = GELUFunction.apply(x, inner, False, qbp)
        golden_out, _, _ = golden_gelu_fwd(x, NF4_LEVELS)
        assert torch.equal(src_out, golden_out)

    def test_backward_matches_golden(self):
        x = _make_input(seed=14)
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)
        src_x = x.clone().requires_grad_(True)
        src_out = GELUFunction.apply(src_x, inner, False, qbp)
        src_out.sum().backward()
        # q_in and phi from golden forward are valid because forward is bit-exact
        _, gold_q_in, gold_phi = golden_gelu_fwd(x, NF4_LEVELS)
        gold_grad = golden_gelu_bwd(torch.ones_like(src_out),
                                    gold_q_in, gold_phi, NF4_LEVELS)
        assert torch.equal(src_x.grad, gold_grad)


# ============================================================================
# STE mode
# ============================================================================


class TestNF4STE:
    """quantize_backprop=False → backward skips intermediate quantization."""

    def test_ste_forward_matches_quantized(self):
        """Forward is the same regardless of quantize_backprop setting."""
        x = _make_input(seed=20)
        nf4_scheme = QuantScheme.per_tensor("nf4")

        cfg_qbp = OpQuantConfig(input=nf4_scheme, grad_input=nf4_scheme)
        cfg_ste = OpQuantConfig(input=nf4_scheme, grad_input=None)

        out_qbp = SigmoidFunction.apply(x, cfg_qbp.input, True)
        out_ste = SigmoidFunction.apply(x, cfg_ste.input, False)
        assert torch.equal(out_qbp, out_ste)

    def test_ste_backward_vs_golden(self):
        """STE backward uses no intermediate quantization — verify against golden."""
        x = _make_input(seed=21)
        nf4_scheme = QuantScheme.per_tensor("nf4")
        cfg_ste = OpQuantConfig(input=nf4_scheme, grad_input=None)

        src_x = x.clone().requires_grad_(True)
        src_out = SigmoidFunction.apply(src_x, cfg_ste.input, False)
        src_out.sum().backward()

        # STE golden: backward uses raw ops (no quantization at each step)
        output = src_out.detach()
        ones = torch.ones_like(output)
        # grad_sigmoid = sigmoid(x) * (1 - sigmoid(x)), raw (no quantization)
        grad_sigmoid_ste = output * (1.0 - output)
        gold_grad = grad_sigmoid_ste * ones

        assert torch.equal(src_x.grad, gold_grad), (
            f"STE backward mismatch\n"
            f"max diff: {(src_x.grad - gold_grad).abs().max()}"
        )


# ============================================================================
# Per-channel NF4
# ============================================================================


class TestNF4PerChannelActivation:
    """Activation operators with per_channel NF4 quantization."""

    def test_sigmoid_forward_per_channel(self):
        """Per-channel NF4: each channel independently scaled."""
        x = _make_input(shape=(2, 8), seed=17)
        inner_scheme = QuantScheme.per_channel("nf4", axis=1)
        cfg = OpQuantConfig(input=inner_scheme, grad_input=inner_scheme)
        qbp = bool(cfg.grad_input)

        out = SigmoidFunction.apply(x, inner_scheme, qbp)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()

    def test_sigmoid_per_channel_vs_per_tensor_differ(self):
        """Per-channel and per-tensor NF4 produce different results."""
        x = _make_input(shape=(2, 8), seed=18)

        pc = QuantScheme.per_channel("nf4", axis=1)
        pt = QuantScheme.per_tensor("nf4")

        out_pc = SigmoidFunction.apply(
            x, pc, True
        )
        out_pt = SigmoidFunction.apply(
            x, pt, True
        )
        assert not torch.equal(out_pc, out_pt)
