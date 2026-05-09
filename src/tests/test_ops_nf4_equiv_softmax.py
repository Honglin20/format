"""
NF4 operator equivalence tests — Softmax.

Since mx has no NF4, equivalence is verified against independent golden
reference implementations that replicate the exact quantization chain.

Every vec_* call in the src operator is replaced by:
  raw torch op → golden_lut_quantize(result, levels).

Mathematical derivation documented inline.
"""
import torch

from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.ops.softmax import SoftmaxFunction

# ============================================================================
# Shared golden quantization
# ============================================================================

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)

LN_2_BF16 = 0.69140625


def _golden_q(x, levels=None):
    """Golden nearest-neighbor LUT quantization."""
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


Q = _golden_q


def _make_input(shape=(4, 8), seed=42):
    """Generate test input scaled to avoid extreme softmax saturation."""
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32) * 0.5


# ============================================================================
# Softmax golden
# ============================================================================
#
# Mathematical derivation — Softmax forward (NF4 per_tensor):
#
#   Given x ∈ R^(N,D), dim d:
#
#   Step 1:  q_in   = Q(x)                           [vec_quantize]
#   Step 2:  max_v  = max(x, dim, keepdim=True)      [torch.max — NOT quantized]
#   Step 3:  sft    = Q(q_in - max_v)                 [vec_sub]
#   Step 4:  exp_v  = Q(exp(sft))                    [vec_exp]
#   Step 5:  exp_s  = Q(Σ exp_v over dim)             [vec_reduce_sum = Q(sum)]
#   Step 6:  out    = Q(exp_v / exp_s)               [vec_div]
#
#   Note: vec_reduce_sum is Q(sum(x, dim)), NOT vec_reduce_mean (no div by N).
#
# Softmax backward (NF4 per_tensor):
#
#   Given grad_output go, saved output y:
#
#   Step 1:  go_q   = Q(go)                          [vec_quantize]
#   Step 2:  g1     = Q(go_q ⊙ y)                    [vec_mul]
#   Step 3:  g2     = Q(Σ g1 over dim)               [vec_reduce_sum]
#   Step 4:  g3     = Q(go_q - g2)                   [vec_sub]
#   Step 5:  g4     = Q(y ⊙ g3)                      [vec_mul]
#   [if softmax_exp2]:
#   Step 6:  grad   = Q(g4 * LN_2_BF16)              [vec_mul]
#
#   Mathematical justification (softmax gradient identity):
#     d/dx softmax(x)_i = y_i * (δ_ij - y_j)
#     where y = softmax(x)
#     So grad_input = y ⊙ (go - Σ(go ⊙ y)_j)
#     This is the standard softmax backward decomposed into quantized steps.


def golden_softmax_fwd(x, dim, levels, softmax_exp2=False):
    """Softmax forward with NF4 per_tensor quantization at every step."""
    # Step 1: Q(input)
    q_in = Q(x, levels)
    # Step 2: max (not quantized)
    max_data, _ = q_in.max(dim, keepdim=True)
    # Step 3: Q(q_in - max_data)
    shifted = Q(q_in - max_data, levels)
    # Step 4: Q(exp(shifted)) or Q(exp2(shifted))
    if softmax_exp2:
        exp_out = Q(torch.exp2(shifted), levels)
    else:
        exp_out = Q(torch.exp(shifted), levels)
    # Step 5: vec_reduce_sum → Q(sum(exp_out, dim))
    exp_sum = Q(torch.sum(exp_out, dim, keepdim=True), levels)
    # Step 6: Q(exp_out / exp_sum)
    output = Q(exp_out / exp_sum, levels)
    return output


def golden_softmax_bwd(grad_output, output, dim, levels, softmax_exp2=False):
    """Softmax backward with NF4 per_tensor quantization."""
    # Step 1: Q(go)
    go_q = Q(grad_output, levels)
    # Step 2: Q(go_q * output)
    g1 = Q(go_q * output, levels)
    # Step 3: vec_reduce_sum → Q(sum(g1, dim))
    g2 = Q(torch.sum(g1, dim, keepdim=True), levels)
    # Step 4: Q(go_q - g2)
    g3 = Q(go_q - g2, levels)
    # Step 5: Q(output * g3)
    g4 = Q(output * g3, levels)
    # Step 6: (if exp2) Q(g4 * LN_2_BF16)
    if softmax_exp2:
        g4 = Q(g4 * LN_2_BF16, levels)
    return g4


# ============================================================================
# Tests
# ============================================================================

def _nf4_per_tensor_cfg(quantize_backprop=True):
    nf4_scheme = QuantScheme.per_tensor("nf4")
    bw = nf4_scheme if quantize_backprop else None
    return OpQuantConfig(input=nf4_scheme, grad_input=bw)


class TestNF4Softmax:
    def test_forward_matches_golden(self):
        x = _make_input(seed=1)
        dim = -1
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)

        src_out = SoftmaxFunction.apply(x, dim, inner, False, qbp)
        golden = golden_softmax_fwd(x, dim, NF4_LEVELS)
        assert torch.equal(src_out, golden), (
            f"Softmax forward mismatch\n"
            f"max diff: {(src_out - golden).abs().max()}"
        )

    def test_backward_matches_golden(self):
        x = _make_input(seed=2)
        dim = -1
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)

        src_x = x.clone().requires_grad_(True)
        src_out = SoftmaxFunction.apply(src_x, dim, inner, False, qbp)
        src_out.sum().backward()

        gold_grad = golden_softmax_bwd(
            torch.ones_like(src_out), src_out.detach(), dim, NF4_LEVELS,
        )
        assert torch.equal(src_x.grad, gold_grad), (
            f"Softmax backward mismatch\n"
            f"max diff: {(src_x.grad - gold_grad).abs().max()}"
        )

    def test_forward_exp2(self):
        """Softmax with exp2 approximation."""
        x = _make_input(seed=3)
        dim = -1
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)

        src_out = SoftmaxFunction.apply(x, dim, inner, True, qbp)
        golden = golden_softmax_fwd(x, dim, NF4_LEVELS, softmax_exp2=True)
        assert torch.equal(src_out, golden), (
            f"Softmax exp2 forward mismatch\n"
            f"max diff: {(src_out - golden).abs().max()}"
        )

    def test_backward_exp2(self):
        """Softmax exp2 backward includes LN_2_BF16 multiplication."""
        x = _make_input(seed=4)
        dim = -1
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)

        src_x = x.clone().requires_grad_(True)
        src_out = SoftmaxFunction.apply(src_x, dim, inner, True, qbp)
        src_out.sum().backward()

        gold_grad = golden_softmax_bwd(
            torch.ones_like(src_out), src_out.detach(), dim, NF4_LEVELS,
            softmax_exp2=True,
        )
        assert torch.equal(src_x.grad, gold_grad), (
            f"Softmax exp2 backward mismatch\n"
            f"max diff: {(src_x.grad - gold_grad).abs().max()}"
        )

    def test_dim0(self):
        """Softmax along dim=0."""
        x = _make_input((8, 4), seed=5)
        dim = 0
        cfg = _nf4_per_tensor_cfg()
        inner = cfg.input
        qbp = bool(cfg.grad_input)

        src_out = SoftmaxFunction.apply(x, dim, inner, False, qbp)
        golden = golden_softmax_fwd(x, dim, NF4_LEVELS)
        assert torch.equal(src_out, golden)


# ============================================================================
# STE mode
# ============================================================================

class TestNF4SoftmaxSTE:
    def test_ste_forward_matches_quantized(self):
        """STE forward is identical to QBP forward."""
        x = _make_input(seed=10)
        dim = -1

        out_qbp = SoftmaxFunction.apply(
            x, dim, QuantScheme.per_tensor("nf4"), False, True,
        )
        out_ste = SoftmaxFunction.apply(
            x, dim, QuantScheme.per_tensor("nf4"), False, False,
        )
        assert torch.equal(out_qbp, out_ste)

    def test_ste_backward(self):
        """STE backward skips intermediate quantization."""
        x = _make_input(seed=11)
        dim = -1
        nf4 = QuantScheme.per_tensor("nf4")

        src_x = x.clone().requires_grad_(True)
        src_out = SoftmaxFunction.apply(src_x, dim, nf4, False, False)
        src_out.sum().backward()

        assert src_x.grad is not None
        assert torch.isfinite(src_x.grad).all()
        assert src_x.grad.abs().sum() > 0


# ============================================================================
# Per-channel NF4
# ============================================================================

class TestNF4SoftmaxPerChannel:
    def test_per_channel(self):
        x = _make_input((2, 8), seed=12)
        pc = QuantScheme.per_channel("nf4", axis=-1)

        out = SoftmaxFunction.apply(x, -1, pc, False, True)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()

    def test_per_channel_vs_per_tensor_differ(self):
        x = _make_input((2, 8), seed=13)
        pc = QuantScheme.per_channel("nf4", axis=-1)
        pt = QuantScheme.per_tensor("nf4")

        out_pc = SoftmaxFunction.apply(x.clone(), -1, pc, False, True)
        out_pt = SoftmaxFunction.apply(x.clone(), -1, pt, False, True)
        assert not torch.equal(out_pc, out_pt)
