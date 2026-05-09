"""
NF4 operator equivalence tests — SIMD elementwise ops.

Since mx has no NF4, equivalence is verified against independent golden
reference implementations that replicate the exact quantization chain.

Every vec_* call in the src operator is replaced by:
  raw torch op → golden_lut_quantize(result, levels).

Mathematical derivation per op is documented inline.
"""
import torch

from src.scheme.quant_scheme import QuantScheme
from src.ops.elemwise import (
    SIMDAdd, SIMDSub, SIMDMul, SIMDDiv,
    SIMDSplit, SIMDSquare, SIMDSqrt, SIMDExp, SIMDLog,
    SIMDReduceSum,
)

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
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32) * 0.5


NF4_SCHEME = QuantScheme.per_tensor("nf4")


# ============================================================================
# SIMD golden reference functions
# ============================================================================
#
# Each SIMD op follows a consistent pattern:
#   Forward:  Q(in1), Q(in2) → raw torch op → Q(result)
#   Backward: Q(grad_output) → compute gradient → Q(intermediate) → ...
#
# The golden functions replicate this exact chain.

# ---------------------------------------------------------------------------
# Add: y = x1 + x2, dy/dx1 = 1, dy/dx2 = 1
# ---------------------------------------------------------------------------

def golden_add_fwd(in1, in2, levels):
    """SIMDAdd forward: Q(Q(in1) + Q(in2))."""
    q1 = Q(in1, levels)
    q2 = Q(in2, levels)
    return Q(q1 + q2, levels)


def golden_add_bwd(g, in_shape, levels):
    """SIMDAdd backward: broadcast Q(g) to input shape."""
    g_q = Q(g, levels)
    if list(g_q.shape) == list(in_shape):
        return g_q
    # _broadcast_gradient: sum over extra dims, quantize sum
    reduce_dims = []
    for i in range(g_q.ndim):
        if i + 1 > len(in_shape):
            reduce_dims.append(-1 - i)
        else:
            if g_q.shape[-1 - i] != in_shape[-1 - i]:
                reduce_dims.append(-1 - i)
    if reduce_dims:
        g_sum = torch.sum(g_q, dim=reduce_dims)
        return Q(g_sum.view(in_shape), levels)
    return g_q.view(in_shape)


# ---------------------------------------------------------------------------
# Sub: y = x1 - x2, dy/dx1 = 1, dy/dx2 = -1
# ---------------------------------------------------------------------------

def golden_sub_fwd(in1, in2, levels):
    """SIMDSub forward: Q(Q(in1) - Q(in2))."""
    q1 = Q(in1, levels)
    q2 = Q(in2, levels)
    return Q(q1 - q2, levels)


# ---------------------------------------------------------------------------
# Mul: y = x1 * x2, dy/dx1 = x2, dy/dx2 = x1
# ---------------------------------------------------------------------------

def golden_mul_fwd(in1, in2, levels):
    """SIMDMul forward: Q(Q(in1) * Q(in2))."""
    q1 = Q(in1, levels)
    q2 = Q(in2, levels)
    return Q(q1 * q2, levels), q1, q2


def golden_mul_bwd(g, in1_q, in2_q, in1_shape, in2_shape, levels):
    """SIMDMul backward: Q(Q(g) * in2_q) and Q(Q(g) * in1_q), then broadcast."""
    g_q = Q(g, levels)
    g1 = Q(g_q * in2_q, levels)
    g2 = Q(g_q * in1_q, levels)
    g1 = golden_add_bwd(g1, in1_shape, levels)
    g2 = golden_add_bwd(g2, in2_shape, levels)
    return g1, g2


# ---------------------------------------------------------------------------
# Div: y = x1 / x2, dy/dx1 = 1/x2, dy/dx2 = -x1/x2^2 = -y/x2
# ---------------------------------------------------------------------------

def golden_div_fwd(in1, in2, levels):
    """SIMDDiv forward: Q(Q(in1) / Q(in2))."""
    q1 = Q(in1, levels)
    q2 = Q(in2, levels)
    out = Q(q1 / q2, levels)
    return out, q2


def golden_div_bwd(g, out, in2_q, in1_shape, in2_shape, levels):
    """SIMDDiv backward: Q(g/in2) for g1, Q(g * Q(-out/in2)) for g2."""
    g_q = Q(g, levels)
    # g1 = Q(g / in2), then broadcast
    g1 = Q(g_q / in2_q, levels)
    g1 = golden_add_bwd(g1, in1_shape, levels)
    # g2_q = Q(-out / in2), then g2 = Q(g * g2_q), then broadcast
    g2_q = Q((-out) / in2_q, levels)
    g2 = Q(g_q * g2_q, levels)
    g2 = golden_add_bwd(g2, in2_shape, levels)
    return g1, g2


# ---------------------------------------------------------------------------
# Square: y = x^2, dy/dx = 2x
# ---------------------------------------------------------------------------

def golden_square_fwd(in1, levels):
    """SIMDSquare forward: Q(Q(in1)^2)."""
    q1 = Q(in1, levels)
    return Q(q1 ** 2, levels), q1


def golden_square_bwd(g, x_q, levels):
    """SIMDSquare backward: Q(Q(g) * Q(x * 2))."""
    g_q = Q(g, levels)
    x2 = Q(x_q * 2, levels)
    return Q(g_q * x2, levels)


# ---------------------------------------------------------------------------
# Sqrt: y = sqrt(x), dy/dx = 0.5/sqrt(x)
# ---------------------------------------------------------------------------

def golden_sqrt_fwd(in1, levels):
    """SIMDSqrt forward: Q(sqrt(Q(in1)))."""
    q1 = Q(in1, levels)
    out = Q(torch.sqrt(q1), levels)
    return out


def golden_sqrt_bwd(g, sqrt_x, levels):
    """SIMDSqrt backward: Q(Q(g) * 0.5 / sqrt_x) = Q(Q(Q(g) * 0.5) / sqrt_x)."""
    g_q = Q(g, levels)
    g_half = Q(g_q * 0.5, levels)
    return Q(g_half / sqrt_x, levels)


# ---------------------------------------------------------------------------
# Exp: y = e^x, dy/dx = e^x = y
# ---------------------------------------------------------------------------

def golden_exp_fwd(in1, levels):
    """SIMDExp forward: Q(exp(Q(in1)))."""
    q1 = Q(in1, levels)
    out = Q(torch.exp(q1), levels)
    return out


def golden_exp_bwd(g, exp_x, levels):
    """SIMDExp backward: Q(Q(g) * exp_x)."""
    g_q = Q(g, levels)
    return Q(g_q * exp_x, levels)


# ---------------------------------------------------------------------------
# Log: y = ln(x), dy/dx = 1/x
# ---------------------------------------------------------------------------

def golden_log_fwd(in1, levels):
    """SIMDLog forward: Q(ln(Q(in1)))."""
    q1 = Q(in1, levels)
    out = torch.log(q1)
    return Q(out, levels), q1


def golden_log_bwd(g, x_q, levels):
    """SIMDLog backward: Q(Q(g) / x_q)."""
    g_q = Q(g, levels)
    return Q(g_q / x_q, levels)


# ---------------------------------------------------------------------------
# Split: y1, y2 = x, x (identity), backward: g1 + g2
# ---------------------------------------------------------------------------

def golden_split_bwd(g1, g2, levels):
    """SIMDSplit backward: Q(Q(g1) + Q(g2))."""
    q1 = Q(g1, levels)
    q2 = Q(g2, levels)
    return Q(q1 + q2, levels)


# ---------------------------------------------------------------------------
# ReduceSum: y = Σx over dim, backward: expand grad to input shape
# ---------------------------------------------------------------------------

def golden_reduce_sum_fwd(in1, dim, keepdim, levels):
    """SIMDReduceSum forward: Q(sum(Q(in1), dim))."""
    q1 = Q(in1, levels)
    return Q(torch.sum(q1, dim=dim, keepdim=keepdim), levels)


# ============================================================================
# Tests — Binary ops
# ============================================================================

class TestNF4SIMDAdd:
    def test_forward(self):
        x1 = _make_input(seed=1)
        x2 = _make_input(seed=2)
        src = SIMDAdd.apply(x1, x2, NF4_SCHEME, True)
        gold = golden_add_fwd(x1, x2, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x1 = _make_input(seed=3).requires_grad_(True)
        x2 = _make_input(seed=4).requires_grad_(True)
        src = SIMDAdd.apply(x1, x2, NF4_SCHEME, True)
        src.sum().backward()

        gold_g = golden_add_bwd(torch.ones_like(src), list(x1.shape), NF4_LEVELS)
        assert torch.equal(x1.grad, gold_g)
        assert torch.equal(x2.grad, gold_g)


class TestNF4SIMDSub:
    def test_forward(self):
        x1 = _make_input(seed=5)
        x2 = _make_input(seed=6)
        src = SIMDSub.apply(x1, x2, NF4_SCHEME, True)
        gold = golden_sub_fwd(x1, x2, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x1 = _make_input(seed=7).requires_grad_(True)
        x2 = _make_input(seed=8).requires_grad_(True)
        src = SIMDSub.apply(x1, x2, NF4_SCHEME, True)
        src.sum().backward()

        # d/dx1 = 1: Q(g) → broadcast = same as Add for g1
        gold_g1 = golden_add_bwd(torch.ones_like(src), list(x1.shape), NF4_LEVELS)
        # d/dx2 = -1: Q(-g) → broadcast
        n_g = Q(-torch.ones_like(src), NF4_LEVELS)
        gold_g2 = golden_add_bwd(n_g, list(x2.shape), NF4_LEVELS)

        assert torch.equal(x1.grad, gold_g1), f"Sub g1 max diff: {(x1.grad - gold_g1).abs().max()}"
        assert torch.equal(x2.grad, gold_g2), f"Sub g2 max diff: {(x2.grad - gold_g2).abs().max()}"


class TestNF4SIMDMul:
    def test_forward(self):
        x1 = _make_input(seed=9)
        x2 = _make_input(seed=10)
        src = SIMDMul.apply(x1, x2, NF4_SCHEME, True)
        gold, _, _ = golden_mul_fwd(x1, x2, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x1 = _make_input(seed=11).requires_grad_(True)
        x2 = _make_input(seed=12).requires_grad_(True)
        src = SIMDMul.apply(x1, x2, NF4_SCHEME, True)
        src.sum().backward()

        _, q1, q2 = golden_mul_fwd(
            x1.detach(), x2.detach(), NF4_LEVELS,
        )
        gold_g1, gold_g2 = golden_mul_bwd(
            torch.ones_like(src), q1, q2,
            list(x1.shape), list(x2.shape), NF4_LEVELS,
        )
        assert torch.equal(x1.grad, gold_g1), f"Mul g1 max diff: {(x1.grad - gold_g1).abs().max()}"
        assert torch.equal(x2.grad, gold_g2), f"Mul g2 max diff: {(x2.grad - gold_g2).abs().max()}"


class TestNF4SIMDDiv:
    def test_forward(self):
        x1 = _make_input(seed=13)
        x2 = _make_input(seed=14).abs() + 0.1  # positive to avoid div-by-zero
        src = SIMDDiv.apply(x1, x2, NF4_SCHEME, True)
        gold, _ = golden_div_fwd(x1, x2, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x1 = _make_input(seed=15).requires_grad_(True)
        x2 = (_make_input(seed=16).abs() + 0.1).requires_grad_(True)
        src = SIMDDiv.apply(x1, x2, NF4_SCHEME, True)
        src.sum().backward()

        gold_out, q2 = golden_div_fwd(
            x1.detach(), x2.detach(), NF4_LEVELS,
        )
        gold_g1, gold_g2 = golden_div_bwd(
            torch.ones_like(src), gold_out, q2,
            list(x1.shape), list(x2.shape), NF4_LEVELS,
        )
        assert torch.equal(x1.grad, gold_g1), f"Div g1 max diff: {(x1.grad - gold_g1).abs().max()}"
        assert torch.equal(x2.grad, gold_g2), f"Div g2 max diff: {(x2.grad - gold_g2).abs().max()}"


# ============================================================================
# Tests — Unary ops
# ============================================================================

class TestNF4SIMDSquare:
    def test_forward(self):
        x = _make_input(seed=17)
        src = SIMDSquare.apply(x, NF4_SCHEME, True)
        gold, _ = golden_square_fwd(x, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x = _make_input(seed=18).requires_grad_(True)
        src = SIMDSquare.apply(x, NF4_SCHEME, True)
        src.sum().backward()

        _, qx = golden_square_fwd(x.detach(), NF4_LEVELS)
        gold_g = golden_square_bwd(torch.ones_like(src), qx, NF4_LEVELS)
        assert torch.equal(x.grad, gold_g)


class TestNF4SIMDSqrt:
    def test_forward(self):
        x = _make_input(seed=19).abs() + 0.1
        src = SIMDSqrt.apply(x, NF4_SCHEME, True)
        gold = golden_sqrt_fwd(x, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x = (_make_input(seed=20).abs() + 0.1).requires_grad_(True)
        src = SIMDSqrt.apply(x, NF4_SCHEME, True)
        src.sum().backward()

        gold_out = golden_sqrt_fwd(x.detach(), NF4_LEVELS)
        gold_g = golden_sqrt_bwd(torch.ones_like(src), gold_out, NF4_LEVELS)
        assert torch.equal(x.grad, gold_g)


class TestNF4SIMDExp:
    def test_forward(self):
        x = _make_input(seed=21)
        src = SIMDExp.apply(x, NF4_SCHEME, True)
        gold = golden_exp_fwd(x, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x = _make_input(seed=22).requires_grad_(True)
        src = SIMDExp.apply(x, NF4_SCHEME, True)
        src.sum().backward()

        gold_out = golden_exp_fwd(x.detach(), NF4_LEVELS)
        gold_g = golden_exp_bwd(torch.ones_like(src), gold_out, NF4_LEVELS)
        assert torch.equal(x.grad, gold_g)


class TestNF4SIMDLog:
    def test_forward(self):
        x = _make_input(seed=23).abs() + 0.1
        src = SIMDLog.apply(x, NF4_SCHEME, True)
        gold, _ = golden_log_fwd(x, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_backward(self):
        x = (_make_input(seed=24).abs() + 0.1).requires_grad_(True)
        src = SIMDLog.apply(x, NF4_SCHEME, True)
        src.sum().backward()

        _, qx = golden_log_fwd(x.detach(), NF4_LEVELS)
        gold_g = golden_log_bwd(torch.ones_like(src), qx, NF4_LEVELS)
        assert torch.equal(x.grad, gold_g)


# ============================================================================
# Tests — Split
# ============================================================================

class TestNF4SIMDSplit:
    def test_forward(self):
        x = _make_input(seed=25)
        s1, s2 = SIMDSplit.apply(x, NF4_SCHEME, True)
        # Split forward: just clones (no quantization)
        assert torch.equal(s1, x)
        assert torch.equal(s2, x)

    def test_backward(self):
        x = _make_input(seed=26).requires_grad_(True)
        s1, s2 = SIMDSplit.apply(x, NF4_SCHEME, True)
        (s1.sum() + s2.sum()).backward()

        gold_g = golden_split_bwd(
            torch.ones_like(s1), torch.ones_like(s2), NF4_LEVELS,
        )
        assert torch.equal(x.grad, gold_g)


# ============================================================================
# Tests — ReduceSum
# ============================================================================

class TestNF4SIMDReduceSum:
    def test_forward(self):
        x = _make_input(seed=27)
        dim = [0]
        src = SIMDReduceSum.apply(x, dim, False, NF4_SCHEME, True)
        gold = golden_reduce_sum_fwd(x, dim, False, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_forward_keepdim(self):
        x = _make_input(seed=28)
        dim = [-1]
        src = SIMDReduceSum.apply(x, dim, True, NF4_SCHEME, True)
        gold = golden_reduce_sum_fwd(x, dim, True, NF4_LEVELS)
        assert torch.equal(src, gold)


# ============================================================================
# Tests — Broadcasting
# ============================================================================

class TestNF4SIMDBroadcast:
    def test_add_broadcast(self):
        """SIMDAdd with broadcasting: (4,8) + (1,8)."""
        x1 = _make_input((4, 8), seed=29)
        x2 = _make_input((1, 8), seed=30)

        src = SIMDAdd.apply(x1, x2, NF4_SCHEME, True)
        gold = golden_add_fwd(x1, x2, NF4_LEVELS)
        assert torch.equal(src, gold)

    def test_mul_broadcast_backward(self):
        """SIMDMul backward with broadcasting."""
        x1 = _make_input((4, 8), seed=31).requires_grad_(True)
        x2 = _make_input((1, 8), seed=32).requires_grad_(True)

        src = SIMDMul.apply(x1, x2, NF4_SCHEME, True)
        src.sum().backward()

        _, q1, q2 = golden_mul_fwd(
            x1.detach(), x2.detach(), NF4_LEVELS,
        )
        gold_g1, gold_g2 = golden_mul_bwd(
            torch.ones_like(src), q1, q2,
            [4, 8], [1, 8], NF4_LEVELS,
        )
        assert torch.equal(x1.grad, gold_g1), f"broadcast g1 max diff: {(x1.grad - gold_g1).abs().max()}"
        assert torch.equal(x2.grad, gold_g2), f"broadcast g2 max diff: {(x2.grad - gold_g2).abs().max()}"


# ============================================================================
# Tests — STE mode
# ============================================================================

class TestNF4SIMDSTE:
    def test_add_ste(self):
        x1 = _make_input(seed=33).requires_grad_(True)
        x2 = _make_input(seed=34).requires_grad_(True)
        src = SIMDAdd.apply(x1, x2, NF4_SCHEME, False)
        src.sum().backward()

    def test_mul_ste(self):
        x1 = _make_input(seed=35).requires_grad_(True)
        x2 = _make_input(seed=36).requires_grad_(True)
        src = SIMDMul.apply(x1, x2, NF4_SCHEME, False)
        src.sum().backward()
        assert torch.isfinite(x1.grad).all()
        assert torch.isfinite(x2.grad).all()
