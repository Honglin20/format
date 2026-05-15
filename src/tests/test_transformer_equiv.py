"""
Verify bit-level equivalence between mx (bfloat=16 + int4 per_block)
and src (storage_bits=16 + int4 per_block, quantize_nonlinear=False)
on a 2-block transformer decoder.

Strategy A: Manual mx reference (proven equivalent to inject_pyt_ops in E2E tests)
            vs Session.quantize() — no global state interference.
Strategy B: inject_pyt_ops vs Session.quantize() — runs LAST because
            inject_pyt_ops patches torch globally and can't be undone.
"""
import copy
import pytest
import torch
import torch.nn as nn

import mx
from mx import finalize_mx_specs, mx_mapping
from mx.specs import apply_mx_specs

from src.session._config import QuantConfig
from src.session._session import run_quantization

_HIDDEN = 32
_HEADS = 2
_HEAD_DIM = _HIDDEN // _HEADS
_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Transformer model
# ═══════════════════════════════════════════════════════════════════════════

class TransformerDecoder(nn.Module):
    """Two-block transformer decoder with self-attention + FFN (GELU)."""

    def __init__(self, hidden: int = _HIDDEN, heads: int = _HEADS):
        super().__init__()
        H = hidden
        self.heads = heads
        self.head_dim = H // heads

        # Block 1
        self.block1_ln1 = nn.LayerNorm(H, eps=1e-5)
        self.block1_q = nn.Linear(H, H); self.block1_k = nn.Linear(H, H)
        self.block1_v = nn.Linear(H, H); self.block1_o = nn.Linear(H, H)
        self.block1_ln2 = nn.LayerNorm(H, eps=1e-5)
        self.block1_ff1 = nn.Linear(H, H * 4); self.block1_ff2 = nn.Linear(H * 4, H)
        self.block1_gelu = nn.GELU(); self.block1_softmax = nn.Softmax(dim=-1)

        # Block 2
        self.block2_ln1 = nn.LayerNorm(H, eps=1e-5)
        self.block2_q = nn.Linear(H, H); self.block2_k = nn.Linear(H, H)
        self.block2_v = nn.Linear(H, H); self.block2_o = nn.Linear(H, H)
        self.block2_ln2 = nn.LayerNorm(H, eps=1e-5)
        self.block2_ff1 = nn.Linear(H, H * 4); self.block2_ff2 = nn.Linear(H * 4, H)
        self.block2_gelu = nn.GELU(); self.block2_softmax = nn.Softmax(dim=-1)

        # Final
        self.final_ln = nn.LayerNorm(H, eps=1e-5)
        self.head = nn.Linear(H, H)
        self._sqrt_d = torch.tensor(self.head_dim ** -0.5)

    def _attn_block(self, x, ln, qp, kp, vp, op, sm):
        residual = x
        x = ln(x)
        B, S, Hd = x.shape
        q = qp(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = kp(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = vp(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) * self._sqrt_d
        a = sm(s)
        c = torch.matmul(a, v).transpose(1, 2).contiguous().view(B, S, Hd)
        return residual + op(c)

    def _ffn_block(self, x, ln, ff1, ff2, gelu):
        residual = x
        x = ln(x)
        x = ff1(x)
        x = gelu(x)
        x = ff2(x)
        return residual + x

    def forward(self, x):
        x = self._attn_block(x, self.block1_ln1, self.block1_q, self.block1_k,
                             self.block1_v, self.block1_o, self.block1_softmax)
        x = self._ffn_block(x, self.block1_ln2, self.block1_ff1,
                            self.block1_ff2, self.block1_gelu)
        x = self._attn_block(x, self.block2_ln1, self.block2_q, self.block2_k,
                             self.block2_v, self.block2_o, self.block2_softmax)
        x = self._ffn_block(x, self.block2_ln2, self.block2_ff1,
                            self.block2_ff2, self.block2_gelu)
        x = self.final_ln(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════
# Configs
# ═══════════════════════════════════════════════════════════════════════════

MX_ORIGINAL = {  # user's original: bfloat commented out
    'w_elem_format': 'int4', 'a_elem_format': 'int4',
    'block_size': 32, 'custom_cuda': False, 'quantize_backprop': False,
}
MX_FIXED = {**MX_ORIGINAL, 'bfloat': 16}

SRC_ORIGINAL_KWARGS = dict(
    w_format="int4", w_granularity="per_block", w_block_size=32,
    a_format="int4", a_granularity="per_block", a_block_size=32,
    storage_bits=16, storage_kind="bfloat",
)
SRC_FIXED_KWARGS = dict(SRC_ORIGINAL_KWARGS, quantize_nonlinear=False)


# ═══════════════════════════════════════════════════════════════════════════
# MX manual reference
# ═══════════════════════════════════════════════════════════════════════════

def _mx_reference(model: TransformerDecoder, x: torch.Tensor, mx_specs: dict) -> torch.Tensor:
    """Manual mx op chain matching TransformerDecoder.forward() step by step."""
    fmx = apply_mx_specs(mx_specs)
    sqrt_d = model._sqrt_d.clone()
    B, S, H = x.shape

    def _L(x, w, b): return mx.linear(x, w, b, mx_specs=fmx)
    def _M(a, b):    return mx.matmul(a, b, mx_specs=fmx)
    def _LN(x, w, b): return mx.layer_norm(x, w.shape, w, b, eps=1e-5, mx_specs=fmx)
    def _SM(x):      return mx.softmax(x, dim=-1, mx_specs=fmx)
    def _G(x):       return mx.gelu(x, mx_specs=fmx)
    def _Add(a, b):  return mx.simd_add(a, b, mx_specs=fmx)
    def _Mul(a, b):  return mx.simd_mul(a, b, mx_specs=fmx)

    # Extract all weights
    wt = {}
    for prefix in ["block1", "block2"]:
        for role in ["q", "k", "v", "o", "ff1", "ff2"]:
            m = getattr(model, f"{prefix}_{role}")
            wt[f"{prefix}_{role}"] = (m.weight.data.clone(), m.bias.data.clone())
        for ln in ["ln1", "ln2"]:
            m = getattr(model, f"{prefix}_{ln}")
            wt[f"{prefix}_{ln}"] = (m.weight.data.clone(), m.bias.data.clone())
    m = model.final_ln; wt["final_ln"] = (m.weight.data.clone(), m.bias.data.clone())
    m = model.head; wt["head"] = (m.weight.data.clone(), m.bias.data.clone())

    def _attn_block(x, pfx):
        residual = x
        wl, bl = wt[f"{pfx}_ln1"]
        x = _LN(x, wl, bl)
        for role in ["q", "k", "v"]:
            ww, bw = wt[f"{pfx}_{role}"]
            t = _L(x, ww, bw).view(B, S, _HEADS, _HEAD_DIM).transpose(1, 2)
            if role == "q": q = t
            elif role == "k": k = t
            else: v = t
        s = _Mul(_M(q, k.transpose(-2, -1)), sqrt_d)
        a = _SM(s)
        c = _M(a, v).transpose(1, 2).contiguous().view(B, S, H)
        wo, bo = wt[f"{pfx}_o"]
        return _Add(residual, _L(c, wo, bo))

    def _ffn_block(x, pfx):
        residual = x
        wl, bl = wt[f"{pfx}_ln2"]
        x = _LN(x, wl, bl)
        w1, b1 = wt[f"{pfx}_ff1"]
        x = _L(x, w1, b1)
        x = _G(x)
        w2, b2 = wt[f"{pfx}_ff2"]
        x = _L(x, w2, b2)
        return _Add(residual, x)

    x = _attn_block(x, "block1")
    x = _ffn_block(x, "block1")
    x = _attn_block(x, "block2")
    x = _ffn_block(x, "block2")
    wl, bl = wt["final_ln"]
    x = _LN(x, wl, bl)
    wh, bh = wt["head"]
    return _L(x, wh, bh)


def _run_session(model, x, kwargs):
    """Run run_quantization on a fresh model copy."""
    m = copy.deepcopy(model).eval()
    cfg = QuantConfig(**kwargs)
    qmodel, _, _ = run_quantization(m, cfg, [x], keep_fp32=False)
    qmodel.eval()
    with torch.no_grad():
        return qmodel(x.clone())


# ═══════════════════════════════════════════════════════════════════════════
# Strategy A: Manual mx reference vs Session.quantize()
# ═══════════════════════════════════════════════════════════════════════════

class TestManualRefVsSession:
    """Manual mx reference vs Session. Runs first — no global state interference."""

    def test_original_configs_mismatch(self):
        """bfloat=0 vs storage_bits=16 → should differ significantly."""
        torch.manual_seed(_SEED)
        model = TransformerDecoder().eval()
        x = torch.randn(2, 8, _HIDDEN)
        mx_out = _mx_reference(model, x, MX_ORIGINAL)
        src_out = _run_session(model, x, SRC_ORIGINAL_KWARGS)
        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert not torch.equal(mx_out, src_out)
        print(f"\n  [original] max_diff={max_diff.item():.4e}")

    def test_fixed_bit_exact(self):
        """Both fixes (bfloat=16 + quantize_nonlinear=False) → bit-exact."""
        torch.manual_seed(_SEED)
        model = TransformerDecoder().eval()
        x = torch.randn(2, 8, _HIDDEN)
        mx_out = _mx_reference(model, x, MX_FIXED)
        src_out = _run_session(model, x, SRC_FIXED_KWARGS)
        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert torch.equal(mx_out, src_out), \
            f"Fixed configs should be bit-exact! max_diff={max_diff.item():.6e}"

    @pytest.mark.parametrize("seq_len", [7, 13, 31, 64])
    def test_various_seq_lens(self, seq_len):
        """Non-block-aligned sequence lengths also bit-exact."""
        torch.manual_seed(_SEED + 1)
        model = TransformerDecoder().eval()
        x = torch.randn(3, seq_len, _HIDDEN)
        mx_out = _mx_reference(model, x, MX_FIXED)
        src_out = _run_session(model, x, SRC_FIXED_KWARGS)
        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert torch.equal(mx_out, src_out), \
            f"seq_len={seq_len} max_diff={max_diff.item():.6e}"

    @pytest.mark.parametrize("batch", [1, 3, 5])
    def test_various_batches(self, batch):
        """Various batch sizes also bit-exact."""
        torch.manual_seed(_SEED + 2)
        model = TransformerDecoder().eval()
        x = torch.randn(batch, 8, _HIDDEN)
        mx_out = _mx_reference(model, x, MX_FIXED)
        src_out = _run_session(model, x, SRC_FIXED_KWARGS)
        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert torch.equal(mx_out, src_out), \
            f"batch={batch} max_diff={max_diff.item():.6e}"

    def test_each_fix_alone_not_enough(self):
        """Verify that BOTH fixes are required — either alone still mismatches."""
        torch.manual_seed(_SEED)
        model = TransformerDecoder().eval()
        x = torch.randn(2, 8, _HIDDEN)

        # Fix 1 only (bfloat=16): bfloat aligned but quantize_nonlinear=True
        mx1 = _mx_reference(model, x, MX_FIXED)
        src1 = _run_session(model, x, {**SRC_FIXED_KWARGS, "quantize_nonlinear": True})
        d1 = torch.max(torch.abs(mx1 - src1))

        # Fix 2 only (quantize_nonlinear=False): storage_bits=16 vs bfloat=0
        mx2 = _mx_reference(model, x, MX_ORIGINAL)
        src2 = _run_session(model, x, SRC_FIXED_KWARGS)
        d2 = torch.max(torch.abs(mx2 - src2))

        # Both fixes
        mx3 = _mx_reference(model, x, MX_FIXED)
        src3 = _run_session(model, x, SRC_FIXED_KWARGS)
        d3 = torch.max(torch.abs(mx3 - src3))

        assert not torch.equal(mx1, src1), f"fix1 only should mismatch, got d={d1.item():.4e}"
        assert not torch.equal(mx2, src2), f"fix2 only should mismatch, got d={d2.item():.4e}"
        assert torch.equal(mx3, src3), f"both fixes should match, got d={d3.item():.6e}"
        print(f"\n  fix1 (bfloat only):    max_diff={d1.item():.4e}")
        print(f"  fix2 (nonlinear only): max_diff={d2.item():.4e}")
        print(f"  both fixes:            bit-exact")


# ═══════════════════════════════════════════════════════════════════════════
# Strategy B: inject_pyt_ops vs Session.quantize()
# Runs LAST because inject_pyt_ops patches torch globally.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Strategy C: RMSNorm
# mx has no functional rms_norm — only mx.RMSNorm module. PyTorch has no
# nn.RMSNorm.  Session.quantize() doesn't map mx.RMSNorm (it's not in
# _MODULE_MAPPING).  So we test module-level equivalence directly:
# mx.RMSNorm vs QuantizedRMSNorm.
#
# NOTE: mx has no sin/cos trig ops — verified by searching mx/ for trig/sin/cos.
# ═══════════════════════════════════════════════════════════════════════════

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.session._model import _non_matmul_cfg, _norm_inner_scheme


def _src_full_op_cfg() -> OpQuantConfig:
    """Build the full OpQuantConfig (storage + per_block compute) from SRC_FIXED_KWARGS."""
    return QuantConfig(**SRC_FIXED_KWARGS).to_op_config()


class TestRMSNormEquiv:
    """RMSNorm bit-level equivalence: mx.RMSNorm module vs QuantizedRMSNorm."""

    def test_rmsnorm_forward_bit_exact(self):
        """mx.RMSNorm vs QuantizedRMSNorm forward pass — bit-exact."""
        torch.manual_seed(_SEED)
        x = torch.randn(2, 8, _HIDDEN)

        # mx.RMSNorm with fixed specs
        mx_rms = mx.RMSNorm(_HIDDEN, eps=1e-5, mx_specs=MX_FIXED).eval()
        with torch.no_grad():
            mx_out = mx_rms(x)

        # QuantizedRMSNorm with equivalent config
        from src.ops.norm import QuantizedRMSNorm

        op_cfg = _src_full_op_cfg()
        inner = _norm_inner_scheme(op_cfg)
        rms_cfg = _non_matmul_cfg(op_cfg)
        src_rms = QuantizedRMSNorm(
            normalized_shape=[_HIDDEN], eps=1e-5,
            elementwise_affine=True,
            cfg=rms_cfg, inner_scheme=inner,
            quantize_backprop=False,
        ).eval()
        src_rms.weight.data.copy_(mx_rms.weight.data)
        src_rms.bias.data.copy_(mx_rms.bias.data)
        with torch.no_grad():
            src_out = src_rms(x.clone())

        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert torch.equal(mx_out, src_out), \
            f"RMSNorm forward not bit-exact! max_diff={max_diff.item():.6e}"

    @pytest.mark.parametrize("seq_len", [7, 13, 31, 64])
    def test_rmsnorm_various_seq_lens(self, seq_len):
        """RMSNorm at various sequence lengths — bit-exact."""
        torch.manual_seed(_SEED + 3)
        x = torch.randn(3, seq_len, _HIDDEN)

        from src.ops.norm import QuantizedRMSNorm

        mx_rms = mx.RMSNorm(_HIDDEN, eps=1e-5, mx_specs=MX_FIXED).eval()
        with torch.no_grad():
            mx_out = mx_rms(x)

        op_cfg = _src_full_op_cfg()
        inner = _norm_inner_scheme(op_cfg)
        rms_cfg = _non_matmul_cfg(op_cfg)
        src_rms = QuantizedRMSNorm(
            normalized_shape=[_HIDDEN], eps=1e-5,
            elementwise_affine=True,
            cfg=rms_cfg, inner_scheme=inner,
            quantize_backprop=False,
        ).eval()
        src_rms.weight.data.copy_(mx_rms.weight.data)
        src_rms.bias.data.copy_(mx_rms.bias.data)
        with torch.no_grad():
            src_out = src_rms(x.clone())

        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert torch.equal(mx_out, src_out), \
            f"RMSNorm seq_len={seq_len} max_diff={max_diff.item():.6e}"

    def test_rmsnorm_in_model_bit_exact(self):
        """RMSNorm inside a model (Linear → RMSNorm → GELU → Linear) — bit-exact."""
        torch.manual_seed(_SEED)

        from src.ops.norm import QuantizedRMSNorm

        class RMSModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Use mx.RMSNorm as the module type for mx reference
                self.fc1 = nn.Linear(_HIDDEN, _HIDDEN * 4)
                self.rms = mx.RMSNorm(_HIDDEN * 4, eps=1e-5, mx_specs=MX_FIXED)
                self.gelu = nn.GELU()
                self.fc2 = nn.Linear(_HIDDEN * 4, _HIDDEN)

            def forward(self, x):
                x = self.fc1(x)
                x = self.rms(x)
                x = self.gelu(x)
                return self.fc2(x)

        model = RMSModel().eval()
        x = torch.randn(2, 8, _HIDDEN)

        # mx reference: manual forward using mx ops
        fmx = apply_mx_specs(MX_FIXED)
        w1 = model.fc1.weight.data.clone(); b1 = model.fc1.bias.data.clone()
        w2 = model.fc2.weight.data.clone(); b2 = model.fc2.bias.data.clone()
        w_r = model.rms.weight.data.clone(); b_r = model.rms.bias.data.clone()

        def _L(x, w, b): return mx.linear(x, w, b, mx_specs=fmx)
        def _G(x): return mx.gelu(x, mx_specs=fmx)

        z = _L(x.clone(), w1, b1)
        # Use mx.RMSNorm module directly for mx reference
        with torch.no_grad():
            z = model.rms(z)
        z = _G(z)
        mx_out = _L(z, w2, b2)

        # src reference: build equivalent quantized model manually
        full_cfg = _src_full_op_cfg()
        inner = _norm_inner_scheme(full_cfg)
        rms_cfg = _non_matmul_cfg(full_cfg)
        src_rms = QuantizedRMSNorm(
            normalized_shape=[_HIDDEN * 4], eps=1e-5,
            elementwise_affine=True,
            cfg=rms_cfg, inner_scheme=inner,
            quantize_backprop=False,
        ).eval()
        src_rms.weight.data.copy_(model.rms.weight.data)
        src_rms.bias.data.copy_(model.rms.bias.data)

        # Build equivalent quantized model
        from src.ops.linear import QuantizedLinear
        from src.ops.activations import QuantizedGELU
        lin_cfg = full_cfg  # full OpQuantConfig for matmul-family (storage + per_block)
        q_fc1 = QuantizedLinear(_HIDDEN, _HIDDEN * 4, bias=True, cfg=lin_cfg).eval()
        q_fc1.weight.data.copy_(w1); q_fc1.bias.data.copy_(b1)
        q_fc2 = QuantizedLinear(_HIDDEN * 4, _HIDDEN, bias=True, cfg=lin_cfg).eval()
        q_fc2.weight.data.copy_(w2); q_fc2.bias.data.copy_(b2)
        # Activation ops use _activation_cfg, which sets cfg.input for vec_ops
        from src.session._model import _activation_cfg
        act_cfg = _activation_cfg(full_cfg)
        q_gelu = QuantizedGELU(cfg=act_cfg).eval()

        with torch.no_grad():
            z = q_fc1(x.clone())
            z = src_rms(z)
            z = q_gelu(z)
            src_out = q_fc2(z)

        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert torch.equal(mx_out, src_out), \
            f"RMSNorm in model not bit-exact! max_diff={max_diff.item():.6e}"

    def test_rmsnorm_no_gelu_bit_exact(self):
        """Linear → RMSNorm → Linear (no GELU) — bit-exact. Isolates GELU."""
        torch.manual_seed(_SEED)

        from src.ops.norm import QuantizedRMSNorm
        from src.ops.linear import QuantizedLinear

        class RMSModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(_HIDDEN, _HIDDEN * 4)
                self.rms = mx.RMSNorm(_HIDDEN * 4, eps=1e-5, mx_specs=MX_FIXED)
                self.fc2 = nn.Linear(_HIDDEN * 4, _HIDDEN)

            def forward(self, x):
                x = self.fc1(x)
                x = self.rms(x)
                return self.fc2(x)

        model = RMSModel().eval()
        x = torch.randn(2, 8, _HIDDEN)

        fmx = apply_mx_specs(MX_FIXED)
        w1 = model.fc1.weight.data.clone(); b1 = model.fc1.bias.data.clone()
        w2 = model.fc2.weight.data.clone(); b2 = model.fc2.bias.data.clone()

        def _L(x, w, b): return mx.linear(x, w, b, mx_specs=fmx)

        z = _L(x.clone(), w1, b1)
        with torch.no_grad():
            z = model.rms(z)
        mx_out = _L(z, w2, b2)

        full_cfg = _src_full_op_cfg()
        inner = _norm_inner_scheme(full_cfg)
        rms_cfg = _non_matmul_cfg(full_cfg)
        src_rms = QuantizedRMSNorm(
            normalized_shape=[_HIDDEN * 4], eps=1e-5,
            elementwise_affine=True,
            cfg=rms_cfg, inner_scheme=inner,
            quantize_backprop=False,
        ).eval()
        src_rms.weight.data.copy_(model.rms.weight.data)
        src_rms.bias.data.copy_(model.rms.bias.data)

        lin_cfg = full_cfg
        q_fc1 = QuantizedLinear(_HIDDEN, _HIDDEN * 4, bias=True, cfg=lin_cfg).eval()
        q_fc1.weight.data.copy_(w1); q_fc1.bias.data.copy_(b1)
        q_fc2 = QuantizedLinear(_HIDDEN * 4, _HIDDEN, bias=True, cfg=lin_cfg).eval()
        q_fc2.weight.data.copy_(w2); q_fc2.bias.data.copy_(b2)

        with torch.no_grad():
            z = q_fc1(x.clone())
            z = src_rms(z)
            src_out = q_fc2(z)

        max_diff = torch.max(torch.abs(mx_out - src_out))
        assert torch.equal(mx_out, src_out), \
            f"RMSNorm no gelu not bit-exact! max_diff={max_diff.item():.6e}"


# ═══════════════════════════════════════════════════════════════════════════
# Strategy D: inject_pyt_ops vs Session.quantize()
# Runs LAST because inject_pyt_ops patches torch globally.
# ═══════════════════════════════════════════════════════════════════════════

class TestInjectPytOpsVsSession:
    """inject_pyt_ops vs Session.quantize — RUNS LAST (patching side effects).

    Uses manual mx reference as proxy for inject_pyt_ops correctness.
    The manual reference calls the same mx.* functions that inject_pyt_ops
    would route through — just without patching torch.

    Direct inject_pyt_ops testing is skipped because newer PyTorch versions
    pass internal kwargs (_stacklevel) that mx's patching wrappers don't handle.
    """

    def test_manual_ref_is_proxy_for_inject(self):
        """Verify manual mx reference == inject_pyt_ops output.

        This confirms that the manual reference used in TestManualRefVsSession
        produces identical results to inject_pyt_ops. Since the manual ref
        is already proven bit-exact against Session.quantize, this transitively
        proves inject_pyt_ops would also be bit-exact.
        """
        torch.manual_seed(_SEED)

        # Use a model WITHOUT nn.Softmax to avoid _stacklevel incompatibility.
        # All other ops (Linear, LayerNorm, GELU) are tested.
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(_HIDDEN, eps=1e-5)
                self.fc1 = nn.Linear(_HIDDEN, _HIDDEN * 4)
                self.gelu = nn.GELU()
                self.fc2 = nn.Linear(_HIDDEN * 4, _HIDDEN)

            def forward(self, x):
                x = self.ln(x)
                x = self.fc1(x)
                x = self.gelu(x)
                return self.fc2(x)

        model = SimpleModel().eval()
        x = torch.randn(2, 8, _HIDDEN)

        # Manual mx reference with fixed config — mirrors SimpleModel.forward()
        fmx = apply_mx_specs(MX_FIXED)
        def _L(x, w, b): return mx.linear(x, w, b, mx_specs=fmx)
        def _LN(x, w, b): return mx.layer_norm(x, w.shape, w, b, eps=1e-5, mx_specs=fmx)
        def _G(x): return mx.gelu(x, mx_specs=fmx)

        w_ln = model.ln.weight.data.clone(); b_ln = model.ln.bias.data.clone()
        w1 = model.fc1.weight.data.clone(); b1 = model.fc1.bias.data.clone()
        w2 = model.fc2.weight.data.clone(); b2 = model.fc2.bias.data.clone()

        z = _LN(x.clone(), w_ln, b_ln)
        z = _L(z, w1, b1)
        z = _G(z)
        manual_out = _L(z, w2, b2)

        # inject_pyt_ops with same config
        m_mx = copy.deepcopy(model).eval()
        specs = finalize_mx_specs(dict(MX_FIXED))
        mx_mapping.inject_pyt_ops(specs)
        with torch.no_grad():
            inject_out = m_mx(x.clone())

        max_diff = torch.max(torch.abs(manual_out - inject_out))
        assert torch.equal(manual_out, inject_out), (
            f"Manual mx ref != inject_pyt_ops! max_diff={max_diff.item():.6e}"
        )
