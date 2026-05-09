#!/usr/bin/env python3
"""
Post-fix verification: MX int4 vs src — end-to-end bit-exact equivalence.

Verifies ALL module types in _MODULE_MAPPING plus ALL inline ops from
_PATCH_TABLE.  The test model chains every operator in a single forward
pass so that quantize_model + model(x) exercises every conversion path.

This is a DEVELOPMENT / DEBUGGING tool for single-format verification.
For parametrized CI testing see:
  - src/tests/test_e2e_all_ops.py (smoke + full MX format parametrization)
  - src/tests/test_ops_equiv_conv.py (Conv1d/2d/3d, MX format parametrization)
  - src/tests/test_ops_equiv_conv_transpose.py (ConvTranspose MX format)
  - src/tests/test_ops_equiv_matmul.py (Linear/MatMul/BMM MX format)

MX baseline config:
  w_elem_format='int4', a_elem_format='int4', block_size=32,
  bfloat=16, custom_cuda=False, quantize_backprop=False
"""
import json
import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

MX_SPECS = {
    'w_elem_format': 'int4',
    'a_elem_format': 'int4',
    'block_size': 32,
    'bfloat': 16,
    'custom_cuda': False,
    'quantize_backprop': False,
}
SEED = 42
_HIDDEN = 32  # = 4*8 = 4*2*4 = 2*2*2*2*2  (every reshape preserves elements)

# ═══════════════════════════════════════════════════════════════════════════
# MX reference helpers
# ═══════════════════════════════════════════════════════════════════════════

import mx
from mx import finalize_mx_specs, mx_mapping
from mx.specs import apply_mx_specs
from mx.batchnorm import batch_norm as _mx_raw_bn
from mx.groupnorm import group_norm as _mx_raw_gn

_fmx = apply_mx_specs(MX_SPECS)


def _mx_linear(x, w, b):
    return mx.linear(x, w, b, mx_specs=_fmx)


def _mx_matmul(a, b):
    return mx.matmul(a, b, mx_specs=_fmx)


def _mx_bmm(a, b):
    return mx.bmm(a, b, mx_specs=_fmx)


def _mx_layernorm(x, w, b):
    return mx.layer_norm(x, w.shape, w, b, eps=1e-5, mx_specs=_fmx)


def _mx_softmax(x):
    return mx.softmax(x, dim=-1, mx_specs=_fmx)


def _mx_relu(x):
    return mx.relu(x, mx_specs=_fmx)


def _mx_gelu(x):
    return mx.gelu(x, mx_specs=_fmx)


def _mx_silu(x):
    return mx.silu(x, mx_specs=_fmx)


def _mx_sigmoid(x):
    return mx.sigmoid(x, mx_specs=_fmx)


def _mx_tanh(x):
    return mx.tanh(x, mx_specs=_fmx)


def _mx_relu6(x):
    return mx.relu6(x, mx_specs=_fmx)


def _mx_leaky_relu(x):
    return mx.leaky_relu(x, mx_specs=_fmx)


def _mx_conv1d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    return mx.conv1d(x, w, b, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, mx_specs=_fmx)


def _mx_conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    return mx.conv2d(x, w, b, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, mx_specs=_fmx)


def _mx_conv3d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    return mx.conv3d(x, w, b, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, mx_specs=_fmx)


def _mx_batch_norm(x, rm, rv, w, b):
    return _mx_raw_bn(x, rm, rv, w, b,
                      is_training=False, momentum=0.1, eps=1e-5,
                      mx_specs=_fmx)


def _mx_group_norm(x, num_groups, w, b):
    return _mx_raw_gn(x, num_groups, w, b, eps=1e-5, mx_specs=_fmx)


def _mx_conv_transpose2d(x, w, b=None, stride=1, padding=0, output_padding=0,
                          dilation=1, groups=1):
    m = mx.ConvTranspose2d(
        w.shape[0], w.shape[1], kernel_size=w.shape[2:],
        stride=stride, padding=padding, output_padding=output_padding,
        dilation=dilation, groups=groups, bias=b is not None,
        mx_specs=_fmx,
    )
    m.weight.data.copy_(w)
    if b is not None:
        m.bias.data.copy_(b)
    return m(x)


def _mx_pool(x, output_size):
    return mx.adaptive_avg_pool2d(x, output_size, mx_specs=_fmx)


def _mx_add(a, b):
    return mx.simd_add(a, b, mx_specs=_fmx)


def _mx_sub(a, b):
    return mx.simd_sub(a, b, mx_specs=_fmx)


def _mx_mul(a, b):
    return mx.simd_mul(a, b, mx_specs=_fmx)


def _mx_div(a, b):
    return mx.simd_div(a, b, mx_specs=_fmx)


def _mx_exp(x):
    return mx.simd_exp(x, mx_specs=_fmx)


def _mx_log(x):
    return mx.simd_log(x, mx_specs=_fmx)


# ═══════════════════════════════════════════════════════════════════════════
# Src helpers
# ═══════════════════════════════════════════════════════════════════════════

from src.session._config import QuantConfig
from src.session._model import quantize_model
from src.tests._compat import (
    op_config_from_mx_specs,
    norm_config_from_mx_specs,
    softmax_config_from_mx_specs,
    activation_config_from_mx_specs,
    simd_config_from_mx_specs,
    pool_config_from_mx_specs,
)

# Compat-based OpQuantConfigs
_CFG_LINEAR = op_config_from_mx_specs(MX_SPECS, op_type="linear")
_CFG_CONV = op_config_from_mx_specs(MX_SPECS, op_type="conv")
_CFG_CONV_TRANSPOSE = op_config_from_mx_specs(MX_SPECS, op_type="conv_transpose")
_CFG_MATMUL = op_config_from_mx_specs(MX_SPECS, op_type="matmul")
_NORM_CFG, _NORM_INNER, _NORM_QBP = norm_config_from_mx_specs(MX_SPECS, op_type="layer_norm")
_BN_CFG, _BN_INNER, _BN_QBP = norm_config_from_mx_specs(MX_SPECS, op_type="batch_norm")
_GN_CFG, _GN_INNER, _GN_QBP = norm_config_from_mx_specs(MX_SPECS, op_type="group_norm")
_SM_CFG, _SM_EXP2 = softmax_config_from_mx_specs(MX_SPECS)
_ACT_CFG = activation_config_from_mx_specs(MX_SPECS)
_POOL_CFG = pool_config_from_mx_specs(MX_SPECS)
_SIMD_INNER, _SIMD_QBP = simd_config_from_mx_specs(MX_SPECS)

# QuantConfig-based equivalent (user-facing API)
_QUANT_CFG = QuantConfig(
    name="MXINT4-verify",
    w_format="int4",
    w_granularity="per_block",
    w_block_size=32,
    a_format="int4",
    a_granularity="per_block",
    a_block_size=32,
    storage_bits=16,
    storage_kind="bfloat",
    transform="none",
)
_OP_CFG_USER = _QUANT_CFG.to_op_config()

# Inline-op configs for matmul family intercepted by QuantizeContext.
# When quantize_model receives a per-module dict, inline ops resolve via
# _resolve_context_cfg which only extracts a storage scheme.  Full MX
# quantization for inline matmul-type ops requires explicit op_cfgs.
_INLINE_OP_CFGS = {
    "matmul": _CFG_MATMUL,
    "mm": _CFG_MATMUL,
    "bmm": _CFG_MATMUL,
    "linear": _CFG_LINEAR,
}


# ═══════════════════════════════════════════════════════════════════════════
# AllOpsModel — exercises every module type AND every inline op
# ═══════════════════════════════════════════════════════════════════════════

class AllOpsModel(nn.Module):
    """Model exercising all quantizeable operator types in a single forward pass.

    Module path (replaced by quantize_model):
      Linear, LayerNorm, ReLU, GELU, SiLU, Sigmoid, Tanh, ReLU6, LeakyReLU,
      Softmax, Conv1d, Conv2d, Conv3d, BatchNorm1d, BatchNorm2d, BatchNorm3d,
      GroupNorm, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
      AdaptiveAvgPool2d

    Inline path (intercepted by QuantizeContext via patched torch.* / Tensor dunders):
      torch.add, torch.sub, torch.mul, torch.div, torch.exp, torch.log,
      torch.matmul, torch.mm, torch.bmm, F.linear (explicit)
    """

    def __init__(self, hidden: int = _HIDDEN):
        super().__init__()
        H = hidden  # 32

        # --- 1D section: Linear + LayerNorm + Activations + Softmax ---
        self.ln1 = nn.LayerNorm(H, eps=1e-5)
        self.linear1 = nn.Linear(H, H * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(H * 2, H)

        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu6 = nn.ReLU6()
        self.leaky_relu = nn.LeakyReLU()

        self.ln2 = nn.LayerNorm(H, eps=1e-5)
        self.softmax = nn.Softmax(dim=-1)

        # --- Conv1d path: (B, H=32) ↔ (B, C=4, L=8) ---
        self.conv1d = nn.Conv1d(4, 4, 3, padding=1)
        self.bn1d = nn.BatchNorm1d(4)
        self.conv_transpose1d = nn.ConvTranspose1d(4, 4, 3, padding=1)

        # --- Conv2d path: (B, H=32) ↔ (B, C=4, H=2, W=4) ---
        self.conv2d = nn.Conv2d(4, 4, 3, padding=1)
        self.bn2d = nn.BatchNorm2d(4)
        self.gn = nn.GroupNorm(2, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))
        self.conv_transpose2d = nn.ConvTranspose2d(4, 4, 3, padding=1)

        # --- Conv3d path: (B, H=32) ↔ (B, C=2, D=2, H=2, W=2) ---
        self.conv3d = nn.Conv3d(2, 2, 3, padding=1)
        self.bn3d = nn.BatchNorm3d(2)
        self.conv_transpose3d = nn.ConvTranspose3d(2, 2, 3, padding=1)

        # --- Output ---
        self.linear3 = nn.Linear(H, H)

        # Buffers for inline ops
        self.register_buffer("matmul_w", torch.randn(H, H) * 0.1)
        self.register_buffer("mm_w", torch.randn(H, H) * 0.1)
        self.register_buffer("bmm_w", torch.randn(2, H, H) * 0.1)
        self.register_buffer("div_val", torch.tensor(2.0))

    def forward(self, x):
        # x: (2, H) where H=_HIDDEN=32
        batch = x.shape[0]
        H = x.shape[1]  # 32

        # ---- 1D section ----
        residual = x
        x = self.ln1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x + residual               # INLINE add (Tensor.__add__)

        # ---- Inline arithmetic ops (between modules) ----
        x = torch.mul(x, torch.sigmoid(x))   # INLINE mul
        x = torch.sub(x, x.mean(dim=1, keepdim=True))  # INLINE sub

        # ---- Activations (all 6 remaining) ----
        x = self.gelu(x)
        x = self.silu(x)
        x = self.sigmoid(x)
        x = self.tanh(x)
        x = self.relu6(x)
        x = self.leaky_relu(x)

        # ---- Conv1d path: (B,32) → (B,4,8) → conv1d → bn1d → conv_transpose1d → (B,32) ----
        c1 = x.reshape(batch, 4, 8)
        c1 = self.conv1d(c1)
        c1 = self.bn1d(c1)
        c1 = self.conv_transpose1d(c1)
        x_c1 = c1.reshape(batch, H)

        # ---- Conv2d path: (B,32) → (B,4,2,4) → conv2d → bn2d → gn → pool → conv_transpose2d → (B,32) ----
        c2 = x.reshape(batch, 4, 2, 4)
        c2 = self.conv2d(c2)
        c2 = self.bn2d(c2)
        c2 = self.gn(c2)
        c2 = self.adaptive_pool(c2)    # output_size=(2,4), same shape so identity
        c2 = self.conv_transpose2d(c2)
        x_c2 = c2.reshape(batch, H)

        # ---- Conv3d path: (B,32) → (B,2,2,2,4) → conv3d → bn3d → conv_transpose3d → (B,32) ----
        c3 = x.reshape(batch, 2, 2, 2, 4)
        c3 = self.conv3d(c3)
        c3 = self.bn3d(c3)
        c3 = self.conv_transpose3d(c3)
        x_c3 = c3.reshape(batch, H)

        # ---- Combine all paths ----
        x = x + x_c1 + x_c2 + x_c3

        # ---- More inline ops ----
        x = self.ln2(x)

        # torch.matmul: needs 2D → unsqueeze to (1, B, H) for 3D matmul, then squeeze
        x = torch.matmul(x.unsqueeze(0), self.matmul_w).squeeze(0)

        # torch.div (inline)
        x = torch.div(x, self.div_val)

        # torch.exp + torch.log
        x = torch.exp(x)
        x = torch.abs(x) + 1e-5
        x = torch.log(x)

        # torch.bmm: (2, H) → (2, 1, H), bmm with (2, H, H), then (2, 1, H) → (2, H)
        x_3d = x.unsqueeze(1)          # (2, 1, H)
        x_3d = torch.bmm(x_3d, self.bmm_w)  # (2, 1, H)
        x = x_3d.squeeze(1)           # (2, H)

        # torch.mm: 2D matmul on a slice
        x = torch.mm(x[:1], self.mm_w)  # (1, H), inline mm

        # Explicit F.linear (should also be intercepted)
        x = nn.functional.linear(x, self.matmul_w)

        # ---- Final ----
        x = self.softmax(x)
        x = self.linear3(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("Post-Fix Verification: MX int4 vs src — ALL Ops End-to-End")
    print("=" * 72)
    print(f"MX Specs: {json.dumps(MX_SPECS)}")
    print(f"Hidden dim: {_HIDDEN}")
    print()

    torch.manual_seed(SEED)

    # ── Test 1: OpQuantConfig compatibility ──
    print("─" * 72)
    print("Test 1: QuantConfig.to_op_config() vs compat ground truth")
    print("─" * 72)
    _print_config_diff(_OP_CFG_USER, _CFG_LINEAR)

    # ── Build model and MX reference ──
    model_compat = AllOpsModel()
    per_module = _build_per_module_config()
    quantize_model(model_compat, per_module, op_cfgs=_INLINE_OP_CFGS)
    model_compat.eval()

    x = torch.randn(2, _HIDDEN)

    with torch.no_grad():
        src_compat_out = model_compat(x.clone())

    # ── MX reference chain ──
    mx_ref = _build_mx_reference(model_compat, x.clone())

    # ── Test 2: compat configs ──
    print("\n─" * 72)
    print("Test 2: quantize_model (per-module compat cfgs) vs MX reference chain")
    print("─" * 72)
    _assert_and_print(mx_ref, src_compat_out, "compat-all-ops")

    # ── Test 3: QuantConfig singleton ──
    print("\n─" * 72)
    print("Test 3: quantize_model (QuantConfig singleton) vs MX reference chain")
    print("─" * 72)

    model_user = AllOpsModel()
    quantize_model(model_user, _OP_CFG_USER)
    model_user.eval()
    model_user.load_state_dict(model_compat.state_dict())

    with torch.no_grad():
        src_user_out = model_user(x.clone())

    _assert_and_print(mx_ref, src_user_out, "QuantConfig-all-ops")

    # ── Test 4: Layer-by-layer ──
    print("\n─" * 72)
    print("Test 4: Layer-by-layer comparison (AllOpsModel, compat configs)")
    print("─" * 72)

    all_ok = _layer_by_layer_test(model_compat, x.clone())

    # ── Summary ──
    ok_compat = torch.equal(mx_ref, src_compat_out)
    ok_user = torch.equal(mx_ref, src_user_out)
    ok_config = _cfg_matches()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Test 1 (Config match):        {'✓ PASS' if ok_config else '✗ FAIL'}")
    print(f"  Test 2 (compat cfgs, all ops): {'✓ PASS' if ok_compat else '✗ FAIL'}")
    print(f"  Test 3 (QuantConfig, all ops):{'✓ PASS' if ok_user else '✗ FAIL'}")
    print(f"  Test 4 (layer-by-layer):      {'✓ ALL BIT-EXACT' if all_ok else '✗ DISCREPANCIES FOUND'}")

    all_pass = ok_config and ok_compat and ok_user and all_ok
    if all_pass:
        print("\n  ALL TESTS PASSED — End-to-end bit-exact equivalence confirmed.")
    else:
        print("\n  SOME TESTS FAILED — see details above.")

    return 0 if all_pass else 1


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_per_module_config():
    """Return a dict mapping every module name → OpQuantConfig."""
    return {
        # 1D ops
        "ln1": _NORM_CFG,
        "ln2": _NORM_CFG,
        "linear1": _CFG_LINEAR,
        "linear2": _CFG_LINEAR,
        "linear3": _CFG_LINEAR,
        # Activations
        "relu": _ACT_CFG,
        "gelu": _ACT_CFG,
        "silu": _ACT_CFG,
        "sigmoid": _ACT_CFG,
        "tanh": _ACT_CFG,
        "relu6": _ACT_CFG,
        "leaky_relu": _ACT_CFG,
        # Conv1d path — conv_transpose1d uses EMPTY_CFG (no MX reference)
        "conv1d": _CFG_CONV,
        "bn1d": _BN_CFG,
        # Conv2d path
        "conv2d": _CFG_CONV,
        "bn2d": _BN_CFG,
        "gn": _GN_CFG,
        "adaptive_pool": _POOL_CFG,
        "conv_transpose2d": _CFG_CONV_TRANSPOSE,
        # Conv3d path — conv_transpose3d uses EMPTY_CFG (no MX reference)
        "conv3d": _CFG_CONV,
        "bn3d": _BN_CFG,
        # Final
        "softmax": _SM_CFG,
    }


def _build_mx_reference(model: AllOpsModel, x: torch.Tensor):
    """Build the MX reference output by manually chaining each MX op.

    This mirrors AllOpsModel.forward() step-by-step using MX functional APIs.
    ops that MX does not support (conv_transpose1d, conv_transpose3d) are
    replaced with unquantized torch calls, so the reference accounts for them.
    """
    batch = x.shape[0]
    H = x.shape[1]  # 32

    # Extract weights from model
    w_ln1 = model.ln1.weight.data.clone()
    b_ln1 = model.ln1.bias.data.clone()
    w_l1 = model.linear1.weight.data.clone()
    b_l1 = model.linear1.bias.data.clone()
    w_l2 = model.linear2.weight.data.clone()
    b_l2 = model.linear2.bias.data.clone()
    w_l3 = model.linear3.weight.data.clone()
    b_l3 = model.linear3.bias.data.clone()
    w_ln2 = model.ln2.weight.data.clone()
    b_ln2 = model.ln2.bias.data.clone()

    # Conv1d weights
    w_c1d = model.conv1d.weight.data.clone()
    b_c1d = model.conv1d.bias.data.clone()
    w_ct1d = model.conv_transpose1d.weight.data.clone()
    b_ct1d = model.conv_transpose1d.bias.data.clone()
    bn1d_rm = model.bn1d.running_mean.clone()
    bn1d_rv = model.bn1d.running_var.clone()
    bn1d_w = model.bn1d.weight.data.clone()
    bn1d_b = model.bn1d.bias.data.clone()

    # Conv2d weights
    w_c2d = model.conv2d.weight.data.clone()
    b_c2d = model.conv2d.bias.data.clone()
    bn2d_rm = model.bn2d.running_mean.clone()
    bn2d_rv = model.bn2d.running_var.clone()
    bn2d_w = model.bn2d.weight.data.clone()
    bn2d_b = model.bn2d.bias.data.clone()
    gn_w = model.gn.weight.data.clone()
    gn_b = model.gn.bias.data.clone()
    w_ct2d = model.conv_transpose2d.weight.data.clone()
    b_ct2d = model.conv_transpose2d.bias.data.clone()

    # Conv3d weights
    w_c3d = model.conv3d.weight.data.clone()
    b_c3d = model.conv3d.bias.data.clone()
    bn3d_rm = model.bn3d.running_mean.clone()
    bn3d_rv = model.bn3d.running_var.clone()
    bn3d_w = model.bn3d.weight.data.clone()
    bn3d_b = model.bn3d.bias.data.clone()
    w_ct3d = model.conv_transpose3d.weight.data.clone()
    b_ct3d = model.conv_transpose3d.bias.data.clone()

    # Buffers
    matmul_w = model.matmul_w.clone()
    mm_w = model.mm_w.clone()
    bmm_w = model.bmm_w.clone()
    div_val = model.div_val.clone()

    # ── Forward chain (mirrors AllOpsModel.forward) ──
    residual = x
    x = _mx_layernorm(x, w_ln1, b_ln1)
    x = _mx_linear(x, w_l1, b_l1)
    x = _mx_relu(x)
    x = _mx_linear(x, w_l2, b_l2)
    x = _mx_add(x, residual)

    # Inline mul + sub
    x = _mx_mul(x, torch.sigmoid(x))
    x = _mx_sub(x, x.mean(dim=1, keepdim=True))

    # Activations
    x = _mx_gelu(x)
    x = _mx_silu(x)
    x = _mx_sigmoid(x)
    x = _mx_tanh(x)
    x = _mx_relu6(x)
    x = _mx_leaky_relu(x)

    # Conv1d path
    c1 = x.reshape(batch, 4, 8)
    c1 = _mx_conv1d(c1, w_c1d, b_c1d, stride=1, padding=1)
    c1 = _mx_batch_norm(c1, bn1d_rm, bn1d_rv, bn1d_w, bn1d_b)
    # MX has no conv_transpose1d — use plain torch
    c1 = nn.functional.conv_transpose1d(c1, w_ct1d, b_ct1d, padding=1)
    x_c1 = c1.reshape(batch, H)

    # Conv2d path
    c2 = x.reshape(batch, 4, 2, 4)
    c2 = _mx_conv2d(c2, w_c2d, b_c2d, stride=1, padding=1)
    c2 = _mx_batch_norm(c2, bn2d_rm, bn2d_rv, bn2d_w, bn2d_b)
    c2 = _mx_group_norm(c2, 2, gn_w, gn_b)
    c2 = _mx_pool(c2, (2, 4))
    c2 = _mx_conv_transpose2d(c2, w_ct2d, b_ct2d, stride=1, padding=1)
    x_c2 = c2.reshape(batch, H)

    # Conv3d path
    c3 = x.reshape(batch, 2, 2, 2, 4)
    c3 = _mx_conv3d(c3, w_c3d, b_c3d, stride=1, padding=1)
    c3 = _mx_batch_norm(c3, bn3d_rm, bn3d_rv, bn3d_w, bn3d_b)
    # MX has no conv_transpose3d — use plain torch
    c3 = nn.functional.conv_transpose3d(c3, w_ct3d, b_ct3d, padding=1)
    x_c3 = c3.reshape(batch, H)

    # Combine paths (left-to-right, matching AllOpsModel.forward associativity)
    x = _mx_add(x, x_c1)
    x = _mx_add(x, x_c2)
    x = _mx_add(x, x_c3)

    # ln2
    x = _mx_layernorm(x, w_ln2, b_ln2)

    # matmul
    x = _mx_matmul(x.unsqueeze(0), matmul_w).squeeze(0)

    # div
    x = _mx_div(x, div_val)

    # exp + log
    x = _mx_exp(x)
    x = torch.abs(x) + 1e-5
    x = _mx_log(x)

    # bmm
    x_3d = x.unsqueeze(1)
    x_3d = _mx_bmm(x_3d, bmm_w)
    x = x_3d.squeeze(1)

    # mm — mx.matmul handles 2D just like torch.mm
    x = _mx_matmul(x[:1], mm_w)

    # Explicit F.linear
    x = _mx_linear(x, matmul_w, None)

    # softmax + final linear
    x = _mx_softmax(x)
    x = _mx_linear(x, w_l3, b_l3)

    return x


def _layer_by_layer_test(model: AllOpsModel, x: torch.Tensor):
    """Step through forward manually, comparing each module to MX reference."""
    batch = x.shape[0]
    H = x.shape[1]

    from src.session._context import QuantizeContext
    from src.session._model import _resolve_context_cfg
    ctx_cfg = _resolve_context_cfg(_build_per_module_config(), _INLINE_OP_CFGS)

    layer_outputs_src = {}

    with QuantizeContext(model, ctx_cfg, op_cfgs=_INLINE_OP_CFGS), torch.no_grad():
        _x = x.clone()
        # Replicate the full forward, capturing module outputs directly
        residual = _x

        _x = model.ln1(_x); layer_outputs_src["ln1"] = _x.clone()
        _x = model.linear1(_x); layer_outputs_src["linear1"] = _x.clone()
        _x = model.relu(_x); layer_outputs_src["relu"] = _x.clone()
        _x = model.linear2(_x); layer_outputs_src["linear2"] = _x.clone()

        _x = _x + residual  # INLINE add

        _x = torch.mul(_x, torch.sigmoid(_x))
        _x = torch.sub(_x, _x.mean(dim=1, keepdim=True))

        _x = model.gelu(_x); layer_outputs_src["gelu"] = _x.clone()
        _x = model.silu(_x); layer_outputs_src["silu"] = _x.clone()
        _x = model.sigmoid(_x); layer_outputs_src["sigmoid"] = _x.clone()
        _x = model.tanh(_x); layer_outputs_src["tanh"] = _x.clone()
        _x = model.relu6(_x); layer_outputs_src["relu6"] = _x.clone()
        _x = model.leaky_relu(_x); layer_outputs_src["leaky_relu"] = _x.clone()

        c1 = _x.reshape(batch, 4, 8)
        c1 = model.conv1d(c1); layer_outputs_src["conv1d"] = c1.reshape(batch, H).clone()
        c1 = model.bn1d(c1); layer_outputs_src["bn1d"] = c1.reshape(batch, H).clone()
        c1 = model.conv_transpose1d(c1)
        x_c1 = c1.reshape(batch, H)

        c2 = _x.reshape(batch, 4, 2, 4)
        c2 = model.conv2d(c2); layer_outputs_src["conv2d"] = c2.reshape(batch, H).clone()
        c2 = model.bn2d(c2); layer_outputs_src["bn2d"] = c2.reshape(batch, H).clone()
        c2 = model.gn(c2); layer_outputs_src["gn"] = c2.reshape(batch, H).clone()
        c2 = model.adaptive_pool(c2)
        layer_outputs_src["adaptive_pool"] = c2.reshape(batch, H).clone()
        c2 = model.conv_transpose2d(c2); layer_outputs_src["conv_transpose2d"] = c2.reshape(batch, H).clone()

        c3 = _x.reshape(batch, 2, 2, 2, 4)
        c3 = model.conv3d(c3); layer_outputs_src["conv3d"] = c3.reshape(batch, H).clone()
        c3 = model.bn3d(c3); layer_outputs_src["bn3d"] = c3.reshape(batch, H).clone()
        c3 = model.conv_transpose3d(c3)
        x_c3 = c3.reshape(batch, H)

        x_c2 = c2.reshape(batch, H)
        _x = _x + x_c1 + x_c2 + x_c3

        _x = model.ln2(_x); layer_outputs_src["ln2"] = _x.clone()

        _x = torch.matmul(_x.unsqueeze(0), model.matmul_w).squeeze(0)
        _x = torch.div(_x, model.div_val)
        _x = torch.exp(_x)
        _x = torch.abs(_x) + 1e-5
        _x = torch.log(_x)
        _x_3d = _x.unsqueeze(1)
        _x_3d = torch.bmm(_x_3d, model.bmm_w)
        _x = _x_3d.squeeze(1)
        _x = torch.mm(_x[:1], model.mm_w)
        _x = nn.functional.linear(_x, model.matmul_w)

        _x = model.softmax(_x); layer_outputs_src["softmax"] = _x.clone()
        _x = model.linear3(_x); layer_outputs_src["linear3"] = _x.clone()

    # --- MX reference layer outputs ---
    layer_outputs_mx = {}

    w_ln1 = model.ln1.weight.data.clone()
    b_ln1 = model.ln1.bias.data.clone()
    w_l1 = model.linear1.weight.data.clone()
    b_l1 = model.linear1.bias.data.clone()
    w_l2 = model.linear2.weight.data.clone()
    b_l2 = model.linear2.bias.data.clone()
    w_l3 = model.linear3.weight.data.clone()
    b_l3 = model.linear3.bias.data.clone()
    w_ln2 = model.ln2.weight.data.clone()
    b_ln2 = model.ln2.bias.data.clone()

    # Conv1d
    w_c1d = model.conv1d.weight.data.clone(); b_c1d = model.conv1d.bias.data.clone()
    w_ct1d = model.conv_transpose1d.weight.data.clone(); b_ct1d = model.conv_transpose1d.bias.data.clone()
    bn1d_rm = model.bn1d.running_mean.clone(); bn1d_rv = model.bn1d.running_var.clone()
    bn1d_w = model.bn1d.weight.data.clone(); bn1d_b = model.bn1d.bias.data.clone()

    # Conv2d
    w_c2d = model.conv2d.weight.data.clone(); b_c2d = model.conv2d.bias.data.clone()
    bn2d_rm = model.bn2d.running_mean.clone(); bn2d_rv = model.bn2d.running_var.clone()
    bn2d_w = model.bn2d.weight.data.clone(); bn2d_b = model.bn2d.bias.data.clone()
    gn_w = model.gn.weight.data.clone(); gn_b = model.gn.bias.data.clone()
    w_ct2d = model.conv_transpose2d.weight.data.clone(); b_ct2d = model.conv_transpose2d.bias.data.clone()

    # Conv3d
    w_c3d = model.conv3d.weight.data.clone(); b_c3d = model.conv3d.bias.data.clone()
    bn3d_rm = model.bn3d.running_mean.clone(); bn3d_rv = model.bn3d.running_var.clone()
    bn3d_w = model.bn3d.weight.data.clone(); bn3d_b = model.bn3d.bias.data.clone()
    w_ct3d = model.conv_transpose3d.weight.data.clone(); b_ct3d = model.conv_transpose3d.bias.data.clone()

    matmul_w = model.matmul_w.clone()
    mm_w = model.mm_w.clone()
    bmm_w = model.bmm_w.clone()
    div_val = model.div_val.clone()

    mx_x = x.clone()
    residual = mx_x.clone()

    mx_x = _mx_layernorm(mx_x, w_ln1, b_ln1)
    layer_outputs_mx["ln1"] = mx_x.clone()

    mx_x = _mx_linear(mx_x, w_l1, b_l1)
    layer_outputs_mx["linear1"] = mx_x.clone()

    mx_x = _mx_relu(mx_x)
    layer_outputs_mx["relu"] = mx_x.clone()

    mx_x = _mx_linear(mx_x, w_l2, b_l2)
    layer_outputs_mx["linear2"] = mx_x.clone()

    mx_x = _mx_add(mx_x, residual)
    layer_outputs_mx["add_residual"] = mx_x.clone()

    mx_x = _mx_mul(mx_x, torch.sigmoid(mx_x))
    layer_outputs_mx["inline_mul"] = mx_x.clone()

    mx_x = _mx_sub(mx_x, mx_x.mean(dim=1, keepdim=True))
    layer_outputs_mx["inline_sub"] = mx_x.clone()

    mx_x = _mx_gelu(mx_x)
    layer_outputs_mx["gelu"] = mx_x.clone()

    mx_x = _mx_silu(mx_x)
    layer_outputs_mx["silu"] = mx_x.clone()

    mx_x = _mx_sigmoid(mx_x)
    layer_outputs_mx["sigmoid"] = mx_x.clone()

    mx_x = _mx_tanh(mx_x)
    layer_outputs_mx["tanh"] = mx_x.clone()

    mx_x = _mx_relu6(mx_x)
    layer_outputs_mx["relu6"] = mx_x.clone()

    mx_x = _mx_leaky_relu(mx_x)
    layer_outputs_mx["leaky_relu"] = mx_x.clone()

    # Conv1d path
    c1 = mx_x.reshape(batch, 4, 8)
    c1 = _mx_conv1d(c1, w_c1d, b_c1d, stride=1, padding=1)
    layer_outputs_mx["conv1d"] = c1.reshape(batch, H).clone()
    c1 = _mx_batch_norm(c1, bn1d_rm, bn1d_rv, bn1d_w, bn1d_b)
    layer_outputs_mx["bn1d"] = c1.reshape(batch, H).clone()
    c1 = nn.functional.conv_transpose1d(c1, w_ct1d, b_ct1d, padding=1)
    layer_outputs_mx["conv_transpose1d"] = c1.reshape(batch, H).clone()

    # Conv2d path
    c2 = mx_x.reshape(batch, 4, 2, 4)
    c2 = _mx_conv2d(c2, w_c2d, b_c2d, stride=1, padding=1)
    layer_outputs_mx["conv2d"] = c2.reshape(batch, H).clone()
    c2 = _mx_batch_norm(c2, bn2d_rm, bn2d_rv, bn2d_w, bn2d_b)
    layer_outputs_mx["bn2d"] = c2.reshape(batch, H).clone()
    c2 = _mx_group_norm(c2, 2, gn_w, gn_b)
    layer_outputs_mx["gn"] = c2.reshape(batch, H).clone()
    c2 = _mx_pool(c2, (2, 4))
    layer_outputs_mx["adaptive_pool"] = c2.reshape(batch, H).clone()
    c2 = _mx_conv_transpose2d(c2, w_ct2d, b_ct2d, stride=1, padding=1)
    layer_outputs_mx["conv_transpose2d"] = c2.reshape(batch, H).clone()

    # Conv3d path
    c3 = mx_x.reshape(batch, 2, 2, 2, 4)
    c3 = _mx_conv3d(c3, w_c3d, b_c3d, stride=1, padding=1)
    layer_outputs_mx["conv3d"] = c3.reshape(batch, H).clone()
    c3 = _mx_batch_norm(c3, bn3d_rm, bn3d_rv, bn3d_w, bn3d_b)
    layer_outputs_mx["bn3d"] = c3.reshape(batch, H).clone()
    c3 = nn.functional.conv_transpose3d(c3, w_ct3d, b_ct3d, padding=1)
    layer_outputs_mx["conv_transpose3d"] = c3.reshape(batch, H).clone()

    # Combine
    x_c1 = c1.reshape(batch, H)
    x_c2 = c2.reshape(batch, H)
    x_c3 = c3.reshape(batch, H)
    mx_x = _mx_add(mx_x, x_c1)
    mx_x = _mx_add(mx_x, x_c2)
    mx_x = _mx_add(mx_x, x_c3)

    mx_x = _mx_layernorm(mx_x, w_ln2, b_ln2)
    layer_outputs_mx["ln2"] = mx_x.clone()

    mx_x = _mx_matmul(mx_x.unsqueeze(0), matmul_w).squeeze(0)
    mx_x = _mx_div(mx_x, div_val)
    mx_x = _mx_exp(mx_x)
    mx_x = torch.abs(mx_x) + 1e-5
    mx_x = _mx_log(mx_x)

    x_3d = mx_x.unsqueeze(1)
    x_3d = _mx_bmm(x_3d, bmm_w)
    mx_x = x_3d.squeeze(1)

    mx_x = _mx_matmul(mx_x[:1], mm_w)
    mx_x = _mx_linear(mx_x, matmul_w, None)

    mx_x = _mx_softmax(mx_x)
    layer_outputs_mx["softmax"] = mx_x.clone()

    mx_x = _mx_linear(mx_x, w_l3, b_l3)
    layer_outputs_mx["linear3"] = mx_x.clone()

    # ── Compare ──
    all_ok = True
    expected_layers = [
        "ln1", "linear1", "relu", "linear2",
        "gelu", "silu", "sigmoid", "tanh", "relu6", "leaky_relu",
        "conv1d", "bn1d", "conv2d", "bn2d", "gn", "adaptive_pool",
        "conv3d", "bn3d", "ln2", "softmax", "linear3",
    ]

    for layer_name in expected_layers:
        mx_out = layer_outputs_mx.get(layer_name)
        src_out = layer_outputs_src.get(layer_name)
        if mx_out is None:
            print(f"  {layer_name:20s}: SKIP (mx missing)")
            continue
        if src_out is None:
            print(f"  {layer_name:20s}: SKIP (src hook missing)")
            continue

        # Flatten both to (B, -1) for comparison (shape may differ slightly)
        diff = float(torch.max(torch.abs(mx_out.reshape(mx_out.shape[0], -1) -
                                          src_out.reshape(src_out.shape[0], -1))))
        ok = diff == 0
        ok = diff == 0
        if not ok:
            all_ok = False
        status = "✓" if ok else f"✗ max_abs_diff={diff:.6e}"
        print(f"  {layer_name:20s}: {status}")

    return all_ok


def _assert_and_print(mx_ref, src_out, label):
    """Print bit-exact comparison result."""
    if torch.equal(mx_ref, src_out):
        print(f"  Result: BIT-EXACT ✓")
        return True
    diff = float(torch.max(torch.abs(mx_ref - src_out)))
    rel = diff / max(float(torch.max(torch.abs(mx_ref))), 1e-10)
    print(f"  Result: MISMATCH ✗ (max_abs_diff={diff:.6e}, max_rel_diff={rel:.6e})")
    return False


def _print_config_diff(user, compat):
    fields = ["storage", "input", "weight", "bias", "output"]
    for f in fields:
        uv = getattr(user, f)
        cv = getattr(compat, f)
        if uv is None and cv is None:
            continue
        u_str = _s(uv); c_str = _s(cv)
        ok = u_str == c_str
        print(f"  {f:10s}: user={u_str:45s} compat={c_str:45s} {'✓' if ok else '✗'}")


def _s(scheme):
    if scheme is None: return "None"
    f = str(scheme.format) if hasattr(scheme, 'format') else '?'
    g = str(scheme.granularity.mode) if hasattr(scheme, 'granularity') else '?'
    bs = getattr(scheme.granularity, 'block_size', None) if hasattr(scheme, 'granularity') else None
    if bs: g = f"{g}(bs={bs})"
    return f"{f} / {g}"


def _cfg_matches():
    user = _OP_CFG_USER
    compat = _CFG_LINEAR
    for f in ["storage", "input", "weight"]:
        uv = _s(getattr(user, f)); cv = _s(getattr(compat, f))
        if uv != cv:
            return False
    if user.output is not None or compat.output is not None:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
