"""
Parametrized end-to-end equivalence: quantize_model(AllOpsModel) vs MX reference.

Covers all 21 module types + 10 inline ops in a single forward pass,
with smoke (fast CI) and full (slow, all 8 MX formats) parametrization.
"""
import pytest
import torch
import torch.nn as nn

import mx
from mx import finalize_mx_specs
from mx.specs import apply_mx_specs
from mx.batchnorm import batch_norm as _mx_raw_bn
from mx.groupnorm import group_norm as _mx_raw_gn

from src.session._model import quantize_model
from src.tests._compat import (
    op_config_from_mx_specs,
    norm_config_from_mx_specs,
    softmax_config_from_mx_specs,
    activation_config_from_mx_specs,
    simd_config_from_mx_specs,
    pool_config_from_mx_specs,
)
from src.tests._formats import smoke_mx_specs_params, full_mx_specs_params

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_HIDDEN = 32  # = 4*8 = 4*2*4 = 2*2*2*2*2 (every reshape preserves elements)
_SEED = 42


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
        H = hidden

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

        # --- Conv1d path ---
        self.conv1d = nn.Conv1d(4, 4, 3, padding=1)
        self.bn1d = nn.BatchNorm1d(4)
        self.conv_transpose1d = nn.ConvTranspose1d(4, 4, 3, padding=1)

        # --- Conv2d path ---
        self.conv2d = nn.Conv2d(4, 4, 3, padding=1)
        self.bn2d = nn.BatchNorm2d(4)
        self.gn = nn.GroupNorm(2, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))
        self.conv_transpose2d = nn.ConvTranspose2d(4, 4, 3, padding=1)

        # --- Conv3d path ---
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
        batch = x.shape[0]
        H = x.shape[1]

        # ---- 1D section ----
        residual = x
        x = self.ln1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x + residual               # INLINE add (Tensor.__add__)

        x = torch.mul(x, torch.sigmoid(x))   # INLINE mul
        x = torch.sub(x, x.mean(dim=1, keepdim=True))  # INLINE sub

        # ---- Activations ----
        x = self.gelu(x)
        x = self.silu(x)
        x = self.sigmoid(x)
        x = self.tanh(x)
        x = self.relu6(x)
        x = self.leaky_relu(x)

        # ---- Conv1d path ----
        c1 = x.reshape(batch, 4, 8)
        c1 = self.conv1d(c1)
        c1 = self.bn1d(c1)
        c1 = self.conv_transpose1d(c1)
        x_c1 = c1.reshape(batch, H)

        # ---- Conv2d path ----
        c2 = x.reshape(batch, 4, 2, 4)
        c2 = self.conv2d(c2)
        c2 = self.bn2d(c2)
        c2 = self.gn(c2)
        c2 = self.adaptive_pool(c2)
        c2 = self.conv_transpose2d(c2)
        x_c2 = c2.reshape(batch, H)

        # ---- Conv3d path ----
        c3 = x.reshape(batch, 2, 2, 2, 4)
        c3 = self.conv3d(c3)
        c3 = self.bn3d(c3)
        c3 = self.conv_transpose3d(c3)
        x_c3 = c3.reshape(batch, H)

        # ---- Combine all paths ----
        x = x + x_c1 + x_c2 + x_c3

        # ---- More inline ops ----
        x = self.ln2(x)
        x = torch.matmul(x.unsqueeze(0), self.matmul_w).squeeze(0)
        x = torch.div(x, self.div_val)
        x = torch.exp(x)
        x = torch.abs(x) + 1e-5
        x = torch.log(x)
        x_3d = x.unsqueeze(1)
        x_3d = torch.bmm(x_3d, self.bmm_w)
        x = x_3d.squeeze(1)
        x = torch.mm(x[:1], self.mm_w)
        x = nn.functional.linear(x, self.matmul_w)

        # ---- Final ----
        x = self.softmax(x)
        x = self.linear3(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# MX reference builder (accepts parametrized mx_specs)
# ═══════════════════════════════════════════════════════════════════════════

def _build_mx_reference(model: AllOpsModel, x: torch.Tensor, mx_specs: dict):
    """Build the MX reference output by manually chaining each MX op.

    Mirrors AllOpsModel.forward() step-by-step using MX functional APIs.
    ConvTranspose1d/3d use plain torch (mx doesn't support them).
    """
    fmx = apply_mx_specs(mx_specs)

    def _mx_linear(x, w, b):
        return mx.linear(x, w, b, mx_specs=fmx)

    def _mx_matmul(a, b):
        return mx.matmul(a, b, mx_specs=fmx)

    def _mx_bmm(a, b):
        return mx.bmm(a, b, mx_specs=fmx)

    def _mx_layernorm(x, w, b):
        return mx.layer_norm(x, w.shape, w, b, eps=1e-5, mx_specs=fmx)

    def _mx_softmax(x):
        return mx.softmax(x, dim=-1, mx_specs=fmx)

    def _mx_relu(x):
        return mx.relu(x, mx_specs=fmx)

    def _mx_gelu(x):
        return mx.gelu(x, mx_specs=fmx)

    def _mx_silu(x):
        return mx.silu(x, mx_specs=fmx)

    def _mx_sigmoid(x):
        return mx.sigmoid(x, mx_specs=fmx)

    def _mx_tanh(x):
        return mx.tanh(x, mx_specs=fmx)

    def _mx_relu6(x):
        return mx.relu6(x, mx_specs=fmx)

    def _mx_leaky_relu(x):
        return mx.leaky_relu(x, mx_specs=fmx)

    def _mx_conv1d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        return mx.conv1d(x, w, b, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, mx_specs=fmx)

    def _mx_conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        return mx.conv2d(x, w, b, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, mx_specs=fmx)

    def _mx_conv3d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        return mx.conv3d(x, w, b, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, mx_specs=fmx)

    def _mx_batch_norm(x, rm, rv, w, b):
        return _mx_raw_bn(x, rm, rv, w, b,
                          is_training=False, momentum=0.1, eps=1e-5,
                          mx_specs=fmx)

    def _mx_group_norm(x, num_groups, w, b):
        return _mx_raw_gn(x, num_groups, w, b, eps=1e-5, mx_specs=fmx)

    def _mx_conv_transpose2d(x, w, b=None, stride=1, padding=0):
        m = mx.ConvTranspose2d(
            w.shape[0], w.shape[1], kernel_size=w.shape[2:],
            stride=stride, padding=padding, bias=b is not None,
            mx_specs=fmx,
        )
        m.weight.data.copy_(w)
        if b is not None:
            m.bias.data.copy_(b)
        return m(x)

    def _mx_pool(x, output_size):
        return mx.adaptive_avg_pool2d(x, output_size, mx_specs=fmx)

    def _mx_add(a, b):
        return mx.simd_add(a, b, mx_specs=fmx)

    def _mx_sub(a, b):
        return mx.simd_sub(a, b, mx_specs=fmx)

    def _mx_mul(a, b):
        return mx.simd_mul(a, b, mx_specs=fmx)

    def _mx_div(a, b):
        return mx.simd_div(a, b, mx_specs=fmx)

    def _mx_exp(x):
        return mx.simd_exp(x, mx_specs=fmx)

    def _mx_log(x):
        return mx.simd_log(x, mx_specs=fmx)

    batch = x.shape[0]
    H = x.shape[1]

    # Extract weights from model
    w_ln1 = model.ln1.weight.data.clone(); b_ln1 = model.ln1.bias.data.clone()
    w_l1 = model.linear1.weight.data.clone(); b_l1 = model.linear1.bias.data.clone()
    w_l2 = model.linear2.weight.data.clone(); b_l2 = model.linear2.bias.data.clone()
    w_l3 = model.linear3.weight.data.clone(); b_l3 = model.linear3.bias.data.clone()
    w_ln2 = model.ln2.weight.data.clone(); b_ln2 = model.ln2.bias.data.clone()

    w_c1d = model.conv1d.weight.data.clone(); b_c1d = model.conv1d.bias.data.clone()
    w_ct1d = model.conv_transpose1d.weight.data.clone(); b_ct1d = model.conv_transpose1d.bias.data.clone()
    bn1d_rm = model.bn1d.running_mean.clone(); bn1d_rv = model.bn1d.running_var.clone()
    bn1d_w = model.bn1d.weight.data.clone(); bn1d_b = model.bn1d.bias.data.clone()

    w_c2d = model.conv2d.weight.data.clone(); b_c2d = model.conv2d.bias.data.clone()
    bn2d_rm = model.bn2d.running_mean.clone(); bn2d_rv = model.bn2d.running_var.clone()
    bn2d_w = model.bn2d.weight.data.clone(); bn2d_b = model.bn2d.bias.data.clone()
    gn_w = model.gn.weight.data.clone(); gn_b = model.gn.bias.data.clone()
    w_ct2d = model.conv_transpose2d.weight.data.clone(); b_ct2d = model.conv_transpose2d.bias.data.clone()

    w_c3d = model.conv3d.weight.data.clone(); b_c3d = model.conv3d.bias.data.clone()
    bn3d_rm = model.bn3d.running_mean.clone(); bn3d_rv = model.bn3d.running_var.clone()
    bn3d_w = model.bn3d.weight.data.clone(); bn3d_b = model.bn3d.bias.data.clone()
    w_ct3d = model.conv_transpose3d.weight.data.clone(); b_ct3d = model.conv_transpose3d.bias.data.clone()

    matmul_w = model.matmul_w.clone()
    mm_w = model.mm_w.clone()
    bmm_w = model.bmm_w.clone()
    div_val = model.div_val.clone()

    # Forward chain (mirrors AllOpsModel.forward)
    residual = x
    x = _mx_layernorm(x, w_ln1, b_ln1)
    x = _mx_linear(x, w_l1, b_l1)
    x = _mx_relu(x)
    x = _mx_linear(x, w_l2, b_l2)
    x = _mx_add(x, residual)

    x = _mx_mul(x, torch.sigmoid(x))
    x = _mx_sub(x, x.mean(dim=1, keepdim=True))

    x = _mx_gelu(x)
    x = _mx_silu(x)
    x = _mx_sigmoid(x)
    x = _mx_tanh(x)
    x = _mx_relu6(x)
    x = _mx_leaky_relu(x)

    c1 = x.reshape(batch, 4, 8)
    c1 = _mx_conv1d(c1, w_c1d, b_c1d, stride=1, padding=1)
    c1 = _mx_batch_norm(c1, bn1d_rm, bn1d_rv, bn1d_w, bn1d_b)
    c1 = nn.functional.conv_transpose1d(c1, w_ct1d, b_ct1d, padding=1)
    x_c1 = c1.reshape(batch, H)

    c2 = x.reshape(batch, 4, 2, 4)
    c2 = _mx_conv2d(c2, w_c2d, b_c2d, stride=1, padding=1)
    c2 = _mx_batch_norm(c2, bn2d_rm, bn2d_rv, bn2d_w, bn2d_b)
    c2 = _mx_group_norm(c2, 2, gn_w, gn_b)
    c2 = _mx_pool(c2, (2, 4))
    c2 = _mx_conv_transpose2d(c2, w_ct2d, b_ct2d, stride=1, padding=1)
    x_c2 = c2.reshape(batch, H)

    c3 = x.reshape(batch, 2, 2, 2, 4)
    c3 = _mx_conv3d(c3, w_c3d, b_c3d, stride=1, padding=1)
    c3 = _mx_batch_norm(c3, bn3d_rm, bn3d_rv, bn3d_w, bn3d_b)
    c3 = nn.functional.conv_transpose3d(c3, w_ct3d, b_ct3d, padding=1)
    x_c3 = c3.reshape(batch, H)

    x = _mx_add(x, x_c1)
    x = _mx_add(x, x_c2)
    x = _mx_add(x, x_c3)

    x = _mx_layernorm(x, w_ln2, b_ln2)
    x = _mx_matmul(x.unsqueeze(0), matmul_w).squeeze(0)
    x = _mx_div(x, div_val)
    x = _mx_exp(x)
    x = torch.abs(x) + 1e-5
    x = _mx_log(x)
    x_3d = x.unsqueeze(1)
    x_3d = _mx_bmm(x_3d, bmm_w)
    x = x_3d.squeeze(1)
    x = _mx_matmul(x[:1], mm_w)
    x = _mx_linear(x, matmul_w, None)
    x = _mx_softmax(x)
    x = _mx_linear(x, w_l3, b_l3)

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Per-module config builder (mirrors _build_per_module_config)
# ═══════════════════════════════════════════════════════════════════════════

def _build_per_module_config(mx_specs: dict):
    """Return a dict mapping every module name -> OpQuantConfig."""
    CFG_LINEAR = op_config_from_mx_specs(mx_specs, op_type="linear")
    CFG_CONV = op_config_from_mx_specs(mx_specs, op_type="conv")
    CFG_CONV_TRANSPOSE = op_config_from_mx_specs(mx_specs, op_type="conv_transpose")
    CFG_MATMUL = op_config_from_mx_specs(mx_specs, op_type="matmul")
    NORM_CFG, _, _ = norm_config_from_mx_specs(mx_specs, op_type="layer_norm")
    BN_CFG, _, _ = norm_config_from_mx_specs(mx_specs, op_type="batch_norm")
    GN_CFG, _, _ = norm_config_from_mx_specs(mx_specs, op_type="group_norm")
    SM_CFG, _ = softmax_config_from_mx_specs(mx_specs)
    ACT_CFG = activation_config_from_mx_specs(mx_specs)
    POOL_CFG = pool_config_from_mx_specs(mx_specs)

    return {
        "ln1": NORM_CFG, "ln2": NORM_CFG,
        "linear1": CFG_LINEAR, "linear2": CFG_LINEAR, "linear3": CFG_LINEAR,
        "relu": ACT_CFG, "gelu": ACT_CFG, "silu": ACT_CFG,
        "sigmoid": ACT_CFG, "tanh": ACT_CFG,
        "relu6": ACT_CFG, "leaky_relu": ACT_CFG,
        "conv1d": CFG_CONV, "bn1d": BN_CFG,
        "conv2d": CFG_CONV, "bn2d": BN_CFG,
        "gn": GN_CFG, "adaptive_pool": POOL_CFG,
        "conv_transpose2d": CFG_CONV_TRANSPOSE,
        "conv3d": CFG_CONV, "bn3d": BN_CFG,
        "softmax": SM_CFG,
    }


_INLINE_OP_CFGS_CACHE: dict = {}

def _get_inline_op_cfgs(mx_specs: dict):
    """Cached inline op cfgs for matmul family intercepted by QuantizeContext."""
    key = frozenset(mx_specs.items())
    if key not in _INLINE_OP_CFGS_CACHE:
        CFG_MATMUL = op_config_from_mx_specs(mx_specs, op_type="matmul")
        CFG_LINEAR = op_config_from_mx_specs(mx_specs, op_type="linear")
        _INLINE_OP_CFGS_CACHE[key] = {
            "matmul": CFG_MATMUL, "mm": CFG_MATMUL,
            "bmm": CFG_MATMUL, "linear": CFG_LINEAR,
        }
    return _INLINE_OP_CFGS_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════
# Smoke E2E tests (fast, runs in CI)
# ═══════════════════════════════════════════════════════════════════════════

SMOKE_SPECS = [p for p in smoke_mx_specs_params() if p.values[1] != {}]
FULL_SPECS = [p for p in full_mx_specs_params() if p.values[1] != {}]


class TestE2EAllOpsSmoke:
    """Full-model E2E equivalence with smoke parametrization (5 configs).

    Uses per-module dict configs — the most granular approach that mirrors
    mx's per-operator quantization.  This is the gold standard for bit-exact
    equivalence verification.
    """

    @pytest.mark.parametrize("name,mx_specs", SMOKE_SPECS)
    def test_per_module_config(self, name, mx_specs):
        torch.manual_seed(_SEED)
        model = AllOpsModel()
        per_module = _build_per_module_config(mx_specs)
        op_cfgs = _get_inline_op_cfgs(mx_specs)
        quantize_model(model, per_module, op_cfgs=op_cfgs)
        model.eval()

        x = torch.randn(2, _HIDDEN)
        with torch.no_grad():
            src_out = model(x.clone())
        mx_ref = _build_mx_reference(model, x.clone(), mx_specs)

        assert torch.equal(mx_ref, src_out), (
            f"E2E per-module mismatch ({name}): "
            f"max diff={torch.max(torch.abs(mx_ref - src_out))}"
        )

    def test_all_module_types_quantized(self):
        """Verify that ALL 21 module types in _MODULE_MAPPING are exercised."""
        from src.ops.linear import QuantizedLinear
        from src.ops.conv import (
            QuantizedConv1d, QuantizedConv2d, QuantizedConv3d,
            QuantizedConvTranspose1d, QuantizedConvTranspose2d, QuantizedConvTranspose3d,
        )
        from src.ops.norm import (
            QuantizedBatchNorm1d, QuantizedBatchNorm2d, QuantizedBatchNorm3d,
            QuantizedLayerNorm, QuantizedGroupNorm,
        )
        from src.ops.activations import (
            QuantizedReLU, QuantizedGELU, QuantizedSiLU, QuantizedSigmoid,
            QuantizedTanh, QuantizedReLU6, QuantizedLeakyReLU,
        )
        from src.ops.softmax import QuantizedSoftmax
        from src.ops.pooling import QuantizedAdaptiveAvgPool2d

        expected_types = {
            QuantizedLinear,
            QuantizedConv1d, QuantizedConv2d, QuantizedConv3d,
            QuantizedConvTranspose1d, QuantizedConvTranspose2d, QuantizedConvTranspose3d,
            QuantizedBatchNorm1d, QuantizedBatchNorm2d, QuantizedBatchNorm3d,
            QuantizedLayerNorm, QuantizedGroupNorm,
            QuantizedReLU, QuantizedGELU, QuantizedSiLU, QuantizedSigmoid,
            QuantizedTanh, QuantizedReLU6, QuantizedLeakyReLU,
            QuantizedSoftmax, QuantizedAdaptiveAvgPool2d,
        }
        assert len(expected_types) == 21, f"Expected 21 types, got {len(expected_types)}"

        torch.manual_seed(_SEED)
        model = AllOpsModel()
        # Use the first smoke config
        first_spec = SMOKE_SPECS[0].values[1]
        per_module = _build_per_module_config(first_spec)
        op_cfgs = _get_inline_op_cfgs(first_spec)
        quantize_model(model, per_module, op_cfgs=op_cfgs)

        found_types = set()
        for name, module in model.named_modules():
            found_types.add(type(module))

        quantized_found = found_types & expected_types
        missing = expected_types - found_types

        assert len(missing) == 0, (
            f"Missing quantized module types: "
            f"{sorted(t.__name__ for t in missing)}"
        )
        assert len(quantized_found) == 21, (
            f"Expected 21 quantized types, found {len(quantized_found)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Full E2E tests (all 8 MX formats, slow)
# ═══════════════════════════════════════════════════════════════════════════

class TestE2EAllOpsFull:
    """Full-model E2E equivalence with all 8 MX format variants."""

    @pytest.mark.slow
    @pytest.mark.parametrize("name,mx_specs", FULL_SPECS)
    def test_per_module_config(self, name, mx_specs):
        torch.manual_seed(_SEED)
        model = AllOpsModel()
        per_module = _build_per_module_config(mx_specs)
        op_cfgs = _get_inline_op_cfgs(mx_specs)
        quantize_model(model, per_module, op_cfgs=op_cfgs)
        model.eval()

        x = torch.randn(2, _HIDDEN)
        with torch.no_grad():
            src_out = model(x.clone())
        mx_ref = _build_mx_reference(model, x.clone(), mx_specs)

        assert torch.equal(mx_ref, src_out), (
            f"E2E full per-module mismatch ({name}): "
            f"max diff={torch.max(torch.abs(mx_ref - src_out))}"
        )
