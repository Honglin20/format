"""
Bit-exact equivalence verification: Transformer block, forward + backward.

Verifies that ``quantize_nonlinear=False`` produces bit-exact identical results
to MX for a standard Transformer encoder block across the full storage × compute
matrix.

Architecture covered:
  - Linear (Q/K/V/O/FFN1/FFN2)       → matmul-family (MX per_block)
  - LayerNorm × 2                      → non-linear (elemwise only in MX)
  - Softmax (attention)                → non-linear (elemwise only in MX)
  - GELU (FFN)                         → non-linear (elemwise only in MX)
  - torch.matmul inline (Q@K^T, A@V)  → matmul-family inline
  - torch.add (residual)               → SIMD inline (elemwise only in MX)
  - torch.div (scale)                  → SIMD inline

Comparison matrix:
  Storage:     none  |  bfloat=16  |  fp=8
  MX compute:  none  |  fp8_e4m3   |  int8  |  int4  (per_block)
  Direction:   forward  |  backward (quantize_backprop=True)

Usage:
  python tools/verify_transformer_equiv.py          # smoke (3 configs)
  python tools/verify_transformer_equiv.py --full   # all 12 configs
"""
import argparse
import sys
import traceback
from typing import Dict, List

import torch
import torch.nn as nn

import mx
from mx.specs import apply_mx_specs, get_backwards_mx_specs

# ══════════════════════════════════════════════════════════════════════════════
# Comparison matrix definition
# ══════════════════════════════════════════════════════════════════════════════

STORAGE_CONFIGS = {
    "none":   {},
    "bf16":   {"bfloat": 16},
    "fp8":    {"fp": 8},
}

COMPUTE_CONFIGS = {
    "none":      {},
    "fp8_e4m3":  {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32},
    "int8":      {"w_elem_format": "int8",     "a_elem_format": "int8",     "block_size": 32},
    "int4":      {"w_elem_format": "int4",     "a_elem_format": "int4",     "block_size": 32},
}

SMOKE_COMBINATIONS = [
    ("none", "fp8_e4m3"),    # pure MX (bfloat=0, fp=0)
    ("bf16", "int4"),         # bf16 storage + int4 compute
    ("bf16", "fp8_e4m3"),    # bf16 storage + fp8 compute
]

SMOKE_COMBINATIONS_FP = [
    ("fp8", "fp8_e4m3"),     # fp8 storage + fp8 compute
    ("fp8", "int4"),          # fp8 storage + int4 compute
]

FULL_STORAGE = list(STORAGE_CONFIGS.keys())
FULL_COMPUTE = list(COMPUTE_CONFIGS.keys())

BLOCK_SIZE = 32
HIDDEN_DIM = 64
NUM_HEADS = 4
SEQ_LEN = 8
BATCH = 2
ATOL = 1e-6

# ══════════════════════════════════════════════════════════════════════════════
# Standard Transformer encoder block (for verification)
# ══════════════════════════════════════════════════════════════════════════════


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block exercising all operator categories.

    Operator inventory per forward pass:
      Matmul-family (module):  Q/K/V/O/FFN1/FFN2  → 6 × nn.Linear
      Matmul-family (inline):  Q@K^T  +  A@V       → 2 × torch.matmul
      Non-linear (module):     LayerNorm × 2        → 2 × nn.LayerNorm
      Non-linear (module):     GELU                 → 1 × nn.GELU
      Non-linear (module):     Softmax              → 1 × nn.Softmax
      SIMD (inline):           residual add × 2     → 2 × torch.add
      SIMD (inline):           scale (div)          → 1 × torch.div
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Attention projections (matmul-family modules)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # LayerNorm (non-linear modules)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # FFN (matmul-family + non-linear)
        self.ffn1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.gelu = nn.GELU()
        self.ffn2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # ---- Self-Attention ----
        residual = x
        x = self.ln1(x)

        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Inline matmul + scale + softmax
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = torch.div(attn_scores, scale)          # SIMD inline
        attn_weights = self.softmax(attn_scores)

        # Inline matmul
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)

        x = torch.add(residual, attn_out)                    # SIMD inline (residual)

        # ---- FFN ----
        residual = x
        x = self.ln2(x)
        x = self.ffn1(x)
        x = self.gelu(x)
        x = self.ffn2(x)
        x = torch.add(residual, x)                           # SIMD inline (residual)

        return x


# ══════════════════════════════════════════════════════════════════════════════
# MX reference: manual forward pass using mx.* functional APIs
# ══════════════════════════════════════════════════════════════════════════════


def _mx_transformer_forward(model: TransformerEncoderBlock, x, mx_specs):
    """Run a transformer encoder block forward pass using mx.* function APIs."""
    fmx = apply_mx_specs(mx_specs)
    bfmx = get_backwards_mx_specs(fmx) if fmx.get("quantize_backprop", True) else None

    # ---- Self-Attention ----
    residual = x
    x_norm = mx.layer_norm(x, model.ln1.weight.shape, model.ln1.weight, model.ln1.bias,
                           eps=model.ln1.eps, mx_specs=fmx)

    B, S, D = x_norm.shape
    H = model.num_heads
    hd = model.head_dim

    # Q/K/V projections via mx.linear
    q = mx.linear(x_norm, model.q_proj.weight, model.q_proj.bias, mx_specs=fmx)
    k = mx.linear(x_norm, model.k_proj.weight, model.k_proj.bias, mx_specs=fmx)
    v = mx.linear(x_norm, model.v_proj.weight, model.v_proj.bias, mx_specs=fmx)
    q = q.view(B, S, H, hd).transpose(1, 2)
    k = k.view(B, S, H, hd).transpose(1, 2)
    v = v.view(B, S, H, hd).transpose(1, 2)

    scale = hd ** 0.5
    attn_scores = mx.matmul(q, k.transpose(-2, -1), mx_specs=fmx)
    attn_scores = mx.simd_div(attn_scores, scale, mx_specs=fmx)
    attn_weights = mx.softmax(attn_scores, dim=-1, mx_specs=fmx)
    attn_out = mx.matmul(attn_weights, v, mx_specs=fmx)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    attn_out = mx.linear(attn_out, model.out_proj.weight, model.out_proj.bias, mx_specs=fmx)

    attn_out = mx.simd_add(residual, attn_out, mx_specs=fmx)

    # ---- FFN ----
    residual = attn_out
    x_norm2 = mx.layer_norm(attn_out, model.ln2.weight.shape, model.ln2.weight, model.ln2.bias,
                            eps=model.ln2.eps, mx_specs=fmx)
    ffn1_out = mx.linear(x_norm2, model.ffn1.weight, model.ffn1.bias, mx_specs=fmx)
    gelu_out = mx.gelu(ffn1_out, mx_specs=fmx)
    ffn2_out = mx.linear(gelu_out, model.ffn2.weight, model.ffn2.bias, mx_specs=fmx)
    output = mx.simd_add(residual, ffn2_out, mx_specs=fmx)

    return output


def _mx_transformer_backward(output):
    """Run backward on the sum of the output to exercise QAT backward paths."""
    loss = output.sum()
    loss.backward()


# ══════════════════════════════════════════════════════════════════════════════
# Src reference: quantize_model(quantize_nonlinear=False) + forward/backward
# ══════════════════════════════════════════════════════════════════════════════


def _build_op_config(storage_key: str, compute_key: str) -> "OpQuantConfig":
    """Build OpQuantConfig from storage + compute keys."""
    from src.scheme.op_config import OpQuantConfig
    from src.scheme.quant_scheme import QuantScheme
    from src.scheme.granularity import GranularitySpec
    from src.formats.base import FormatBase

    # Storage (elemwise)
    storage = None
    storage_cfg = STORAGE_CONFIGS[storage_key]
    if storage_cfg.get("bfloat", 0) > 0:
        from src.formats.bf16_fp16 import BFloat16Format
        storage = QuantScheme(
            format=BFloat16Format(),
            granularity=GranularitySpec.per_tensor(),
        )
    elif storage_cfg.get("fp", 0) > 0:
        storage = QuantScheme(
            format=FormatBase.from_str("fp8_e5m2"),
            granularity=GranularitySpec.per_tensor(),
        )

    # MX compute (per_block)
    input_scheme = None
    weight_scheme = None
    compute_cfg = COMPUTE_CONFIGS[compute_key]
    if "w_elem_format" in compute_cfg:
        fmt = FormatBase.from_str(compute_cfg["w_elem_format"])
        block_size = compute_cfg.get("block_size", 32)
        weight_scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec.per_block(size=block_size, axis=-1),
        )
        input_scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec.per_block(size=block_size, axis=-1),
        )

    return OpQuantConfig(input=input_scheme, weight=weight_scheme, storage=storage)


def _src_forward_and_backward(model, x, op_cfg):
    """quantize model, run forward + backward, return output + gradients."""
    import copy
    from src.session._model import quantize_model
    from src.session._context import _EMPTY_CFG

    qmodel = quantize_model(
        copy.deepcopy(model),
        op_cfg,
        quantize_nonlinear=False,
    )
    qmodel.eval()

    # Clone input for gradient tracking
    x = x.clone().requires_grad_(True)
    output = qmodel(x)
    loss = output.sum()
    loss.backward()

    # Collect gradients for comparison
    grads = {}
    for name, param in qmodel.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    grads["x"] = x.grad.clone() if x.grad is not None else None

    return output.detach(), grads, qmodel


def _run_mx_forward_and_backward(model, x, mx_specs):
    """Run MX reference forward + backward, return output + gradients."""
    # Clone model for MX grading
    import copy
    mx_model = copy.deepcopy(model)
    x = x.clone().requires_grad_(True)

    output = _mx_transformer_forward(mx_model, x, mx_specs)
    _mx_transformer_backward(output)

    grads = {}
    for name, param in mx_model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    grads["x"] = x.grad.clone() if x.grad is not None else None

    return output.detach(), grads


# ══════════════════════════════════════════════════════════════════════════════
# Comparison driver
# ══════════════════════════════════════════════════════════════════════════════


def _bit_exact_check(a, b, label: str) -> bool:
    """Check bit-exact equality, reporting max diff on failure."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        print(f"  ✗ {label}: one is None (a={a is not None}, b={b is not None})")
        return False
    if torch.equal(a, b):
        return True
    max_diff = (a.float() - b.float()).abs().max().item() if a.dtype != b.dtype else (a - b).abs().max().item()
    print(f"  ✗ {label}: NOT bit-exact, max diff={max_diff:.6e}")
    return False


def compare_config(storage_key: str, compute_key: str,
                   model: TransformerEncoderBlock, x: torch.Tensor) -> Dict:
    """Compare src vs MX for one storage×compute combination."""
    from src.session._model import _is_mx_compute

    mx_specs = {}
    mx_specs.update(STORAGE_CONFIGS[storage_key])
    mx_specs.update(COMPUTE_CONFIGS[compute_key])
    mx_specs["quantize_backprop"] = True

    op_cfg = _build_op_config(storage_key, compute_key)

    result = {
        "storage": storage_key,
        "compute": compute_key,
        "op_cfg": str(op_cfg),
        "forward": None,
        "backward": None,
        "errors": [],
    }

    # ---- Forward comparison ----
    try:
        mx_out, mx_grads = _run_mx_forward_and_backward(model, x, mx_specs)
        src_out, src_grads, _ = _src_forward_and_backward(model, x, op_cfg)

        fwd_ok = _bit_exact_check(mx_out, src_out, "forward/output")
        result["forward"] = "PASS" if fwd_ok else "FAIL"
        if not fwd_ok:
            result["errors"].append("forward/output")

        # ---- Backward comparison (gradients) ----
        bwd_all_ok = True
        all_param_names = set(list(mx_grads.keys()) + list(src_grads.keys()))
        for pname in sorted(all_param_names):
            mx_g = mx_grads.get(pname)
            src_g = src_grads.get(pname)
            ok = _bit_exact_check(mx_g, src_g, f"backward/{pname}")
            if not ok:
                bwd_all_ok = False
                result["errors"].append(f"backward/{pname}")

        result["backward"] = "PASS" if bwd_all_ok else "FAIL"

    except Exception as e:
        result["forward"] = "ERROR"
        result["backward"] = "ERROR"
        result["errors"].append(f"{type(e).__name__}: {e}")
        traceback.print_exc()

    return result


def print_summary(results: List[Dict]):
    """Print a formatted summary table."""
    print()
    print("=" * 100)
    print("Transformer Bit-Exact Equivalence Summary")
    print("=" * 100)
    print(f"{'Storage':<8} {'Compute':<12} {'Forward':<10} {'Backward':<10} {'Errors'}")
    print("-" * 100)

    n_pass = 0
    n_fail = 0
    n_error = 0

    for r in results:
        fwd = r["forward"] or "N/A"
        bwd = r["backward"] or "N/A"
        errs = ", ".join(r["errors"]) if r["errors"] else "—"

        fwd_mark = "✓" if fwd == "PASS" else "✗" if fwd == "FAIL" else "⚠"
        bwd_mark = "✓" if bwd == "PASS" else "✗" if bwd == "FAIL" else "⚠"

        print(f"{r['storage']:<8} {r['compute']:<12} {fwd_mark} {fwd:<8} {bwd_mark} {bwd:<8} {errs}")

        if fwd == "PASS":
            n_pass += 1
        elif fwd == "FAIL":
            n_fail += 1
        else:
            n_error += 1

        if bwd == "PASS":
            n_pass += 1
        elif bwd == "FAIL":
            n_fail += 1
        else:
            n_error += 1

    print("-" * 100)
    total = n_pass + n_fail + n_error
    print(f"Total checks: {total}  |  PASS: {n_pass}  |  FAIL: {n_fail}  |  ERROR: {n_error}")
    print("=" * 100)

    if n_fail > 0 or n_error > 0:
        print("\nFAILURES/ERRORS DETECTED — see details above.")
        return False
    else:
        print("\nALL CHECKS PASSED — src (quantize_nonlinear=False) is bit-exact with MX.")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Transformer bit-exact equivalence verification (quantize_nonlinear=False)"
    )
    parser.add_argument("--full", action="store_true", help="Run all 12 configs")
    parser.add_argument("--fp", action="store_true", help="Include fp8 storage configs")
    args = parser.parse_args()

    torch.manual_seed(42)

    model = TransformerEncoderBlock(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)

    print("Transformer encoder block:")
    print(f"  hidden_dim={HIDDEN_DIM}, num_heads={NUM_HEADS}")
    print(f"  batch={BATCH}, seq_len={SEQ_LEN}")
    print(f"  modules: 6×Linear, 2×LayerNorm, 1×GELU, 1×Softmax")
    print(f"  inline:  2×matmul, 2×add, 1×div")

    if args.full:
        storage_keys = FULL_STORAGE
        compute_keys = FULL_COMPUTE
    else:
        storage_keys = ["none", "bf16"]
        compute_keys = list(dict.fromkeys(c for _, c in SMOKE_COMBINATIONS))  # deduplicate
        if args.fp:
            storage_keys = FULL_STORAGE
            compute_keys = list(set(compute_keys + [c for _, c in SMOKE_COMBINATIONS_FP]))

    results = []
    for sk in storage_keys:
        for ck in compute_keys:
            label = f"{sk}+{ck}"
            print(f"\n--- {label} ---")
            r = compare_config(sk, ck, model, x)
            results.append(r)

    ok = print_summary(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
