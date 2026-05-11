"""
Debug: step-by-step error propagation through a pure linear stack.

Compares quantized model vs fp32 reference, layer by layer.
Traces signal power and error power to understand why QSNR changes with depth.

Run: python scripts/test_true_error_accumulation.py
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import Session


# ═══════════════════════════════════════════════════════════════════════════════
# Model & Data
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(n_layers=8, hidden=128):
    """Pure Linear stack — no activation, no bias."""
    layers = [nn.Linear(64, hidden, bias=False)]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden, bias=False)]
    return nn.Sequential(*layers)


def calib_data(n_batches=8, bs=16):
    return [torch.randn(bs, 64) for _ in range(n_batches)]


def qsnr_db(fp32, quant):
    num = fp32.pow(2).mean()
    den = (fp32 - quant).pow(2).mean().clamp_min(1e-30)
    return (10 * torch.log10(num / den)).item()


# ═══════════════════════════════════════════════════════════════════════════════
# Hook capture
# ═══════════════════════════════════════════════════════════════════════════════

def capture_linear_outputs(model, data):
    outputs = {}
    hooks = []

    def _hook(module, inp, out, idx):
        outputs[idx] = (inp[0].detach().clone(), out.detach().clone())

    idx = 0
    for m in model:
        if hasattr(m, "weight"):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, i=idx: _hook(mod, inp, out, i)))
            idx += 1

    with torch.no_grad():
        model(data[0])

    for h in hooks:
        h.remove()
    return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# Model structure check
# ═══════════════════════════════════════════════════════════════════════════════

def print_model_structure(model, label):
    print(f"\n  {label}:")
    for i, m in enumerate(model):
        w = hasattr(m, "weight")
        print(f"    [{i}] {type(m).__name__:<30} weight={list(m.weight.shape) if w else '-'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run & trace
# ═══════════════════════════════════════════════════════════════════════════════

def run_and_trace(n_layers=8, seed=42):
    torch.manual_seed(seed)

    fp32_model = build_model(n_layers=n_layers)
    quant_model = copy.deepcopy(fp32_model)

    data = calib_data()

    cfg = QuantConfig(w_format="int8", a_format="int8", calibrator="max",
                      quantize_nonlinear=False)
    session = Session(quant_model, cfg)
    result = session.run(data, outputs=["qsnr", "mse"])

    qmodel = session._quant_session.qmodel

    print_model_structure(fp32_model, "fp32 model")
    print_model_structure(qmodel, "quant model (qmodel)")

    # Observer per-role QSNR
    input_q, _ = result.qsnr_per_role(role="input")
    weight_q, _ = result.qsnr_per_role(role="weight")

    # Capture (input, output) at each Linear layer
    fp32_captured = capture_linear_outputs(fp32_model, data)
    quant_captured = capture_linear_outputs(qmodel, data)

    print(f"\n  fp32 captured layers: {len(fp32_captured)}")
    print(f"  quant captured layers: {len(quant_captured)}")

    # ── Per-layer trace ──
    n = len(fp32_captured)
    print(f"\n  {'L':<4} {'sig_pwr':>10} {'err_pwr':>10} {'QSNR':>8} "
          f"{'obs_in':>8} {'obs_wt':>8}  "
          f"{'Δin_pwr':>10} {'Δout_pwr':>10} {'Δwt_pwr':>10}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*8} "
          f"{'-'*8} {'-'*8}  "
          f"{'-'*10} {'-'*10} {'-'*10}")

    for i in range(n):
        fp32_in, fp32_out = fp32_captured[i]
        quant_in, quant_out = quant_captured[i]

        sig_pwr = fp32_out.pow(2).mean().item()
        err_pwr = (fp32_out - quant_out).pow(2).mean().item()
        qsnr = qsnr_db(fp32_out, quant_out)

        # Input error power (how much did the input diverge from fp32 ref)
        in_err_pwr = (fp32_in - quant_in).pow(2).mean().item()

        # Output error power
        out_err_pwr = (fp32_out - quant_out).pow(2).mean().item()

        # Weight error power (for this layer)
        # Get weight from both models
        fp32_w = None
        quant_w = None
        for j, m in enumerate(fp32_model):
            if hasattr(m, "weight") and j // 2 == i:  # no ReLU, every module is Linear
                fp32_w = m.weight.detach()
                break
        # Actually, use the index directly since there are no ReLUs
        fp32_w = list(fp32_model)[i].weight.detach()
        quant_w = list(qmodel)[i].weight.detach()
        wt_err_pwr = (fp32_w - quant_w).pow(2).mean().item()

        obs_in = input_q.get(str(i), float("nan"))
        obs_wt = weight_q.get(str(i), float("nan"))

        print(f"  L{i:<3} {sig_pwr:>10.4f} {err_pwr:>10.6f} {qsnr:>8.1f} "
              f"{obs_in:>8.1f} {obs_wt:>8.1f}  "
              f"{in_err_pwr:>10.6f} {out_err_pwr:>10.6f} {wt_err_pwr:>10.6f}")

    # ── Observer key mapping ──
    print(f"\n  Observer keys → layers:")
    for k in sorted(obs.keys(), key=int):
        roles = list(obs[k].keys())
        idx = int(k)
        module = list(qmodel)[idx]
        w_shape = tuple(module.weight.shape) if hasattr(module, "weight") else "-"
        print(f"    key={k}  [{idx}] {type(module).__name__:<30} weight={w_shape}  roles={roles}")

    # ── Quant weight vs fp32 weight for each layer ──
    print(f"\n  Weight quantization error per layer:")
    for i, (fp32_m, quant_m) in enumerate(zip(fp32_model, qmodel)):
        if hasattr(fp32_m, "weight") and hasattr(quant_m, "weight"):
            w_fp32 = fp32_m.weight.detach()
            w_q = quant_m.weight.detach()
            # Check if they differ
            max_diff = (w_fp32 - w_q).abs().max().item()
            wt_qsnr = qsnr_db(w_fp32, w_q)
            print(f"    [{i}] {type(quant_m).__name__:<25} "
                  f"shape={tuple(w_fp32.shape)} max|Δ|={max_diff:.6f} wt_QSNR={wt_qsnr:.1f} dB")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Error Propagation Trace — Linear Stack, No Activation")
    print("=" * 70)
    run_and_trace(n_layers=8)
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
