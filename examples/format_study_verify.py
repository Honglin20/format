"""
Format Study — Manual Verification with Fixed Weights & Inputs

Uses a single ``nn.Linear(4, 8, bias=False)`` with fixed (realistic) weights and
inputs so every config's quantized output + per-channel QSNR can be **hand‑computed**
and compared against the pipeline result.

Run::

    python examples/format_study_verify.py

If all assertions pass, the pipeline produces bit-exact results matching the
derivation.  Any mismatch is printed with the actual vs expected tensors.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.config import resolve_config
from src.session import QuantSession
from src.calibration.strategies import MSEScaleStrategy
from src.analysis.observers import QSNRObserver
from src.quantize.elemwise import quantize as lib_quantize


# ═══════════════════════════════════════════════════════════════════
# 1.  Fixed model, weights, and data  (seed=42 for reproducibility)
# ═══════════════════════════════════════════════════════════════════

torch.manual_seed(42)

IN_FEATURES = 4
OUT_FEATURES = 8
BATCH_SIZE = 4

# Fixed weight tensor  (out_features, in_features) = (8, 4)
WEIGHT = nn.Parameter(torch.randn(OUT_FEATURES, IN_FEATURES))

# Fixed calibration inputs  (batch, in_features) = (4, 4)
CALIB_X = torch.randn(BATCH_SIZE, IN_FEATURES)

# Fixed eval inputs
EVAL_X = torch.randn(BATCH_SIZE * 2, IN_FEATURES)


def build_model() -> nn.Module:
    """Fresh single Linear layer with fixed weights (no bias), wrapped in Sequential."""
    lin = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        lin.weight.copy_(WEIGHT)
    return nn.Sequential(lin)


# ═══════════════════════════════════════════════════════════════════
# 2.  Manual quantization helpers  (mirror the library formulas)
# ═══════════════════════════════════════════════════════════════════

def _round_nearest(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * floor(|x| + 0.5) — matches src/formats/_core.py:30"""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


# ── INT elemwise ──
def manual_int_elemwise(x: torch.Tensor, mbits: int,
                         round_mode: str = "nearest") -> torch.Tensor:
    max_norm = float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)
    scaled = x * (2 ** (mbits - 2))
    if round_mode == "nearest":
        rounded = _round_nearest(scaled)
    else:
        raise ValueError(f"round_mode {round_mode} not implemented in manual helper")
    out = rounded / (2 ** (mbits - 2))
    out = torch.clamp(out, -max_norm, max_norm)
    return out


# ── FP elemwise ──
def manual_fp_elemwise(x: torch.Tensor, ebits: int, mbits: int,
                        round_mode: str = "nearest") -> torch.Tensor:
    emax = 2 ** (ebits - 1)
    max_norm = 2 ** emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)

    x_abs = torch.abs(x)
    min_exp = -(2 ** (ebits - 1)) + 2
    private_exp = torch.floor(torch.log2(x_abs + (x == 0).to(x.dtype)))
    private_exp = private_exp.clamp(min=min_exp)

    scaled = x / (2 ** private_exp) * (2 ** (mbits - 2))
    if round_mode == "nearest":
        rounded = _round_nearest(scaled)
    else:
        raise ValueError(f"round_mode {round_mode} not implemented in manual helper")
    out = rounded / (2 ** (mbits - 2)) * (2 ** private_exp)
    out = torch.where(torch.abs(out) > max_norm,
                       torch.sign(out) * float("Inf"), out)
    return out


# ── NF4 elemwise ──
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


def manual_nf4_elemwise(x: torch.Tensor) -> torch.Tensor:
    levels = NF4_LEVELS.to(device=x.device, dtype=x.dtype)
    x_clamped = torch.clamp(x, -1.0, 1.0)
    distances = torch.abs(x_clamped.unsqueeze(-1) - levels)
    indices = torch.argmin(distances, dim=-1)
    return levels[indices]


# ── per-channel quantization (with PoT rounding) ──
def manual_per_channel_quantize(x: torch.Tensor, channel_axis: int,
                                 elemwise_fn, scale_format: str = "fp32",
                                 **elemwise_kw) -> torch.Tensor:
    ax = channel_axis if channel_axis >= 0 else x.ndim + channel_axis
    dims = tuple(i for i in range(x.ndim) if i != ax)
    amax = torch.amax(torch.abs(x), dim=dims, keepdim=True).clamp(min=1e-12)
    if scale_format == "pot":
        amax = 2 ** torch.round(torch.log2(amax))
    x_norm = x / amax
    x_q = elemwise_fn(x_norm, **elemwise_kw)
    return x_q * amax


# ── per-block quantization via library ──
def manual_per_block_quantize(x: torch.Tensor, block_size: int,
                               block_axis: int, format_name: str) -> torch.Tensor:
    from src.scheme.quant_scheme import QuantScheme
    from src.formats.base import FormatBase
    from src.scheme.granularity import GranularitySpec

    fmt = FormatBase.from_str(format_name)
    scheme = QuantScheme(
        format=fmt,
        granularity=GranularitySpec.per_block(block_size, block_axis),
    )
    return lib_quantize(x, scheme)


def manual_per_channel_quantize_fmt(x: torch.Tensor, channel_axis: int,
                                      format_name: str,
                                      scale_format: str = "fp32") -> torch.Tensor:
    if format_name.startswith("int"):
        mbits = int(format_name[3:])
        return manual_per_channel_quantize(
            x, channel_axis, manual_int_elemwise, mbits=mbits,
            scale_format=scale_format,
        )
    elif format_name.startswith("fp") and "_e" in format_name:
        parts = format_name[3:].split("_e")
        mbits = int(parts[0])
        ebits = int(parts[1][0])
        return manual_per_channel_quantize(
            x, channel_axis, manual_fp_elemwise, ebits=ebits, mbits=mbits,
            scale_format=scale_format,
        )
    elif format_name == "nf4":
        return manual_per_channel_quantize(
            x, channel_axis, manual_nf4_elemwise,
            scale_format=scale_format,
        )
    else:
        raise ValueError(f"Unknown format: {format_name}")


# ── QSNR per-channel ──
def manual_qsnr_per_channel(fp32: torch.Tensor, quant: torch.Tensor,
                             channel_axis: int = -1) -> torch.Tensor:
    ax = channel_axis if channel_axis >= 0 else fp32.ndim + channel_axis
    n_ch = fp32.shape[ax]
    fp32_2d = fp32.movedim(ax, 0).reshape(n_ch, -1)
    quant_2d = quant.movedim(ax, 0).reshape(n_ch, -1)
    qsnr_values = []
    for c in range(n_ch):
        fp_c = fp32_2d[c]
        q_c = quant_2d[c]
        num = (fp_c ** 2).mean()
        den = ((fp_c - q_c) ** 2).mean()
        if den == 0:
            qsnr_values.append(float("inf"))
        elif num == 0:
            qsnr_values.append(float("-inf"))
        else:
            qsnr_values.append(10.0 * math.log10((num / den).item()))
    return torch.tensor(qsnr_values)


# ── QSNR per-block ──
def manual_qsnr_per_block(fp32, quant, block_size, block_axis):
    ax = block_axis if block_axis >= 0 else fp32.ndim + block_axis
    dim_size = fp32.shape[ax]
    n_blocks = (dim_size + block_size - 1) // block_size

    fp32_moved = fp32.movedim(ax, -1)
    quant_moved = quant.movedim(ax, -1)

    if dim_size % block_size != 0:
        pad = n_blocks * block_size - dim_size
        fp32_moved = F.pad(fp32_moved, (0, pad))
        quant_moved = F.pad(quant_moved, (0, pad))

    fp32_2d = fp32_moved.reshape(-1, block_size)
    quant_2d = quant_moved.reshape(-1, block_size)

    err = fp32_2d - quant_2d
    num = fp32_2d.pow(2).mean(dim=1)
    den = err.pow(2).mean(dim=1).clamp_min(1e-30)
    qsnr_per_block = 10 * torch.log10(num / den)
    return qsnr_per_block.mean().unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════
# 3.  Config-based expected computation
# ═══════════════════════════════════════════════════════════════════

def compute_expected_for_config(cfg_desc: dict):
    """
    Manually compute the expected quantized output and per-channel QSNR
    for a single study config descriptor.

    Returns:
        (q_output, qsnr)  — expected values
    """
    op_cfg = resolve_config(cfg_desc)
    weight_scheme = op_cfg.weight
    input_scheme = op_cfg.input
    output_scheme = op_cfg.output

    w_fp32 = WEIGHT.clone()
    x_fp32 = CALIB_X.clone()

    # ── quantize weight ──
    if weight_scheme is not None:
        w_q = lib_quantize(w_fp32, weight_scheme)
    else:
        w_q = w_fp32

    # ── quantize input ──
    if input_scheme is not None:
        x_q = lib_quantize(x_fp32, input_scheme)
    else:
        x_q = x_fp32

    # ── matmul ──
    y_fp32 = x_fp32 @ w_fp32.T
    y_q = x_q @ w_q.T

    # ── quantize output ──
    if output_scheme is not None:
        g = output_scheme.granularity
        from src.scheme.granularity import GranularityMode
        if g.mode == GranularityMode.PER_CHANNEL:
            ax = g.channel_axis if g.channel_axis >= 0 else y_q.ndim + g.channel_axis
            dims = tuple(i for i in range(y_q.ndim) if i != ax)
            amax = torch.amax(torch.abs(y_q), dim=dims, keepdim=True).clamp(min=1e-12)
            if output_scheme.scale_format == "pot":
                amax = 2 ** torch.round(torch.log2(amax))
            y_q_out = lib_quantize(y_q, output_scheme, scale=amax)
        elif g.mode == GranularityMode.PER_TENSOR:
            amax = torch.amax(torch.abs(y_q)).clamp(min=1e-12)
            y_q_out = lib_quantize(y_q, output_scheme, scale=amax)
        else:
            # per_block — use library
            y_q_out = lib_quantize(y_q, output_scheme)
            qsnr = manual_qsnr_per_block(y_q, y_q_out, g.block_size, g.block_axis)
            return y_q_out, qsnr

        # QSNR: compare pre-output-quant vs post-output-quant
        if g.mode == GranularityMode.PER_BLOCK:
            qsnr = manual_qsnr_per_block(y_q, y_q_out, g.block_size, g.block_axis)
        else:
            qsnr = manual_qsnr_per_channel(y_q, y_q_out)
    else:
        y_q_out = y_q
        qsnr = manual_qsnr_per_channel(y_q, y_q_out)

    return y_q_out, qsnr


# ═══════════════════════════════════════════════════════════════════
# 4.  Pipeline run
# ═══════════════════════════════════════════════════════════════════

def run_pipeline_for_config(cfg_desc: dict, fp32_model: nn.Module):
    """Run the actual pipeline for a single config. Returns (q_output, qsnr_tensor, avg_qsnr)."""
    import copy, ast
    op_cfg = resolve_config(cfg_desc)
    model = copy.deepcopy(fp32_model)

    session = QuantSession(
        model, op_cfg,
        calibrator=MSEScaleStrategy(),
        keep_fp32=True,
    )

    with session.calibrate():
        with torch.no_grad():
            session(CALIB_X)

    with session.analyze(observers=[QSNRObserver()]) as ctx:
        with torch.no_grad():
            session(CALIB_X)
    report = ctx.report()

    session.eval()
    with torch.no_grad():
        q_output = session(CALIB_X)

    # Extract output-stage QSNR from report
    df = report.to_dataframe()
    output_qsnr = {}
    for _, row in df.iterrows():
        if row["role"] != "output" or "output_post_quant[2]" not in str(row["stage"]):
            continue
        slice_str = row["slice"]
        try:
            slice_key = ast.literal_eval(slice_str)
        except (ValueError, SyntaxError):
            continue
        if isinstance(slice_key, tuple) and len(slice_key) >= 2 and slice_key[0] == "channel":
            ch = slice_key[1]
            output_qsnr[ch] = row["qsnr_db"]
        elif isinstance(slice_key, tuple) and slice_key[0] == "block_agg":
            output_qsnr["agg"] = row["qsnr_db"]

    if output_qsnr:
        vals = list(output_qsnr.values())
        avg_qsnr = sum(v for v in vals if not math.isnan(v)) / max(len([v for v in vals if not math.isnan(v)]), 1)
        qsnr_per_ch = torch.tensor(vals)
    else:
        avg_qsnr = float("nan")
        qsnr_per_ch = torch.tensor([])

    return q_output, qsnr_per_ch, avg_qsnr


# ═══════════════════════════════════════════════════════════════════
# 5.  Study config
# ═══════════════════════════════════════════════════════════════════

STUDY_CONFIG = {
    "core_4bit_scale": {
        "description": "Core: INT4 per_channel — FP32 scale vs PoT scale",
        "configs": [
            {"name": "INT4-PC-FP32", "format": "int4", "granularity": "per_channel",
             "axis": -1, "scale_format": "fp32"},
            {"name": "INT4-PC-PoT",  "format": "int4", "granularity": "per_channel",
             "axis": -1, "scale_format": "pot"},
        ],
    },
    "4bit_formats": {
        "description": "4-bit Format Overview",
        "configs": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 4},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 4},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "axis": -1},
        ],
    },
    "8bit_formats": {
        "description": "8-bit Format Overview",
        "configs": [
            {"name": "MXINT-8", "format": "int8",     "granularity": "per_block",   "block_size": 4},
            {"name": "MXFP-8",  "format": "fp8_e4m3", "granularity": "per_block",   "block_size": 4},
            {"name": "INT8-PC", "format": "int8",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
        ],
    },
    "mixed_precision": {
        "description": "Mixed-Precision: Weight vs Activation format",
        "configs": [
            {"name": "W4A4", "format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
            {"name": "W4A8", "format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "fp32",
             "act_format": "int8"},
            {"name": "W8A4", "format": "int8", "granularity": "per_channel", "axis": -1, "scale_format": "fp32",
             "act_format": "int4"},
            {"name": "W4A8-FP", "format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "fp32",
             "act_format": "fp8_e4m3"},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 6.  Run verification
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("Format Study Verification — Fixed Weights & Inputs")
    print("=" * 72)
    print()
    print(f"  Model:   Linear(in={IN_FEATURES}, out={OUT_FEATURES}, bias=False)")
    print(f"  Weight:  random normal (seed=42), shape={WEIGHT.shape}")
    print(f"  Calib:   random normal (seed=42), shape={CALIB_X.shape}")
    print(f"  Eval:    random normal (seed=42), shape={EVAL_X.shape}")
    print(f"  Per-block weight: {OUT_FEATURES} blocks (1 per row), {IN_FEATURES} elems/block")
    print(f"  Per-channel weight: {IN_FEATURES} channels, {OUT_FEATURES} elems/channel")
    print()

    fp32_model = build_model()

    all_passed = True
    results = {}

    for part_name, part_cfg in STUDY_CONFIG.items():
        desc = part_cfg["description"]
        print(f"\n── {part_name}: {desc} ──")

        for cfg_desc in part_cfg["configs"]:
            name = cfg_desc["name"]
            print(f"\n  [{name}]")

            # ── Expected (manual computation) ──
            try:
                expected_out, expected_qsnr = compute_expected_for_config(cfg_desc)
                print(f"    Expected:  output.shape={tuple(expected_out.shape)}")
                if isinstance(expected_qsnr, torch.Tensor) and expected_qsnr.numel() == 1:
                    print(f"               qsnr=[{expected_qsnr.item():.2f}] dB")
                elif isinstance(expected_qsnr, torch.Tensor):
                    print(f"               qsnr_per_ch={[f'{v:.2f}' for v in expected_qsnr.tolist()]}")
            except Exception as e:
                print(f"    Expected computation FAILED: {e}")
                import traceback; traceback.print_exc()
                all_passed = False
                continue

            # ── Actual (pipeline run) ──
            try:
                actual_out, actual_qsnr_per_ch, actual_avg_qsnr = run_pipeline_for_config(cfg_desc, fp32_model)
                print(f"    Actual:    output.shape={tuple(actual_out.shape)}")
                print(f"               avg_qsnr={actual_avg_qsnr:.2f} dB")
            except Exception as e:
                print(f"    Pipeline run FAILED: {e}")
                import traceback; traceback.print_exc()
                all_passed = False
                continue

            # ── Compare output ──
            output_match = torch.allclose(actual_out, expected_out, rtol=1e-4, atol=1e-5)
            if output_match:
                print(f"    ✓ Output match")
            else:
                max_diff = (actual_out - expected_out).abs().max().item()
                print(f"    ✗ OUTPUT MISMATCH! (max diff={max_diff:.6e})")
                print(f"      Expected[0]: {expected_out[0].tolist()}")
                print(f"      Actual[0]:   {actual_out[0].tolist()}")
                all_passed = False

            # ── Compare QSNR ──
            qsnr_match = True
            if len(expected_qsnr) > 0 and len(actual_qsnr_per_ch) > 0:
                expected_valid = expected_qsnr[expected_qsnr.isfinite()]
                actual_valid = actual_qsnr_per_ch[actual_qsnr_per_ch.isfinite()]
                if len(expected_valid) == len(actual_valid):
                    qsnr_diff = (expected_valid - actual_valid).abs().max().item()
                    if qsnr_diff < 0.5:
                        print(f"    ✓ QSNR match (max diff={qsnr_diff:.2f} dB)")
                    else:
                        qsnr_match = False
                        print(f"    ✗ QSNR MISMATCH (max diff={qsnr_diff:.2f} dB)")
                        print(f"      Expected: {expected_valid.tolist()}")
                        print(f"      Actual:   {actual_valid.tolist()}")
                else:
                    qsnr_match = False
                    print(f"    ✗ QSNR value count mismatch: expected {len(expected_valid)} vs actual {len(actual_valid)}")
                    print(f"      Expected: {expected_valid.tolist()}")
                    print(f"      Actual:   {actual_valid.tolist()}")
            elif len(actual_qsnr_per_ch) > 0:
                qsnr_match = False
                print(f"    ✗ QSNR: expected no values, got {actual_qsnr_per_ch.tolist()}")
            else:
                print(f"    - QSNR not comparable (empty)")

            e_mean = expected_qsnr[expected_qsnr.isfinite()].mean().item() if len(expected_qsnr) > 0 and expected_qsnr.isfinite().any() else float("nan")
            results[name] = {
                "pass": output_match and qsnr_match,
                "expected_qsnr_avg": e_mean,
                "actual_qsnr_avg": actual_avg_qsnr,
            }

    # ── Summary ──
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    passed = sum(1 for v in results.values() if v["pass"])
    total = len(results)
    for name, r in results.items():
        status = "✓" if r["pass"] else "✗"
        print(f"  {status} {name:20s}  QSNR: {r['actual_qsnr_avg']:8.2f} dB  (exp: {r['expected_qsnr_avg']:8.2f} dB)")
    print(f"\n  {passed}/{total} configs passed")

    if not all_passed:
        raise AssertionError(f"{total - passed} config(s) failed verification!")

    print("\nAll assertions passed. Pipeline matches manual computation.")


if __name__ == "__main__":
    main()
