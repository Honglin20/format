"""L1: Sparse vs MXINT QSNR baseline comparison.

Answers: "Does sparse QSNR exceed MXINT across granularity modes?"

Run: PYTHONPATH=. python research/sparse/experiments/l1_baseline.py
"""
import json
import sys
from itertools import product
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.formats.base import FormatBase
from src.quantize.elemwise import quantize
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.scheme.quant_scheme import QuantScheme

from research.sparse.configs.experiments import L1 as CFG, generate_tensor


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_qsnr(x_fp32: torch.Tensor, x_quant: torch.Tensor) -> float:
    num = x_fp32.pow(2).mean()
    den = (x_fp32 - x_quant).pow(2).mean().clamp_min(1e-30)
    return (10 * torch.log10(num / den)).item()


def compute_b_eff(shape, granularity: GranularitySpec, outlier_ratio: float,
                  b: int = 4, b_o: int = 4, s: int = 8) -> float:
    """Effective bitwidth including mask + scale overhead."""
    N = 1
    for d in shape:
        N *= d

    if granularity.mode == GranularityMode.PER_TENSOR:
        group_size = N
    elif granularity.mode == GranularityMode.PER_CHANNEL:
        axis = granularity.channel_axis
        C = shape[axis]
        group_size = N // C
    elif granularity.mode == GranularityMode.PER_BLOCK:
        group_size = granularity.block_size
    elif granularity.mode == GranularityMode.BANK:
        group_size = granularity.bank_size
    else:
        group_size = N

    if outlier_ratio == 0.0:
        return b + s / group_size
    else:
        b_mask = 1.0
        b_scale = 2 * s / group_size
        return (1 - outlier_ratio) * b + outlier_ratio * b_o + b_mask + b_scale


# ---------------------------------------------------------------------------
# Scheme construction
# ---------------------------------------------------------------------------

def make_scheme(fmt: FormatBase, mode: GranularityMode, outlier_ratio: float,
                block_size: int = 32, bank_size: int = 16,
                scale_storage: str = "pot") -> QuantScheme:
    if mode == GranularityMode.PER_TENSOR:
        g = GranularitySpec(mode=mode, outlier_ratio=outlier_ratio)
    elif mode == GranularityMode.PER_CHANNEL:
        g = GranularitySpec(mode=mode, channel_axis=0, outlier_ratio=outlier_ratio)
    elif mode == GranularityMode.PER_BLOCK:
        # MX requires block_size > 0. When sparse with block_size=0 the
        # GranularitySpec constructor raises ValueError; always pass block_size.
        g = GranularitySpec(mode=mode, block_size=block_size, outlier_ratio=outlier_ratio)
    elif mode == GranularityMode.BANK:
        g = GranularitySpec(mode=mode, bank_size=bank_size, bank_axis=-1,
                            outlier_ratio=outlier_ratio)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return QuantScheme(format=fmt, granularity=g, scale_storage=scale_storage)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_l1(cfg: dict = None, output_dir: str = None) -> list:
    if cfg is None:
        cfg = CFG
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "results"

    fmt = FormatBase.from_str(cfg["format"])
    modes = cfg["granularity_modes"]
    sparse_modes = cfg["sparse_modes"]
    distributions = cfg["distributions"]
    shapes = cfg["tensor_shapes"]
    n_seeds = cfg["n_seeds"]
    base_seed = cfg["base_seed"]

    total = len(modes) * len(sparse_modes) * len(distributions) * len(shapes)
    results = []
    count = 0

    for mode, sm, dist, shape in product(modes, sparse_modes, distributions, shapes):
        count += 1
        outlier_ratio = sm["outlier_ratio"]
        sparse_label = sm["label"]
        mode_name = mode.value

        # PER_BLOCK static sparse (pre-computed mask) is not implemented.
        # Dynamic sparse (outlier_ratio > 0, mask=None) works via
        # _quantize_outlier_bank — only the static path raises NotImplementedError.
        if mode == GranularityMode.PER_BLOCK and outlier_ratio > 0.0:
            # Dynamic sparse is fine; proceed normally.
            pass

        qsnr_vals = []
        mse_vals = []

        for seed in range(base_seed, base_seed + n_seeds):
            x = generate_tensor(shape, dist, seed=seed)
            scheme = make_scheme(fmt, mode, outlier_ratio,
                                 block_size=cfg.get("mxint_block_size", 32),
                                 bank_size=cfg.get("bank_size", 16),
                                 scale_storage=cfg.get("scale_storage", "pot"))

            try:
                x_q = quantize(x, scheme)
            except NotImplementedError:
                qsnr_vals.append(float("nan"))
                mse_vals.append(float("nan"))
                continue

            qsnr_vals.append(compute_qsnr(x, x_q))
            mse_vals.append((x - x_q).pow(2).mean().item())

        qsnr_t = torch.tensor([v for v in qsnr_vals if not torch.isnan(torch.tensor(v))])
        mse_t = torch.tensor([v for v in mse_vals if not torch.isnan(torch.tensor(v))])

        if qsnr_t.numel() > 0:
            qsnr_mean = qsnr_t.mean().item()
            qsnr_std = qsnr_t.std().item() if qsnr_t.numel() > 1 else 0.0
            mse_mean = mse_t.mean().item()
            mse_std = mse_t.std().item() if mse_t.numel() > 1 else 0.0
        else:
            qsnr_mean = qsnr_std = mse_mean = mse_std = None

        b_eff = compute_b_eff(shape, scheme.granularity, outlier_ratio, b=4, b_o=4, s=8)

        results.append({
            "granularity": mode_name,
            "sparse": sparse_label,
            "distribution": dist,
            "shape": list(shape),
            "outlier_ratio": outlier_ratio,
            "qsnr_mean": qsnr_mean,
            "qsnr_std": qsnr_std,
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "b_eff": b_eff,
            "status": "ok",
        })
        status = f"QSNR={qsnr_mean:.2f}±{qsnr_std:.2f}" if qsnr_mean is not None else "FAILED"
        print(f"  [{count}/{total}] {mode_name:12s} {sparse_label:6s} {dist:16s} {str(shape):16s} → {status}")

    # Write JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "l1_baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    # Print summary table
    _print_table(results)

    return results


def _print_table(results: list):
    """Print a markdown-formatted summary table."""
    print()
    print("| Granularity     | Mode   | Distribution    | Shape        | QSNR (dB)        | b_eff |")
    print("|-----------------|--------|-----------------|--------------|------------------|-------|")
    for r in results:
        if r["status"] == "skipped":
            qsnr_str = "N/A"
        elif r["qsnr_mean"] is not None:
            qsnr_str = f"{r['qsnr_mean']:.2f} ± {r['qsnr_std']:.2f}"
        else:
            qsnr_str = "FAILED"
        b_eff_str = f"{r['b_eff']:.2f}" if r["b_eff"] is not None else "N/A"
        print(f"| {r['granularity']:15s} | {r['sparse']:6s} | {r['distribution']:15s} | {str(r['shape']):12s} | {qsnr_str:16s} | {b_eff_str:5s} |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_l1()
