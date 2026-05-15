"""L3: Bank granularity sweet-spot analysis.

Answers: "What is the optimal bank_size for a given tensor dimension?"

Run: PYTHONPATH=. python research/sparse/experiments/l3_bank_sweetspot.py
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

from research.sparse.configs.experiments import L3 as CFG, generate_tensor


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_qsnr(x_fp32: torch.Tensor, x_quant: torch.Tensor) -> float:
    num = x_fp32.pow(2).mean()
    den = (x_fp32 - x_quant).pow(2).mean().clamp_min(1e-30)
    return (10 * torch.log10(num / den)).item()


def compute_b_eff(bank_size: int, fixed_dim: int, outlier_ratio: float,
                  b: int = 4, b_o: int = 4, s: int = 8) -> float:
    """Effective bitwidth for BANK granularity.

    group_size = bank_size * fixed_dim because one amax per bank is
    shared across all fixed_dim rows in that bank segment.
    """
    group_size = bank_size * fixed_dim
    b_mask = 1.0
    b_scale = 2 * s / group_size
    return (1 - outlier_ratio) * b + outlier_ratio * b_o + b_mask + b_scale


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_l3(cfg: dict = None, output_dir: str = None) -> list:
    if cfg is None:
        cfg = CFG
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "results"

    fmt = FormatBase.from_str(cfg["format"])
    mode = cfg["granularity_mode"]
    outlier_ratio = cfg["outlier_ratio"]
    bank_sizes = cfg["bank_sizes"]
    tensor_dims = cfg["tensor_dims"]
    fixed_dim = cfg["fixed_dim"]
    distributions = cfg["distributions"]
    n_seeds = cfg["n_seeds"]
    base_seed = cfg["base_seed"]

    total = len(bank_sizes) * len(tensor_dims) * len(distributions)
    results = []
    count = 0

    for bank_size, tensor_dim, dist in product(bank_sizes, tensor_dims, distributions):
        count += 1

        # Tensor shape: (fixed_dim, tensor_dim) — bank_axis=-1 splits tensor_dim
        shape = (fixed_dim, tensor_dim)

        # Validate divisibility
        if tensor_dim % bank_size != 0:
            results.append({
                "bank_size": bank_size,
                "tensor_dim": tensor_dim,
                "distribution": dist,
                "shape": list(shape),
                "outlier_ratio": outlier_ratio,
                "qsnr_mean": None, "qsnr_std": None,
                "mse_mean": None, "mse_std": None,
                "b_eff": None,
                "qsnr_per_b_eff": None,
                "status": "skipped",
                "note": f"tensor_dim {tensor_dim} not divisible by bank_size {bank_size}",
            })
            print(f"  [{count}/{total}] bs={bank_size:3d} dim={tensor_dim:4d} {dist:10s} → SKIPPED (indivisible)")
            continue

        qsnr_vals = []
        mse_vals = []

        g = GranularitySpec(
            mode=mode,
            bank_size=bank_size,
            bank_axis=-1,
            outlier_ratio=outlier_ratio,
        )
        scheme = QuantScheme(format=fmt, granularity=g,
                             scale_storage=cfg.get("scale_storage", "pot"))

        for seed in range(base_seed, base_seed + n_seeds):
            x = generate_tensor(shape, dist, seed=seed)
            x_q = quantize(x, scheme)
            qsnr_vals.append(compute_qsnr(x, x_q))
            mse_vals.append((x - x_q).pow(2).mean().item())

        qsnr_t = torch.tensor(qsnr_vals)
        mse_t = torch.tensor(mse_vals)
        s_bits = 8 if cfg.get("scale_storage", "pot") == "pot" else 32
        b_eff = compute_b_eff(bank_size, fixed_dim, outlier_ratio, b=4, b_o=4, s=s_bits)
        qsnr_mean = qsnr_t.mean().item()
        qsnr_per_b_eff = qsnr_mean / b_eff if b_eff > 0 else None

        results.append({
            "bank_size": bank_size,
            "tensor_dim": tensor_dim,
            "distribution": dist,
            "shape": list(shape),
            "outlier_ratio": outlier_ratio,
            "qsnr_mean": qsnr_mean,
            "qsnr_std": qsnr_t.std().item() if n_seeds > 1 else 0.0,
            "mse_mean": mse_t.mean().item(),
            "mse_std": mse_t.std().item() if n_seeds > 1 else 0.0,
            "b_eff": b_eff,
            "qsnr_per_b_eff": qsnr_per_b_eff,
            "status": "ok",
        })
        print(f"  [{count}/{total}] bs={bank_size:3d} dim={tensor_dim:4d} {dist:10s} → QSNR={qsnr_t.mean().item():.2f}  b_eff={b_eff:.2f}  Q/b={qsnr_per_b_eff:.2f}")

    # Write JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "l3_bank_sweetspot.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    # Print summary table
    _print_table(results)

    return results


def _print_table(results: list):
    print()
    print("| Bank Size | Tensor Dim | Distribution | QSNR (dB)        | b_eff | QSNR/b_eff |")
    print("|-----------|------------|--------------|------------------|-------|------------|")
    for r in results:
        if r["status"] == "skipped":
            qsnr_str = "N/A"
            b_str = "N/A"
            qb_str = "N/A"
        else:
            qsnr_str = f"{r['qsnr_mean']:.2f} ± {r['qsnr_std']:.2f}"
            b_str = f"{r['b_eff']:.2f}"
            qb_str = f"{r['qsnr_per_b_eff']:.2f}"
        print(f"| {r['bank_size']:>9d} | {r['tensor_dim']:>10d} | {r['distribution']:12s} | {qsnr_str:16s} | {b_str:5s} | {qb_str:10s} |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_l3()
