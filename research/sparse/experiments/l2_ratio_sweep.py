"""L2: Sparse ratio sweep — QSNR vs effective bitwidth.

Answers: "How does outlier_ratio affect QSNR and effective bitwidth?"

Run: PYTHONPATH=. python research/sparse/experiments/l2_ratio_sweep.py
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

from research.sparse.configs.experiments import L2 as CFG, generate_tensor


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_qsnr(x_fp32: torch.Tensor, x_quant: torch.Tensor) -> float:
    num = x_fp32.pow(2).mean()
    den = (x_fp32 - x_quant).pow(2).mean().clamp_min(1e-30)
    return (10 * torch.log10(num / den)).item()


def compute_b_eff(shape, mode: GranularityMode, outlier_ratio: float,
                  bank_size: int = 16, b: int = 4, b_o: int = 4, s: int = 8) -> float:
    N = 1
    for d in shape:
        N *= d

    if mode == GranularityMode.PER_TENSOR:
        group_size = N
    elif mode == GranularityMode.PER_CHANNEL:
        group_size = shape[0] if len(shape) == 2 else N // shape[0]
    elif mode == GranularityMode.BANK:
        group_size = bank_size
    else:
        group_size = N

    if outlier_ratio == 0.0:
        return b + s / group_size
    else:
        b_mask = 1.0
        b_scale = 2 * s / group_size
        return (1 - outlier_ratio) * b + outlier_ratio * b_o + b_mask + b_scale


def make_scheme(fmt: FormatBase, mode: GranularityMode, outlier_ratio: float,
                bank_size: int = 16,
                scale_storage: str = "pot") -> QuantScheme:
    if mode == GranularityMode.PER_TENSOR:
        g = GranularitySpec(mode=mode, outlier_ratio=outlier_ratio)
    elif mode == GranularityMode.PER_CHANNEL:
        g = GranularitySpec(mode=mode, channel_axis=0, outlier_ratio=outlier_ratio)
    elif mode == GranularityMode.BANK:
        g = GranularitySpec(mode=mode, bank_size=bank_size, bank_axis=-1,
                            outlier_ratio=outlier_ratio)
    else:
        raise ValueError(f"Unsupported mode for L2: {mode}")
    return QuantScheme(format=fmt, granularity=g, scale_storage=scale_storage)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_l2(cfg: dict = None, output_dir: str = None) -> list:
    if cfg is None:
        cfg = CFG
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "results"

    fmt = FormatBase.from_str(cfg["format"])
    modes = cfg["granularity_modes"]
    ratios = cfg["outlier_ratios"]
    distributions = cfg["distributions"]
    shape = cfg["tensor_shape"]
    n_seeds = cfg["n_seeds"]
    base_seed = cfg["base_seed"]
    bank_size = cfg["bank_size"]
    scale_storage = cfg.get("scale_storage", "pot")

    total = len(modes) * len(ratios) * len(distributions)
    results = []
    count = 0

    for mode, ratio, dist in product(modes, ratios, distributions):
        count += 1
        mode_name = mode.value

        qsnr_vals = []
        mse_vals = []

        try:
            scheme = make_scheme(fmt, mode, ratio, bank_size=bank_size,
                                 scale_storage=scale_storage)
        except ValueError as e:
            print(f"  [{count}/{total}] {mode_name:12s} r={ratio:.2f} {dist:12s} → SKIPPED ({e})")
            results.append({
                "granularity": mode_name,
                "outlier_ratio": ratio,
                "distribution": dist,
                "qsnr_mean": None, "qsnr_std": None,
                "mse_mean": None, "mse_std": None,
                "b_eff": None,
                "status": "skipped",
            })
            continue

        for seed in range(base_seed, base_seed + n_seeds):
            x = generate_tensor(shape, dist, seed=seed)
            x_q = quantize(x, scheme)
            qsnr_vals.append(compute_qsnr(x, x_q))
            mse_vals.append((x - x_q).pow(2).mean().item())

        qsnr_t = torch.tensor(qsnr_vals)
        mse_t = torch.tensor(mse_vals)
        b_eff = compute_b_eff(shape, mode, ratio, bank_size=bank_size, b=4, b_o=4, s=8)

        results.append({
            "granularity": mode_name,
            "outlier_ratio": ratio,
            "distribution": dist,
            "qsnr_mean": qsnr_t.mean().item(),
            "qsnr_std": qsnr_t.std().item() if n_seeds > 1 else 0.0,
            "mse_mean": mse_t.mean().item(),
            "mse_std": mse_t.std().item() if n_seeds > 1 else 0.0,
            "b_eff": b_eff,
            "status": "ok",
        })
        print(f"  [{count}/{total}] {mode_name:12s} r={ratio:.2f} {dist:12s} → QSNR={qsnr_t.mean().item():.2f}±{qsnr_t.std().item():.2f}  b_eff={b_eff:.2f}")

    # Write JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "l2_ratio_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_l2()
