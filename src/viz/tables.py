"""Parameterized table generation functions.

All functions are PURE: receive data, return formatted text.
File I/O is self-contained (each function creates its own CSV).
"""
import math
import os
from collections import defaultdict
from typing import Dict

from src.viz._helpers import _compute_best_transform_per_layer


def accuracy_table(results: dict, *, title: str, output_dir: str, filename: str) -> str:
    """Generate CSV accuracy + avg QSNR/MSE table from a flat results dict.

    Args:
        results: Dict mapping config name to result dict with keys
            ``accuracy``, ``qsnr_per_layer``, ``mse_per_layer``.
        title: Table title for the text header.
        output_dir: Output root directory. CSV saved to ``<output_dir>/tables/``.
        filename: CSV filename.

    Returns:
        Formatted text representation of the table.
    """
    rows = []
    for name, data in sorted(results.items()):
        acc = data.get("accuracy", {})
        if isinstance(acc, dict) and len(acc) == 1:
            acc_val = list(acc.values())[0]
            acc_str = f"{acc_val:.4f}"
        elif isinstance(acc, dict):
            acc_str = ", ".join(f"{k}: {v:.4f}" for k, v in acc.items())
        elif isinstance(acc, (int, float)):
            acc_str = f"{acc:.4f}"
        else:
            acc_str = str(acc)
        qsnr_dict = data.get("qsnr_per_layer", {})
        mse_dict = data.get("mse_per_layer", {})
        avg_qsnr = sum(qsnr_dict.values()) / max(len(qsnr_dict), 1)
        avg_mse = sum(mse_dict.values()) / max(len(mse_dict), 1)
        rows.append((name, acc_str, avg_qsnr, avg_mse))

    max_name = max((len(r[0]) for r in rows), default=20)
    name_w = max(max_name + 2, 20)
    header_line = (
        f"{'Config':<{name_w}} {'Accuracy':<20} {'Avg QSNR (dB)':<15} {'Avg MSE':<15}"
    )
    sep_line = "-" * len(header_line)
    lines = [f"\n{'=' * len(header_line)}", title, "=" * len(header_line)]
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(
            f"{row[0]:<{name_w}} {row[1]:<20} {row[2]:<15.2f} {row[3]:<15.6f}"
        )

    csv_dir = os.path.join(output_dir, "tables")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, filename)
    with open(csv_path, "w") as f:
        f.write("Config,Accuracy,Avg_QSNR_dB,Avg_MSE\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.6f}\n")

    return "\n".join(lines)


def format_comparison_table(results: dict, *, title: str, output_dir: str, filename: str = "comparison.csv") -> str:
    """Alias for accuracy_table with a default filename.

    Args:
        results: Dict mapping config name to result dict.
        title: Table title for the text header.
        output_dir: Output root directory. CSV saved to ``<output_dir>/tables/``.
        filename: CSV filename (default ``comparison.csv``).

    Returns:
        Formatted text representation of the table.
    """
    return accuracy_table(results, title=title, output_dir=output_dir, filename=filename)


# ---------------------------------------------------------------------------
# Table 3 — FP32 vs PoT delta
# ---------------------------------------------------------------------------

def pot_delta_table(part_c: dict, output_dir: str) -> str:
    """FP32 vs PoT accuracy delta table (formerly generate_table_3)."""
    baseline_acc = 0.0
    for name, data in part_c.items():
        if name == "FP32 (baseline)":
            acc = data.get("accuracy", {})
            baseline_acc = float(acc.get("accuracy", 0.0)) if isinstance(acc, dict) else float(acc or 0.0)
            break

    rows = []
    for name, data in part_c.items():
        if name == "FP32 (baseline)":
            continue
        acc = data.get("accuracy", {})
        if isinstance(acc, dict):
            acc_val = float(acc.get("accuracy", 0.0))
            acc_str = ", ".join(f"{k}: {v:.4f}" for k, v in acc.items())
        else:
            acc_val = float(acc) if isinstance(acc, (int, float)) else 0.0
            acc_str = f"{acc_val:.4f}"
        qsnr_d = data.get("qsnr_per_layer", {})
        mse_d = data.get("mse_per_layer", {})
        rows.append((
            name, acc_str, acc_val - baseline_acc,
            sum(qsnr_d.values()) / max(len(qsnr_d), 1),
            sum(mse_d.values()) / max(len(mse_d), 1),
        ))

    lines = [f"\n{'='*85}", "Table 3: FP32 vs PoT Scaling", "=" * 85,
             f"{'Config':<20} {'Accuracy':<20} {'Delta':<12} {'Avg QSNR (dB)':<15} {'Avg MSE':<15}",
             "-" * 85]
    for r in rows:
        lines.append(f"{r[0]:<20} {r[1]:<20} {r[2]:<+12.4f} {r[3]:<15.2f} {r[4]:<15.6f}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table3_pot.csv", "w") as f:
        f.write("Config,Accuracy,Delta,Avg_QSNR_dB,Avg_MSE\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.4f},{r[4]:.6f}\n")
    return result


# ---------------------------------------------------------------------------
# Table 4 — Format x Transform accuracy matrix
# ---------------------------------------------------------------------------

def transform_matrix_table(part_d: dict, output_dir: str, *, suffix: str = "") -> str:
    """Format x Transform accuracy matrix table (formerly generate_table_4)."""
    fmt_names = sorted(part_d.keys())
    tx_variants = sorted({tx for fmt_data in part_d.values() for tx in fmt_data})

    def _acc(fmt_data, tx):
        if tx not in fmt_data:
            return float("nan")
        acc = fmt_data[tx].get("accuracy", {})
        return float(acc.get("accuracy", 0.0)) if isinstance(acc, dict) else (
            float(acc) if isinstance(acc, (int, float)) else float("nan")
        )

    lines = [f"\n{'='*80}", "Table 4: Format x Transform Accuracy Matrix", "=" * 80,
             f"{'Format':<16}" + "".join(f" {tx:<20}" for tx in tx_variants),
             "-" * (16 + 21 * len(tx_variants))]
    for fmt_name in fmt_names:
        row = f"{fmt_name:<16}"
        for tx in tx_variants:
            v = _acc(part_d[fmt_name], tx)
            row += f" {'N/A':<20}" if math.isnan(v) else f" {v:<20.4f}"
        lines.append(row)
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table4_format_x_transform{suffix}.csv", "w") as f:
        f.write("Format," + ",".join(tx_variants) + "\n")
        for fmt_name in fmt_names:
            vals = [
                f"{_acc(part_d[fmt_name], tx):.6f}" if not math.isnan(_acc(part_d[fmt_name], tx)) else "N/A"
                for tx in tx_variants
            ]
            f.write(f"{fmt_name}," + ",".join(vals) + "\n")
    return result


# ---------------------------------------------------------------------------
# Table 5 — Transform distribution
# ---------------------------------------------------------------------------

def transform_distribution_table(part_d: dict, output_dir: str) -> str:
    """Per-layer optimal transform distribution table (formerly generate_table_5)."""
    distribution: Dict[str, Dict[str, int]] = {}
    all_tx_set: set = set()

    for fmt_name, fmt_data in part_d.items():
        variant_qsnr = {
            tx: fmt_data[tx]["qsnr_per_layer"]
            for tx in ("None", "SmoothQuant", "Hadamard")
            if tx in fmt_data and "qsnr_per_layer" in fmt_data[tx]
        }
        tx_counts: Dict[str, int] = defaultdict(int)
        for best_tx in _compute_best_transform_per_layer(variant_qsnr).values():
            tx_counts[best_tx] += 1
        distribution[fmt_name] = dict(tx_counts)
        all_tx_set.update(tx_counts.keys())

    all_tx = sorted(all_tx_set)
    hdr = f"{'Format':<16}" + "".join(f" {tx:<18}" for tx in all_tx) + " Total"
    lines = [f"\n{'='*80}", "Table 5: Per-Layer Optimal Transform Distribution", "=" * 80, hdr, "-" * len(hdr)]
    for fmt_name in sorted(distribution.keys()):
        r = f"{fmt_name:<16}"
        total = 0
        for tx in all_tx:
            cnt = distribution[fmt_name].get(tx, 0)
            r += f" {cnt:<18}"
            total += cnt
        lines.append(r + f" {total}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table5_transform_distribution.csv", "w") as f:
        f.write("Format," + ",".join(all_tx) + ",Total\n")
        for fmt_name in sorted(distribution.keys()):
            vals = [str(distribution[fmt_name].get(tx, 0)) for tx in all_tx]
            vals.append(str(sum(distribution[fmt_name].values())))
            f.write(f"{fmt_name}," + ",".join(vals) + "\n")
    return result


# ---------------------------------------------------------------------------
# Table 6 — Top-10 sensitivity
# ---------------------------------------------------------------------------

def sensitivity_table(all_results: dict, output_dir: str) -> str:
    """Top-10 most sensitive layers table (formerly generate_table_6)."""
    layer_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: {"mse": [], "qsnr": []})
    for part_name, part_data in all_results.items():
        if not isinstance(part_data, dict):
            continue
        for config_data in part_data.values():
            if not isinstance(config_data, dict):
                continue
            for key in ("qsnr_per_layer", "mse_per_layer"):
                if key not in config_data:
                    continue
                metric = "qsnr" if "qsnr" in key else "mse"
                for layer, val in config_data[key].items():
                    layer_metrics[layer][metric].append(val)

    ranking = sorted(
        (
            (layer,
             max(m["mse"]) if m["mse"] else 0.0,
             min(m["qsnr"]) if m["qsnr"] else 0.0)
            for layer, m in layer_metrics.items()
        ),
        key=lambda x: x[1], reverse=True,
    )[:10]

    lines = [f"\n{'='*80}", "Table 6: Top-10 Most Sensitive Layers", "=" * 80,
             f"{'#':<4} {'Layer':<28} {'Max MSE':<18} {'Min QSNR (dB)':<15}", "-" * 80]
    for i, (layer, mse, qsnr) in enumerate(ranking, 1):
        lines.append(f"{i:<4} {layer:<28} {mse:<18.6e} {qsnr:<15.2f}")
    result = "\n".join(lines)

    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    with open(f"{output_dir}/tables/table6_sensitivity.csv", "w") as f:
        f.write("Rank,Layer,Max_MSE,Min_QSNR_dB\n")
        for i, (layer, mse, qsnr) in enumerate(ranking, 1):
            f.write(f"{i},{layer},{mse:.6e},{qsnr:.4f}\n")
    return result
