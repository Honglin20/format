"""Parameterized table generation functions.

All functions are PURE: receive data, return formatted text.
File I/O is self-contained (each function creates its own CSV).
"""
import os
from collections import defaultdict
from typing import Dict


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

    lines = [f"\n{'=' * 70}", title, "=" * 70]
    lines.append(
        f"{'Config':<20} {'Accuracy':<20} {'Avg QSNR (dB)':<15} {'Avg MSE':<15}"
    )
    lines.append("-" * 70)
    for row in rows:
        lines.append(
            f"{row[0]:<20} {row[1]:<20} {row[2]:<15.2f} {row[3]:<15.6f}"
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
