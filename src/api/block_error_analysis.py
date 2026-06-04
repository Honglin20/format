"""Block-level error analysis: per-block/per-channel QSNR decomposition.

Prerequisite: Session must have been run with PerBlockQSNRObserver attached.

Usage::

    from src.analysis.observers import PerBlockQSNRObserver
    from src.api.block_error_analysis import block_error_analysis

    session = Session(model, config, observers=[PerBlockQSNRObserver()])
    result = session.run(calib_data)
    report = block_error_analysis(result, layer="fc2", role="weight")
    print(report.worst_units[:5])
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session._result import SessionResult


@dataclass
class BlockErrorReport:
    """Per-unit (block or channel) error breakdown for one layer + role."""

    layer: str
    role: str
    unit_type: str                      # "block" | "channel" | "bank" | "tensor"
    per_unit_qsnr: Dict[int, float]     # unit_idx → qsnr_db
    per_unit_mse: Dict[int, float]      # unit_idx → mse
    worst_units: List[Tuple[int, float]]  # [(unit_idx, qsnr_db)] worst-first
    stats: Dict[str, float]             # mean, std, min, max, p10, p90
    config_name: str = ""
    outlier_unit_stats: Optional[Dict[int, dict]] = None

    def summary(self) -> str:
        n = len(self.per_unit_qsnr)
        lines = [
            f"Block Error Report: {self.layer} ({self.role})",
            f"  Units: {n} ({self.unit_type})",
            f"  QSNR: mean={self.stats.get('mean', 0):.1f} "
            f"std={self.stats.get('std', 0):.1f} "
            f"min={self.stats.get('min', 0):.1f} "
            f"max={self.stats.get('max', 0):.1f} dB",
        ]
        if self.worst_units:
            lines.append(f"  Worst {min(5, len(self.worst_units))}:")
            for idx, qsnr in self.worst_units[:5]:
                lines.append(f"    {self.unit_type} {idx}: {qsnr:.1f} dB")
        return "\n".join(lines)


def block_error_analysis(
    result: SessionResult,
    layer: str,
    role: str = "weight",
    top_k: int = 10,
) -> BlockErrorReport:
    """Extract per-block QSNR ranking from PerBlockQSNRObserver data.

    Args:
        result: SessionResult with PerBlockQSNRObserver data.
        layer: Module name.
        role: ``"input"`` / ``"weight"`` / ``"output"``.
        top_k: Number of worst units to include in worst_units.

    Returns:
        BlockErrorReport with per-unit QSNR breakdown.
    """
    obs_data = result.observers_data
    if not obs_data:
        return _empty_report(layer, role, result.name or "")

    layer_data = obs_data.get(layer, {})
    role_data = layer_data.get(role, {})
    if not role_data:
        return _empty_report(layer, role, result.name or "")

    # Find the stage with per-unit data (skip "block_agg" entries)
    per_unit_qsnr: Dict[int, float] = {}
    per_unit_mse: Dict[int, float] = {}
    unit_type = "block"

    for stage_key, slices in role_data.items():
        for slice_key, metrics in slices.items():
            if not isinstance(slice_key, tuple) or len(slice_key) < 2:
                continue
            tag = slice_key[0]
            if tag in ("block", "channel", "bank"):
                unit_type = tag
                idx = int(slice_key[1])
                qsnr_val = metrics.get("qsnr_db")
                mse_val = metrics.get("mse")
                if qsnr_val is not None and math.isfinite(qsnr_val):
                    per_unit_qsnr[idx] = qsnr_val
                if mse_val is not None and math.isfinite(mse_val):
                    per_unit_mse[idx] = mse_val

    if not per_unit_qsnr:
        return _empty_report(layer, role, result.name or "")

    # Compute stats
    values = list(per_unit_qsnr.values())
    values_sorted = sorted(values)
    n = len(values)
    mean_v = sum(values) / n
    variance = sum((v - mean_v) ** 2 for v in values) / max(n - 1, 1)
    std_v = math.sqrt(variance)

    stats = {
        "mean": mean_v,
        "std": std_v,
        "min": min(values),
        "max": max(values),
        "p10": values_sorted[max(0, int(n * 0.1))],
        "p90": values_sorted[min(n - 1, int(n * 0.9))],
        "count": n,
    }

    # Worst units sorted ascending (worst QSNR first)
    worst = sorted(per_unit_qsnr.items(), key=lambda x: x[1])[:top_k]

    return BlockErrorReport(
        layer=layer,
        role=role,
        unit_type=unit_type,
        per_unit_qsnr=per_unit_qsnr,
        per_unit_mse=per_unit_mse,
        worst_units=worst,
        stats=stats,
        config_name=result.name or "",
    )


def _empty_report(layer: str, role: str, config_name: str) -> BlockErrorReport:
    return BlockErrorReport(
        layer=layer,
        role=role,
        unit_type="unknown",
        per_unit_qsnr={},
        per_unit_mse={},
        worst_units=[],
        stats={},
        config_name=config_name,
    )
