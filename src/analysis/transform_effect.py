"""Transform effect report: quantify per-config transform recovery.

Compares config pairs (with/without transform) matched by (w_bits, a_bits).

Usage::

    from src.report._study_report import StudyReport
    from src.analysis.transform_effect import TransformEffectReport

    study_report = StudyReport.from_file(output_dir)
    report = TransformEffectReport.from_study(study_report)
    print(report.summary())
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.report._study_report import StudyReport
    from src.session._result import SessionResult


class TransformEffectReport:
    """Quantify how much each transform recovers precision, per config.

    Auto-detects transform/no-transform config pairs from config names:
      - "W4A4" and "W4A4+SQ" → smoothquant pair
      - "W4A4" and "W4A4+HD" → hadamard pair
      - Any config name containing "+SQ" / "smoothquant" / "+HD" / "hadamard"
        is treated as a transform variant.
    """

    # Recognized transform suffixes in config names
    _TRANSFORM_PATTERNS = [
        (r"\+SQ$", "smoothquant"),
        (r"\+smoothquant$", "smoothquant"),
        (r"\+HD$", "hadamard"),
        (r"\+hadamard$", "hadamard"),
    ]

    def __init__(self, pairs: List[dict]):
        """Initialize with matched pairs.

        Each pair: {
            "base_config": str,
            "transform_config": str,
            "transform": str,         # "smoothquant" | "hadamard"
            "base_accuracy": float | None,
            "transform_accuracy": float | None,
            "fp32_accuracy": float | None,
            "accuracy_gain": float | None,
            "recovery_pct": float | None,
            "base_avg_qsnr": float | None,
            "transform_avg_qsnr": float | None,
            "qsnr_gain_db": float | None,
        }
        """
        self._pairs = pairs

    @classmethod
    def from_study(cls, study_report: StudyReport) -> TransformEffectReport:
        """Auto-detect transform/no-transform pairs from StudyReport."""
        configs: Dict[str, SessionResult] = {}
        for part_results in study_report._results.values():
            for r in part_results:
                name = r.name or ""
                configs[name] = r

        # Match base ↔ transform pairs
        pairs: List[dict] = []
        for name, result in configs.items():
            base_name = None
            transform = None

            for pattern, transform_name in cls._TRANSFORM_PATTERNS:
                match = re.search(pattern, name)
                if match:
                    base_name = name[:match.start()]
                    transform = transform_name
                    break

            if base_name is None or base_name not in configs:
                continue

            base_result = configs[base_name]

            # Extract metrics
            base_acc = _extract_accuracy(base_result)
            trans_acc = _extract_accuracy(result)
            fp32_acc = base_result.fp32_metrics.get("accuracy") if base_result.fp32_metrics else None
            if fp32_acc is None and result.fp32_metrics:
                fp32_acc = result.fp32_metrics.get("accuracy")

            accuracy_gain = None
            if base_acc is not None and trans_acc is not None:
                accuracy_gain = trans_acc - base_acc

            recovery_pct = None
            if fp32_acc is not None and base_acc is not None and accuracy_gain is not None:
                gap = fp32_acc - base_acc
                if abs(gap) > 1e-10:
                    recovery_pct = (accuracy_gain / gap) * 100

            # Avg QSNR
            base_qsnr = _avg_qsnr(base_result)
            trans_qsnr = _avg_qsnr(result)
            qsnr_gain = None
            if base_qsnr is not None and trans_qsnr is not None:
                qsnr_gain = trans_qsnr - base_qsnr

            pairs.append({
                "base_config": base_name,
                "transform_config": name,
                "transform": transform,
                "base_accuracy": base_acc,
                "transform_accuracy": trans_acc,
                "fp32_accuracy": fp32_acc,
                "accuracy_gain": accuracy_gain,
                "recovery_pct": recovery_pct,
                "base_avg_qsnr": base_qsnr,
                "transform_avg_qsnr": trans_qsnr,
                "qsnr_gain_db": qsnr_gain,
            })

        return cls(pairs)

    @property
    def pairs(self) -> List[dict]:
        return list(self._pairs)

    def per_config_recovery(self) -> List[dict]:
        """Accuracy recovery per config pair.

        Returns [{base_config, transform, accuracy_gain, recovery_pct, qsnr_gain_db}].
        """
        results = []
        for p in self._pairs:
            entry = {
                "base_config": p["base_config"],
                "transform": p["transform"],
                "accuracy_gain": p["accuracy_gain"],
                "recovery_pct": p["recovery_pct"],
                "qsnr_gain_db": p["qsnr_gain_db"],
            }
            results.append(entry)
        return results

    def summary(self) -> str:
        """Formatted summary table."""
        if not self._pairs:
            return "(No transform pairs detected in study results.)"

        lines = ["Transform Effect Summary"]
        lines.append(f"{'Config':<12} {'Transform':<14} {'Base Acc':>10} {'Trans Acc':>10} "
                     f"{'Gain':>8} {'Recovery':>10} {'QSNR Δ':>8}")
        lines.append("-" * 76)

        for p in self._pairs:
            base_acc = f"{p['base_accuracy']:.4f}" if p["base_accuracy"] is not None else "N/A"
            trans_acc = f"{p['transform_accuracy']:.4f}" if p["transform_accuracy"] is not None else "N/A"
            gain = f"{p['accuracy_gain']:+.4f}" if p["accuracy_gain"] is not None else "N/A"
            recovery = f"{p['recovery_pct']:.1f}%" if p["recovery_pct"] is not None else "N/A"
            qsnr_d = f"{p['qsnr_gain_db']:+.1f}" if p["qsnr_gain_db"] is not None else "N/A"

            lines.append(
                f"{p['base_config']:<12} {p['transform']:<14} {base_acc:>10} {trans_acc:>10} "
                f"{gain:>8} {recovery:>10} {qsnr_d:>8}"
            )

        return "\n".join(lines)


def _extract_accuracy(result: SessionResult) -> Optional[float]:
    """Extract accuracy from SessionResult metrics."""
    if result.quant_metrics:
        for key in ("accuracy", "acc", "eval_accuracy"):
            if key in result.quant_metrics:
                return result.quant_metrics[key]
    return None


def _avg_qsnr(result: SessionResult) -> Optional[float]:
    """Compute average per-layer QSNR."""
    if not result.qsnr_per_layer:
        return None
    values = [v for v in result.qsnr_per_layer.values()
              if v is not None and v == v and v != float("inf")]
    return sum(values) / len(values) if values else None
