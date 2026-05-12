"""InterventionAccessor — apply and compare intervention plans on SessionResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session._result import SessionResult
    from src.analysis._intervention import InterventionPlan


@dataclass
class InterventionComparison:
    """Side-by-side comparison of baseline vs intervention SessionResults.

    Attributes:
        baseline: The original SessionResult (before intervention).
        intervention: The SessionResult after applying overrides.
        plan: The InterventionPlan that was applied.
    """

    baseline: "SessionResult"
    intervention: "SessionResult"
    plan: "InterventionPlan"

    def summary(self) -> str:
        """Human-readable comparison table."""
        lines = [
            "Intervention Comparison",
            "=" * 80,
            f"  Plan: {self.plan.metadata.get('description', '(unnamed)')}",
            f"  Layers modified: {len(self.plan.overrides)}",
            "",
        ]

        # Accuracy comparison (quant vs quant)
        if (self.baseline.quant_metrics and self.intervention.quant_metrics
                and self.baseline.fp32_metrics):
            lines.append(f"  {'Metric':<12} {'FP32':<12} {'Baseline':<12} {'Intervention':<12} {'Change':<12}")
            lines.append(f"  {'-' * 60}")
            for k in self.baseline.fp32_metrics:
                fp32_v = self.baseline.fp32_metrics[k]
                b_val = self.baseline.quant_metrics.get(k)
                i_val = self.intervention.quant_metrics.get(k)
                if b_val is None or i_val is None:
                    continue
                delta = i_val - b_val
                lines.append(f"  {k:<12} {fp32_v:<12.4f} {b_val:<12.4f} {i_val:<12.4f} {delta:<+12.4f}")

        # QSNR comparison (average across all layers)
        b_qsnr = self.baseline.qsnr_per_layer
        i_qsnr = self.intervention.qsnr_per_layer
        if b_qsnr and i_qsnr:
            b_finite = [v for v in b_qsnr.values() if v == v and v != float('inf') and v != float('-inf')]
            i_finite = [v for v in i_qsnr.values() if v == v and v != float('inf') and v != float('-inf')]
            if b_finite and i_finite:
                b_avg = sum(b_finite) / len(b_finite)
                i_avg = sum(i_finite) / len(i_finite)
                lines.append(f"\n  Avg QSNR: baseline={b_avg:.1f} dB → intervention={i_avg:.1f} dB (Δ={i_avg - b_avg:+.1f} dB)")

        # Per-override QSNR delta
        lines.append(f"\n  {'Layer':<30} {'Role':<10} {'QSNR Before':>12} {'QSNR After':>12} {'Δ':>10}")
        lines.append(f"  {'-' * 78}")
        changes = self.plan.metadata.get("changes", {})
        for layer in sorted(self.plan.overrides):
            b_q = b_qsnr.get(layer, float("nan"))
            i_q = i_qsnr.get(layer, float("nan"))
            b_str = f"{b_q:.1f}" if b_q == b_q else "N/A"
            i_str = f"{i_q:.1f}" if i_q == i_q else "N/A"
            delta_str = f"{i_q - b_q:+.1f}" if (b_q == b_q and i_q == i_q) else "N/A"
            role_info = changes.get(layer, {}).get("what", "?")
            lines.append(f"  {layer:<30} {role_info:<10} {b_str:>12} {i_str:>12} {delta_str:>10}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "plan": self.plan.to_dict(),
            "baseline_qsnr": self.baseline.qsnr_per_layer,
            "intervention_qsnr": self.intervention.qsnr_per_layer,
        }


class InterventionAccessor:
    """Apply and compare intervention plans on a SessionResult.

    Usage::

        plan = result.plan.top_k_boost(k=5)
        comparison = result.intervention.compare(model, calib_data, plan)
        print(comparison.summary())
    """

    def __init__(self, result: "SessionResult"):
        self._result = result

    def compare(
        self,
        model,
        calib_data,
        plan: "InterventionPlan",
        *,
        eval_data=None,
        eval_fn: Optional[Callable] = None,
    ) -> InterventionComparison:
        """Run a new Session with *plan.overrides* and compare against baseline.

        Args:
            model: Original (unquantized) fp32 model.
            calib_data: Calibration data for the new Session.
            plan: InterventionPlan with per-layer overrides.
            eval_data: Optional evaluation data.
            eval_fn: Optional ``(model, data) -> Dict[str, float]``.

        Returns:
            ``InterventionComparison`` with baseline and intervention results.
        """
        from src.session._session import Session

        if not plan.overrides:
            return InterventionComparison(
                baseline=self._result,
                intervention=self._result,
                plan=plan,
            )

        session = Session(
            model,
            self._result.config,
            keep_fp32=True,
            overrides=plan.overrides,
        )
        intervention_result = session.run(
            calib_data,
            eval_data=eval_data,
            eval_fn=eval_fn,
        )

        return InterventionComparison(
            baseline=self._result,
            intervention=intervention_result,
            plan=plan,
        )
