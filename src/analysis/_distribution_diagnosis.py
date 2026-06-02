"""DistributionDiagnosis — causal linking of distribution features to quantisation failure modes.

Requires DistributionObserver data (``outputs=["distribution"]``) alongside
QSNRObserver for the full causal analysis.  All methods degrade gracefully
when distribution data is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from src.session._result import SessionResult

# ── Degradation taxonomy ──────────────────────────────────────────────────────
# Each rule is (label, condition_fn, suggestion).
# Rules are tried in priority order; the first match wins.

DegradationRule = Tuple[str, str, str]  # (label, description, suggestion)


def _make_rules() -> list:
    """Build the ordered degradation classification rules.

    Each entry: (label, description, suggestion).
    The lambda receives a metrics dict (from DistributionObserver) and returns
    True if the degradation applies.
    """

    def _outlier(metrics: dict) -> bool:
        return (
            metrics.get("outlier_ratio", 0.0) > 0.02
            and metrics.get("crest_factor", 0.0) > 10.0
        )

    def _high_dr(metrics: dict) -> bool:
        return (
            metrics.get("dynamic_range_bits", 0.0) > 8.0
            and metrics.get("outlier_ratio", 0.0) <= 0.02
        )

    def _heavy_tailed(metrics: dict) -> bool:
        return (
            metrics.get("excess_kurtosis", 0.0) > 3.0
            and metrics.get("outlier_ratio", 0.0) > 0.01
        )

    def _bimodal(metrics: dict) -> bool:
        return metrics.get("bimodality_coefficient", 0.0) > 0.7

    def _low_entropy(metrics: dict) -> bool:
        return metrics.get("norm_entropy", 0.5) < 0.3

    return [
        (
            "outlier_dominated",
            "Outlier-dominated — per_tensor scale is hijacked by extreme values, "
            "crushing the majority of values to near-zero effective bits.",
            "Consider per_channel granularity, hadamard transform, or boost bit-width.",
            _outlier,
        ),
        (
            "high_dynamic_range",
            "High dynamic range — the ratio between max and min non-zero values "
            "exceeds 8 bits, wasting quantisation levels on empty range.",
            "Consider per_channel granularity or hadamard transform.",
            _high_dr,
        ),
        (
            "heavy_tailed",
            "Heavy-tailed — excess kurtosis > 3 with moderate outliers; "
            "the distribution has fatter tails than Gaussian, causing "
            "occasional large quantisation errors.",
            "Consider smoothquant or pre-scale transform.",
            _heavy_tailed,
        ),
        (
            "bimodal",
            "Bimodal — the distribution appears to have two distinct modes, "
            "making uniform quantisation inefficient.",
            "Consider retaining this layer in higher precision.",
            _bimodal,
        ),
        (
            "low_entropy",
            "Low entropy — the distribution is highly concentrated; "
            "quantisation is naturally efficient here.",
            "Safe to quantise aggressively (lower bit-width).",
            _low_entropy,
        ),
        (
            "benign",
            "Benign — no problematic distribution features detected. "
            "Quantisation error is likely structural rather than data-driven.",
            "Check other causes: layer position, upstream error propagation, "
            "or model architecture.",
            lambda m: True,
        ),
    ]


_RULES = _make_rules()


def classify_distribution(metrics: dict) -> Tuple[str, str, str]:
    """Classify a single distribution metrics dict into a degradation label.

    Args:
        metrics: A dict from DistributionObserver containing keys like
            ``outlier_ratio``, ``crest_factor``, ``dynamic_range_bits``,
            ``excess_kurtosis``, ``bimodality_coefficient``, ``norm_entropy``.

    Returns:
        ``(label, description, suggestion)``.
    """
    for label, desc, suggestion, condition in _RULES:
        try:
            if condition(metrics):
                return label, desc, suggestion
        except (KeyError, TypeError):
            continue
    return "unknown", "Unable to classify distribution.", "Check raw data."


# ── DistributionDiagnosis accessor ────────────────────────────────────────────


class DistributionDiagnosis:
    """Distribution-based quantisation failure diagnosis on a single SessionResult.

    Usage::

        diag = result.characterize
        print(diag.profile("layer3.linear", role="weight"))
        print(diag.causal_analysis())
    """

    def __init__(self, result: SessionResult):
        self._result = result

    # ------------------------------------------------------------------
    # profile
    # ------------------------------------------------------------------

    def profile(self, layer: str, role: str = "weight") -> str:
        """Deep-dive diagnosis for a single layer and role.

        Reports the distribution family (best fit), key statistical features,
        and a human-readable degradation classification with actionable
        suggestion.

        Args:
            layer: Module name (as in ``named_modules()``).
            role: ``"input"`` / ``"weight"`` / ``"output"``.

        Returns:
            Formatted multi-line diagnostic text.
        """
        # Get QSNR for this layer/role
        qsnr = self._result.qsnr_by_role.get(role, {}).get(layer)
        mse = self._result.mse_by_role.get(role, {}).get(layer)

        # Get distribution metrics from observers_data
        dist_metrics = self._get_dist_metrics(layer, role)

        lines = [
            f"{layer} ({role})",
        ]
        if qsnr is not None:
            lines.append(f"  QSNR: {qsnr:.1f} dB")
        if mse is not None:
            lines.append(f"  MSE:  {mse:.2e}")

        if dist_metrics is None:
            lines.append("  Distribution: (no DistributionObserver data)")
            lines.append(
                "  Enable via: session.analyze(calib_data, "
                "outputs=['distribution'])"
            )
            return "\n".join(lines)

        # Best fit distribution
        fit_name = dist_metrics.get("best_fit")
        if fit_name:
            ks = dist_metrics.get("best_fit_ks", float("nan"))
            params = dist_metrics.get("best_fit_params")
            lines.append(f"  Distribution: {fit_name} (KS={ks:.3f})"
                         + (f" params={params}" if params else ""))

        # Key stats
        for key, label in [
            ("crest_factor", "Crest factor"),
            ("outlier_ratio", "Outlier ratio (>3σ)"),
            ("dynamic_range_bits", "Dynamic range"),
            ("excess_kurtosis", "Excess kurtosis"),
            ("norm_entropy", "Normalised entropy"),
            ("bimodality_coefficient", "Bimodality coeff"),
            ("skewness", "Skewness"),
            ("sparse_ratio", "Sparsity ratio"),
        ]:
            v = dist_metrics.get(key)
            if v is not None:
                if key in ("outlier_ratio", "sparse_ratio"):
                    lines.append(f"  {label}: {v:.1%}")
                elif key == "dynamic_range_bits":
                    lines.append(f"  {label}: {v:.1f} bits")
                else:
                    lines.append(f"  {label}: {v:.2f}")

        # Classification
        label, desc, suggestion = classify_distribution(dist_metrics)
        lines.append(f"\n  Diagnosis: {label}")
        lines.append(f"  {desc}")
        lines.append(f"\n  Suggested: {suggestion}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # classify (single layer/role)
    # ------------------------------------------------------------------

    def classify(self, layer: str, role: str = "weight") -> str:
        """Return the degradation label for a specific layer and role.

        Args:
            layer: Module name.
            role: ``"input"`` / ``"weight"`` / ``"output"``.

        Returns:
            Classification label string (e.g. ``"outlier_dominated"``).
        """
        metrics = self._get_dist_metrics(layer, role)
        if metrics is None:
            return "no_data"
        label, _, _ = classify_distribution(metrics)
        return label

    # ------------------------------------------------------------------
    # causal_analysis
    # ------------------------------------------------------------------

    def causal_analysis(self) -> str:
        """Full causal matrix: every (layer, role) ranked by QSNR with
        distribution features and classification.

        Returns:
            Formatted text table.
        """
        qsnr_by_role = self._result.qsnr_by_role
        if not qsnr_by_role:
            return "(No per-role QSNR data.)"

        # Collect rows
        rows: list = []
        roles_order = ["input", "weight", "output"]
        for role in roles_order:
            for layer, qsnr in qsnr_by_role.get(role, {}).items():
                if qsnr != qsnr:
                    continue
                metrics = self._get_dist_metrics(layer, role)
                crest = metrics.get("crest_factor") if metrics else None
                ol_ratio = metrics.get("outlier_ratio") if metrics else None
                dr_bits = metrics.get("dynamic_range_bits") if metrics else None
                classification = (
                    self.classify(layer, role) if metrics else "no_data"
                )
                rows.append((layer, role, qsnr, crest, ol_ratio, dr_bits, classification))

        if not rows:
            return "(No causal data available — enable DistributionObserver with outputs=['distribution'])"

        rows.sort(key=lambda r: r[2])

        hdr = (
            f"{'Layer':<30} {'Role':<10} {'QSNR':>8} {'Crest':>7} "
            f"{'OL%':>7} {'DR':>6}  Classification"
        )
        lines = [hdr, "-" * len(hdr)]
        for layer, role, qsnr, crest, ol, dr, cls in rows:
            c_str = f"{crest:.1f}" if crest is not None else "N/A"
            o_str = f"{ol * 100:.1f}%" if ol is not None else "N/A"
            d_str = f"{dr:.1f}" if dr is not None else "N/A"
            lines.append(
                f"{layer:<30} {role:<10} {qsnr:>8.1f} {c_str:>7} "
                f"{o_str:>7} {d_str:>6}  {cls}"
            )

        dist_warning = (
            "" if self._has_dist_data()
            else "\n(Distribution data not collected. "
                 "Enable via: session.analyze(calib_data, outputs=['distribution']))"
        )
        return "\n".join(lines) + dist_warning

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dist_metrics(self, layer: str, role: str) -> Optional[dict]:
        """Extract the distribution metrics dict for a (layer, role) pair.

        Returns the first non-empty metrics slice from observer data.
        """
        obs = self._result.observers_data
        layer_data = obs.get(layer, {})
        stages = layer_data.get(role, {})
        if not stages:
            return None
        for _stage, slices in stages.items():
            for _slice_key, metrics in slices.items():
                # DistributionObserver stores crest_factor, skewness, etc.
                if "crest_factor" in metrics:
                    return metrics
        # Check for DistributionFitObserver data
        for _stage, slices in stages.items():
            for _slice_key, metrics in slices.items():
                if "best_fit" in metrics or "best_fit_ks" in metrics:
                    return metrics
        return None

    def _has_dist_data(self) -> bool:
        """Check if any distribution data exists in observers_data."""
        for _layer, roles in self._result.observers_data.items():
            for _role, stages in roles.items():
                for _stage, slices in stages.items():
                    for _slice_key, metrics in slices.items():
                        if "crest_factor" in metrics or "best_fit" in metrics:
                            return True
        return False
