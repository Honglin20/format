"""Diagnostic Pipeline API — composable three-stage quantization analysis.

Stages:
  coarse_pass()  — Multi-config accuracy gaps + layer ranking + distribution overview
  deep_dive()    — Single-config deep analysis: distribution diagnosis + block error
                   + error provenance + layer sensitivity
  prescribe()    — Intervention planning + recovery strategies

Each function is independently callable and returns a dataclass with
``.summary()`` and ``.to_dict()`` for downstream consumption.

Observer requirements (attach during ``Session.run()``):
  coarse_pass:    QSNRObserver
  deep_dive:      QSNRObserver + DistributionObserver + PerBlockQSNRObserver
  prescribe:      QSNRObserver
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session._result import SessionResult

from src.api._chart_helpers import _linear_layer_names

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    # Data types
    "GapEntry", "BottleneckAssessment", "TransformRecovery", "RankedLayer",
    "DistributionClassEntry", "ErrorRangeBucket",
    "LayerDiagnosis", "BlockDiagnosis", "DepthDecayPoint", "ErrorSourceEntry",
    "SensitivityEntry", "LayerTypeAgg", "BoostTarget", "RecoveryStrategy",
    "DistOverlayData",
    # Report types
    "CoarseReport", "DeepDiveReport", "PrescriptionReport",
    # Stage functions
    "coarse_pass", "deep_dive", "prescribe",
    # Pluggable utilities
    "detect_wxa_bottleneck",
    # Save
    "save_diagnostic_data",
]


# =====================================================================
# Shared data types
# =====================================================================

@dataclass
class GapEntry:
    """Per-config accuracy gap from FP32 baseline."""
    config: str
    accuracy: Optional[float] = None
    delta_from_fp32: Optional[float] = None
    avg_qsnr_db: Optional[float] = None


@dataclass
class BottleneckAssessment:
    """Which axis (weight or activation) dominates accuracy degradation."""
    weight_degradation: float = 0.0
    activation_degradation: float = 0.0
    primary: str = "unknown"  # "weight" | "activation" | "both" | "unknown"


@dataclass
class TransformRecovery:
    """Precision recovery from a transform (SmoothQuant / Hadamard)."""
    config: str = ""
    transform: str = ""
    accuracy_gain: float = 0.0
    recovery_pct: float = 0.0
    qsnr_gain_db: Optional[float] = None


@dataclass
class RankedLayer:
    """A layer ranked by cross-config QSNR analysis."""
    layer: str = ""
    avg_qsnr_db: float = 0.0
    worst_config: str = ""
    worst_qsnr_db: float = 0.0
    dominant_role: str = ""


@dataclass
class DistributionClassEntry:
    """Summary of a distribution archetype across layers."""
    name: str = ""
    count: int = 0
    percentage: str = ""
    avg_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ErrorRangeBucket:
    """Error statistics for a dynamic-range bucket."""
    range_label: str = ""
    avg_qsnr: float = 0.0
    count: int = 0
    verdict: str = ""


@dataclass
class LayerDiagnosis:
    """Per-layer distribution diagnosis."""
    layer: str = ""
    role: str = ""
    qsnr_db: float = 0.0
    classification: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""


@dataclass
class BlockDiagnosis:
    """Per-layer block/channel error analysis."""
    layer: str = ""
    role: str = ""
    unit_type: str = ""  # "block" | "channel"
    stats: Dict[str, float] = field(default_factory=dict)
    worst_units: List[Tuple[int, float]] = field(default_factory=list)
    error_pattern: str = ""


@dataclass
class DepthDecayPoint:
    """Single point on the depth-vs-QSNR curve."""
    depth: int = 0
    layer: str = ""
    qsnr_db: float = 0.0


@dataclass
class ErrorSourceEntry:
    """Per-layer error source classification."""
    layer: str = ""
    output_qsnr: float = 0.0
    accum_qsnr: Optional[float] = None
    dominant_role: str = ""
    error_source: str = ""


@dataclass
class SensitivityEntry:
    """Layer sensitivity ranking entry."""
    layer: str = ""
    role: str = ""
    value: float = 0.0
    layer_type: str = ""


@dataclass
class LayerTypeAgg:
    """Aggregated metrics per layer type."""
    layer_type: str = ""
    count: int = 0
    avg_qsnr_db: float = 0.0
    avg_mse: float = 0.0


@dataclass
class BoostTarget:
    """Intervention target layer."""
    layer: str = ""
    current_qsnr: float = 0.0
    dominant_role: str = ""
    action: str = ""
    reason: str = ""


@dataclass
class RecoveryStrategy:
    """A candidate recovery strategy."""
    strategy_type: str = ""  # "mixed_precision" | "transform" | "format_change"
    description: str = ""
    target_layers: List[str] = field(default_factory=list)
    expected_recovery_pct: float = 0.0
    priority: str = "medium"


@dataclass
class DistOverlayData:
    """Pre-computed histogram data for render_chart dist_overlay.

    Agent reads this directly and passes to render_chart:
        render_chart(data, "dist_overlay", x="bin", series=[...])
    """
    bins: List[float] = field(default_factory=list)
    fp32: List[float] = field(default_factory=list)
    quant: List[float] = field(default_factory=list)
    error: List[float] = field(default_factory=list)

    def to_chart_data(self) -> List[dict]:
        """Convert to render_chart data format."""
        return [
            {"bin": round(b, 4),
             "fp32": int(self.fp32[i]),
             "quant": int(self.quant[i]),
             "error": int(self.error[i])}
            for i, b in enumerate(self.bins)
            if i < len(self.fp32)
        ]


# =====================================================================
# Report dataclasses
# =====================================================================

@dataclass
class CoarseReport:
    """Output of ``coarse_pass()``: multi-config overview."""

    fp32_accuracy: Optional[float] = None
    gaps: List[GapEntry] = field(default_factory=list)
    bottleneck: BottleneckAssessment = field(default_factory=BottleneckAssessment)
    transform_effects: List[TransformRecovery] = field(default_factory=list)
    consistent_worst: List[RankedLayer] = field(default_factory=list)
    config_specific_worst: List[RankedLayer] = field(default_factory=list)
    distribution_taxonomy: List[DistributionClassEntry] = field(default_factory=list)
    error_by_range: List[ErrorRangeBucket] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=== Coarse Analysis ==="]
        if self.fp32_accuracy is not None:
            lines.append(f"FP32 baseline: {self.fp32_accuracy:.4f}")
        lines.append(f"Configs: {len(self.gaps)}")
        for g in self.gaps:
            delta = f"{g.delta_from_fp32:+.4f}" if g.delta_from_fp32 is not None else "N/A"
            lines.append(f"  {g.config}: acc={g.accuracy}, delta={delta}")
        lines.append(f"Bottleneck: {self.bottleneck.primary}")
        lines.append(f"Consistent worst: {len(self.consistent_worst)} layers")
        if self.transform_effects:
            lines.append(f"Transform effects: {len(self.transform_effects)}")
        if self.distribution_taxonomy:
            lines.append(f"Distribution classes: {len(self.distribution_taxonomy)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeepDiveReport:
    """Output of ``deep_dive()``: per-layer deep analysis."""

    layer_diagnoses: List[LayerDiagnosis] = field(default_factory=list)
    block_analyses: List[BlockDiagnosis] = field(default_factory=list)
    depth_decay: List[DepthDecayPoint] = field(default_factory=list)
    error_sources: List[ErrorSourceEntry] = field(default_factory=list)
    sensitivity_topk: List[SensitivityEntry] = field(default_factory=list)
    layer_type_aggregation: List[LayerTypeAgg] = field(default_factory=list)
    dist_overlays: Dict[str, Dict[str, "DistOverlayData"]] = field(default_factory=dict)
    # {layer_name: {role: DistOverlayData}} — pre-computed for worst layers

    def summary(self) -> str:
        lines = ["=== Deep Dive ==="]
        lines.append(f"Layer diagnoses: {len(self.layer_diagnoses)}")
        lines.append(f"Block analyses: {len(self.block_analyses)}")
        lines.append(f"Depth decay points: {len(self.depth_decay)}")
        lines.append(f"Error source entries: {len(self.error_sources)}")
        lines.append(f"Sensitivity top-K: {len(self.sensitivity_topk)}")
        if self.layer_type_aggregation:
            lines.append(f"Layer type groups: {len(self.layer_type_aggregation)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PrescriptionReport:
    """Output of ``prescribe()``: intervention plan + strategies."""

    boost_targets: List[BoostTarget] = field(default_factory=list)
    strategies: List[RecoveryStrategy] = field(default_factory=list)
    best_strategy: str = ""

    def summary(self) -> str:
        lines = ["=== Prescription ==="]
        lines.append(f"Boost targets: {len(self.boost_targets)}")
        for t in self.boost_targets[:5]:
            lines.append(f"  {t.layer}: {t.action} (QSNR={t.current_qsnr:.1f})")
        lines.append(f"Strategies: {len(self.strategies)}")
        if self.best_strategy:
            lines.append(f"Best: {self.best_strategy}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# Stage 1: coarse_pass
# =====================================================================

def coarse_pass(
    results: Dict[str, "SessionResult"],
    *,
    fp32_accuracy: Optional[float] = None,
    k: int = 5,
    bottleneck_fn: Optional[Callable[[List[GapEntry]], BottleneckAssessment]] = None,
) -> CoarseReport:
    """Multi-config gap analysis + cross-config ranking + distribution overview.

    Args:
        results: ``{config_name: SessionResult}`` for all quantization configs.
        fp32_accuracy: FP32 baseline accuracy.  If *None*, extracted from the
            first result's ``fp32_metrics``.
        k: Number of worst layers for cross-config ranking.
        bottleneck_fn: Custom bottleneck detector.  Receives the list of
            ``GapEntry`` and returns a ``BottleneckAssessment``.  Defaults to
            :func:`detect_wxa_bottleneck` which handles WxAy naming convention;
            pass a custom function for non-standard config naming.

    Returns:
        ``CoarseReport`` with gaps, bottleneck, rankings, taxonomy.
    """
    if not results:
        return CoarseReport()

    if fp32_accuracy is None:
        first = next(iter(results.values()))
        fp32_metrics = first.fp32_metrics or {}
        fp32_accuracy = fp32_metrics.get("accuracy")

    gaps = _build_gaps(results, fp32_accuracy)
    bottleneck = (bottleneck_fn or detect_wxa_bottleneck)(gaps)
    consistent, specific = _cross_config_ranking(results, k)
    transform_effects = _transform_effects(results)
    taxonomy, error_range = _distribution_overview(results)

    return CoarseReport(
        fp32_accuracy=fp32_accuracy,
        gaps=gaps,
        bottleneck=bottleneck,
        transform_effects=transform_effects,
        consistent_worst=consistent,
        config_specific_worst=specific,
        distribution_taxonomy=taxonomy,
        error_by_range=error_range,
    )


def detect_wxa_bottleneck(gaps: List[GapEntry]) -> BottleneckAssessment:
    """Default bottleneck detector: W8A8 → W4A8 (weight) → W4A4 (activation).

    Works with MXInt-style config naming (WxAy).  For other formats, pass a
    custom ``bottleneck_fn`` to :func:`coarse_pass`.
    """
    config_acc: Dict[str, Optional[float]] = {g.config: g.accuracy for g in gaps}

    w8a8 = config_acc.get("W8A8")
    w4a8 = config_acc.get("W4A8")
    w4a4 = config_acc.get("W4A4")

    # Fallback: substring match for compound names like "W4A4+SQ"
    if w8a8 is None:
        for g in gaps:
            if "W8A8" in g.config or g.config.endswith("8x8"):
                w8a8 = g.accuracy
            elif "W4A8" in g.config or g.config.endswith("4x8"):
                w4a8 = g.accuracy
            elif "W4A4" in g.config or g.config.endswith("4x4"):
                w4a4 = g.accuracy

    if w8a8 is None or w4a8 is None:
        return BottleneckAssessment()

    weight_deg = (w8a8 - w4a8) if w8a8 is not None and w4a8 is not None else 0.0
    activation_deg = (w4a8 - w4a4) if w4a8 is not None and w4a4 is not None else 0.0

    if weight_deg <= 0 and activation_deg <= 0:
        primary = "unknown"
    elif max(weight_deg, activation_deg) < 1e-9:
        primary = "unknown"
    elif abs(weight_deg - activation_deg) / max(weight_deg, activation_deg) < 0.2:
        primary = "both"
    elif weight_deg > activation_deg:
        primary = "weight"
    else:
        primary = "activation"

    return BottleneckAssessment(
        weight_degradation=weight_deg,
        activation_degradation=activation_deg,
        primary=primary,
    )


# =====================================================================
# Stage 2: deep_dive
# =====================================================================

def deep_dive(
    result: "SessionResult",
    *,
    layers: Optional[List[str]] = None,
    k: int = 5,
    top_k_blocks: int = 10,
) -> DeepDiveReport:
    """Single-config deep analysis: distribution + block + provenance + sensitivity.

    Each sub-analysis degrades independently when its required observer data
    is absent, so the report can be partially populated.

    Args:
        result: ``SessionResult`` with observers data.
        layers: Specific layers to analyse.  If *None*, the worst *k* layers
            (by output QSNR) are selected automatically.
        k: Number of worst layers when *layers* is *None*.
        top_k_blocks: Per-layer worst block/channel count.

    Returns:
        ``DeepDiveReport`` with diagnoses, block analyses, depth decay, etc.
    """
    target_layers = layers if layers is not None else _worst_layer_names(result, k)

    diagnoses = _diagnose_layers(result, target_layers)
    block_analyses = _analyse_blocks(result, target_layers, top_k_blocks)
    depth_decay = _depth_decay(result)
    error_sources = _error_sources(result)
    sensitivity, type_agg = _sensitivity_analysis(result, k)
    dist_overlays = _extract_dist_overlays(result, target_layers)

    return DeepDiveReport(
        layer_diagnoses=diagnoses,
        block_analyses=block_analyses,
        depth_decay=depth_decay,
        error_sources=error_sources,
        sensitivity_topk=sensitivity,
        layer_type_aggregation=type_agg,
        dist_overlays=dist_overlays,
    )


# =====================================================================
# Stage 3: prescribe
# =====================================================================

def prescribe(
    result: "SessionResult",
    *,
    k: int = 5,
    strategy: str = "conservative",
) -> PrescriptionReport:
    """Intervention planning + recovery strategy generation.

    Args:
        result: ``SessionResult`` with observers data.
        k: Number of top layers to consider for intervention.
        strategy: ``"conservative"`` or ``"aggressive"``.

    Returns:
        ``PrescriptionReport`` with boost targets and strategies.
    """
    boost_targets = _boost_targets(result, k)
    strategies = _recovery_strategies(result, k, strategy, boost_targets)

    best = ""
    if strategies:
        best = strategies[0].description

    return PrescriptionReport(
        boost_targets=boost_targets,
        strategies=strategies,
        best_strategy=best,
    )


# =====================================================================
# Internal helpers — coarse_pass
# =====================================================================

def _extract_accuracy(result: "SessionResult") -> Optional[float]:
    if result.quant_metrics:
        for key in ("accuracy", "acc", "eval_accuracy"):
            if key in result.quant_metrics:
                return result.quant_metrics[key]
    return None


def _avg_qsnr(result: "SessionResult") -> Optional[float]:
    if not result.qsnr_per_layer:
        return None
    values = [v for v in result.qsnr_per_layer.values()
              if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def _build_gaps(
    results: Dict[str, "SessionResult"],
    fp32_accuracy: Optional[float],
) -> List[GapEntry]:
    gaps: List[GapEntry] = []
    for name, r in results.items():
        acc = _extract_accuracy(r)
        delta = None
        if acc is not None and fp32_accuracy is not None:
            delta = acc - fp32_accuracy
        gaps.append(GapEntry(
            config=name,
            accuracy=acc,
            delta_from_fp32=delta,
            avg_qsnr_db=_avg_qsnr(r),
        ))
    return gaps


# Transform suffix patterns — own copy to avoid coupling to private attrs.
_TRANSFORM_SUFFIXES: List[Tuple[str, str]] = [
    (r"\+SQ$", "smoothquant"),
    (r"\+smoothquant$", "smoothquant"),
    (r"\+HD$", "hadamard"),
    (r"\+hadamard$", "hadamard"),
]


def _cross_config_ranking(
    results: Dict[str, "SessionResult"],
    k: int,
) -> Tuple[List[RankedLayer], List[RankedLayer]]:
    """Cross-config consistent and config-specific worst layers."""
    from src.analysis.cross_config_ranking import CrossConfigLayerRanking

    if len(results) < 2:
        return [], []

    ranking = CrossConfigLayerRanking.from_results(results)

    # Dominant role from qsnr_by_role (more reliable)
    def _dominant(layer: str) -> str:
        for r in results.values():
            qbr = r.qsnr_by_role or {}
            worst_role, worst_val = "", float("inf")
            for role in ("input", "weight", "output"):
                v = qbr.get(role, {}).get(layer)
                if v is not None and v < worst_val:
                    worst_val, worst_role = v, role
            if worst_role:
                return worst_role
        return ""

    # Consistent worst
    consistent: List[RankedLayer] = []
    for layer, avg_q in ranking.consistent_worst(k=k):
        # Find worst config for this layer
        worst_cfg, worst_q = "", float("inf")
        for cfg_name, r in results.items():
            q = r.qsnr_per_layer.get(layer)
            if q is not None and q < worst_q:
                worst_q, worst_cfg = q, cfg_name
        consistent.append(RankedLayer(
            layer=layer,
            avg_qsnr_db=round(avg_q, 1),
            worst_config=worst_cfg,
            worst_qsnr_db=round(worst_q, 1),
            dominant_role=_dominant(layer),
        ))

    # Config-specific worst
    specific: List[RankedLayer] = []
    for cfg_name in results:
        for layer, qsnr in ranking.config_specific_worst(cfg_name, k=3):
            specific.append(RankedLayer(
                layer=layer,
                avg_qsnr_db=round(qsnr, 1),
                worst_config=cfg_name,
                worst_qsnr_db=round(qsnr, 1),
                dominant_role=_dominant(layer),
            ))

    return consistent, specific


def _transform_effects(
    results: Dict[str, "SessionResult"],
) -> List[TransformRecovery]:
    """Detect and quantify transform recovery effects."""
    config_names = list(results.keys())

    effects: List[TransformRecovery] = []
    for name in config_names:
        base_name = None
        transform = None
        for pattern, transform_name in _TRANSFORM_SUFFIXES:
            match = re.search(pattern, name)
            if match:
                base_name = name[:match.start()]
                transform = transform_name
                break

        if base_name is None or base_name not in results:
            continue

        base_result = results[base_name]
        trans_result = results[name]

        base_acc = _extract_accuracy(base_result)
        trans_acc = _extract_accuracy(trans_result)
        fp32_acc = None
        for r in (base_result, trans_result):
            if r.fp32_metrics:
                fp32_acc = r.fp32_metrics.get("accuracy")
                if fp32_acc is not None:
                    break

        accuracy_gain = None
        if base_acc is not None and trans_acc is not None:
            accuracy_gain = trans_acc - base_acc

        recovery_pct = None
        if fp32_acc is not None and base_acc is not None and accuracy_gain is not None:
            gap = fp32_acc - base_acc
            if abs(gap) > 1e-10:
                recovery_pct = (accuracy_gain / gap) * 100

        base_q = _avg_qsnr(base_result)
        trans_q = _avg_qsnr(trans_result)
        qsnr_gain = None
        if base_q is not None and trans_q is not None:
            qsnr_gain = trans_q - base_q

        effects.append(TransformRecovery(
            config=base_name,
            transform=transform or "",
            accuracy_gain=accuracy_gain or 0.0,
            recovery_pct=recovery_pct or 0.0,
            qsnr_gain_db=qsnr_gain,
        ))

    return effects


def _distribution_overview(
    results: Dict[str, "SessionResult"],
) -> Tuple[List[DistributionClassEntry], List[ErrorRangeBucket]]:
    """Distribution taxonomy and error-by-range from the worst config."""
    taxonomy_entries: List[DistributionClassEntry] = []
    error_entries: List[ErrorRangeBucket] = []

    # Pick worst config by avg QSNR (skip results with no QSNR data)
    candidates = [(r, _avg_qsnr(r)) for r in results.values()]
    with_qsnr = [(r, q) for r, q in candidates if q is not None]
    if not with_qsnr:
        return taxonomy_entries, error_entries
    worst_result = min(with_qsnr, key=lambda x: x[1])[0]

    try:
        report = worst_result.report
        if report is None:
            return taxonomy_entries, error_entries

        from src.analysis.correlation import DistributionTaxonomy, ErrorByDistribution

        taxon = DistributionTaxonomy.from_report(report)
        classes = taxon.classify()
        for name, data in sorted(classes.items(), key=lambda x: x[1]["count"], reverse=True):
            taxonomy_entries.append(DistributionClassEntry(
                name=name,
                count=data["count"],
                percentage=data.get("percentage", ""),
                avg_metrics=data.get("avg_metrics", {}),
            ))

        ebd = ErrorByDistribution(report)
        range_groups = ebd.group_by_range()
        for label, stats in sorted(range_groups.items()):
            error_entries.append(ErrorRangeBucket(
                range_label=label,
                avg_qsnr=stats.get("avg_qsnr", 0),
                count=stats.get("count", 0),
                verdict=stats.get("verdict", ""),
            ))
    except (AttributeError, KeyError, TypeError) as exc:
        logger.debug("distribution overview unavailable: %s", exc)

    return taxonomy_entries, error_entries


# =====================================================================
# Internal helpers — deep_dive
# =====================================================================

def _worst_layer_names(result: "SessionResult", k: int) -> List[str]:
    """Return names of the k worst layers by output QSNR."""
    qsnr = result.accum_qsnr_per_layer or result.qsnr_per_layer
    if not qsnr:
        return []
    obs = result.observers_data or {}
    linear = _linear_layer_names(obs) if obs else None

    candidates = [(l, q) for l, q in qsnr.items()
                  if math.isfinite(q) and (not linear or l in linear)]
    candidates.sort(key=lambda x: x[1])
    return [name for name, _ in candidates[:k]]


def _diagnose_layers(
    result: "SessionResult",
    layers: List[str],
) -> List[LayerDiagnosis]:
    """Distribution diagnosis for target layers."""
    from src.analysis._distribution_diagnosis import classify_distribution

    diagnoses: List[LayerDiagnosis] = []
    obs = result.observers_data
    qsnr_by_role = result.qsnr_by_role or {}

    for layer in layers:
        layer_data = obs.get(layer, {})
        for role in ("weight", "input", "output"):
            stages = layer_data.get(role, {})
            if not stages:
                continue

            qsnr = qsnr_by_role.get(role, {}).get(layer)
            if qsnr is None or not math.isfinite(qsnr):
                continue

            # Get distribution metrics
            dist_metrics = None
            for _stage, slices in stages.items():
                for _key, metrics in slices.items():
                    if "crest_factor" in metrics:
                        dist_metrics = metrics
                        break
                if dist_metrics:
                    break

            if dist_metrics is None:
                continue

            label, _desc, suggestion = classify_distribution(dist_metrics)
            features = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in dist_metrics.items()
                if isinstance(v, (int, float))
            }

            diagnoses.append(LayerDiagnosis(
                layer=layer,
                role=role,
                qsnr_db=round(qsnr, 1),
                classification=label,
                features=features,
                suggestion=suggestion,
            ))

    return diagnoses


def _analyse_blocks(
    result: "SessionResult",
    layers: List[str],
    top_k: int,
) -> List[BlockDiagnosis]:
    """Per-layer block/channel error analysis."""
    from src.api.block_error_analysis import block_error_analysis

    analyses: List[BlockDiagnosis] = []
    for layer in layers:
        for role in ("weight", "input"):
            report = block_error_analysis(result, layer=layer, role=role, top_k=top_k)
            if not report.per_unit_qsnr:
                continue

            # Classify error pattern
            values = list(report.per_unit_qsnr.values())
            mean_v = report.stats.get("mean", 0)
            std_v = report.stats.get("std", 0)
            cv = std_v / abs(mean_v) if mean_v != 0 else 0

            if cv > 0.5:
                pattern = "concentrated"
            elif cv < 0.15:
                pattern = "uniform"
            else:
                pattern = "mixed"

            analyses.append(BlockDiagnosis(
                layer=layer,
                role=role,
                unit_type=report.unit_type,
                stats={k: round(v, 2) for k, v in report.stats.items()},
                worst_units=[(idx, round(q, 1)) for idx, q in report.worst_units],
                error_pattern=pattern,
            ))

    return analyses


def _depth_decay(result: "SessionResult") -> List[DepthDecayPoint]:
    """QSNR vs network depth data."""
    try:
        data = result.diagnose.depth_decay_data(role="output")
        return [
            DepthDecayPoint(depth=d, layer=l, qsnr_db=round(q, 1))
            for d, l, q in data
            if math.isfinite(q)
        ]
    except (AttributeError, KeyError) as exc:
        logger.debug("depth decay unavailable: %s", exc)
        return []


def _classify_error_source(accum_qsnr: Optional[float], local_qsnr: float) -> str:
    if accum_qsnr is None or not math.isfinite(accum_qsnr):
        return "Local"
    diff = abs(accum_qsnr - local_qsnr)
    if diff < 3.0:
        return "Source"
    elif diff < 10.0:
        return "Mixed"
    return "Propagated"


def _error_sources(result: "SessionResult") -> List[ErrorSourceEntry]:
    """Per-layer error source classification."""
    obs = result.observers_data
    qsnr_by_role = result.qsnr_by_role or {}
    accum = result.accum_qsnr_per_layer or {}

    # Source layers: prefer observers_data, fall back to qsnr_by_role keys
    if obs:
        linear = _linear_layer_names(obs)
        source_layers = [l for l in obs if l in linear] if linear else list(obs.keys())
    else:
        # No observers — use union of all role keys
        all_layers: set = set()
        for role_map in qsnr_by_role.values():
            all_layers.update(role_map.keys())
        source_layers = sorted(all_layers)

    entries: List[ErrorSourceEntry] = []
    for layer in source_layers:
        role_qsnrs: Dict[str, float] = {}
        for role in ("input", "weight", "output"):
            v = qsnr_by_role.get(role, {}).get(layer)
            if v is not None and math.isfinite(v):
                role_qsnrs[role] = v

        if not role_qsnrs:
            continue

        local_qsnr = role_qsnrs.get("output", min(role_qsnrs.values()))
        accum_q = accum.get(layer)
        dominant = min(role_qsnrs, key=role_qsnrs.get)
        source = _classify_error_source(accum_q, local_qsnr)

        entries.append(ErrorSourceEntry(
            layer=layer,
            output_qsnr=round(local_qsnr, 1),
            accum_qsnr=round(accum_q, 1) if accum_q is not None and math.isfinite(accum_q) else None,
            dominant_role=dominant,
            error_source=source,
        ))

    return entries


def _sensitivity_analysis(
    result: "SessionResult",
    k: int,
) -> Tuple[List[SensitivityEntry], List[LayerTypeAgg]]:
    """Layer sensitivity ranking + layer-type aggregation."""
    try:
        report = result.report
        if report is None:
            return [], []

        from src.analysis.correlation import LayerSensitivity

        ls = LayerSensitivity(report)

        # Top-K by MSE (most sensitive)
        sensitivity: List[SensitivityEntry] = []
        for layer, role, mse, ltype in ls.topk(k=k, metric="mse"):
            sensitivity.append(SensitivityEntry(
                layer=layer, role=role, value=round(mse, 6), layer_type=ltype,
            ))

        # Layer type aggregation
        type_agg: List[LayerTypeAgg] = []
        by_type = ls.by_layer_type()
        for ltype, data in sorted(by_type.items()):
            type_agg.append(LayerTypeAgg(
                layer_type=ltype,
                count=data.get("count", 0),
                avg_qsnr_db=round(data.get("avg_qsnr_db", 0), 1),
                avg_mse=round(data.get("avg_mse", 0), 6),
            ))

        return sensitivity, type_agg
    except (AttributeError, KeyError, TypeError) as exc:
        logger.debug("sensitivity analysis unavailable: %s", exc)
        return [], []


# =====================================================================
# Internal helpers — prescribe
# =====================================================================

def _boost_targets(result: "SessionResult", k: int) -> List[BoostTarget]:
    """Generate boost targets from InterventionPlanner."""
    try:
        plan = result.plan.top_k_boost(k=k, role="auto", target_bits=8)
    except (AttributeError, KeyError) as exc:
        logger.debug("boost targets unavailable: %s", exc)
        return []

    changes = plan.metadata.get("changes", {})
    qsnr_by_role = result.qsnr_by_role or {}

    targets: List[BoostTarget] = []
    for layer in sorted(plan.overrides):
        info = changes.get(layer, {})
        # Find dominant role and QSNR
        worst_role, worst_q = "", float("inf")
        for role in ("input", "weight", "output"):
            v = qsnr_by_role.get(role, {}).get(layer)
            if v is not None and v < worst_q:
                worst_q, worst_role = v, role

        targets.append(BoostTarget(
            layer=layer,
            current_qsnr=round(worst_q, 1) if worst_q != float("inf") else 0.0,
            dominant_role=worst_role,
            action=info.get("what", "boost precision"),
            reason=info.get("why", ""),
        ))

    return targets


def _recovery_strategies(
    result: "SessionResult",
    k: int,
    strategy: str,
    boost_targets: List[BoostTarget],
) -> List[RecoveryStrategy]:
    """Generate recovery strategies from InterventionPlanner."""
    strategies: List[RecoveryStrategy] = []

    # Estimate recovery from QSNR data when available
    def _estimate_recovery(layers: List[str]) -> float:
        if not result.qsnr_by_role:
            return 0.0
        # Rough heuristic: sum of (QSNR deficit / 60) across target layers
        total_deficit = 0.0
        for layer in layers:
            worst_q = float("inf")
            for role in ("input", "weight", "output"):
                v = result.qsnr_by_role.get(role, {}).get(layer)
                if v is not None and v < worst_q:
                    worst_q = v
            if worst_q != float("inf"):
                total_deficit += max(0, 60.0 - worst_q)
        # Normalize: full deficit recovery ≈ 100%, scale by layer count
        return min(100.0, total_deficit / max(len(layers), 1) * 2)

    # Strategy 1: Mixed precision (top-k boost)
    try:
        plan = result.plan.top_k_boost(k=k, role="auto", target_bits=8)
        if plan.overrides:
            layers = list(plan.overrides.keys())
            strategies.append(RecoveryStrategy(
                strategy_type="mixed_precision",
                description=f"Boost top-{len(layers)} worst layers to higher precision",
                target_layers=layers,
                expected_recovery_pct=round(_estimate_recovery(layers), 1),
                priority="high",
            ))
    except (AttributeError, KeyError) as exc:
        logger.debug("mixed-precision strategy unavailable: %s", exc)

    # Strategy 2: Conservative recommendation
    try:
        rec = result.plan.recommend(strategy="conservative")
        if rec.overrides:
            layers = list(rec.overrides.keys())
            strategies.append(RecoveryStrategy(
                strategy_type="mixed_precision",
                description=f"Conservative: boost {len(layers)} layers",
                target_layers=layers,
                expected_recovery_pct=round(_estimate_recovery(layers), 1),
                priority="medium",
            ))
    except (AttributeError, KeyError) as exc:
        logger.debug("conservative strategy unavailable: %s", exc)

    # Strategy 3: Transform suggestion
    try:
        ranking_text = result.plan.transform_ranking(k=k)
        if ranking_text and "no transform" not in ranking_text.lower():
            target_layers = [t.layer for t in boost_targets][:k]
            strategies.append(RecoveryStrategy(
                strategy_type="transform",
                description="Apply transform to worst-K layers",
                target_layers=target_layers,
                expected_recovery_pct=round(_estimate_recovery(target_layers), 1),
                priority="medium",
            ))
    except (AttributeError, KeyError) as exc:
        logger.debug("transform strategy unavailable: %s", exc)

    return strategies


# =====================================================================
# Dist overlay extraction
# =====================================================================

def _extract_dist_overlays(
    result: "SessionResult",
    layers: List[str],
) -> Dict[str, Dict[str, DistOverlayData]]:
    """Extract histogram data for dist_overlay charts (worst layers only)."""
    from src.api._chart_helpers import _get_hist_data

    obs = result.observers_data
    overlays: Dict[str, Dict[str, DistOverlayData]] = {}

    for layer in layers:
        layer_data = obs.get(layer, {})
        for role in ("weight", "input"):
            hist = _get_hist_data(obs, layer, role)
            if hist is None or "fp32_hist" not in hist:
                continue

            fp32_hist = hist["fp32_hist"]
            if hasattr(fp32_hist, "tolist"):
                fp32_hist = fp32_hist.tolist()

            n_bins = len(fp32_hist)
            fp32_min = hist.get("fp32_min", 0)
            fp32_max = hist.get("fp32_max", 1)
            bin_width = (fp32_max - fp32_min) / n_bins if n_bins > 0 else 1

            bins = [round(fp32_min + (i + 0.5) * bin_width, 4) for i in range(n_bins)]
            fp32 = fp32_hist

            quant = hist.get("quant_hist", [])
            if hasattr(quant, "tolist"):
                quant = quant.tolist()
            quant = list(quant) if quant else [0] * n_bins

            error = hist.get("err_hist", [])
            if hasattr(error, "tolist"):
                error = error.tolist()
            error = list(error) if error else [0] * n_bins

            # Pad/truncate to match n_bins
            quant = (quant + [0] * n_bins)[:n_bins]
            error = (error + [0] * n_bins)[:n_bins]

            overlays.setdefault(layer, {})[role] = DistOverlayData(
                bins=bins, fp32=fp32, quant=quant, error=error,
            )

    return overlays


# =====================================================================
# save_diagnostic_data — incremental JSON storage
# =====================================================================

def save_diagnostic_data(
    coarse: CoarseReport,
    deep_dive_report: DeepDiveReport,
    prescription: PrescriptionReport,
    output_dir: str,
) -> str:
    """Save diagnostic data as incrementally-loadable JSON files.

    Creates an ``index.json`` + split files under ``<output_dir>/diagnostic/``.
    Reporter agent reads ``index.json`` first, then loads specific files on
    demand.

    Args:
        coarse: Output of ``coarse_pass()``.
        deep_dive_report: Output of ``deep_dive()``.
        prescription: Output of ``prescribe()``.
        output_dir: Base directory (typically the Study output dir).

    Returns:
        Path to the ``diagnostic/`` directory.
    """
    base = os.path.join(output_dir, "diagnostic")
    _mkdir(base, f"{base}/coarse", f"{base}/deep_dive", f"{base}/prescription")

    # ── Coarse files ────────────────────────────────────────────────
    _write(f"{base}/coarse/gaps.json", coarse.gaps)
    _write(f"{base}/coarse/bottleneck.json", coarse.bottleneck)
    _write(f"{base}/coarse/consistent_worst.json", coarse.consistent_worst)
    _write(f"{base}/coarse/config_specific_worst.json", coarse.config_specific_worst)
    _write(f"{base}/coarse/transform_effects.json", coarse.transform_effects)
    _write(f"{base}/coarse/distribution_taxonomy.json", coarse.distribution_taxonomy)
    _write(f"{base}/coarse/error_by_range.json", coarse.error_by_range)

    # ── Deep dive files ────────────────────────────────────────────
    _write(f"{base}/deep_dive/depth_decay.json", deep_dive_report.depth_decay)
    _write(f"{base}/deep_dive/error_sources.json", deep_dive_report.error_sources)
    _write(f"{base}/deep_dive/sensitivity.json", {
        "topk": deep_dive_report.sensitivity_topk,
        "layer_type_aggregation": deep_dive_report.layer_type_aggregation,
    })

    # Per-layer files (diagnoses + blocks + dist_overlay)
    layer_names = sorted({d.layer for d in deep_dive_report.layer_diagnoses}
                         | {b.layer for b in deep_dive_report.block_analyses}
                         | set(deep_dive_report.dist_overlays.keys()))

    layer_index: Dict[str, str] = {}
    for layer in layer_names:
        safe = _safe_layer_name(layer)
        layer_data: Dict[str, Any] = {}

        layer_diags = [asdict(d) for d in deep_dive_report.layer_diagnoses if d.layer == layer]
        if layer_diags:
            layer_data["diagnoses"] = layer_diags

        layer_blocks = []
        for b in deep_dive_report.block_analyses:
            if b.layer == layer:
                bd = asdict(b)
                bd["worst_units"] = [list(u) for u in b.worst_units]
                layer_blocks.append(bd)
        if layer_blocks:
            layer_data["blocks"] = layer_blocks

        layer_overlays = deep_dive_report.dist_overlays.get(layer, {})
        if layer_overlays:
            overlay_data = {}
            for role, od in layer_overlays.items():
                overlay_data[role] = asdict(od)
            layer_data["dist_overlay"] = overlay_data

        fname = f"layer_{safe}.json"
        _write(f"{base}/deep_dive/{fname}", layer_data)

        parts = []
        if layer_diags:
            parts.append("diagnoses")
        if layer_blocks:
            parts.append("blocks")
        if layer_overlays:
            parts.append("dist_overlay")
        layer_index[layer] = f"{fname} — {', '.join(parts)}"

    _write(f"{base}/deep_dive/index.json", {
        "layers": layer_index,
        "global_files": {
            "depth_decay.json": "Network depth vs QSNR curve",
            "error_sources.json": "Per-layer error source classification",
            "sensitivity.json": "Layer sensitivity ranking + type aggregation",
        },
    })

    # ── Prescription files ─────────────────────────────────────────
    _write(f"{base}/prescription/boost_targets.json", prescription.boost_targets)
    _write(f"{base}/prescription/strategies.json", prescription.strategies)

    # ── Top-level index ────────────────────────────────────────────
    index = {
        "fp32_accuracy": coarse.fp32_accuracy,
        "config_names": [g.config for g in coarse.gaps],
        "bottleneck_primary": coarse.bottleneck.primary,
        "n_consistent_worst": len(coarse.consistent_worst),
        "n_transform_effects": len(coarse.transform_effects),
        "available_data": {
            "coarse": {
                "description": "Global accuracy overview + cross-config ranking",
                "files": {
                    "gaps.json": "Per-config accuracy and delta from FP32",
                    "bottleneck.json": "Weight vs activation bottleneck assessment",
                    "consistent_worst.json": "Layers worst across ALL configs",
                    "config_specific_worst.json": "Layers worst in specific configs only",
                    "transform_effects.json": "SmoothQuant/Hadamard recovery effects",
                    "distribution_taxonomy.json": "Distribution archetype classification",
                    "error_by_range.json": "Error stats by dynamic range bucket",
                },
            },
            "deep_dive": {
                "description": "Per-layer deep analysis (single worst config)",
                "index": "deep_dive/index.json — lists available layers and their data",
                "global_files": {
                    "depth_decay.json": "QSNR vs network depth curve",
                    "error_sources.json": "Source/Mixed/Propagated per layer",
                    "sensitivity.json": "Top-K sensitivity + layer type aggregation",
                },
            },
            "prescription": {
                "description": "Intervention recommendations",
                "files": {
                    "boost_targets.json": "Layers to boost with current QSNR",
                    "strategies.json": "Recovery strategies with expected impact",
                },
            },
        },
    }
    _write(f"{base}/index.json", index)

    return base


# ── Save helpers ────────────────────────────────────────────────────

def _mkdir(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _write(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


_UNSAFE_FILENAME_RE = re.compile(r"[^\w\-]+")


def _safe_layer_name(layer: str) -> str:
    return _UNSAFE_FILENAME_RE.sub("_", layer).strip("_") or "unknown_layer"


def _json_default(obj: Any) -> Any:
    """Handle non-serializable types in asdict output."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return round(obj, 6)
        return None
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
