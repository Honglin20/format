"""Convert SessionResult lists to the dict formats expected by viz/tables and viz/figures.

These functions bridge the gap between the typed ``SessionResult`` and
the loosely-typed dicts that the visualisation layer consumes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.report._session_report import SessionReport
from src.session._result import SessionResult


def results_to_viz_dict(results: List[SessionResult]) -> dict:
    """Convert SessionResult list to flat dict format for viz functions.

    Returns::

        {"config_name": {"accuracy": {...}, "qsnr_per_layer": {...}, ...}}
    """
    viz_dict: Dict[str, dict] = {}
    for r in results:
        entry: Dict[str, Any] = {}
        if r.quant_metrics is not None:
            entry["accuracy"] = r.quant_metrics
        if r.qsnr_per_layer:
            entry["qsnr_per_layer"] = r.qsnr_per_layer
        if r.mse_per_layer:
            entry["mse_per_layer"] = r.mse_per_layer
        if r.delta is not None:
            entry["delta"] = r.delta
        if r.fp32_metrics is not None:
            entry["fp32_accuracy"] = r.fp32_metrics
        viz_dict[r.name] = entry
    return viz_dict


def results_to_nested_viz_dict(
    results: List[SessionResult],
    config_descriptors: List[dict],
) -> dict:
    """Convert SessionResult list to nested ``{format: {transform_label: data}}``.

    Uses config descriptors to infer format grouping and transform labels.
    A config without a ``"transform"`` key (or ``"transform": "none"``) is
    the baseline (key ``"None"``).  ``"hadamard"`` -> ``"Hadamard"``,
    ``"smoothquant"`` -> ``"SmoothQuant"``.

    Returns::

        {"fmt_base": {"None": {...}, "Hadamard": {...}, ...}}
    """
    TX_LABEL = {
        "none": "None", "hadamard": "Hadamard",
        "smoothquant": "SmoothQuant", "prescale": "PreScale",
    }

    # Build a lookup: config_name -> (base_format, transform_label)
    name_to_group: Dict[str, tuple] = {}
    for desc in config_descriptors:
        name = desc.get("name")
        if not name:
            continue
        tx_raw = desc.get("transform", "none")
        if tx_raw is None:
            tx_raw = "none"
        tx_label = TX_LABEL.get(str(tx_raw).lower(), str(tx_raw))
        # Infer base format name: strip known transform suffixes
        base = name
        for suffix in ("-Had", "-None", "-SmoothQuant", "-Hadamard", "-SQ"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        name_to_group[name] = (base, tx_label)

    # Also try to fill from SessionResult.config for results without descriptors
    for r in results:
        if r.name not in name_to_group:
            tx_raw = r.config.transform
            tx_label = TX_LABEL.get(tx_raw, tx_raw)
            name_to_group[r.name] = (r.name, tx_label)

    nested: Dict[str, dict] = {}
    for r in results:
        if r.name not in name_to_group:
            base_fmt = r.name
            tx_label = "None"
        else:
            base_fmt, tx_label = name_to_group[r.name]

        entry: Dict[str, Any] = {}
        if r.quant_metrics is not None:
            entry["accuracy"] = r.quant_metrics
        if r.qsnr_per_layer:
            entry["qsnr_per_layer"] = r.qsnr_per_layer
        if r.mse_per_layer:
            entry["mse_per_layer"] = r.mse_per_layer
        if r.delta is not None:
            entry["delta"] = r.delta
        if r.fp32_metrics is not None:
            entry["fp32_accuracy"] = r.fp32_metrics

        nested.setdefault(base_fmt, {})[tx_label] = entry
    return nested


def results_to_combined_viz_dict(
    all_results: Dict[str, List[SessionResult]],
) -> dict:
    """Convert all results to nested dict format for cross-part viz functions.

    Returns::

        {"part_name": {"config_name": {"accuracy": {...}, ...}}}
    """
    combined: Dict[str, dict] = {}
    for part_name, part_results in all_results.items():
        combined[part_name] = results_to_viz_dict(part_results)
    return combined


def extract_metric_per_layer(
    report: SessionReport,
    metric: str,
) -> Dict[str, float]:
    """Extract per-layer average of a metric from a SessionReport.

    Args:
        report: A ``SessionReport`` wrapping a single ``SessionResult``.
        metric: Metric key, e.g. ``"qsnr"`` or ``"mse"``.

    Returns:
        ``{layer_name: mean_value}``
    """
    df = report.to_dataframe()
    if isinstance(df, list):
        result: Dict[str, list] = defaultdict(list)
        for row in df:
            name = row.get("layer", "unknown")
            val = row.get(metric)
            if val is not None:
                result[name].append(val)
        return {k: sum(v) / len(v) for k, v in result.items()}
    else:
        # Assume pandas DataFrame
        grouped = df.groupby("layer")[metric].mean()
        return grouped.to_dict()
