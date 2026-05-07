"""StudyReport — declaration-driven output layer for format study results.

Usage::

    report = StudyReport(results)
    report.print_summary()
    report.save(output_dir, config=config)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from src.pipeline.runner import ExperimentResult

# ---------------------------------------------------------------------------
# Lazy-loaded registries (populated on first use)
# ---------------------------------------------------------------------------

_TABLE_REGISTRY: Dict[str, Callable] = {}
_FIGURE_REGISTRY: Dict[str, Callable] = {}


def _ensure_registries():
    """Lazily import and register all viz functions on first use."""
    if _TABLE_REGISTRY:
        return  # Already populated

    # Tables
    from src.viz.tables import (
        accuracy_table,
        pot_delta_table,
        transform_matrix_table,
        transform_distribution_table,
        sensitivity_table,
    )
    _TABLE_REGISTRY["accuracy"] = accuracy_table
    _TABLE_REGISTRY["pot_delta"] = pot_delta_table
    _TABLE_REGISTRY["transform_matrix"] = transform_matrix_table
    _TABLE_REGISTRY["transform_distribution"] = transform_distribution_table
    _TABLE_REGISTRY["sensitivity"] = sensitivity_table

    # Figures
    from src.viz.figures import (
        block_sweep_line_chart,
        error_vs_distribution,
        hierarchical_delta_bar,
        histogram_overlay,
        layer_type_qsnr,
        mse_box_plot,
        pot_delta_bar,
        qsnr_line_chart,
        transform_delta,
        transform_heatmap,
        transform_pie,
    )
    from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS

    _FIGURE_REGISTRY["qsnr_line"] = (
        lambda d, od: qsnr_line_chart(d, title="QSNR per Layer", colors=FORMAT_COLORS, output_dir=od)
    )
    _FIGURE_REGISTRY["mse_box"] = (
        lambda d, od: mse_box_plot(d, title="MSE Distribution", colors=FORMAT_COLORS, output_dir=od)
    )
    _FIGURE_REGISTRY["pot_delta_bar"] = (
        lambda d, od: pot_delta_bar(d, output_dir=od)
    )
    _FIGURE_REGISTRY["transform_heatmap"] = (
        lambda d, od: transform_heatmap(d, colors=FORMAT_COLORS, output_dir=od)
    )
    _FIGURE_REGISTRY["transform_pie"] = (
        lambda d, od: transform_pie(d, colors=TRANSFORM_COLORS, output_dir=od)
    )
    _FIGURE_REGISTRY["transform_delta"] = (
        lambda d, od: transform_delta(d, colors=TRANSFORM_COLORS, output_dir=od)
    )
    _FIGURE_REGISTRY["histogram"] = (
        lambda d, od: histogram_overlay(d, output_dir=od)
    )
    _FIGURE_REGISTRY["error_vs_dist"] = (
        lambda d, od: error_vs_distribution(d, output_dir=od)
    )
    _FIGURE_REGISTRY["layer_type_qsnr"] = (
        lambda d, od: layer_type_qsnr(d, output_dir=od)
    )
    _FIGURE_REGISTRY["block_sweep"] = (
        lambda d, od: block_sweep_line_chart(d, output_dir=od)
    )
    _FIGURE_REGISTRY["hierarchical_delta"] = (
        lambda d, od: hierarchical_delta_bar(d, colors=FORMAT_COLORS, output_dir=od)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_table(func, table_key: str, part_name: str, viz_dict: dict, output_dir: str):
    """Call a table function with the appropriate keyword arguments.

    Adapts the variadic signatures of viz table functions to the uniform
    ``(results, output_dir)`` call site in :meth:`StudyReport.save`.
    """
    filename = f"{part_name}_{table_key}.csv"
    title = part_name
    try:
        func(viz_dict, title=title, output_dir=output_dir, filename=filename)
    except TypeError:
        # Older viz functions may not accept title/filename kwargs
        func(viz_dict, output_dir)


def _results_to_viz_dict(results: List[ExperimentResult]) -> dict:
    """Convert ExperimentResult list to the flat dict format expected by viz functions.

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


def _results_to_nested_viz_dict(
    results: List[ExperimentResult],
    config_descriptors: List[dict],
) -> dict:
    """Convert ExperimentResult list to nested ``{format: {transform_label: data}}``.

    Uses config descriptors to infer format grouping and transform labels.
    A config without a ``"transform"`` key (or ``"transform": "none"``) is
    the baseline (key ``"None"``).  ``"hadamard"`` → ``"Hadamard"``,
    ``"smoothquant"`` → ``"SmoothQuant"``.

    Returns::

        {"fmt_base": {"None": {...}, "Hadamard": {...}, ...}}
    """
    TX_LABEL = {"none": "None", "hadamard": "Hadamard", "smoothquant": "SmoothQuant"}

    # Build a lookup: config_name → (base_format, transform_label)
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

    nested: Dict[str, dict] = {}
    for r in results:
        if r.name not in name_to_group:
            # Fallback: use raw name as both format and transform
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


def _results_to_combined_viz_dict(
    all_results: Dict[str, List[ExperimentResult]],
) -> dict:
    """Convert all results to the nested dict format for cross-part viz functions.

    Returns::

        {"part_name": {"config_name": {"accuracy": {...}, ...}}}
    """
    combined: Dict[str, dict] = {}
    for part_name, part_results in all_results.items():
        combined[part_name] = _results_to_viz_dict(part_results)
    return combined


# ---------------------------------------------------------------------------
# StudyReport
# ---------------------------------------------------------------------------

class StudyReport:
    """Declaration-driven output layer for format study results.

    Takes raw ``Dict[str, List[ExperimentResult]]`` from ``ExperimentRunner``
    and handles all terminal output (terminal summary, CSV tables, figures,
    JSON export) through a registry pattern.
    """

    def __init__(self, results: Dict[str, List[ExperimentResult]]):
        self._results = results

    # ── properties ──────────────────────────────────────────────────────

    @property
    def parts(self) -> List[str]:
        return list(self._results.keys())

    @property
    def total_experiments(self) -> int:
        return sum(len(v) for v in self._results.values())

    # ── print_summary ───────────────────────────────────────────────────

    def print_summary(self):
        """Print a terminal comparison table per part."""
        for part_name, part_results in self._results.items():
            print(f"\n{'='*70}")
            print(f"  Part: {part_name}")
            print(f"{'='*70}")
            if not part_results:
                print("  (no results)")
                continue

            # Header
            hdr = f"{'Config':<24} {'Avg QSNR':<12} {'Avg MSE':<12} {'Acc Delta':<12}"
            print(f"  {hdr}")
            print(f"  {'-'*len(hdr)}")

            for r in part_results:
                delta_str = ""
                if r.delta:
                    vals = [f"{k}={v:+.4f}" for k, v in r.delta.items()]
                    delta_str = ", ".join(vals)
                print(f"  {r.name:<24} {r.avg_qsnr:<12.2f} {r.avg_mse:<12.6f} {delta_str}")

    # ── to_serializable ─────────────────────────────────────────────────

    def to_serializable(self) -> dict:
        """Return a JSON-serializable dict of all results."""
        serializable: Dict[str, dict] = {}
        for part_name, part_results in self._results.items():
            serializable[part_name] = {}
            for r in part_results:
                entry: Dict[str, Any] = {}
                if r.quant_metrics is not None:
                    entry["accuracy"] = r.quant_metrics
                if r.fp32_metrics is not None:
                    entry["fp32_accuracy"] = r.fp32_metrics
                if r.delta is not None:
                    entry["delta"] = r.delta
                if r.qsnr_per_layer:
                    entry["qsnr_per_layer"] = r.qsnr_per_layer
                if r.mse_per_layer:
                    entry["mse_per_layer"] = r.mse_per_layer
                serializable[part_name][r.name] = entry
        return serializable

    # ── save ────────────────────────────────────────────────────────────

    def save(self, output_dir: str, config: Optional[dict] = None):
        """Generate CSV tables and figures based on per-part output declarations.

        Args:
            output_dir: Output root directory.
            config: Full study config dict. Each part can declare an
                ``output`` key with ``{"tables": [...], "figures": [...]}``.
                If not given, all parts default to ``{"tables": ["accuracy"],
                "figures": ["qsnr_line"]}``.
        """
        _ensure_registries()
        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        if config is None:
            config = {}

        # ---- Per-part tables and figures ----
        for part_name, part_results in self._results.items():
            part_output = config.get(part_name, {}).get("output", {})
            table_keys = part_output.get("tables", ["accuracy"])
            figure_keys = part_output.get("figures", ["qsnr_line"])

            viz_dict = _results_to_viz_dict(part_results)

            # Transform figures expect nested {fmt: {tx: data}} — build on demand
            _nested_cache: Optional[dict] = None
            _transform_figures = {"transform_heatmap", "transform_pie", "transform_delta"}

            for table_key in table_keys:
                if table_key == "sensitivity":
                    continue  # Handled separately below
                func = _TABLE_REGISTRY.get(table_key)
                if func is None:
                    print(f"  Warning: unknown table '{table_key}' for {part_name}, skipped")
                    continue
                try:
                    _call_table(func, table_key, part_name, viz_dict, output_dir)
                except Exception as e:
                    print(f"  Warning: table '{table_key}' for {part_name} failed: {e}")

            for fig_key in figure_keys:
                func = _FIGURE_REGISTRY.get(fig_key)
                if func is None:
                    print(f"  Warning: unknown figure '{fig_key}' for {part_name}, skipped")
                    continue
                try:
                    if fig_key in _transform_figures:
                        if _nested_cache is None:
                            descriptors = config.get(part_name, {}).get("configs", [])
                            _nested_cache = _results_to_nested_viz_dict(part_results, descriptors)
                        func(_nested_cache, output_dir)
                    else:
                        func(viz_dict, output_dir)
                except Exception as e:
                    print(f"  Warning: figure '{fig_key}' for {part_name} failed: {e}")

        # ---- Cross-part tables ----
        any_sensitivity = any(
            "sensitivity" in config.get(p, {}).get("output", {}).get("tables", [])
            for p in self._results
        )
        if any_sensitivity:
            try:
                combined = _results_to_combined_viz_dict(self._results)
                _TABLE_REGISTRY["sensitivity"](combined, output_dir)
            except Exception as e:
                print(f"  Warning: sensitivity table failed: {e}")

        # ---- Save results.json ----
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(self.to_serializable(), f, indent=2, default=str)
        print(f"  results.json: saved to {output_dir}/results.json")
