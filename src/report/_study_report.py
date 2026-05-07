"""StudyReport — aggregated report across multiple format studies.

Takes raw ``Dict[str, List[SessionResult]]`` from a multi-part session
workflow and handles all terminal output (terminal summary, CSV tables,
figures, JSON export) through a registry pattern.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.report._converters import (
    results_to_combined_viz_dict,
    results_to_nested_viz_dict,
    results_to_viz_dict,
)
from src.report._registry import get_figure_fn, get_table_fn
from src.session._config import QuantConfig
from src.session._session import SessionResult

# Figures that consume nested {format: {transform: data}} dicts
_TRANSFORM_FIGURES = frozenset({"transform_heatmap", "transform_pie", "transform_delta"})


class StudyReport:
    """Aggregated report across multiple format study parts.

    Takes raw ``Dict[str, List[SessionResult]]`` and handles all terminal
    output (terminal summary, CSV tables, figures, JSON export) through a
    registry pattern.
    """

    def __init__(self, results: Dict[str, List[SessionResult]]):
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
            print(f"\n{'=' * 70}")
            print(f"  Part: {part_name}")
            print(f"{'=' * 70}")
            if not part_results:
                print("  (no results)")
                continue

            hdr = f"{'Config':<24} {'Avg QSNR':<12} {'Avg MSE':<12} {'Acc Delta':<12}"
            print(f"  {hdr}")
            print(f"  {'-' * len(hdr)}")

            for r in part_results:
                avg_qsnr = (
                    sum(r.qsnr_per_layer.values()) / len(r.qsnr_per_layer)
                    if r.qsnr_per_layer else float("nan")
                )
                avg_mse = (
                    sum(r.mse_per_layer.values()) / len(r.mse_per_layer)
                    if r.mse_per_layer else float("nan")
                )
                delta_str = ""
                if r.delta:
                    vals = [f"{k}={v:+.4f}" for k, v in r.delta.items()]
                    delta_str = ", ".join(vals)
                print(
                    f"  {r.name:<24} "
                    f"{avg_qsnr:<12.2f} "
                    f"{avg_mse:<12.6f} "
                    f"{delta_str}"
                )

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
                If not given, all parts default to
                ``{"tables": ["accuracy"], "figures": ["qsnr"]}``.
        """
        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        if config is None:
            config = {}

        # ── Per-part tables and figures ──────────────────────────────
        for part_name, part_results in self._results.items():
            part_output = config.get(part_name, {}).get("output", {})
            table_keys = part_output.get("tables", ["accuracy"])
            figure_keys = part_output.get("figures", ["qsnr"])

            viz_dict = results_to_viz_dict(part_results)

            # Nested cache built lazily for transform figures
            _nested_cache: Optional[dict] = None

            for table_key in table_keys:
                if table_key == "sensitivity":
                    continue  # Handled as cross-part below
                try:
                    func = get_table_fn(table_key)
                    func(
                        viz_dict, output_dir,
                        title=part_name,
                        filename=f"{part_name}_{table_key}.csv",
                    )
                except KeyError:
                    print(
                        f"  Warning: unknown table '{table_key}' "
                        f"for {part_name}, skipped"
                    )
                except Exception as e:
                    print(
                        f"  Warning: table '{table_key}' "
                        f"for {part_name} failed: {e}"
                    )

            for fig_key in figure_keys:
                try:
                    func = get_figure_fn(fig_key)
                    if fig_key in _TRANSFORM_FIGURES:
                        if _nested_cache is None:
                            descriptors = config.get(part_name, {}).get(
                                "configs", []
                            )
                            _nested_cache = results_to_nested_viz_dict(
                                part_results, descriptors
                            )
                        func(_nested_cache, output_dir)
                    else:
                        func(viz_dict, output_dir)
                except KeyError:
                    print(
                        f"  Warning: unknown figure '{fig_key}' "
                        f"for {part_name}, skipped"
                    )
                except Exception as e:
                    print(
                        f"  Warning: figure '{fig_key}' "
                        f"for {part_name} failed: {e}"
                    )

        # ── Cross-part tables ────────────────────────────────────────
        any_sensitivity = any(
            "sensitivity" in config.get(p, {}).get("output", {}).get("tables", [])
            for p in self._results
        )
        if any_sensitivity:
            try:
                combined = results_to_combined_viz_dict(self._results)
                get_table_fn("sensitivity")(combined, output_dir)
            except Exception as e:
                print(f"  Warning: sensitivity table failed: {e}")

        # ── Save results.json ────────────────────────────────────────
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(self.to_serializable(), f, indent=2, default=str)
        print(f"  results.json: saved to {output_dir}/results.json")

    # ── from_file ───────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str) -> StudyReport:
        """Reload a ``StudyReport`` from a saved ``results.json`` file.

        Args:
            path: Directory containing ``results.json``.

        Returns:
            A new ``StudyReport`` populated with the saved results.

        Raises:
            FileNotFoundError: If ``<path>/results.json`` does not exist.
        """
        json_path = os.path.join(path, "results.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"results.json not found at {json_path}"
            )
        with open(json_path) as f:
            data = json.load(f)

        results: Dict[str, List[SessionResult]] = {}
        for part_name, part_data in data.items():
            part_results: List[SessionResult] = []
            for name, entry in part_data.items():
                result = SessionResult(
                    name=name,
                    config=QuantConfig(name=name),
                    quant_metrics=entry.get("accuracy"),
                    fp32_metrics=entry.get("fp32_accuracy"),
                    delta=entry.get("delta"),
                    qsnr_per_layer=entry.get("qsnr_per_layer", {}),
                    mse_per_layer=entry.get("mse_per_layer", {}),
                )
                part_results.append(result)
            results[part_name] = part_results
        return cls(results)
