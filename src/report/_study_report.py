"""StudyReport — aggregated report across multiple format studies.

Takes raw ``Dict[str, List[SessionResult]]`` from a multi-part session
workflow and provides terminal output, tidy DataFrame export, post-hoc
visualization via ``.plot``, and JSON serialization.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from src.session._config import QuantConfig
from src.session._result import SessionResult


def _jsonify_keys(obj):
    """Recursively convert non-string dict keys (tuples) to strings for JSON."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k) if not isinstance(k, str) else k
            out[key] = _jsonify_keys(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonify_keys(item) for item in obj]
    return obj


def _dejsonify_keys(obj):
    """Recursively convert stringified tuple keys back to tuples."""
    import ast
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
                try:
                    parsed = ast.literal_eval(k)
                    if isinstance(parsed, tuple):
                        out[parsed] = _dejsonify_keys(v)
                        continue
                except (ValueError, SyntaxError):
                    pass
            out[k] = _dejsonify_keys(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_dejsonify_keys(item) for item in obj]
    return obj


class StudyReport:
    """Aggregated report across multiple format study parts.

    Takes raw ``Dict[str, List[SessionResult]]`` and provides terminal
    output, tidy DataFrame export, post-hoc visualization via ``.plot``,
    and JSON serialization.
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

    # ── to_dataframe ────────────────────────────────────────────────────

    def to_dataframe(self):
        """Flatten all results into a tidy DataFrame.

        Each row is one ``(part, config, format, layer, role)`` with columns
        for every metric collected by observers (``qsnr_db``, ``mse``,
        ``crest_factor``, ``peak``, ``rms``, ``mean``, ``std``,
        ``skewness``, ``kurtosis``, ...).

        Per-channel / per-block slices are aggregated by mean so that every
        ``(layer, role)`` pair contributes exactly one row per config.

        Returns:
            ``pandas.DataFrame``, or ``None`` if pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        rows = []
        for part_name, part_results in self._results.items():
            for r in part_results:
                obs = r.observers_data
                if not obs:
                    continue
                for layer, roles in obs.items():
                    for role, stages in roles.items():
                        # Collect all metric dicts across stages and slices
                        all_metrics = []
                        for _stage, slices in stages.items():
                            for _slice_key, metrics in slices.items():
                                all_metrics.append(metrics)

                        if not all_metrics:
                            continue

                        row = {
                            "part": part_name,
                            "config": r.name,
                            "format": r.config.w_format,
                            "layer": layer,
                            "role": role,
                        }

                        # Aggregate each metric by mean across slices
                        all_keys = set()
                        for m in all_metrics:
                            all_keys.update(m.keys())
                        for key in sorted(all_keys):
                            values = [m[key] for m in all_metrics if key in m]
                            numeric = [v for v in values if isinstance(v, (int, float))]
                            if numeric:
                                row[key] = sum(numeric) / len(numeric)

                        rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Push part/config/layer/role to the front
        leading = ["part", "config", "format", "layer", "role"]
        cols = [c for c in leading if c in df.columns] + \
               [c for c in df.columns if c not in leading]
        return df[cols]

    # ── plot accessor ────────────────────────────────────────────────────

    @property
    def plot(self) -> "StudyPlotAccessor":
        """Post-hoc visualization accessor.

        Returns a :class:`StudyPlotAccessor` with methods like
        :meth:`~StudyPlotAccessor.qsnr_comparison` and
        :meth:`~StudyPlotAccessor.crest_vs_qsnr`.
        """
        from src.report._plot import StudyPlotAccessor

        return StudyPlotAccessor(self)

    @property
    def tables(self) -> "StudyTablesAccessor":
        """Terminal table output accessor.

        Returns a :class:`StudyTablesAccessor` with methods like
        :meth:`~StudyTablesAccessor.per_layer_qsnr`.
        """
        from src.report._tables import StudyTablesAccessor

        return StudyTablesAccessor(self)

    # ── correlate_hook_observer ─────────────────────────────────────────

    def correlate_hook_observer(self, role: str = "output") -> dict:
        """Correlate accumulated (hook) QSNR with local (observer) QSNR
        across all results.

        Delegates to :meth:`SessionResult.correlate_hook_observer` for each
        result, then aggregates by config name.

        Args:
            role: Tensor role for observer data (default ``"output"``).

        Returns:
            ``{config_name: {"matched": [...], "observer_only": [...],
            "hook_only": [...]}}``. Returns empty dict if no correlation
            data is available.
        """
        result: dict = {}
        for part_results in self._results.values():
            for r in part_results:
                cfg_name = r.name or "(unnamed)"
                corr = r.correlate_hook_observer(role=role)
                if corr:
                    result[cfg_name] = corr
        return result

    # ── _avg_qsnr_mse ────────────────────────────────────────────────────

    @staticmethod
    def _avg_qsnr_mse(r: SessionResult, qsnr_type: str = "local") -> Tuple[float, float]:
        """Return ``(avg_qsnr, avg_mse)`` for a single session result.

        Args:
            r: SessionResult to extract metrics from.
            qsnr_type: ``"local"`` (default) reads ``qsnr_per_layer`` /
                ``mse_per_layer``. ``"accum"`` reads
                ``accum_qsnr_per_layer`` / ``accum_mse_per_layer``.
        """
        if qsnr_type == "accum":
            qsnr_dict = r.accum_qsnr_per_layer
            mse_dict = r.accum_mse_per_layer
        else:
            qsnr_dict = r.qsnr_per_layer
            mse_dict = r.mse_per_layer

        avg_qsnr = (
            sum(qsnr_dict.values()) / len(qsnr_dict)
            if qsnr_dict else float("nan")
        )
        avg_mse = (
            sum(mse_dict.values()) / len(mse_dict)
            if mse_dict else float("nan")
        )
        return avg_qsnr, avg_mse

    # ── summary_dataframe ───────────────────────────────────────────────

    def summary_dataframe(self, qsnr_type: str = "local"):
        """One-row-per-config summary DataFrame across all parts.

        Each row represents a single config with columns for part, config,
        format, FP32 baseline metrics, quantized metrics, delta values, and
        average QSNR/MSE. All parts are flattened into one unified table.

        Columns are ordered: ``part``, ``config``, ``format``, then metric
        columns grouped by category (``fp32_*``, ``quant_*``, ``delta_*``,
        ``avg_*``).

        Args:
            qsnr_type: ``"local"`` (default) reads per-op observer QSNR.
                ``"accum"`` reads end-to-end accumulated hook QSNR.

        Returns:
            ``pandas.DataFrame``, or ``None`` if pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        rows = []
        for part_name, part_results in self._results.items():
            for r in part_results:
                row: dict = {
                    "part": part_name,
                    "config": r.name,
                    "format": r.config.w_format if r.config else "",
                }
                if r.fp32_metrics:
                    for k, v in r.fp32_metrics.items():
                        row[f"fp32_{k}"] = v
                if r.quant_metrics:
                    for k, v in r.quant_metrics.items():
                        row[f"quant_{k}"] = v
                if r.delta:
                    for k, v in r.delta.items():
                        row[f"delta_{k}"] = v
                avg_qsnr, avg_mse = self._avg_qsnr_mse(r, qsnr_type=qsnr_type)
                row["avg_qsnr_db"] = avg_qsnr
                row["avg_mse"] = avg_mse
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        leading = ["part", "config", "format"]
        metric_cols = [c for c in df.columns if c not in leading]

        def _col_sort_key(c):
            if c.startswith("fp32_"):
                return (0, c)
            if c.startswith("quant_"):
                return (1, c)
            if c.startswith("delta_"):
                return (2, c)
            return (3, c)

        metric_cols = sorted(metric_cols, key=_col_sort_key)
        return df[leading + metric_cols]

    # ── summary ──────────────────────────────────────────────────────────

    def summary(self, qsnr_type: str = "local") -> str:
        """Return a unified comparison table across all parts as a string.

        All configs from all parts appear in a single table with FP32
        baseline, quantized metrics, deltas, and average QSNR/MSE.
        Uses pandas DataFrame display when available; falls back to a
        manually formatted table otherwise.

        Args:
            qsnr_type: ``"local"`` (default) reads per-op observer QSNR.
                ``"accum"`` reads end-to-end accumulated hook QSNR.

        Returns:
            Formatted table string.
        """
        label = "Accum QSNR" if qsnr_type == "accum" else "Avg QSNR"
        df = self.summary_dataframe(qsnr_type=qsnr_type)
        if df is None:
            # Fallback: manual formatting without pandas
            all_results = []
            for part_name, part_results in self._results.items():
                for r in part_results:
                    all_results.append((part_name, r))

            if not all_results:
                return ""

            lines = [f"{'Part':<20} {'Config':<24} {label:<12} {'Avg MSE':<12}"]
            lines.append("-" * len(lines[0]))
            for part_name, r in all_results:
                avg_qsnr, avg_mse = self._avg_qsnr_mse(r, qsnr_type=qsnr_type)
                lines.append(
                    f"{part_name:<20} {r.name:<24} "
                    f"{avg_qsnr:<12.2f} {avg_mse:<12.6f}"
                )
            return "\n".join(lines)

        if df.empty:
            return ""

        return df.to_string(index=False)

    def print_summary(self, qsnr_type: str = "local"):
        """Deprecated: use :meth:`summary` instead.

        Prints the summary table. Kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "print_summary() is deprecated, use summary() instead.",
            DeprecationWarning, stacklevel=2,
        )
        text = self.summary(qsnr_type=qsnr_type)
        if text:
            print(text)

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
                if r.accum_qsnr_per_layer:
                    entry["accum_qsnr_per_layer"] = r.accum_qsnr_per_layer
                if r.accum_mse_per_layer:
                    entry["accum_mse_per_layer"] = r.accum_mse_per_layer
                if r.observers_data:
                    entry["observers_data"] = _jsonify_keys(r.observers_data)
                if r.qsnr_by_role:
                    entry["qsnr_by_role"] = r.qsnr_by_role
                if r.mse_by_role:
                    entry["mse_by_role"] = r.mse_by_role
                serializable[part_name][r.name] = entry
        return serializable

    # ── save ────────────────────────────────────────────────────────────

    def save(self, output_dir: str):
        """Generate CSV tables, figures, and results.json.

        Produces (conditionally, based on available data):
        - ``tables/accuracy.csv`` — per-config accuracy comparison
        - ``figures/qsnr_comparison.png`` — per-layer QSNR overlay
        - ``figures/crest_vs_qsnr.png`` — crest factor vs QSNR by role
        - ``figures/outlier_analysis.png`` — outlier analysis by role
        - ``figures/per_block_qsnr.png`` — per-block QSNR stats by role
        - ``figures/correlation_heatmap.png`` — feature correlation matrix
        - ``figures/role_distribution.png`` — per-role distribution comparison
        - ``figures/pareto_qsnr.png`` / ``pareto_accuracy.png`` — Pareto frontier
        - ``figures/error_propagation.png`` — accumulated vs local QSNR decomposition
        - ``figures/accumulated_vs_local.png`` — scatter: accumulated vs local QSNR
        - ``tables/error_source.txt`` — per-layer error source diagnosis
        - ``figures/cost_decomposition.png`` — cost FLOPs breakdown
        - ``results.json`` — full serialized results
        """
        import matplotlib.pyplot as plt

        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        # ── Accuracy table ───────────────────────────────────────────
        df = self.to_dataframe()
        any_eval = any(
            r.quant_metrics is not None
            for part_results in self._results.values()
            for r in part_results
        )
        if any_eval:
            self._save_accuracy_csv(output_dir)

        # ── Per-layer QSNR CSV ──────────────────────────────────────
        any_qsnr = any(
            r.qsnr_per_layer
            for part_results in self._results.values()
            for r in part_results
        )
        if any_qsnr:
            csv_path = self.tables.save_per_layer_qsnr_csv(output_dir)
            if csv_path:
                print(f"  per_layer_qsnr.csv: saved to {csv_path}")

        # ── QSNR comparison figure ───────────────────────────────────
        if df is not None and not df.empty and "qsnr_db" in df.columns:
            try:
                fig = self.plot.qsnr_comparison()
                fig.savefig(f"{output_dir}/figures/qsnr_comparison.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: qsnr_comparison figure failed: {e}")

        # ── Crest factor vs QSNR figure ──────────────────────────────
        if df is not None and not df.empty and "crest_factor" in df.columns:
            try:
                fig = self.plot.crest_vs_qsnr()
                fig.savefig(f"{output_dir}/figures/crest_vs_qsnr.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: crest_vs_qsnr failed: {e}")

        # ── Outlier analysis figure ──────────────────────────────────
        if df is not None and not df.empty and "outlier_ratio" in df.columns:
            try:
                fig = self.plot.outlier_analysis()
                fig.savefig(f"{output_dir}/figures/outlier_analysis.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: outlier_analysis failed: {e}")

        # ── Per-block QSNR figure ────────────────────────────────────
        if df is not None and not df.empty and "qsnr_db_std" in df.columns:
            try:
                fig = self.plot.per_block_qsnr()
                fig.savefig(f"{output_dir}/figures/per_block_qsnr.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: per_block_qsnr failed: {e}")

        # ── Correlation heatmap ──────────────────────────────────────
        if df is not None and not df.empty and "skewness" in df.columns:
            try:
                fig = self.plot.correlation_heatmap()
                fig.savefig(f"{output_dir}/figures/correlation_heatmap.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: correlation_heatmap failed: {e}")

        # ── Role distribution comparison ─────────────────────────────
        if df is not None and not df.empty and "skewness" in df.columns:
            try:
                fig = self.plot.role_distribution_comparison()
                fig.savefig(f"{output_dir}/figures/role_distribution.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: role_distribution_comparison failed: {e}")

        # ── Per-layer role distribution histograms ───────────────────
        if self._results:  # uses raw observer buffers, not dataframe
            try:
                fig = self.plot.per_layer_role_histogram(k=5)
                fig.savefig(f"{output_dir}/figures/per_layer_role_histogram.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: per_layer_role_histogram failed: {e}")

        # ── Pareto frontier ──────────────────────────────────────────
        any_cost = any(
            r.cost is not None
            for part_results in self._results.values()
            for r in part_results
        )
        if any_cost:
            for metric in ("qsnr", "accuracy"):
                try:
                    fig = self.plot.pareto_frontier(metric=metric)
                    fig.savefig(f"{output_dir}/figures/pareto_{metric}.png",
                                dpi=300, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"  Warning: pareto_frontier({metric}) failed: {e}")

        # ── Cost decomposition ───────────────────────────────────────
        if any_cost:
            try:
                fig = self.plot.cost_decomposition()
                fig.savefig(f"{output_dir}/figures/cost_decomposition.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: cost_decomposition failed: {e}")

        # ── Error propagation figures ────────────────────────────────
        corr = self.correlate_hook_observer()
        any_corr = any(
            bool(info["matched"]) for info in corr.values()
        )
        if any_corr:
            for fig_name, fig_method in [
                ("error_propagation", self.plot.error_propagation),
                ("accumulated_vs_local", self.plot.accumulated_vs_local),
            ]:
                try:
                    fig = fig_method()
                    fig.savefig(
                        f"{output_dir}/figures/{fig_name}.png",
                        dpi=300, bbox_inches="tight",
                    )
                    plt.close(fig)
                    print(f"  {fig_name}.png: saved")
                except Exception as e:
                    print(f"  Warning: {fig_name} failed: {e}")

            # Error source table
            try:
                table_text = self.tables.error_source_analysis()
                if table_text and "No " not in table_text[:30]:
                    with open(
                        f"{output_dir}/tables/error_source.txt", "w"
                    ) as f:
                        f.write(table_text)
                    print(
                        "  error_source.txt: saved to "
                        f"{output_dir}/tables/error_source.txt"
                    )
            except Exception as e:
                print(f"  Warning: error_source_analysis failed: {e}")

        # ── results.json ─────────────────────────────────────────────
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(self.to_serializable(), f, indent=2, default=str)
        print(f"  results.json: saved to {output_dir}/results.json")

    def _save_accuracy_csv(self, output_dir: str):
        """Write per-config accuracy comparison CSV."""
        rows = []
        for part_name, part_results in self._results.items():
            for r in part_results:
                fp32_v = None
                q_v = None
                if r.fp32_metrics:
                    fp32_v = list(r.fp32_metrics.values())[0] if len(r.fp32_metrics) == 1 else str(r.fp32_metrics)
                if r.quant_metrics:
                    q_v = list(r.quant_metrics.values())[0] if len(r.quant_metrics) == 1 else str(r.quant_metrics)
                avg_qsnr, avg_mse = self._avg_qsnr_mse(r)
                rows.append({
                    "part": part_name,
                    "config": r.name,
                    "fp32": fp32_v if fp32_v is not None else "",
                    "quant": q_v if q_v is not None else "",
                    "avg_qsnr_db": avg_qsnr,
                    "avg_mse": avg_mse,
                })

        if not rows:
            return

        csv_path = f"{output_dir}/tables/accuracy.csv"
        keys = ["part", "config", "fp32", "quant", "avg_qsnr_db", "avg_mse"]
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

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
                    accum_qsnr_per_layer=entry.get("accum_qsnr_per_layer", {}),
                    accum_mse_per_layer=entry.get("accum_mse_per_layer", {}),
                    observers_data=_dejsonify_keys(entry.get("observers_data", {})),
                    qsnr_by_role=entry.get("qsnr_by_role", {}),
                    mse_by_role=entry.get("mse_by_role", {}),
                )
                part_results.append(result)
            results[part_name] = part_results
        return cls(results)
