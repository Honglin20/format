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
                            if values:
                                row[key] = sum(values) / len(values)

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

    # ── _avg_qsnr_mse ────────────────────────────────────────────────────

    @staticmethod
    def _avg_qsnr_mse(r: SessionResult) -> Tuple[float, float]:
        """Return ``(avg_qsnr, avg_mse)`` for a single session result."""
        avg_qsnr = (
            sum(r.qsnr_per_layer.values()) / len(r.qsnr_per_layer)
            if r.qsnr_per_layer else float("nan")
        )
        avg_mse = (
            sum(r.mse_per_layer.values()) / len(r.mse_per_layer)
            if r.mse_per_layer else float("nan")
        )
        return avg_qsnr, avg_mse

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
                avg_qsnr, avg_mse = self._avg_qsnr_mse(r)
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

    def save(self, output_dir: str):
        """Generate CSV tables, figures, and results.json.

        Produces:
        - ``tables/accuracy.csv`` — per-config accuracy comparison
        - ``figures/qsnr_comparison.png`` — per-layer QSNR overlay
        - ``figures/crest_vs_qsnr.png`` — crest factor vs QSNR scatter
          (if DistributionObserver data is present)
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
            for role in ("input", "weight"):
                try:
                    fig = self.plot.crest_vs_qsnr(role=role)
                    fig.savefig(f"{output_dir}/figures/crest_vs_qsnr_{role}.png",
                                dpi=300, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"  Warning: crest_vs_qsnr({role}) failed: {e}")

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
                )
                part_results.append(result)
            results[part_name] = part_results
        return cls(results)
