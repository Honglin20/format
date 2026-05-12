"""SessionResult dataclass and its accessor methods.

SessionResult is the output of running one Session (one QuantConfig).
It holds accuracy deltas, per-layer QSNR/MSE metrics (both local observer
and accumulated hook paths), raw observer data, and cost estimates, plus
user-facing accessor methods for display.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.session._config import QuantConfig

if TYPE_CHECKING:
    from src.report._session_tables import SessionTablesAccessor
    from src.report._plot import SessionPlotAccessor


@dataclass
class SessionResult:
    """Result of running a single Session (one QuantConfig).

    Replaces pipeline/runner.py:ExperimentResult with the addition of
    the config field, sq_transforms cache, and user-facing accessor methods.
    """

    name: str
    config: QuantConfig
    fp32_metrics: Optional[Dict[str, float]] = None
    quant_metrics: Optional[Dict[str, float]] = None
    delta: Optional[Dict[str, float]] = None
    qsnr_per_layer: Dict[str, float] = field(default_factory=dict)
    mse_per_layer: Dict[str, float] = field(default_factory=dict)
    qsnr_by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """QSNR per role per layer: ``{role: {layer: qsnr_db}}``."""
    mse_by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """MSE per role per layer: ``{role: {layer: mse}}``."""
    accum_qsnr_per_layer: Dict[str, float] = field(default_factory=dict)
    accum_mse_per_layer: Dict[str, float] = field(default_factory=dict)
    observers_data: Dict[str, Any] = field(default_factory=dict)
    cost: Any = None
    cost_fp32: Any = None
    sq_transforms: Optional[Dict[str, Any]] = None
    sq_distrib_comparison: Optional[Any] = None
    """SmoothQuant pre/post distribution comparison.

    Populated when ``transform="smoothquant"`` and
    ``"smoothquant_distrib"`` is in the analyze outputs.  Call
    :meth:`sq_comparison` for a terminal table, or use
    :func:`src.viz.figures.smoothquant_distrib_comparison` for a plot.
    """

    # ------------------------------------------------------------------
    # Accessor properties
    # ------------------------------------------------------------------

    @property
    def sq_comparison(self) -> str:
        """Human-readable SmoothQuant distribution comparison table.

        Shortcut that delegates to
        :meth:`SmoothQuantDistribComparison.summary_table`.

        Raises:
            RuntimeError: If ``sq_distrib_comparison`` is ``None``
                (not collected or transform is not smoothquant).
        """
        if self.sq_distrib_comparison is None:
            raise RuntimeError(
                "sq_distrib_comparison is not available. "
                "Run session.analyze(outputs=['smoothquant_distrib']) "
                "with transform='smoothquant'."
            )
        return self.sq_distrib_comparison.summary_table()

    @property
    def tables(self) -> "SessionTablesAccessor":
        """Terminal table output accessor.

        Returns a :class:`SessionTablesAccessor` with methods like
        :meth:`~SessionTablesAccessor.error_source_analysis` and
        :meth:`~SessionTablesAccessor.per_layer_qsnr`.
        """
        from src.report._session_tables import SessionTablesAccessor

        return SessionTablesAccessor(self)

    @property
    def report(self) -> "AnalysisReport":
        """Distribution analysis accessor.

        Returns an :class:`AnalysisReport` wrapping this result's
        ``observers_data``, enabling taxonomy, profile, and sensitivity
        analysis.

        Example::

            result.report.taxonomy.classify()
            DistributionProfile.from_report(result.report)
            ErrorByDistribution(result.report)
        """
        from src.analysis.report import AnalysisReport

        return AnalysisReport(self.observers_data)

    @property
    def characterize(self) -> "DistributionDiagnosis":
        """Distribution-based quantisation failure diagnosis accessor.

        Returns a :class:`DistributionDiagnosis` that links distribution
        features (crest factor, outlier ratio, kurtosis, etc.) to known
        quantisation failure modes.

        Example::

            print(result.characterize.profile("layer3.linear", role="weight"))
            print(result.characterize.causal_analysis())
        """
        from src.analysis._distribution_diagnosis import DistributionDiagnosis

        return DistributionDiagnosis(self)

    @property
    def diagnose(self) -> "ErrorProvenance":
        """Systematic error provenance accessor.

        Returns an :class:`ErrorProvenance` with per-role per-layer
        QSNR attribution, top-K worst layers, and error source analysis.

        Example::

            print(result.diagnose.summary())
            print(result.diagnose.per_role_table())
            for name, qsnr in result.diagnose.top_k(5, role="weight"):
                print(f"{name}: {qsnr:.1f} dB")
        """
        from src.analysis._error_provenance import ErrorProvenance

        return ErrorProvenance(self)

    @property
    def plan(self) -> "InterventionPlanner":
        """Intervention planner accessor.

        Returns an :class:`InterventionPlanner` that generates per-layer
        precision-boost and transform plans from QSNR data.

        Example::

            plan = result.plan.top_k_boost(k=5, role="weight", target_bits=8)
            print(plan.explain())
        """
        from src.analysis._intervention import InterventionPlanner

        return InterventionPlanner(self)

    @property
    def intervention(self) -> "InterventionAccessor":
        """Intervention application and comparison accessor.

        Returns an :class:`InterventionAccessor` that can apply an
        InterventionPlan to a new Session and compare results.

        Example::

            plan = result.plan.top_k_boost(k=5)
            comparison = result.intervention.compare(model, data, plan)
            print(comparison.summary())
        """
        from src.analysis._intervention_accessor import InterventionAccessor

        return InterventionAccessor(self)

    @property
    def plot(self) -> "SessionPlotAccessor":
        """Post-hoc visualization accessor.

        Returns a :class:`SessionPlotAccessor` with methods like
        :meth:`~SessionPlotAccessor.qsnr_comparison` and
        :meth:`~SessionPlotAccessor.error_propagation`.

        Example::

            result.plot.qsnr_comparison()
            result.plot.error_propagation(role="output")
        """
        from src.report._plot import SessionPlotAccessor

        return SessionPlotAccessor(self)

    # ------------------------------------------------------------------
    # Accessor methods
    # ------------------------------------------------------------------

    def correlate_hook_observer(self, role: str = "output") -> dict:
        """Correlate accumulated (hook) QSNR with local (observer) QSNR.

        Matches observer keys to hook keys by prefix: ``obs_key == hook_key``
        or ``obs_key.startswith(hook_key + ".")``. For each matched hook key,
        the minimum local QSNR across sub-slices is taken.

        Args:
            role: Tensor role for observer data (default ``"output"``).

        Returns:
            ``{"matched": [(hook_key, accum_qsnr, local_qsnr), ...],
            "observer_only": [(obs_key, local_qsnr), ...],
            "hook_only": [(hook_key, accum_qsnr), ...]}``.
            Returns empty dict if no hook or local data.
        """
        accum = self.accum_qsnr_per_layer
        if not accum:
            return {}

        local, _ = self.qsnr_per_role(role=role)
        if not local:
            return {}

        hook_keys = set(accum.keys())
        obs_by_hook: dict = {}
        unmatched_obs: list = []

        for obs_key, local_qsnr in sorted(local.items()):
            matched_hk = None
            for hk in hook_keys:
                if obs_key == hk or obs_key.startswith(hk + "."):
                    matched_hk = hk
                    break
            if matched_hk:
                obs_by_hook.setdefault(matched_hk, []).append(
                    (obs_key, local_qsnr)
                )
            else:
                unmatched_obs.append((obs_key, local_qsnr))

        matched_list = []
        for hk in sorted(hook_keys):
            if hk in obs_by_hook:
                min_local = min(v for _, v in obs_by_hook[hk])
                matched_list.append((hk, accum[hk], min_local))

        matched_hks = set(hk for hk, _, _ in matched_list)
        hook_only = [
            (hk, accum[hk])
            for hk in sorted(hook_keys)
            if hk not in matched_hks
        ]

        return {
            "matched": matched_list,
            "observer_only": unmatched_obs,
            "hook_only": hook_only,
        }

    def qsnr_per_role(
        self, role: str = "output"
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Extract per-layer QSNR and MSE for a given role.

        Reads from pre-computed ``qsnr_by_role`` / ``mse_by_role`` dicts
        (populated during ``Session.analyze()``).  Falls back to iterating
        ``observers_data`` directly if the cached dicts are empty (e.g.
        when loading from a serialized file that predates multi-role support).

        Args:
            role: Tensor role to extract (``"input"`` / ``"weight"`` /
                ``"output"`` / ``"bias"``). Default ``"output"``.

        Returns:
            ``(qsnr_per_layer, mse_per_layer)`` — each is ``Dict[str, float]``.
        """
        if self.qsnr_by_role and self.mse_by_role:
            return self.qsnr_by_role.get(role, {}), self.mse_by_role.get(role, {})

        # Fallback: iterate observers_data directly (backward compat)
        qsnr_map: Dict[str, float] = {}
        mse_map: Dict[str, float] = {}
        for layer, roles in self.observers_data.items():
            stages = roles.get(role)
            if stages is None:
                continue
            for _stage, slices in stages.items():
                for _slice_key, metrics in slices.items():
                    if "qsnr_db" in metrics:
                        v = metrics["qsnr_db"]
                        if v is not None and v == v and v != float("-inf"):
                            prev = qsnr_map.get(layer)
                            if prev is None or v < prev:
                                qsnr_map[layer] = v
                    if "mse" in metrics:
                        mse_map[layer] = max(
                            mse_map.get(layer, 0.0),
                            metrics["mse"],
                        )
        return qsnr_map, mse_map

    def summary(self, qsnr_type: str = "local") -> str:
        """One-line human-readable summary of the quantization result.

        Args:
            qsnr_type: ``"local"`` (default) reads per-op observer QSNR.
                ``"accum"`` reads end-to-end accumulated hook QSNR.

        Example::

            >>> print(result.summary())
            Config: int8-pc | loss: fp32=0.1234 quant=0.1456 | avg QSNR=34.2 dB
        """
        parts = [f"Config: {self.name or '(unnamed)'}"]

        if self.fp32_metrics and self.quant_metrics:
            metric_strs = []
            for k in self.fp32_metrics:
                fp32_v = self.fp32_metrics[k]
                q_v = self.quant_metrics.get(k, float("nan"))
                metric_strs.append(f"{k}: fp32={fp32_v:.4f} quant={q_v:.4f}")
            if metric_strs:
                parts.append(" | ".join(metric_strs))

        qsnr_dict = self.accum_qsnr_per_layer if qsnr_type == "accum" else self.qsnr_per_layer
        if qsnr_dict:
            finite = [v for v in qsnr_dict.values() if v is not None and v == v and v != float('inf') and v != float('-inf')]
            if finite:
                avg_qsnr = sum(finite) / len(finite)
                label = "accum QSNR" if qsnr_type == "accum" else "avg QSNR"
                parts.append(f"{label}={avg_qsnr:.1f} dB")
            else:
                parts.append("avg QSNR=N/A")

        if self.delta:
            delta_strs = []
            for k, v in self.delta.items():
                delta_strs.append(f"Δ{k}={v:+.4f}")
            parts.append(" ".join(delta_strs))

        return " | ".join(parts)

    def accuracy_table(self) -> str:
        """Formatted accuracy comparison table.

        Example::

            >>> print(result.accuracy_table())
            Metric    FP32      Quant     Δ
            ------------------------------------
            loss      0.1234    0.1456    +0.0222
            acc       0.9500    0.9300    -0.0200
        """
        if not self.fp32_metrics:
            return (
                "(no accuracy metrics — pass eval_fn to run() or evaluate())\n"
                "eval_fn signature: (model, data) -> Dict[str, float]\n"
                "Example: def eval_fn(m, d): return {'loss': sum(m(b).sum() for b in d).item()}"
            )

        lines = []
        header = f"{'Metric':<12} {'FP32':<10} {'Quant':<10} {'Δ':<10}"
        lines.append(header)
        lines.append("-" * len(header))

        for k in self.fp32_metrics:
            fp32_v = self.fp32_metrics[k]
            q_v = self.quant_metrics.get(k, float("nan")) if self.quant_metrics else float("nan")
            d_v = self.delta.get(k, float("nan")) if self.delta else float("nan")

            lines.append(
                f"{k:<12} {fp32_v:<10.4f} {q_v:<10.4f} {d_v:<+10.4f}"
            )

        return "\n".join(lines)

    def top_k_qsnr(self, k: int = 10, reverse: bool = False, qsnr_type: str = "local") -> List[Tuple[str, float]]:
        """Top-k layers by QSNR.

        Args:
            k: Number of layers to return (default 10).
            reverse: If False (default), returns the k layers with the **lowest**
                QSNR (worst quality), sorted ascending. If True, returns the k
                layers with the **highest** QSNR (best quality), sorted descending.
            qsnr_type: ``"local"`` (default) reads per-op observer QSNR.
                ``"accum"`` reads end-to-end accumulated hook QSNR.

        Returns:
            List of ``(layer_name, qsnr_db)`` tuples.

        Example::

            >>> # Worst 3 layers
            >>> for name, qsnr in result.top_k_qsnr(3):
            ...     print(f"{name}: {qsnr:.1f} dB")
            layer1.linear: 12.3 dB
            layer2.conv: 18.7 dB
            layer3.norm: 25.1 dB

            >>> # Best 3 layers
            >>> for name, qsnr in result.top_k_qsnr(3, reverse=True):
            ...     print(f"{name}: {qsnr:.1f} dB")
            layer5.embed: 52.1 dB
            layer4.norm: 48.3 dB
            layer3.norm: 45.7 dB
        """
        qsnr_dict = self.accum_qsnr_per_layer if qsnr_type == "accum" else self.qsnr_per_layer
        sorted_layers = sorted(qsnr_dict.items(), key=lambda x: x[1], reverse=reverse)
        return sorted_layers[:k]

    def layer_report(self) -> "Any":
        """Per-layer DataFrame with local and accumulated QSNR/MSE.

        Columns: ``layer``, ``qsnr_db`` (local observer QSNR), ``mse`` (local),
        ``accum_qsnr_db`` (accumulated hook QSNR), ``accum_mse`` (accumulated).
        Accumulated columns are omitted when no hook data is present.

        Returns:
            ``pandas.DataFrame``, or ``None`` if pandas is not available.

        Example::

            >>> df = result.layer_report()
            >>> print(df.sort_values("qsnr_db").head(5))
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        all_layers = (
            set(self.qsnr_per_layer.keys())
            | set(self.mse_per_layer.keys())
            | set(self.accum_qsnr_per_layer.keys())
            | set(self.accum_mse_per_layer.keys())
        )
        has_accum = bool(self.accum_qsnr_per_layer or self.accum_mse_per_layer)

        rows = []
        for layer in sorted(all_layers):
            row = {
                "layer": layer,
                "qsnr_db": self.qsnr_per_layer.get(layer),
                "mse": self.mse_per_layer.get(layer),
            }
            if has_accum:
                row["accum_qsnr_db"] = self.accum_qsnr_per_layer.get(layer)
                row["accum_mse"] = self.accum_mse_per_layer.get(layer)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Serialization & save
    # ------------------------------------------------------------------

    def to_serializable(self) -> dict:
        """Return a JSON-serializable dict of this result."""
        entry: Dict[str, Any] = {"name": self.name}
        if self.config:
            entry["config"] = self.config.name
        if self.quant_metrics is not None:
            entry["accuracy"] = self.quant_metrics
        if self.fp32_metrics is not None:
            entry["fp32_accuracy"] = self.fp32_metrics
        if self.delta is not None:
            entry["delta"] = self.delta
        if self.qsnr_per_layer:
            entry["qsnr_per_layer"] = self.qsnr_per_layer
        if self.mse_per_layer:
            entry["mse_per_layer"] = self.mse_per_layer
        if self.accum_qsnr_per_layer:
            entry["accum_qsnr_per_layer"] = self.accum_qsnr_per_layer
        if self.accum_mse_per_layer:
            entry["accum_mse_per_layer"] = self.accum_mse_per_layer
        return entry

    def save(self, output_dir: str) -> None:
        """Save single-config analysis to ``output_dir``.

        Generates (conditionally, based on available data):
        - ``results.json`` — full serialized result
        - ``tables/accuracy.txt`` — FP32 vs Quant accuracy comparison
        - ``tables/per_layer_qsnr.csv`` — per-layer QSNR
        - ``tables/error_source.txt`` — error source diagnosis
        - ``figures/qsnr_comparison.png`` — per-layer QSNR bar chart
        - ``figures/error_propagation.png`` — accumulated vs local QSNR
        - ``figures/accumulated_vs_local.png`` — scatter plot
        """
        import json

        import matplotlib.pyplot as plt

        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        # Accuracy table
        if self.fp32_metrics:
            txt = self.accuracy_table()
            with open(f"{output_dir}/tables/accuracy.txt", "w") as f:
                f.write(txt + "\n")
            print(f"  tables/accuracy.txt: saved")

        # Per-layer QSNR CSV
        if self.qsnr_per_layer:
            csv_path = f"{output_dir}/tables/per_layer_qsnr.csv"
            with open(csv_path, "w") as f:
                f.write("Layer,QSNR_dB\n")
                for layer, qsnr in sorted(self.qsnr_per_layer.items(),
                                          key=lambda x: x[1]):
                    f.write(f"{layer},{qsnr:.4f}\n")
            print(f"  per_layer_qsnr.csv: saved to {csv_path}")

        # Figures — QSNR bar chart (always if qsnr data)
        if self.qsnr_per_layer:
            try:
                fig = self.plot.qsnr_comparison()
                fig.savefig(f"{output_dir}/figures/qsnr_comparison.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  qsnr_comparison.png: saved")
            except Exception as e:
                print(f"  Warning: qsnr_comparison failed: {e}")

        # Error propagation figures
        has_accum = any(
            v == v and v != float("inf") and v != float("-inf")
            for v in self.accum_qsnr_per_layer.values()
        )
        if has_accum:
            for fig_name, fig_method in [
                ("error_propagation", lambda: self.plot.error_propagation()),
                ("accumulated_vs_local", lambda: self.plot.accumulated_vs_local()),
            ]:
                try:
                    fig = fig_method()
                    fig.savefig(f"{output_dir}/figures/{fig_name}.png",
                                dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  {fig_name}.png: saved")
                except Exception as e:
                    print(f"  Warning: {fig_name} failed: {e}")

            # Error source table
            try:
                text = self.tables.error_source_analysis()
                if text and "No " not in text[:30]:
                    with open(f"{output_dir}/tables/error_source.txt", "w") as f:
                        f.write(text)
                    print(f"  error_source.txt: saved")
            except Exception as e:
                print(f"  Warning: error_source_analysis failed: {e}")

        # results.json
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(self.to_serializable(), f, indent=2, default=str)
        print(f"  results.json: saved to {output_dir}/results.json")
