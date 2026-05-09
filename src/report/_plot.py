"""StudyPlotAccessor — post-hoc visualization on StudyReport."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.report._study_report import StudyReport

_VALID_PLOT_ROLES = frozenset({"input", "weight", "output", "bias"})


class StudyPlotAccessor:
    """Post-hoc visualization methods on :class:`StudyReport`.

    Usage::

        report = Study(configs, model).run(data, eval_fn)
        report.plot.qsnr_comparison()
        report.plot.crest_vs_qsnr(role="input")
    """

    def __init__(self, report: "StudyReport"):
        self._report = report

    # ── QSNR comparison ─────────────────────────────────────────────────

    def qsnr_comparison(self) -> plt.Figure:
        """Per-layer QSNR overlay for all configs.

        One line per config. Layers are aligned by name across configs so
        that the same layer in different configs shares an x-position.

        Returns:
            matplotlib Figure. The caller is responsible for ``show()`` or
            ``savefig()``.

        Raises:
            ValueError: If ``qsnr_db`` is not available (QSNRObserver not active).
        """
        df = self._report.to_dataframe()
        if df is None or df.empty or "qsnr_db" not in df.columns:
            raise ValueError(
                "QSNR data not available. "
                "Add QSNRObserver() to session observers before the analysis pass."
            )

        # Collect union of all layer names across configs
        all_layers = list(dict.fromkeys(df["layer"]))  # preserve order, deduplicate
        configs = sorted(df["config"].unique())

        fig, ax = plt.subplots(figsize=(12, 6))
        x_positions = list(range(len(all_layers)))

        for cfg in configs:
            cfg_df = df[df["config"] == cfg]
            # Average QSNR across roles per layer, then map to layer order
            per_layer = cfg_df.groupby("layer")["qsnr_db"].mean()
            values = [per_layer.get(l, float("nan")) for l in all_layers]
            ax.plot(x_positions, values, marker="o", label=cfg, linewidth=2)

        short_names = [_short_layer_name(l) for l in all_layers]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("QSNR (dB)")
        ax.set_title("QSNR per Layer")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ── Crest factor vs QSNR scatter ────────────────────────────────────

    def crest_vs_qsnr(self, role: str = "input") -> plt.Figure:
        """Crest factor vs QSNR scatter, one point per (config, layer).

        Args:
            role: Tensor role to plot. Must be one of
                ``"input"``, ``"weight"``, ``"output"``, ``"bias"``.

        Returns:
            matplotlib Figure.

        Raises:
            ValueError: If *role* is invalid, or if required observer data
                (``crest_factor`` from DistributionObserver, ``qsnr_db`` from
                QSNRObserver) is not present.
        """
        if role not in _VALID_PLOT_ROLES:
            raise ValueError(
                f"Invalid role {role!r}. Must be one of "
                f"{sorted(_VALID_PLOT_ROLES)}."
            )

        df = self._report.to_dataframe()
        needed = {"crest_factor", "qsnr_db"}
        if df is None or df.empty or not needed.issubset(df.columns):
            missing = needed - (set(df.columns) if df is not None and not df.empty else set())
            raise ValueError(
                f"Required metrics not available: {sorted(missing)}. "
                "Ensure DistributionObserver and QSNRObserver are both active "
                "in session observers during the analysis pass."
            )

        role_df = df[df["role"] == role]
        if role_df.empty:
            available = sorted(df["role"].unique()) if "role" in df.columns else []
            raise ValueError(
                f"No data for role {role!r}. "
                f"Roles present in the report: {available or '(none)'}."
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        configs = sorted(role_df["config"].unique())

        for cfg in configs:
            cfg_df = role_df[role_df["config"] == cfg]
            ax.scatter(cfg_df["crest_factor"], cfg_df["qsnr_db"],
                       label=cfg, alpha=0.7, s=40)

        ax.set_xlabel("Crest Factor (peak / RMS)")
        ax.set_ylabel("QSNR (dB)")
        ax.set_title(f"Crest Factor vs QSNR — {role}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


    # ── Outlier analysis ───────────────────────────────────────────────

    def outlier_analysis(self, role: str = "input") -> plt.Figure:
        """Outlier ratio per-layer bar chart + outlier vs QSNR scatter.

        Args:
            role: Tensor role to plot. Must be one of
                ``"input"``, ``"weight"``, ``"output"``, ``"bias"``.

        Returns:
            matplotlib Figure with two panels.

        Raises:
            ValueError: If *role* is invalid, or if ``outlier_ratio`` from
                DistributionObserver is not available.
        """
        if role not in _VALID_PLOT_ROLES:
            raise ValueError(
                f"Invalid role {role!r}. Must be one of "
                f"{sorted(_VALID_PLOT_ROLES)}."
            )

        df = self._report.to_dataframe()
        if df is None or df.empty or "outlier_ratio" not in df.columns:
            raise ValueError(
                "Outlier ratio data not available. "
                "Ensure DistributionObserver is active during the analysis pass."
            )

        role_df = df[df["role"] == role]
        if role_df.empty:
            available = sorted(df["role"].unique()) if "role" in df.columns else []
            raise ValueError(
                f"No data for role {role!r}. "
                f"Roles present in the report: {available or '(none)'}."
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: per-layer outlier_ratio bar chart
        layers = list(dict.fromkeys(role_df["layer"]))
        configs = sorted(role_df["config"].unique())
        x = np.arange(len(layers))
        width = 0.8 / max(len(configs), 1)

        for i, cfg in enumerate(configs):
            cfg_df = role_df[role_df["config"] == cfg]
            per_layer = cfg_df.groupby("layer")["outlier_ratio"].mean()
            values = [per_layer.get(l, 0) for l in layers]
            ax1.bar(x + i * width, values, width, label=cfg, alpha=0.7)

        ax1.set_xticks(x + width * (len(configs) - 1) / 2)
        ax1.set_xticklabels([_short_layer_name(l) for l in layers],
                            rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("Outlier Ratio")
        ax1.set_title(f"Outlier Ratio per Layer — {role}")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: outlier_ratio vs QSNR scatter
        has_qsnr = "qsnr_db" in df.columns
        for cfg in configs:
            cfg_df = role_df[role_df["config"] == cfg]
            if has_qsnr:
                ax2.scatter(cfg_df["outlier_ratio"], cfg_df["qsnr_db"],
                           label=cfg, alpha=0.7, s=40)
            else:
                ax2.scatter(cfg_df["outlier_ratio"],
                           [0] * len(cfg_df), label=cfg, alpha=0.7, s=40)

        ax2.set_xlabel("Outlier Ratio")
        ax2.set_ylabel("QSNR (dB)" if has_qsnr else "(no QSNR)")
        ax2.set_title(f"Outlier Ratio vs QSNR — {role}")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ── Per-block QSNR distribution ─────────────────────────────────────

    def per_block_qsnr(self, role: str = "input") -> plt.Figure:
        """Per-block QSNR statistics (std + min-vs-mean scatter).

        Uses ``qsnr_db_std``, ``qsnr_db_min``, ``qsnr_db_max`` collected by
        QSNRObserver in per-block mode.

        Args:
            role: Tensor role to plot.

        Returns:
            matplotlib Figure with two panels.

        Raises:
            ValueError: If per-block QSNR statistics are not available
                (QSNRObserver not active with per-block granularity).
        """
        if role not in _VALID_PLOT_ROLES:
            raise ValueError(
                f"Invalid role {role!r}. Must be one of "
                f"{sorted(_VALID_PLOT_ROLES)}."
            )

        df = self._report.to_dataframe()
        needed = {"qsnr_db_std", "qsnr_db_min", "qsnr_db_max"}
        if df is None or df.empty:
            raise ValueError(
                "Per-block QSNR data not available. "
                "Ensure QSNRObserver is active with per-block granularity "
                "to collect per-block statistics (qsnr_db_std/min/max)."
            )
        available_cols = set(df.columns)
        if not needed.issubset(available_cols):
            missing = needed - available_cols
            raise ValueError(
                f"Per-block QSNR statistics not available: {sorted(missing)}. "
                "QSNRObserver only collects qsnr_db_std/min/max in per-block "
                "mode. Use per_tensor/per_channel granularity or ensure "
                "per-block analysis was run."
            )

        role_df = df[df["role"] == role]
        if role_df.empty:
            available = sorted(df["role"].unique()) if "role" in df.columns else []
            raise ValueError(
                f"No data for role {role!r}. "
                f"Roles present in the report: {available or '(none)'}."
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        layers = list(dict.fromkeys(role_df["layer"]))
        configs = sorted(role_df["config"].unique())

        # Panel 1: per-layer qsnr_db_std box
        layer_std_data = {}
        for layer in layers:
            ldf = role_df[role_df["layer"] == layer]
            if "qsnr_db_std" in ldf.columns:
                vals = ldf["qsnr_db_std"].dropna().tolist()
                if vals:
                    layer_std_data[layer] = vals

        if layer_std_data:
            positions = range(len(layer_std_data))
            ax1.boxplot(layer_std_data.values(), positions=positions,
                        widths=0.6, patch_artist=True)
            ax1.set_xticks(positions)
            ax1.set_xticklabels([_short_layer_name(l) for l in layer_std_data],
                                rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("QSNR Std Dev (dB)")
        ax1.set_title(f"Per-Block QSNR Std Dev — {role}")
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: qsnr_db_min vs qsnr_db mean scatter
        for cfg in configs:
            cfg_df = role_df[role_df["config"] == cfg]
            if "qsnr_db_min" in cfg_df.columns and "qsnr_db" in cfg_df.columns:
                ax2.scatter(cfg_df["qsnr_db"], cfg_df["qsnr_db_min"],
                           label=cfg, alpha=0.7, s=40)

        ax2.set_xlabel("Mean QSNR (dB)")
        ax2.set_ylabel("Min QSNR (dB)")
        ax2.set_title(f"Min vs Mean QSNR per Block — {role}")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # Diagonal reference line
        if "qsnr_db" in role_df.columns:
            vals = role_df["qsnr_db"].dropna()
            if not vals.empty:
                lo, hi = vals.min(), vals.max()
                ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.5)

        fig.tight_layout()
        return fig

    # ── Pareto frontier ─────────────────────────────────────────────────

    def pareto_frontier(self, metric: str = "qsnr") -> plt.Figure:
        """Pareto frontier: quality vs cost trade-off scatter.

        Plots accuracy or average QSNR against bit-width, latency, and
        memory for each config, colour-coded by config name.

        Args:
            metric: Quality metric for the y-axis. ``"qsnr"`` (default) or
                ``"accuracy"``.

        Returns:
            matplotlib Figure with up to three panels.

        Raises:
            ValueError: If no results with cost data are available.
        """
        if metric not in ("qsnr", "accuracy"):
            raise ValueError(
                f"Invalid metric {metric!r}. Must be 'qsnr' or 'accuracy'."
            )

        # Build per-config data from raw results
        points = []
        for part_name, part_results in self._report._results.items():
            for r in part_results:
                avg_qsnr, avg_mse = self._report._avg_qsnr_mse(r)
                acc = None
                if r.quant_metrics and len(r.quant_metrics) == 1:
                    acc = list(r.quant_metrics.values())[0]
                elif r.quant_metrics:
                    acc = sum(r.quant_metrics.values()) / max(len(r.quant_metrics), 1)

                # Derive bit width from format names
                w_bits = _fmt_bits(r.config.w_format)
                a_bits = _fmt_bits(r.config.a_format)
                avg_bits = (w_bits + a_bits) / 2

                # Cost data
                latency = None
                memory_mb = None
                if r.cost is not None:
                    latency = getattr(r.cost, "total_latency_us", None)
                    memory_mb = getattr(r.cost, "total_memory_bytes", 0) / 1e6

                points.append({
                    "part": part_name,
                    "config": r.name,
                    "avg_qsnr": avg_qsnr,
                    "accuracy": acc,
                    "avg_bits": avg_bits,
                    "latency_us": latency,
                    "memory_mb": memory_mb,
                })

        if not points:
            raise ValueError(
                "No results available for Pareto frontier. "
                "Run at least one session with cost estimation enabled."
            )

        y_key = "accuracy" if metric == "accuracy" else "avg_qsnr"
        valid = [p for p in points if p[y_key] is not None
                 and not (isinstance(p[y_key], float) and math.isnan(p[y_key]))]
        if not valid:
            raise ValueError(
                f"No valid {metric} values available for Pareto frontier."
            )

        has_latency = any(p["latency_us"] is not None for p in valid)
        has_memory = any(p["memory_mb"] is not None and p["memory_mb"] > 0 for p in valid)

        n_panels = 1 + has_latency + has_memory
        if n_panels == 1:
            fig, ax_bits = plt.subplots(figsize=(7, 5))
            axes = [ax_bits]
        else:
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
            if n_panels == 1:
                axes = [axes]

        ax_idx = 0
        configs = sorted(set(p["config"] for p in valid))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 1)))

        # Panel: quality vs bit-width
        ax = axes[ax_idx]
        for i, cfg in enumerate(configs):
            cfg_pts = [p for p in valid if p["config"] == cfg]
            xs = [p["avg_bits"] for p in cfg_pts]
            ys = [p[y_key] for p in cfg_pts]
            ax.scatter(xs, ys, label=cfg, color=colors[i], alpha=0.7, s=60)
        ax.set_xlabel("Avg Bit Width")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} vs Bit Width")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

        # Panel: quality vs latency
        if has_latency:
            ax = axes[ax_idx]
            for i, cfg in enumerate(configs):
                cfg_pts = [p for p in valid if p["config"] == cfg
                          and p["latency_us"] is not None]
                xs = [p["latency_us"] for p in cfg_pts]
                ys = [p[y_key] for p in cfg_pts]
                ax.scatter(xs, ys, label=cfg, color=colors[i], alpha=0.7, s=60)
            ax.set_xlabel("Latency (us)")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} vs Latency")
            ax.grid(True, alpha=0.3)
            ax_idx += 1

        # Panel: quality vs memory
        if has_memory:
            ax = axes[ax_idx]
            for i, cfg in enumerate(configs):
                cfg_pts = [p for p in valid if p["config"] == cfg
                          and p["memory_mb"] is not None and p["memory_mb"] > 0]
                xs = [p["memory_mb"] for p in cfg_pts]
                ys = [p[y_key] for p in cfg_pts]
                ax.scatter(xs, ys, label=cfg, color=colors[i], alpha=0.7, s=60)
            ax.set_xlabel("Memory (MB)")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} vs Memory")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Pareto Frontier — {metric.capitalize()} Trade-off",
                     fontsize=13)
        fig.tight_layout()
        return fig

    # ── Correlation heatmap ──────────────────────────────────────────────

    def correlation_heatmap(self) -> plt.Figure:
        """Pearson correlation heatmap of distribution features vs QSNR/MSE.

        Computes pairwise Pearson correlation across all distribution features
        collected by DistributionObserver (crest_factor, skewness, kurtosis,
        sparse_ratio, etc.) plus QSNR and MSE where available.

        Returns:
            matplotlib Figure with a single heatmap axes.

        Raises:
            ValueError: If distribution feature data is not available.
        """
        df = self._report.to_dataframe()
        if df is None or df.empty:
            raise ValueError(
                "No data available for correlation analysis. "
                "Ensure DistributionObserver is active during the analysis pass."
            )

        feat_cols = [
            "crest_factor", "skewness", "kurtosis", "excess_kurtosis",
            "bimodality_coefficient", "sparse_ratio", "dynamic_range_bits",
            "outlier_ratio", "norm_entropy",
        ]
        available = [c for c in feat_cols if c in df.columns]

        # Add QSNR/MSE only if present in most rows (avoid sparse columns
        # causing dropna to remove too much data)
        for c in ["qsnr_db", "mse"]:
            if c in df.columns and df[c].notna().sum() > len(df) * 0.5:
                available.append(c)

        if len(available) < 2:
            raise ValueError(
                "Insufficient distribution feature data for correlation heatmap. "
                "Ensure DistributionObserver is active during the analysis pass."
            )

        sub = df[available].dropna()
        if len(sub) < 3:
            raise ValueError(
                "Too few data points for correlation analysis "
                f"({len(sub)} rows after dropping NaN)."
            )

        corr = sub.corr()
        labels = available

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1),
                                        max(8, len(labels) * 0.9)))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(len(labels)):
            for j in range(len(labels)):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=7, color="white" if abs(v) > 0.5 else "black")

        cbar = fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
        ax.set_title("Distribution Features × QSNR/MSE Correlation", fontsize=12)
        fig.tight_layout()
        return fig

    # ── Cost decomposition ───────────────────────────────────────────────

    def cost_decomposition(self) -> plt.Figure:
        """Cost decomposition stacked bar chart (FLOPs per config).

        Shows math / quantize / transform FLOPs broken down per config
        from the cost model. Requires cost estimation to have been run.

        Returns:
            matplotlib Figure.

        Raises:
            ValueError: If no cost data is available.
        """
        rows = []
        for part_name, part_results in self._report._results.items():
            for r in part_results:
                if r.cost is None:
                    continue
                c = r.cost
                rows.append({
                    "part": part_name,
                    "config": r.name,
                    "flops_math": getattr(c, "total_flops_math",
                                          sum(getattr(l, "flops_math", 0)
                                              for l in getattr(c, "layers", []))),
                    "flops_quantize": getattr(c, "total_flops_quantize",
                                              sum(getattr(l, "flops_quantize", 0)
                                                  for l in getattr(c, "layers", []))),
                    "flops_transform": getattr(c, "total_flops_transform",
                                               sum(getattr(l, "flops_transform", 0)
                                                   for l in getattr(c, "layers", []))),
                })

        if not rows:
            raise ValueError(
                "No cost data available for decomposition. "
                "Run the cost model (needs_cost=True) before generating this figure."
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        configs = [r["config"] for r in rows]
        math_vals = [r["flops_math"] for r in rows]
        quant_vals = [r["flops_quantize"] for r in rows]
        trans_vals = [r["flops_transform"] for r in rows]

        x = np.arange(len(configs))
        width = 0.6

        p1 = ax.bar(x, math_vals, width, label="Math FLOPs", color="#0072B2", alpha=0.8)
        p2 = ax.bar(x, quant_vals, width, bottom=math_vals, label="Quantize FLOPs",
                    color="#D55E00", alpha=0.8)
        bottoms2 = [m + q for m, q in zip(math_vals, quant_vals)]
        p3 = ax.bar(x, trans_vals, width, bottom=bottoms2, label="Transform FLOPs",
                    color="#009E73", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("FLOPs")
        ax.set_title("Cost Decomposition by Config")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Add total labels on top
        totals = [m + q + t for m, q, t in zip(math_vals, quant_vals, trans_vals)]
        for bar, total in zip(p3, totals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height(),
                   f"{total:,}", ha="center", va="bottom", fontsize=7, rotation=90)

        fig.tight_layout()
        return fig

    # ── Role distribution comparison ─────────────────────────────────────

    def role_distribution_comparison(self) -> plt.Figure:
        """Per-role distribution feature comparison (boxplots).

        Compares skewness, kurtosis, and normalized entropy across
        input / weight / output roles.

        Returns:
            matplotlib Figure with 1×3 boxplot panels.

        Raises:
            ValueError: If distribution feature data is not available.
        """
        df = self._report.to_dataframe()
        needed = {"skewness", "kurtosis", "norm_entropy", "role"}
        if df is None or df.empty:
            raise ValueError(
                "Distribution data not available. "
                "Ensure DistributionObserver is active during the analysis pass."
            )
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(
                f"Required metrics not available: {sorted(missing)}. "
                "Ensure DistributionObserver is active during the analysis pass."
            )

        # Filter to plot-relevant roles
        plot_roles = [r for r in ["input", "weight", "output"] if r in df["role"].values]
        if not plot_roles:
            raise ValueError(
                "No input/weight/output role data found in report. "
                f"Roles present: {sorted(df['role'].unique())}."
            )

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors_cycle = plt.cm.tab10.colors

        for ax, feature, ylabel in [
            (axes[0], "skewness", "Skewness"),
            (axes[1], "kurtosis", "Kurtosis"),
            (axes[2], "norm_entropy", "Normalized Entropy"),
        ]:
            data_groups = []
            labels = []
            for i, role in enumerate(plot_roles):
                vals = df[df["role"] == role][feature].dropna().tolist()
                if vals:
                    data_groups.append(vals)
                    labels.append(role)

            if data_groups:
                bp = ax.boxplot(data_groups, tick_labels=labels, patch_artist=True)
                for patch, role in zip(bp["boxes"], labels):
                    idx = plot_roles.index(role) if role in plot_roles else 0
                    patch.set_facecolor(colors_cycle[idx % len(colors_cycle)])
                    patch.set_alpha(0.6)

            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} by Role")
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Distribution Feature Comparison Across Roles", fontsize=13)
        fig.tight_layout()
        return fig


# ── Helpers ──────────────────────────────────────────────────────────────

def _short_layer_name(name: str) -> str:
    """Shorten a full module path for x-axis labels."""
    name = name.replace("module.", "").replace("Quantized", "")
    return name[:20]


def _fmt_bits(fmt_name: str) -> int:
    """Extract approximate bit-width from a format name string.

    Parses the first contiguous digits from the format name.
    ``"int8"`` → 8, ``"fp8_e4m3"`` → 8, ``"nf4"`` → 4.
    """
    m = re.search(r"(\d+)", str(fmt_name))
    return int(m.group(1)) if m else 32
