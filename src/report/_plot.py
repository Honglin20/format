"""Plot accessors — post-hoc visualization on SessionResult and StudyReport."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.report._study_report import StudyReport
    from src.session._result import SessionResult

_VALID_PLOT_ROLES = frozenset({"input", "weight", "output", "bias"})

# ── Error message helpers ──────────────────────────────────────────────────
# Concrete HOW, not just WHAT. Each observer key maps to the `outputs`
# string the user must pass to `session.run()` or `session.analyze()`.

_OBSERVER_HOW: dict = {
    "qsnr":          'outputs=["qsnr"] (or "default" / "all")',
    "mse":           'outputs=["mse"] (or "default" / "all")',
    "distribution":  'outputs=["distribution"] (or "all")',
    "histogram":     'outputs=["histogram"] (or "all")',
    "fit":           'outputs=["fit"] (or "all") — requires scipy',
    "per_block_qsnr":'outputs=["per_block_qsnr"] (or "all")',
    "outlier":       'outputs=["distribution", "qsnr"] (or "all")',
}

def _how_to(*keys: str) -> str:
    """Return the last sentence of an error message: what to pass to run()."""
    clauses = []
    for k in keys:
        clauses.append(_OBSERVER_HOW.get(k, k))
    if len(clauses) == 1:
        return f"Enable via: session.run(calib_data, {clauses[0]})."
    return f"Enable via: session.run(calib_data, outputs=[{', '.join(repr(k) for k in keys)}])."


# ── Shared rendering functions (used by both SessionPlotAccessor and  ──
# StudyPlotAccessor). Accept data dicts, return Figures.                ──

def _render_error_propagation(corr_data: dict, role: str) -> plt.Figure:
    """Render 3-row error propagation panel from correlation data.

    Args:
        corr_data: ``{config_name: {"matched": [...], "observer_only": [...],
            "hook_only": [...]}}`` — 1 or N configs.
        role: Tensor role for the title label.

    Returns:
        matplotlib Figure with 3 rows × 1 column.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(corr_data), 1)))

    all_hook_keys: list = []
    for idx, (cfg_name, info) in enumerate(corr_data.items()):
        matched = info["matched"]
        if not matched:
            continue

        hook_keys = [m[0] for m in matched]
        acc_qsnrs = [m[1] for m in matched]
        loc_qsnrs = [m[2] for m in matched]
        if not all_hook_keys:
            all_hook_keys = hook_keys

        x = np.arange(len(hook_keys))
        width = 0.35
        n_cfgs = max(len(corr_data), 1)
        group_width = width * 2 + 0.1
        offset = (idx - (n_cfgs - 1) / 2) * group_width

        color = colors[idx % len(colors)]

        # Row 1: accumulated vs local grouped bars
        ax0 = axes[0]
        ax0.bar(x + offset, acc_qsnrs, width,
                label=f"{cfg_name} (accum)", color=color, alpha=0.8)
        ax0.bar(x + offset + width, loc_qsnrs, width,
                label=f"{cfg_name} (local)", color=color, alpha=0.35,
                hatch="//")

        # Row 2: delta-QSNR
        ax1 = axes[1]
        deltas = [0.0]
        for i in range(1, len(matched)):
            deltas.append(matched[i - 1][1] - matched[i][1])
        bars_delta = ax1.bar(x + offset + width / 2, deltas,
                             group_width * 0.9, color=color, alpha=0.8)
        for bar, d in zip(bars_delta, deltas):
            if d > 5.0:
                bar.set_color("#e74c3c")
            elif d > 1.0:
                bar.set_color("#f39c12")
            else:
                bar.set_color("#2ecc71")

        # Row 3: headroom (local - accumulated)
        ax2 = axes[2]
        headrooms = [loc - acc for loc, acc in zip(loc_qsnrs, acc_qsnrs)]
        bars_head = ax2.bar(x + offset + width / 2, headrooms,
                            group_width * 0.9, color=color, alpha=0.8)
        for bar, h in zip(bars_head, headrooms):
            if h < 3.0:
                bar.set_color("#e74c3c")
            elif h < 10.0:
                bar.set_color("#f39c12")
            else:
                bar.set_color("#2ecc71")

    # Labels
    axes[0].set_ylabel("QSNR (dB)")
    axes[0].set_title(f"Accumulated vs Local QSNR [{role}]")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].set_ylabel("Δ QSNR (dB)")
    axes[1].set_title(
        f"Δ-QSNR: Accumulated Drop Between Layers [{role}]  "
        "(red > 5 dB, amber > 1 dB)"
    )
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].set_ylabel("Headroom (dB)")
    axes[2].set_xlabel("Layer")
    axes[2].set_title(
        f"Local Headroom = Local − Accumulated [{role}]  "
        "(red < 3 dB = source, amber < 10 dB, green = propagated)"
    )
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    if all_hook_keys:
        short_names = [_short_layer_name(k) for k in all_hook_keys]
        for ax in axes:
            ax.set_xticks(np.arange(len(all_hook_keys)))
            ax.set_xticklabels(short_names, rotation=45, ha="right",
                               fontsize=7)

    fig.suptitle("Error Propagation Analysis", fontsize=14)
    fig.tight_layout()
    return fig


def _render_accumulated_vs_local(corr_data: dict, role: str) -> plt.Figure:
    """Render accumulated vs local QSNR scatter from correlation data.

    Args:
        corr_data: ``{config_name: {"matched": [...], ...}}`` — 1 or N configs.
        role: Tensor role for the title label.

    Returns:
        matplotlib Figure with single scatter axes.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    colors_cfg = plt.cm.tab10(np.linspace(0, 1, max(len(corr_data), 1)))

    all_acc, all_loc = [], []

    for idx, (cfg_name, info) in enumerate(corr_data.items()):
        matched = info["matched"]
        if not matched:
            continue

        xs = [m[1] for m in matched]
        ys = [m[2] for m in matched]
        labels = [m[0] for m in matched]
        all_acc.extend(xs)
        all_loc.extend(ys)

        ax.scatter(xs, ys, label=cfg_name,
                   color=colors_cfg[idx % len(colors_cfg)],
                   alpha=0.7, s=60)

        for x, y, label in zip(xs, ys, labels):
            headroom = y - x
            if headroom > 15.0:
                short = _short_layer_name(label)
                ax.annotate(short + "↑", (x, y), fontsize=6, alpha=0.7,
                            xytext=(4, 4), textcoords="offset points")
            elif headroom < 3.0:
                short = _short_layer_name(label)
                ax.annotate(short + "×", (x, y), fontsize=6, alpha=0.7,
                            xytext=(4, 4), textcoords="offset points")

    if all_acc:
        lo = min(all_acc + all_loc) - 2
        hi = max(all_acc + all_loc) + 2
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.5,
                label="y=x (source)")

    ax.set_xlabel("Accumulated QSNR (dB)")
    ax.set_ylabel("Local QSNR (dB)")
    ax.set_title(f"Accumulated vs Local QSNR [{role}]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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

    def qsnr_comparison(self, qsnr_cap=None, skip_activations: bool = True, op_types=None) -> plt.Figure:
        """Per-layer QSNR overlay for all configs.

        One line per config. Layers are aligned by name across configs so
        that the same layer in different configs shares an x-position.

        Args:
            qsnr_cap: If set, clip QSNR values to this maximum.
            skip_activations: If True (default), exclude ReLU/GELU/etc. layers.
            op_types: Operator types to include, e.g. ``["linear", "conv"]``.
                ``None`` = all types.

        Returns:
            matplotlib Figure. The caller is responsible for ``show()`` or
            ``savefig()``.

        Raises:
            ValueError: If ``qsnr_db`` is not available (QSNRObserver not active).
        """
        from src.analysis._error_provenance import is_activation_layer
        from src.viz._layer_classify import filter_layers_by_type

        df = self._report.to_dataframe()
        if df is None or df.empty or "qsnr_db" not in df.columns:
            raise ValueError(
                "QSNR data not available — QSNRObserver was not active. "
                + _how_to("qsnr")
            )

        # Collect union of all layer names across configs
        all_layers = list(dict.fromkeys(df["layer"]))  # preserve order, deduplicate
        if skip_activations:
            all_layers = [n for n in all_layers if not is_activation_layer(n)]
        if op_types:
            all_layers = filter_layers_by_type(all_layers, op_types)
        configs = sorted(df["config"].unique())

        fig, ax = plt.subplots(figsize=(12, 6))
        x_positions = list(range(len(all_layers)))

        for cfg in configs:
            cfg_df = df[df["config"] == cfg]
            # Average QSNR across roles per layer, then map to layer order
            per_layer = cfg_df.groupby("layer")["qsnr_db"].mean()
            values = [per_layer.get(l, float("nan")) for l in all_layers]
            if qsnr_cap is not None:
                values = [min(v, qsnr_cap) if v == v else v for v in values]
            ax.plot(x_positions, values, marker="o", label=cfg, linewidth=2)

        short_names = [_short_layer_name(l) for l in all_layers]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("QSNR (dB)")
        ax.set_title("QSNR per Layer (avg across input/weight/output)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ── Crest factor vs QSNR scatter ────────────────────────────────────

    def crest_vs_qsnr(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Crest factor vs QSNR scatter, one panel per role.

        Args:
            roles: Tensor roles to plot. Default all three
                ``("input", "weight", "output")``. Each must be one of
                ``"input"``, ``"weight"``, ``"output"``, ``"bias"``.

        Returns:
            matplotlib Figure with 1 × len(roles) subplots.

        Raises:
            ValueError: If any role is invalid, or if required observer data
                is not present.
        """
        for role in roles:
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
                f"Crest factor data not available — missing: {sorted(missing)}. "
                + _how_to("distribution", "qsnr")
            )

        available = [r for r in roles if r in df["role"].values]
        if not available:
            all_roles = sorted(df["role"].unique()) if "role" in df.columns else []
            raise ValueError(
                f"No data for roles {list(roles)}. "
                f"Roles present in the report: {all_roles or '(none)'}."
            )

        fig, axes = plt.subplots(1, len(available),
                                 figsize=(6 * len(available), 5),
                                 squeeze=False)

        for ax, role in zip(axes[0], available):
            role_df = df[df["role"] == role]
            configs = sorted(role_df["config"].unique())
            for cfg in configs:
                cfg_df = role_df[role_df["config"] == cfg]
                ax.scatter(cfg_df["crest_factor"], cfg_df["qsnr_db"],
                           label=cfg, alpha=0.7, s=40)
            ax.set_xlabel("Crest Factor (peak / RMS)")
            ax.set_ylabel("QSNR (dB)")
            ax.set_title(f"{role}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Crest Factor vs QSNR by Role", fontsize=13)
        fig.tight_layout()
        return fig


    # ── Outlier analysis ───────────────────────────────────────────────

    def outlier_analysis(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Outlier ratio per-layer bar + outlier vs QSNR scatter, one row per role.

        Args:
            roles: Tensor roles to plot. Default all three
                ``("input", "weight", "output")``.

        Returns:
            matplotlib Figure with len(roles) rows × 2 columns.

        Raises:
            ValueError: If any role is invalid, or if ``outlier_ratio`` from
                DistributionObserver is not available.
        """
        for role in roles:
            if role not in _VALID_PLOT_ROLES:
                raise ValueError(
                    f"Invalid role {role!r}. Must be one of "
                    f"{sorted(_VALID_PLOT_ROLES)}."
                )

        df = self._report.to_dataframe()
        if df is None or df.empty or "outlier_ratio" not in df.columns:
            raise ValueError(
                "Outlier ratio data not available — DistributionObserver was not active. "
                + _how_to("distribution")
            )

        available = [r for r in roles if r in df["role"].values]
        if not available:
            all_roles = sorted(df["role"].unique()) if "role" in df.columns else []
            raise ValueError(
                f"No data for roles {list(roles)}. "
                f"Roles present in the report: {all_roles or '(none)'}."
            )

        n_rows = len(available)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows),
                                 squeeze=False)

        for row_idx, role in enumerate(available):
            ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1]
            role_df = df[df["role"] == role]

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
            ax1.set_title(f"Outlier Ratio per Layer [{role}]")
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
            ax2.set_title(f"Outlier Ratio vs QSNR [{role}]")
            ax2.legend(fontsize=7)
            ax2.grid(True, alpha=0.3)

        fig.suptitle("Outlier Analysis by Role", fontsize=13)
        fig.tight_layout()
        return fig

    # ── Per-block QSNR distribution ─────────────────────────────────────

    def per_block_qsnr(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Per-block QSNR statistics (std + min-vs-mean), one row per role.

        Uses ``qsnr_db_std``, ``qsnr_db_min``, ``qsnr_db_max`` collected by
        QSNRObserver in per-block mode.

        Args:
            roles: Tensor roles to plot. Default all three
                ``("input", "weight", "output")``.

        Returns:
            matplotlib Figure with len(roles) rows × 2 columns.

        Raises:
            ValueError: If per-block QSNR statistics are not available.
        """
        for role in roles:
            if role not in _VALID_PLOT_ROLES:
                raise ValueError(
                    f"Invalid role {role!r}. Must be one of "
                    f"{sorted(_VALID_PLOT_ROLES)}."
                )

        df = self._report.to_dataframe()
        needed = {"qsnr_db_std", "qsnr_db_min", "qsnr_db_max"}
        if df is None or df.empty:
            raise ValueError(
                "Per-block QSNR data not available — QSNRObserver was not active. "
                + _how_to("qsnr")
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

        available = [r for r in roles if r in df["role"].values]
        if not available:
            all_roles = sorted(df["role"].unique()) if "role" in df.columns else []
            raise ValueError(
                f"No data for roles {list(roles)}. "
                f"Roles present in the report: {all_roles or '(none)'}."
            )

        n_rows = len(available)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows),
                                 squeeze=False)

        for row_idx, role in enumerate(available):
            ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1]
            role_df = df[df["role"] == role]
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
            ax1.set_title(f"Per-Block QSNR Std Dev [{role}]")
            ax1.grid(True, alpha=0.3, axis="y")

            # Panel 2: qsnr_db_min vs qsnr_db mean scatter
            for cfg in configs:
                cfg_df = role_df[role_df["config"] == cfg]
                if "qsnr_db_min" in cfg_df.columns and "qsnr_db" in cfg_df.columns:
                    ax2.scatter(cfg_df["qsnr_db"], cfg_df["qsnr_db_min"],
                               label=cfg, alpha=0.7, s=40)

            ax2.set_xlabel("Mean QSNR (dB)")
            ax2.set_ylabel("Min QSNR (dB)")
            ax2.set_title(f"Min vs Mean QSNR per Block [{role}]")
            ax2.legend(fontsize=7)
            ax2.grid(True, alpha=0.3)

            # Diagonal reference line
            if "qsnr_db" in role_df.columns:
                vals = role_df["qsnr_db"].dropna()
                if not vals.empty:
                    lo, hi = vals.min(), vals.max()
                    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.5)

        fig.suptitle("Per-Block QSNR Distribution by Role", fontsize=13)
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
                "No data available for correlation analysis — "
                "DistributionObserver was not active. "
                + _how_to("distribution")
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
                + _how_to("distribution")
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

    # ── Error propagation ───────────────────────────────────────────────

    def error_propagation(self, role: str = "output") -> plt.Figure:
        """Accumulated vs local QSNR decomposition, 3-row panel.

        Row 1: Grouped bar — accumulated QSNR (hook) vs local QSNR (observer).
        Row 2: Delta-QSNR — drop in accumulated QSNR between consecutive layers.
        Row 3: Headroom — local minus accumulated QSNR.

        Args:
            role: Tensor role to analyse (default ``"output"``).

        Returns:
            matplotlib Figure with 3 rows × 1 column.

        Raises:
            ValueError: If correlation data is not available (requires both
                ``keep_fp32=True`` and QSNRObserver active (default)).
        """
        if role not in _VALID_PLOT_ROLES:
            raise ValueError(
                f"Invalid role {role!r}. Must be one of "
                f"{sorted(_VALID_PLOT_ROLES)}."
            )

        data = self._report.correlate_hook_observer(role)
        if not any(bool(info["matched"]) for info in data.values()):
            raise ValueError(
                "Error propagation data not available. "
                "Requires qsnr observer (included in default outputs) and keep_fp32=True. "
                + _how_to("qsnr")
            )
        return _render_error_propagation(data, role)

    def accumulated_vs_local(self, role: str = "output") -> plt.Figure:
        """Scatter plot: accumulated QSNR vs local QSNR.

        Points on the y=x diagonal indicate that this layer's local
        quantization is the dominant error source. Points above the
        diagonal mean error is propagated from earlier layers.

        Args:
            role: Tensor role to analyse (default ``"output"``).

        Returns:
            matplotlib Figure with single scatter axes.

        Raises:
            ValueError: If correlation data is not available.
        """
        if role not in _VALID_PLOT_ROLES:
            raise ValueError(
                f"Invalid role {role!r}. Must be one of "
                f"{sorted(_VALID_PLOT_ROLES)}."
            )

        data = self._report.correlate_hook_observer(role)
        if not any(bool(info["matched"]) for info in data.values()):
            raise ValueError(
                "Accumulated vs local data not available. "
                "Requires qsnr observer (included in default outputs) and keep_fp32=True. "
                + _how_to("qsnr")
            )
        return _render_accumulated_vs_local(data, role)

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
                "Distribution data not available — DistributionObserver was not active. "
                + _how_to("distribution")
            )
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(
                f"Distribution features not available — missing: {sorted(missing)}. "
                + _how_to("distribution")
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

    # ── Kurtosis analysis ───────────────────────────────────────────────

    def kurtosis_analysis(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Kurtosis distribution, QSNR relationship, and top-layer ranking.

        Three-panel figure: kurtosis histogram with reference lines,
        kurtosis vs QSNR scatter by role, and top-15 (layer, role) ranked
        by kurtosis.

        Args:
            roles: Tensor roles to include (default input/weight/output).

        Returns:
            matplotlib Figure with 1×3 panels.
        """
        df = self._report.to_dataframe()
        if df is None or df.empty or "kurtosis" not in df.columns:
            raise ValueError(
                "Kurtosis data not available — DistributionObserver was not active. "
                + _how_to("distribution", "qsnr")
            )

        role_df = df[df["role"].isin(roles)]
        if role_df.empty:
            raise ValueError(f"No data for roles {list(roles)}.")

        return _render_kurtosis_analysis(role_df, roles)

    # ── Histogram overlay ──────────────────────────────────────────────

    def histogram_overlay(
        self,
        top_k: int = 5,
        role: str | None = None,
        layer: str | None = None,
        op_types: list | None = None,
        qsnr_type: str = "local",
    ) -> plt.Figure:
        """Three-channel histogram overlay across all study configs.

        Two modes:

        - **Top-K mode** (default): most sensitive (layer, role) pairs.
        - **Single-layer mode** (``layer="fc1"``): input/weight/output for
          one layer.

        Args:
            top_k: Number of most-sensitive pairs (top-K mode).
            role: Filter by tensor role. ``None`` = all.
            layer: Specific layer name → single-layer mode.
            op_types: Operator types to include, e.g. ``["linear"]``.
            qsnr_type: ``"local"`` or ``"accum"`` for ranking/labelling.

        Returns:
            matplotlib Figure.
        """
        import torch as _torch
        from src.viz._layer_classify import filter_layers_by_type

        if qsnr_type not in ("local", "accum"):
            raise ValueError(f"qsnr_type must be 'local' or 'accum', got {qsnr_type!r}")

        roles = ("input", "weight", "output") if role is None else (role,)
        if qsnr_type == "accum" and role is None:
            roles = ("output",)

        # Collect histogram data and QSNR across all results
        layer_hists: dict = {}
        layer_error: dict = {}

        for part_name, part_results in self._report._results.items():
            for r in part_results:
                accum_lookup = _build_accum_qsnr_lookup(r) if qsnr_type == "accum" else {}
                obs = r.observers_data
                if not obs:
                    continue
                for lname, roles_dict in obs.items():
                    if op_types and filter_layers_by_type([lname], op_types) == []:
                        continue
                    for role_str, stages in roles_dict.items():
                        if role_str not in roles:
                            continue
                        key = f"{lname} [{role_str}]"
                        if key in layer_hists:
                            continue
                        for _stage, slices in stages.items():
                            for _slice_key, metrics in slices.items():
                                if "fp32_hist" not in metrics or "quant_hist" not in metrics:
                                    continue
                                entry = {}
                                for k in ("fp32_hist", "quant_hist", "err_hist",
                                          "fp32_min", "fp32_max"):
                                    v = metrics.get(k)
                                    if v is not None:
                                        if k in ("fp32_min", "fp32_max"):
                                            entry[k] = float(v)
                                        elif isinstance(v, _torch.Tensor):
                                            entry[k] = v.cpu().float().numpy()
                                        elif not isinstance(v, np.ndarray):
                                            entry[k] = np.asarray(v)
                                        else:
                                            entry[k] = v
                                if "fp32_hist" in entry and "quant_hist" in entry:
                                    layer_hists[key] = entry

                                # QSNR for ranking
                                if qsnr_type == "accum":
                                    qsnr = accum_lookup.get(lname)
                                else:
                                    qsnr = metrics.get("qsnr_db")
                                if qsnr is not None and not (math.isnan(qsnr) if isinstance(qsnr, float) else False):
                                    layer_error[key] = qsnr

        if layer is not None:
            return self._histogram_single_layer_study(layer, layer_hists, layer_error)

        if not layer_hists:
            raise ValueError("Histogram data not available. Run with outputs=[\"histogram\"].")

        if layer_error:
            top_layers = sorted(
                layer_hists.items(),
                key=lambda x: layer_error.get(x[0], float("inf")),
            )[:top_k]
        else:
            top_layers = sorted(
                layer_hists.items(),
                key=lambda x: x[1].get("fp32_hist", np.array(0)).sum(),
                reverse=True,
            )[:top_k]

        n = len(top_layers)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for ax, (lk, hd) in zip(axes[0], top_layers):
            _render_hist_subplot(ax, hd)
            q_label = f"  local={layer_error.get(lk, float('nan')):.1f}dB" if layer_error else ""
            ax.set_title(f"{lk}{q_label}", fontsize=9)

        role_label = "" if role is None else f" ({role})"
        q_label2 = "accum" if qsnr_type == "accum" else "local"
        fig.suptitle(f"Activation Histograms (fp32 / quant / error){role_label} — "
                     f"Most Sensitive Layers ({q_label2} QSNR)", fontsize=13)
        fig.tight_layout()
        return fig

    def _histogram_single_layer_study(self, layer: str, layer_hists: dict,
                                      layer_error: dict) -> plt.Figure:
        """Single-layer mode for StudyPlotAccessor."""
        roles = ("input", "weight", "output")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
        short = _short_layer_name(layer)

        for col_idx, r in enumerate(roles):
            ax = axes[0, col_idx]
            key = f"{layer} [{r}]"
            hd = layer_hists.get(key)
            if hd is not None:
                _render_hist_subplot(ax, hd)
                qsnr = layer_error.get(key)
                q_str = f"local={qsnr:.1f}dB" if qsnr is not None else ""
                ax.set_title(f"{short}\n{r}  {q_str}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "no histogram data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
                ax.set_title(f"{short}\n{r}", fontsize=9)

        fig.suptitle(f"Activation Histograms — {short}", fontsize=13)
        fig.tight_layout()
        return fig

    # ── Per-layer role QSNR line ─────────────────────────────────────

    def per_layer_role_qsnr_line(
        self,
        role: str = "output",
        qsnr_type: str = "local",
        op_types: list | None = None,
        skip_activations: bool = True,
        qsnr_cap=None,
    ) -> plt.Figure:
        """Per-layer QSNR line chart comparing configs for a single role.

        One line per config. Layers are aligned by name union across configs.

        Args:
            role: Tensor role to plot (``"input"`` / ``"weight"`` /
                ``"output"``). In accum mode, only ``"output"`` is
                meaningful.
            qsnr_type: ``"local"`` (observer) or ``"accum"`` (hook).
            op_types: Operator types to include, e.g. ``["linear"]``.
            skip_activations: If True (default), exclude activation layers.
            qsnr_cap: If set, clip QSNR values to this maximum.

        Returns:
            matplotlib Figure.
        """
        from src.analysis._error_provenance import is_activation_layer
        from src.viz._layer_classify import filter_layers_by_type

        if role not in _VALID_PLOT_ROLES:
            raise ValueError(f"Invalid role {role!r}.")
        if qsnr_type not in ("local", "accum"):
            raise ValueError(f"qsnr_type must be 'local' or 'accum', got {qsnr_type!r}")

        # Collect per-result QSNR
        config_data: dict = {}
        for part_name, part_results in self._report._results.items():
            for r in part_results:
                label = f"{part_name}/{r.name}" if part_name else r.name

                if qsnr_type == "accum":
                    lookup = _build_accum_qsnr_lookup(r)
                    qsnr_dict = lookup
                else:
                    qsnr_dict = r.qsnr_by_role.get(role, {})
                    if not qsnr_dict:
                        # Fallback to qsnr_per_layer for output
                        if role == "output" and r.qsnr_per_layer:
                            qsnr_dict = r.qsnr_per_layer

                if not qsnr_dict:
                    continue

                config_data[label] = qsnr_dict

        if not config_data:
            raise ValueError(
                f"No QSNR data for role={role!r}. " + _how_to("qsnr")
            )

        # Union of all layer names
        all_layers = list(dict.fromkeys(
            n for qd in config_data.values() for n in qd
        ))
        if skip_activations:
            all_layers = [n for n in all_layers if not is_activation_layer(n)]
        if op_types:
            all_layers = filter_layers_by_type(all_layers, op_types)

        if not all_layers:
            raise ValueError("No layers remaining after filtering.")

        fig, ax = plt.subplots(figsize=(max(12, len(all_layers) * 0.3), 6))
        x = np.arange(len(all_layers))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(config_data), 1)))

        for idx, (label, qsnr_dict) in enumerate(config_data.items()):
            vals = [qsnr_dict.get(n, float("nan")) for n in all_layers]
            if qsnr_cap is not None:
                vals = [min(v, qsnr_cap) if v == v else v for v in vals]
            color = colors[idx % len(colors)]
            ax.plot(x, vals, "o-", color=color, linewidth=1.5, markersize=5,
                    label=label)

        short_names = [_short_layer_name(n) for n in all_layers]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("QSNR (dB)")
        q_label = "accum" if qsnr_type == "accum" else "local"
        ax.set_title(f"Per-Layer QSNR — {role} role ({q_label})")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    # ── Per-layer role distribution histogram ──────────────────────────

    def per_layer_role_histogram(
        self, k: int = 5, log_y: bool = False,
        op_types: list | None = None,
        qsnr_type: str = "local",
    ) -> plt.Figure:
        """Per-layer, per-role fp32 value distribution for worst-QSNR layers.

        For the *k* layers with the lowest QSNR, plots the fp32 value
        distribution histogram for each role (input / weight / output)
        in a grid: rows = layers, cols = roles.

        Uses ``HistogramObserver`` data (``fp32_hist``) from the raw
        observer buffer. Falls back to text-only distribution summary
        from ``DistributionObserver`` when histogram data is absent.

        Args:
            k: Number of worst-QSNR layers to show (default 5).
            log_y: If True, use log scale for histogram y-axis.

        Returns:
            matplotlib Figure with *k* rows × 3 columns.

        Raises:
            ValueError: If no histogram or distribution data is available.
        """
        import torch as _torch

        roles = ("input", "weight", "output")

        # Collect histogram data from raw observer buffers
        # key → {fp32_hist, quant_hist, err_hist, fp32_min, fp32_max}
        layer_role_hists: dict = {}
        layer_role_dist: dict = {}  # DistributionObserver fallback

        for part_name, part_results in self._report._results.items():
            for r in part_results:
                obs = r.observers_data
                if not obs:
                    continue
                for layer, roles_dict in obs.items():
                    for role, stages in roles_dict.items():
                        if role not in roles:
                            continue
                        key = (layer, role)
                        if key in layer_role_hists or key in layer_role_dist:
                            continue
                        for _stage, slices in stages.items():
                            for _slice_key, metrics in slices.items():
                                if "fp32_hist" in metrics:
                                    h = metrics["fp32_hist"]
                                    if isinstance(h, _torch.Tensor):
                                        h = h.cpu().float().numpy()
                                    elif not isinstance(h, np.ndarray):
                                        h = np.asarray(h)
                                    if len(h) > 0:
                                        entry = {"fp32_hist": h}
                                        for ch in ("quant_hist", "err_hist"):
                                            v = metrics.get(ch)
                                            if v is not None:
                                                if isinstance(v, _torch.Tensor):
                                                    v = v.cpu().float().numpy()
                                                elif not isinstance(v, np.ndarray):
                                                    v = np.asarray(v)
                                                if len(v) > 0:
                                                    entry[ch] = v
                                        for bound in ("fp32_min", "fp32_max"):
                                            v = metrics.get(bound)
                                            if v is not None:
                                                entry[bound] = float(v)
                                        layer_role_hists[key] = entry
                                if key not in layer_role_dist:
                                    dkeys = ("mean", "std", "skewness",
                                             "kurtosis", "min", "max")
                                    d = {dk: metrics[dk]
                                         for dk in dkeys if dk in metrics}
                                    if d:
                                        layer_role_dist[key] = d

        if not layer_role_hists and not layer_role_dist:
            raise ValueError(
                "No histogram or distribution data available. "
                + _how_to("histogram", "distribution")
            )

        # Rank layers by QSNR
        from src.viz._layer_classify import filter_layers_by_type

        accum_avg: dict = {}
        if qsnr_type == "accum":
            accum_sums: dict = {}
            accum_counts: dict = {}
            for part_results in self._report._results.values():
                for r in part_results:
                    lookup = _build_accum_qsnr_lookup(r)
                    for lname, val in lookup.items():
                        accum_sums[lname] = accum_sums.get(lname, 0.0) + val
                        accum_counts[lname] = accum_counts.get(lname, 0) + 1
            for lname in accum_sums:
                accum_avg[lname] = accum_sums[lname] / accum_counts[lname]

        df = self._report.to_dataframe()
        bottom_k: list = []
        if qsnr_type == "accum" and accum_avg:
            ranked = sorted(accum_avg, key=lambda l: accum_avg[l])
            if op_types:
                ranked = filter_layers_by_type(ranked, op_types)
            all_layers_set = set(
                l for l, r in set(layer_role_hists.keys()) | set(layer_role_dist.keys())
            )
            bottom_k = [l for l in ranked if l in all_layers_set][:k]
        elif df is not None and not df.empty and "qsnr_db" in df.columns:
            layer_qsnr = df.groupby("layer")["qsnr_db"].mean()
            ranked = layer_qsnr.nsmallest(len(layer_qsnr)).index.tolist()
            if op_types:
                ranked = filter_layers_by_type(ranked, op_types)
            bottom_k = ranked[:k]
        else:
            all_layers = sorted(set(
                l for l, r in set(layer_role_hists.keys()) | set(layer_role_dist.keys())
            ))
            if op_types:
                all_layers = filter_layers_by_type(all_layers, op_types)
            bottom_k = all_layers[:k]

        if not bottom_k:
            raise ValueError("No layers found with distribution data.")

        present_roles = [r for r in roles if any(
            (l, r) in layer_role_hists or (l, r) in layer_role_dist
            for l in bottom_k
        )]
        if not present_roles:
            present_roles = list(roles)

        n_rows = len(bottom_k)
        n_cols = len(present_roles)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 3 * n_rows),
                                 squeeze=False)

        for row_idx, layer in enumerate(bottom_k):
            for col_idx, role in enumerate(present_roles):
                ax = axes[row_idx, col_idx]
                key = (layer, role)

                if key in layer_role_hists:
                    entry = layer_role_hists[key]
                    fp32_counts = entry.get("fp32_hist")
                    quant_counts = entry.get("quant_hist")
                    err_counts = entry.get("err_hist")
                    n_bins = len(fp32_counts) if fp32_counts is not None else 64
                    x_bins = np.arange(n_bins)

                    # fp32: blue fill (matching histogram_overlay)
                    if fp32_counts is not None:
                        ax.fill_between(x_bins, fp32_counts, alpha=0.25,
                                        color="#3498db", label="fp32", step="mid")

                    # quant: red dashed outline
                    if quant_counts is not None:
                        ax.plot(x_bins, quant_counts, color="#e74c3c",
                                linewidth=1.2, linestyle="--", label="quant")

                    # x-axis: real value range
                    lo = entry.get("fp32_min")
                    hi = entry.get("fp32_max")
                    if lo is not None and hi is not None:
                        edges = np.linspace(lo, hi, n_bins + 1)
                        tick_idx = np.linspace(0, n_bins - 1, min(5, n_bins)).astype(int)
                        ax.set_xticks(tick_idx)
                        ax.set_xticklabels([f"{edges[i]:.3g}" for i in tick_idx], fontsize=6)
                        ax.set_xlabel("Value", fontsize=7)
                    else:
                        ax.set_xlabel("Bin", fontsize=7)

                    # error: twin axis so small errors are visible
                    if err_counts is not None and err_counts.sum() > 0:
                        ax2 = ax.twinx()
                        ax2.fill_between(x_bins, err_counts, alpha=0.3,
                                         color="#95a5a6", label="error", step="mid")
                        ax2.set_ylabel("Error", fontsize=6)
                        ax2.tick_params(axis="y", labelsize=5)
                        handles1, labels1 = ax.get_legend_handles_labels()
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        ax2.legend(handles1 + handles2, labels1 + labels2,
                                  fontsize=5, loc="upper right")
                    else:
                        ax.legend(fontsize=5, loc="upper right")
                elif key in layer_role_dist:
                    d = layer_role_dist[key]
                    lines = [
                        f"mean={d.get('mean', 0):.2g}",
                        f"std={d.get('std', 0):.2g}",
                        f"skew={d.get('skewness', 0):.2f}",
                        f"kurt={d.get('kurtosis', 0):.2f}",
                    ]
                    ax.text(0.5, 0.5, "\n".join(lines),
                            transform=ax.transAxes,
                            ha="center", va="center", fontsize=8,
                            bbox=dict(boxstyle="round", facecolor="#f0f0f0",
                                      alpha=0.8))
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9,
                            color="gray")

                # QSNR label per (layer, role)
                qsnr_str = ""
                if qsnr_type == "accum":
                    aq = accum_avg.get(layer)
                    if aq is not None:
                        qsnr_str = f"accum={aq:.1f}dB"
                elif df is not None and not df.empty and "qsnr_db" in df.columns:
                    sub = df[(df["layer"] == layer) & (df["role"] == role)]
                    if not sub.empty:
                        vals = sub["qsnr_db"].dropna()
                        if not vals.empty:
                            qsnr_str = f"local={vals.mean():.1f}dB"

                short = _short_layer_name(layer)
                ax.set_title(f"{short}\n{role} {qsnr_str}", fontsize=7)
                if row_idx == n_rows - 1 and key not in layer_role_hists:
                    ax.set_xlabel("Bin", fontsize=7)
                if col_idx == 0:
                    ax.set_ylabel("Count", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)
                if log_y and key in layer_role_hists:
                    ax.set_yscale("log")

        fig.suptitle("Per-Layer fp32 Value Distribution — Worst-QSNR Layers",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        return fig


class SessionPlotAccessor:
    """Post-hoc visualization on a single :class:`SessionResult`.

    Usage::

        result = Session(model, cfg).run(calib_data)
        result.plot.qsnr_comparison()
        result.plot.error_propagation(role="output")
    """

    def __init__(self, result: "SessionResult"):
        self._result = result
        self._df = None

    # ── DataFrame builder ──────────────────────────────────────────────────

    def _build_df(self):
        """Build a tidy DataFrame from observers_data, same schema as
        :meth:`StudyReport.to_dataframe` but for a single config."""
        if self._df is not None:
            return self._df
        try:
            import pandas as pd
        except ImportError:
            self._df = False
            return False

        obs = self._result.observers_data
        if not obs:
            self._df = False
            return False

        rows = []
        for layer, roles in obs.items():
            for role, stages in roles.items():
                all_metrics = []
                for _stage, slices in stages.items():
                    for _slice_key, metrics in slices.items():
                        all_metrics.append(metrics)
                if not all_metrics:
                    continue
                row = {
                    "config": self._result.name or "(unnamed)",
                    "format": self._result.config.w_format if self._result.config else "",
                    "layer": layer,
                    "role": role,
                }
                all_keys = set()
                for m in all_metrics:
                    all_keys.update(m.keys())
                for key in sorted(all_keys):
                    values = [m[key] for m in all_metrics if key in m]
                    numeric = [v for v in values if isinstance(v, (int, float))]
                    if numeric:
                        row[key] = sum(numeric) / len(numeric)
                rows.append(row)
        self._df = pd.DataFrame(rows) if rows else False
        return self._df

    # ── QSNR comparison ─────────────────────────────────────────────────────

    def qsnr_comparison(self, qsnr_cap=None, skip_activations: bool = True, op_types=None) -> plt.Figure:
        """Per-layer QSNR line chart for a single config.

        Args:
            qsnr_cap: If set, clip QSNR values to this maximum.
            skip_activations: If True (default), exclude ReLU/GELU/etc. layers.
            op_types: Operator types to include, e.g. ``["linear", "conv"]``.
                ``None`` = all types.

        Returns:
            matplotlib Figure.
        """
        from src.analysis._error_provenance import is_activation_layer
        from src.viz._layer_classify import filter_layers_by_type

        qsnr = self._result.qsnr_per_layer
        if not qsnr:
            raise ValueError(
                "QSNR data not available — QSNRObserver was not active. "
                + _how_to("qsnr")
            )

        layers = list(qsnr.keys())
        if skip_activations:
            layers = [n for n in layers if not is_activation_layer(n)]
        if op_types:
            layers = filter_layers_by_type(layers, op_types)
        values = [qsnr[n] for n in layers]
        if qsnr_cap is not None:
            values = [min(v, qsnr_cap) for v in values]
        short_names = [_short_layer_name(l) for l in layers]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(layers))
        ax.bar(x, values, color="#3498db", alpha=0.8, label=self._result.name or "QSNR")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("QSNR (dB)")
        ax.set_title(f"Per-Layer QSNR — {self._result.name or '(unnamed)'}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    # ── Crest factor vs QSNR scatter ────────────────────────────────────────

    def crest_vs_qsnr(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Crest factor vs QSNR scatter, one panel per role.

        Args:
            roles: Tensor roles to plot.
        """
        for role in roles:
            if role not in _VALID_PLOT_ROLES:
                raise ValueError(f"Invalid role {role!r}.")

        df = self._build_df()
        needed = {"crest_factor", "qsnr_db"}
        if df is False or df.empty or not needed.issubset(df.columns):
            raise ValueError(
                "Crest factor data not available. "
                + _how_to("distribution", "qsnr")
            )

        available = [r for r in roles if r in df["role"].values]
        if not available:
            raise ValueError(f"No data for roles {list(roles)}.")

        fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), squeeze=False)
        for ax, role in zip(axes[0], available):
            role_df = df[df["role"] == role]
            ax.scatter(role_df["crest_factor"], role_df["qsnr_db"],
                       alpha=0.7, s=40, color="#3498db")
            ax.set_xlabel("Crest Factor (peak / RMS)")
            ax.set_ylabel("QSNR (dB)")
            ax.set_title(role, fontsize=11)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Crest Factor vs QSNR by Role", fontsize=13)
        fig.tight_layout()
        return fig

    # ── Outlier analysis ────────────────────────────────────────────────────

    def outlier_analysis(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Outlier ratio per-layer bar + outlier vs QSNR scatter."""
        for role in roles:
            if role not in _VALID_PLOT_ROLES:
                raise ValueError(f"Invalid role {role!r}.")

        df = self._build_df()
        if df is False or df.empty or "outlier_ratio" not in df.columns:
            raise ValueError(
                "Outlier ratio data not available. "
                + _how_to("distribution")
            )

        available = [r for r in roles if r in df["role"].values]
        if not available:
            raise ValueError(f"No data for roles {list(roles)}.")

        n_rows = len(available)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows), squeeze=False)

        for row_idx, role in enumerate(available):
            ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1]
            role_df = df[df["role"] == role]
            layers = list(dict.fromkeys(role_df["layer"]))
            x = np.arange(len(layers))

            per_layer = role_df.groupby("layer")["outlier_ratio"].mean()
            values = [per_layer.get(l, 0) for l in layers]
            ax1.bar(x, values, color="#e74c3c", alpha=0.7)
            ax1.set_xticks(x)
            ax1.set_xticklabels([_short_layer_name(l) for l in layers],
                                rotation=45, ha="right", fontsize=7)
            ax1.set_ylabel("Outlier Ratio")
            ax1.set_title(f"Outlier Ratio per Layer [{role}]")
            ax1.grid(True, alpha=0.3, axis="y")

            has_qsnr = "qsnr_db" in df.columns
            if has_qsnr:
                ax2.scatter(role_df["outlier_ratio"], role_df["qsnr_db"],
                           alpha=0.7, s=40, color="#e74c3c")
            ax2.set_xlabel("Outlier Ratio")
            ax2.set_ylabel("QSNR (dB)" if has_qsnr else "(no QSNR)")
            ax2.set_title(f"Outlier Ratio vs QSNR [{role}]")
            ax2.grid(True, alpha=0.3)

        fig.suptitle("Outlier Analysis by Role", fontsize=13)
        fig.tight_layout()
        return fig

    # ── Per-block QSNR distribution ─────────────────────────────────────────

    def per_block_qsnr(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Per-block QSNR statistics (std + min-vs-mean)."""
        for role in roles:
            if role not in _VALID_PLOT_ROLES:
                raise ValueError(f"Invalid role {role!r}.")

        df = self._build_df()
        needed = {"qsnr_db_std", "qsnr_db_min", "qsnr_db_max"}
        if df is False or df.empty:
            raise ValueError("Per-block QSNR data not available. " + _how_to("qsnr"))
        if not needed.issubset(df.columns):
            raise ValueError(
                f"Per-block QSNR statistics not available. "
                "Use per_block granularity."
            )

        available = [r for r in roles if r in df["role"].values]
        if not available:
            raise ValueError(f"No data for roles {list(roles)}.")

        n_rows = len(available)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows), squeeze=False)

        for row_idx, role in enumerate(available):
            ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1]
            role_df = df[df["role"] == role]
            layers = list(dict.fromkeys(role_df["layer"]))

            layer_std_data = {}
            for layer in layers:
                ldf = role_df[role_df["layer"] == layer]
                if "qsnr_db_std" in ldf.columns:
                    vals = ldf["qsnr_db_std"].dropna().tolist()
                    if vals:
                        layer_std_data[layer] = vals

            if layer_std_data:
                positions = range(len(layer_std_data))
                ax1.boxplot(layer_std_data.values(), positions=positions, widths=0.6, patch_artist=True)
                ax1.set_xticks(positions)
                ax1.set_xticklabels([_short_layer_name(l) for l in layer_std_data],
                                    rotation=45, ha="right", fontsize=7)
            ax1.set_ylabel("QSNR Std Dev (dB)")
            ax1.set_title(f"Per-Block QSNR Std Dev [{role}]")
            ax1.grid(True, alpha=0.3, axis="y")

            if "qsnr_db_min" in role_df.columns and "qsnr_db" in role_df.columns:
                ax2.scatter(role_df["qsnr_db"], role_df["qsnr_db_min"],
                           alpha=0.7, s=40, color="#3498db")
                vals = role_df["qsnr_db"].dropna()
                if not vals.empty:
                    lo, hi = vals.min(), vals.max()
                    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.5)
            ax2.set_xlabel("Mean QSNR (dB)")
            ax2.set_ylabel("Min QSNR (dB)")
            ax2.set_title(f"Min vs Mean QSNR per Block [{role}]")
            ax2.grid(True, alpha=0.3)

        fig.suptitle("Per-Block QSNR Distribution by Role", fontsize=13)
        fig.tight_layout()
        return fig

    # ── Correlation heatmap ─────────────────────────────────────────────────

    def correlation_heatmap(self) -> plt.Figure:
        """Pearson correlation heatmap of distribution features vs QSNR/MSE."""
        df = self._build_df()
        if df is False or df.empty:
            raise ValueError("No data available. " + _how_to("distribution"))

        feat_cols = [
            "crest_factor", "skewness", "kurtosis", "excess_kurtosis",
            "bimodality_coefficient", "sparse_ratio", "dynamic_range_bits",
            "outlier_ratio", "norm_entropy",
        ]
        available = [c for c in feat_cols if c in df.columns]
        for c in ["qsnr_db", "mse"]:
            if c in df.columns and df[c].notna().sum() > len(df) * 0.5:
                available.append(c)

        if len(available) < 2:
            raise ValueError("Insufficient distribution feature data. " + _how_to("distribution"))

        sub = df[available].dropna()
        if len(sub) < 3:
            raise ValueError(f"Too few data points ({len(sub)} rows).")

        corr = sub.corr()
        labels = available

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), max(8, len(labels) * 0.9)))
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
        fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
        ax.set_title("Distribution Features × QSNR/MSE Correlation", fontsize=12)
        fig.tight_layout()
        return fig

    # ── Cost decomposition ──────────────────────────────────────────────────

    def cost_decomposition(self) -> plt.Figure:
        """Cost decomposition stacked bar chart (FLOPs)."""
        c = self._result.cost
        if c is None:
            raise ValueError("No cost data available. Run session.cost() first.")

        flops_math = getattr(c, "total_flops_math", sum(
            getattr(ly, "flops_math", 0) for ly in getattr(c, "layers", [])))
        flops_quant = getattr(c, "total_flops_quantize", sum(
            getattr(ly, "flops_quantize", 0) for ly in getattr(c, "layers", [])))
        flops_trans = getattr(c, "total_flops_transform", sum(
            getattr(ly, "flops_transform", 0) for ly in getattr(c, "layers", [])))

        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["Math", "Quantize", "Transform"]
        values = [flops_math, flops_quant, flops_trans]
        colors_list = ["#0072B2", "#D55E00", "#009E73"]

        bottom = 0
        for i, (cat, val, col) in enumerate(zip(categories, values, colors_list)):
            bar = ax.bar(0, val, 0.5, bottom=bottom, label=cat, color=col, alpha=0.8)
            if val > 0:
                ax.text(0, bottom + val / 2, f"{cat}\n{val:,}", ha="center",
                       va="center", fontsize=9, color="white", fontweight="bold")
            bottom += val

        ax.set_xticks([0])
        ax.set_xticklabels([self._result.name or "(unnamed)"])
        ax.set_ylabel("FLOPs")
        ax.set_title("Cost Decomposition")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    # ── Error propagation ───────────────────────────────────────────────────

    def error_propagation(self, role: str = "output") -> plt.Figure:
        """Accumulated vs local QSNR decomposition, 3-row panel.

        Delegates to :func:`_render_error_propagation`.
        """
        if role not in _VALID_PLOT_ROLES:
            raise ValueError(f"Invalid role {role!r}.")

        corr = self._result.correlate_hook_observer(role=role)
        if not corr or not corr.get("matched"):
            raise ValueError(
                "Error propagation data not available. "
                "Requires qsnr observer (included in default outputs) and keep_fp32=True. "
                + _how_to("qsnr")
            )
        data = {self._result.name or "(unnamed)": corr}
        return _render_error_propagation(data, role)

    # ── Accumulated vs local ────────────────────────────────────────────────

    def accumulated_vs_local(self, role: str = "output") -> plt.Figure:
        """Scatter plot: accumulated QSNR vs local QSNR.

        Delegates to :func:`_render_accumulated_vs_local`.
        """
        if role not in _VALID_PLOT_ROLES:
            raise ValueError(f"Invalid role {role!r}.")

        corr = self._result.correlate_hook_observer(role=role)
        if not corr or not corr.get("matched"):
            raise ValueError(
                "Accumulated vs local data not available. "
                "Requires qsnr observer (included in default outputs) and keep_fp32=True. "
                + _how_to("qsnr")
            )
        data = {self._result.name or "(unnamed)": corr}
        return _render_accumulated_vs_local(data, role)

    # ── Role distribution comparison ────────────────────────────────────────

    def role_distribution_comparison(self) -> plt.Figure:
        """Per-role distribution feature comparison (boxplots)."""
        df = self._build_df()
        needed = {"skewness", "kurtosis", "norm_entropy", "role"}
        if df is False or df.empty:
            raise ValueError("Distribution data not available. " + _how_to("distribution"))
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Distribution features missing: {sorted(missing)}. "
                             + _how_to("distribution"))

        plot_roles = [r for r in ["input", "weight", "output"] if r in df["role"].values]
        if not plot_roles:
            raise ValueError(f"No input/weight/output role data. Roles: {sorted(df['role'].unique())}.")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors_cycle = plt.cm.tab10.colors
        for ax, feature, ylabel in [
            (axes[0], "skewness", "Skewness"),
            (axes[1], "kurtosis", "Kurtosis"),
            (axes[2], "norm_entropy", "Normalized Entropy"),
        ]:
            data_groups, labels = [], []
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

    # ── Kurtosis analysis ───────────────────────────────────────────────

    def kurtosis_analysis(self, roles=("input", "weight", "output")) -> plt.Figure:
        """Kurtosis distribution, QSNR relationship, and top-layer ranking.

        Three-panel figure: kurtosis histogram with reference lines,
        kurtosis vs QSNR scatter by role, and top-15 (layer, role) ranked
        by kurtosis.

        Args:
            roles: Tensor roles to include (default input/weight/output).

        Returns:
            matplotlib Figure with 1×3 panels.
        """
        df = self._build_df()
        if df is False or df.empty or "kurtosis" not in df.columns:
            raise ValueError(
                "Kurtosis data not available — DistributionObserver was not active. "
                + _how_to("distribution", "qsnr")
            )

        role_df = df[df["role"].isin(roles)]
        if role_df.empty:
            raise ValueError(f"No data for roles {list(roles)}.")

        return _render_kurtosis_analysis(role_df, roles)

    # ── Per-layer role distribution histogram ───────────────────────────────

    def per_layer_role_histogram(
        self, k: int = 5, log_y: bool = False,
        op_types: list | None = None,
        qsnr_type: str = "local",
    ) -> plt.Figure:
        """Per-layer, per-role fp32 value distribution for worst-QSNR layers.

        Args:
            k: Number of worst-QSNR layers to show.
            log_y: Use log scale for histogram y-axis.
            op_types: Operator types to include, e.g. ``["linear"]``.
            qsnr_type: ``"local"`` or ``"accum"`` for ranking and labels.
        """
        import torch as _torch

        roles = ("input", "weight", "output")
        obs = self._result.observers_data

        layer_role_hists: dict = {}
        layer_role_dist: dict = {}
        if obs:
            for layer, roles_dict in obs.items():
                for role, stages in roles_dict.items():
                    if role not in roles:
                        continue
                    key = (layer, role)
                    if key in layer_role_hists or key in layer_role_dist:
                        continue
                    for _stage, slices in stages.items():
                        for _slice_key, metrics in slices.items():
                            if "fp32_hist" in metrics:
                                h = metrics["fp32_hist"]
                                if isinstance(h, _torch.Tensor):
                                    h = h.cpu().float().numpy()
                                elif not isinstance(h, np.ndarray):
                                    h = np.asarray(h)
                                if len(h) > 0:
                                    entry = {"fp32_hist": h}
                                    for ch in ("quant_hist", "err_hist"):
                                        v = metrics.get(ch)
                                        if v is not None:
                                            if isinstance(v, _torch.Tensor):
                                                v = v.cpu().float().numpy()
                                            elif not isinstance(v, np.ndarray):
                                                v = np.asarray(v)
                                            if len(v) > 0:
                                                entry[ch] = v
                                    for bound in ("fp32_min", "fp32_max"):
                                        v = metrics.get(bound)
                                        if v is not None:
                                            entry[bound] = float(v)
                                    layer_role_hists[key] = entry
                            if key not in layer_role_dist:
                                dkeys = ("mean", "std", "skewness", "kurtosis", "min", "max")
                                d = {dk: metrics[dk] for dk in dkeys if dk in metrics}
                                if d:
                                    layer_role_dist[key] = d

        if not layer_role_hists and not layer_role_dist:
            raise ValueError("No histogram or distribution data. " + _how_to("histogram", "distribution"))

        # Rank layers by QSNR
        from src.viz._layer_classify import filter_layers_by_type

        accum_lookup = _build_accum_qsnr_lookup(self._result) if qsnr_type == "accum" else {}
        if qsnr_type == "accum":
            qsnr = accum_lookup
        else:
            qsnr = self._result.qsnr_per_layer

        all_layers_set = set(l for l, r in set(layer_role_hists.keys()) | set(layer_role_dist.keys()))
        if qsnr:
            sorted_layers = sorted(qsnr, key=lambda l: qsnr.get(l, float("inf")))
            if op_types:
                sorted_layers = filter_layers_by_type(sorted_layers, op_types)
            bottom_k = [l for l in sorted_layers if l in all_layers_set][:k]
        else:
            all_layers = sorted(all_layers_set)
            if op_types:
                all_layers = filter_layers_by_type(all_layers, op_types)
            bottom_k = all_layers[:k]

        if not bottom_k:
            raise ValueError("No layers found with distribution data.")

        present_roles = [r for r in roles if any(
            (l, r) in layer_role_hists or (l, r) in layer_role_dist for l in bottom_k
        )]
        if not present_roles:
            present_roles = list(roles)

        n_rows, n_cols = len(bottom_k), len(present_roles)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows), squeeze=False)

        for row_idx, layer in enumerate(bottom_k):
            for col_idx, role in enumerate(present_roles):
                ax = axes[row_idx, col_idx]
                key = (layer, role)
                if key in layer_role_hists:
                    entry = layer_role_hists[key]
                    fp32_counts = entry.get("fp32_hist")
                    quant_counts = entry.get("quant_hist")
                    err_counts = entry.get("err_hist")
                    n_bins = len(fp32_counts) if fp32_counts is not None else 64
                    x_bins = np.arange(n_bins)

                    # fp32: blue fill (matching histogram_overlay)
                    if fp32_counts is not None:
                        ax.fill_between(x_bins, fp32_counts, alpha=0.25,
                                        color="#3498db", label="fp32", step="mid")

                    # quant: red dashed outline
                    if quant_counts is not None:
                        ax.plot(x_bins, quant_counts, color="#e74c3c",
                                linewidth=1.2, linestyle="--", label="quant")

                    # x-axis: real value range
                    lo = entry.get("fp32_min")
                    hi = entry.get("fp32_max")
                    if lo is not None and hi is not None:
                        edges = np.linspace(lo, hi, n_bins + 1)
                        tick_idx = np.linspace(0, n_bins - 1, min(5, n_bins)).astype(int)
                        ax.set_xticks(tick_idx)
                        ax.set_xticklabels([f"{edges[i]:.3g}" for i in tick_idx], fontsize=6)
                        ax.set_xlabel("Value", fontsize=7)
                    else:
                        ax.set_xlabel("Bin", fontsize=7)

                    # error: twin axis so small errors are visible
                    if err_counts is not None and err_counts.sum() > 0:
                        ax2 = ax.twinx()
                        ax2.fill_between(x_bins, err_counts, alpha=0.3,
                                         color="#95a5a6", label="error", step="mid")
                        ax2.set_ylabel("Error", fontsize=6)
                        ax2.tick_params(axis="y", labelsize=5)
                        handles1, labels1 = ax.get_legend_handles_labels()
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        ax2.legend(handles1 + handles2, labels1 + labels2,
                                  fontsize=5, loc="upper right")
                    else:
                        ax.legend(fontsize=5, loc="upper right")
                elif key in layer_role_dist:
                    d = layer_role_dist[key]
                    lines = [f"mean={d.get('mean', 0):.2g}", f"std={d.get('std', 0):.2g}",
                             f"skew={d.get('skewness', 0):.2f}", f"kurt={d.get('kurtosis', 0):.2f}"]
                    ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
                            ha="center", va="center", fontsize=8,
                            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="gray")

                qsnr_str = ""
                if qsnr_type == "accum":
                    qsnr_val = accum_lookup.get(layer)
                    if qsnr_val is not None:
                        qsnr_str = f"accum={qsnr_val:.1f}dB"
                else:
                    qsnr_val = self._result.qsnr_per_layer.get(layer)
                    if qsnr_val is not None and qsnr_val == qsnr_val:
                        qsnr_str = f"local={qsnr_val:.1f}dB"
                short = _short_layer_name(layer)
                ax.set_title(f"{short}\n{role} {qsnr_str}", fontsize=7)
                if row_idx == n_rows - 1 and key not in layer_role_hists:
                    ax.set_xlabel("Bin", fontsize=7)
                if col_idx == 0:
                    ax.set_ylabel("Count", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)
                if log_y and key in layer_role_hists:
                    ax.set_yscale("log")

        fig.suptitle("Per-Layer fp32 Value Distribution — Worst-QSNR Layers", fontsize=12, y=1.01)
        fig.tight_layout()
        return fig

    # ── Error propagation (new viz) ──────────────────────────────────────

    def propagation_dag(self, *, qsnr_cap=None, skip_activations=True) -> plt.Figure:
        """Horizontal bar chart: local QSNR per layer with accum markers.

        Args:
            qsnr_cap: If set, clip QSNR values to this maximum.
            skip_activations: If True (default), exclude ReLU/GELU/etc. layers.
        """
        from src.viz._propagation import plot_propagation_dag
        return plot_propagation_dag(self._result, qsnr_cap=qsnr_cap,
                                    skip_activations=skip_activations)

    def error_waterfall(self, *, qsnr_cap=60.0, skip_activations=True) -> plt.Figure:
        """Waterfall chart: accumulated QSNR dropping layer by layer.

        Args:
            qsnr_cap: Clip accumulated QSNR values to this maximum (default 60).
            skip_activations: If True (default), exclude ReLU/GELU/etc. layers.
        """
        from src.viz._propagation import plot_error_waterfall
        return plot_error_waterfall(self._result, qsnr_cap=qsnr_cap,
                                    skip_activations=skip_activations)

    def local_vs_accum_scatter(self, *, qsnr_cap=None, skip_activations=True) -> plt.Figure:
        """Scatter: local vs accumulated QSNR with headroom colouring.

        Args:
            qsnr_cap: If set, clip QSNR values to this maximum.
            skip_activations: If True (default), exclude ReLU/GELU/etc. layers.
        """
        from src.viz._propagation import plot_local_vs_accum_scatter
        return plot_local_vs_accum_scatter(self._result, qsnr_cap=qsnr_cap,
                                           skip_activations=skip_activations)

    # ── Per-role ────────────────────────────────────────────────────────

    def per_role_qsnr_bars(self, max_layers: int = 30, sort_by: str = "worst",
                           qsnr_cap=None, skip_activations: bool = True) -> plt.Figure:
        """Grouped bar chart: input / weight / output QSNR per layer.

        Args:
            max_layers: Maximum number of layers to display.
            sort_by: ``"worst"`` — sort by the lowest QSNR across all roles.
                     ``"depth"`` — keep model order.
            qsnr_cap: If set, clip QSNR values to this maximum.
            skip_activations: If True (default), exclude ReLU/GELU/etc. layers.
        """
        from src.viz._per_role import plot_per_role_qsnr_bars
        return plot_per_role_qsnr_bars(self._result, max_layers=max_layers,
                                       sort_by=sort_by, qsnr_cap=qsnr_cap,
                                       skip_activations=skip_activations)

    def depth_decay(self, role: str = "output", *, qsnr_cap=None,
                    skip_activations: bool = True) -> plt.Figure:
        """QSNR vs depth line plot for a single role.

        Args:
            role: ``"input"`` / ``"weight"`` / ``"output"``.
            qsnr_cap: If set, clip QSNR values to this maximum.
            skip_activations: If True (default), exclude ReLU/GELU/etc. layers.
        """
        from src.viz._per_role import plot_depth_decay
        return plot_depth_decay(self._result, role=role, qsnr_cap=qsnr_cap,
                                skip_activations=skip_activations)

    def per_layer_role_qsnr_line(
        self,
        qsnr_type: str = "local",
        op_types: list | None = None,
        skip_activations: bool = True,
        qsnr_cap=None,
    ) -> plt.Figure:
        """Per-layer QSNR line chart with one line per role (local) or output only (accum).

        Args:
            qsnr_type: ``"local"`` → input/weight/output lines.
                       ``"accum"`` → output line only.
            op_types: Operator types to include, e.g. ``["linear"]``.
            skip_activations: If True (default), exclude activation layers.
            qsnr_cap: If set, clip QSNR values to this maximum.
        """
        from src.viz._per_role import plot_per_layer_role_qsnr_line
        return plot_per_layer_role_qsnr_line(
            self._result,
            qsnr_type=qsnr_type,
            op_types=op_types,
            skip_activations=skip_activations,
            qsnr_cap=qsnr_cap,
        )

    # ── Distribution ───────────────────────────────────────────────────

    def layer_histogram(self, layer: str, role: str = "weight", log_y: bool = False) -> plt.Figure:
        """Histogram overlay: fp32 vs quantised distribution for a layer/role.

        Requires HistogramObserver data (``outputs=["histogram"]``).
        Falls back to a text placeholder when histogram data is absent.

        Args:
            layer: Module name.
            role: ``"input"`` / ``"weight"`` / ``"output"``.
            log_y: If True, use log scale for the y-axis (counts).
        """
        hist_data = self._get_histogram_data(layer, role)
        if hist_data is None:
            fig, ax = plt.subplots()
            ax.text(
                0.5, 0.5,
                f"No histogram data for {layer} ({role}).\n"
                "Run session.analyze(calib_data, outputs=['histogram'])\n"
                "to collect distribution data.",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
            )
            ax.set_title(f"Distribution: {layer} ({role})")
            return fig

        fp32_h = hist_data.get("fp32_hist")
        quant_h = hist_data.get("quant_hist")
        err_h = hist_data.get("err_hist")

        import torch
        n_bins = len(fp32_h) if fp32_h is not None else 128
        x = range(n_bins)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        # Top panel: fp32 vs quant overlay
        if fp32_h is not None:
            ax1.fill_between(x, fp32_h.float().numpy(), alpha=0.5, color="#3498db",
                             step="mid", label="fp32")
        if quant_h is not None:
            ax1.plot(x, quant_h.float().numpy(), color="#e74c3c", linewidth=1.2,
                     alpha=0.9, label="quant")
        ax1.legend(fontsize=8)
        ax1.set_ylabel("Count", fontsize=8)
        ax1.grid(alpha=0.2)

        qsnr = self._result.qsnr_by_role.get(role, {}).get(layer)
        title = f"{layer} ({role})"
        if qsnr is not None:
            title += f"  —  QSNR={qsnr:.1f} dB"
        ax1.set_title(title)

        # Bottom panel: error histogram
        if err_h is not None:
            err_vals = err_h.float().numpy()
            pos_mask = err_vals >= 0
            neg_mask = ~pos_mask
            ax2.fill_between(x, err_vals, where=pos_mask, alpha=0.5,
                             color="#e74c3c", step="mid", label="error > 0")
            ax2.fill_between(x, err_vals, where=neg_mask, alpha=0.5,
                             color="#2ecc71", step="mid", label="error < 0")
            ax2.legend(fontsize=8)
        ax2.set_xlabel("Bin", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        ax2.grid(alpha=0.2)

        if log_y:
            ax1.set_yscale("log")
            ax2.set_yscale("log")

        fig.tight_layout()
        return fig

    def channel_heterogeneity(self, layer: str, role: str = "weight") -> plt.Figure:
        """Per-channel QSNR distribution (violin or box plot) for a layer/role.

        Computes per-channel QSNR from observers_data and displays a box plot.
        Falls back to a text placeholder when per-channel data is absent.
        """
        per_ch_qsnr = self._get_per_channel_qsnr(layer, role)
        if per_ch_qsnr is None or len(per_ch_qsnr) == 0:
            fig, ax = plt.subplots()
            ax.text(
                0.5, 0.5,
                f"No per-channel data for {layer} ({role}).",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(f"Channel Heterogeneity: {layer} ({role})")
            return fig

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(per_ch_qsnr, vert=False, widths=0.6,
                   medianprops={"color": "black", "linewidth": 1},
                   flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
        ax.set_xlabel("QSNR (dB)")
        ax.set_title(f"Per-Channel QSNR: {layer} ({role})  —  "
                     f"{len(per_ch_qsnr)} channels")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    # ── Histogram overlay ──────────────────────────────────────────────────

    def histogram_overlay(
        self,
        top_k: int = 5,
        role: str | None = None,
        layer: str | None = None,
        op_types: list | None = None,
        qsnr_type: str = "local",
    ) -> plt.Figure:
        """Three-channel histogram overlay (fp32 / quant / error).

        Two modes:

        - **Top-K mode** (default, ``layer=None``): shows the most
          quantization-sensitive (layer, role) pairs ranked by QSNR.
        - **Single-layer mode** (``layer="fc1"``): shows the input / weight /
          output histograms for one specific layer in three subplots.

        Requires ``outputs=["histogram", "qsnr"]`` (or ``"all"``).

        Args:
            top_k: Number of most-sensitive (layer, role) pairs (top-K mode).
            role: Filter by tensor role (top-K mode). ``None`` = all roles.
            layer: Specific layer name → single-layer mode.
            op_types: Operator types to include, e.g. ``["linear", "conv"]``.
                ``None`` = all types.
            qsnr_type: ``"local"`` (observer) or ``"accum"`` (hook) for
                ranking and labelling. Accum only applies to output role.

        Returns:
            matplotlib Figure.
        """
        from src.viz._layer_classify import filter_layers_by_type

        if qsnr_type not in ("local", "accum"):
            raise ValueError(f"qsnr_type must be 'local' or 'accum', got {qsnr_type!r}")

        if layer is not None:
            return self._histogram_single_layer(layer, op_types, qsnr_type)

        roles = ("input", "weight", "output") if role is None else (role,)

        # In accum mode only output role has data
        if qsnr_type == "accum" and role is None:
            roles = ("output",)

        accum_lookup = _build_accum_qsnr_lookup(self._result) if qsnr_type == "accum" else {}

        # Collect histogram data and QSNR
        layer_hists: dict = {}
        layer_error: dict = {}

        for lname in sorted(self._result.observers_data.keys()):
            if op_types and filter_layers_by_type([lname], op_types) == []:
                continue
            for r in roles:
                metrics = self._get_histogram_data(lname, r)
                if metrics is None:
                    continue
                fp32_hist = metrics.get("fp32_hist")
                quant_hist = metrics.get("quant_hist")
                if fp32_hist is None or quant_hist is None:
                    continue
                key = f"{lname} [{r}]"
                hist_data = {}
                for k in ("fp32_hist", "quant_hist", "err_hist",
                          "fp32_min", "fp32_max"):
                    v = metrics.get(k)
                    if v is not None:
                        if k in ("fp32_min", "fp32_max"):
                            hist_data[k] = float(v)
                        else:
                            hist_data[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v
                if "fp32_hist" not in hist_data or "quant_hist" not in hist_data:
                    continue
                layer_hists[key] = hist_data

                # Get QSNR for ranking
                if qsnr_type == "accum":
                    qsnr = accum_lookup.get(lname)
                else:
                    role_dict = self._result.qsnr_by_role.get(r, {})
                    qsnr = role_dict.get(lname)
                if qsnr is not None and not (math.isnan(qsnr) if isinstance(qsnr, float) else False):
                    layer_error[key] = qsnr

        if not layer_hists:
            raise ValueError(
                "Histogram data not available. "
                "Run Session with outputs=[\"histogram\"] (or \"all\") "
                "to enable HistogramObserver."
            )

        # Rank by sensitivity: lowest QSNR first
        if layer_error:
            top_layers = sorted(
                layer_hists.items(),
                key=lambda x: layer_error.get(x[0], float("inf")),
            )[:top_k]
        else:
            print("  Warning: No QSNR data for sensitivity ranking, "
                  "falling back to histogram magnitude")
            top_layers = sorted(
                layer_hists.items(),
                key=lambda x: x[1].get("fp32_hist", np.array(0)).sum(),
                reverse=True,
            )[:top_k]

        n = len(top_layers)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

        for ax, (layer_key, hist_data) in zip(axes[0], top_layers):
            _render_hist_subplot(ax, hist_data)

            # Title with QSNR
            q_label = f"  local={layer_error.get(layer_key, float('nan')):.1f}dB" if layer_error else ""
            ax.set_title(f"{layer_key}{q_label}", fontsize=9)

        role_label = "" if role is None else f" ({role})"
        q_label2 = "accum" if qsnr_type == "accum" else "local"
        fig.suptitle(f"Activation Histograms (fp32 / quant / error){role_label} — "
                     f"Most Sensitive Layers ({q_label2} QSNR)", fontsize=13)
        fig.tight_layout()
        return fig

    def _histogram_single_layer(
        self, layer: str, op_types: list | None, qsnr_type: str
    ) -> plt.Figure:
        """Single-layer mode: input / weight / output histograms in 1×3 grid."""
        accum_lookup = _build_accum_qsnr_lookup(self._result)
        roles = ("input", "weight", "output")

        hist_data_per_role: dict = {}
        for r in roles:
            metrics = self._get_histogram_data(layer, r)
            if metrics is None:
                hist_data_per_role[r] = None
                continue
            fp32_hist = metrics.get("fp32_hist")
            quant_hist = metrics.get("quant_hist")
            if fp32_hist is None or quant_hist is None:
                hist_data_per_role[r] = None
                continue
            entry = {}
            for k in ("fp32_hist", "quant_hist", "err_hist", "fp32_min", "fp32_max"):
                v = metrics.get(k)
                if v is not None:
                    if k in ("fp32_min", "fp32_max"):
                        entry[k] = float(v)
                    else:
                        entry[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v
            hist_data_per_role[r] = entry

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
        short = _short_layer_name(layer)

        for col_idx, r in enumerate(roles):
            ax = axes[0, col_idx]
            hd = hist_data_per_role.get(r)
            if hd is not None:
                _render_hist_subplot(ax, hd)

                # Build QSNR title line
                local_q = None
                role_dict = self._result.qsnr_by_role.get(r, {})
                if layer in role_dict:
                    local_q = role_dict[layer]
                accum_q = accum_lookup.get(layer) if r == "output" else None

                parts = []
                if local_q is not None and not (math.isnan(local_q) if isinstance(local_q, float) else False):
                    parts.append(f"local={local_q:.1f}dB")
                if accum_q is not None:
                    parts.append(f"accum={accum_q:.1f}dB")

                q_str = "  ".join(parts) if parts else ""
                ax.set_title(f"{short}\n{r}  {q_str}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "no histogram data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
                ax.set_title(f"{short}\n{r}", fontsize=9)

        fig.suptitle(f"Activation Histograms — {short}", fontsize=13)
        fig.tight_layout()
        return fig

    # ── Distribution data helpers ───────────────────────────────────────

    def _get_histogram_data(self, layer: str, role: str) -> dict | None:
        """Extract HistogramObserver data for a (layer, role) pair."""
        obs = self._result.observers_data
        stages = obs.get(layer, {}).get(role, {})
        for _stage, slices in stages.items():
            for _slice_key, metrics in slices.items():
                if "fp32_hist" in metrics:
                    return metrics
        return None

    def _get_per_channel_qsnr(self, layer: str, role: str) -> list | None:
        """Extract per-channel QSNR values from observers_data.

        This works when the observer was configured for per-channel measurement
        (the ``_measure_per_unit`` path in SliceAwareObserver).
        """
        obs = self._result.observers_data
        stages = obs.get(layer, {}).get(role, {})
        values = []
        for _stage, slices in stages.items():
            for _slice_key, metrics in slices.items():
                if isinstance(metrics, list):
                    for item in metrics:
                        if isinstance(item, dict) and "qsnr_db" in item:
                            v = item["qsnr_db"]
                            if v == v:
                                values.append(v)
                elif isinstance(metrics, dict) and "qsnr_db" in metrics:
                    pass
        return values if values else None


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_accum_qsnr_lookup(result) -> dict:
    """Map observer layer names to accum QSNR values via prefix matching.

    Uses the same matching logic as
    :meth:`SessionResult.correlate_hook_observer`: an observer key matches
    a hook key if they are equal or the observer key starts with the hook
    key followed by a dot.
    """
    accum = result.accum_qsnr_per_layer
    if not accum:
        return {}
    hook_keys = sorted(accum.keys(), key=len, reverse=True)
    obs_keys = set(result.observers_data.keys())
    lookup: dict = {}
    for obs in obs_keys:
        for hk in hook_keys:
            if obs == hk or obs.startswith(hk + "."):
                lookup[obs] = accum[hk]
                break
    return lookup


def _render_hist_subplot(ax, hist_data: dict) -> None:
    """Render fp32 fill + quant dashed outline + error twin-axis on *ax*."""
    fp32_counts = hist_data.get("fp32_hist")
    quant_counts = hist_data.get("quant_hist")
    err_counts = hist_data.get("err_hist")

    n_bins = (
        len(fp32_counts) if fp32_counts is not None
        else len(quant_counts) if quant_counts is not None
        else 64
    )
    x_bins = np.arange(n_bins)

    if fp32_counts is not None and isinstance(fp32_counts, np.ndarray):
        ax.fill_between(x_bins, fp32_counts, alpha=0.25,
                        color="#3498db", label="fp32", step="mid")

    if quant_counts is not None and isinstance(quant_counts, np.ndarray):
        ax.plot(x_bins, quant_counts, color="#e74c3c",
                linewidth=1.2, linestyle="--", label="quant")

    fp32_min = hist_data.get("fp32_min")
    fp32_max = hist_data.get("fp32_max")
    if fp32_min is not None and fp32_max is not None:
        edges = np.linspace(fp32_min, fp32_max, n_bins + 1)
        tick_idx = np.linspace(0, n_bins - 1, min(5, n_bins)).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{edges[i]:.3g}" for i in tick_idx], fontsize=7)
        ax.set_xlabel("Value")
    else:
        ax.set_xlabel("Bin")

    if err_counts is not None and isinstance(err_counts, np.ndarray) and err_counts.sum() > 0:
        ax2 = ax.twinx()
        ax2.fill_between(x_bins, err_counts, alpha=0.3,
                         color="#95a5a6", label="error", step="mid")
        ax2.set_ylabel("Error count", fontsize=7)
        ax2.tick_params(axis="y", labelsize=6)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles1 + handles2, labels1 + labels2,
                   fontsize=7, loc="upper right")
    else:
        ax.legend(fontsize=7, loc="upper right")

    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)


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


def _render_kurtosis_analysis(df, roles=("input", "weight", "output")) -> plt.Figure:
    """Shared kurtosis analysis rendering for StudyPlotAccessor and SessionPlotAccessor.

    Args:
        df: DataFrame with columns ``kurtosis``, ``qsnr_db``, ``role``, ``layer``.
        roles: Roles to include.

    Returns:
        matplotlib Figure with 1×3 panels.
    """
    role_colors = {"input": "#0072B2", "weight": "#D55E00", "output": "#009E73"}
    fallback = plt.cm.tab10.colors

    kurt_vals = df["kurtosis"].dropna().tolist()
    if not kurt_vals:
        raise ValueError("No valid kurtosis values found.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # -- Panel 1: kurtosis histogram ---------------------------------------
    ax = axes[0]
    counts, bins, _ = ax.hist(kurt_vals, bins=40, color="#3498db", alpha=0.7,
                               edgecolor="white")
    y_max = counts.max()
    for threshold, label, style in [
        (3.0, "normal (3)", "dashed"),
        (6.0, "heavy-tailed (6)", "solid"),
        (10.0, "extreme (10)", "dotted"),
    ]:
        ax.axvline(x=threshold, color="black", linestyle=style,
                   linewidth=1.0, alpha=0.6)
        ax.text(threshold, y_max * 0.92, label,
                rotation=90, va="top", ha="right", fontsize=7,
                color="black", alpha=0.7)
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("Count (layers × roles)")
    ax.set_title("Kurtosis Distribution\n(vertical lines: normal / heavy / extreme)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xscale("log")

    # -- Panel 2: kurtosis vs QSNR scatter --------------------------------
    ax = axes[1]
    for role in roles:
        role_df = df[df["role"] == role]
        if role_df.empty:
            continue
        color = role_colors.get(role, fallback[0])
        has_qsnr = "qsnr_db" in df.columns
        if has_qsnr:
            ax.scatter(role_df["kurtosis"], role_df["qsnr_db"],
                      label=role, color=color, alpha=0.7, s=35)
        else:
            ax.scatter(role_df["kurtosis"], [0] * len(role_df),
                      label=role, color=color, alpha=0.7, s=35)

    ax.axvline(x=3.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="k=3 (normal)")
    ax.axvline(x=6.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
               label="k=6 (heavy-tailed)")
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("QSNR (dB)" if "qsnr_db" in df.columns else "(no QSNR)")
    ax.set_title("Kurtosis vs QSNR by Role")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # -- Panel 3: top-15 (layer, role) ranked by kurtosis -----------------
    ax = axes[2]
    top_n = min(15, len(df))
    ranked = df.nlargest(top_n, "kurtosis")
    labels = []
    kv = []
    colors_list = []
    for _, row in ranked.iterrows():
        short = _short_layer_name(row["layer"])[:18]
        labels.append(f"{short}|{row['role'][:3]}")
        kv.append(row["kurtosis"])
        colors_list.append(role_colors.get(row["role"], fallback[0]))

    y_pos = range(len(labels))
    ax.barh(y_pos, kv, color=colors_list, alpha=0.7, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(x=3.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axvline(x=6.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Kurtosis")
    ax.set_title(f"Top-{top_n} by Kurtosis (layer | role)")
    ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=r) for r, c in role_colors.items()
                      if r in roles]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")

    fig.suptitle("Kurtosis Analysis — Distribution, QSNR Relationship, Top Layers",
                 fontsize=13)
    fig.tight_layout()
    return fig
