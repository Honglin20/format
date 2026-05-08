"""StudyPlotAccessor — post-hoc visualization on StudyReport."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.report._study_report import StudyReport


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
        """
        df = self._report.to_dataframe()
        if df is None or df.empty or "qsnr_db" not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No QSNR data available",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title("QSNR per Layer")
            return fig

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
            role: Tensor role to plot (``"input"``, ``"weight"``, ``"output"``).

        Returns:
            matplotlib Figure.
        """
        df = self._report.to_dataframe()
        needed = {"crest_factor", "qsnr_db"}
        if df is None or df.empty or not needed.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5,
                    "Need crest_factor + qsnr_db\n"
                    "(ensure DistributionObserver and QSNRObserver are active)",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title("Crest Factor vs QSNR")
            return fig

        role_df = df[df["role"] == role]
        if role_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"No data for role '{role}'",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title("Crest Factor vs QSNR")
            return fig

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


def _short_layer_name(name: str) -> str:
    """Shorten a full module path for x-axis labels."""
    name = name.replace("module.", "").replace("Quantized", "")
    return name[:20]
