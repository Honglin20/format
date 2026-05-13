"""StudyTablesAccessor — terminal table output on StudyReport."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.report._study_report import StudyReport


class StudyTablesAccessor:
    """Terminal table methods on :class:`StudyReport`.

    Usage::

        report = Study(configs, model).run(data, eval_fn)
        print(report.tables.per_layer_qsnr())
    """

    def __init__(self, report: "StudyReport"):
        self._report = report

    # ── Per-Layer QSNR ─────────────────────────────────────────────────

    def per_layer_qsnr(self, max_layers: int = 60, qsnr_type: str = "local") -> str:
        """Per-layer QSNR comparison table across all configs.

        One row per layer, one column per config. Rows sorted by worst QSNR
        across configs so the most quantization-sensitive layers appear first.

        Args:
            max_layers: Maximum layers to display (default 60). Extra layers
                are summarised at the bottom. Use 0 for unlimited.
            qsnr_type: ``"local"`` (default) reads per-op observer QSNR.
                ``"accum"`` reads end-to-end accumulated hook QSNR.

        Returns:
            Formatted text table.
        """
        from src.viz.tables import _format_per_layer_qsnr_table

        all_layers: dict[str, dict[str, float]] = {}
        configs: list[str] = []

        for part_results in self._report._results.values():
            for r in part_results:
                cfg_name = r.name or "(unnamed)"
                if cfg_name not in configs:
                    configs.append(cfg_name)
                qsnr_dict = r.accum_qsnr_per_layer if qsnr_type == "accum" else r.qsnr_per_layer
                for layer, qsnr in qsnr_dict.items():
                    all_layers.setdefault(layer, {})[cfg_name] = qsnr

        if not all_layers:
            return (
                "No QSNR per-layer data found.\n"
                "Ensure session.analyze() or session.run() is called "
                "with outputs=['qsnr'] (enabled by default)."
            )

        configs.sort()
        label = "accum" if qsnr_type == "accum" else "output"
        return _format_per_layer_qsnr_table(
            all_layers, configs, max_layers=max_layers,
            title=f"Per-Layer QSNR (dB, {label}) — Lower = more quantization-sensitive",
        )

    # ── Error source analysis ─────────────────────────────────────────

    def error_source_analysis(self, role: str = "output") -> str:
        """Per-layer error source diagnosis: accumulated vs local QSNR.

        Delegates to :meth:`SessionTablesAccessor.error_source_analysis`
        for each config, concatenating the per-config tables.

        Args:
            role: Tensor role to analyse (default ``"output"``).

        Returns:
            Formatted text table.
        """
        from src.report._session_tables import SessionTablesAccessor

        blocks: list[str] = []
        for part_results in self._report._results.values():
            for r in part_results:
                block = SessionTablesAccessor(r).error_source_analysis(role=role)
                if block and not block.startswith("No "):
                    blocks.append(block)

        if not blocks:
            return (
                "No error propagation data available.\n"
                "Requires QSNRObserver active (included in default outputs)\n"
                "and keep_fp32=True (default).\n"
                "session.run(calib_data, outputs=['qsnr'])"
            )

        return "\n".join(blocks)

    # ── CSV helpers ─────────────────────────────────────────────────────

    def _per_layer_qsnr_rows(self) -> list[dict]:
        """Return tidy rows for CSV export. Used by save()."""
        rows: list[dict] = []
        for part_results in self._report._results.values():
            for r in part_results:
                for layer, qsnr in r.qsnr_per_layer.items():
                    rows.append({
                        "part": next(
                            (p for p, v in self._report._results.items()
                             if r in v), ""
                        ),
                        "config": r.name,
                        "layer": layer,
                        "qsnr_db": qsnr,
                    })
        return rows

    def save_per_layer_qsnr_csv(self, output_dir: str) -> str | None:
        """Write per_layer_qsnr.csv and return the path, or None if no data."""
        rows = self._per_layer_qsnr_rows()
        if not rows:
            return None

        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        csv_path = f"{output_dir}/tables/per_layer_qsnr.csv"

        # Pivot: layer × config
        configs: list[str] = sorted(
            set(r["config"] for r in rows if r["config"])
        )

        # Collect layer → {config → qsnr}
        layer_data: dict[str, dict[str, float]] = {}
        for row in rows:
            layer_data.setdefault(row["layer"], {})[row["config"]] = row["qsnr_db"]

        # Sort by worst QSNR
        ranked = []
        for layer, cfg_qsnrs in layer_data.items():
            vals = [v for v in cfg_qsnrs.values() if v == v]
            ranked.append((layer, min(vals) if vals else float("inf")))
        ranked.sort(key=lambda x: x[1])
        layer_order = [l for l, _ in ranked]

        with open(csv_path, "w") as f:
            f.write("Layer," + ",".join(f"{c}_QSNR_dB" for c in configs) + "\n")
            for layer in layer_order:
                vals = []
                for cfg in configs:
                    v = layer_data[layer].get(cfg)
                    vals.append(f"{v:.4f}" if isinstance(v, (int, float)) and v == v else "")
                f.write(f"{layer}," + ",".join(vals) + "\n")

        return csv_path
