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

    def per_layer_qsnr(self, max_layers: int = 60) -> str:
        """Per-layer QSNR comparison table across all configs.

        One row per layer, one column per config. Rows sorted by worst QSNR
        across configs so the most quantization-sensitive layers appear first.

        Args:
            max_layers: Maximum layers to display (default 60). Extra layers
                are summarised at the bottom. Use 0 for unlimited.

        Returns:
            Formatted text table.
        """
        # Collect {layer: {config_name: qsnr}} from all parts
        all_layers: dict[str, dict[str, float]] = {}
        configs: list[str] = []

        for part_results in self._report._results.values():
            for r in part_results:
                cfg_name = r.name or "(unnamed)"
                if cfg_name not in configs:
                    configs.append(cfg_name)
                for layer, qsnr in r.qsnr_per_layer.items():
                    all_layers.setdefault(layer, {})[cfg_name] = qsnr

        if not all_layers:
            return (
                "No QSNR per-layer data found.\n"
                "Ensure session.analyze() or session.run() is called "
                "with outputs=['qsnr'] (enabled by default)."
            )

        configs.sort()

        # Rank layers by worst QSNR (minimum across configs)
        ranked: list[tuple[str, float]] = []
        for layer in all_layers:
            vals = [v for v in all_layers[layer].values() if v == v]
            min_qsnr = min(vals) if vals else float("inf")
            ranked.append((layer, min_qsnr))
        ranked.sort(key=lambda x: x[1])

        layer_order = [l for l, _ in ranked]

        name_w = max(
            len(l.replace("module.", "").replace("Quantized", ""))
            for l in layer_order
        )
        name_w = min(max(name_w + 2, 10), 32)
        val_w = 12

        header = f"{'Layer':<{name_w}}" + "".join(
            f" {cfg:<{val_w}}" for cfg in configs
        )
        sep = "-" * len(header)
        lines = [
            f"\n{'=' * len(header)}",
            "Per-Layer QSNR (dB) — Lower = more quantization-sensitive",
            "=" * len(header),
            header,
            sep,
        ]

        shown = layer_order if max_layers <= 0 else layer_order[:max_layers]
        hidden_count = (
            0 if max_layers <= 0 else len(layer_order) - max_layers
        )

        for layer in shown:
            short = layer.replace("module.", "").replace("Quantized", "")[:name_w]
            row = f"{short:<{name_w}}"
            for cfg in configs:
                val = all_layers[layer].get(cfg)
                if val is not None and val == val:
                    row += f" {val:<{val_w}.2f}"
                else:
                    row += f" {'-':<{val_w}}"
            lines.append(row)

        if hidden_count > 0:
            lines.append(
                f"\n  ... {hidden_count} more layers (omit for brevity; "
                f"use max_layers=0 for full list)"
            )

        return "\n".join(lines)

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
