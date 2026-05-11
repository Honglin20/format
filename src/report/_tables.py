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
        from src.viz.tables import _format_per_layer_qsnr_table

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
        return _format_per_layer_qsnr_table(
            all_layers, configs, max_layers=max_layers,
        )

    # ── Error source analysis ─────────────────────────────────────────

    def error_source_analysis(self, role: str = "output") -> str:
        """Per-layer error source diagnosis: accumulated vs local QSNR.

        For each config and matched layer, shows accumulated QSNR (hook
        path, true_error), local QSNR (observer path), delta-QSNR
        (drop from previous layer), headroom, and a diagnosis.

        Diagnosis thresholds:
          - headroom < 3 dB  → Source
          - headroom 3-10 dB → Mixed
          - headroom > 10 dB → Propagated

        Args:
            role: Tensor role to analyse (default ``"output"``).

        Returns:
            Formatted text table.
        """
        data = self._report._correlate_hook_observer(role)

        if not data:
            return (
                "No error propagation data available.\n"
                "Requires both true_error=True and QSNRObserver active.\n"
                "session.run(calib_data, outputs=['qsnr'], true_error=True)"
            )

        lines: list[str] = []
        for cfg_name, info in data.items():
            matched = info["matched"]
            if not matched:
                lines.append(
                    f"\n  {cfg_name}: no matched hook/observer layers — "
                    "skipping"
                )
                continue

            lines.append(f"\n{'=' * 105}")
            lines.append(
                f"  Error Source Analysis — {cfg_name} [{role}]"
            )
            lines.append(f"{'=' * 105}")

            hdr = (
                f"{'Layer':<28} "
                f"{'Accum QSNR':>12} {'Local QSNR':>12} "
                f"{'Delta':>10} {'Headroom':>10}  Diagnosis"
            )
            lines.append(hdr)
            lines.append("-" * len(hdr))

            prev_acc = None
            sources = propagated = mixed = 0
            for hook_key, acc_qsnr, loc_qsnr in matched:
                delta = (
                    prev_acc - acc_qsnr if prev_acc is not None else 0.0
                )
                headroom = loc_qsnr - acc_qsnr
                prev_acc = acc_qsnr

                if headroom < 3.0:
                    diagnosis = "Source"
                    sources += 1
                elif headroom < 10.0:
                    diagnosis = "Mixed"
                    mixed += 1
                else:
                    diagnosis = "Propagated"
                    propagated += 1

                short = hook_key.replace("module.", "").replace(
                    "Quantized", ""
                )[:28]
                lines.append(
                    f"{short:<28} "
                    f"{acc_qsnr:>12.2f} {loc_qsnr:>12.2f} "
                    f"{delta:>+10.2f} {headroom:>+10.2f}  {diagnosis}"
                )

            # Summary line
            if len(matched) >= 2:
                total_drop = matched[0][1] - matched[-1][1]
                headrooms = [l - a for _, a, l in matched]
                avg_headroom = sum(headrooms) / len(headrooms)
                lines.append("-" * len(hdr))
                lines.append(
                    f"{'Summary:':<28} "
                    f"{'':>12} {'':>12} "
                    f"drop={total_drop:>+.1f} "
                    f"avg_headroom={avg_headroom:>+.1f}  "
                    f"{sources} source, {mixed} mixed, "
                    f"{propagated} propagated"
                )

            # Observer-only layers
            obs_only = info["observer_only"]
            if obs_only:
                lines.append(
                    f"\n  Observer-only (no hook data):"
                )
                for obs_key, loc_qsnr in obs_only:
                    short = obs_key[:36]
                    lines.append(
                        f"    {short:<36} local={loc_qsnr:.2f} dB"
                    )

            # Hook-only layers
            hook_only = info["hook_only"]
            if hook_only:
                lines.append(
                    f"\n  Hook-only (no observer data):"
                )
                for hk, acc_qsnr in hook_only:
                    short = hk[:36]
                    lines.append(
                        f"    {short:<36} accum={acc_qsnr:.2f} dB"
                    )

        return "\n".join(lines) if lines else "No data."

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
