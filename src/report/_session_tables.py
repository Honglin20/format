"""SessionTablesAccessor — terminal table output on SessionResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.session._result import SessionResult


class SessionTablesAccessor:
    """Terminal table methods on :class:`SessionResult`.

    Usage::

        result = Session(model, cfg).run(calib_data)
        print(result.tables.error_source_analysis())
        print(result.tables.per_layer_qsnr())
    """

    def __init__(self, result: "SessionResult"):
        self._result = result

    # ── Per-Layer QSNR ─────────────────────────────────────────────────

    def per_layer_qsnr(self, max_layers: int = 60, qsnr_type: str = "local") -> str:
        """Per-layer QSNR table for a single result.

        One row per layer, sorted by worst QSNR first.

        Args:
            max_layers: Maximum layers to display (default 60). Use 0 for
                unlimited.
            qsnr_type: ``"local"`` (default) reads per-op observer QSNR.
                ``"accum"`` reads end-to-end accumulated hook QSNR.

        Returns:
            Formatted text table.
        """
        from src.viz.tables import _format_per_layer_qsnr_table

        qsnr = (
            self._result.accum_qsnr_per_layer if qsnr_type == "accum"
            else self._result.qsnr_per_layer
        )
        if not qsnr:
            return (
                "No QSNR per-layer data found.\n"
                "Ensure session.analyze() or session.run() is called "
                "with outputs=['qsnr'] (enabled by default)."
            )

        cfg_name = self._result.name or "(unnamed)"
        label = "accum" if qsnr_type == "accum" else "output"
        all_layers = {layer: {cfg_name: v} for layer, v in qsnr.items()}
        return _format_per_layer_qsnr_table(
            all_layers, [cfg_name], max_layers=max_layers,
            title=f"Per-Layer QSNR (dB, {label}) — {cfg_name}",
        )

    # ── Error source analysis ─────────────────────────────────────────

    def error_source_analysis(self, role: str = "output") -> str:
        """Per-layer error source diagnosis: accumulated vs local QSNR.

        Compares accumulated QSNR (hook path comparing quant vs fp32 output)
        against local QSNR (observer path, pre-quant vs post-quant) to
        diagnose whether each layer's error is self-inflicted or propagated
        from upstream.

        Diagnosis thresholds:
          - headroom < 3 dB  → Source
          - headroom 3-10 dB → Mixed
          - headroom > 10 dB → Propagated

        Args:
            role: Tensor role to analyse (default ``"output"``).

        Returns:
            Formatted text table.
        """
        accum = self._result.accum_qsnr_per_layer
        if not accum:
            return (
                "No accumulated QSNR data available.\n"
                "Ensure session.run() is called with outputs=['qsnr'] "
                "(enabled by default) and keep_fp32=True (default)."
            )

        local, _ = self._result.qsnr_per_role(role=role)
        if not local:
            return (
                f"No local QSNR data for role='{role}'.\n"
                "Ensure QSNRObserver is active (outputs=['qsnr'], enabled by default)."
            )

        cfg_name = self._result.name or "(unnamed)"
        corr = self._result.correlate_hook_observer(role=role)
        matched = corr.get("matched", [])
        obs_only = corr.get("observer_only", [])
        hook_only = corr.get("hook_only", [])

        if not matched:
            return (
                f"No matched hook/observer layers for {cfg_name} — "
                "skipping"
            )

        lines: list[str] = []
        lines.append(f"\n{'=' * 105}")
        lines.append(f"  Error Source Analysis — {cfg_name} [{role}]")
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
        if obs_only:
            lines.append(f"\n  Observer-only (no hook data):")
            for obs_key, loc_qsnr in obs_only:
                short = obs_key[:36]
                lines.append(
                    f"    {short:<36} local={loc_qsnr:.2f} dB"
                )

        # Hook-only layers
        if hook_only:
            lines.append(f"\n  Hook-only (no observer data):")
            for hk, acc_qsnr in hook_only:
                short = hk[:36]
                lines.append(
                    f"    {short:<36} accum={acc_qsnr:.2f} dB"
                )

        return "\n".join(lines)

    # ── Per-Role QSNR ──────────────────────────────────────────────────

    def per_role_qsnr(self, max_layers: int = 40) -> str:
        """Per-layer table with input / weight / output QSNR columns.

        Sorted by worst QSNR first.  Delegates to
        :meth:`ErrorProvenance.per_role_table`.

        Args:
            max_layers: Maximum number of layers to display.

        Returns:
            Formatted text table.
        """
        from src.analysis._error_provenance import ErrorProvenance

        prov = ErrorProvenance(self._result)
        return prov.per_role_table(max_layers=max_layers)
