"""Report for a single SessionResult."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from src.session._result import SessionResult


class SessionReport:
    """Report for a single ``SessionResult``.

    Provides human-readable summaries and dataframe-style access to the
    per-layer metrics collected during a session.
    """

    def __init__(self, result: SessionResult):
        self.result = result

    # ── metrics_table ───────────────────────────────────────────────────

    def metrics_table(self) -> str:
        """Return a human-readable string summary of all metrics."""
        lines: List[str] = []
        lines.append(f"=== {self.result.name} ===")
        if self.result.quant_metrics:
            for k, v in self.result.quant_metrics.items():
                lines.append(f"  {k}: {v}")
        if self.result.delta:
            for k, v in self.result.delta.items():
                lines.append(f"  delta_{k}: {v}")
        if self.result.qsnr_per_layer:
            vals = list(self.result.qsnr_per_layer.values())
            avg_q = sum(vals) / len(vals)
            lines.append(f"  avg_qsnr: {avg_q:.2f}")
        if self.result.mse_per_layer:
            vals = list(self.result.mse_per_layer.values())
            avg_m = sum(vals) / len(vals)
            lines.append(f"  avg_mse: {avg_m:.6f}")
        return "\n".join(lines)

    # ── to_dataframe ────────────────────────────────────────────────────

    def to_dataframe(self) -> List[Dict[str, Any]]:
        """Return per-layer metrics as a list of dicts.

        Each dict has keys ``"layer"``, ``"qsnr"`` (if available),
        and ``"mse"`` (if available).

        Returns a list of dicts, each with ``"layer"``, ``"qsnr"``,
        and ``"mse"`` keys (when available).
        """
        rows: List[Dict[str, Any]] = []
        layers: set = set()
        layers.update(self.result.qsnr_per_layer.keys())
        layers.update(self.result.mse_per_layer.keys())
        for layer in sorted(layers):
            row: Dict[str, Any] = {"layer": layer}
            if layer in self.result.qsnr_per_layer:
                row["qsnr"] = self.result.qsnr_per_layer[layer]
            if layer in self.result.mse_per_layer:
                row["mse"] = self.result.mse_per_layer[layer]
            rows.append(row)
        return rows

    # ── save ────────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """Save the metrics summary as a text file in ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "metrics.txt")
        with open(path, "w") as f:
            f.write(self.metrics_table())
            f.write("\n")
