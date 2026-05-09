"""SessionResult dataclass and its accessor methods.

SessionResult is the output of running one Session (one QuantConfig).
It holds accuracy deltas, per-layer QSNR/MSE metrics, raw observer data,
and cost estimates, plus user-facing accessor methods for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.session._config import QuantConfig


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
    observers_data: Dict[str, Any] = field(default_factory=dict)
    cost: Any = None
    cost_fp32: Any = None
    sq_transforms: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Accessor methods
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line human-readable summary of the quantization result.

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

        if self.qsnr_per_layer:
            finite = [v for v in self.qsnr_per_layer.values() if v == v and v != float('inf') and v != float('-inf')]
            if finite:
                avg_qsnr = sum(finite) / len(finite)
                parts.append(f"avg QSNR={avg_qsnr:.1f} dB")
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
            return "(no accuracy metrics — run with eval_fn)"

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

    def top_k_qsnr(self, k: int = 10, reverse: bool = False) -> List[Tuple[str, float]]:
        """Top-k layers by QSNR.

        Args:
            k: Number of layers to return (default 10).
            reverse: If False (default), returns the k layers with the **lowest**
                QSNR (worst quality), sorted ascending. If True, returns the k
                layers with the **highest** QSNR (best quality), sorted descending.

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
        sorted_layers = sorted(self.qsnr_per_layer.items(), key=lambda x: x[1], reverse=reverse)
        return sorted_layers[:k]

    def layer_report(self) -> "Any":
        """Per-layer DataFrame with QSNR and MSE metrics.

        Returns:
            ``pandas.DataFrame`` with columns: layer, qsnr_db, mse.
            Returns ``None`` if pandas is not available.

        Example::

            >>> df = result.layer_report()
            >>> print(df.sort_values("qsnr_db").head(5))
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        rows = []
        all_layers = set(self.qsnr_per_layer.keys()) | set(self.mse_per_layer.keys())
        for layer in sorted(all_layers):
            rows.append({
                "layer": layer,
                "qsnr_db": self.qsnr_per_layer.get(layer),
                "mse": self.mse_per_layer.get(layer),
            })
        return pd.DataFrame(rows)
