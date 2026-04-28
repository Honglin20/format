"""Model-level task performance evaluation.

evaluate_performance: compare quantized models against an FP32 baseline
using a user-provided eval_fn. Measures the real-world impact of quantization
(accuracy drop, loss increase, etc.) rather than per-layer signal fidelity.
"""
from typing import Callable

import torch.nn as nn


class PerformanceReport:
    """Holds task-level metrics for an FP32 baseline and quantized variants.

    Provides delta computation, formatted summary, and DataFrame export.
    """

    def __init__(self, baseline_metrics: dict, quantized_metrics: dict[str, dict]):
        self.baseline = baseline_metrics
        self.quantized = quantized_metrics

    # ---- Data access -------------------------------------------------------

    def summary(self) -> dict:
        """Per-model metrics with delta from FP32 baseline.

        Returns:
            {model_name: {"metric_name": value, ..., "delta_<metric>": diff, ...}}
            "fp32_baseline" entry has no delta fields.
        """
        result = {"fp32_baseline": dict(self.baseline)}

        for name, metrics in self.quantized.items():
            entry = dict(metrics)
            for k, v in metrics.items():
                if k in self.baseline:
                    entry[f"delta_{k}"] = v - self.baseline[k]
            result[name] = entry

        return result

    def to_dataframe(self):
        """Flatten all model metrics into a DataFrame (pandas) or list of dicts."""
        rows = []
        baseline_row = {"model": "fp32_baseline"}
        baseline_row.update(self.baseline)
        rows.append(baseline_row)

        for name, metrics in self.quantized.items():
            row = {"model": name}
            row.update(metrics)
            for k, v in metrics.items():
                if k in self.baseline:
                    row[f"delta_{k}"] = v - self.baseline[k]
            rows.append(row)

        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    # ---- Printing ----------------------------------------------------------

    def print_summary(self, top_k: int = 0):
        """Print formatted performance comparison.

        Args:
            top_k: ignored; included for API compatibility. All models shown.
        """
        print("=== Model Performance ===")
        print()

        # Baseline
        baseline_items = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.baseline.items()
        )
        print(f"  FP32 baseline: {baseline_items}")
        print()

        if not self.quantized:
            print("  (no quantized models)")
            return

        # Determine column widths
        metric_names = list(self.baseline.keys())
        name_width = max(max(len(n) for n in self.quantized), 16)

        # Header
        header = f"  {'Model':<{name_width}}"
        for m in metric_names:
            header += f" {m:>12}"
        header += " |"
        for m in metric_names:
            header += f" {'delta_' + m:>14}"
        print(header)

        sep = f"  {'-' * name_width}"
        for _ in metric_names:
            sep += f" {'-' * 12}"
        sep += "-+"
        for _ in metric_names:
            sep += f" {'-' * 14}"
        print(sep)

        # Quantized rows
        for name, metrics in self.quantized.items():
            row = f"  {name:<{name_width}}"
            for m in metric_names:
                v = metrics.get(m, 0)
                row += f" {v:>12.4f}" if isinstance(v, float) else f" {str(v):>12}"
            row += " |"
            for m in metric_names:
                if m in self.baseline and m in metrics:
                    delta = metrics[m] - self.baseline[m]
                    row += f" {delta:>+14.4f}" if isinstance(delta, float) else f" {str(delta):>14}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

        print()

        # Highlight the best
        if len(self.quantized) > 1:
            print("Best per metric:")
            for m in metric_names:
                hi_better = m not in ("loss", "perplexity")
                best = max if hi_better else min
                candidates = {n: d.get(m) for n, d in self.quantized.items() if m in d}
                if candidates:
                    best_name = best(candidates, key=candidates.get)
                    print(f"  {m}: {best_name} ({candidates[best_name]:.4f})")
            print()

        # Top degradation warnings
        degrades = []
        for name, metrics in self.quantized.items():
            for m in metric_names:
                if m in self.baseline and m in metrics:
                    delta = metrics[m] - self.baseline[m]
                    hi_better = m not in ("loss", "perplexity")
                    degraded = delta < 0 if hi_better else delta > 0
                    if degraded:
                        degrades.append((name, m, delta))

        if degrades:
            print("Degradation warnings:")
            for name, metric, delta in degrades:
                direction = "↓" if metric not in ("loss", "perplexity") else "↑"
                print(f"  {name}: {metric} {delta:+.4f} {direction}")


# ---------------------------------------------------------------------------
# evaluate_performance entry point
# ---------------------------------------------------------------------------

def evaluate_performance(
    fp32_model: nn.Module,
    quantized_models: dict[str, nn.Module],
    eval_dataloader,
    eval_fn: Callable[[nn.Module, object], dict],
    device: str = "cpu",
) -> PerformanceReport:
    """Evaluate quantized models against an FP32 baseline on task performance.

    Args:
        fp32_model: Unquantized baseline model.
        quantized_models: Dict of {name: quantized_model} to evaluate.
        eval_dataloader: DataLoader yielding batches for evaluation.
        eval_fn: callable(model, dataloader) -> dict of metric_name: float.
            The caller defines what metrics to compute (accuracy, loss, etc.).
        device: Device to run evaluation on ("cpu" or "cuda").

    Returns:
        PerformanceReport with baseline + per-model metrics and deltas.

    Example:
        >>> def cls_eval(model, dl):
        ...     correct, total, loss = 0, 0, 0.0
        ...     for x, y in dl:
        ...         out = model(x)
        ...         correct += (out.argmax(1) == y).sum().item()
        ...         total += y.size(0)
        ...     return {"accuracy": correct / total}
        >>> report = evaluate_performance(fp32, {"fp8": q_model}, loader, cls_eval)
    """
    if not quantized_models:
        raise ValueError("quantized_models must contain at least one model")

    fp32_model.to(device)
    fp32_model.eval()
    baseline_metrics = eval_fn(fp32_model, eval_dataloader)

    quantized_metrics = {}
    for name, model in quantized_models.items():
        model.to(device)
        model.eval()
        quantized_metrics[name] = eval_fn(model, eval_dataloader)

    return PerformanceReport(baseline_metrics, quantized_metrics)
