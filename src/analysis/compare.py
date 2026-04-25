"""Multi-format comparison framework (Phase 4.5).

compare_formats: run the same calibration data through multiple quantization
configurations and produce a ComparisonReport for side-by-side analysis.
"""
import torch
import torch.nn as nn

from src.analysis.context import AnalysisContext
from src.analysis.observers import QSNRObserver, DistributionObserver
from src.analysis.report import Report


class ComparisonReport:
    """Holds reports from multiple format configurations for comparison.

    Provides aggregation, ranking, and per-layer recommendation.
    """

    def __init__(self, reports: dict[str, Report]):
        self.reports = reports

    def to_dataframe(self):
        """Flatten all format reports into a single DataFrame.

        Adds a 'format' column to distinguish configurations.
        """
        frames = []
        for fmt_name, report in self.reports.items():
            df = report.to_dataframe()
            if isinstance(df, list):
                for row in df:
                    row["format"] = fmt_name
                    frames.append(row)
            else:
                df = df.copy()
                df["format"] = fmt_name
                frames.append(df)

        if not frames:
            return []

        if all(isinstance(f, dict) for f in frames):
            return frames

        try:
            import pandas as pd
            return pd.concat(frames, ignore_index=True)
        except ImportError:
            return frames

    def summary(self) -> dict:
        """Aggregate metrics per format.

        Returns:
            {format_name: {"avg_qsnr_db": ..., "avg_mse": ..., "total_layers": ...}}
        """
        result = {}
        for fmt_name, report in self.reports.items():
            role_summary = report.summary(by=("role",))
            total_qsnr = 0.0
            total_mse = 0.0
            total_roles = 0
            layer_set = set()

            for _, stats in role_summary.items():
                total_qsnr += stats.get("avg_qsnr_db", 0)
                total_mse += stats.get("avg_mse", 0)
                total_roles += 1
            # Count unique layers
            layer_set.update(report.keys())

            result[fmt_name] = {
                "avg_qsnr_db": total_qsnr / total_roles if total_roles > 0 else 0,
                "avg_mse": total_mse / total_roles if total_roles > 0 else 0,
                "total_layers": len(layer_set),
                "total_roles": total_roles,
            }
        return result

    def rank_formats(self, metric="qsnr_db") -> list:
        """Rank formats by a metric (higher QSNR = better, lower MSE = better).

        Returns:
            [(format_name, metric_value), ...] sorted best-to-worst.
        """
        summary = self.summary()
        reverse = True if metric == "qsnr_db" else False
        items = [(name, s.get(f"avg_{metric}", 0)) for name, s in summary.items()]
        return sorted(items, key=lambda x: x[1], reverse=reverse)

    def recommend(self) -> dict:
        """Per-layer best format recommendation based on QSNR.

        For each layer, pick the format with highest average QSNR across all roles.

        Returns:
            {layer_name: {"best_format": str, "qsnr_by_format": {fmt: qsnr, ...}}}
        """
        # Build per-layer per-format QSNR map
        layer_qsnr = {}
        for fmt_name, report in self.reports.items():
            for layer, roles in report._raw.items():
                entry = layer_qsnr.setdefault(layer, {})
                total_qsnr = 0.0
                role_count = 0
                for role, stages in roles.items():
                    for stage, slices in stages.items():
                        for slice_key, metrics in slices.items():
                            if "qsnr_db" in metrics:
                                total_qsnr += metrics["qsnr_db"]
                                role_count += 1
                entry[fmt_name] = total_qsnr / role_count if role_count > 0 else 0

        recommendations = {}
        for layer, fmt_qsnr in layer_qsnr.items():
            if not fmt_qsnr:
                continue
            best_fmt = max(fmt_qsnr, key=fmt_qsnr.get)
            recommendations[layer] = {
                "best_format": best_fmt,
                "qsnr_by_format": fmt_qsnr,
            }
        return recommendations

    def print_comparison(self):
        """Print formatted comparison summary."""
        print("=== Format Comparison ===")
        print(f"Configurations: {list(self.reports.keys())}")
        print()

        summary = self.summary()
        header = f"  {'Format':<20} {'Avg QSNR':>10} {'Avg MSE':>12} {'Layers':>8}"
        print(header)
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8}")
        for fmt_name, stats in summary.items():
            print(f"  {fmt_name:<20} {stats['avg_qsnr_db']:>10.1f} "
                  f"{stats['avg_mse']:>12.2e} {stats['total_layers']:>8}")

        ranked = self.rank_formats()
        print(f"\nFormat ranking (best → worst by QSNR):")
        for i, (name, qsnr) in enumerate(ranked, 1):
            print(f"  {i}. {name} — {qsnr:.1f} dB")

        recs = self.recommend()
        if recs:
            print(f"\nPer-layer recommendations:")
            fmt_width = max(len(r["best_format"]) for r in recs.values()) + 2
            header = f"  {'Layer':<20} {'Best Format':<{fmt_width}} {'QSNR (dB)':>10}"
            print(header)
            print(f"  {'-'*20} {'-'*fmt_width} {'-'*10}")
            for layer, rec in recs.items():
                best_qsnr = rec["qsnr_by_format"].get(rec["best_format"], 0)
                print(f"  {layer:<20} {rec['best_format']:<{fmt_width}} {best_qsnr:>10.1f}")


def compare_formats(
    build_model_fn,
    calibration_data,
    configs: dict,
    observers=None,
) -> ComparisonReport:
    """Run calibration data through multiple quantization configs and compare.

    Args:
        build_model_fn: callable(config_name, config) -> fresh quantized nn.Module.
        calibration_data: a single tensor or list of tensors (batches).
        configs: dict of {name: OpQuantConfig} mapping config names to configs.
        observers: list of Observer instances to use. Defaults to
            [QSNRObserver(), DistributionObserver()].

    Returns:
        ComparisonReport with per-format Report objects.
    """
    if not configs:
        raise ValueError("configs must contain at least one configuration")

    if observers is None:
        observers = [QSNRObserver(), DistributionObserver()]

    if isinstance(calibration_data, (list, tuple)):
        batches = calibration_data
    else:
        batches = [calibration_data]

    reports = {}
    for cfg_name, config in configs.items():
        model = build_model_fn(cfg_name, config)
        model.eval()

        with AnalysisContext(model, observers) as ctx:
            for batch in batches:
                with torch.no_grad():
                    model(batch)

        reports[cfg_name] = ctx.report()

    return ComparisonReport(reports)
