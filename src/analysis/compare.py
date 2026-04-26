"""Multi-format comparison framework (Phase 4.5).

compare_formats: run the same calibration data through multiple quantization
configurations and produce a ComparisonReport for side-by-side analysis.
"""
import torch
import torch.nn as nn

from src.analysis.context import AnalysisContext
from src.analysis.observers import QSNRObserver, DistributionObserver
from src.analysis.report import Report


# ---------------------------------------------------------------------------
# Metric metadata — higher_is_better for ranking / recommendation
# ---------------------------------------------------------------------------

_HIGHER_IS_BETTER = {
    "qsnr_db": True,
    "mse": False,
    "dynamic_range_bits": False,
    "sparse_ratio": False,
    "skewness": False,
    "kurtosis": False,
    "norm_entropy": False,
    "outlier_ratio": False,
    "mean": False,
    "std": False,
    "excess_kurtosis": False,
    "bimodality_coefficient": False,
}


def higher_is_better(metric: str) -> bool:
    """True if higher values of this metric are better. Unknown metrics default to True."""
    return _HIGHER_IS_BETTER.get(metric, True)


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------

class ComparisonReport:
    """Holds reports from multiple format configurations for comparison.

    Provides aggregation, ranking, per-layer recommendation, and
    formatted printing.
    """

    def __init__(self, reports: dict[str, Report]):
        self.reports = reports

    # ---- Data access -------------------------------------------------------

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
            layer_set.update(report.keys())

            result[fmt_name] = {
                "avg_qsnr_db": total_qsnr / total_roles if total_roles > 0 else 0,
                "avg_mse": total_mse / total_roles if total_roles > 0 else 0,
                "total_layers": len(layer_set),
                "total_roles": total_roles,
            }
        return result

    # ---- Ranking -----------------------------------------------------------

    def rank_formats(self, metric="qsnr_db", role=None) -> list:
        """Rank formats by a metric, respecting higher_is_better.

        Args:
            metric: Metric name (e.g. "qsnr_db", "mse").
            role: Optional role filter ("input", "weight", "output", "bias").
                  None = all roles combined.

        Returns:
            [(format_name, metric_value), ...] sorted best-to-worst.
        """
        hi_better = higher_is_better(metric)

        if role:
            return self._rank_formats_by_role(metric, role, hi_better)

        summary = self.summary()
        avg_key = f"avg_{metric}"
        items = [(name, s.get(avg_key, 0)) for name, s in summary.items()]
        return sorted(items, key=lambda x: x[1], reverse=hi_better)

    def _rank_formats_by_role(self, metric: str, role: str, hi_better: bool) -> list:
        """Rank formats for a specific tensor role."""
        role_avg = {}
        role_count = {}
        for fmt_name, report in self.reports.items():
            for _, roles in report._raw.items():
                if role not in roles:
                    continue
                stages = roles[role]
                for _, slices in stages.items():
                    for _, metrics in slices.items():
                        if metric in metrics:
                            role_avg.setdefault(fmt_name, 0.0)
                            role_avg[fmt_name] += metrics[metric]
                            role_count.setdefault(fmt_name, 0)
                            role_count[fmt_name] += 1
        items = []
        for fmt_name in role_avg:
            avg = role_avg[fmt_name] / role_count[fmt_name] if role_count.get(fmt_name, 0) > 0 else 0
            items.append((fmt_name, avg))
        return sorted(items, key=lambda x: x[1], reverse=hi_better)

    # ---- Recommendation ----------------------------------------------------

    def recommend(self, metric="qsnr_db") -> dict:
        """Per-layer best format recommendation.

        For each layer, picks the format with the best metric value across
        all roles, respecting higher_is_better.

        Args:
            metric: Metric to use for recommendation (e.g. "qsnr_db", "mse").

        Returns:
            {layer_name: {"best_format": str, "scores_by_format": {fmt: score, ...}}}
        """
        hi_better = higher_is_better(metric)
        layer_scores = {}

        for fmt_name, report in self.reports.items():
            for layer, roles in report._raw.items():
                entry = layer_scores.setdefault(layer, {})
                total = 0.0
                count = 0
                for _, stages in roles.items():
                    for _, slices in stages.items():
                        for _, mdict in slices.items():
                            if metric in mdict:
                                total += mdict[metric]
                                count += 1
                entry[fmt_name] = total / count if count > 0 else 0

        recommendations = {}
        for layer, fmt_scores in layer_scores.items():
            if not fmt_scores:
                continue
            best = max if hi_better else min
            best_fmt = best(fmt_scores, key=fmt_scores.get)
            recommendations[layer] = {
                "best_format": best_fmt,
                "scores_by_format": fmt_scores,
            }
        return recommendations

    # ---- Printing ----------------------------------------------------------

    def print_comparison(self):
        """Print formatted comparison summary."""
        print("=== Format Comparison ===")
        print(f"Configurations: {list(self.reports.keys())}")
        print()

        # Overall summary
        summary = self.summary()
        header = f"  {'Format':<20} {'Avg QSNR':>10} {'Avg MSE':>12} {'Layers':>8}"
        print(header)
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8}")
        for fmt_name, stats in summary.items():
            print(f"  {fmt_name:<20} {stats['avg_qsnr_db']:>10.1f} "
                  f"{stats['avg_mse']:>12.2e} {stats['total_layers']:>8}")

        # Overall ranking
        ranked = self.rank_formats()
        print(f"\nOverall ranking (best → worst by QSNR):")
        for i, (name, qsnr) in enumerate(ranked, 1):
            print(f"  {i}. {name} — {qsnr:.1f} dB")

        # Per-role ranking
        print(f"\nPer-role QSNR (dB):")
        roles = ["input", "weight", "output", "bias"]
        active_roles = []
        for role in roles:
            r = self.rank_formats(role=role)
            if any(v > 0 for _, v in r):
                active_roles.append((role, r))

        if active_roles:
            fmt_names = [fmt for fmt in self.reports]
            role_header = f"  {'Role':<12}"
            for fmt in fmt_names:
                role_header += f" {fmt:>12}"
            print(role_header)
            print(f"  {'-'*12}{' ' + '-'*12 * len(fmt_names)}")
            for role, ranking in active_roles:
                fmt_map = dict(ranking)
                row = f"  {role:<12}"
                for fmt in fmt_names:
                    row += f" {fmt_map.get(fmt, 0):>12.1f}"
                print(row)

        # Per-layer recommendations
        recs = self.recommend()
        if recs:
            print(f"\nPer-layer recommendations:")
            fmt_width = max(len(r["best_format"]) for r in recs.values()) + 2
            rec_header = f"  {'Layer':<20} {'Best Format':<{fmt_width}}"
            print(rec_header)
            print(f"  {'-'*20} {'-'*fmt_width}")
            for layer, rec in recs.items():
                print(f"  {layer:<20} {rec['best_format']:<{fmt_width}}")


# ---------------------------------------------------------------------------
# compare_formats entry point
# ---------------------------------------------------------------------------

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
