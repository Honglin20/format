"""Distribution profiling, taxonomy, and error correlation tools."""
import numpy as np
from src.analysis.report import Report


class DistributionProfile:
    """Vertical summary of fp32 distribution fingerprints across all layers.

    Groups per-tensor statistics by role and computes percentile summaries.
    """

    def __init__(self, metrics_by_role: dict):
        self._data = metrics_by_role

    @classmethod
    def from_report(cls, report: Report):
        collected = {}

        for layer, roles in report._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        if "dynamic_range_bits" not in metrics:
                            continue
                        collected.setdefault(role, []).append(metrics)

        metrics_by_role = {}
        for role, metrics_list in collected.items():
            metrics_by_role[role] = cls._summarize_metrics(metrics_list)

        return cls(metrics_by_role)

    @staticmethod
    def _summarize_metrics(metrics_list: list) -> dict:
        if not metrics_list:
            return {"sample_count": 0}

        result = {"sample_count": len(metrics_list)}
        numeric_fields = [
            "mean", "std", "skewness", "kurtosis", "sparse_ratio",
            "dynamic_range_bits", "outlier_ratio", "norm_entropy",
            "bimodality_coefficient", "min", "max", "excess_kurtosis",
        ]

        for field in numeric_fields:
            values = [m[field] for m in metrics_list if field in m]
            if not values:
                continue
            arr = np.array(values)
            result[field] = {
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
            }

        return result

    def by_role(self, role: str) -> dict:
        return self._data.get(role, {"sample_count": 0})

    def all_roles(self) -> dict:
        return dict(self._data)

    def print_profile(self):
        print("=== Distribution Profile ===")
        for role, summary in self._data.items():
            n = summary.get("sample_count", 0)
            print(f"\n{role} ({n} samples):")
            if "dynamic_range_bits" in summary:
                dr = summary["dynamic_range_bits"]
                print(f"  Dynamic range (bits): "
                      f"min={dr['min']:.1f} p50={dr['p50']:.1f} "
                      f"p75={dr['p75']:.1f} max={dr['max']:.1f}")
            if "sparse_ratio" in summary:
                sp = summary["sparse_ratio"]
                print(f"  Sparse ratio: "
                      f"min={sp['min']:.2%} p50={sp['p50']:.2%} max={sp['max']:.2%}")
            if "std" in summary:
                sd = summary["std"]
                print(f"  Std dev: "
                      f"min={sd['min']:.4f} p50={sd['p50']:.4f} max={sd['max']:.4f}")


class DistributionTaxonomy:
    """Classify per-tensor distributions into predefined archetypes."""

    RULES = [
        ("zero-centered-gaussian", lambda m:
            abs(m.get("skewness", 0)) < 0.5
            and 2.5 < m.get("kurtosis", 0) < 4.0
            and m.get("sparse_ratio", 0) < 0.1),
        ("bimodal", lambda m:
            m.get("bimodality_coefficient", 0) > 0.555),
        ("zero-inflated", lambda m:
            m.get("sparse_ratio", 0) > 0.3),
        ("uniform-like", lambda m:
            m.get("norm_entropy", 0) > 0.85
            and abs(m.get("skewness", 0)) < 0.5),
        ("heavy-tailed", lambda m:
            m.get("kurtosis", 0) > 6.0),
        ("positive-skewed", lambda m:
            m.get("skewness", 0) > 0.5
            and m.get("bimodality_coefficient", 0) <= 0.555),
        ("negative-skewed", lambda m:
            m.get("skewness", 0) < -0.5),
        ("log-normal-like", lambda m:
            m.get("skewness", 0) > 1.0
            and m.get("kurtosis", 0) < 6.0
            and m.get("bimodality_coefficient", 0) <= 0.555
            and m.get("sparse_ratio", 0) < 0.3),
    ]

    def __init__(self, classifications: dict):
        self._clusters = classifications

    @classmethod
    def from_report(cls, report: Report):
        clusters = {}

        for layer, roles in report._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        if "skewness" not in metrics:
                            continue

                        cluster_name = cls._classify_one(metrics)
                        entry = clusters.setdefault(cluster_name, {
                            "count": 0,
                            "layers": [],
                            "metrics_sum": {},
                            "metric_count": 0,
                        })
                        entry["count"] += 1
                        entry["layers"].append((layer, role))
                        for k in ("skewness", "kurtosis", "sparse_ratio",
                                  "dynamic_range_bits", "norm_entropy"):
                            if k in metrics:
                                entry["metrics_sum"][k] = \
                                    entry["metrics_sum"].get(k, 0.0) + metrics[k]
                        entry["metric_count"] += 1

        result = {}
        total = sum(c["count"] for c in clusters.values())
        for name, data in clusters.items():
            avg_metrics = {}
            for k, s in data["metrics_sum"].items():
                avg_metrics[k] = s / data["metric_count"] if data["metric_count"] > 0 else 0

            result[name] = {
                "count": data["count"],
                "percentage": f"{100 * data['count'] / total:.0f}%" if total > 0 else "0%",
                "avg_metrics": avg_metrics,
                "representative_layers": data["layers"][:3],
            }

        return cls(result)

    @classmethod
    def _classify_one(cls, metrics: dict) -> str:
        for name, rule in cls.RULES:
            try:
                if rule(metrics):
                    return name
            except Exception:
                continue
        return "unclassified"

    def classify(self) -> dict:
        return dict(self._clusters)

    def classify_by_role(self, role: str) -> dict:
        return dict(self._clusters)

    def get_exemplars(self, cluster: str, n: int = 3) -> list:
        cluster_data = self._clusters.get(cluster)
        if not cluster_data:
            return []
        return [{"layer": l, "role": r}
                for l, r in cluster_data["representative_layers"][:n]]

    def print_taxonomy(self, ascii_plots: bool = False):
        print("=== Distribution Taxonomy ===")
        for name, data in sorted(self._clusters.items(),
                                  key=lambda x: x[1]["count"], reverse=True):
            print(f"\n{name} ({data['count']} layers, {data['percentage']}):")
            if data["avg_metrics"]:
                m = data["avg_metrics"]
                print(f"  avg skew={m.get('skewness', 0):.2f}  "
                      f"kurt={m.get('kurtosis', 0):.2f}  "
                      f"sparse={m.get('sparse_ratio', 0):.2%}")
            if data["representative_layers"]:
                print(f"  Examples: {data['representative_layers'][:3]}")
            if ascii_plots:
                example = (data["representative_layers"][0]
                           if data["representative_layers"] else "N/A")
                print(f"  [ASCII histogram: see HistogramObserver data for {example}]")


class ErrorByDistribution:
    """Correlate quantization error with distribution features."""

    def __init__(self, report: Report):
        self._report = report
        self._samples = []
        for layer, roles in report._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        if "qsnr_db" in metrics or "mse" in metrics:
                            self._samples.append({
                                "layer": layer,
                                "role": role,
                                "stage": stage,
                                **metrics,
                            })

    def rank_layers(self, by="qsnr_db", role=None, ascending=True, k=None):
        samples = self._samples
        if role:
            samples = [s for s in samples if s["role"] == role]
        if by not in ("qsnr_db", "mse"):
            raise ValueError(f"Unknown metric: {by}")

        reverse = not ascending if by == "qsnr_db" else ascending
        sorted_samples = sorted(samples, key=lambda s: s.get(by, 0), reverse=reverse)
        if k:
            sorted_samples = sorted_samples[:k]
        return [(s["layer"], s["role"], s.get(by, 0)) for s in sorted_samples]

    def group_by_range(self, role=None, bins=None):
        if bins is None:
            bins = [0, 4, 7, 999]

        samples = self._samples
        if role:
            samples = [s for s in samples if s["role"] == role]
        samples = [s for s in samples if "dynamic_range_bits" in s]

        groups = {}
        for s in samples:
            dr = s["dynamic_range_bits"]
            bucket = None
            for i in range(len(bins) - 1):
                if bins[i] <= dr < bins[i + 1]:
                    bucket = f"{bins[i]}-{bins[i+1]} bits"
                    break
            if bucket is None:
                bucket = f"{bins[-1]}+ bits"

            entry = groups.setdefault(bucket, {
                "count": 0, "_qsnr_sum": 0.0, "_mse_sum": 0.0,
            })
            entry["count"] += 1
            if "qsnr_db" in s:
                entry["_qsnr_sum"] += s["qsnr_db"]
            if "mse" in s:
                entry["_mse_sum"] += s["mse"]

        result = {}
        for bucket, entry in groups.items():
            avg_qsnr = entry["_qsnr_sum"] / entry["count"] if entry["count"] > 0 else 0
            if avg_qsnr >= 35:
                verdict = "excellent"
            elif avg_qsnr >= 25:
                verdict = "good"
            elif avg_qsnr >= 15:
                verdict = "acceptable"
            elif avg_qsnr >= 10:
                verdict = "poor — format insufficient"
            else:
                verdict = "critical"

            result[bucket] = {
                "avg_qsnr": avg_qsnr,
                "avg_mse": entry["_mse_sum"] / entry["count"] if entry["count"] > 0 else 0,
                "count": entry["count"],
                "verdict": verdict,
            }
        return result

    def print_correlation(self):
        print("=== Error by Distribution ===")
        groups = self.group_by_range()
        for bucket, stats in sorted(groups.items()):
            print(f"  {bucket}: avg QSNR={stats['avg_qsnr']:.1f} dB, "
                  f"MSE={stats['avg_mse']:.2e}, count={stats['count']}, "
                  f"verdict: {stats['verdict']}")


class LayerSensitivity:
    """Global sensitivity ranking for quantized layers."""

    def __init__(self, report: Report):
        self._report = report
        self._samples = []
        for layer, roles in report._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        if "mse" not in metrics and "qsnr_db" not in metrics:
                            continue
                        name_lower = layer.lower()
                        if "linear" in name_lower:
                            ltype = "Linear"
                        elif "conv" in name_lower:
                            ltype = "Conv"
                        elif "norm" in name_lower:
                            ltype = "Norm"
                        elif "act" in name_lower or "relu" in name_lower or "gelu" in name_lower:
                            ltype = "Activation"
                        else:
                            ltype = "Other"
                        self._samples.append({
                            "layer": layer,
                            "layer_type": ltype,
                            "role": role,
                            **metrics,
                        })

    def topk(self, k=10, role=None, metric="mse"):
        samples = self._samples
        if role:
            samples = [s for s in samples if s["role"] == role]
        reverse = True if metric == "mse" else False
        sorted_samples = sorted(samples, key=lambda s: s.get(metric, 0), reverse=reverse)
        return [(s["layer"], s["role"], s.get(metric, 0), s.get("layer_type", "?"))
                for s in sorted_samples[:k]]

    def by_layer_type(self) -> dict:
        result = {}
        for s in self._samples:
            lt = s["layer_type"]
            entry = result.setdefault(lt, {
                "count": 0, "_mse_sum": 0.0, "_qsnr_sum": 0.0, "layers": [],
            })
            entry["count"] += 1
            entry["_mse_sum"] += s.get("mse", 0)
            entry["_qsnr_sum"] += s.get("qsnr_db", 0)
            if s["layer"] not in entry["layers"]:
                entry["layers"].append(s["layer"])

        for lt, entry in result.items():
            entry["avg_mse"] = entry["_mse_sum"] / entry["count"] if entry["count"] > 0 else 0
            entry["avg_qsnr_db"] = entry["_qsnr_sum"] / entry["count"] if entry["count"] > 0 else 0
            del entry["_mse_sum"], entry["_qsnr_sum"]

        return result

    def above_threshold(self, metric="mse", threshold=0.01) -> list:
        return [(s["layer"], s["role"], s.get(metric, 0))
                for s in self._samples if s.get(metric, 0) > threshold]
