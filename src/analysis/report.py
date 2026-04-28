class Report:
    """Analysis report wrapper with Python API and print formatting."""

    def __init__(self, raw: dict):
        self._raw = raw

    def keys(self):
        return list(self._raw.keys())

    def layer(self, name: str) -> dict:
        return self._raw.get(name, {})

    def roles(self, layer: str) -> list:
        return list(self._raw.get(layer, {}).keys())

    def stages(self, layer: str, role: str) -> list:
        return list(self._raw.get(layer, {}).get(role, {}).keys())

    def to_dataframe(self):
        """Flatten to rows — one per slice."""
        rows = []
        for layer, roles in self._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        row = {
                            "layer": layer,
                            "role": role,
                            "stage": stage,
                            "slice": str(slice_key),
                            **metrics,
                        }
                        rows.append(row)
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def summary(self, by=("role",)):
        """Aggregate metrics grouped by given keys."""
        flat = []
        for layer, roles in self._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        layer_type = "Linear" if "linear" in layer.lower() else \
                                     "Conv" if "conv" in layer.lower() else "Other"
                        flat.append({
                            "layer_type": layer_type,
                            "role": role,
                            **metrics,
                        })

        if not flat:
            return {}

        result = {}
        for row in flat:
            key_parts = [str(row.get(b, "unknown")) for b in by]
            key = "/".join(key_parts)
            entry = result.setdefault(key, {"count": 0, "_mse_sum": 0.0, "_qsnr_sum": 0.0})
            entry["count"] += 1
            if "mse" in row:
                entry["_mse_sum"] += row["mse"]
            if "qsnr_db" in row:
                entry["_qsnr_sum"] += row["qsnr_db"]

        for key, entry in result.items():
            entry["avg_mse"] = entry["_mse_sum"] / entry["count"] if entry["count"] > 0 else 0
            entry["avg_qsnr_db"] = entry["_qsnr_sum"] / entry["count"] if entry["count"] > 0 else 0
            del entry["_mse_sum"], entry["_qsnr_sum"]

        return result

    def print_summary(self, top_k: int = 10):
        print("=== Quantization Analysis Summary ===")
        print(f"Total layers: {len(self._raw)}")

        role_summary = self.summary(by=("role",))
        if role_summary:
            print("\nRole Summary:")
            header = f"  {'Role':<12} {'Avg QSNR':>10} {'Avg MSE':>12} {'Count':>6}"
            print(header)
            print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*6}")
            for role, stats in role_summary.items():
                print(f"  {role:<12} {stats.get('avg_qsnr_db', 0):>10.1f} "
                      f"{stats.get('avg_mse', 0):>12.2e} {stats['count']:>6}")

        flat = self.to_dataframe()
        if isinstance(flat, list) and flat:
            sorted_rows = sorted(flat, key=lambda r: r.get("mse", 0), reverse=True)
            print(f"\nTop-{top_k} layers by MSE (worst -> best):")
            header = f"  {'Layer':<24} {'Role':<12} {'MSE':>12} {'QSNR':>8}"
            print(header)
            print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*8}")
            for row in sorted_rows[:top_k]:
                print(f"  {row['layer']:<24} {row['role']:<12} "
                      f"{row.get('mse', 0):>12.2e} {row.get('qsnr_db', 0):>8.1f}")

    def to_json(self, path: str):
        import json
        def _convert(obj):
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            return obj
        with open(path, "w") as f:
            json.dump(_convert(self._raw), f, indent=2)

    def to_csv(self, path: str):
        rows = self.to_dataframe()
        if not rows:
            return
        if hasattr(rows, "to_csv"):
            rows.to_csv(path, index=False)
        else:
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
