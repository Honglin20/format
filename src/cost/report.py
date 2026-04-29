"""CostReport: aggregate and format per-layer cost data."""
from __future__ import annotations


class CostReport:
    def __init__(self, layers: list, model_name: str = ""):
        self.layers = list(layers)
        self.model_name = model_name

    @property
    def total_latency_us(self) -> float:
        return sum(l.latency_us for l in self.layers)

    @property
    def total_memory_bytes(self) -> int:
        weight_sum = sum(l.memory_weight_bytes for l in self.layers)
        max_act = max((l.memory_activation_bytes for l in self.layers), default=0)
        return weight_sum + max_act

    def summary(self) -> dict:
        return {
            "model": self.model_name or "model",
            "num_layers": len(self.layers),
            "total_latency_us": self.total_latency_us,
            "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
            "total_flops_math": sum(l.flops_math for l in self.layers),
            "total_flops_quantize": sum(l.flops_quantize for l in self.layers),
            "total_flops_transform": sum(l.flops_transform for l in self.layers),
        }

    def to_dataframe(self):
        rows = []
        for l in self.layers:
            rows.append({
                "op_name": l.op_name, "op_type": l.op_type,
                "flops_math": l.flops_math,
                "flops_quantize": l.flops_quantize,
                "flops_transform": l.flops_transform,
                "latency_us": round(l.latency_us, 2),
                "mem_weight_kb": round(l.memory_weight_bytes / 1024, 2),
                "mem_act_kb": round(l.memory_activation_bytes / 1024, 2),
            })
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def print_summary(self):
        s = self.summary()
        print(f"=== Cost Report: {s['model']} ===")
        print(f"  Layers: {s['num_layers']}")
        print(f"  Total latency: {s['total_latency_us']:.1f} us")
        print(f"  Total memory:  {s['total_memory_mb']:.2f} MB")
        print(f"  Math FLOPs:    {s['total_flops_math']:,}")
        print(f"  Quant FLOPs:   {s['total_flops_quantize']:,}")
        print(f"  Transform FLOPs: {s['total_flops_transform']:,}")

    def print_per_layer(self):
        print(f"{'Layer':<24} {'Type':<16} {'Lat(us)':>10} {'Mem(kB)':>10}")
        print("-" * 62)
        for l in self.layers:
            mem_total = l.memory_weight_bytes + l.memory_activation_bytes
            print(f"  {l.op_name:<22} {l.op_type:<16} {l.latency_us:>10.2f} {mem_total/1024:>10.1f}")

    def print_comparison(self, baseline: "CostReport"):
        print(f"=== Cost Comparison: {self.model_name} vs {baseline.model_name} ===")
        lat_ratio = self.total_latency_us / max(baseline.total_latency_us, 1e-9)
        mem_ratio = self.total_memory_bytes / max(baseline.total_memory_bytes, 1)
        print(f"  Latency: {self.total_latency_us:.1f} / {baseline.total_latency_us:.1f} us "
              f"({lat_ratio:.2f}x)")
        print(f"  Memory:  {self.total_memory_bytes/1e6:.2f} / {baseline.total_memory_bytes/1e6:.2f} MB "
              f"({mem_ratio:.2f}x)")
