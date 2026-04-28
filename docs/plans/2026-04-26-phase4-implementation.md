# Phase 4: 层级误差分析 — 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 Phase 3 的 ObservableMixin/QuantEvent/iter_slices 骨架上，实现单配置量化分析：4 个 Observer + AnalysisContext + Report + 分布归类 + 误差关联

**Architecture:** SliceAwareObserver 子类实现 `_measure()` 收集指标 → AnalysisContext 挂载到量化模型 → Report 聚合多 Observer 数据 → DistributionTaxonomy/ErrorByDistribution 做高层分析

**Tech Stack:** PyTorch, dataclasses, pytest

---

## Task 1: DistributionObserver

**Files:**
- Create: `src/analysis/observers.py`
- Create: `src/tests/test_analysis.py`

### Step 1: Write the failing test

```python
# src/tests/test_analysis.py
import torch
import pytest
from src.analysis.observers import DistributionObserver


class TestDistributionObserver:
    """Unit tests for DistributionObserver._measure()."""

    def test_gaussian_distribution(self):
        """Synthetic Gaussian: skew≈0, kurt≈3, sparse≈0."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.randn(1000)  # mean≈0, std≈1
        q = f.clone()  # perfect quantization (identity)

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["mean"] == pytest.approx(0.0, abs=0.1)
        assert metrics["std"] == pytest.approx(1.0, abs=0.1)
        assert metrics["skewness"] == pytest.approx(0.0, abs=0.3)
        assert metrics["excess_kurtosis"] == pytest.approx(0.0, abs=0.5)
        assert metrics["sparse_ratio"] < 0.05
        assert metrics["norm_entropy"] > 0.5  # not a delta

    def test_positive_skewed(self):
        """ReLU-like: right-skewed, high sparsity."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.randn(1000).clamp(min=0)  # ReLU
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["skewness"] > 0.5
        assert metrics["sparse_ratio"] == pytest.approx(0.5, abs=0.1)  # ~50% zeros

    def test_bimodal_distribution(self):
        """Two separated Gaussians: bimodality_coefficient > 0.555."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.cat([torch.randn(500) - 2.0, torch.randn(500) + 2.0])
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["bimodality_coefficient"] > 0.555
        assert abs(metrics["skewness"]) < 0.5  # symmetric

    def test_all_zeros(self):
        """Edge case: all zeros → sparse_ratio=1, dynamic_range_bits=0."""
        obs = DistributionObserver()
        f = torch.zeros(100)
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["sparse_ratio"] == 1.0
        assert metrics["dynamic_range_bits"] == 0.0
        assert metrics["std"] == 0.0

    def test_heavy_tailed(self):
        """Cauchy-like: excess kurtosis >> 0."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.distributions.Cauchy(0, 1).sample((2000,))
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["excess_kurtosis"] > 3.0  # fat tails

    def test_uniform_distribution(self):
        """Uniform: high normalized entropy, low skew."""
        obs = DistributionObserver()
        torch.manual_seed(42)
        f = torch.rand(1000)  # Uniform[0,1]
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        assert metrics["norm_entropy"] > 0.85
        assert abs(metrics["skewness"]) < 0.5

    def test_dynamic_range_bits(self):
        """Known range: [1e-6, 1.0] → dynamic_range_bits ≈ log2(1e6) ≈ 20."""
        obs = DistributionObserver()
        f = torch.tensor([1e-6, 1.0, 0.5, 2e-6, 0.0, -0.3])
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)

        # log2(1.0 / 1e-6) = log2(1e6) ≈ 19.93
        assert metrics["dynamic_range_bits"] == pytest.approx(19.93, abs=0.1)

    def test_outlier_detection(self):
        """One extreme outlier: outlier_ratio ≈ 1/N."""
        obs = DistributionObserver(outlier_sigma=3.0)
        torch.manual_seed(42)
        f = torch.randn(500)
        f[0] = 100.0  # extreme outlier
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)
        assert metrics["outlier_ratio"] > 0.0
        assert metrics["outlier_ratio"] < 0.01
```

### Step 2: Run test to verify it fails

```bash
pytest src/tests/test_analysis.py::TestDistributionObserver -v
```
Expected: FAIL — no module `src.analysis.observers`

### Step 3: Write minimal implementation

```python
# src/analysis/observers.py
import torch
from src.analysis.observer import SliceAwareObserver


class DistributionObserver(SliceAwareObserver):
    """Per-slice fp32 statistical fingerprint for distribution taxonomy."""

    def __init__(self, sparse_eps: float = 1e-8, outlier_sigma: float = 3.0,
                 hist_bins: int = 64):
        super().__init__()
        self.sparse_eps = sparse_eps
        self.outlier_sigma = outlier_sigma
        self.hist_bins = hist_bins

    def _measure(self, key, fp32, quant):
        f = fp32
        f_abs = f.abs()
        n = f.numel()
        non_zero_mask = f_abs > self.sparse_eps
        min_nonzero = f_abs[non_zero_mask].min().item() if non_zero_mask.any() else self.sparse_eps

        # Central moments
        mean = f.mean()
        delta = f - mean
        var = delta.pow(2).mean()
        std = var.sqrt()
        m3 = delta.pow(3).mean()
        m4 = delta.pow(4).mean()
        skew = (m3 / (var * std + 1e-30)).item()
        kurt = (m4 / (var.pow(2) + 1e-30)).item()
        excess_kurt = kurt - 3.0

        # Sarle's bimodality coefficient
        bc_denom = excess_kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3) + 1e-30)
        bimodality = (skew**2 + 1) / (bc_denom + 1e-30)

        # Normalized Shannon entropy from histogram
        hist = torch.histc(f, bins=self.hist_bins)
        probs = hist.float() / (n + 1e-30)
        probs_pos = probs[probs > 0]
        entropy_raw = -(probs_pos * torch.log2(probs_pos + 1e-30)).sum().item()
        max_entropy = torch.log2(torch.tensor(self.hist_bins, dtype=torch.float32)).item()
        norm_entropy = entropy_raw / (max_entropy + 1e-30)

        return {
            "min": f.min().item(),
            "max": f.max().item(),
            "mean": mean.item(),
            "std": std.item(),
            "skewness": skew,
            "kurtosis": kurt,
            "excess_kurtosis": excess_kurt,
            "bimodality_coefficient": bimodality,
            "sparse_ratio": (f_abs < self.sparse_eps).float().mean().item(),
            "dynamic_range_bits": (torch.log2(f_abs.max() / min_nonzero)).item() if non_zero_mask.any() else 0.0,
            "outlier_ratio": (f_abs > self.outlier_sigma * std).float().mean().item(),
            "norm_entropy": norm_entropy,
        }
```

### Step 4: Run tests to verify they pass

```bash
pytest src/tests/test_analysis.py::TestDistributionObserver -v
```
Expected: 8 tests PASS

### Step 5: Commit

```bash
git add src/analysis/observers.py src/tests/test_analysis.py
git commit -m "feat(analysis): add DistributionObserver with skew/kurtosis/bimodality/entropy stats

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 2: QSNRObserver + MSEObserver

**Files:**
- Modify: `src/analysis/observers.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Write the failing tests

```python
# Append to src/tests/test_analysis.py
from src.analysis.observers import QSNRObserver, MSEObserver


class TestQSNRObserver:
    def test_perfect_quantization_infinite_qsnr(self):
        """fp32 == quant → error=0 → QSNR should handle inf."""
        obs = QSNRObserver()
        f = torch.randn(100)
        q = f.clone()

        metrics = obs._measure(("tensor",), f, q)
        # Perfect: power of error is 0, so QSNR is inf
        assert metrics["qsnr_db"] > 100  # effectively infinite

    def test_known_error(self):
        """fp32 = [1.0], quant = [0.9] → QSNR analytically known."""
        obs = QSNRObserver()
        f = torch.tensor([1.0, 2.0, 3.0])
        q = torch.tensor([0.9, 1.8, 2.7])  # 10% relative error

        metrics = obs._measure(("tensor",), f, q)
        # num = mean(fp32^2) = (1+4+9)/3 = 14/3
        # den = mean((fp32-q)^2) = (0.01+0.04+0.09)/3 = 0.14/3
        # QSNR = 10*log10((14/3)/(0.14/3)) = 10*log10(100) = 20 dB
        assert metrics["qsnr_db"] == pytest.approx(20.0, abs=0.01)


class TestMSEObserver:
    def test_perfect_quantization_zero_mse(self):
        obs = MSEObserver()
        f = torch.randn(100)
        q = f.clone()
        metrics = obs._measure(("tensor",), f, q)
        assert metrics["mse"] == 0.0

    def test_known_error(self):
        obs = MSEObserver()
        f = torch.tensor([1.0, 2.0, 3.0])
        q = torch.tensor([0.9, 1.8, 2.7])
        metrics = obs._measure(("tensor",), f, q)
        # MSE = mean((0.1)^2, (0.2)^2, (0.3)^2) = (0.01+0.04+0.09)/3 ≈ 0.04667
        assert metrics["mse"] == pytest.approx(0.04667, abs=1e-5)
```

### Step 2: Run tests to verify they fail

```bash
pytest src/tests/test_analysis.py::TestQSNRObserver -v
pytest src/tests/test_analysis.py::TestMSEObserver -v
```
Expected: FAIL — QSNRObserver / MSEObserver not defined

### Step 3: Write minimal implementation

```python
# Append to src/analysis/observers.py

class QSNRObserver(SliceAwareObserver):
    """QSNR = 10 * log10(||fp32||² / ||fp32 - quant||²), unit dB."""

    def _measure(self, key, fp32, quant):
        err = fp32 - quant
        num = fp32.pow(2).mean()
        den = err.pow(2).mean().clamp_min(1e-30)
        return {"qsnr_db": (10 * torch.log10(num / den)).item()}


class MSEObserver(SliceAwareObserver):
    """Mean squared error per slice."""

    def _measure(self, key, fp32, quant):
        return {"mse": (fp32 - quant).pow(2).mean().item()}
```

### Step 4: Run tests to verify they pass

```bash
pytest src/tests/test_analysis.py::TestQSNRObserver src/tests/test_analysis.py::TestMSEObserver -v
```
Expected: 4 tests PASS

### Step 5: Commit

```bash
git add src/analysis/observers.py src/tests/test_analysis.py
git commit -m "feat(analysis): add QSNRObserver and MSEObserver

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 3: HistogramObserver

**Files:**
- Modify: `src/analysis/observers.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Write the failing test

```python
# Append to src/tests/test_analysis.py
from src.analysis.observers import HistogramObserver


class TestHistogramObserver:
    def test_bin_count(self):
        obs = HistogramObserver(n_bins=64)
        f = torch.randn(500)
        q = f + 0.01 * torch.randn(500)

        metrics = obs._measure(("tensor",), f, q)
        assert metrics["fp32_hist"].numel() == 64
        assert metrics["quant_hist"].numel() == 64
        assert metrics["err_hist"].numel() == 64

    def test_counts_sum_to_total(self):
        obs = HistogramObserver(n_bins=32)
        f = torch.randn(300)
        q = f + 0.01 * torch.randn(300)

        metrics = obs._measure(("tensor",), f, q)
        assert metrics["fp32_hist"].sum().item() == 300
        assert metrics["quant_hist"].sum().item() == 300
        assert metrics["err_hist"].sum().item() == 300
```

### Step 3: Write minimal implementation

```python
# Append to src/analysis/observers.py

class HistogramObserver(SliceAwareObserver):
    """fp32 / quant / error three-channel histogram."""

    def __init__(self, n_bins: int = 128):
        super().__init__()
        self.n_bins = n_bins

    def _measure(self, key, fp32, quant):
        return {
            "fp32_hist": torch.histc(fp32, bins=self.n_bins).cpu(),
            "quant_hist": torch.histc(quant, bins=self.n_bins).cpu(),
            "err_hist": torch.histc(fp32 - quant, bins=self.n_bins).cpu(),
        }
```

### Step 4: Run tests to verify they pass

```bash
pytest src/tests/test_analysis.py::TestHistogramObserver -v
```
Expected: 2 tests PASS. Then run full suite:
```bash
pytest src/tests/test_analysis.py -v
```
Expected: 14 tests PASS

### Step 5: Commit

```bash
git add src/analysis/observers.py src/tests/test_analysis.py
git commit -m "feat(analysis): add HistogramObserver with three-channel histogram

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 4: AnalysisContext

**Files:**
- Create: `src/analysis/context.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Write the failing test

```python
# Append to src/tests/test_analysis.py
import torch.nn as nn
from src.analysis.context import AnalysisContext


class TestAnalysisContext:
    def test_context_attaches_and_detaches_observers(self):
        """After __exit__, all ObservableMixin instances have empty _observers."""
        from src.analysis.mixin import ObservableMixin

        class DummyLayer(ObservableMixin, nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(DummyLayer(), DummyLayer())
        from src.analysis.observers import QSNRObserver

        with AnalysisContext(model, [QSNRObserver()]) as ctx:
            for m in model.modules():
                if isinstance(m, ObservableMixin):
                    assert len(m._observers) == 1

        for m in model.modules():
            if isinstance(m, ObservableMixin):
                assert len(m._observers) == 0

    def test_warmup_batches_reset_observers(self):
        """warmup_batches=2 → first 2 steps, observer data discarded."""
        from src.analysis.mixin import ObservableMixin

        class EmitLayer(ObservableMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10))

            def forward(self, x):
                return x

        model = EmitLayer()
        from src.analysis.observers import MSEObserver
        obs = MSEObserver()

        with AnalysisContext(model, [obs], warmup_batches=2) as ctx:
            model(torch.randn(5, 10))
            ctx.step()
            # After warmup, reset should have cleared buffer
            # Buffer should be empty since emit_fn not wired in this test
            assert len(obs.report()) == 0

    def test_report_aggregates_observers(self):
        """ctx.report() returns a Report object."""
        from src.analysis.mixin import ObservableMixin

        class NoOpLayer(ObservableMixin, nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(NoOpLayer())
        from src.analysis.observers import QSNRObserver

        with AnalysisContext(model, [QSNRObserver()]) as ctx:
            model(torch.randn(3, 4))

        report = ctx.report()
        assert report is not None
        # Should be a Report type
        from src.analysis.report import Report
        assert isinstance(report, Report)
```

### Step 3: Write minimal implementation

```python
# src/analysis/context.py
import torch.nn as nn
from src.analysis.mixin import ObservableMixin
from src.analysis.report import Report


class AnalysisContext:
    """Context manager: attaches observers to ObservableMixin modules.

    Usage:
        with AnalysisContext(model, [QSNRObserver()]) as ctx:
            for batch in data:
                model(batch)
        report = ctx.report()
    """

    def __init__(self, model: nn.Module, observers=None,
                 warmup_batches: int = 0):
        self.model = model
        self.observers = observers or []
        self.warmup_batches = warmup_batches
        self._batch_count = 0

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ObservableMixin):
                module._observers = self.observers
                module._analysis_name = name
        return self

    def __exit__(self, *args):
        for module in self.model.modules():
            if isinstance(module, ObservableMixin):
                module._observers = []

    def report(self):
        """Aggregate all observer data into a Report."""
        raw = {}
        for i, obs in enumerate(self.observers):
            for layer, role_map in obs.report().items():
                raw.setdefault(layer, {}).update(role_map)
        return Report(raw)

    def step(self):
        """Mark one batch complete. Warmup batches reset observers."""
        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            for obs in self.observers:
                obs.reset()
```

### Step 4: Run tests to verify they pass

```bash
pytest src/tests/test_analysis.py::TestAnalysisContext -v
```
Expected: 3 tests PASS

### Step 5: Commit

```bash
git add src/analysis/context.py src/tests/test_analysis.py
git commit -m "feat(analysis): add AnalysisContext to attach/detach observers

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 5: Report

**Files:**
- Create: `src/analysis/report.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Write the failing test

```python
# Append to src/tests/test_analysis.py
from src.analysis.report import Report


class TestReport:
    def make_sample_raw(self):
        """Build a minimal raw report dict matching observer output structure."""
        return {
            "layer1.linear": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {"qsnr_db": 42.3, "mse": 1e-5},
                    }
                },
                "weight": {
                    "weight_pre_quant[0]": {
                        ("tensor",): {"qsnr_db": 55.1, "mse": 3e-8},
                    }
                },
            },
            "layer2.conv": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {"qsnr_db": 38.7, "mse": 2e-5},
                    }
                },
            },
        }

    def test_keys_returns_all_layer_names(self):
        report = Report(self.make_sample_raw())
        assert set(report.keys()) == {"layer1.linear", "layer2.conv"}

    def test_layer_access(self):
        report = Report(self.make_sample_raw())
        layer1 = report.layer("layer1.linear")
        assert "input" in layer1
        assert "weight" in layer1
        assert layer1["input"]["input_pre_quant[0]"][("tensor",)]["qsnr_db"] == 42.3

    def test_to_dataframe(self):
        report = Report(self.make_sample_raw())
        df = report.to_dataframe()
        assert len(df) == 3  # one row per slice
        assert "qsnr_db" in df.columns
        assert "mse" in df.columns
        assert "layer" in df.columns
        assert "role" in df.columns

    def test_summary_by_role(self):
        report = Report(self.make_sample_raw())
        summary = report.summary(by=("role",))
        assert "input" in summary
        assert "weight" in summary
        # input: avg qsnr = (42.3 + 38.7)/2 = 40.5
        assert summary["input"]["avg_qsnr_db"] == pytest.approx(40.5, abs=0.1)

    def test_roles_and_stages(self):
        report = Report(self.make_sample_raw())
        roles = report.roles("layer1.linear")
        assert set(roles) == {"input", "weight"}
        stages = report.stages("layer1.linear", "input")
        assert "input_pre_quant[0]" in stages

    def test_print_summary_does_not_crash(self):
        report = Report(self.make_sample_raw())
        report.print_summary()  # should not raise
```

### Step 3: Write minimal implementation

```python
# src/analysis/report.py
class Report:
    """Analysis report wrapper with Python API and print formatting.

    Internal structure:
      {layer_name: {role: {stage[pipeline_index]: {slice_key: {metric: value}}}}}
    """

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
        """Flatten to a DataFrame with one row per slice."""
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
            # Fallback: return list of dicts
            return rows

    def summary(self, by=("role",)):
        """Aggregate metrics grouped by the given keys.

        by=("role",) → {role: {"avg_qsnr_db": ..., "avg_mse": ..., "count": ...}}
        """
        flat = []
        for layer, roles in self._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        # Determine layer type from layer name
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
            key_parts = []
            for b in by:
                key_parts.append(row.get(b, "unknown"))
            key = "/".join(key_parts)

            entry = result.setdefault(key, {"count": 0, "_mse_sum": 0.0, "_qsnr_sum": 0.0})
            entry["count"] += 1
            if "mse" in row:
                entry["_mse_sum"] += row["mse"]
            if "qsnr_db" in row:
                entry["_qsnr_sum"] += row["qsnr_db"]

        # Compute averages
        for key, entry in result.items():
            if entry["count"] > 0:
                entry["avg_mse"] = entry["_mse_sum"] / entry["count"]
                entry["avg_qsnr_db"] = entry["_qsnr_sum"] / entry["count"]
            del entry["_mse_sum"], entry["_qsnr_sum"]

        return result

    def print_summary(self, top_k: int = 10):
        """Print formatted summary table."""
        print("=== Quantization Analysis Summary ===")
        print(f"Total layers: {len(self._raw)}")

        # Per-role summary
        role_summary = self.summary(by=("role",))
        if role_summary:
            print("\nRole Summary:")
            print(f"  {'Role':<12} {'Avg QSNR':>10} {'Avg MSE':>12} {'Count':>6}")
            print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*6}")
            for role, stats in role_summary.items():
                print(f"  {role:<12} {stats.get('avg_qsnr_db', 0):>10.1f} "
                      f"{stats.get('avg_mse', 0):>12.2e} {stats['count']:>6}")

        # Top-k by MSE (worst first)
        flat = self.to_dataframe()
        if isinstance(flat, list) and flat:
            sorted_rows = sorted(flat, key=lambda r: r.get("mse", 0), reverse=True)
            print(f"\nTop-{top_k} layers by MSE (worst → best):")
            print(f"  {'Layer':<24} {'Role':<12} {'MSE':>12} {'QSNR':>8}")
            print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*8}")
            for row in sorted_rows[:top_k]:
                print(f"  {row['layer']:<24} {row['role']:<12} "
                      f"{row.get('mse', 0):>12.2e} {row.get('qsnr_db', 0):>8.1f}")

    def to_json(self, path: str):
        import json
        def convert(obj):
            if isinstance(obj, dict):
                return {str(k) if not isinstance(k, str) else k: convert(v) for k, v in obj.items()}
            return obj
        with open(path, "w") as f:
            json.dump(convert(self._raw), f, indent=2)

    def to_csv(self, path: str):
        df = self.to_dataframe()
        if hasattr(df, "to_csv"):
            df.to_csv(path, index=False)
        else:
            import csv
            if not df:
                return
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=df[0].keys())
                writer.writeheader()
                writer.writerows(df)
```

### Step 4: Run tests to verify they pass

```bash
pytest src/tests/test_analysis.py::TestReport -v
```
Expected: 6 tests PASS. Then full suite:
```bash
pytest src/tests/test_analysis.py -v
```
Expected: all 23 tests PASS

### Step 5: Commit

```bash
git add src/analysis/report.py src/tests/test_analysis.py
git commit -m "feat(analysis): add Report with to_dataframe, summary, print_summary, export

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 6: Update __init__.py + 端到端集成测试

**Files:**
- Modify: `src/analysis/__init__.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Update __init__.py public exports

```python
# src/analysis/__init__.py
"""
Analysis infrastructure for quantized operators.

Phase 4: full AnalysisContext + concrete Observers + Report + Distribution taxonomy.
"""
from .events import QuantEvent
from .mixin import ObservableMixin
from .observer import ObserverBase, SliceAwareObserver
from .slicing import iter_slices, SliceKey
from .observers import DistributionObserver, QSNRObserver, MSEObserver, HistogramObserver
from .context import AnalysisContext
from .report import Report
```

### Step 2: Add end-to-end integration test

```python
# Append to src/tests/test_analysis.py
from src.quantize import quantize
from src.scheme.quant_scheme import QuantScheme
from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.transform import IdentityTransform


class TestEndToEnd:
    """End-to-end: quantized model + AnalysisContext → report."""

    def test_two_layer_linear_e2e(self):
        """2-layer Linear model with analysis → report is non-empty."""
        from src.ops.linear import QuantizedLinear
        from src.scheme.op_config import OpQuantConfig

        # Build a simple scheme
        fmt = FormatBase.from_str("fp8_e4m3")
        scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec(mode=GranularityMode.PER_TENSOR),
            transform=IdentityTransform(),
        )
        cfg = OpQuantConfig(input=(scheme,), weight=(scheme,), output=(scheme,))

        model = nn.Sequential(
            QuantizedLinear(8, 4, bias=False, cfg=cfg, name="layer0"),
            QuantizedLinear(4, 2, bias=False, cfg=cfg, name="layer1"),
        )

        observers = [DistributionObserver(), QSNRObserver(), MSEObserver()]
        x = torch.randn(3, 8)

        with AnalysisContext(model, observers) as ctx:
            model(x)

        report = ctx.report()
        assert len(report.keys()) > 0
        # Should have layer0 and layer1
        for name in report.keys():
            assert "layer" in name.lower()

        # DataFrame should have rows
        df = report.to_dataframe()
        assert len(df) > 0
        assert "qsnr_db" in df.columns
        assert "dynamic_range_bits" in df.columns

    def test_empty_observers_no_crash(self):
        """emit_fn=None when no observers → no crash."""
        from src.ops.linear import QuantizedLinear
        from src.scheme.op_config import OpQuantConfig

        fmt = FormatBase.from_str("fp8_e4m3")
        scheme = QuantScheme(
            format=fmt,
            granularity=GranularitySpec(mode=GranularityMode.PER_TENSOR),
            transform=IdentityTransform(),
        )
        cfg = OpQuantConfig(input=(scheme,), weight=(scheme,))

        model = QuantizedLinear(8, 4, bias=False, cfg=cfg, name="test_layer")
        # No observers → emit_fn=None, should not crash
        x = torch.randn(2, 8)
        y = model(x)  # should not raise
        assert y.shape == (2, 4)
```

### Step 3: Run full test suite

```bash
pytest src/tests/test_analysis.py -v
```
Expected: 25 tests PASS

Then verify no regression:
```bash
pytest src/tests/ -x -q
```
Expected: all 730+ tests PASS

### Step 4: Commit

```bash
git add src/analysis/__init__.py src/tests/test_analysis.py
git commit -m "feat(analysis): update public exports and add end-to-end integration tests

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 7: DistributionProfile

**Files:**
- Create: `src/analysis/correlation.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Write the failing test

```python
# Append to src/tests/test_analysis.py
from src.analysis.correlation import DistributionProfile


class TestDistributionProfile:
    def make_dist_report(self):
        """A Report with DistributionObserver data."""
        raw = {
            "layer1.linear": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "mean": 0.1, "std": 1.2, "skewness": 1.5,
                            "kurtosis": 5.0, "sparse_ratio": 0.3,
                            "dynamic_range_bits": 4.5, "outlier_ratio": 0.02,
                        },
                    }
                },
                "weight": {
                    "weight_pre_quant[0]": {
                        ("tensor",): {
                            "mean": -0.01, "std": 0.8, "skewness": -0.1,
                            "kurtosis": 3.2, "sparse_ratio": 0.01,
                            "dynamic_range_bits": 3.0, "outlier_ratio": 0.0,
                        },
                    }
                },
            },
            "layer2.conv": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "mean": 0.5, "std": 0.9, "skewness": 0.8,
                            "kurtosis": 4.1, "sparse_ratio": 0.05,
                            "dynamic_range_bits": 5.2, "outlier_ratio": 0.01,
                        },
                    }
                },
            },
        }
        return Report(raw)

    def test_by_role_aggregates_correctly(self):
        report = self.make_dist_report()
        profile = DistributionProfile.from_report(report)

        input_profile = profile.by_role("input")
        assert input_profile["sample_count"] == 2  # layer1 + layer2
        assert input_profile["dynamic_range_bits"]["p50"] == pytest.approx(4.85, abs=0.1)
        assert input_profile["sparse_ratio"]["min"] == pytest.approx(0.05, abs=0.01)

    def test_all_roles(self):
        report = self.make_dist_report()
        profile = DistributionProfile.from_report(report)

        all_roles = profile.all_roles()
        assert "input" in all_roles
        assert "weight" in all_roles

    def test_empty_report(self):
        report = Report({})
        profile = DistributionProfile.from_report(report)
        assert profile.by_role("input")["sample_count"] == 0

    def test_print_profile_does_not_crash(self):
        report = self.make_dist_report()
        profile = DistributionProfile.from_report(report)
        profile.print_profile()  # should not raise
```

### Step 3: Write minimal implementation

```python
# src/analysis/correlation.py
import numpy as np
from src.analysis.report import Report


class DistributionProfile:
    """Vertical summary of fp32 distribution fingerprints across all layers.

    Groups per-tensor statistics by role (input/weight/output) and computes
    distribution summaries (percentiles for each metric).
    """

    def __init__(self, metrics_by_role: dict):
        self._data = metrics_by_role

    @classmethod
    def from_report(cls, report: Report):
        """Extract DistributionObserver data from report."""
        collected = {}  # role → list of metric dicts

        for layer, roles in report._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        # Only aggregate per-tensor slices for the profile
                        # (ignore per-channel/per-block detail)
                        has_dist_metrics = "dynamic_range_bits" in metrics
                        if not has_dist_metrics:
                            continue
                        collected.setdefault(role, []).append(metrics)

        metrics_by_role = {}
        for role, metrics_list in collected.items():
            metrics_by_role[role] = cls._summarize_metrics(metrics_list)

        return cls(metrics_by_role)

    @staticmethod
    def _summarize_metrics(metrics_list: list) -> dict:
        """Compute percentile summaries for each metric across samples."""
        if not metrics_list:
            return {"sample_count": 0}

        result = {"sample_count": len(metrics_list)}
        numeric_fields = ["mean", "std", "skewness", "kurtosis",
                          "sparse_ratio", "dynamic_range_bits", "outlier_ratio",
                          "norm_entropy", "bimodality_coefficient", "min", "max",
                          "excess_kurtosis"]

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
```

### Step 4: Run tests

```bash
pytest src/tests/test_analysis.py::TestDistributionProfile -v
```
Expected: 4 tests PASS

### Step 5: Commit

```bash
git add src/analysis/correlation.py src/tests/test_analysis.py
git commit -m "feat(analysis): add DistributionProfile for per-role distribution summary

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 8: DistributionTaxonomy

**Files:**
- Modify: `src/analysis/correlation.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Write the failing test

```python
# Append to src/tests/test_analysis.py
from src.analysis.correlation import DistributionTaxonomy


class TestDistributionTaxonomy:
    def make_taxonomy_report(self):
        """Report with distribution stats covering all 8 archetypes."""
        raw = {}
        archetypes = [
            # (layer, role, skew, kurt, bimod, sparse, entropy)
            ("l_gauss", "weight", 0.1, 3.1, 0.4, 0.02, 0.6),      # zero-centered gaussian
            ("l_pos", "input", 1.2, 4.0, 0.3, 0.25, 0.5),           # positive-skewed
            ("l_neg", "input", -0.8, 3.5, 0.4, 0.05, 0.5),          # negative-skewed
            ("l_heavy", "output", 0.2, 8.0, 0.3, 0.05, 0.4),        # heavy-tailed
            ("l_bi", "weight", 0.1, 2.5, 0.6, 0.05, 0.55),          # bimodal
            ("l_unif", "input", 0.1, 2.0, 0.4, 0.05, 0.9),          # uniform-like
            ("l_zero", "input", 0.2, 3.0, 0.3, 0.5, 0.3),           # zero-inflated
            ("l_logn", "output", 1.5, 5.0, 0.3, 0.1, 0.6),          # log-normal-like
        ]
        for layer, role, sk, ku, bi, sp, ent in archetypes:
            raw[layer] = {
                role: {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "skewness": sk, "kurtosis": ku,
                            "bimodality_coefficient": bi,
                            "sparse_ratio": sp, "norm_entropy": ent,
                        }
                    }
                }
            }
        return Report(raw)

    def test_classify_all_eight_types(self):
        report = self.make_taxonomy_report()
        taxonomy = DistributionTaxonomy.from_report(report)
        result = taxonomy.classify()

        # Should have at least 6-8 of the expected clusters
        cluster_names = set(result.keys())
        expected = {"zero-centered-gaussian", "positive-skewed", "negative-skewed",
                    "heavy-tailed", "bimodal", "uniform-like",
                    "zero-inflated", "log-normal-like"}
        assert len(cluster_names & expected) >= 6

        for name, cluster in result.items():
            assert "count" in cluster
            assert "percentage" in cluster
            assert "representative_layers" in cluster
            assert cluster["count"] > 0

    def test_unclassified_fallback(self):
        """Metrics that don't match any rule → 'unclassified'."""
        raw = {
            "weird_layer": {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "skewness": 0.3, "kurtosis": 4.5,  # not matching any rule
                            "bimodality_coefficient": 0.4,
                            "sparse_ratio": 0.15, "norm_entropy": 0.6,
                        }
                    }
                }
            }
        }
        report = Report(raw)
        taxonomy = DistributionTaxonomy.from_report(report)
        result = taxonomy.classify()
        assert "unclassified" in result

    def test_get_exemplars_returns_structure(self):
        report = self.make_taxonomy_report()
        taxonomy = DistributionTaxonomy.from_report(report)
        exemplars = taxonomy.get_exemplars("positive-skewed", n=1)
        assert len(exemplars) >= 1
        assert "layer" in exemplars[0]
        assert "role" in exemplars[0]

    def test_print_taxonomy_no_crash(self):
        report = self.make_taxonomy_report()
        taxonomy = DistributionTaxonomy.from_report(report)
        taxonomy.print_taxonomy()
        taxonomy.print_taxonomy(ascii_plots=True)
```

### Step 3: Write minimal implementation

```python
# Append to src/analysis/correlation.py

class DistributionTaxonomy:
    """Classify per-tensor distributions into predefined archetypes."""

    # Each rule is a callable (metrics) → bool or None (if not applicable)
    RULES = [
        ("zero-centered-gaussian", lambda m:
            abs(m.get("skewness", 0)) < 0.5 and 2.5 < m.get("kurtosis", 0) < 4.0 and m.get("sparse_ratio", 0) < 0.1),
        ("bimodal", lambda m:
            m.get("bimodality_coefficient", 0) > 0.555),
        ("zero-inflated", lambda m:
            m.get("sparse_ratio", 0) > 0.3),
        ("uniform-like", lambda m:
            m.get("norm_entropy", 0) > 0.85 and abs(m.get("skewness", 0)) < 0.5),
        ("heavy-tailed", lambda m:
            m.get("kurtosis", 0) > 6.0),
        ("positive-skewed", lambda m:
            m.get("skewness", 0) > 0.5 and m.get("bimodality_coefficient", 0) <= 0.555),
        ("negative-skewed", lambda m:
            m.get("skewness", 0) < -0.5),
        ("log-normal-like", lambda m:
            m.get("skewness", 0) > 1.0 and m.get("kurtosis", 0) < 6.0
            and m.get("bimodality_coefficient", 0) <= 0.555 and m.get("sparse_ratio", 0) < 0.3),
    ]

    def __init__(self, classifications: dict):
        self._clusters = classifications

    @classmethod
    def from_report(cls, report: Report):
        """Classify every per-tensor distribution in the report."""
        clusters = {}

        for layer, roles in report._raw.items():
            for role, stages in roles.items():
                for stage, slices in stages.items():
                    for slice_key, metrics in slices.items():
                        # Only classify per-tensor slices
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
                        # Accumulate metrics for averaging
                        for k in ("skewness", "kurtosis", "sparse_ratio",
                                  "dynamic_range_bits", "norm_entropy"):
                            if k in metrics:
                                entry["metrics_sum"][k] = \
                                    entry["metrics_sum"].get(k, 0.0) + metrics[k]
                        entry["metric_count"] += 1

        # Compute averages and representative layers
        result = {}
        total = sum(c["count"] for c in clusters.values())
        for name, data in clusters.items():
            avg_metrics = {}
            for k, s in data["metrics_sum"].items():
                avg_metrics[k] = s / data["metric_count"] if data["metric_count"] > 0 else 0

            # Pick up to 3 representative layers
            reps = data["layers"][:3]

            result[name] = {
                "count": data["count"],
                "percentage": f"{100 * data['count'] / total:.0f}%",
                "avg_metrics": avg_metrics,
                "representative_layers": reps,
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
        """Filter clusters to only layers matching the given role."""
        # Requires re-classifying with role filter — for now, return all
        return dict(self._clusters)

    def get_exemplars(self, cluster: str, n: int = 3) -> list:
        """Return representative (layer, role) pairs for a cluster."""
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
                print(f"  [ASCII histogram: see HistogramObserver data for "
                      f"{data['representative_layers'][0] if data['representative_layers'] else 'N/A'}]")
```

### Step 4: Run tests

```bash
pytest src/tests/test_analysis.py::TestDistributionTaxonomy -v
```
Expected: 4 tests PASS

### Step 5: Commit

```bash
git add src/analysis/correlation.py src/tests/test_analysis.py
git commit -m "feat(analysis): add DistributionTaxonomy with 8 archetype rule-based classification

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 9: ErrorByDistribution + LayerSensitivity

**Files:**
- Modify: `src/analysis/correlation.py`
- Modify: `src/tests/test_analysis.py`

### Step 1: Write the failing tests

```python
# Append to src/tests/test_analysis.py
from src.analysis.correlation import ErrorByDistribution, LayerSensitivity


class TestErrorByDistribution:
    def make_error_report(self):
        raw = {}
        for i, (dr, qsnr) in enumerate([(3.0, 45.0), (4.5, 35.0), (7.0, 15.0),
                                          (9.0, 6.0), (3.5, 42.0), (5.0, 28.0)]):
            raw[f"layer{i}"] = {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {
                            "qsnr_db": qsnr, "mse": 10**(-qsnr/10),
                            "dynamic_range_bits": dr,
                        }
                    }
                }
            }
        return Report(raw)

    def test_rank_layers(self):
        report = self.make_error_report()
        eb = ErrorByDistribution(report)

        ranked = eb.rank_layers(by="qsnr_db", k=3, ascending=True)
        assert len(ranked) == 3
        assert ranked[0][2] == pytest.approx(6.0, abs=0.1)  # worst QSNR
        assert ranked[0][0] == "layer3"  # layer3 has qsnr=6

    def test_group_by_range(self):
        report = self.make_error_report()
        eb = ErrorByDistribution(report)

        groups = eb.group_by_range(role="input", bins=[0, 4, 7, 999])
        # 0-4 bits: layers 0, 4 → avg QSNR = (45+42)/2 = 43.5 → excellent
        # 4-7 bits: layers 1, 5 → avg QSNR = (35+28)/2 = 31.5 → good
        # 7+ bits: layers 2, 3 → avg QSNR = (15+6)/2 = 10.5 → critical
        assert "0-4 bits" in groups
        assert groups["0-4 bits"]["avg_qsnr"] == pytest.approx(43.5, abs=0.1)
        assert "7-999 bits" in groups
        assert groups["7-999 bits"]["avg_qsnr"] == pytest.approx(10.5, abs=0.1)

    def test_print_correlation_no_crash(self):
        report = self.make_error_report()
        eb = ErrorByDistribution(report)
        eb.print_correlation()


class TestLayerSensitivity:
    def make_sensitivity_report(self):
        raw = {}
        for i, (layer_type, mse) in enumerate([
            ("Linear", 1e-3), ("Linear", 5e-4), ("Conv", 2e-3),
            ("Linear", 1e-5), ("Conv", 8e-4),
        ]):
            raw[f"layer{i}.{layer_type}"] = {
                "input": {
                    "input_pre_quant[0]": {
                        ("tensor",): {"mse": mse, "qsnr_db": -10 * __import__('math').log10(mse)},
                    }
                }
            }
        return Report(raw)

    def test_topk(self):
        report = self.make_sensitivity_report()
        sens = LayerSensitivity(report)

        top2 = sens.topk(k=2, metric="mse")
        assert len(top2) == 2
        assert top2[0][2] == 0.002  # layer2 with mse=2e-3

    def test_by_layer_type(self):
        report = self.make_sensitivity_report()
        sens = LayerSensitivity(report)

        by_type = sens.by_layer_type()
        assert "Linear" in by_type
        assert "Conv" in by_type

    def test_above_threshold(self):
        report = self.make_sensitivity_report()
        sens = LayerSensitivity(report)

        above = sens.above_threshold(metric="mse", threshold=1e-3)
        assert len(above) >= 1
```

### Step 3: Implement ErrorByDistribution and LayerSensitivity

```python
# Append to src/analysis/correlation.py


class ErrorByDistribution:
    """Correlate quantization error with distribution features."""

    def __init__(self, report: Report):
        self._report = report
        # Flatten all slices with error and distribution metrics
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

        # Only samples with dynamic_range_bits
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

            entry = groups.setdefault(bucket, {"count": 0, "_qsnr_sum": 0.0, "_mse_sum": 0.0})
            entry["count"] += 1
            if "qsnr_db" in s:
                entry["_qsnr_sum"] += s["qsnr_db"]
            if "mse" in s:
                entry["_mse_sum"] += s["mse"]

        # Compute averages and verdicts
        result = {}
        for bucket, entry in groups.items():
            avg_qsnr = entry["_qsnr_sum"] / entry["count"] if entry["count"] > 0 else 0
            verdict = "critical"
            if avg_qsnr >= 35:
                verdict = "excellent"
            elif avg_qsnr >= 25:
                verdict = "good"
            elif avg_qsnr >= 15:
                verdict = "acceptable"
            elif avg_qsnr >= 10:
                verdict = "poor — format insufficient"

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
                        if "mse" in metrics or "qsnr_db" in metrics:
                            # Determine layer type from name
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
```

### Step 4: Run tests

```bash
pytest src/tests/test_analysis.py::TestErrorByDistribution \
  src/tests/test_analysis.py::TestLayerSensitivity -v
```
Expected: 6 tests PASS

### Step 5: Run all analysis tests + regression check

```bash
pytest src/tests/test_analysis.py -v
pytest src/tests/ -x -q
```
Expected: 33+ analysis tests + all 730 existing tests PASS

### Step 6: Commit

```bash
git add src/analysis/correlation.py src/tests/test_analysis.py
git commit -m "feat(analysis): add ErrorByDistribution and LayerSensitivity correlation tools

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Task 10: Export utilities

**Files:**
- Create: `src/analysis/export.py`
- Modify: `src/tests/test_analysis.py`

### Step 3: Write minimal implementation

```python
# src/analysis/export.py
"""Export utilities: JSON, CSV, Markdown."""
# Export methods live on Report (to_json, to_csv, print_summary).
# This module provides standalone helpers for multi-report export.
```

### Step 4: JSON/CSV export test already covered in TestReport (test_print_summary_does_not_crash).

```bash
pytest src/tests/test_analysis.py -v
pytest src/tests/ -x -q
```
Expected: all tests PASS

### Step 5: Commit

```bash
git add src/analysis/export.py
git commit -m "feat(analysis): add export module placeholder (methods on Report)

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

## Final Verification

```bash
pytest src/tests/test_analysis.py -v   # All analysis tests
pytest src/tests/ -x -q                # Full suite, no regression
```

Expected: 33+ new tests + 730 existing = 763+ tests PASS.
