# Phase 4: 层级误差分析 — 设计文档

**状态**: 已确认
**日期**: 2026-04-26
**依赖**: Phase 3 完成（730 tests，bit-exact）

---

## 1. 目标和验收标准

Phase 4 在 Phase 3 已铺设的 `ObservableMixin` / `QuantEvent` / `iter_slices` 骨架上，构建完整的**单配置**量化分析能力。

### 用户需求

1. **分布分析**：分析每一层 input / weight / output 的数值分布，自动归类到常见分布簇（高斯/偏态/重尾/双峰等），纵向汇总（如"所有 input 有哪些分布类型"），并能画出各类代表性分布
2. **量化误差分析**：每层每个 tensor 的 QSNR / MSE，关联分布特征回答"当前格式对哪些分布适用"
3. **灵敏度排名**：全局排序，快速定位问题层

### 验收标准

- 所有已有 730 tests 保持全绿（零 regression）
- `pytest src/tests/test_analysis.py -x` 全绿
- `AnalysisContext` 可以在任意量化模型上挂载 observer 并收集数据
- `DistributionObserver` 产出 per-tensor 统计指纹（含 dynamic_range_bits）
- `report.to_dataframe()` 和 `report.print_summary()` 正常工作
- 不保留任何 fp32 原始张量（零 OOM 风险）

### 不在 Phase 4 范围

- 多格式对比（Phase 4.5）
- 格式推荐引擎（Phase 4.5）
- QAT backward 分析（Phase 4 inference-only）
- RNN 家族

---

## 2. 已有基础设施（Phase 3 遗留）

```
src/analysis/
├── __init__.py        # 导出 QuantEvent, ObservableMixin, ObserverBase, SliceAwareObserver, iter_slices
├── events.py          # QuantEvent frozen dataclass（完整 type guard）
├── mixin.py           # ObservableMixin._emit() + _observers property
├── observer.py        # ObserverBase (ABC), SliceAwareObserver (ABC)
└── slicing.py         # iter_slices: PER_TENSOR / PER_CHANNEL / PER_BLOCK / DYNAMIC_GROUP
```

所有量化算子已通过 `emit_fn` 回调模式在关键点发射事件：
- forward: `input_pre_quant` / `weight_pre_quant` / `output_post_quant`
- backward: `grad_output_pre_quant` / `grad_weight_post_quant` / `grad_input_post_quant`

### 关键设计约定（不变）

- `emit_fn` 由 `QuantizedXxx.forward()` 传入 Function：`emit_fn = self._emit if self._observers else None`
- Function 内通过 `if emit_fn: emit_fn(...)` 零开销触发
- `ObservableMixin._emit()` 创建 `QuantEvent` 并分发给所有 observer

---

## 3. Phase 4 新增组件

### 3.1 子阶段划分

| 子阶段 | 内容 | 新增/修改文件 |
|---|---|---|
| P4.1 | 4 个具体 Observer + AnalysisContext + Report | `observers.py`, `context.py`, `report.py`, `__init__.py` |
| P4.2 | 分布画像汇总 + 误差关联 + 灵敏度排名 | `correlation.py` |
| P4.3 | 导出（JSON/CSV/Markdown） | `export.py` |

依赖链：P4.1 → P4.2 → P4.3

---

## 4. P4.1 详细设计：Observer + Context + Report

### 4.1 DistributionObserver

捕捉 fp32 张量的**统计指纹**，是需求 1（分布分析）的核心。

```python
# src/analysis/observers.py

class DistributionObserver(SliceAwareObserver):
    """Per-slice 收集 fp32 统计特征，用于分布归类和格式匹配。

    每个 slice 记录：
      - min, max, mean, std
      - skewness: E[(x-μ)³] / σ³ —— 正=右偏（ReLU激活），负=左偏，≈0=对称（高斯）
      - kurtosis: E[(x-μ)⁴] / σ⁴（excess = kurtosis - 3）
      - bimodality_coefficient: Sarle 系数 (skew²+1)/(kurtosis+3*(n-1)²/((n-2)*(n-3))) >0.555→双峰
      - sparse_ratio: |x| < eps 的比例
      - dynamic_range_bits: log2(max(|x|) / min_nonzero)
      - outlier_ratio: 超过 3σ 的比例
      - entropy: 基于直方图的归一化 Shannon 熵（0=单点, 1=均匀）

    限制：只存标量统计，不保留原始张量。
    """

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

        # Normalized entropy from histogram
        hist = torch.histc(f, bins=self.hist_bins)
        probs = hist.float() / (n + 1e-30)
        probs_pos = probs[probs > 0]
        entropy = -(probs_pos * torch.log2(probs_pos + 1e-30)).sum().item()
        max_entropy = torch.log2(torch.tensor(self.hist_bins, dtype=torch.float32)).item()
        norm_entropy = entropy / (max_entropy + 1e-30)

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

`dynamic_range_bits` 是连接"分布"和"格式选择"的桥梁：
- 值 ≤ 3 → FP8 E4M3 可覆盖
- 值 ≤ 6 → FP8 E5M2 可覆盖
- 值 > 8 → 需要 FP16 或 block-wise 量化

### 4.2 QSNRObserver

```python
class QSNRObserver(SliceAwareObserver):
    """QSNR = 10 * log10(||fp32||² / ||fp32 - quant||²)，单位 dB。"""

    def _measure(self, key, fp32, quant):
        err = fp32 - quant
        num = fp32.pow(2).mean()
        den = err.pow(2).mean().clamp_min(1e-30)
        return {"qsnr_db": (10 * torch.log10(num / den)).item()}
```

### 4.3 MSEObserver

```python
class MSEObserver(SliceAwareObserver):
    def _measure(self, key, fp32, quant):
        return {"mse": (fp32 - quant).pow(2).mean().item()}
```

### 4.4 HistogramObserver

```python
class HistogramObserver(SliceAwareObserver):
    """fp32 / quant / error 三通道直方图。只用 per-tensor 粒度（不分 slice），
    内部在 _measure 中判断 key == ("tensor",)，非 tensor 级不收集直方图。"""

    def __init__(self, n_bins: int = 128):
        super().__init__()
        self.n_bins = n_bins

    def _measure(self, key, fp32, quant):
        # 直方图只在 per-tensor 粒度收集，避免数据膨胀
        return {
            "fp32_hist": torch.histc(fp32, bins=self.n_bins).cpu(),
            "quant_hist": torch.histc(quant, bins=self.n_bins).cpu(),
            "err_hist": torch.histc(fp32 - quant, bins=self.n_bins).cpu(),
        }
```

说明：HistogramObserver 继承 `SliceAwareObserver`，每个 slice 都会调用 `_measure`。
当 granularity 为 PER_CHANNEL 或 PER_BLOCK 时，会产生大量直方图。这是预期行为——
用户选择细粒度分析时，自然获得更多数据。如需控制规模，可通过 `n_bins` 或 granularity 选择来调节。

### 4.5 AnalysisContext

```python
# src/analysis/context.py

class AnalysisContext:
    """上下文管理器：挂载 observers 到模型中所有 ObservableMixin 模块。

    用法：
        with AnalysisContext(model, [QSNRObserver(), DistributionObserver()]) as ctx:
            for batch in calibration_data:
                model(batch)
        report = ctx.report()
    """

    def __init__(self, model: nn.Module, observers=None,
                 warmup_batches: int = 0):
        self.model = model
        self.observers = observers or [QSNRObserver()]
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

    def report(self) -> "Report":
        """聚合所有 observer 的数据，返回 Report 对象。"""
        ...

    def step(self):
        """标记一个 batch 完成。warmup_batches 内的数据被丢弃。"""
        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            for obs in self.observers:
                obs.reset()
```

### 4.6 Report

```python
# src/analysis/report.py

class Report:
    """分析报告的 Python API 包装。

    内部结构：
      {layer_name: {role: {stage[pipeline_index]: {slice_key: {metric: value}}}}}
    """

    def __init__(self, raw: dict):
        self._raw = raw

    def keys(self) -> list[str]: ...
    def layer(self, name: str) -> dict: ...
    def to_dataframe(self) -> "pd.DataFrame": ...
    def summary(self, by=("role",)) -> dict: ...
    def print_summary(self, top_k: int = 10): ...
    def to_json(self, path: str): ...
    def to_csv(self, path: str): ...

    # 查询权限
    def roles(self, layer: str) -> list[str]: ...
    def stages(self, layer: str, role: str) -> list[str]: ...
```

`print_summary()` 输出示例：
```
=== Quantization Analysis Summary ===
Total layers: 52 | Format: FP8_E4M3 | Granularity: per-tensor

Top-10 layers by QSNR (worst → best):
  Rank  Layer              Role     QSNR(dB)  MSE       Range(bits)
  1     layer12.conv2       input     8.3     2.1e-03      9.7 ⚠
  2     layer7.linear       weight   12.1     6.3e-04      7.2
  ...

Role Summary:
  Role     Avg QSNR   Avg MSE    Layers  >30dB%
  input     38.2      8.1e-06      52     84%
  weight    45.1      2.2e-07      48     96%
  output    42.7      3.5e-06      36     91%
```

---

## 5. P4.2 详细设计：分布画像 + 误差关联

### 5.1 DistributionProfile

```python
# src/analysis/correlation.py

class DistributionProfile:
    """按 role 纵向汇总所有层的 fp32 分布指纹。

    从 Report 中提取 DistributionObserver 的数据，按 role 分组聚合。
    """

    @classmethod
    def from_report(cls, report: Report) -> "DistributionProfile": ...

    def by_role(self, role: str) -> dict:
        """返回该 role 的汇总统计：
        {
          "sample_count": N,
          "dynamic_range_bits": {"min": ..., "p25": ..., "p50": ..., "p75": ..., "max": ...},
          "sparse_ratio": {...},
          "outlier_ratio": {...},
          "std": {...},
        }
        """

    def all_roles(self) -> dict[str, dict]: ...
    def print_profile(self): ...
```

### 5.2 DistributionTaxonomy — 分布归类

将 per-tensor 的统计特征（skewness/kurtosis/bimodality/sparsity/entropy）
匹配到预定义的分布簇中，回答"这个模型里都有什么类型的分布"。

**8 个预定义分布簇 + 判定规则：**

| 分布类型 | 判定规则 | 典型出现位置 |
|---|---|---|
| 零中心高斯 | \|skew\| < 0.5, 2.5 < kurt < 4, sparse < 0.1 | 初始化后的 Linear weight |
| 正偏态 | skew > 0.5, bimod < 0.555 | ReLU 后 activation |
| 负偏态 | skew < -0.5 | 某些 pre-activation / grad |
| 重尾 | kurt > 6.0 | Transformer attention output |
| 双峰 | bimod > 0.555 | Norm 后分布，量化后 weight |
| 均匀 | norm_entropy > 0.85, \|skew\| < 0.5 | Embedding / bias |
| 零膨胀 | sparse_ratio > 0.3 | ReLU/GELU 后，pruned weight |
| 对数正态 | skew > 1.0, kurt < 6, bimod < 0.555, sparse < 0.3 | LayerNorm 输出 |

判定顺序：从上往下匹配，先命中的为准。都不命中 → `"unclassified"`。

```python
# src/analysis/correlation.py

class DistributionTaxonomy:
    """按统计特征将每层张量归入分布簇。"""

    # 预定义分布类型
    PATTERNS = {
        "zero-centered-gaussian": ...,
        "positive-skewed": ...,
        "negative-skewed": ...,
        "heavy-tailed": ...,
        "bimodal": ...,
        "uniform-like": ...,
        "zero-inflated": ...,
        "log-normal-like": ...,
    }

    @classmethod
    def from_report(cls, report: Report) -> "DistributionTaxonomy": ...

    def classify(self) -> dict:
        """返回分类结果：
        {
          "zero-centered-gaussian": {
            "count": 23, "percentage": "46%",
            "avg_metrics": {"skewness": 0.02, "kurtosis": 3.1, ...},
            "representative_layers": [
              ("layer3.linear", "weight"),
              ("layer7.conv2", "weight"),
              ...
            ],
          },
          "positive-skewed": { ... },
          ...
        }
        """

    def classify_by_role(self, role: str) -> dict:
        """按 role 过滤后的归类。"""

    def print_taxonomy(self, ascii_plots: bool = False):
        """打印分布归类摘要表。

        当 ascii_plots=True 时，每类画 1 个代表性层的 ASCII 直方图（基于
        HistogramObserver 数据）。"""

    def get_exemplars(self, cluster: str, n: int = 3) -> list[dict]:
        """返回该分布簇中 n 个代表性层的 histogram 数据。

        每项 = {"layer": str, "role": str, "hist_bins": [...], "hist_counts": [...]}
        用户可直接用 matplotlib 画图：
            for ex in taxonomy.get_exemplars("positive-skewed"):
                plt.bar(ex["hist_bins"], ex["hist_counts"])
        """
```

**ASCII 直方图示例**（`print_taxonomy(ascii_plots=True)` 输出）：
```
positive-skewed (38 layers, 54%)
  Representative: layer5.relu.input
    min=-0.12  max=2.87  mean=0.34  skew=+1.42  sparse=22%
     -0.1 [          ] 
      0.0 [██████    ] 
      0.5 [██████████] 
      1.0 [██████    ] 
      1.5 [███       ] 
      2.0 [█         ] 
      2.5 [          ] 
```

### 5.3 ErrorByDistribution

```python
class ErrorByDistribution:
    """将量化误差与分布特征关联。

    核心方法 group_by_range：按 dynamic_range_bits 分桶，统计每桶的平均 QSNR。
    """

    def __init__(self, report: Report): ...

    def rank_layers(self, by="qsnr_db", role=None, ascending=True, k=None) -> list:
        """返回 (layer_name, role, metric_value) 排名列表。"""

    def group_by_range(self, role=None, bins=None) -> dict:
        """按 dynamic_range_bits 分桶：
        {
          "2-4 bits":  {"avg_qsnr": 48.2, "count": 12, "verdict": "excellent"},
          "4-6 bits":  {"avg_qsnr": 35.7, "count": 20, "verdict": "good"},
          "6-8 bits":  {"avg_qsnr": 18.3, "count": 8,  "verdict": "poor — format insufficient"},
          "8+ bits":   {"avg_qsnr": 6.1,  "count": 3,  "verdict": "critical"},
        }
        """

    def print_correlation(self): ...
```

verdict 阈值（可配置）：
- QSNR ≥ 35 dB → "excellent"
- QSNR ≥ 25 dB → "good"
- QSNR ≥ 15 dB → "acceptable"
- QSNR < 15 dB → "poor — format insufficient"
- QSNR < 10 dB → "critical"

### 5.4 LayerSensitivity

```python
class LayerSensitivity:
    """全局灵敏度排名。"""

    def __init__(self, report: Report): ...

    def topk(self, k=10, role=None, metric="mse") -> list[tuple]:
        """最敏感的 k 层（误差最大）。"""

    def by_layer_type(self) -> dict:
        """按 Linear/Conv/Norm/Activation 分类聚合灵敏度。"""

    def above_threshold(self, metric, threshold) -> list:
        """筛选超过阈值的层。"""
```

---

## 6. P4.3 详细设计：导出

```python
# src/analysis/export.py

# Report 方法：
#   report.to_json(path)   — 完整 dump
#   report.to_csv(path)    — DataFrame 导出
#   report.to_dataframe()  — 返回 pd.DataFrame

# 独立函数：
#   from src.analysis.export import reports_to_excel
#   reports_to_excel({"run1": report1, "run2": report2}, "comparison.xlsx")
#   # 每个 sheet 一个 run，方便 Excel 横向对比（Phase 4.5 过渡用）
```

---

## 7. 最终文件结构

```
src/analysis/
├── __init__.py        # 更新导出：加 QSNRObserver, MSEObserver, DistributionObserver,
│                      #   HistogramObserver, AnalysisContext, Report,
│                      #   DistributionProfile, DistributionTaxonomy, ErrorByDistribution, LayerSensitivity
├── events.py          # (已有，不改)
├── mixin.py           # (已有，不改)
├── observer.py        # (已有，不改)
├── slicing.py         # (已有，不改)
├── observers.py       # NEW — 4 个具体 Observer
├── context.py         # NEW — AnalysisContext
├── report.py          # NEW — Report 包装类
├── correlation.py     # NEW — DistributionProfile / DistributionTaxonomy /
│                      #        ErrorByDistribution / LayerSensitivity
└── export.py          # NEW — JSON/CSV/Markdown 导出
```

---

## 8. 测试策略

### 新增测试文件：`src/tests/test_analysis.py`

| # | 测试内容 | 类型 |
|---|---|---|
| 1 | `DistributionObserver._measure` — 合成 fp32/quant 对，验证 min/max/mean/std 正确 | 单元 |
| 2 | `DistributionObserver._measure` — 全零张量（sparse_ratio=1, dynamic_range_bits=0） | 边界 |
| 3 | `DistributionObserver._measure` — 含异常值（outlier_ratio > 0）| 单元 |
| 3a | `DistributionObserver._measure` — skew/kurtosis/bimodality/norm_entropy 与理论值验证 | 单元 |
| 4 | `QSNRObserver._measure` — fp32=quant → QSNR=inf 处理 | 边界 |
| 5 | `QSNRObserver._measure` — 已知误差的解析验证 | 单元 |
| 6 | `MSEObserver._measure` — 与 torch 直接计算一致 | 单元 |
| 7 | `HistogramObserver._measure` — bin 数和形状验证 | 单元 |
| 8 | `AnalysisContext` — 小模型挂载/卸载 observer | 集成 |
| 9 | `AnalysisContext` — warmup_batches 跳过数据 | 集成 |
| 10 | `Report.to_dataframe()` — 结构验证 | 单元 |
| 11 | `Report.summary(by=...)` — 聚合值验证 | 单元 |
| 12 | `DistributionProfile.from_report()` — 合成 report 验证聚合 | 单元 |
| 13 | `DistributionTaxonomy.classify()` — 合成统计特征，8 种类型各归入正确簇 | 单元 |
| 14 | `DistributionTaxonomy.classify()` — "unclassified" 回退覆盖未知分布 | 单元 |
| 15 | `DistributionTaxonomy.get_exemplars()` — 返回 histogram 数据结构正确 | 单元 |
| 16 | `DistributionTaxonomy.print_taxonomy(ascii_plots=True)` — 无崩溃输出 | 单元 |
| 17 | `ErrorByDistribution.group_by_range()` — 分桶正确性 | 单元 |
| 18 | `ErrorByDistribution.rank_layers()` — 排序正确性 | 单元 |
| 19 | `LayerSensitivity.topk()` — top-k 正确性 | 单元 |
| 20 | 小模型端到端 — 2 层 Linear 量化模型 + AnalysisContext → report 非空 | 集成 |
| 21 | `report.to_json()` / `report.to_csv()` — 文件存在性 | 集成 |
| 22 | `ObservableMixin` with empty observers → emit_fn=None → 不崩溃 | 回归 |

所有测试 `pytest src/tests/ -x` 必须通过，不引入对已有 730 tests 的 regression。

---

## 9. 关键设计决策汇总

| 决策 | 结论 |
|---|---|
| 分析范围 | Inference only（forward pass） |
| 触发方式 | `with AnalysisContext(model, observers):` 上下文管理器 |
| 多格式对比 | 拆分到 Phase 4.5，Phase 4 只做单配置分析 |
| 报告交互 | Python API + `print_summary()` 两者都要 |
| 分布归类 | 规则匹配（8 个预定义簇），确定性强，零额外依赖 |
| 可视化 | ASCII 终端直方图 + `get_exemplars()` histogram data export |
| 内存策略 | `_measure` 只存标量/直方图，不保留原始张量 |
| 依赖链 | P4.1 → P4.2 → P4.3（按序推进） |

---

## 10. 子任务清单（总控）

### P4.1 — Observer + Context + Report
- [ ] P4.1-1: `DistributionObserver` + 测试 #1-3a, #6
- [ ] P4.1-2: `QSNRObserver` + 测试 #4-5
- [ ] P4.1-3: `MSEObserver` + 测试 #6 → 重新编号
- [ ] P4.1-4: `HistogramObserver` + 测试 #7
- [ ] P4.1-5: `AnalysisContext` + 测试 #8-9
- [ ] P4.1-6: `Report` + 测试 #10-11
- [ ] P4.1-7: 更新 `__init__.py` 公开导出
- [ ] P4.1-8: 小模型端到端集成测试 #20, #22

### P4.2 — 分布画像 + 分布归类 + 误差关联
- [ ] P4.2-1: `DistributionProfile.from_report()` + 测试 #12
- [ ] P4.2-2: `DistributionTaxonomy.classify()` + `get_exemplars()` + `print_taxonomy()` + 测试 #13-16
- [ ] P4.2-3: `ErrorByDistribution` + 测试 #17-18
- [ ] P4.2-4: `LayerSensitivity` + 测试 #19

### P4.3 — 导出
- [ ] P4.3-1: JSON / CSV 导出 + 测试 #21
- [ ] P4.3-2: `print_summary()` Markdown 格式化

---

## 11. Phase 4.5 展望（本设计不在范围内）

- `src/analysis/compare.py` — `compare_formats(model_fn, data, configs, observers)`
- 对比矩阵：每层 × 每种配置 × 每个角色的 QSNR / MSE
- 格式推荐引擎：基于 DistributionProfile + ErrorByDistribution 自动推荐最佳格式
- 校准数据稳定性分析：多个 calibration 子集对比误差波动
