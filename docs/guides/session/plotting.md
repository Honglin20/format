# 绘图 & 可视化

> 第 5 章 · [Session 文档索引](INDEX.md)

`SessionResult.plot` 和 `StudyReport.plot` 提供丰富的内置图表。所有方法返回 `matplotlib.Figure`，可 `plt.show()` 或 `fig.savefig()`。

## 基础用法

```python
from src.session import Study, QuantConfig

configs = [
    QuantConfig(name="int8", w_format="int8"),
    QuantConfig(name="int4", w_format="int4"),
    QuantConfig(name="fp4-mx", w_format="fp4_e2m1", w_granularity="per_block",
                w_block_size=32),
]

report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn)
```

## 单结果模式

`SessionResult.plot` 提供以下方法：

```python
result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)

# 质量评估
result.plot.qsnr_comparison()                  # 逐层 QSNR 柱状图
result.plot.crest_vs_qsnr()                    # crest factor vs QSNR（三 role 面板）
result.plot.outlier_analysis()                 # outlier 分析（双面板 × role）
result.plot.per_block_qsnr()                   # 逐 block QSNR 分布

# 误差传播
result.plot.error_propagation(role="output")   # 累积 vs 本地 QSNR 三行面板
result.plot.accumulated_vs_local(role="output")  # 累积 vs 本地散点图
result.plot.propagation_dag()                  # 误差传播水平柱状图
result.plot.error_waterfall()                  # 累积 QSNR 瀑布图
result.plot.local_vs_accum_scatter()           # 本地 vs 累积散点（headroom 着色）

# 直方图
result.plot.histogram_overlay(top_k=5)         # Top-5 敏感层三通道直方图
result.plot.histogram_overlay(layer="fc1")     # 单层三 role 直方图
result.plot.layer_histogram("fc1", role="weight")   # 单层单 role 详细直方图
result.plot.per_layer_role_histogram(k=5)      # 最差 k 层 × 三 role 网格

# 分布特征
result.plot.role_distribution_comparison()     # 三 role 分布特征箱线图
result.plot.correlation_heatmap()              # 分布特征 × 误差相关矩阵
result.plot.kurtosis_analysis()                # 峰度分析三面板

# Per-role QSNR
result.plot.per_role_qsnr_bars()               # 单层三 role QSNR 分组柱状图
result.plot.depth_decay(role="output")         # QSNR 随深度衰减
result.plot.per_layer_role_qsnr_line()         # 三 role QSNR 折线（或 accum 单线）

# 通道分析
result.plot.channel_heterogeneity("fc1", role="weight")  # 单层通道 QSNR 箱线图

# 成本
result.plot.cost_decomposition()               # FLOPs 堆叠柱状图

# 保存
result.save("results/my_config/")
```

单结果模式不含 `pareto_frontier()` — 需要多配置对比才有 trade-off 曲线。

## 一键保存（Study 模式）

```python
report.save("results/")
```

`save()` 自动检测哪些数据存在，只生成有数据的图表。输出结构：

```
results/
├── results.json
├── tables/
│   ├── accuracy.csv
│   ├── per_layer_qsnr.csv
│   └── ...
└── figures/
    ├── qsnr_comparison.png
    ├── histogram_overlay.png
    ├── per_layer_role_histogram.png
    ├── pareto_qsnr.png
    └── ...
```

## 图表前提条件速查

每个图表依赖特定的 observer 和前置步骤。不满足时抛出 `ValueError` 并说明缺少的 observer。

### SessionResult.plot 方法

| 方法 | 必要的 `outputs` key | 额外条件 |
|------|---------------------|---------|
| `qsnr_comparison()` | `"qsnr"`（默认） | — |
| `crest_vs_qsnr(roles)` | `"distribution"` + `"qsnr"` | — |
| `outlier_analysis(roles)` | `"distribution"`（推荐 `"qsnr"`） | — |
| `per_block_qsnr(roles)` | `"qsnr"` | per_block 粒度 |
| `correlation_heatmap()` | `"distribution"` + `"qsnr"` | — |
| `cost_decomposition()` | — | cost 已执行 |
| `role_distribution_comparison()` | `"distribution"` | — |
| `kurtosis_analysis(roles)` | `"distribution"` + `"qsnr"` | — |
| `histogram_overlay(top_k, role, layer, op_types, qsnr_type)` | `"histogram"` + `"qsnr"` | QSNR 可选，缺失时按幅度排序 |
| `layer_histogram(layer, role)` | `"histogram"` | — |
| `channel_heterogeneity(layer, role)` | `"qsnr"` | per-channel 模式 |
| `per_layer_role_histogram(k)` | `"histogram"` 或 `"distribution"` | — |
| `error_propagation(role)` | `"qsnr"`（默认） | `keep_fp32=True`（默认） |
| `accumulated_vs_local(role)` | `"qsnr"`（默认） | `keep_fp32=True`（默认） |
| `propagation_dag()` | `"qsnr"`（默认） | `keep_fp32=True`（默认） |
| `error_waterfall()` | `"qsnr"`（默认） | `keep_fp32=True`（默认） |
| `local_vs_accum_scatter()` | `"qsnr"`（默认） | `keep_fp32=True`（默认） |
| `per_role_qsnr_bars()` | `"qsnr"` | — |
| `depth_decay(role)` | `"qsnr"` | — |
| `per_layer_role_qsnr_line()` | `"qsnr"` | — |

### StudyReport.plot 方法

| 方法 | 必要的 `outputs` key | 额外条件 |
|------|---------------------|---------|
| `qsnr_comparison()` | `"qsnr"`（默认） | — |
| `crest_vs_qsnr(roles)` | `"distribution"` + `"qsnr"` | — |
| `outlier_analysis(roles)` | `"distribution"`（推荐 `"qsnr"`） | — |
| `per_block_qsnr(roles)` | `"qsnr"` | per_block 粒度 |
| `pareto_frontier(metric)` | — | cost 已执行；`metric="accuracy"` 还需 `eval_fn` |
| `correlation_heatmap()` | `"distribution"` + `"qsnr"` | — |
| `cost_decomposition()` | — | cost 已执行 |
| `role_distribution_comparison()` | `"distribution"` | — |
| `kurtosis_analysis(roles)` | `"distribution"` + `"qsnr"` | — |
| `histogram_overlay(top_k, role, layer, op_types, qsnr_type)` | `"histogram"` + `"qsnr"` | QSNR 可选，缺失时按幅度排序 |
| `per_layer_role_histogram(k)` | `"histogram"` 或 `"distribution"` | — |
| `per_layer_role_qsnr_line(role, qsnr_type)` | `"qsnr"` | — |
| `error_propagation(role)` | `"qsnr"`（默认） | `keep_fp32=True`（默认） |
| `accumulated_vs_local(role)` | `"qsnr"`（默认） | `keep_fp32=True`（默认） |

---

# 图表一览

## 质量评估（核心）

### QSNR 逐层对比 `qsnr_comparison()`

每个 config 一条线，横轴为层名，纵轴为 QSNR (dB)：

```python
fig = report.plot.qsnr_comparison()
fig.savefig("qsnr_comparison.png", dpi=150)
```

**所需 observer**: `"qsnr"`（默认开启）

---

### Crest Factor vs QSNR 散点 `crest_vs_qsnr(roles)`

Crest Factor（峰值 / RMS）越大，量化越困难。每个 role 一个子图：

```python
fig = result.plot.crest_vs_qsnr()                          # 默认 input/weight/output 三面板
fig = result.plot.crest_vs_qsnr(roles=("input", "weight"))  # 只选 input + weight
```

`roles` 默认 `("input", "weight", "output")`，可选值含 `"bias"`。

**所需 observer**: `"distribution"` + `"qsnr"`

---

## 误差传播分析

### 误差传播面板 `error_propagation(role)`

三行面板图，逐层对比累积误差与本地误差：

```python
fig = result.plot.error_propagation(role="output")
```

- **Row 1**：分组柱状图 — Accumulated QSNR（深色）vs Local QSNR（浅色）
- **Row 2**：δ-QSNR 柱状图 — 逐层累积 QSNR 下降量，按阈值着色
- **Row 3**：Headroom 柱状图 — local − accumulated，按诊断阈值着色（绿=Source / 橙=Mixed / 红=Propagated）

**所需 observer**: `"qsnr"`（默认启用）+ `keep_fp32=True`（默认）

---

### Accumulated vs Local 散点 `accumulated_vs_local(role)`

```python
fig = result.plot.accumulated_vs_local(role="output")
```

- X 轴：Accumulated QSNR（hook，累积误差）
- Y 轴：Local QSNR（observer，本地误差）
- y=x 对角线：线上 = Source（本地误差主导），高于线 = Propagated（累积误差主导）
- 自动标注 outlier（headroom > 15 dB 或 < 3 dB）

**所需 observer**: `"qsnr"`（默认启用）+ `keep_fp32=True`（默认）

---

### 误差传播 DAG `propagation_dag()`

水平柱状图：每层 local QSNR + accum 标记点。

```python
fig = result.plot.propagation_dag()
fig = result.plot.propagation_dag(qsnr_cap=60.0, skip_activations=True)
```

---

### 误差瀑布 `error_waterfall()`

瀑布图：累积 QSNR 逐层下降量。

```python
fig = result.plot.error_waterfall(qsnr_cap=60.0)
```

---

### Local vs Accum 散点 `local_vs_accum_scatter()`

散点图：local vs accumulated QSNR，按 headroom 着色。

```python
fig = result.plot.local_vs_accum_scatter()
```

---

## 分布特征分析

### Outlier 分析 `outlier_analysis(roles)`

双面板图表（每个 role 一行）：左为逐层 outlier ratio 柱状图，右为 outlier ratio vs QSNR 散点图：

```python
fig = result.plot.outlier_analysis()                          # 默认三 role
fig = result.plot.outlier_analysis(roles=("input", "weight"))  # 只选两个 role
```

`roles` 默认 `("input", "weight", "output")`。

**所需 observer**: `"distribution"` + `"qsnr"`

---

### 逐 Block QSNR 统计 `per_block_qsnr(roles)`

双面板图表（每个 role 一行）：左为逐层 QSNR 标准差箱线图，右为 QSNR 最小值 vs 平均值散点图。仅在 per_block 粒度下产生有意义数据：

```python
fig = result.plot.per_block_qsnr()  # 默认三 role
```

`roles` 默认 `("input", "weight", "output")`。

**所需 observer**: `"qsnr"`（per_block 模式下自动采集 `qsnr_db_std/min/max`）

---

### 分布特征 × QSNR 相关性热力图 `correlation_heatmap()`

9 种分布特征（crest_factor、skewness、kurtosis、sparse_ratio、outlier_ratio 等）与 QSNR/MSE 的 Pearson 相关系数矩阵：

```python
fig = result.plot.correlation_heatmap()
```

**所需 observer**: `"distribution"` + `"qsnr"` + `"mse"`（至少 2 种特征）

---

### 跨角色分布对比 `role_distribution_comparison()`

1×3 面板箱线图，对比 input / weight / output 三个角色的 skewness、kurtosis、normalized entropy：

```python
fig = result.plot.role_distribution_comparison()
```

**所需 observer**: `"distribution"`（需 `skewness`、`kurtosis`、`norm_entropy`）

---

### 峰度分析 `kurtosis_analysis(roles)`

三面板图表：
1. 峰度值直方图 + 参考线（normal=3 / heavy-tailed=6 / extreme=10）
2. 峰度 vs QSNR 散点，按 role 着色
3. Top-15 (layer, role) 按峰度排名的水平柱状图

```python
fig = result.plot.kurtosis_analysis()
fig = result.plot.kurtosis_analysis(roles=("input", "weight"))
```

`roles` 默认 `("input", "weight", "output")`。

**所需 observer**: `"distribution"` + `"qsnr"`

**解读**：
- 高 QSNR + 高峰度 → 可以安全地激进量化
- 低 QSNR + 高峰度 → 重尾是根因 → 尝试 SmoothQuant
- 低 QSNR + 正常峰度 → 根因在其他地方 → 检查 outlier ratio

---

## 直方图与分布可视化

### 直方图叠加 `histogram_overlay(top_k, role, layer, op_types, qsnr_type)`

三通道半透明叠加直方图（fp32 蓝色填充 / quant 红色虚线 / error 灰色）。两种模式：

**Top-K 模式**（默认，`layer=None`）：展示 QSNR 最低（最敏感）的 `top_k` 个 `(layer, role)` 对：

```python
result = Session(model, cfg).run(
    calib_data, eval_fn=eval_fn,
    outputs=["histogram", "qsnr"],
)

# 全部 role 混合，取 top-5
fig = result.plot.histogram_overlay(top_k=5)

# 只看 weight 的直方图
fig = result.plot.histogram_overlay(top_k=5, role="weight")

# 只看 input 的直方图
fig = result.plot.histogram_overlay(top_k=5, role="input")

# 只看 output 的直方图
fig = result.plot.histogram_overlay(top_k=5, role="output")

# 用累积 QSNR 排序（而非默认的 local QSNR）
fig = result.plot.histogram_overlay(top_k=5, qsnr_type="accum")

# 只包含特定算子类型
fig = result.plot.histogram_overlay(top_k=5, op_types=["linear", "conv"])
```

**单层模式**（`layer="fc1"`）：展示指定层的 input / weight / output 三个 role 的直方图，排列为 1×3 面板：

```python
# 单层三 role 可视化
fig = result.plot.histogram_overlay(layer="fc1")

# 同样支持 op_types 和 qsnr_type
fig = result.plot.histogram_overlay(layer="module.0.QuantizedLinear", qsnr_type="accum")
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_k` | `int` | `5` | Top-K 模式下展示的 (layer, role) 对数 |
| `role` | `str` or `None` | `None` | 过滤 role（Top-K 模式）。`None` = 全部 role |
| `layer` | `str` or `None` | `None` | 指定层名 → 进入单层模式（1×3 面板：input/weight/output） |
| `op_types` | `list` or `None` | `None` | 过滤算子类型，如 `["linear", "conv"]` |
| `qsnr_type` | `str` | `"local"` | QSNR 类型：`"local"`（observer）/ `"accum"`（hook）。accum 仅 output role 有数据 |

**所需 observer**: `"histogram"` + `"qsnr"`（QSNR 用于排序，缺失时回退为按幅度排序）

> 此方法在 `SessionResult.plot` 和 `StudyReport.plot` 上均可使用。

---

### 单层单 Role 直方图 `layer_histogram(layer, role, log_y)`

展示单个 (layer, role) 的 fp32 vs quant 直方图叠加 + error 直方图（上下两面板）：

```python
fig = result.plot.layer_histogram("fc1", role="weight")
fig = result.plot.layer_histogram("fc1", role="input", log_y=True)
```

**所需 observer**: `"histogram"`

---

### 逐层 Per-Role 直方图 `per_layer_role_histogram(k)`

展示 QSNR 最差的 k 层 × 三 role 的 fp32 值分布直方图网格：

```python
fig = result.plot.per_layer_role_histogram(k=5)
fig = result.plot.per_layer_role_histogram(k=3, log_y=True)
fig = result.plot.per_layer_role_histogram(k=5, qsnr_type="accum")
fig = result.plot.per_layer_role_histogram(k=5, op_types=["linear"])
```

**所需 observer**: `"histogram"` 或 `"distribution"`（优先直方图，回退为文本摘要）

---

### 通道异质性 `channel_heterogeneity(layer, role)`

展示单个 layer/role 的逐通道 QSNR 分布（箱线图）：

```python
fig = result.plot.channel_heterogeneity("fc1", role="weight")
```

**所需 observer**: `"qsnr"`（需 per-channel 采集模式）

---

## Per-Role QSNR 对比

### 逐层 Per-Role QSNR 柱状图 `per_role_qsnr_bars()`

分组柱状图：每个 layer 显示 input / weight / output 三个 role 的 QSNR：

```python
fig = result.plot.per_role_qsnr_bars()
fig = result.plot.per_role_qsnr_bars(max_layers=20, sort_by="worst")
fig = result.plot.per_role_qsnr_bars(sort_by="depth")
```

- `sort_by="worst"`：按最低 QSNR 的 role 排序
- `sort_by="depth"`：保持模型原始深度顺序

---

### QSNR 深度衰减 `depth_decay(role)`

指定 role 的 QSNR 随深度变化的折线图：

```python
fig = result.plot.depth_decay(role="output")
fig = result.plot.depth_decay(role="input", qsnr_cap=60.0)
```

---

### 逐层 Per-Role QSNR 折线 `per_layer_role_qsnr_line()`

折线图，每个 role 一条线（local 模式）或仅 output 一条线（accum 模式）：

```python
fig = result.plot.per_layer_role_qsnr_line()                    # 三 role 折线（local）
fig = result.plot.per_layer_role_qsnr_line(qsnr_type="accum")    # output 单线（accum）
fig = result.plot.per_layer_role_qsnr_line(op_types=["linear"])  # 只含 linear 层
```

在 `StudyReport.plot` 上，此方法额外支持 `role` 参数用于多 config 对比单 role：

```python
fig = report.plot.per_layer_role_qsnr_line(role="weight", qsnr_type="local")
```

---

## 成本分析

### Pareto 前沿 `pareto_frontier(metric)` <small>仅 StudyReport</small>

质量与成本的权衡散点图，横轴为 bit-width / latency / memory，纵轴为 QSNR 或 Accuracy：

```python
fig = report.plot.pareto_frontier(metric="qsnr")      # QSNR vs bit-width/latency/memory
fig = report.plot.pareto_frontier(metric="accuracy")   # Accuracy vs 成本
```

`metric` 可选值：`"qsnr"`（默认）/ `"accuracy"`。

**所需数据**: cost 模型 + eval（accuracy 模式）

---

### 成本分解 `cost_decomposition()`

每个 config 的 FLOPs 堆叠柱状图（数学计算 / 量化 / 变换）：

```python
fig = result.plot.cost_decomposition()
# 或
fig = report.plot.cost_decomposition()
```

**所需数据**: cost 模型（`needs_cost=True`）

---

## 控制输出的 Observer

通过 `outputs` 参数启用需要的 observer：

```python
# 默认（QSNR + accuracy table）
report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn)

# 全部 observer + 全部图表
report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn, outputs="all")

# 按需组合
report = Study(configs, model=model).run(
    calib_data, eval_fn=eval_fn,
    outputs=["qsnr", "distribution"]  # crest_vs_qsnr + outlier_analysis + correlation_heatmap
)
```

可用的 output key：

| Key | Observer | 产生的图表/表格 |
|-----|----------|-------------|
| `"qsnr"` | QSNRObserver | qsnr_comparison, per_block_qsnr, per_role_qsnr_bars, depth_decay, per_layer_role_qsnr_line |
| `"mse"` | MSEObserver | 增强 correlation_heatmap |
| `"histogram"` | HistogramObserver | histogram_overlay, layer_histogram, per_layer_role_histogram |
| `"distribution"` | DistributionObserver | crest_vs_qsnr, outlier_analysis, correlation_heatmap, role_distribution_comparison, kurtosis_analysis |
| `"fit"` | DistributionFitObserver | distribution_fit table（需 `scipy`） |
| `"accuracy"` | — （无 observer） | accuracy table（需 `eval_fn`） |
| `"cost"` | — （无 observer） | pareto_frontier, cost_decomposition（需 `needs_cost=True`） |
| `"error_propagation"` | `"qsnr"`（默认） | error_propagation, accumulated_vs_local, propagation_dag, error_waterfall, local_vs_accum_scatter |

---

## 自定义分析：导出 DataFrame

`report.to_dataframe()` 将所有结果展开为 tidy DataFrame，可自定义任意分析：

```python
df = report.to_dataframe()
# 列: part, config, format, layer, role, qsnr_db, mse,
#     crest_factor, skewness, kurtosis, outlier_ratio, ...

import seaborn as sns
import matplotlib.pyplot as plt

# 自定义 1：QSNR 分组箱线图
sns.boxplot(data=df, x="config", y="qsnr_db")

# 自定义 2：分布特征散点矩阵
features = ["skewness", "kurtosis", "outlier_ratio", "qsnr_db"]
sns.pairplot(df[features].dropna(), diag_kind="kde")

# 自定义 3：按 role 分面
g = sns.FacetGrid(df, col="role", height=4)
g.map_dataframe(sns.scatterplot, x="crest_factor", y="qsnr_db")
```

---

## 重新加载

```python
from src.report import StudyReport
report = StudyReport.from_file("results/")
# 可以继续调用 report.plot.* 方法（但 observer 数据从 results.json 恢复不全）
```

---
← [上一章：结果查看](result.md) | [Session 文档索引](INDEX.md) | [下一章：误差分析](analysis.md) →
