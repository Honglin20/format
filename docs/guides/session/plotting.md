# 绘图 & 可视化

通过 `Study` 对比多个配置时，`StudyReport.plot` 提供 10 种内置图表。所有方法返回 `matplotlib.Figure`，可 `plt.show()` 或 `fig.savefig()`。

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

## 一键保存

```python
report.save("results/")
```

`save()` 自动检测哪些数据存在，只生成有数据的图表。输出结构：

```
results/
├── results.json
├── tables/
│   └── accuracy.csv
└── figures/
    ├── qsnr_comparison.png
    ├── crest_vs_qsnr_input.png
    ├── crest_vs_qsnr_weight.png
    ├── crest_vs_qsnr_output.png
    ├── outlier_input.png             # P0.1
    ├── per_block_qsnr_input.png      # P0.2
    ├── correlation_heatmap.png       # P1.5
    ├── role_distribution.png         # P1.7
    ├── pareto_qsnr.png               # P0.4
    ├── pareto_accuracy.png           # P0.4
    └── cost_decomposition.png        # P1.6
```

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

### Crest Factor vs QSNR 散点 `crest_vs_qsnr(role)`

Crest Factor（峰值 / RMS）越大，量化越困难。每个点代表一个 `(config, layer)`：

```python
fig = report.plot.crest_vs_qsnr(role="input")
fig = report.plot.crest_vs_qsnr(role="weight")
fig = report.plot.crest_vs_qsnr(role="output")
```

`role` 可选值：`"input"` / `"weight"` / `"output"` / `"bias"`。数据不足时抛出 `ValueError` 并提示缺少的 observer。

**所需 observer**: `"distribution"` + `"qsnr"`

---

## 分布特征分析（P0.1–P1.7）

### Outlier 分析 `outlier_analysis(role)` <small>P0.1</small>

双面板图表：左为逐层 outlier ratio 柱状图，右为 outlier ratio vs QSNR 散点图。outlier 比例越高的层，量化损失通常越大：

```python
fig = report.plot.outlier_analysis(role="input")
fig = report.plot.outlier_analysis(role="weight")
```

**所需 observer**: `"distribution"` + `"qsnr"`

---

### 逐 Block QSNR 统计 `per_block_qsnr(role)` <small>P0.2</small>

双面板图表：左为逐层 QSNR 标准差箱线图（反映 block 间质量波动），右为 QSNR 最小值 vs 平均值散点图。仅在 per_block 粒度下产生有意义数据：

```python
fig = report.plot.per_block_qsnr(role="input")
```

**所需 observer**: `"qsnr"`（per_block 模式下自动采集 `qsnr_db_std/min/max`）

---

### Pareto 前沿 `pareto_frontier(metric)` <small>P0.4</small>

质量与成本的权衡散点图，横轴为 bit-width / latency / memory，纵轴为 QSNR 或 Accuracy。每个 config 一个点：

```python
fig = report.plot.pareto_frontier(metric="qsnr")      # QSNR vs bit-width/latency/memory
fig = report.plot.pareto_frontier(metric="accuracy")   # Accuracy vs 成本
```

`metric` 可选值：`"qsnr"`（默认）/ `"accuracy"`。需要 `cost()` 已执行且 `eval_fn` 已传入。

**所需数据**: cost 模型 + eval（accuracy 模式）

---

### 分布特征 × QSNR 相关性热力图 `correlation_heatmap()` <small>P1.5</small>

8 种分布特征（crest_factor、skewness、kurtosis、sparse_ratio、outlier_ratio 等）与 QSNR/MSE 的 Pearson 相关系数矩阵：

```python
fig = report.plot.correlation_heatmap()
```

颜色越深（红/蓝）相关性越强，可用于识别哪些分布特征最能预测量化误差。

**所需 observer**: `"distribution"` + `"qsnr"` + `"mse"`（至少 2 种特征）

---

### 成本分解 `cost_decomposition()` <small>P1.6</small>

每个 config 的 FLOPs 堆叠柱状图（数学计算 / 量化 / 变换），直观对比量化开销占比：

```python
fig = report.plot.cost_decomposition()
```

**所需数据**: cost 模型（`needs_cost=True`）

---

### 跨角色分布对比 `role_distribution_comparison()` <small>P1.7</small>

1×3 面板箱线图，对比 input / weight / output 三个角色的 skewness、kurtosis、normalized entropy。揭示不同 tensor 角色的分布差异：

```python
fig = report.plot.role_distribution_comparison()
```

**所需 observer**: `"distribution"`（需 `skewness`、`kurtosis`、`norm_entropy`）

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

| Key | Observer | 产生的图表 |
|-----|----------|-----------|
| `"qsnr"` | QSNRObserver | qsnr_comparison, per_block_qsnr |
| `"mse"` | MSEObserver | 增强 correlation_heatmap |
| `"histogram"` | HistogramObserver | histogram_overlay |
| `"distribution"` | DistributionObserver | crest_vs_qsnr, outlier_analysis, correlation_heatmap, role_distribution_comparison |
| `"fit"` | DistributionFitObserver | distribution_fit table |

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
