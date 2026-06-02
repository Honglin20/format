# 多配置对比 (Study)

> 第 7 章 · [Session 文档索引](INDEX.md)

`Study` 对多个 `QuantConfig` 并行运行 Session，产出 `StudyReport` 做聚合对比。
一个 `Study` = N 个独立的 `Session` 运行 + 聚合对比表 + 多配置可视化。

## 基础用法

```python
from src.session import Study, QuantConfig

configs = [
    QuantConfig(name="int8-pc", w_format="int8", w_granularity="per_channel"),
    QuantConfig(name="int4-pc", w_format="int4", w_granularity="per_channel"),
    QuantConfig(name="fp4-mx", w_format="fp4_e2m1", w_granularity="per_block",
                w_block_size=32, a_format="fp4_e2m1", a_granularity="per_block",
                a_block_size=32),
]

report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn)
print(report.summary())
```

`Study` 对每个 config 创建一个独立 Session，model 自动 deepcopy 互不干扰。

## 控制输出

通过 `outputs` 参数控制分析深度：

```python
# 默认输出（accuracy + qsnr）
report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn)

# 全部 observer + 全部图表表格
report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn, outputs="all")
```

可用的 output key 和对应的 observer 依赖：

| Output Key | 依赖的 Observer | 用途 |
|------------|---------------|------|
| `"qsnr"` | QSNRObserver | QSNR 线图、comparison 图 |
| `"mse"` | MSEObserver | MSE 箱线图 |
| `"distribution"` | DistributionObserver | crest_factor / skewness / kurtosis / outlier_ratio / ... |
| `"histogram"` | HistogramObserver | fp32 / quant / error 三通道直方图 |
| `"fit"` | DistributionFitObserver | 参数化分布拟合（需 scipy） |
| `"accuracy"` | eval_fn | 精度对比表 |
| `"cost"` | cost model | Pareto 前沿、成本分解 |

> **自定义组合**: `outputs=["qsnr", "distribution"]` — 只启用指定 observer，其余跳过。

## 终端输出

```python
report.print_summary()

# 输出：
# ======================================================================
#   Part: int8
# ======================================================================
#   Config    Avg QSNR     Avg MSE      Acc Delta
#   ----------------------------------------------------------------
#   int8      34.21        0.001200     loss=-0.0300
#   int4      22.50        0.005000     loss=-0.1000
#   ...
```

## 保存 & 重新加载

```python
# 保存：生成 tables/ + figures/ + results.json
report.save("results/my_study/")

# 从保存的结果重新加载
from src.report import StudyReport
report = StudyReport.from_file("results/my_study/")
```

产出文件结构（根据启用的 observer 有选择地生成）：

```
results/my_study/
├── results.json
├── tables/
│   ├── accuracy.csv                      # accuracy / avg_qsnr / avg_mse（需 eval_fn）
│   └── per_layer_qsnr.csv               # 逐层逐配置 QSNR（需 qsnr observer）
└── figures/
    ├── qsnr_comparison.png               # 多配置 QSNR 覆盖线图
    ├── crest_vs_qsnr_input.png           # crest factor vs QSNR 散点（需 distribution）
    ├── crest_vs_qsnr_weight.png
    ├── crest_vs_qsnr_output.png
    ├── outlier_input.png                 # outlier 分析（需 distribution）
    ├── outlier_weight.png
    ├── outlier_output.png
    ├── per_block_qsnr_input.png          # per-block QSNR 统计（需 per_block 粒度）
    ├── per_block_qsnr_weight.png
    ├── per_block_qsnr_output.png
    ├── correlation_heatmap.png           # 分布特征 × 误差相关矩阵（需 distribution）
    ├── role_distribution.png             # 各 role 分布特征对比（需 distribution）
    ├── pareto_qsnr.png                   # 品质 vs 成本 Pareto 前沿（需 cost）
    ├── pareto_accuracy.png               # 精度 vs 成本 Pareto 前沿（需 cost + eval_fn）
    └── cost_decomposition.png            # FLOPs 分解（需 cost）
```

## 导出 DataFrame

`report.to_dataframe()` 返回 tidy DataFrame — 每行 `(part, config, format, layer, role)`：

```python
df = report.to_dataframe()

# 每个配置的平均 QSNR
print(df.groupby("config")["qsnr_db"].mean())

# 找出质量差的层
print(df[df["qsnr_db"] < 20][["layer", "role", "qsnr_db"]])

# 比较不同格式在某层的 QSNR
print(df[df["layer"] == "module.0.linear"]["qsnr_db"])

# 按 role 分组看 QSNR 差异
print(df.groupby(["config", "role"])["qsnr_db"].mean().unstack())
```

## 可视化

### 单 Study 图表

`report.plot` 提供 post-hoc 可视化方法：

```python
report.plot.qsnr_comparison()       # 所有 config 的 QSNR 覆盖线图
report.plot.crest_vs_qsnr("input")  # crest factor vs QSNR 散点
report.plot.outlier_analysis("weight")  # outlier 分析
report.plot.per_block_qsnr("input") # per-block QSNR 统计
report.plot.correlation_heatmap()    # 分布特征 x 误差相关矩阵
report.plot.pareto_frontier()        # 品质 vs 成本 Pareto 前沿
report.plot.cost_decomposition()     # FLOPs 分解
```

### 通过注册表使用

`src/viz/figures.py` 的独立函数可以脱离 Study 直接使用：

```python
from src.viz.figures import qsnr_line_chart

fig = qsnr_line_chart(
    results,                       # {config_name: {"qsnr_per_layer": {...}}}
    title="8-bit Format QSNR Comparison",
    colors={"int8": "#3498db", "fp8_e4m3": "#e74c3c"},
    output_dir="results/",
)
```

## 理解 QSNR 对比结果

### 为什么同 bit-width 格式的 QSNR 线图看起来很接近？

QSNR per layer 是对层内所有 tensor role（input/weight/output）取 **最小值** 的聚合结果。
同一层的 bottleneck tensor 角色通常不受 format 影响（例如某层的 input activation 本身就难量化），
所以不同 format 的 QSNR 曲线会呈现**相似的形状**，区别主要在**整体水平**上偏移。

要看到格式之间的差异：
1. 从表格而非线图中对比 — 数值比视觉更精确（见下方 per-layer QSNR 表）
2. 按 role 拆分 — `df.groupby(["config", "role"])["qsnr_db"].mean()`
3. 用 `report.plot.crest_vs_qsnr()` 散点图 — 分布特征 + QSNR 的二维关系
4. 观察 `pareto_frontier()` — QSNR vs 位宽/延迟/内存 的 trade-off

## 对比表格

### Per-Layer QSNR 对比表

显示同一模型下，不同配置在同一层的 QSNR 数值，通过 `report.tables` 访问器调用：

```python
# 终端打印逐层 QSNR 对比表
print(report.tables.per_layer_qsnr())

# 限制显示层数（默认 60）
print(report.tables.per_layer_qsnr(max_layers=10))
```

输出示例：

```
====================================
Per-Layer QSNR (dB) — Lower = more quantization-sensitive
====================================
Layer      int8-pc    int4-pc    fp4-mx     nf4
-------------------------------------------------
0.linear   34.2       22.5       20.1       21.3
1.norm     45.1       38.7       36.2        -
2.linear   32.8       19.3       18.7       20.1
...
```

> 只量化权重的格式（如 NF4 weight-only）对 activation 不量化，对应列显示 `-`。

`report.save()` 会自动生成 `tables/per_layer_qsnr.csv`。

### Accuracy 对比表

自动由 `report.save()` 生成 `tables/accuracy.csv`，包含 avg QSNR 和 avg MSE：

| Config | Accuracy | Avg QSNR (dB) | Avg MSE |
|--------|----------|---------------|---------|
| int8-pc | 0.9300 | 34.21 | 0.001200 |
| int4-pc | 0.8700 | 22.50 | 0.005000 |

### 变换矩阵表

`tables/table4_format_x_transform.csv` — 格式 × 变换的 accuracy 矩阵。

## 自定义配置组合

### 不同模型

```python
study = Study(configs, model=model).run(
    calib_data, eval_fn=eval_fn,
    model_factory=lambda cfg: build_model(cfg)  # 按 config 构建不同模型
)
```

### 变换比较

```python
configs = [
    # 同一格式 + 不同变换
    QuantConfig(name="int8-none", w_format="int8", w_granularity="per_block",
                w_block_size=32, transform="none"),
    QuantConfig(name="int8-hadamard", w_format="int8", w_granularity="per_block",
                w_block_size=32, transform="hadamard"),
    QuantConfig(name="int8-smoothquant", w_format="int8", w_granularity="per_block",
                w_block_size=32, transform="smoothquant"),
]

report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn,
                                          outputs="all")
# 自动生成 transform_heatmap / transform_pie / transform_delta 图表
# 自动生成 transform_matrix / transform_distribution / transform_benefit 表格
```

## 预定义实验

`src/session/study_config.py` 提供 6 组预定义实验，可直接运行：

| Part | 描述 |
|------|------|
| `part_a` | 8-bit 格式对比 (MXINT-8 / MXFP-8 / INT8-PC) |
| `part_b` | 4-bit 格式对比 (MXINT-4 / MXFP-4 / INT4-PC / NF4-PC) |
| `part_c` | FP32 vs PoT scale 对比 |
| `part_d` | Transform 研究 (None / Hadamard / SmoothQuant × 4 格式) |
| `block_sweep` | Block size 敏感度 (16/32/64/128) |
| `part_hierarchical` | 两级量化 (PoT pre-scale + MX per-block) |

使用方式见 `examples/format_study_random.py`。

---
← [上一章：误差分析](analysis.md) | [Session 文档索引](INDEX.md) | [下一章：ONNX 导出](export.md) →
