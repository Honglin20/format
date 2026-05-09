# 误差分析

Session 内置 5 种 Observer，在 `analyze()` 阶段自动挂载，记录每层每个 tensor 角色的量化误差。

## Observer 类型

| Observer | Output Key | 测量内容 |
|----------|-----------|---------|
| `QSNRObserver` | `"qsnr"` | 量化信噪比 `10*log10(||fp32||² / ||fp32-quant||²)`  dB |
| `MSEObserver` | `"mse"` | 均方误差 |
| `HistogramObserver` | `"histogram"` | fp32 / quant / error 三通道直方图 |
| `DistributionObserver` | `"distribution"` | 统计指纹：mean/std/skewness/kurtosis + peak/rms/crest_factor + 稀疏度/动态范围 |
| `DistributionFitObserver` | `"fit"` | 参数化分布拟合（scipy MLE + KS）：best_fit / best_fit_params / best_fit_ks |

`DistributionFitObserver` 需要 `pip install scipy`。

## 可视化输出

Observer 数据通过 `StudyReport` 的图表和表格系统自动可视化。详见 [绘图 & 可视化](plotting.md)。

### 分布拟合分类表

开启 `"fit"` observer 后，`save()` 生成 `tables/table7_distribution_fit.csv`，统计各分布类型（norm / laplace / cauchy / lognorm / ...）在不同 config 下的出现次数：

```python
result = Session(model, cfg).run(calib_data, outputs=["fit", "distribution"])
# save() 自动生成 tables/table7_distribution_fit.csv
```

也可独立调用：

```python
from src.viz.tables import distribution_fit_table

text = distribution_fit_table(all_results, output_dir="results/")
print(text)
```

## 在 Session 中使用

```python
# 默认输出（只启用 QSNR + MSE）
session.analyze(calib_data)

# 全部打开
session.analyze(calib_data, outputs="all")

# 自定义组合
session.analyze(calib_data, outputs=["qsnr", "histogram"])
```

或在 `run()` 中指定：

```python
result = Session(model, cfg).run(calib_data, outputs=["qsnr", "distribution"])
```

**注意**：`evaluate()` 阶段（accuracy 相关输出）需要 `eval_fn`，`cost()` 阶段（pareto / cost_decomposition）需要 `needs_cost=True`。详见 [结果查看](result.md) 的前提条件表。

## 独立使用 Observer

Observer 可以脱离 Session，直接挂载到任意模型：

```python
from src.analysis import QSNRObserver, MSEObserver, AnalysisContext

observers = [QSNRObserver(), MSEObserver()]
with AnalysisContext(model, observers) as ctx:
    for batch in calib_data:
        model(batch)
report = ctx.report()
```

## 分布分析与误差关联

拿到 observer 数据后，可以对分布指纹做聚合分析：

```python
from src.analysis.correlation import (
    DistributionProfile,
    DistributionTaxonomy,
    ErrorByDistribution,
    LayerSensitivity,
)

# 从报吿构建分布画像
profile = DistributionProfile.from_report(ctx.report())
profile.print_profile()

# 自动将层分类到 8 种分布类型
taxonomy = DistributionTaxonomy.from_report(ctx.report())
taxonomy.print_taxonomy()
taxonomy.print_taxonomy(ascii_plots=True)          # ASCII 分布图
exemplars = taxonomy.get_exemplars("heavy-tailed", n=3)

# 找出误差最大的层
eb = ErrorByDistribution(ctx.report())
for layer, role, qsnr in eb.rank_layers(by="qsnr_db", k=5, ascending=True):
    print(f"{layer}/{role}: {qsnr:.1f} dB")

# 按动态范围分组统计
groups = eb.group_by_range(role="input", bins=[0, 4, 7, 999])
for name, info in groups.items():
    print(f"{name}: avg_qsnr={info['avg_qsnr']:.1f} dB, {info['verdict']}")

# 最敏感的层
sens = LayerSensitivity(ctx.report())
for layer, role, mse in sens.topk(k=5, metric="mse"):
    print(f"{layer}/{role}: MSE={mse:.6f}")
```

## 分布类型识别

开启 `"distribution"` 和 `"fit"` 两个 observer 后，可通过 `report.taxonomy` 访问器做分布分类：

```python
result = Session(model, cfg).run(calib_data, outputs=["distribution", "fit"])

# 分类结果
result.report().taxonomy.classify()
# {"norm": {"count": 12, ...}, "heavy-tailed": {"count": 3, ...}, ...}

# 打印分类报告
result.report().taxonomy.print()

# 获取某类的代表层
result.report().taxonomy.exemplars("heavy-tailed", n=3)
```
