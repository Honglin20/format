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

## 误差传播分析：累积 vs 本地

量化误差有两条独立的测量路径——**累积误差**（hook）和**本地误差**（QSNRObserver）。单独看任一条都只能回答「某层现在有多差」；把两条路径关联起来才能回答「某层差是因为自己还是因为上游」。

### 两条路径

| 路径 | 存储字段 | 测量方式 | 每层度量 |
|------|---------|---------|---------|
| Hook（累积） | `accum_qsnr_per_layer` | 量化模型 forward 时，逐层 hook 抓取输出，与 fp32 参考输出计算 QSNR | 从第一层到当前层的**累积误差** |
| Observer（本地） | `qsnr_per_layer` | 量化算子内部 `_emit()` 事件，比较量化前后 tensor | 仅此层量化引入的**本地误差** |

两条路径在 `"qsnr"` observer 启用时（默认）**同时采集**，分别存入独立字段，不再需要用 `true_error` 参数切换。

**关键不等式**：Observer 覆盖范围 ⊃ Hook 覆盖范围（`hook ⊂ observer`）。Hook 只覆盖 `_MODULE_MAPPING` 中的模块（Linear/Conv/Norm/...），Observer 额外覆盖 patched inline ops（如 `torch.matmul`、`torch.add`），可捕获未被 hook 覆盖的自定义模块。

### 启用累积误差

无需额外参数——当 `"qsnr"` 在 outputs 中时（默认），两条路径自动同时采集：

```python
result = Session(model, cfg).run(
    calib_data,
    # outputs=["qsnr"] 是默认值，无需显式指定
)

# 累积 QSNR（hook 数据）
print(result.accum_qsnr_per_layer)
# {"embedding": 33.8, "layers.0.ffn.2": 27.7, "output": 27.8, ...}

# 本地 QSNR（observer 数据，仅 output role）
print(result.qsnr_per_layer)
# {"embedding": 55.5, "layers.0.self_attn.matmul": 55.4, ...}

# 按其他 role 提取本地 QSNR
local_weight, _ = result.qsnr_per_role(role="weight")
local_input, _ = result.qsnr_per_role(role="input")
```

**实现细节**：
- Hook 只覆盖 `cfg_causes_quantization(cfg) == True` 的模块——空配置/全 None 配置的模块输出与 fp32 位精确等价，不参与对比
- Observer 覆盖所有 `ObservableMixin` 模块（QuantizedLinear 等） + 所有 patched inline ops（`torch.matmul`、`F.linear` 等）
- 两类数据的 key 命名规则不同，匹配时使用前缀规则：`obs_key == hook_key` 或 `obs_key.startswith(hook_key + ".")`

### 关联分析

```python
# 直接从 SessionResult 获取关联数据（推荐）
corr = result.correlate_hook_observer(role="output")
# 返回 {"matched": [...], "observer_only": [...], "hook_only": [...]}

for hk, accum, local in corr["matched"]:
    headroom = local - accum  # 越大 = 越可能是传播
    print(f"{hk}: accum={accum:.1f}  local={local:.1f}  headroom={headroom:+.1f}")

# 多配置对比时，通过 StudyReport 聚合：
from src.report._study_report import StudyReport

report = StudyReport({"my_config": [result]})
corr_all = report.correlate_hook_observer(role="output")
# 返回 {"my_config": {"matched": [...], ...}}
```

### 诊断表

单结果模式下，直接从 SessionResult 输出诊断表：

```python
# 单结果：直接使用 result.tables
print(result.tables.error_source_analysis(role="output"))
```

多配置对比时，使用 StudyReport 的 tables accessor：

```python
print(report.tables.error_source_analysis(role="output"))
```

输出示例（Transformer Encoder）：

```
=========================================================================================================
  Error Source Analysis — int8_bf16_storage [output]
=========================================================================================================
Layer                          Accum QSNR   Local QSNR      Delta   Headroom  Diagnosis
---------------------------------------------------------------------------------------
embedding                           33.79        55.54      +0.00     +21.75  Propagated
layers.0.ffn.2                      27.65        55.16      +3.17     +27.51  Propagated
layers.1.ffn.2                      27.38        55.46      +2.84     +28.07  Propagated
layers.2.ffn.2                      26.99        55.41      +2.62     +28.42  Propagated
layers.3.ffn.2                      26.82        55.63      +2.30     +28.81  Propagated
output                              27.78        55.47      +1.23     +27.69  Propagated
---------------------------------------------------------------------------------------
Summary:                                   drop=+6.0  avg_headroom=+82.8  0 source, 0 mixed, 34 propagated

  Observer-only (no hook data):
    layers.0.self_attn.matmul     local=55.44 dB

  Hook-only (no observer data):
    layers.0.ffn.1                accum=30.48 dB
```

**列含义**：

| 列 | 含义 |
|----|------|
| Accum QSNR | 累积 QSNR（hook，越低越差） |
| Local QSNR | 本地 QSNR（observer，理论上限） |
| Delta | `accum[i-1] − accum[i]`，此层引入的**增量误差**。负值 = 本地恶化；正值 = 实际改善（极少见） |
| Headroom | `local − accum`，此层还有多少精度余量。越高越说明误差来自上游传播 |
| Diagnosis | `Source`（≤3 dB）/ `Mixed`（3–10 dB）/ `Propagated`（>10 dB） |

**Summary 行**：
- `drop`：首层到最后层的总累积 QSNR 下降（总传播损失）
- `avg_headroom`：平均余量
- 三种诊断的计数

### 可视化

```python
# 三行面板图
# Row 1: 分组柱状图（accumulated vs local per layer）
# Row 2: δ-QSNR 柱状图（acc[i-1] - acc[i]），按阈值着色
# Row 3: Headroom 柱状图（local - accumulated），按诊断着色
report.plot.error_propagation(role="output")

# 散点图：X=accumulated, Y=local，每点一个 matched layer
# y=x 对角线：在线上 = Source，高于线 = Propagated
# 标注 outlier（headroom > 15dB 或 < 3dB）
report.plot.accumulated_vs_local(role="output")
```

### 诊断解读

**全部 Propagated 是正常的**——说明本地量化精度（int8 ~55 dB）远高于累积误差（~27-34 dB），误差主要来自深度传播而非本地引入。这意味着：

- 该量化方案的本地精度充足
- 要改善整体精度，应优先减少层数、增大位宽、或在关键层使用更高精度格式
- 如果某层出现 **Source** 诊断，说明它是精度瓶颈，应重点优化该层的量化配置

**Hook-only 层**（如 GELU 激活）说明 observer 未对该模块 emit output 事件——通常是 passthrough 算子或 observer 不支持该算子类型，不影响分析。

**Observer-only 层**（如 attention matmul）说明存在自定义模块调用了 patched inline op（`torch.matmul`），但它们不在 `_MODULE_MAPPING` 中故不被 hook。可在诊断表中单独查看其本地 QSNR。

### 演示脚本

```bash
# 4 层简单模型 + CustomMatMul（验证 hook ⊂ observer）
python scripts/test_error_propagation.py

# 4 层 Transformer Encoder（真实场景）
python scripts/test_transformer_error_propagation.py
```

## 常见错误

以下错误表示需要的 Observer 未启用，解决方案是传递对应的 `outputs` 选项：

| 错误中的关键字 | 缺少的 Observer | 解决方案 |
|--------------|---------------|---------|
| `QSNR data not available` | QSNRObserver | `outputs=["qsnr"]` 或 `outputs="default"` |
| `MSE data not available` | MSEObserver | `outputs=["mse"]` 或 `outputs="default"` |
| `crest_factor` / `Distribution data not available` | DistributionObserver | `outputs=["distribution"]` |
| `Histogram data not available` | HistogramObserver | `outputs=["histogram"]` |
| `Distribution fit data not available` | DistributionFitObserver | `outputs=["fit"]`（需 scipy） |
| `Outlier ratio data not available` | DistributionObserver | `outputs=["distribution"]` |
| `Per-block QSNR statistics not available` | QSNRObserver + per_block 格式 | 使用 `per_block` granularity + `outputs=["qsnr"]` |

> **提示**: 传 `outputs="all"` 可以一次性启用所有 Observer（更慢但不会漏）。

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

拿到 observer 数据后，可以对分布指纹做聚合分析。单结果模式通过 `result.report()` 获取 `AnalysisReport`，独立模式通过 `ctx.report()`：

```python
from src.analysis.correlation import (
    DistributionProfile,
    DistributionTaxonomy,
    ErrorByDistribution,
    LayerSensitivity,
)

# SessionResult 模式（推荐）
report = result.report()

# 独立 AnalysisContext 模式
# report = ctx.report()

# 从报吿构建分布画像
profile = DistributionProfile.from_report(report)
profile.print_profile()

# 自动将层分类到 8 种分布类型
taxonomy = DistributionTaxonomy.from_report(report)
taxonomy.print_taxonomy()
taxonomy.print_taxonomy(ascii_plots=True)          # ASCII 分布图
exemplars = taxonomy.get_exemplars("heavy-tailed", n=3)

# 找出误差最大的层
eb = ErrorByDistribution(report)
for layer, role, qsnr in eb.rank_layers(by="qsnr_db", k=5, ascending=True):
    print(f"{layer}/{role}: {qsnr:.1f} dB")

# 按动态范围分组统计
groups = eb.group_by_range(role="input", bins=[0, 4, 7, 999])
for name, info in groups.items():
    print(f"{name}: avg_qsnr={info['avg_qsnr']:.1f} dB, {info['verdict']}")

# 最敏感的层
sens = LayerSensitivity(report)
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
