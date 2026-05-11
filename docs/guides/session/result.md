# SessionResult & 结果查看

`SessionResult` 是 Session 的输出，包含精度对比、逐层误差、原始 observer 数据。

## eval_fn 合约

多个方法依赖 `eval_fn`。它的签名和返回值必须满足：

```python
def eval_fn(model, data) -> dict[str, float]:
    """在 data 上运行 model，返回 {指标名: 浮点值}。

    model 是当前推理模式下的模型（fp32 或 quant），
    data 是 eval_data（或 calib_data 当 eval_data 未单独传入时）。
    """
    ...
    return {"loss": 0.1234}          # 单指标
    # 或
    return {"loss": 0.12, "acc": 0.95}  # 多指标
```

**约束**：
- 返回值必须是 `Dict[str, float]` —— 字符串 key，浮点 value。
- `accuracy_table()` / `summary()` 的准确率列 / `pareto_frontier(metric="accuracy")` 都依赖这个返回值。
- 不传 `eval_fn` 时 `evaluate()` 阶段被跳过，`fp32_metrics` / `quant_metrics` / `delta` 全部为 `None`。

## 快速查看

```python
result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)

# 一行摘要
print(result.summary())
# Config: int8 | loss: fp32=0.1234 quant=0.1456 | avg QSNR=34.2 dB | Δloss=+0.0222

# 精度对比表（需要 eval_fn）
print(result.accuracy_table())
# Metric    FP32      Quant     Δ
# --------------------------------
# loss      0.1234    0.1456    +0.0222
# acc       0.9500    0.9300    -0.0200
```

不传 `eval_fn` 时的输出：

```python
>>> print(result.accuracy_table())
(no accuracy metrics — run with eval_fn)

>>> print(result.summary())
Config: int8 | avg QSNR=34.2 dB
```

## 逐层误差

```python
# QSNR 最差的 3 层（定位问题层）
for name, qsnr in result.top_k_qsnr(3):
    print(f"  {name}: {qsnr:.1f} dB")

# QSNR 最好的 3 层
for name, qsnr in result.top_k_qsnr(3, reverse=True):
    print(f"  {name}: {qsnr:.1f} dB")

# 逐层 DataFrame（需要 pandas）
df = result.layer_report()
print(df.sort_values("qsnr_db").head(5))
```

## 功能前提条件一览

每个方法/图表依赖特定的前置步骤。未满足时给出可操作的错误提示。

### SessionResult 方法

| 方法 | 前提条件 | 不满足时 |
|------|---------|---------|
| `summary()` — 准确率列 | `eval_fn` 已传入 `run()`/`evaluate()` | 只显示 QSNR，跳过准确率 |
| `summary()` — QSNR 列 | QSNRObserver（`outputs` 默认含 `"qsnr"`） | 显示 `avg QSNR=N/A` |
| `accuracy_table()` | `eval_fn` 已传入 | `(no accuracy metrics — run with eval_fn)` |
| `top_k_qsnr()` | QSNRObserver（默认开启） | 返回空列表 |
| `layer_report()` | QSNRObserver / MSEObserver（默认含 qsnr） | 返回含 NaN 的 DataFrame |

### StudyReport.plot 方法

| 方法 | 必要的 outputs key | 额外条件 |
|------|-------------------|---------|
| `qsnr_comparison()` | `"qsnr"`（默认） | — |
| `crest_vs_qsnr(role)` | `"distribution"` + `"qsnr"` | — |
| `outlier_analysis(role)` | `"distribution"`（推荐 `"qsnr"`） | — |
| `per_block_qsnr(role)` | `"qsnr"` | per_block 粒度（`qsnr_db_std/min/max` 只在 per_block 时采集） |
| `pareto_frontier(metric)` | cost 已执行 | `metric="accuracy"` 还需 `eval_fn` |
| `correlation_heatmap()` | `"distribution"` + `"qsnr"` | — |
| `cost_decomposition()` | cost 已执行 | — |
| `role_distribution_comparison()` | `"distribution"` | — |

### 表格输出

| 输出 key | 必要的 outputs key | 额外条件 |
|---------|-------------------|---------|
| `"accuracy"` | — （needs_eval=True） | `eval_fn` |
| `"distribution_fit"` | `"fit"` | `pip install scipy` |
| `"transform_benefit"` | `"qsnr"` | `eval_fn` |
| `"cost"` | — （needs_cost=True） | 调用 `session.cost()` 或 `outputs` 含 `"cost"` |

### outputs key → observer 映射

| outputs key | 实例化的 observer 类 | 产生的 metrics |
|------------|-------------------|--------------|
| `"qsnr"` | `QSNRObserver` | `qsnr_db`, `qsnr_db_std/min/max`（per_block） |
| `"mse"` | `MSEObserver` | `mse`, `mse_std/min/max`（per_block） |
| `"histogram"` | `HistogramObserver` | `fp32_hist`, `quant_hist`, `err_hist` |
| `"distribution"` | `DistributionObserver` | `crest_factor`, `skewness`, `kurtosis`, `outlier_ratio`, ... |
| `"fit"` | `DistributionFitObserver` | `best_fit`, `best_fit_params`, `best_fit_ks` |

### `outputs` 常用组合

```python
# 默认：精度表 + QSNR 对比图
outputs="default"  # → ["accuracy", "qsnr"]

# 全部：所有 17 种输出
outputs="all"

# 按需组合
outputs=["qsnr"]                                              # 只看 QSNR
outputs=["qsnr", "distribution"]                              # crest_vs_qsnr + outlier
outputs=["qsnr", "distribution", "mse"]                       # + correlation_heatmap
outputs=["qsnr", "accuracy", "cost"]                          # pareto + accuracy table + cost
outputs=["qsnr", "distribution", "fit", "accuracy"]           # + distribution_fit table
```

## 完整属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `result.name` | `str` | 配置名 |
| `result.config` | `QuantConfig` | 原始配置对象 |
| `result.qsnr_per_layer` | `Dict[str, float]` | 本地 QSNR（observer 测量的逐层量化噪声，output role） |
| `result.mse_per_layer` | `Dict[str, float]` | 本地 MSE（observer 测量，output role） |
| `result.accum_qsnr_per_layer` | `Dict[str, float]` | 累积 QSNR（hook 测量，quant vs fp32 参考输出的逐层对比） |
| `result.accum_mse_per_layer` | `Dict[str, float]` | 累积 MSE（hook 测量） |
| `result.fp32_metrics` | `Dict[str, float]` | eval_fn 在 fp32 模型上的输出 |
| `result.quant_metrics` | `Dict[str, float]` | eval_fn 在量化模型上的输出 |
| `result.delta` | `Dict[str, float]` | 精度差（fp32 - quant） |
| `result.observers_data` | `dict` | 原始 observer 数据（供高级分析） |
| `result.cost` | `CostResult` | 量化模型延迟 & 内存估算 |
| `result.cost_fp32` | `CostResult` | fp32 模型延迟 & 内存估算 |

## 方法速查

| 方法 | 返回 | 说明 |
|------|------|------|
| `.summary()` | `str` | 单行摘要 |
| `.accuracy_table()` | `str` | FP32 vs Quant 对比表（需 eval_fn） |
| `.top_k_qsnr(k, reverse=False)` | `List[Tuple]` | QSNR 最差/最好的 k 层 |
| `.layer_report()` | `DataFrame` | 逐层 QSNR + MSE（需 pandas） |
| `.qsnr_per_role(role)` | `Tuple[Dict, Dict]` | 从 observers_data 提取指定 role 的 (qsnr, mse) |
| `.tables` | `SessionTablesAccessor` | 终端表格输出（per_layer_qsnr, error_source_analysis） |

## 获取 observer 原始数据

```python
# observers_data 结构: {layer: {role: {stage: {slice_key: metrics}}}
for layer, roles in result.observers_data.items():
    for role, stages in roles.items():
        for stage, slices in stages.items():
            for slice_key, metrics in slices.items():
                print(f"{layer}/{role}/{stage}/{slice_key}: {metrics}")
```
