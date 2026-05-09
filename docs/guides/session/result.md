# SessionResult & 结果查看

`SessionResult` 是 Session 的输出，包含精度对比、逐层误差、原始 observer 数据。

## 快速查看

```python
result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)

# 一行摘要
print(result.summary())
# Config: int8 | loss: fp32=0.1234 quant=0.1456 | avg QSNR=34.2 dB | Δloss=+0.0222

# 精度对比表
print(result.accuracy_table())
# Metric    FP32      Quant     Δ
# --------------------------------
# loss      0.1234    0.1456    +0.0222
# acc       0.9500    0.9300    -0.0200
```

## 逐层误差

```python
# QSNR 最差的 3 层（定位问题层）
for name, qsnr in result.top_k_qsnr(3):
    print(f"  {name}: {qsnr:.1f} dB")

# QSNR 最好的 3 层
for name, qsnr in result.top_k_qsnr(3, reverse=True):
    print(f"  {name}: {qsnr:.1f} dB")

# 逐层 DataFrame
df = result.layer_report()
print(df.sort_values("qsnr_db").head(5))
```

## 完整属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `result.name` | `str` | 配置名 |
| `result.config` | `QuantConfig` | 原始配置对象 |
| `result.qsnr_per_layer` | `Dict[str, float]` | `{层名: QSNR dB}` |
| `result.mse_per_layer` | `Dict[str, float]` | `{层名: MSE}` |
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
| `.accuracy_table()` | `str` | FP32 vs Quant 对比表 |
| `.top_k_qsnr(k, reverse=False)` | `List[Tuple]` | QSNR 最差/最好的 k 层 |
| `.layer_report()` | `DataFrame` | 逐层 QSNR + MSE（需 pandas） |

## 获取 observer 原始数据

```python
# observers_data 结构: {layer: {role: {stage: {slice_key: metrics}}}
for layer, roles in result.observers_data.items():
    for role, stages in roles.items():
        for stage, slices in stages.items():
            for slice_key, metrics in slices.items():
                print(f"{layer}/{role}/{stage}/{slice_key}: {metrics}")
```
