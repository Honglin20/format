# SessionResult API 参考

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `.name` | `str` | 配置名 |
| `.config` | `QuantConfig` | 原始配置 |
| `.qsnr_per_layer` | `Dict[str, float]` | `{层名: QSNR dB}` |
| `.mse_per_layer` | `Dict[str, float]` | `{层名: MSE}` |
| `.fp32_metrics` | `Dict[str, float]\|None` | fp32 模型指标 |
| `.quant_metrics` | `Dict[str, float]\|None` | 量化模型指标 |
| `.delta` | `Dict[str, float]\|None` | 精度损失（fp32 - quant） |
| `.observers_data` | `dict` | 原始 observer 数据 |
| `.cost` | `CostResult\|None` | 量化模型延迟 & 内存 |
| `.cost_fp32` | `CostResult\|None` | fp32 模型延迟 & 内存 |

## 方法

### summary()

```python
>>> print(result.summary())
Config: int8 | loss: fp32=0.1234 quant=0.1456 | avg QSNR=34.2 dB | Δloss=+0.0222
```

### accuracy_table()

```python
>>> print(result.accuracy_table())
Metric    FP32      Quant     Δ
--------------------------------
loss      0.1234    0.1456    +0.0222
acc       0.9500    0.9300    -0.0200
```

需要 run() 时传入 eval_fn。

### top_k_qsnr(k, reverse=False)

```python
# QSNR 最差的 3 层
result.top_k_qsnr(3)
# → [("layer1.linear", 12.3), ("layer2.conv", 18.7), ...]

# QSNR 最好的 3 层
result.top_k_qsnr(3, reverse=True)
```

### layer_report()

```python
df = result.layer_report()           # → pandas DataFrame
df.sort_values("qsnr_db").head(5)    # QSNR 最差的 5 层
```

需要 `pandas`。

## observers_data 结构

```python
{
    "layer_name": {           # 层名
        "input": {            # tensor 角色 (input/weight/output/grad_input/...)
            "post_quantize": {  # 阶段 (post_transform/pre_quantize/post_quantize)
                "": {           # slice key (空字符串 = 整个张量)
                    "qsnr_db": 34.2,
                    "mse": 0.0012,
                    ...
                }
            }
        }
    }
}
```
