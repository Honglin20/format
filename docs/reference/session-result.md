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

## eval_fn 合约

`accuracy_table()`、`summary()` 的准确率列、`pareto_frontier(metric="accuracy")` 依赖 `eval_fn`。

```python
def eval_fn(model, data) -> dict[str, float]:
    """在 data 上运行 model，返回 {指标名: 浮点值}。

    model — 当前推理模式下的模型（fp32 或 quant）。
    data — eval_data（未单独传入时用 calib_data）。
    """
    ...
    return {"loss": 0.1234}           # 单指标
    # 或
    return {"loss": 0.12, "acc": 0.95}  # 多指标
```

**约束**：
- 返回值必须是 `Dict[str, float]`。
- 不传 `eval_fn` 时 `evaluate()` 被跳过 → `fp32_metrics` / `quant_metrics` / `delta` 全为 `None`。
- `accuracy_table()` 在 `fp32_metrics` 为 `None` 时返回 `"(no accuracy metrics — run with eval_fn)"`。

## 方法

### summary()

```python
>>> print(result.summary())
Config: int8 | loss: fp32=0.1234 quant=0.1456 | avg QSNR=34.2 dB | Δloss=+0.0222
```

不传 `eval_fn` 时跳过准确率部分：

```python
>>> print(result.summary())
Config: int8 | avg QSNR=34.2 dB
```

### accuracy_table()

**需要** `run()` 时传入 `eval_fn`。

```python
>>> print(result.accuracy_table())
Metric    FP32      Quant     Δ
--------------------------------
loss      0.1234    0.1456    +0.0222
acc       0.9500    0.9300    -0.0200
```

不传 `eval_fn` 时：

```python
>>> print(result.accuracy_table())
(no accuracy metrics — run with eval_fn)
```

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

**需要** `pandas`。未安装时返回 `None`。

## 功能前提条件

| 方法 | 前提条件 | 不满足时的行为 |
|------|---------|-------------|
| `summary()` 准确率列 | `eval_fn` | 跳过准确率，只显示 QSNR |
| `summary()` QSNR | QSNRObserver（默认开启） | `avg QSNR=N/A` |
| `accuracy_table()` | `eval_fn` | 返回 `"(no accuracy metrics — run with eval_fn)"` |
| `top_k_qsnr()` | QSNRObserver | 返回空列表 |
| `layer_report()` | pandas 已安装 | 返回 `None` |

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
