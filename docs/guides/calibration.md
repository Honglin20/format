# 校准策略

校准决定每个量化层的 scale。四种策略通过 `QuantConfig.calibrator` 选择。

## 四种策略

| 策略 | 配置值 | 原理 | 适用场景 |
|------|--------|------|---------|
| MSE | `"mse"` | 网格搜索最小化 MSE（默认） | 通用，鲁棒 |
| Max | `"max"` | scale = max(|x|) | 快速、保守 |
| Percentile | `"percentile"` | scale = N 分位数（q=99） | 有 outlier 但不丢太多信息 |
| KL | `"kl"` | 最小化 KL 散度 | 分布敏感任务（分类） |

## 用法

```python
from src.session import Session, QuantConfig

# 默认 MSE
cfg = QuantConfig(w_format="int4")
# 等效于 cfg = QuantConfig(w_format="int4", calibrator="mse")

# 快速原型
cfg = QuantConfig(w_format="int4", calibrator="max")

# 有 outlier 的场景
cfg = QuantConfig(w_format="int4", calibrator="percentile")

# 分布敏感
cfg = QuantConfig(w_format="int4", calibrator="kl")
```

## 比较不同校准策略

```python
for cal in ["mse", "max", "percentile", "kl"]:
    cfg = QuantConfig(name=cal, w_format="int4", calibrator=cal)
    result = Session(model, cfg).run(calib_data)
    print(f"{cal}: {result.summary()}")
```

## MX per_block 自动跳过

MX per_block 格式的 scale 在推理时动态计算，`calibrate()` 会自动检测并跳过校准步骤。
