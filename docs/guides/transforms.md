# Transform

Transform 在量化前/后对数据做预处理，主要用于处理 outlier 降低量化误差。

## 五种 Transform

| Transform | 原理 | 适用场景 |
|-----------|------|---------|
| `"none"` | 不做处理 | 默认，大多数情况 |
| `"hadamard"` | Hadamard 正交旋转，O(n log n) | 激活有聚集 outlier |
| `"smoothquant"` | 激活平滑因子迁移到权重 | LLM 激活 outlier 严重时 |
| `"prescale"` | 前置可学习 scale + LSQ 优化 | 需要极致精度 |
| `"adaptive"` | 逐层自动选择最优变换 | 不确定用哪个时 |

## 基础用法

```python
# 无 transform
cfg = QuantConfig(w_format="int8")

# Hadamard 正交旋转
cfg = QuantConfig(w_format="int8", transform="hadamard")

# SmoothQuant
cfg = QuantConfig(w_format="int8", transform="smoothquant")
```

## SmoothQuant

SmoothQuant 用校准数据计算平滑因子，将激活 outlier 迁移到权重：

```python
cfg = QuantConfig(w_format="int8", transform="smoothquant")
session = Session(model, cfg)

# calib_data 必须传入（计算平滑因子用）
session.quantize(calib_data=calib_data)
```

`smoothquant` 只在 `weight_only=False` 时生效。

`sq_alpha` 控制平滑强度（默认 0.5）：
- 更接近 0：更多平滑压力给激活
- 更接近 1：更多平滑压力给权重

## Prescale + LSQ

```python
cfg = QuantConfig(
    w_format="int4",
    transform="prescale",
    lsq_steps=100,
    lsq_lr=1e-3,
    prescale_init="amax",
)

# calib_data 必须传入（初始化 pre_scale 用）
session = Session(model, cfg).quantize(calib_data=calib_data)
```

见 [LSQ 可学习量化](../advanced/lsq.md)。

## Adaptive：逐层自动选择

```python
cfg = QuantConfig(w_format="int8", transform="adaptive")
session = Session(model, cfg)

# calibrate() 阶段自动做一次前向，对每层评估 none/hadamard/smoothquant
# 的 QSNR，自动选择最优方案
session.quantize(calib_data=calib_data)
session.calibrate(calib_data)

# 查看逐层选择结果
print(session._adaptive_selection)
# {"none": 5, "hadamard": 3, "smoothquant": 2}
```

见 [自适应 Transform](../advanced/adaptive-transform.md)。
