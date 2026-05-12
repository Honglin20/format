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

### SmoothQuant 分布前后对比

`analyze(outputs=["smoothquant_distrib"])` 自动对比 SmoothQuant 前后每层
activation 和 weight 的分布变化，输出 per-layer 对比表和关键指标变化：

```python
session = Session(model, cfg)
session.quantize(calib_data=calib_data)
session.analyze(calib_data, outputs=["smoothquant_distrib"])
print(session.result.sq_comparison)
```

输出示例：

```
Layer                            DR raw  DR smooth    Δ DR  Outlier raw  Outlier smooth
-----------------------------------------------------------------------
0                                  8.63       8.09    0.54      0.0000        0.0000
2                                  5.83      22.19  -16.35      0.0410        0.0059
-----------------------------------------------------------------------
Mean DR reduction: -7.91 bits
Mean outlier reduction: 0.0176
```

也可以手动调用：

```python
session.compare_smoothquant_distributions(calib_data, eval_fn=my_eval)
```

对比指标包括：
- `dynamic_range_bits`：动态范围（bits），SmoothQuant 应压缩首层
- `outlier_ratio`：outlier 比例，核心卖点
- `crest_factor`：峰值/RMS 比
- `skewness`：偏度
- 自动按 DR 压缩幅度排名 `improved_layers`

可视化：

```python
from src.viz import smoothquant_distrib_comparison

fig = smoothquant_distrib_comparison(result.sq_distrib_comparison, k=5,
                                     output_dir="./output")
```

见 [绘图 & 可视化](plotting.md)。

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
