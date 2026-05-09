# 自适应 Transform

`transform="adaptive"` 在 `calibrate()` 阶段自动为每一层选择最优变换（none / hadamard / smoothquant），通过 QSNR 估计做决策。

## 工作原理

1. Hook → 捕获每层 matmul 的输入 activation
2. 对每个候选（none / hadamard / smoothquant）构造 QuantScheme，运行真实 `quantize()`
3. 计算量化后的 matmul 输出 QSNR（dB）
4. 选择 QSNR 最高的候选，patch 回该层的 `OpQuantConfig`
5. SmoothQuant 当选时，额外 fuse 平滑因子到权重

## 用法

```python
from src.session import Session, QuantConfig

cfg = QuantConfig(
    name="adaptive-int8",
    w_format="int8",
    w_granularity="per_channel",
    transform="adaptive",
)

session = Session(model, cfg)

# calibrate() 阶段自动做 transform 选择
session.quantize(calib_data=calib_data)
session.calibrate(calib_data)

# 查看逐层选择结果
print(session._adaptive_selection)
# {"none": 12, "hadamard": 5, "smoothquant": 3}
```

## QSNR 估计

逐层 QSNR 估计考虑 matmul 输出误差，而非逐元素误差。这使得选择更准确：

- **none**: fp32 输出 vs 直接量化的输出
- **hadamard**: fp32 输出 vs Hadamard 旋转 + 量化 + 逆旋转后的输出
- **smoothquant**: fp32 输出 vs SmoothQuant 平滑 + 量化后的输出（权重已提前融合平滑因子）

## 约束

- 仅对 Linear 和 Conv2d 模块做选择（matmul 算子）
- `weight_only=True` 时跳过 smoothquant 候选
- 如果某层无法估计 QSNR，默认 fallback 到 `"none"`
