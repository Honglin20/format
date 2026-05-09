# LSQ 可学习量化

当静态校准不够时，用 LSQ（Learned Step Size Quantization）通过梯度下降学习最优 pre-scale。

## 用法

```python
from src.session import Session, QuantConfig

cfg = QuantConfig(
    w_format="int4",
    w_granularity="per_channel",
    transform="prescale",     # 必须用 prescale
    lsq_steps=100,            # 每层优化步数
    lsq_lr=1e-3,              # 学习率
    prescale_init="amax",     # pre_scale 初始化方式
)

# calib_data 必须传入（初始化 pre_scale + 逐层优化用）
session = Session(model, cfg).quantize(calib_data=calib_data)
```

## 优化方式

LSQ 使用逐层（layer-wise）BRECQ 式优化：

1. 顺序遍历量化模块
2. 用前面的量化层跑出当前层的输入
3. 梯度优化当前层的 pre-scale，最小化量化输出与 fp32 输出的 MSE

## pre_scale 初始化

| init 值 | 方式 |
|---------|------|
| `"ones"` | 全部初始化为 1.0 |
| `"amax"` | scale = max(|x|) 归一化 |
| `"pot_amax"` | amax 后投影到 2 的幂 |

## pre_scale 参数

| 参数 | 说明 |
|------|------|
| `prescale_init` | 初始化方式 |
| `prescale_pot` | True = 优化过程中保持 2 的幂 |
| `prescale_granularity` | pre_scale 的粒度（默认跟随 a_granularity） |

## 约束

- `lsq_steps > 0` 必须配合 `transform="prescale"`
- 需要校准数据跑前向，不能纯静态
