# 粒度配置

粒度决定量化 scale 的精细程度——粒越细，精度越高，但 scale 开销越大。

## 三种粒度

```python
# per_tensor：整个张量共享一个 scale（最省，最粗糙）
cfg = QuantConfig(w_format="int8", w_granularity="per_tensor")

# per_channel：每个通道一个 scale（CNN/Linear 权重标准做法）
cfg = QuantConfig(w_format="int8", w_granularity="per_channel")

# per_block：每 N 个元素共享一个指数（MX 规格）
cfg = QuantConfig(
    w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32,
    a_format="fp4_e2m1", a_granularity="per_block", a_block_size=32,
)
```

## 粒度选择建议

| 场景 | 推荐粒度 |
|------|---------|
| 原型验证、最低风险 | per_tensor |
| 权重部署（CNN/Transformer） | per_channel |
| 激活量化、MX 规格 | per_block(32) |

## MX per_block

MX per_block 格式的 scale（shared exponent）在推理时动态计算，因此 `calibrate()` 会自动跳过。

可通过 `w_axis` / `a_axis` 控制量化轴（默认 -1，即最后一维）。

## Outlier Bank

`GranularitySpec` 支持 `outlier_ratio` 参数（∈ [0, 1]），将 block 内的 outlier 拆分单独量化：

```python
# 内部由 GranularitySpec 控制，由 QuantConfig 暴露时通过特定需求使用
from src.scheme.granularity import GranularitySpec

spec = GranularitySpec.per_block(size=32)
# outlier_ratio=0.05: 将 5% 的最大值作为 outlier 单独处理
```
