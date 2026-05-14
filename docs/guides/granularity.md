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

## Sparse Outlier 隔离

`outlier_ratio`（∈ [0, 1]）将每组内 top-k 离群点分离到独立 scale 组，三种粒度均支持：

```python
# 通过 QuantConfig 设置，所有粒度模式均可用
cfg = QuantConfig(w_format="int8", outlier_ratio=0.05)          # per_tensor + sparse
cfg = QuantConfig(w_format="int4", w_granularity="per_channel",
                  outlier_ratio=0.02)                            # per_channel + sparse
cfg = QuantConfig(w_format="fp4_e2m1", w_granularity="per_block",
                  w_block_size=32, outlier_ratio=0.05)           # per_block + sparse
```

| 模式 | 无 sparse | outlier_ratio=0.05 |
|------|----------|-------------------|
| per_tensor | 1 个 amax | 2 个 amax（outlier + normal） |
| per_channel | C 个 amax | 2C 个 amax |
| per_block | 每 block 1 个 shared exp | 每 block 2 个 shared exp |

> 详见 [ADR-011: Sparse 泛化](../../architecture/011-sparse-generalization.md)
