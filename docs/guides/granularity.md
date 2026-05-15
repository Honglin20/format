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

> 详见 [ADR-011: Sparse 泛化](../architecture/011-sparse-generalization.md)

## Group Sparse 格式分配（ADR-013）

与上述 per-element sparse 不同，group sparse 在**粒度组**边界上分配格式：某些 channel（或 block、bank）整体用高精度格式，其余用低精度格式。这比 per-element sparse 更结构化、更利于硬件加速。

```python
# 30% channel 用 int8 高精度，其余用 int4
cfg = QuantConfig(w_format="int4", w_granularity="per_channel",
                  group_format="int8", group_ratio=0.3)

# BANK 模式：50% bank 用 fp8
cfg = QuantConfig(w_format="int4", w_granularity="bank", w_block_size=16,
                  group_format="fp8_e4m3", group_ratio=0.5)
```

| 粒度 | group 数 | group_mask 形状 | 行为 |
|------|---------|----------------|------|
| per_tensor | 1 | `()` scalar | 退化：全用 group_format |
| per_channel | C | `(C,)` | 整个 channel 统一 H 或 L |
| per_block | B | block-group shape | 整个 block 统一 H 或 L |
| bank | G | `(num_banks,)` | 整个 bank 统一 H 或 L |

`group_format` 和 `outlier_format` 互斥：同一配置中只能使用一种 sparse 模式。

> 详见 [ADR-013: Group Sparse](../../architecture/013-group-sparse-format-assignment.md)
