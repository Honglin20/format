# QuantConfig 配置

> 第 2 章 · [Session 文档索引](INDEX.md)

`QuantConfig` 是 Session 的配置入口。一个 dataclass，所有字段有默认值，覆盖需要的即可。四个核心维度控制量化行为：

```
format × granularity × transform × calibration
```

## 快速示例

```python
from src.session import QuantConfig

# 默认（INT8 per-tensor，无 transform，MSE 校准）
cfg = QuantConfig()

# 仅改格式
cfg = QuantConfig(w_format="int4", w_granularity="per_channel")

# W/A 全量化 + MX per_block
cfg = QuantConfig(
    w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32,
    a_format="fp4_e2m1", a_granularity="per_block", a_block_size=32,
)
```

传给 Session：

```python
from src.session import Session
result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)
```

## 四大维度

### 格式（format）

控制数值的表示精度。通过 `w_format` / `a_format` 设置。

| 类别 | 示例 | 位宽 |
|------|------|------|
| INT | `"int8"`, `"int4"`, `"int2"` | 8/4/2 |
| FP | `"fp8_e4m3"`, `"fp4_e2m1"`, `"fp6_e3m2"` | 8/4/6 |
| NF | `"nf4"` | 4 |
| BF16 | `"bfloat16"` | 16 |

`a_format=None`（默认）时激活格式与权重格式相同。

→ 详见 [格式选择](../formats.md)

### 粒度（granularity）

控制 scale 的共享范围。通过 `w_granularity` / `a_granularity` 设置。

| 值 | scale 数量 | 适用场景 |
|----|-----------|---------|
| `"per_tensor"` | 1 个/张量 | 默认，最快 |
| `"per_channel"` | N 个/通道 | 权重首选 |
| `"per_block"` | N 个/block | MX 格式，最高精度 |

`per_block` 必须同时指定 `w_block_size` / `a_block_size`（通常 32）。

→ 详见 [粒度配置](../granularity.md)

### Transform

在量化前/后对数据做预处理，主要处理 outlier。通过 `transform` 设置。

| 值 | 原理 | 何时用 |
|----|------|--------|
| `"none"` | 不做处理 | 默认 |
| `"hadamard"` | 正交旋转，O(n log n) | 激活有聚集 outlier |
| `"smoothquant"` | 激活平滑因子迁移到权重 | LLM 激活 outlier 严重 |
| `"prescale"` | 前置可学习 scale | 需要极致精度，配合 LSQ |
| `"adaptive"` | 逐层自动选择 | 不确定用哪个 |

→ 详见 [Transform](../transforms.md) 和 [精度优化方法](optimization.md)

### 校准（calibration）

控制 scale 的计算策略。通过 `calibrator` 设置。

| 值 | 原理 | 适用场景 |
|----|------|---------|
| `"mse"` | 最小化 MSE | 默认，通用 |
| `"max"` | absmax | 极简场景 |
| `"percentile"` | 按分位数截断 | 有 outlier 时 |
| `"kl"` | 最小化 KL 散度 | 分布敏感场景 |

→ 详见 [校准策略](../calibration.md)

## 其他常用字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `name` | `""` | 配置名（显示用） |
| `weight_only` | `False` | 仅量化权重，激活保持 fp32 |
| `quantize_nonlinear` | `True` | False 时跳过 Norm/Activation/Pool 量化 |
| `scale_storage` | `"pot"` | scale 存储格式：`"pot"`（2 的幂）或 `"fp32"` |
| `storage_format` | `None` | 逐元素存储格式，如 `"bfloat16"`、`"fp8_e4m3"` |
| `sq_alpha` | `0.5` | SmoothQuant 平滑强度（0=激活侧重，1=权重侧重） |
| `static_input_scale` | `False` | True = 用校准期计算的 `_input_scale` 做激活量化，推理时 scale 固定不变（而非每 batch 动态计算） |
| `outlier_ratio` | `0.0` | Sparse outlier 比例，∈ [0, 1]。将 top-k 离群点分离到独立 scale 组 |

精度优化相关字段（LSQ、GPTQ、Prescale）见 [第 3 章 · 精度优化方法](optimization.md)。

完整字段表见 [QuantConfig API 参考](../../reference/quant-config.md)。

---
← [上一章：Session 概览](overview.md) | [Session 文档索引](INDEX.md) | [下一章：精度优化方法](optimization.md) →
