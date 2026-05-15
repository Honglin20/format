# QuantConfig 字段参考

`QuantConfig` 是用户面向的量化配置 dataclass，所有字段有默认值，覆盖需要的即可。

## 完整字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `""` | 配置名（显示用） |
| `w_format` | `str` | `"int8"` | 权重格式（注册名或自动解析字符串） |
| `w_granularity` | `str` | `"per_tensor"` | per_tensor / per_channel / per_block |
| `w_block_size` | `int\|None` | `None` | per_block 的 block 大小 |
| `w_axis` | `int` | `-1` | 权重量化轴 |
| `a_format` | `str\|None` | `None` | 激活格式（None = 同权重） |
| `a_granularity` | `str` | `"per_tensor"` | 激活粒度 |
| `a_block_size` | `int\|None` | `None` | 激活 per_block 大小 |
| `a_axis` | `int` | `-1` | 激活量化轴 |
| `transform` | `str` | `"none"` | none / hadamard / smoothquant / prescale / adaptive |
| `sq_alpha` | `float` | `0.5` | SmoothQuant 平滑强度 |
| `prescale_init` | `str` | `"ones"` | prescale 初始化：ones / amax / pot_amax |
| `prescale_pot` | `bool` | `False` | prescale 投影到 2 的幂 |
| `prescale_granularity` | `str\|None` | `None` | None = 跟随 a_granularity |
| `lsq_steps` | `int` | `0` | LSQ 优化步数（>0 需 transform="prescale"） |
| `lsq_lr` | `float` | `1e-3` | LSQ 学习率 |
| `scale_storage` | `str` | `"pot"` | scale 存储格式：pot（2 的幂）/ fp32 |
| `calibrator` | `str` | `"mse"` | 校准策略：mse / max / percentile / kl |
| `storage_bits` | `int` | `0` | Element-wise 存储位宽（16=bf16，0=禁用） |
| `storage_kind` | `str` | `"bfloat"` | 存储类型：bfloat / fp |
| `storage_format` | `str\|None` | `None` | 显式格式名："fp8_e4m3"、"fp4_e2m1" 等。优先于 storage_bits/storage_kind |
| `weight_only` | `bool` | `False` | 仅量化权重 |
| `quantize_nonlinear` | `bool` | `True` | False = 非线性算子保持 fp32 |
| `static_input_scale` | `bool` | `False` | True = 用校准期计算的 input_scale 做激活量化（推理时 scale 固定不变，而非每 batch 动态计算） |
| `outlier_ratio` | `float` | `0.0` | Sparse outlier 比例，∈ [0, 1]。>0 时将 top-k 离群点分离到独立 scale 组，适用于 per_tensor / per_channel / per_block |
| `outlier_format` | `str\|None` | `None` | Outlier 组格式（None = 使用主格式） |
| `a_outlier_format` | `str\|None` | `None` | 激活 outlier 格式覆盖（None = 跟随 `outlier_format`） |
| `group_ratio` | `float` | `0.0` | Group sparse H 组占比 ∈ [0, 1]。与 `outlier_ratio` 互斥 |
| `group_format` | `str\|None` | `None` | Group sparse H 格式（如 "int8"）。与 `outlier_format` 互斥 |
| `a_group_ratio` | `float\|None` | `None` | 激活 group_ratio 覆盖（None = 跟随 `group_ratio`） |
| `a_group_format` | `str\|None` | `None` | 激活 group_format 覆盖（None = 跟随 `group_format`） |

## 精度优化字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gptq` | `bool` | `False` | 启用 GPTQ Hessian 引导权重量化 |
| `gptq_block_size` | `int` | `128` | GPTQ 列块大小 |
| `gptq_damp` | `float` | `0.01` | Hessian 对角阻尼分数，范围 (0, 1] |
| `gptq_act_order` | `bool` | `False` | True = 按 Hessian 影响降序量化列 |

## 常用组合

```python
# 默认（INT8 per-tensor）
QuantConfig()

# 仅量化权重
QuantConfig(w_format="int4", weight_only=True)

# W/A 全量化
QuantConfig(w_format="fp8_e4m3", a_format="fp8_e4m3")

# MX per_block + bfloat16 storage
QuantConfig(w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32,
            a_format="fp4_e2m1", a_granularity="per_block", a_block_size=32,
            storage_format="bfloat16")

# LSQ 可学习量化
QuantConfig(w_format="int4", transform="prescale", lsq_steps=100)

# 自适应变换
QuantConfig(w_format="int8", transform="adaptive")

# 非线性算子保持 fp32
QuantConfig(w_format="int8", quantize_nonlinear=False)

# Group sparse: 30% channel 用 int8 高精度，其余用 int4
QuantConfig(w_format="int4", w_granularity="per_channel",
            group_format="int8", group_ratio=0.3)
```

## 从 legacy dict 构造

```python
cfg = QuantConfig.from_descriptor({
    "format": "int4",
    "granularity": "per_channel",
    "transform": "hadamard",
    "calibrator": "mse",
    "group_format": "int8",
    "group_ratio": 0.3,
})
```
