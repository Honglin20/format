# 精度优化方法

> 第 3 章 · [Session 文档索引](INDEX.md)

量化不可避免引入误差。本章涵盖 7 种精度优化方法，全部通过 `QuantConfig` 配置，由 Session 在 `quantize()` / `calibrate()` 阶段自动执行。

**前提**：已阅读 [第 2 章 · QuantConfig 配置](config.md)，理解 format / granularity / transform / calibration 四大维度。

## 方法速览

| 方法 | 触发方式 | 执行阶段 | 适用场景 |
|------|---------|---------|---------|
| [Prescale](#31-prescale) | `transform="prescale"` | `quantize()` | 静态缩放因子，快速改善激活量化 |
| [LSQ](#32-lsq) | `prescale` + `lsq_steps>0` | `quantize()` | 需要极致精度，梯度优化步长 |
| [GPTQ](#33-gptq) | `gptq=True` | `quantize()` | 大模型权重优化，逐列误差补偿 |
| [Hadamard](#34-hadamard) | `transform="hadamard"` | 量化时内联 | 正交旋转分散 outlier |
| [SmoothQuant](#35-smoothquant) | `transform="smoothquant"` | `quantize()` | 激活 outlier 严重的 LLM |
| [Adaptive](#36-adaptive) | `transform="adaptive"` | `calibrate()` | 不确定用哪个 Transform 时 |
| [per_layer_optimal](#37-per_layer_optimal) | 独立函数 | 后处理 | 多结果逐层最优组合 |

---

## 3.1 Prescale

在量化前对激活插入可学习的缩放因子，静态缩小激活的动态范围，降低量化误差。

### 用法

```python
from src.session import Session, QuantConfig

cfg = QuantConfig(
    w_format="int8",
    transform="prescale",
    prescale_init="amax",        # 初始化策略
    prescale_pot=False,           # 是否约束为 2 的幂
)
result = Session(model, cfg).run(calib_data)
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prescale_init` | `"ones"` | 初始化方式：`"ones"` / `"amax"`（max(\|x\|) 归一化）/ `"pot_amax"`（amax 后投影到 2 的幂） |
| `prescale_pot` | `False` | True = 缩放因子保持 2 的幂（硬件友好） |
| `prescale_granularity` | None | 缩放粒度，默认跟随 `a_granularity` |

### 限制

- 需要 `calib_data`（用于初始化 pre_scale）
- 静态优化，不经过梯度下降（如需梯度优化见 LSQ）

---

## 3.2 LSQ（Learned Step Size Quantization）

通过梯度下降逐层学习最优量化步长。BRECQ 式逐层优化：用已量化的前层跑出当前层输入，梯度优化 pre_scale 最小化量化输出与 fp32 输出的 MSE。

### 用法

```python
cfg = QuantConfig(
    w_format="int4",
    w_granularity="per_channel",
    transform="prescale",        # LSQ 必须配合 prescale
    lsq_steps=100,               # 每层优化步数
    lsq_lr=1e-3,                 # 学习率
    prescale_init="amax",
)
session = Session(model, cfg).quantize(calib_data=calib_data)
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lsq_steps` | `0` | 优化步数。0 = 禁用 LSQ。>0 需要 `transform="prescale"` |
| `lsq_lr` | `1e-3` | 学习率 |

### 约束

- `lsq_steps > 0` 必须配合 `transform="prescale"`（`__post_init__` 强制校验）
- 需要校准数据跑前向，不能纯静态
- 逐层优化，随模型深度线性增时

→ 深入原理见 [LSQ 可学习量化](../../advanced/lsq.md)

---

## 3.3 GPTQ（Hessian 引导权重量化）

使用 Hessian（二阶）信息进行逐列权重量化，在量化每一列后补偿剩余列的误差。显著优于逐列独立量化，适合大模型 weight-only 场景。

### 用法

```python
cfg = QuantConfig(
    w_format="int4",
    w_granularity="per_channel",
    gptq=True,                   # 启用 GPTQ
    gptq_block_size=128,         # 列块大小
    gptq_damp=0.01,              # Hessian 对角阻尼
    gptq_act_order=False,        # 按 Hessian 影响降序量化
)
result = Session(model, cfg).run(calib_data)
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gptq` | `False` | 启用 GPTQ |
| `gptq_block_size` | `128` | 列块大小，影响内存和精度权衡 |
| `gptq_damp` | `0.01` | Hessian 对角线阻尼分数，范围 (0, 1] |
| `gptq_act_order` | `False` | True = 按 Hessian 影响降序量化列 |

### 限制

- **仅支持 `nn.Linear`**（v1 限制）
- 仅支持 `per_channel` / `per_tensor` 粒度，**不支持 per_block**
- 仅优化权重（weight-only 场景最适用）
- 需要 `calib_data`（用于计算 Hessian）

---

## 3.4 Hadamard

Hadamard 正交旋转将激活/权重变换到旋转空间后再量化，利用正交矩阵的范数保持性分散 outlier 能量，降低量化范围。

### 用法

```python
cfg = QuantConfig(w_format="int8", transform="hadamard")
result = Session(model, cfg).run(calib_data)
```

无需额外参数。变换在量化时在线应用：`rotate → quantize → inverse_rotate`。

### 适用场景

- 激活有聚集 outlier 时
- 计算开销 O(n log n)，通常可接受

→ 详见 [Transform 参考](../transforms.md)

---

## 3.5 SmoothQuant

通过逐通道平滑因子将激活的量化难度部分迁移到权重：

```
X' = X · diag(s)^(-1)      激活缩小
W' = diag(s) · W             权重放大
```

`sq_alpha` 控制迁移强度：0 = 全部给激活，1 = 全部给权重（默认 0.5）。

### 用法

```python
cfg = QuantConfig(
    w_format="int8",
    transform="smoothquant",
    sq_alpha=0.5,              # 平滑强度
)
session = Session(model, cfg).quantize(calib_data=calib_data)
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sq_alpha` | `0.5` | 0~1，越小激活承受越多平滑 |

### 限制

- 需要 `calib_data`（计算平滑因子）
- `weight_only=True` 时不生效
- 仅对 Linear / Conv2d 生效

### 分布对比

`analyze()` 时可通过 SmoothQuant 分布对比 observer 查看变换前后每层的动态范围和 outlier 比例变化：

```python
session.analyze(calib_data, outputs=["smoothquant_distrib"])
print(session.result.sq_comparison)
```

→ 详见 [Transform 参考](../transforms.md)

---

## 3.6 Adaptive（逐层自动选择）

对每一层自动评估 `none`、`hadamard`、`smoothquant` 三种 Transform 的 matmul 输出 QSNR，选择最高 QSNR 的方案。

在 `calibrate()` 阶段执行，不修改 `quantize()` 流程。

### 用法

```python
cfg = QuantConfig(w_format="int8", transform="adaptive")
session = Session(model, cfg)
session.quantize(calib_data=calib_data)
session.calibrate(calib_data)

# 查看逐层选择结果
print(session._adaptive_selection)
# {"none": 5, "hadamard": 3, "smoothquant": 2}
```

### 限制

- 仅对 Linear / Conv2d 模块评估
- `weight_only=True` 时跳过 smoothquant 候选
- `calibrate()` 阶段增加一次完整前向的开销

→ 深入机制见 [自适应 Transform](../../advanced/adaptive-transform.md)

---

## 3.7 per_layer_optimal（后处理逐层最优）

独立函数，接受多个 `SessionResult`（如不同 Transform 变体），逐层选择 QSNR 最高的 Transform，构建逐层 `OpQuantConfig` 并重新运行校准。

与 Adaptive 的区别：Adaptive 在 `calibrate()` 内自动完成；per_layer_optimal 是后处理，可从任意来源的结果中择优。

### 用法

```python
from src.session import per_layer_optimal

# 先跑不同 Transform 的 Session
results = []
for transform in ["none", "hadamard", "smoothquant"]:
    cfg = QuantConfig(w_format="int8", transform=transform)
    results.append(Session(model, cfg).run(calib_data))

# 逐层择优
best = per_layer_optimal(
    results, calib_data, fp32_model=model, eval_fn=eval_fn
)
print(best.summary())
```

### 参数

| 参数 | 说明 |
|------|------|
| `part_results` | `List[SessionResult]` — 各变体的结果 |
| `calib_data` | 校准数据 |
| `fp32_model` | 原始 fp32 模型 |
| `eval_fn` | 评估函数 |
| `eval_data` | 可选，评估数据（不传则用 calib_data） |
| `sq_transforms` | 可选，预计算的 SmoothQuantTransform dict |

---
← [上一章：QuantConfig 配置](config.md) | [Session 文档索引](INDEX.md) | [下一章：结果查看](result.md) →
