# ADR-006: Learnable Pre-Scale — 可学习前置 scale

**状态**: 已决策
**日期**: 2026-04-28

---

## 背景与问题

当前量化管线的 scale 有两种来源，均为非可学习的：

| 来源 | 机制 | 何时计算 |
|---|---|---|
| 现场计算 | `torch.amax(\|x\|)` 或 `_shared_exponents()` | 每次 forward 现场算 |
| 校准持久化 | `CalibrationSession` → 策略（max/percentile/MSE/KL）→ `_output_scale` buffer | 校准阶段一次性算 |

没有机制让 scale 基于梯度优化学习。LSQ（Learned Step Size Quantization）和 PACT 类方法的 scale 通过梯度下降逼近局部最优，效果优于统计/搜索方法。

**核心需求**：让量化 scale 可学习，同时不影响格式内部逻辑。

---

## 设计约束

1. **不改格式内部**：新增格式时不需要考虑"如何让 scale 可学习"
2. **所有格式通用**：int、fp、MX shared exponent、NF4 lookup table 全部兼容
3. **与 QuantSession / 校准管线兼容**：不另开接口
4. **校准后可选 LSQ**：校准提供初始 scale，LSQ 可选地优化
5. **逐层重建优化**：以 FP32 层输出为目标，逐层梯度优化，可配置 batch 数/步数/优化器
6. **内部 scale 固定**：LSQ 优化 pre-scale 时，格式内部 scale（amax / shared_exp）固定不变
7. **不新增 `scale_mode` 字段**：Transform 自身声明行为，不引入冗余模式标志

---

## 决策：Pre-Scale via Transform

### 核心概念

**Pre-scale** 是独立于格式内部 scale 的**前置乘法因子**。它不替代内部 scale，而是与之叠加：

```
x → x * pre_scale → format.quantize(x) → x_q / pre_scale
                         ↑
                    内部 scale（amax / shared_exp）依然生效
```

复合效果：

```
effective_scale_per_element = internal_scale / pre_scale
```

对 MX 格式尤其有价值：shared_exp 是整数（power of 2），pre_scale 是 float32。两者分工——pre_scale 做 coarse channel-wise 调整，shared_exp 做 fine block-wise 调整。

### Transform 自身即声明

**`QuantScheme` 不新增任何字段。** Transform 槽位已经定义了 scale 策略：

| Transform | 含义 | 内部 scale 行为 |
|---|---|---|
| `IdentityTransform()` | 无 pre-scale | 现场计算（当前默认） |
| `PreScaleTransform(scale)` | pre-scale 由外部提供 | 固定（校准值传入或现场算） |
| `SmoothQuantTransform(scale)` | 解析计算的 per-channel scale | 固定 |
| `HadamardTransform()` | 正交旋转 | 现场计算 |

用户想表达"这个 quantization step 使用可学习 scale"时，只需将 `transform` 设置为持有 `nn.Parameter` 引用的 `PreScaleTransform`：

```python
# 可学习 scale：pre_scale 是 nn.Parameter
scheme = QuantScheme(
    format=FP8E4M3Format(),
    granularity=GranularitySpec.per_block(32),
    transform=PreScaleTransform(scale=learnable_param),  # ← 由这行定义
)
```

`IdentityTransform` = "我自己算"，`PreScaleTransform` = "用这个 scale"。**不需要额外的 mode 字段。**

### PreScaleTransform

```python
class PreScaleTransform(TransformBase):
    """Pre-scale transform: x → x * scale, x_q → x_q / scale.
    
    scale is a reference to an externally-owned tensor (buffer or Parameter).
    It is NOT copied — mutations to the original tensor take effect
    on the next forward pass.  This allows:
    - calibration to write the scale into a buffer
    - an optimizer to update the scale during LSQ
    - the same tensor to be shared across multiple QuantScheme instances
    
    invertible = True (the inverse is always defined).
    """
    invertible = True
    
    def __init__(self, scale: torch.Tensor):
        self.scale = scale  # reference, not copy
    
    def forward(self, x):
        return x * self.scale
    
    def inverse(self, x_q):
        return x_q / self.scale
    
    def __eq__(self, other):
        if not isinstance(other, PreScaleTransform):
            return NotImplemented
        return self.scale is other.scale  # identity compare (shared reference)
    
    def __hash__(self):
        return hash(("PreScaleTransform", id(self.scale)))
```

关键设计点：
- `self.scale` 持有对 module 上 `nn.Parameter`（或 buffer）的**引用**，不是拷贝
- optimizer 更新 Parameter → 下次 forward 自动生效
- `__eq__`/`__hash__` 基于 `id(scale)`，避免循环依赖（Parameter 的 data 参与 hash 会导致 QuantScheme hash 不稳定）
- `invertible = True` 确保 `quantize()` 的 inverse 路径生效

---

## 架构集成

### 与格式层的兼容性

所有四种粒度路径已验证，无需改任何格式代码：

```
Per-tensor:  x * pre_scale → quantize_elemwise(x)          → / pre_scale   ✅
Per-channel: x * pre_scale → / amax → elemwise → * amax   → / pre_scale   ✅
Per-block:   x * pre_scale → / 2^se → elemwise → * 2^se   → / pre_scale   ✅
Lookup:      x * pre_scale → clamp → nn.levels            → / pre_scale   ✅
```

**NF4 + block 量化走标准 `quantize()` → `format.quantize()` → `_quantize_per_block()` 路径，完全支持。**

### 内部 scale 固定策略

LSQ 优化期间，格式内部 scale 固定不变：

| 粒度 | 内部 scale | 固定方式 |
|---|---|---|
| Per-tensor | 无 | 不需处理 |
| Per-channel | amax | 校准值通过 `scale=` 参数传入 `quantize()` |
| Per-block | shared_exp | 现场计算，取整数值（不可微，自然固定） |
| Lookup | levels | 格式常量，不变 |

per-channel 路径的关键：`FormatBase._quantize_per_channel()` 已有 `scale=` 参数。校准后将 amax 作为固定 scale 传入，pre_scale 在其上做细粒度调整。

### quantize_mx() 修复

`quantize_mx()` 公共函数当前绕过了 ADR-001 三步流程（见 `src/quantize/mx_quantize.py:250-253`），违反了设计原则。修复方式：

```python
def quantize_mx(A, scheme, ...):
    if scheme is None:
        return A
    if not isinstance(scheme.transform, IdentityTransform):
        # Delegate to the standard three-step flow (ADR-001 compliant)
        return quantize(A, scheme)
    # ... existing direct MX logic (no-transform fast path)
```

修复后 `quantize_mx()` 不再是"另一种量化方式"，而是 `quantize()` 的无 transform 快速路径。所有格式（包括 NF4、LookupFormat）通过 `quantize()` 主入口均可使用 block 量化 + 任意 transform。

### QuantSession 新方法

```python
session = QuantSession(model, cfg)

# 阶段 1：校准（收集统计 + 初始 scale）
with session.calibrate(store_activations=True):
    for batch in calib_data:
        session(batch)

# 阶段 2：初始化 pre_scale Parameters（从校准 scale 或 absmax 或 ones）
session.initialize_pre_scales(init="calibrated")  # "calibrated" | "absmax" | "ones"

# 阶段 3：逐层 LSQ 优化
session.optimize_scales(
    LayerwiseScaleOptimizer(
        num_steps=200,
        num_batches=8,
        optimizer="adam",
        lr=1e-3,
        loss="mse",
    )
)
# optimizer 只更新 pre_scale Parameters，内部 scale 固定

# 推理
session.compare(test_loader, eval_fn)
session.export_onnx("model.onnx")
```

### LayerwiseScaleOptimizer（路径 A）

逐层优化，使用真实量化中间激活（BRECQ 风格）：

```
For layer N (from 0 to L-1):
  1. 用 session.qmodel（前 N-1 层已量化）跑校准 batches
     → 得到 layer N 的真实输入 x_input（含量化噪声）
  2. 用 session.fp32_model 的 layer N 跑 x_input → y_fp32（target）
  3. 梯度优化 layer N 的 pre_scale：
     minimize MSE(y_fp32, layer_N_quant(x_input))
  4. 冻结 layer N 的 pre_scale，继续 layer N+1
```

存储策略：只存校准 batches 原始输入。中间激活全部按需重算。

复杂度：O(L² × B) forward（无 backward）+ O(L × B × S) forward+backward（优化步）。
L=层数，B=batch 数，S=每层优化步数。

---

## 被拒绝的方案

**方案 A：让 scale 成为 `nn.Parameter` 替代内部 amax**

拒绝原因：每种格式的 scale 语义不同（amax、shared exponent、level range），改动需要侵入每个格式的 `quantize()` 实现。违背"加新能力不改已有代码"原则。

**方案 B：`QuantScheme` 新增 `scale_mode` 字段**

拒绝原因：`scale_mode` 和 `transform` 槽位表达的语义重叠。`IdentityTransform` 就是 "computed"，`PreScaleTransform` 就是 "learnable/calibrated"。引入 `scale_mode` 会产生不一致状态（`scale_mode="learnable"` + `transform=IdentityTransform` 语义矛盾）。Transform 自身足够声明行为。

**方案 C：LSQ 优化直接在 `CalibrationSession` 内执行**

拒绝原因：`CalibrationSession` 是上下文管理器，负责 hook 注册和数据收集。LSQ 优化是独立的多步迭代过程，需要不同生命周期的控制流。两者合并会破坏单一职责。

**方案 D：保持 `quantize_mx()` 不变，仅加文档说明**

拒绝原因：`quantize_mx()` 绕开 ADR-001 三步流程是设计缺陷，不是文档问题。它阻止所有格式（包括 NF4）在 block 量化场景下使用 transform。修复该函数使其委托到 `quantize()` 主入口。

---

## 判断标准

- [x] `PreScaleTransform` 实现（持有外部 scale tensor 引用，`__eq__`/`__hash__` 基于 id，含 PoT 支持）
- [x] `QuantScheme` 无新增字段（Transform 自身声明行为）
- [x] `quantize_mx()` 修复：非 IdentityTransform 时委托到 `quantize()`
- [x] `LayerwiseScaleOptimizer` 实现路径 A（逐层 + 真实量化中间激活 + 内部 scale 固定 + PoT 投影梯度下降）
- [x] `QuantSession.initialize_pre_scales()` 和 `optimize_scales()` 方法
- [x] 所有已有 1305 tests 通过（无 regression）
- [x] 新增 P5 测试 40 tests（PreScaleTransform 22 + LayerwiseScaleOptimizer 9 + QuantSession 集成 8 + quantize_mx 委托 1）
