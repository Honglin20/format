# Phase 3 缺陷修复规范

**日期**: 2026-04-25  
**来源**: `docs/plans/2026-04-25-phase3-review.md` 中 C1、M1、M2 三项  
**状态**: 待实现（仅规范，不含代码）

---

## 总览

| 缺陷 | 影响 | 修复工作量 | 推荐实现顺序 |
|---|---|---|---|
| M2：iter_slices PER_BLOCK 忽略 block_axis | Phase 4 分析切错维度 | 小 | 1（独立，无依赖）|
| M1：Activation/Softmax/Pool 绕过 OpQuantConfig | 接口不统一，阻断 P3.6 mapping 均匀处理 | 中 | 2（M2 后，C1 前）|
| C1：`_emit()` 未在任何算子中调用 | Phase 4 observer 收不到任何事件 | 中（机械化） | 3（M1 后）|

---

## M2：修复 `iter_slices` PER_BLOCK 模式的 block_axis 处理

### 问题

`src/analysis/slicing.py` 的 PER_BLOCK 分支写死沿 last dim 切片，忽略 `GranularitySpec.block_axis` 字段。

**当前行为**：
- Conv 量化用 `block_axis=1`（channel dim），但 `iter_slices` 沿 `axis=-1` 切 → 分析结果错误
- Linear 用 `block_axis=-1`（last dim）→ 碰巧正确

**影响时机**：Phase 4 `AnalysisContext` 激活后立即暴露。当前 Phase 3 的 `_emit()` 从未被调用，所以现在无实际影响，但属于定时炸弹。

### 修复规范

**修改文件**：`src/analysis/slicing.py`

**算法变更**（PER_BLOCK 分支替换）：

原逻辑（伪代码）：
```
last_dim = fp32.shape[-1]
按 last_dim 切 block
```

新逻辑（伪代码）：
```
axis = granularity.block_axis
若 axis < 0：axis = fp32.ndim + axis
若 axis 越界：raise ValueError（含 block_axis 值和 ndim 的清晰错误信息）
dim_size = fp32.shape[axis]
构造通用 index tuple，仅 axis 维度切片，其余 slice(None)
按 dim_size 和 block_size 切 n_blocks 块
```

**新增测试**（`src/tests/test_slicing.py`）：

补充以下场景（每个场景一条 `pytest.raises` 或断言测试）：

1. `test_per_block_non_last_axis`：`block_axis=1`，验证切片沿 axis=1，而非 last dim
   - 输入形状 `(4, 8, 16)`，`block_size=4, block_axis=1`
   - 期望：2 个 block，每块形状 `(4, 4, 16)`
2. `test_per_block_negative_axis`：`block_axis=-2`（等效 axis=1），与上面结果相同
3. `test_per_block_axis_out_of_range`：`block_axis=5` 对 ndim=3 → 应 `raise ValueError` 含 "block_axis" 关键字
4. `test_per_block_last_axis_unchanged`：`block_axis=-1`（默认），验证行为与修复前一致（回归）

**验收标准**：
- 上述 4 条测试全通过
- 现有所有 slicing 测试无回归
- `test_slicing.py::test_per_block_*` 可独立运行

---

## M1：统一 Activation / Softmax / Pool 的配置接口为 OpQuantConfig

### 问题

当前三类算子用 `inner_scheme: QuantScheme` 作为唯一量化参数，而 ADR-005 要求所有算子对外暴露 `cfg: OpQuantConfig`。

**影响**：
- P3.6 mapping 模块无法用统一逻辑处理所有算子（需 `isinstance` 分支）
- Phase 4 `AnalysisContext` 挂载 observer 时无法统一调用 `cfg.is_training`
- `emit_fn` 机制（C1）无法在不知道是哪种接口的情况下统一传递

### 修复方案：A'（模块层接口统一，Function 内部不变）

**核心原则**：只改 `QuantizedXxx` 模块类的 `__init__` 和 `forward`；`XxxFunction` 静态方法不动，bit-exact 等价性零风险。

### 修改范围

**涉及文件**：
- `src/ops/activations.py`（7 个模块类）
- `src/ops/softmax.py`（1 个模块类）
- `src/ops/pooling.py`（1 个模块类）
- `src/tests/_compat.py`（适配器返回值变更）
- `src/tests/test_ops_equiv_activations.py`（调用方式更新）
- `src/tests/test_ops_equiv_softmax.py`（调用方式更新）
- `src/tests/test_ops_equiv_pooling.py`（调用方式更新）

### 模块类 `__init__` 规范

每个 `QuantizedXxx.__init__` 按以下规则重写：

**新参数签名**：
```
def __init__(self, cfg: OpQuantConfig = None,
             inner_scheme: QuantScheme = None,   # 向下兼容，自动转换
             quantize_backprop: bool = True,
             name: str = None, **原有特殊参数如 inplace/negative_slope/dim 等)
```

**`__init__` 内部逻辑（伪代码）**：
```
若 cfg 和 inner_scheme 同时非 None → raise ValueError("不能同时指定 cfg 和 inner_scheme")

若 inner_scheme 不为 None 且 cfg 为 None：
    fwd_pipeline = (inner_scheme,)
    bw_pipeline  = (inner_scheme,) if quantize_backprop else ()
    cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
    # 注：Softmax backward 用 grad_input，Pooling backward 也用 grad_input

若 cfg 为 None：
    cfg = OpQuantConfig()    # 全空 = passthrough

self.cfg = cfg
# 不再单独存 self.inner_scheme 和 self.quantize_backprop
```

**`forward()` 内部逻辑（伪代码）**：
```
inner_scheme = cfg.input[0] if cfg.input else None
quantize_backprop = bool(cfg.grad_input)   # 有 backward scheme 即为 QAT 模式

若 inner_scheme 为 None：
    return super().forward(input)   # passthrough，与现在行为一致

调用 XxxFunction.apply(input, ..., inner_scheme, quantize_backprop, ...)
```

注意：`XxxFunction.apply()` 的参数列表不变，`inner_scheme` 和 `quantize_backprop` 依然是函数层的参数，只是现在从 `cfg` 中提取，而非由外部传入。

### 特殊算子的 cfg 字段约定

| 算子 | forward scheme 字段 | backward scheme 字段 | 说明 |
|---|---|---|---|
| Sigmoid / Tanh / ReLU / ReLU6 / LeakyReLU / SiLU / GELU | `cfg.input` | `cfg.grad_input` | 激活无 weight |
| Softmax | `cfg.input` | `cfg.grad_input` | 无 weight/bias |
| AdaptiveAvgPool2d | `cfg.input` | `cfg.grad_input` | 无 weight/bias |

其余字段（`weight`, `bias`, `output`, `grad_weight`, `grad_bias`, `input_gw` 等）均为空 `()`，符合 ADR-005 "非 matmul 算子只填 input/output/grad_*"。

### `_compat.py` 适配器变更规范

`activation_config_from_mx_specs`、`softmax_config_from_mx_specs`、`pool_config_from_mx_specs` 当前返回 `(inner_scheme, quantize_backprop)` 元组。

**变更后返回值**：改为返回 `OpQuantConfig`，与 `norm_config_from_mx_specs` 的对齐方式一致（但不再需要额外返回 `inner_scheme`，因为调用方从 `cfg.input[0]` 自取）。

```
activation_config_from_mx_specs(mx_specs) → OpQuantConfig
softmax_config_from_mx_specs(mx_specs)    → (OpQuantConfig, softmax_exp2: bool)
pool_config_from_mx_specs(mx_specs)       → OpQuantConfig
```

### 测试变更规范

等价性测试文件只需更新调用适配器的方式：

```
# 旧调用方式：
inner, qbp = activation_config_from_mx_specs(mx_specs)
src_out = SigmoidFunction.apply(src_x, inner, qbp)

# 新调用方式：
cfg = activation_config_from_mx_specs(mx_specs)
src_mod = QuantizedSigmoid(cfg=cfg)
src_out = src_mod(src_x)   # 或直接用 Function 层（inner_scheme 从 cfg 提取）
```

**新增测试**（`test_op_config.py` 或独立文件）：

1. `test_activation_cfg_from_inner_scheme`：验证 `QuantizedSigmoid(inner_scheme=s)` 的 `cfg.input == (s,)`
2. `test_activation_cfg_from_opquantconfig`：验证 `QuantizedSigmoid(cfg=OpQuantConfig(input=(s,)))` 正常构造
3. `test_activation_cfg_both_raises`：同时传 `cfg` 和 `inner_scheme` 应 `raise ValueError`
4. `test_activation_passthrough_empty_cfg`：`QuantizedSigmoid(cfg=OpQuantConfig())` 等价于原生 Sigmoid
5. `test_activation_is_training_from_cfg`：`cfg.grad_input` 非空时 `cfg.is_training == True`

**验收标准**：
- 所有已有等价性测试（bit-exact）全绿（无回归）
- 新增 5 条单元测试通过
- `isinstance(mod.cfg, OpQuantConfig)` 对所有 9 个模块类成立

---

## C1：在所有算子中接入 `_emit()` 调用

### 问题

所有算子（Linear、Conv、Norm、Activation、Softmax、Pool）继承 `ObservableMixin` 但从未调用 `_emit()`，Phase 4 的 `AnalysisContext` 挂载 observer 后无法收到任何事件。

### 修复方案：`emit_fn` 回调模式

**核心设计**：`QuantizedXxx.forward()` 将 `self._emit`（或 `None`）作为 `emit_fn` 参数传入 `XxxFunction.apply()`；Function 内部在量化关键点调用 `if emit_fn: emit_fn(...)`。

**零开销保证**：
- 无 observers 时：`self._observers` 为空 → 传 `emit_fn = None` → Function 内所有 `if emit_fn:` 都不执行
- 有 observers 时：`emit_fn = self._emit`，其内部再调用 `QuantEvent` 构造和 observer dispatch

### 修改范围

**涉及文件**：
- `src/ops/linear.py`（`LinearFunction` + `QuantizedLinear`）
- `src/ops/conv.py`（`ConvFunction` + `ConvTransposeFunction` + 6 个模块类）
- `src/ops/norm.py`（4 个 Function + 7 个模块类）
- `src/ops/matmul.py`（`MatMulFunction`）
- `src/ops/bmm.py`（`BMMFunction`）
- `src/ops/activations.py`（7 个 Function + 7 个模块类）— 完成 M1 后一并处理
- `src/ops/softmax.py`（`SoftmaxFunction` + 模块）
- `src/ops/pooling.py`（`AdaptiveAvgPool2dFunction` + 模块）

### Function 层变更规范

**参数签名变更**：所有 `XxxFunction.forward()` 在参数列表末尾增加 `emit_fn = None`（可选）。

```
原签名：def forward(ctx, x, w, b, cfg, name=None)
新签名：def forward(ctx, x, w, b, cfg, name=None, emit_fn=None)
```

**`ctx` 保存**：在 `forward()` 开头执行 `ctx.emit_fn = emit_fn`，使 backward 也能访问。

**emit 调用时机与参数**：

在每次 `x = quantize(x, scheme)` 的前后，按如下规则 emit：

```
fp32_tensor = x（量化前）
quant_tensor = quantize(x, scheme)
if emit_fn:
    emit_fn(role, pipeline_index, stage, fp32_tensor, quant_tensor, scheme)
x = quant_tensor
```

**stage 字符串约定**（来自 CLAUDE.md Section 3.3）：

| 位置 | role 字符串 | stage 字符串 |
|---|---|---|
| forward 输入量化前 | `"input"` | `"input_pre_quant"` |
| forward 权重量化前 | `"weight"` | `"weight_pre_quant"` |
| forward 偏置量化前 | `"bias"` | `"weight_pre_quant"` |
| forward 输出量化后 | `"output"` | `"output_post_quant"` |
| backward grad_output 量化前 | `"grad_output"` | `"grad_output_pre_quant"` |
| backward grad_weight 量化后 | `"grad_weight"` | `"grad_weight_post_quant"` |
| backward grad_input 量化后 | `"grad_input"` | `"grad_input_post_quant"` |

**pipeline_index**：对同一 role 的 pipeline，按 enumerate 顺序从 0 开始。

### 模块层变更规范

每个 `QuantizedXxx.forward()` 在调用 `XxxFunction.apply()` 前确定 `emit_fn`：

```
emit_fn = self._emit if self._observers else None
返回 XxxFunction.apply(...现有参数..., emit_fn)
```

**注意**：`self._observers` 已由 `ObservableMixin` 提供，无需额外改动 mixin。

### Norm 算子的特殊处理

Norm 算子的中间 vec_ops（`vec_add`、`vec_mul` 等）均由 `inner_scheme` 驱动，不在 `OpQuantConfig` 的字段中。对这些中间步骤，emit 时使用特殊 role 字符串 `"inner"` 以区分入口/出口量化：

```
role = "inner"
stage = "inner_pre_quant"
pipeline_index = 按调用顺序（0, 1, 2, ...）
```

这样 Phase 4 的 observer 可以选择性地只关注 `role in ("input", "weight", "output")` 而忽略内部步骤。

### 新增测试规范

**测试文件**：`src/tests/test_observable_mixin.py`（已存在，补充以下测试）

1. `test_emit_not_called_without_observers`：无 observers 时，`LinearFunction.apply()` 后 `observer.on_event` 未被调用（用 mock observer 验证）
2. `test_emit_called_with_observer`：attach 一个简单 observer，`QuantizedLinear.forward()` 后 `observer.events` 非空
3. `test_emit_forward_roles_present`：observer 收到的 event roles 包含 `"input"`、`"weight"`、`"output"`
4. `test_emit_backward_roles_present`：QAT 模式（cfg.grad_output 非空）backward 后，observer 收到 `"grad_output"`、`"grad_weight"`、`"grad_input"` roles
5. `test_emit_zero_overhead_no_observers`：无 observers 时，forward 耗时与原始无 emit 版本相差 < 5%（可选，性能基准）

**验收标准**：
- 上述 4 条功能测试（不含性能测试）通过
- 所有已有等价性测试全绿（emit 调用不影响计算结果）
- `grep -c "_emit" src/ops/linear.py` ≥ 4（至少 input/weight/output/grad_output 4 个调用点）

---

## 实现顺序与 Commit 规范

### 推荐顺序

```
subtask-fix-1: M2 — slicing.py block_axis
  commit: fix(analysis): iter_slices PER_BLOCK respects block_axis, add axis-aware tests

subtask-fix-2: M1 — 接口统一
  commit: refactor(ops): unify Activation/Softmax/Pool to OpQuantConfig interface

subtask-fix-3: M3 — _compat matmul round_key
  commit: fix(tests): correct round_key for matmul grad_weight in _compat adapter

subtask-fix-4: C1 — emit_fn 接入（Linear/Conv/Matmul/BMM）
  commit: feat(ops): wire _emit via emit_fn callback in matmul-family operators

subtask-fix-5: C1 — emit_fn 接入（Norm/Activation/Softmax/Pool）
  commit: feat(ops): wire _emit via emit_fn callback in norm/activation/softmax/pool operators
```

每个 subtask 遵循 CLAUDE.md §4 生命周期：TDD → 实现 → 测试通过 → review agent → commit → 更新 CURRENT.md → 发出 clear context 信号。

### 与后续 Phase 的衔接

| 修复项 | Phase 4 解锁能力 |
|---|---|
| M2 | per-block 粒度分析正确切片 |
| M1 | P3.6 mapping 可统一处理所有算子；Phase 4 AnalysisContext 统一注入 |
| C1 | Phase 4 observer 能收到量化事件，QSNR/MSE/Histogram 计算可启动 |
