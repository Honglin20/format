# Phase 3 代码 Review 报告

**日期**: 2026-04-25  
**Review 范围**: Phase 3 已完成部分（P3.0 ~ P3.4）  
**Branch**: `claude/review-phase-3-code-q5A6h`

---

## 总体评估

| 维度 | 评分 | 备注 |
|---|---|---|
| 三轴设计（QuantScheme 三元组）遵循度 | ✅ 优 | 所有生产代码无 mx_specs 污染 |
| OpQuantConfig 接口一致性 | ⚠️ 部分 | Activation/Softmax/Pool 绕过 OpQuantConfig |
| bit-exact 等价性 | ✅ 优 | 已完成算子全部通过 torch.equal |
| 测试覆盖 | ⚠️ 部分 | ops 层缺负面测试；P3.5/P3.6 测试未建 |
| Observer 骨架集成 | ❌ 缺陷 | _emit() 未在任何算子中实际调用 |
| Phase 3 完成度 | ⚠️ 约 70% | P3.5/P3.6 未实现；P3.4 尚可 |

---

## Critical 级别缺陷

### C1：`_emit()` 未在任何算子中调用

**位置**: 全部 `src/ops/*.py`  
**影响程度**: 阻断 Phase 4

**问题**：所有算子（Linear、Conv、Norm、Activation 等）都继承了 `ObservableMixin`，但没有任何算子在量化关键点调用 `_emit()`。

CLAUDE.md 明确要求：
> "量化算子在关键点发事件（`ObservableMixin._emit(role, pipeline_index, stage, fp32, quant, scheme, group_map=None)`），不做 analysis 计算"

`ObservableMixin._observers` 为空时 `_emit()` 零开销，所以 Phase 3 就应该把调用点埋好。但现在的状态是：即使 Phase 4 的 `AnalysisContext` 实现完成、observers 挂载成功，所有 observer 的 `on_event()` 也永远不会被触发，因为没有 `_emit()` 调用。

**修复方向**：每个算子的量化关键点（每次 `quantize(x, s)` 前后）增加 `_emit()` 调用，传入正确的 `role`、`pipeline_index`、`stage`、`fp32`、`quant`、`scheme`。可用辅助函数避免重复：

```python
def _q_emit(self, x, scheme, role, idx, stage):
    fp32 = x.detach()
    quant = quantize(x, scheme)
    self._emit(role, idx, stage, fp32, quant, scheme)
    return quant
```

注意：`_emit()` 需要在 `QuantizedXxx.forward()` 中调用（不能在 `XxxFunction.forward()` 中，因为 autograd.Function 不持有 self）。需要把 `ObservableMixin` 整合到模块的 `forward()` 方法中，而不是 `XxxFunction` 的静态方法中。

---

## Major 级别缺陷

### M1：Activation/Softmax/Pool 绕过 OpQuantConfig，接口不一致

**位置**: `src/ops/activations.py:62-75`, `src/ops/softmax.py:76-92`, `src/ops/pooling.py:109-124`  
**影响**: Phase 3.6 mapping 模块、Phase 4 analysis 均需特殊分支处理

CLAUDE.md 接口契约要求：
```python
class QuantizedXxx(ObservableMixin, nn.Xxx):
    def __init__(self, ..., cfg: OpQuantConfig, name: str | None = None):
```

但实际实现：
```python
class QuantizedSigmoid(ObservableMixin, nn.Sigmoid):
    def __init__(self, inner_scheme: QuantScheme = None, ...)  # ❌ 无 cfg: OpQuantConfig
```

当前存在三种不同的量化配置模式：

| 算子族 | 配置方式 | 是否有 OpQuantConfig | 是否有 inner_scheme |
|---|---|---|---|
| Linear / Conv / Matmul / BMM | OpQuantConfig (12字段) | ✅ | ❌ |
| Norm (BatchNorm/LN/GN/RMS) | OpQuantConfig (entry) + inner_scheme | ✅ | ✅ |
| Activation / Softmax / Pool | inner_scheme only | ❌ | ✅ |

ADR-005 明确说："非 matmul 算子（activation / softmax / norm / elemwise）只使用 input / output / grad_output / grad_input，其余留空" —— 即仍然要用 `OpQuantConfig`，只是字段填的少。

**修复方向（供讨论）**：

方案 A（推荐）：为 Activation/Softmax/Pool 增加薄包装，将 `inner_scheme` 自动打包成 `OpQuantConfig(input=(inner_scheme,), ...)` 存储，保持接口一致。内部消费时从 `cfg.input[0]` 取 inner_scheme：
```python
@classmethod
def from_inner_scheme(cls, inner_scheme, **kw):
    cfg = OpQuantConfig(input=(inner_scheme,)) if inner_scheme else OpQuantConfig()
    return cls(cfg=cfg, **kw)
```

方案 B（最小改动）：接受双轨制，在 CLAUDE.md 和 ADR-005 中明确文档化"非matmul族使用 inner_scheme 模式"，在 P3.6 mapping 和 Phase 4 analysis 中做适配。

### M2：`iter_slices` PER_BLOCK 模式忽略 `block_axis`

**位置**: `src/analysis/slicing.py:49-55`  
**影响**: Phase 4 per-block 误差分析会切错维度

```python
elif mode == GranularityMode.PER_BLOCK:
    bs = granularity.block_size
    last_dim = fp32.shape[-1]      # ❌ 写死 last dim
    n_blocks = (last_dim + bs - 1) // bs
    for b in range(n_blocks):
        sl = slice(b * bs, min((b + 1) * bs, last_dim))
        yield ("block", b), fp32[..., sl], quant[..., sl]  # ❌ 忽略 block_axis
```

`GranularitySpec.block_axis` 存储了实际量化轴（Conv forward: axis=1，matmul in2 backward: axis=-2），但 `iter_slices` 完全不使用它，始终对最后一维切片。

对 Conv 的量化事件做分析时，`block_axis=1`（channel 维），但 `iter_slices` 会沿最后一维切，产生错误的 per-block 指标。

**修复**：
```python
elif mode == GranularityMode.PER_BLOCK:
    bs = granularity.block_size
    axis = granularity.block_axis
    if axis < 0:
        axis = fp32.ndim + axis
    dim_size = fp32.shape[axis]
    n_blocks = (dim_size + bs - 1) // bs
    for b in range(n_blocks):
        sl = [slice(None)] * fp32.ndim
        sl[axis] = slice(b * bs, min((b + 1) * bs, dim_size))
        sl = tuple(sl)
        yield ("block", b), fp32[sl], quant[sl]
```

### M3：`_compat.py::_matmul_backward_pipelines` grad_weight 使用错误的 round_key

**位置**: `src/tests/_compat.py:325`

```python
# _matmul_backward_pipelines 中：
gw_elem = _elem_scheme(mx_specs, "round_grad_input")  # ❌ 应为 "round_grad_weight"
gw_pipeline = (gw_elem,) if gw_elem is not None else ()
```

对比 `_linear_backward_pipelines` (line 260)：
```python
gw_elem = _elem_scheme(mx_specs, "round_grad_weight")  # ✅ 正确
```

当前测试中所有 `mx_specs` 均无 `round_grad_weight` / `round_grad_input` 键，两者都 fallback 到同一默认值，所以测试通过。但若用户提供含这些键的 `mx_specs`，matmul 的 grad_weight 量化会静默使用错误的 round_mode。

### M4：P3.5 Elemwise/SIMD/Vector ops 未实现

**缺失文件**:
- `src/ops/elemwise.py`（re-export 层）
- `src/tests/test_ops_equiv_elemwise.py`

`src/ops/vec_ops.py` 提供了部分向量原语，但未对齐 `mx/elemwise_ops.py`、`mx/simd_ops.py`、`mx/vector_ops.py` 的完整 API 清单。Phase 3 验收门之一：`pytest src/tests/test_ops_equiv_elemwise.py` 无法执行。

### M5：P3.6 Mapping + 端到端测试未实现

**缺失文件**:
- `src/mapping/quantize_model.py`
- `src/tests/test_e2e_small_model.py`

Phase 3 最终验收门包含：
```bash
pytest src/tests/test_e2e_small_model.py -x -q  # 端到端小模型 bit-exact 验证
```
当前无法执行。

---

## Minor 级别缺陷

### m1：`QuantizedLinear.forward` 每次创建 `OpQuantConfig()` 比较

**位置**: `src/ops/linear.py:173`
```python
def forward(self, x):
    if self.cfg == OpQuantConfig():   # 每次 forward 都构造新对象
        return F.linear(x, self.weight, self.bias)
```
应在 `__init__` 中缓存判断结果（`self._is_passthrough = (cfg == OpQuantConfig())`）。

### m2：Norm 的 inner_scheme 与 cfg.input/weight/bias 隐式绑定

**位置**: `src/tests/_compat.py:481-484`, `src/ops/norm.py:206-213`

`norm_config_from_mx_specs` 把同一个 `inner_scheme` 同时填入 `cfg.input/weight/bias` 和返回的 `inner_scheme`。Norm 的 forward 实现分别使用：
- `cfg.input/weight/bias` 做入口量化
- `inner_scheme` 做所有中间 vec_ops 量化

这创建了一个隐式约定：两者永远相同。若将来某个 Norm 变体需要不同的入口方案 vs 中间方案，当前设计无法表达。建议在 ADR 中明确文档化此约束。

### m3：`QuantizedGELU.forward` 条件导致 `first_order=True, scheme=None` 走 GELUFunction

**位置**: `src/ops/activations.py:411-417`
```python
def forward(self, input):
    if self.inner_scheme is None and not self.first_order_gelu:
        return super().forward(input)   # 只有这一种情况走原生 GELU
    return GELUFunction.apply(...)     # inner_scheme=None 且 first_order=True 也走这里
```
当 `inner_scheme=None, first_order=True` 时，`GELUFunction` 以无量化模式运行一阶近似 GELU，而非标准 GELU。这可能是意外行为。应明确：若 `inner_scheme=None`，无论 `first_order` 如何，均应走 passthrough。

### m4：Activation/Softmax 缺少 `_compat.py` 适配器负面测试

`activation_config_from_mx_specs`、`softmax_config_from_mx_specs`、`pool_config_from_mx_specs` 无 `test_op_config_compat.py` 覆盖。当前 `test_op_config_compat.py` (135 行) 只测 linear/matmul/conv 路径。

### m5：`src/ops/nn/` 目录为空残留

```
src/ops/nn/__init__.py   # 空文件
src/ops/nn/norm/         # 空目录（只有 __init__.py）
```
这是未使用的目录存根，应清理。

### m6：Conv 等价性测试的 stride/dilation/groups 覆盖不完整

`test_ops_equiv_conv.py` 仅 7 个测试函数，`test_ops_equiv_conv_transpose.py` 仅 7 个。计划要求 "stride / padding / dilation / groups 四个维度各至少 2 个取值"，但当前测试看起来对高级组合覆盖不足（需对照 mx/convolution.py 确认）。

---

## 测试覆盖评估

| 测试文件 | 测试数 | 类型 | 评价 |
|---|---|---|---|
| `test_op_config.py` | ~40 | 单元测试 | ✅ 完整（每字段负面测试） |
| `test_observable_mixin.py` | ~15 | 单元测试 | ✅ 良好 |
| `test_slicing.py` | ~20 | 单元测试 | ⚠️ PER_BLOCK axis 行为未测 |
| `test_ops_equiv_matmul.py` | 18 | 等价性 | ✅ 覆盖 Linear+Matmul+BMM |
| `test_ops_equiv_conv.py` | 7 | 等价性 | ⚠️ 高级组合不足 |
| `test_ops_equiv_conv_transpose.py` | 7 | 等价性 | ⚠️ 同上 |
| `test_ops_equiv_norm.py` | 19 | 等价性 | ✅ 覆盖 BN/LN/GN/RMS |
| `test_ops_equiv_activations.py` | 19 | 等价性 | ✅ 覆盖 7 种激活 |
| `test_ops_equiv_softmax.py` | 6 | 等价性 | ✅ 基本覆盖 |
| `test_ops_equiv_pooling.py` | 4 | 等价性 | ⚠️ 只有 4 个 |
| `test_ops_equiv_elemwise.py` | — | 等价性 | ❌ **不存在** |
| `test_e2e_small_model.py` | — | 端到端 | ❌ **不存在** |

**缺失的负面测试**：
- `iter_slices` PER_BLOCK 对 non-last-dim 轴的行为
- 算子层的无效输入维度（如 AdaptiveAvgPool 传入 2D 张量）
- `_compat.py` activation/softmax/pool 适配器

---

## 软件设计原则评估

### 符合的原则

1. **Strategy 模式（Format）**: FormatBase 子类统一封装不同量化格式，新增格式只需继承，不改核心量化函数 ✅
2. **零依赖原则**: 生产代码无 `from mx` 导入 ✅
3. **单一职责**: `QuantScheme` 只描述一次量化；`OpQuantConfig` 只聚合配置；`vec_ops` 只做带量化的基本运算 ✅
4. **开闭原则（OCP）**: 新 granularity 模式只需在 `iter_slices` 加分支、`GranularitySpec` 加 factory method，不改量化核心 ✅
5. **Observer 模式（Phase 3 骨架）**: 事件发送与接收解耦，零开销保证 ✅（但 C1 表明发送端未接入）

### 违反的原则

1. **接口一致性（LSP/DIP）**: 违反 ← M1：Activation/Pool/Softmax 接口与其他算子不同
2. **开闭原则局部违反**: `iter_slices` PER_BLOCK 不支持 block_axis ← M2：新的 block_axis 值无法无改代码正确工作
3. **信息隐藏**: Norm 的 inner_scheme 与 cfg 字段隐式绑定未文档化 ← m2

---

## 修复优先级建议

| 优先级 | 任务 | 估计工作量 | 建议时机 |
|---|---|---|---|
| P0（阻断 P4） | C1: 在所有算子 forward() 中埋 `_emit()` 调用 | 中（需统一设计） | P3.6 前完成 |
| P0（阻断 P4） | M2: 修复 `iter_slices` PER_BLOCK 不支持 block_axis | 小 | P3.6 前完成 |
| P1（P3.6 前） | M1: 讨论并确定 Activation/Pool/Softmax 接口方案 | 中（设计决策） | 与用户讨论后决定 |
| P1（P3 完成） | M4/M5: 实现 P3.5 Elemwise + P3.6 mapping+e2e | 大 | P3 收尾 |
| P2（最终收口） | M3: 修复 matmul backward round_key | 微小 | 下次改 _compat 时顺带 |
| P2 | m1: 缓存 QuantizedLinear passthrough 判断 | 微小 | 随时 |
| P2 | m3: 修复 GELUFunction inner_scheme=None 条件 | 微小 | 随时 |
| P3（清理） | m5: 删除 `src/ops/nn/` 空目录 | 微小 | 随时 |

---

## Phase 3 验收门当前状态

```bash
# 1. 生产代码零 MxSpecs 残留
grep -rn "MxSpecs|mx_specs|from.*specs import" src/ops/ src/scheme/ src/formats/ src/quantize/ src/analysis/
→ ✅ 无命中

# 2. 全部算子 bit-exact 等价性测试通过
pytest src/tests/test_ops_equiv_*.py -x -q
→ ⚠️ test_ops_equiv_elemwise.py 不存在（P3.5 未做）

# 3. 总测试全绿
pytest src/tests/ -x -q
→ ⚠️ 需实际运行确认（torch 环境）

# 4. 端到端 smoke
pytest src/tests/test_e2e_small_model.py -x -q
→ ❌ 文件不存在（P3.6 未做）
```

**结论**：Phase 3 约完成 **70%**。P3.0 ~ P3.4 已完成，bit-exact 等价性通过。P3.5（Elemwise）和 P3.6（Mapping + e2e）未实现，且 C1（_emit 未接入）和 M2（iter_slices block_axis 缺陷）需在进入 Phase 4 前修复。
