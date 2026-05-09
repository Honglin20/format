# OpQuantConfig 两阶段重构 — 实现计划 v2

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 OpQuantConfig 从 `tuple[QuantScheme, ...]` pipeline 重构为 `QuantScheme | None` 两阶段模型（storage + compute），消除所有算子中的 for 循环和 GranularityMode 拆分。

**Architecture:** 直接 breaking change — 同时修改 OpQuantConfig、所有算子、所有消费者、所有测试。不保留向后兼容。量化调用统一为 `quantize(x, cfg.storage)` + `quantize(x, cfg.role)` 两句模式。storage 先行（per-tensor elemwise），compute 后行（per-block MX 等）。

**Tech Stack:** Python 3.10+, PyTorch, dataclasses

**Design doc:** `docs/plans/2026-04-27-opconfig-two-level-design.md`

**v2 Improvements over v1:**
- 补全 matmul.py / bmm.py 两个遗漏算子
- 新增 `_compat.py` adapter 逐字段映射规范
- 修复 norm backward 重复量化 bug
- 明确 _patches.py `_simd_inner_scheme` 变更
- 补全 vec_ops.py 不变更声明
- passthrough 检查语义验证
- 全部 test 文件的精确替换模式

---

## 波及文件索引

| 层 | 文件 | 当前 for-in 循环数 | 变更类型 |
|---|---|---|---|
| scheme | `src/scheme/op_config.py` | 2 (__post_init__) | 重写 |
| ops | `src/ops/linear.py` | 14 | 重构所有量化点 |
| ops | `src/ops/conv.py` | 22 (Conv + ConvTranspose) | 重构所有量化点 |
| ops | `src/ops/matmul.py` | 15 | 重构所有量化点 |
| ops | `src/ops/bmm.py` | 14 | 重构所有量化点 |
| ops | `src/ops/norm.py` | 34 (4 norm × fwd+bwd) | 重构 + 修 bug |
| ops | `src/ops/activations.py` | 0 (vec_* driven) | cfg.input[0] → cfg.input + storage 注入 |
| ops | `src/ops/softmax.py` | 0 (vec_* driven) | cfg.input[0] → cfg.input + storage 注入 |
| ops | `src/ops/pooling.py` | 0 (vec_* driven) | cfg.input[0] → cfg.input + storage 注入 |
| ops | `src/ops/elemwise.py` | 0 (vec_* driven) | 不变（只经过 vec_*，不直接用 cfg） |
| ops | `src/ops/vec_ops.py` | — | **不变**（接口稳定） |
| context | `src/context/_patches.py` | 0 | `_simd_inner_scheme` cfg.input[0] → cfg.input |
| context | `src/context/quantize_context.py` | 0 | 不变（只持有 cfg 引用） |
| mapping | `src/mapping/quantize_model.py` | 0 | 不变（只传递 cfg） |
| onnx | `src/onnx/helpers.py` | 0 | 不变（委托给 Format.export_onnx()） |
| session | `src/session.py` | 0 | storage= 参数支持 |
| tests | `src/tests/_compat.py` | 0 | 重写 adapter（tuple → scalar） |
| tests | `src/tests/test_*.py` (~14 files) | 0 | OpQuantConfig 构造点替换 |
| docs | `docs/architecture/005-op-quant-config.md` | — | 更新 ADR |
| docs | `CLAUDE.md` | — | 更新 Section 3.2 |

---

### Task 0: 确认基线 + 创建分支

**Step 1: 确认当前基线测试全部通过**

```bash
cd /Users/mozzie/Desktop/Projects/Analyser/microxcaling
pytest src/tests/ -q
```

Expected: 1247+ passed, 0 xfail

**Step 2: 创建/确认工作分支**

```bash
git checkout feature/refactor-src
git checkout -b refactor/opconfig-two-level
```

**Commit:**
```bash
git commit --allow-empty -m "chore: start OpQuantConfig two-level refactoring"
```

---

### Task 1: 重构 OpQuantConfig 核心

**Files:**
- Modify: `src/scheme/op_config.py`（全文重写）

**Step 1: 重写 OpQuantConfig**

将所有字段从 `Tuple[QuantScheme, ...] = ()` 改为 `QuantScheme | None = None`，新增 `storage` 字段。

```python
"""
OpQuantConfig: operator-level quantization configuration — two-level model.

Quantization has exactly two types:
- storage: storage precision (per-tensor elemwise cast), uniform across all tensors
- compute: compute quantization (per-block MX etc.), per-role

Each field is QuantScheme | None. No tuples, no pipelines, no iteration.
"""
from dataclasses import dataclass, fields
from typing import Optional

from .quant_scheme import QuantScheme

_BACKWARD_FIELD_NAMES = frozenset((
    "grad_output", "grad_input", "grad_weight", "grad_bias",
    "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi",
))

# All field names that carry QuantScheme|None (for __post_init__ validation)
_ALL_FIELD_NAMES = frozenset((
    "storage",
    "input", "weight", "bias", "output",
    "grad_output", "grad_input", "grad_weight", "grad_bias",
    "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi",
))

@dataclass(frozen=True)
class OpQuantConfig:
    """Operator-level quantization configuration.

    Two-level quantization model:
    - storage: applied to EVERY tensor at every quantization point,
      always first (elemwise storage precision cast, e.g. bfloat16)
    - compute: role-specific compute quantization (e.g. fp8 MX per-block)

    Default construction (no arguments) = no quantization on any role.
    """

    # ---- Storage (uniform across all tensors in the model) ----
    storage: Optional[QuantScheme] = None

    # ---- Compute quantization (one per role, None = no compute quant) ----
    input:  Optional[QuantScheme] = None
    weight: Optional[QuantScheme] = None
    bias:   Optional[QuantScheme] = None
    output: Optional[QuantScheme] = None

    # ---- Backward (QAT) ----
    grad_output: Optional[QuantScheme] = None
    grad_input:  Optional[QuantScheme] = None
    grad_weight: Optional[QuantScheme] = None
    grad_bias:   Optional[QuantScheme] = None

    # ---- Backward gemm re-quantization ----
    input_gw:       Optional[QuantScheme] = None
    grad_output_gw: Optional[QuantScheme] = None
    weight_gi:       Optional[QuantScheme] = None
    grad_output_gi:  Optional[QuantScheme] = None

    def __post_init__(self):
        for f in fields(self):
            if f.name not in _ALL_FIELD_NAMES:
                continue
            value = getattr(self, f.name)
            if value is not None and not isinstance(value, QuantScheme):
                raise TypeError(
                    f"OpQuantConfig.{f.name} must be QuantScheme or None, "
                    f"got {type(value).__name__}"
                )

    @property
    def is_training(self) -> bool:
        """True if any backward field is non-None (QAT active)."""
        return any(getattr(self, name) is not None for name in _BACKWARD_FIELD_NAMES)
```

**Step 2: 验证导入和基本行为**

```bash
python -c "
from src.scheme.op_config import OpQuantConfig
cfg = OpQuantConfig()
print(cfg)
assert cfg == OpQuantConfig()
assert not cfg.is_training

cfg2 = OpQuantConfig(input=None, storage=None)
assert cfg2 == OpQuantConfig()  # None == None → equal

from src.scheme.quant_scheme import QuantScheme
s = QuantScheme(format='bfloat16')
cfg3 = OpQuantConfig(storage=s)
assert cfg3 != OpQuantConfig()
"
```

**Step 3: 验证 __post_init__ 错误路径**

```bash
python -c "
from src.scheme.op_config import OpQuantConfig
# Tuple no longer accepted
try:
    OpQuantConfig(input=(1,))
    assert False, 'should have raised'
except TypeError as e:
    assert 'must be QuantScheme or None' in str(e)
print('OK')
"
```

**Step 4: Commit**

```bash
git add src/scheme/op_config.py
git commit -m "refactor(scheme): change OpQuantConfig from tuple pipeline to QuantScheme|None two-level model"
```

---

### Task 2: 更新 Linear 算子

**Files:**
- Modify: `src/ops/linear.py`（全文）

**变更清单：**
1. 移除 `from src.scheme.granularity import GranularityMode`（第 19 行）
2. `forward()`: 将 GranularityMode 拆分替换为 storage→compute 两步
3. `backward()`: 所有 `for s in cfg.xxx:` → `if cfg.xxx: quantize(x, cfg.xxx)`
4. `symbolic()`: `for scheme in cfg.xxx:` → `if cfg.xxx:` 判断
5. `QuantizedLinear`: 保持不变（passthrough check 自动适应新 cfg）

**Step 1: 更新 LinearFunction.forward() — 第 43-129 行**

```python
@staticmethod
def forward(ctx, x, w, b, cfg: OpQuantConfig, name=None, emit_fn=None,
            output_scale=None):
    ctx.emit_fn = emit_fn
    x_raw, w_raw = x, w

    # input: storage → compute
    if cfg.storage is not None:
        fp_x = x; x = quantize(x, cfg.storage)
        if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_x, x, cfg.storage)
    x_post_storage = x
    if cfg.input is not None:
        fp_x = x; x = quantize(x, cfg.input)
        if emit_fn: emit_fn("input", 1, "input_pre_quant", fp_x, x, cfg.input)

    # weight: storage → compute
    if cfg.storage is not None:
        fp_w = w; w = quantize(w, cfg.storage)
        if emit_fn: emit_fn("weight", 0, "weight_pre_quant", fp_w, w, cfg.storage)
    w_post_storage = w
    if cfg.weight is not None:
        fp_w = w; w = quantize(w, cfg.weight)
        if emit_fn: emit_fn("weight", 1, "weight_pre_quant", fp_w, w, cfg.weight)

    # bias: storage only (no compute for bias)
    q_bias = b
    if b is not None and cfg.storage is not None:
        fp_b = q_bias; q_bias = quantize(q_bias, cfg.storage)
        if emit_fn: emit_fn("bias", 0, "weight_pre_quant", fp_b, q_bias, cfg.storage)

    # Save for backward
    if cfg.is_training:
        ctx.save_for_backward(x_post_storage, w_post_storage)
    else:
        ctx.save_for_backward(x_raw, w_raw)

    ctx.cfg = cfg
    ctx.has_bias = b is not None
    ctx.in_dim = w_raw.shape[1]
    ctx.out_dim = w_raw.shape[0]
    ctx.name = name

    # matmul
    y = _F_linear(x, w)

    # output step 1 (post-matmul): storage
    if cfg.storage is not None:
        fp_y = y; y = quantize(y, cfg.storage, scale=output_scale)
        if emit_fn: emit_fn("output", 0, "output_post_quant", fp_y, y, cfg.storage)

    # bias add + output step 2 (post-bias): storage
    if q_bias is not None:
        y = y + q_bias
        if cfg.storage is not None:
            fp_y = y; y = quantize(y, cfg.storage, scale=output_scale)
            if emit_fn: emit_fn("output", 1, "output_post_quant", fp_y, y, cfg.storage)

    # output compute (applied after all storage steps)
    if cfg.output is not None:
        fp_y = y; y = quantize(y, cfg.output, scale=output_scale)
        if emit_fn: emit_fn("output", 2, "output_post_quant", fp_y, y, cfg.output)

    return y
```

**Step 2: 更新 LinearFunction.backward() — 第 131-192 行**

每个角色的量化从 `for s in cfg.xxx:` loop 替换为 `if cfg.storage:` + `if cfg.xxx:` 两步：

```python
@staticmethod
def backward(ctx, grad_y):
    x, w = ctx.saved_tensors
    cfg: OpQuantConfig = ctx.cfg
    emit_fn = ctx.emit_fn

    # grad_output
    if cfg.storage is not None:
        grad_y = quantize(grad_y, cfg.storage)
    if cfg.grad_output is not None:
        grad_y = quantize(grad_y, cfg.grad_output)

    # grad_weight gemm
    x_gw = x
    if cfg.storage is not None:
        x_gw = quantize(x_gw, cfg.storage)
    if cfg.input_gw is not None:
        x_gw = quantize(x_gw, cfg.input_gw)

    g_gw = grad_y
    if cfg.storage is not None:
        g_gw = quantize(g_gw, cfg.storage)
    if cfg.grad_output_gw is not None:
        g_gw = quantize(g_gw, cfg.grad_output_gw)

    g_gw_2d = g_gw.reshape(-1, ctx.out_dim)
    x_gw_2d = x_gw.reshape(-1, ctx.in_dim)
    grad_w = g_gw_2d.T @ x_gw_2d

    if cfg.storage is not None:
        grad_w = quantize(grad_w, cfg.storage)
    if cfg.grad_weight is not None:
        grad_w = quantize(grad_w, cfg.grad_weight)

    # grad_input gemm
    w_gi = w
    if cfg.storage is not None:
        w_gi = quantize(w_gi, cfg.storage)
    if cfg.weight_gi is not None:
        w_gi = quantize(w_gi, cfg.weight_gi)

    g_gi = grad_y
    if cfg.storage is not None:
        g_gi = quantize(g_gi, cfg.storage)
    if cfg.grad_output_gi is not None:
        g_gi = quantize(g_gi, cfg.grad_output_gi)

    grad_x = g_gi @ w_gi

    if cfg.storage is not None:
        grad_x = quantize(grad_x, cfg.storage)
    if cfg.grad_input is not None:
        grad_x = quantize(grad_x, cfg.grad_input)

    # grad_bias
    grad_b = None
    if ctx.has_bias:
        grad_b = grad_y.reshape(-1, ctx.out_dim).sum(0)
        if cfg.storage is not None:
            grad_b = quantize(grad_b, cfg.storage)
        if cfg.grad_bias is not None:
            grad_b = quantize(grad_b, cfg.grad_bias)

    return grad_x, grad_w, grad_b, None, None, None, None
```

**Step 3: 更新 symbolic() — 第 194-220 行**

```python
@staticmethod
def symbolic(g, x, w, b, cfg, name, emit_fn, output_scale=None):
    from src.onnx.helpers import _emit_quantize_node

    if cfg.storage is not None:
        x = _emit_quantize_node(g, x, cfg.storage)
    if cfg.input is not None:
        x = _emit_quantize_node(g, x, cfg.input)

    if cfg.storage is not None:
        w = _emit_quantize_node(g, w, cfg.storage)
    if cfg.weight is not None:
        w = _emit_quantize_node(g, w, cfg.weight)

    wt = g.op("Transpose", w, perm_i=[1, 0])
    y = g.op("MatMul", x, wt)

    if cfg.storage is not None:
        y = _emit_quantize_node(g, y, cfg.storage)

    if b is not None:
        if cfg.storage is not None:
            b = _emit_quantize_node(g, b, cfg.storage)
        y = g.op("Add", y, b)
        if cfg.storage is not None:
            y = _emit_quantize_node(g, y, cfg.storage)

    return y
```

**Step 4: 移除 GranularityMode import**

第 19 行 `from src.scheme.granularity import GranularityMode` 删除。

**Step 5: 更新 QuantizedLinear（passthrough 检查验证）**

`self._is_passthrough = self.cfg == OpQuantConfig()` 保持不变 — 当任何字段非 None 时 cfg != OpQuantConfig()，自动正确。

**Step 6: 验证 Linear 导入无 import 错误**

```bash
python -c "from src.ops.linear import LinearFunction, QuantizedLinear; print('OK')"
```

**Step 7: Commit**

```bash
git add src/ops/linear.py
git commit -m "refactor(ops): simplify Linear quantization to two-level storage+compute model"
```

---

### Task 3: 更新 MatMul / BMM 算子

**Files:**
- Modify: `src/ops/matmul.py`（全文）
- Modify: `src/ops/bmm.py`（全文）

**注意：** 这俩算子被原 v1 plan 遗漏。它们和 Linear 共享完全相同的 GranularityMode 拆分 + for 循环模式。

**Step 1: 更新 MatMulFunction（forward + backward + symbolic）**

同 Linear 的变更模式：
- 移除 `from src.scheme.granularity import GranularityMode`
- `input_elem`/`input_mx` 拆分 → `cfg.storage` + `cfg.input` 两步
- `for s in cfg.xxx:` → `if cfg.xxx:` 单步
- symbolic() 中 `for scheme in cfg.xxx:` → `if cfg.xxx:`

**MatMul 特有的是 `mode_config`（'aa', 'aw', 'wa'），决定 MX axis**。但 axis 信息存在 QuantScheme.granularity.block_axis 中，算子层不再关心（format.quantize() 内部从 granularity 读取 axis）。所以 mode_config 仅影响 backward 的 gemm axis 选择，不影响量化代码。

**Step 2: 更新 BMMFunction（forward + backward）**

同 MatMul，移除 GranularityMode 拆分。

**Step 3: 验证导入**

```bash
python -c "from src.ops.matmul import MatMulFunction; from src.ops.bmm import BMMFunction; print('OK')"
```

**Step 4: Commit**

```bash
git add src/ops/matmul.py src/ops/bmm.py
git commit -m "refactor(ops): simplify MatMul/BMM quantization to two-level storage+compute model"
```

---

### Task 4: 更新 Conv / ConvTranspose 算子

**Files:**
- Modify: `src/ops/conv.py`（全文，~540 行）

**Conv 有两个 Function 类：** `ConvFunction`（Conv1d/2d/3d）和 `ConvTransposeFunction`（ConvTranspose1d/2d/3d）。每个都有 forward + backward + symbolic。

**Step 1: 更新 ConvFunction.forward()**

同 Linear 模式。注意：
- Conv 的 bias 在 `F.conv2d(..., bias=q_bias, ...)` 内部，output 只有一步
- 移除 input_elem/input_mx/weight_elem/weight_mx 四行拆分

```python
# input: storage → compute
if cfg.storage is not None:
    input = quantize(input, cfg.storage)
input_post_storage = input
if cfg.input is not None:
    input = quantize(input, cfg.input)

# weight: storage → compute
if cfg.storage is not None:
    weight = quantize(weight, cfg.storage)
weight_post_storage = weight
if cfg.weight is not None:
    weight = quantize(weight, cfg.weight)

# bias: storage only
q_bias = bias
if bias is not None and cfg.storage is not None:
    q_bias = quantize(q_bias, cfg.storage)

# Save for backward ... (use post_storage tensors)
```

**Step 2: 更新 ConvFunction.backward()**

```python
# grad_output
if cfg.storage is not None:
    grad_output = quantize(grad_output, cfg.storage)
if cfg.grad_output is not None:
    grad_output = quantize(grad_output, cfg.grad_output)

# grad_weight gemm: input → storage → compute
input_gw = input
if cfg.storage is not None:
    input_gw = quantize(input_gw, cfg.storage)
if cfg.input_gw is not None:
    input_gw = quantize(input_gw, cfg.input_gw)
# ... same for grad_output_gw ...
```

**Step 3: 更新 ConvFunction.symbolic()**

```python
if cfg.storage is not None:
    input = _emit_quantize_node(g, input, cfg.storage)
if cfg.input is not None:
    input = _emit_quantize_node(g, input, cfg.input)
# ... weight, bias, output ...
```

**Step 4: 更新 ConvTransposeFunction（forward + backward + symbolic）**

同 Conv。特别注意 ConvTranspose backward 的 axis 约定由 granularity 内的 block_axis 决定，算子层不关心。

**Step 5: 移除 GranularityMode import**

```bash
python -c "from src.ops.conv import ConvFunction, ConvTransposeFunction; print('OK')"
```

**Step 6: Commit**

```bash
git add src/ops/conv.py
git commit -m "refactor(ops): simplify Conv/ConvTranspose quantization to two-level storage+compute model"
```

---

### Task 5: 更新 Norm 算子 + 修复 duplicate backward bug

**Files:**
- Modify: `src/ops/norm.py`（全文）

**Norm 算子特殊之处：**
- 独立的 `inner_scheme` 参数用于所有 `vec_*` 中间计算
- `cfg.input`/`cfg.weight`/`cfg.bias` 用于**入口量化**（在 inner_scheme 之前）
- `cfg.output`/`cfg.grad_*` 用于出口量化
- **存在 bug**：LayerNorm、GroupNorm、RMSNorm 的 backward 中有两段重复的 `for s in cfg.grad_output` 循环（带 emit_fn 和不带 emit_fn 各一段），应合并为一段

**涉及的 4 个 Function 类：** BatchNormFunction, LayerNormFunction, GroupNormFunction, RMSNormFunction

**Step 1: 更新入口量化模式**

每个 Norm Function 的 forward 开头：
```python
# Before (for BN):
in_idx = 0
for s in cfg.input:
    x = quantize(x, s)
    if emit_fn: emit_fn("input", in_idx, ...)

# After:
if cfg.storage is not None:
    x = quantize(x, cfg.storage)
if cfg.input is not None:
    x = quantize(x, cfg.input)
```

同样的模式应用到 weight 和 bias 入口量化。

**Step 2: 保持 inner_scheme 为独立参数**

Norm 的 `_norm_forward()` 调用仍然传递单独的 `inner_scheme` 参数 — 这是 `vec_*` 调用的配置，不同于 `cfg.input`（entry quant）。inner_scheme 仍通过 `QuantizedBatchNorm2d.__init__` 的 `inner_scheme=` 参数传入。

**Step 3: 更新出口量化**

```python
if cfg.storage is not None:
    output = quantize(output, cfg.storage)
if cfg.output is not None:
    output = quantize(output, cfg.output)
```

**Step 4: 修复 LayerNormFunction.backward() 重复量化 bug**

当前 L527-546 有两段重复代码：
```python
# 第一段（L534-539，带 emit_fn）
go_idx = 0
for s in cfg.grad_output:
    grad_output = quantize(grad_output, s)
    if emit_fn: ...

# 第二段（L544-546，无 emit_fn — BUG: 重复量化）
for s in cfg.grad_output:
    grad_output = quantize(grad_output, s)
```

修复为一段（合并 storage + compute）：
```python
if cfg.storage is not None:
    grad_output = quantize(grad_output, cfg.storage)
if cfg.grad_output is not None:
    grad_output = quantize(grad_output, cfg.grad_output)
if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_output, cfg.grad_output)
```

**同样修复 GroupNormFunction.backward()（L679-690）和 RMSNormFunction.backward()（L831-842）**

**Step 5: 验证导入**

```bash
python -c "from src.ops.norm import BatchNormFunction, LayerNormFunction; print('OK')"
```

**Step 6: Commit**

```bash
git add src/ops/norm.py
git commit -m "refactor(ops): simplify Norm quantization to two-level model + fix duplicate backward quantization bug"
```

---

### Task 6: 更新 Activation 算子

**Files:**
- Modify: `src/ops/activations.py`

**涉及的 7 个类：** QuantizedSigmoid, QuantizedTanh, QuantizedReLU, QuantizedReLU6, QuantizedLeakyReLU, QuantizedSiLU, QuantizedGELU

每个类的变更模式完全相同。以 Sigmoid 为例：

**Step 1: 更新 __init__ — inner_scheme → cfg.input 转换**

```python
# Before:
if inner_scheme is not None and cfg is None:
    fwd_pipeline = (inner_scheme,)
    bw_pipeline = (inner_scheme,) if quantize_backprop else ()
    cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)

# After:
if inner_scheme is not None and cfg is None:
    cfg = OpQuantConfig(
        input=inner_scheme,
        grad_input=inner_scheme if quantize_backprop else None,
    )
```

**Step 2: 更新 forward — cfg.input[0] → cfg.input**

```python
# Before:
inner_scheme = self.cfg.input[0] if self.cfg.input else None
quantize_backprop = bool(self.cfg.grad_input)

# After:
inner_scheme = self.cfg.input
quantize_backprop = self.cfg.grad_input is not None
```

**Step 3: 注入 storage（在 vec_quantize 之前）**

```python
def forward(self, input):
    inner_scheme = self.cfg.input
    if inner_scheme is None:
        return super().forward(input)

    # Storage applied BEFORE vec_quantize
    if self.cfg.storage is not None:
        input = quantize(input, self.cfg.storage)

    emit_fn = self._emit if self._observers else None
    result = XxxFunction.apply(input, inner_scheme, quantize_backprop, ...)

    # Storage applied AFTER activation
    if self.cfg.storage is not None:
        result = quantize(result, self.cfg.storage)

    return result
```

**注意：** 需要新增 `from src.quantize import quantize` import。

**Step 4: 对所有 7 个类重复 Step 1-3**

**Step 5: Commit**

```bash
git add src/ops/activations.py
git commit -m "refactor(ops): simplify Activation quantization to two-level model with storage injection"
```

---

### Task 7: 更新 Softmax + Pool 算子

**Files:**
- Modify: `src/ops/softmax.py`
- Modify: `src/ops/pooling.py`

**Step 1: Softmax — inner_scheme → cfg.input 转换**

```python
# __init__:
if inner_scheme is not None and cfg is None:
    cfg = OpQuantConfig(
        input=inner_scheme,
        grad_input=inner_scheme if quantize_backprop else None,
    )

# forward:
inner_scheme = self.cfg.input
quantize_backprop = self.cfg.grad_input is not None
if inner_scheme is None:
    return super().forward(input)
if self.cfg.storage is not None:
    input = quantize(input, self.cfg.storage)
result = SoftmaxFunction.apply(...)
if self.cfg.storage is not None:
    result = quantize(result, self.cfg.storage)
return result
```

**Step 2: Pool — 同样的模式**

```python
# forward:
inner_scheme = self.cfg.input
quantize_backprop = self.cfg.grad_input is not None
if inner_scheme is None:
    return _f_adaptive_avg_pool2d(input, self.output_size)
if self.cfg.storage is not None:
    input = quantize(input, self.cfg.storage)
result = AdaptiveAvgPool2dFunction.apply(...)
if self.cfg.storage is not None:
    result = quantize(result, self.cfg.storage)
return result
```

**Step 3: Commit**

```bash
git add src/ops/softmax.py src/ops/pooling.py
git commit -m "refactor(ops): simplify Softmax/Pool to two-level model with storage injection"
```

---

### Task 8: 确认 Elemwise / Vec ops 不需变更

**Files:**
- **不变更:** `src/ops/elemwise.py`（SIMD ops 只接收 inner_scheme，不直接消费 OpQuantConfig）
- **不变更:** `src/ops/vec_ops.py`（`vec_*` 函数接口稳定，只接收 QuantScheme|None）

**Step 1: 验证 elemwise.py 不引用 cfg**

```bash
grep -n "cfg\." src/ops/elemwise.py || echo "No cfg references — no changes needed"
```

**Step 2: 验证 vec_ops.py 不引用 cfg**

```bash
grep -n "cfg\|OpQuantConfig" src/ops/vec_ops.py || echo "No cfg references — no changes needed"
```

**Step 3: Commit（或跳过，如无变更）**

---

### Task 9: 更新 context / _patches.py

**Files:**
- Modify: `src/context/_patches.py:76-78`

**Step 1: `_simd_inner_scheme` 更新**

```python
# Before (line 78):
return cfg.input[0] if cfg.input else None

# After:
return cfg.input
```

`_make_emit_fn` 和其他 inline op patches 中的 `cfg == _EMPTY_CFG` 检查保持不变 — `_EMPTY_CFG = OpQuantConfig()` 自动正确。

**Step 2: 确认 quantize_context.py 无变更**

```bash
grep -n "for s in cfg\|cfg\.\[0\]\|cfg\.\[1\]" src/context/quantize_context.py || echo "No tuple indexing — no changes needed"
```

**Step 3: Commit**

```bash
git add src/context/_patches.py
git commit -m "refactor(context): cfg.input[0] → cfg.input in _simd_inner_scheme"
```

---

### Task 10: 确认 mapping / onnx 不需变更

**Files:**
- **不变更:** `src/mapping/quantize_model.py`
- **不变更:** `src/onnx/helpers.py`

**验证:**

```bash
grep -n "cfg\.\[" src/mapping/quantize_model.py || echo "OK"
grep -n "for s in cfg" src/onnx/helpers.py || echo "OK"
```

`quantize_model.py` 只通过 `cfg` 传递，不消费字段。  
`helpers.py` 的 `_emit_quantize_node` 委托给 `Format.export_onnx()`，不受 cfg 结构影响。

---

### Task 11: 更新 _compat.py adapter

**Files:**
- Modify: `src/tests/_compat.py`

这是**关键变更点** — adapter 是测试和 cfg 之间的桥梁。

**Step 1: `op_config_from_mx_specs` 重写**

每个 pipeline 从 `tuple(s for s in [a, b] if s is not None)` → 直接赋值：

```python
def op_config_from_mx_specs(mx_specs: dict, op_type: str = "linear") -> OpQuantConfig:
    block_size = mx_specs.get("block_size", 0)
    quantize_backprop = mx_specs.get("quantize_backprop", True)

    # Storage: elemwise bfloat/fp scheme (shared by all roles)
    storage = _elem_scheme(mx_specs, "round_output")

    # input compute: MX scheme
    input_mx_axis = 1 if op_type in ("conv", "conv_transpose") else -1
    input_compute = _mx_scheme(mx_specs, "a_elem_format", block_size,
                                "round_mx_output", block_axis=input_mx_axis)

    # weight compute: MX scheme
    if op_type == "conv":
        weight_mx_axis = 1
    elif op_type == "conv_transpose":
        weight_mx_axis = 0
    elif op_type == "matmul":
        weight_mx_axis = -2
    else:
        weight_mx_axis = -1
    weight_compute = _mx_scheme(mx_specs, "w_elem_format", block_size,
                                 "round_mx_output", block_axis=weight_mx_axis)

    # output: linear uses 2 storage casts (post-matmul + post-bias)
    # but in the new model, output is compute quant after all storage
    # Actually: output=storage for post-matmul storage cast.
    # Linear forward applies storage at output step 1, not output compute.
    # output compute field is for extra quant after all storage steps.
    output_pipeline = None  # mx doesn't set output compute after bias in linear

    if not quantize_backprop:
        return OpQuantConfig(
            storage=storage,
            input=input_compute, weight=weight_compute,
        )

    # Backward pipelines ...
    return _backward_op_config(
        mx_specs, block_size, op_type,
        storage,
        input_compute, weight_compute,
    )
```

**Step 2: `_linear_backward_pipelines` → 返回 OpQuantConfig**

```python
def _linear_backward_pipelines(mx_specs, block_size, storage):
    """Build backward fields for linear, returning a complete OpQuantConfig."""
    # grad_output: storage only
    go_elem = _elem_scheme(mx_specs, "round_grad_input")
    grad_output = storage  # grad_output uses same storage as forward

    # grad_weight gemm
    a_fmt_bp_ex = mx_specs.get("a_elem_format_bp_ex") or mx_specs.get("a_elem_format_bp")
    input_gw = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_ex},
                           "a_elem_format", block_size,
                           "round_mx_input_grad_weight", block_axis=-2) if a_fmt_bp_ex else None
    grad_output_gw = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_ex},
                                 "a_elem_format", block_size,
                                 "round_mx_grad_output_grad_weight", block_axis=-2) if a_fmt_bp_ex else None

    grad_weight = _elem_scheme(mx_specs, "round_grad_weight")

    # grad_input gemm
    w_fmt_bp = mx_specs.get("w_elem_format_bp")
    a_fmt_bp_os = mx_specs.get("a_elem_format_bp_os")
    weight_gi = _mx_scheme({**mx_specs, "w_elem_format": w_fmt_bp},
                            "w_elem_format", block_size,
                            "round_mx_weight_grad_input", block_axis=0) if w_fmt_bp else None
    grad_output_gi = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_os},
                                 "a_elem_format", block_size,
                                 "round_mx_grad_output_grad_input", block_axis=-1) if a_fmt_bp_os else None

    grad_input = _elem_scheme(mx_specs, "round_grad_input")
    grad_bias = _elem_scheme(mx_specs, "round_grad_weight")

    return dict(
        grad_output=grad_output,
        grad_input=grad_input,
        grad_weight=grad_weight,
        grad_bias=grad_bias,
        input_gw=input_gw,
        grad_output_gw=grad_output_gw,
        weight_gi=weight_gi,
        grad_output_gi=grad_output_gi,
    )
```

**Step 3: `op_config_from_mx_specs` 完整返回值**

```python
# 组装 forward + backward config
kwargs = dict(storage=storage, input=input_compute, weight=weight_compute)
if quantize_backprop:
    kwargs.update(_linear_backward_pipelines(mx_specs, block_size, storage))
return OpQuantConfig(**kwargs)
```

**Step 4: 更新 norm/activation/softmax/pool/simd adapter 函数**

- `norm_config_from_mx_specs`: `input_pipeline=(inner_scheme,)` → `storage=inner_scheme`
- `activation_config_from_mx_specs`: `input=(inner_scheme,)` → `input=inner_scheme`
- `softmax_config_from_mx_specs`: 同 activation
- `pool_config_from_mx_specs`: 同 activation
- `simd_config_from_mx_specs`: 不变（返回 inner_scheme，非 OpQuantConfig）

**Step 5: 验证导入**

```bash
python -c "from src.tests._compat import op_config_from_mx_specs; print('OK')"
```

**Step 6: Commit**

```bash
git add src/tests/_compat.py
git commit -m "refactor(tests): update compat adapter for two-level OpQuantConfig"
```

---

### Task 12: 更新所有测试文件

**Files:**
全部包含 `OpQuantConfig(` 构造的测试文件。

**Step 1: 全局替换模式清单**

```
Pattern                                    → Replacement
──────────────────────────────────────────────────────────────────
OpQuantConfig()                            → 不变（空构造签名相同）
OpQuantConfig(input=(s,))                  → OpQuantConfig(input=s)  或 OpQuantConfig(storage=s)
OpQuantConfig(input=(s,), weight=(s,))     → OpQuantConfig(input=s, weight=s)
OpQuantConfig(input=(s,), output=(s,))     → OpQuantConfig(input=s, output=s)
OpQuantConfig(input=(s,), weight=(s,), output=(s,)) → OpQuantConfig(input=s, weight=s, output=s)
OpQuantConfig(input=(s,), grad_input=(s,)) → OpQuantConfig(input=s, grad_input=s)
```

**关键区分：** 如果 `s` 是 elemwise/bfloat scheme，且原来只有 `input=(s,)`（单元素），它在新模型中应该是 `storage=s` 还是 `input=s`？取决于上下文：
- 如果该 scheme 的 granularity 是 per_tensor → 它是 storage
- 如果 per_block → 它是 compute

**Step 2: 逐文件更新**

| 文件 | 变更数量 | 特殊注意 |
|---|---|---|
| `test_ops_equiv_linear.py` | ~8 处 | adpater 返回新 cfg，测试本身可能不用改 |
| `test_ops_equiv_conv.py` | ~5 处 | 同上 |
| `test_ops_equiv_matmul.py` | ~5 处 | 同上 |
| `test_ops_equiv_norm.py` | ~5 处 | 同上 |
| `test_ops_equiv_activation.py` | ~3 处 | 同上 |
| `test_ops_equiv_softmax.py` | ~3 处 | 同上 |
| `test_ops_equiv_pool.py` | ~3 处 | 同上 |
| `test_ops_equiv_elemwise.py` | ~3 处 | 同上 |
| `test_op_config.py` | ~10 处 | **直接构造 OpQuantConfig，需逐行改** |
| `test_op_config_compat.py` | ~3 处 | **测试 tuple → None 行为变更** |
| `test_analysis.py` | ~5 处 | 同上 |
| `test_onnx_export.py` | ~5 处 | 同上 |
| `test_compare.py` | ~3 处 | 同上 |
| `test_quantize_context.py` | ~5 处 | 同上 |
| `test_e2e_small_model.py` | ~10 处 | 同上 |
| `test_integration_quantize_context.py` | ~3 处 | 同上 |
| `test_observable_mixin.py` | ~3 处 | 同上 |

**Step 3: 更新 test_op_config.py**

此文件直接测试 OpQuantConfig 构造和验证，需重写 tuple 相关测试：

```python
# Before:
cfg = OpQuantConfig(input=(s1, s2), weight=(s3,))

# After:
cfg = OpQuantConfig(storage=s1, input=s2, weight=s3)
```

**Step 4: 更新 test_op_config_compat.py**

此文件测试 __post_init__ 对 tuple 的 rejection，需更新：

```python
# 旧测试: 验证 tuple 中非 QuantScheme 元素被拒绝
# → 删除（tuple 不再接受）

# 新测试: 验证非 QuantScheme|None 被拒绝
def test_rejects_non_scheme_non_none():
    with pytest.raises(TypeError, match="must be QuantScheme or None"):
        OpQuantConfig(input="fp4")
```

**Step 5: 验证测试文件导入**

```bash
python -c "import src.tests.test_op_config; print('OK')"
python -c "import src.tests._compat; print('OK')"
```

**Step 6: Commit**

```bash
git add src/tests/
git commit -m "test: update all tests for two-level OpQuantConfig"
```

---

### Task 13: 全量测试 + 修复 regression

**Step 1: 运行全量测试**

```bash
pytest src/tests/ -x -q
```

**Step 2: 逐个修复 failure**

常见 failure 模式：
1. `TypeError: OpQuantConfig.input must be QuantScheme or None, got tuple` → 测试构造点还在用 tuple
2. `AttributeError: 'tuple' object has no attribute 'granularity'` → 有遗漏的 `cfg.input[0]` → 改为 `cfg.input`
3. `for s in cfg.xxx:` → 遗漏的 for 循环
4. 等价性测试 bit-exact failure → adapter 映射不正确

**Step 3: 确认 0 xfail**

```bash
pytest src/tests/ -q
```

Expected: all passed, count >= 1247, 0 xfail.

**Step 4: Commit**

```bash
git add -A
git commit -m "fix: repair all test failures from OpQuantConfig refactoring"
```

---

### Task 14: 更新文档

**Files:**
- Modify: `docs/architecture/005-op-quant-config.md`
- Modify: `CLAUDE.md` Section 3.2

**Step 1: 更新 ADR-005**

反映两阶段设计，移除 tuple pipeline 描述。

**Step 2: 更新 CLAUDE.md**

Section 3.2 "算子级配置"更新为两阶段描述。

**Step 3: Commit**

```bash
git add docs/architecture/005-op-quant-config.md CLAUDE.md
git commit -m "docs: update ADR-005 and CLAUDE.md for two-level OpQuantConfig"
```

---

### Task 15: 最终 Review

**派遣 review agent 检查以下清单：**

| 检查项 | 验证方法 |
|---|---|
| 所有 `for s in cfg.xxx:` 已消灭 | `grep -rn "for s in cfg\." src/ops/` |
| 所有 `GranularityMode` 已从算子移除 | `grep -rn "GranularityMode" src/ops/` |
| 所有 `cfg.xxx[0]` / `cfg.xxx[1]` 已替换 | `grep -rn "cfg\.\w\+\[0\]" src/` |
| storage 语义一致（所有算子 storage 先行） | 抽查 Linear/Conv/Norm/Activation forward |
| inner_scheme 向下兼容参数保留 | 检查 Activation/Softmax/Pool __init__ |
| 全量测试通过 | `pytest src/tests/ -q` |
| 等价性 bit-exact | `pytest src/tests/test_ops_equiv_*.py -x` |
| Norm backward duplicate 量化已修复 | 目视检查 LayerNorm/GroupNorm/RMSNorm backward |

---

## 执行顺序依赖

```
Task 0 (基线确认)
  ↓
Task 1 (OpQuantConfig)         ← 先做，阻塞所有后续
  ↓
Task 2-8 (算子族)              ← 相互独立，可按序
  ├─ Task 2: Linear
  ├─ Task 3: MatMul / BMM
  ├─ Task 4: Conv / ConvTranspose
  ├─ Task 5: Norm + bug fix
  ├─ Task 6: Activation
  ├─ Task 7: Softmax / Pool
  └─ Task 8: Elemwise / Vec（确认不变更）
  ↓
Task 9 (context/_patches)
  ↓
Task 10 (mapping/onnx 确认)
  ↓
Task 11 (_compat.py adapter)
  ↓
Task 12 (全量测试更新)
  ↓
Task 13 (全量测试验证)
  ↓
Task 14 (文档更新)
  ↓
Task 15 (最终 Review)
```

## 验收标准

- [ ] `OpQuantConfig` 所有字段为 `QuantScheme | None`（非 tuple）
- [ ] `storage` 字段存在、验证正确
- [ ] 所有算子 forward/backward 中无 `for s in cfg.xxx:` 循环
- [ ] 所有算子 forward/backward 中无 `GranularityMode` import
- [ ] `cfg.input[0]` / `cfg.output[0]` / `cfg.output[1]` 已消灭
- [ ] ONNX export symbolic() 中无 tuple 遍历
- [ ] 全部 1247+ 测试通过（0 xfail, 0 regression）
- [ ] 等价性测试 bit-exact（与 mx reference 一致）
- [ ] inner_scheme 向下兼容参数保留但 deprecated
- [ ] Norm backward duplicate 量化 bug 已修复
