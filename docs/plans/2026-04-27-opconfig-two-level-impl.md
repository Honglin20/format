# OpQuantConfig 两阶段重构 — 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 OpQuantConfig 从 `tuple[QuantScheme, ...]` pipeline 重构为 `QuantScheme | None` 两阶段模型（storage + compute），消除所有算子中的 for 循环和 GranularityMode 拆分。

**Architecture:** 直接 breaking change — 同时修改 OpQuantConfig、所有算子、所有消费者、所有测试。不保留向后兼容。量化调用统一为 `quantize(x, cfg.storage)` + `quantize(x, cfg.role)` 两句模式。

**Tech Stack:** Python 3.10+, PyTorch, dataclasses

**Design doc:** `docs/plans/2026-04-27-opconfig-two-level-design.md`

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

**Step 2: 验证导入**

```bash
python -c "from src.scheme.op_config import OpQuantConfig; print(OpQuantConfig())"
```

Expected: `OpQuantConfig(storage=None, input=None, weight=None, ...)`

**Step 3: Commit**

```bash
git add src/scheme/op_config.py
git commit -m "refactor(scheme): change OpQuantConfig from tuple pipeline to QuantScheme|None two-level model"
```

---

### Task 2: 更新 Linear 算子

**Files:**
- Modify: `src/ops/linear.py`（全文）

**Step 1: 更新 LinearFunction.forward()**

移除 `GranularityMode` import，所有 `for s in cfg.xxx:` 循环替换为 `if cfg.storage:` / `if cfg.xxx:` 判断。

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

    # bias: storage only
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

**Step 2: 更新 LinearFunction.backward()**

同样的模式：storage → compute，每 role 一句。

```python
@staticmethod
def backward(ctx, grad_y):
    x, w = ctx.saved_tensors
    cfg: OpQuantConfig = ctx.cfg
    emit_fn = ctx.emit_fn

    # grad_output: storage → compute
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

    grad_w = g_gw.reshape(-1, ctx.out_dim).T @ x_gw.reshape(-1, ctx.in_dim)

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

**Step 3: 更新 LinearFunction.symbolic()**

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

**Step 4: 更新 QuantizedLinear（移除 GranularityMode import）**

`QuantizedLinear.__init__` 和 `forward()` 基本不变，只需移除不再需要的 import。

**Step 5: Commit**

```bash
git add src/ops/linear.py
git commit -m "refactor(ops): simplify Linear quantization to two-level storage+compute model"
```

---

### Task 3: 更新 Conv / ConvTranspose 算子

**Files:**
- Modify: `src/ops/conv.py`（全文）

**Step 1: 更新 ConvFunction.forward()**

和 Linear 相同的模式。Conv 特点：bias 在 conv 内部，output 只有一步。

关键差异：
- input MX 沿 axis=1（channel dim），由 scheme 的 granularity 决定，算子层不关心
- weight MX 沿 axis=1（Conv）或 axis=0（ConvTranspose）
- 移除所有 `input_elem`/`input_mx` 拆分逻辑

**Step 2: 更新 ConvFunction.backward()**

同 Linear backward 模式。

**Step 3: 更新 ConvTransposeFunction（forward + backward）**

同 Conv，注意 weight transpose 的 axis 差异（MX axis=0）。

**Step 4: 更新所有 symbolic() 方法**

**Step 5: Commit**

```bash
git add src/ops/conv.py
git commit -m "refactor(ops): simplify Conv/ConvTranspose quantization to two-level storage+compute model"
```

---

### Task 4: 更新 Norm 算子

**Files:**
- Modify: `src/ops/norm.py`

**Step 1: 更新 inner_scheme 提取**

所有 `cfg.input[0]` → `cfg.input`。`inner_scheme` 参数保留但自动转换：

```python
# QuantizedBatchNorm2d.__init__() 中
if inner_scheme is not None and cfg is None:
    cfg = OpQuantConfig(input=inner_scheme)  # 不再 tuple 包装
```

**Step 2: 注入 storage 到 vec_ops 调用**

Norm 的 vec_quantize / vec_add / vec_mul 等内部调用传入 `inner_scheme`。storage 的注入在 entry/exit 点（QuantizedBatchNorm2d.forward() 中）：

```python
def forward(self, x):
    if cfg.storage is not None:
        x = quantize(x, cfg.storage)
    # ... norm forward with inner_scheme ...
    if cfg.storage is not None:
        output = quantize(output, cfg.storage)
    return output
```

**Step 3: Commit**

---

### Task 5: 更新 Activation 算子

**Files:**
- Modify: `src/ops/activations.py`

**Step 1: 更新所有 7 个 Quantized* 类**

每个类的 `__init__` 中：`inner_scheme` → 若提供则 `cfg = OpQuantConfig(input=inner_scheme)`。
每个类的 `forward()` 中：`cfg.input[0]` → `cfg.input`。

**Step 2: 注入 storage**

```python
def forward(self, input):
    inner_scheme = self.cfg.input
    if inner_scheme is None:
        return super().forward(input)
    if self.cfg.storage is not None:
        input = quantize(input, self.cfg.storage)
    result = XxxFunction.apply(input, inner_scheme, ...)
    if self.cfg.storage is not None:
        result = quantize(result, self.cfg.storage)
    return result
```

**Step 3: Commit**

---

### Task 6: 更新 Softmax

**Files:**
- Modify: `src/ops/softmax.py`

同样的模式：`inner_scheme` → `cfg.input`，storage 注入 entry/exit。

---

### Task 7: 更新 Pool

**Files:**
- Modify: `src/ops/pooling.py`

---

### Task 8: 更新 Elemwise / Vec ops

**Files:**
- Modify: `src/ops/elemwise.py`
- Modify: `src/ops/vec_ops.py`

**Step 1: vec_quantize 支持 storage**

`vec_quantize(x, scheme)` 保持不变。storage 在调用方注入。

**Step 2: elemwise SIMD ops 更新**

```python
# simd_add 中
if cfg.storage is not None:
    in1 = quantize(in1, cfg.storage)
    in2 = quantize(in2, cfg.storage)
qin1 = vec_quantize(in1, inner_scheme)
qin2 = vec_quantize(in2, inner_scheme)
result = vec_add(qin1, qin2, inner_scheme)
if cfg.storage is not None:
    result = quantize(result, cfg.storage)
```

---

### Task 9: 更新 mapping / quantize_model

**Files:**
- Modify: `src/mapping/quantize_model.py`

**Step 1: 更新 _EMPTY_CFG**

```python
_EMPTY_CFG = OpQuantConfig()  # 不变，但 new OpQuantConfig() 现在所有字段为 None
```

**Step 2: storage 传递**

`quantize_model(model, cfg)` 不变。如果需要全局 storage，用户直接在 cfg 中设置 `storage=...`。

---

### Task 10: 更新 context / QuantizeContext

**Files:**
- Modify: `src/context/quantize_context.py`

**Step 1: inline op cfg 消费更新**

QuantizeContext 中所有 `for s in cfg.input:` → `if cfg.input:` 模式。

---

### Task 11: 更新 ONNX helpers

**Files:**
- Modify: `src/onnx/helpers.py`

**Step 1: _emit_quantize_node 不变**

该函数委托给 `scheme.format.export_onnx(g, x, scheme)`，签名不变。

**Step 2: 各算子 symbolic() 的 for 循环已在 Task 2-8 中更新**

---

### Task 12: 更新 session.py

**Files:**
- Modify: `src/session.py`

**Step 1: storage 参数传递**

`QuantSession(model, cfg, storage=...)` — 如果提供了 storage，将其注入到 cfg 中（或构造新 cfg）。

---

### Task 13: 更新兼容层

**Files:**
- Modify: `src/tests/_compat.py`

**Step 1: 重写 op_config_from_mx_specs**

当前适配器生成 `input=(bf16, fp4)` 等 tuple。改为：

```python
def op_config_from_mx_specs(mx_specs: dict, op_type: str = "linear") -> OpQuantConfig:
    bfloat = mx_specs.get("bfloat", 0)
    elem_format_name = mx_specs.get("elem_format", "int8")
    block_size = mx_specs.get("block_size", 32)

    storage = None
    if bfloat == 16:
        storage = QuantScheme(BFloat16Format(), GranularitySpec.per_tensor())

    quant = None
    if elem_format_name:
        fmt = get_format(elem_format_name)
        quant = QuantScheme(fmt, GranularitySpec.per_block(block_size))

    # Per-op axis conventions...
    return OpQuantConfig(storage=storage, input=quant, weight=quant, ...)
```

---

### Task 14: 更新所有测试文件

**Files:**
- Modify: `src/tests/test_ops_equiv_linear.py`
- Modify: `src/tests/test_ops_equiv_conv.py`
- Modify: `src/tests/test_ops_equiv_norm.py`
- Modify: `src/tests/test_ops_equiv_activation.py`
- Modify: `src/tests/test_ops_equiv_softmax.py`
- Modify: `src/tests/test_ops_equiv_pool.py`
- Modify: `src/tests/test_ops_equiv_elemwise.py`
- Modify: `src/tests/test_mapping.py`
- Modify: `src/tests/test_onnx_export.py`
- Modify: `src/tests/test_analysis.py`
- Modify: `src/tests/test_context.py`
- Modify: `src/tests/test_formats.py`
- Modify: `src/tests/test_quantize.py`
- （以及其他受影响的测试文件）

**Step 1: 全局替换模式**

查找所有 `OpQuantConfig(input=(` 模式，替换为：
- 如果 tuple 有两个元素：`OpQuantConfig(storage=<first>, input=<second>)`
- 如果 tuple 有一个元素：`OpQuantConfig(input=<only>)`（或 storage，取决于语义）

**Step 2: 逐文件验证**

每个测试文件修改后，运行该文件的测试确保通过。

**Step 3: Commit**

---

### Task 15: 全量测试 + 修复 regression

**Step 1: 运行全量测试**

```bash
pytest src/tests/ -x -q
```

**Step 2: 修复所有 failure**

逐个修复，每个修复后重新运行。

**Step 3: 确认 0 xfail**

```bash
pytest src/tests/ -q
```

Expected: 所有测试通过，数量 ≥ 当前 1247。

**Step 4: Commit**

---

### Task 16: 更新文档

**Files:**
- Modify: `docs/architecture/005-op-quant-config.md`
- Modify: `CLAUDE.md` Section 3.2

**Step 1: 更新 ADR-005**

反映新的两阶段设计。

**Step 2: 更新 CLAUDE.md**

同步更新 Section 3.2 "算子级配置"的描述。

**Step 3: Commit**

---

### Task 17: 最终 Review

派遣 review agent 检查：
- 所有 `for s in cfg.xxx:` 已消灭
- 所有 `GranularityMode` import 已从算子中移除
- 所有 `cfg.xxx[0]` / `cfg.xxx[1]` 已替换
- storage 语义在所有算子中一致
- inner_scheme 向下兼容参数保留
- 全量测试通过

---

## 执行顺序依赖

```
Task 1 (OpQuantConfig)           ← 先做，阻塞所有后续
  ↓
Task 2-8 (各算子族)              ← 可并行，但建议按序（每个 commit 独立）
  ↓
Task 9-12 (mapping/context/onnx/session) ← 依赖算子更新完成
  ↓
Task 13-14 (compat + tests)      ← 依赖所有消费者更新完成
  ↓
Task 15 (全量测试)               ← 依赖 tests 更新完成
  ↓
Task 16-17 (docs + review)       ← 最后
```
