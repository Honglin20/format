# OpQuantConfig 两阶段重构设计

**日期**: 2026-04-27
**状态**: 待实现
**分支**: feature/refactor-src

---

## 1. 问题诊断

### 1.1 核心缺陷：tuple pipeline 是一个不存在的抽象

当前 `OpQuantConfig` 的每个字段是 `tuple[QuantScheme, ...]`，设计意图是支持"多步量化 pipeline"。但实践中：

- **量化只有两种**：存储格式（storage precision，per-tensor 浮点 cast）和计算格式（compute format，per-block MX）
- **pipeline 长度从未超过 2**：`(storage_scheme, compute_scheme)` — 且 storage 在每一个非空 pipeline 中重复出现
- **不存在 3+ 步的场景**，也从未出现过"同一个 role 用两个不同的 compute format 串联"

```python
# 当前：storage scheme 在每个 role 中重复
cfg = OpQuantConfig(
    input=(bf16, fp4),     # bf16 出现了
    weight=(bf16, fp4),    # bf16 又出现了
    bias=(bf16,),          # bf16 又出现了
    output=(bf16, bf16),   # bf16 出现了两次
)
```

### 1.2 算子层泄露了 granularity 知识

每个算子的 forward 中都有这段完全相同的代码：

```python
input_elem = tuple(s for s in cfg.input if s.granularity.mode != GranularityMode.PER_BLOCK)
input_mx = tuple(s for s in cfg.input if s.granularity.mode == GranularityMode.PER_BLOCK)
for s in input_elem: x = quantize(x, s)
x_post_elem = x
for s in input_mx: x = quantize(x, s)
```

这段逻辑在 Linear、Conv、ConvTranspose 的 forward 和 backward 中**一字不差地重复**。算子不应该知道 PER_BLOCK vs 非 PER_BLOCK 的区别——这是 `FormatBase.quantize()` 的内部职责。

### 1.3 遍历模式掩盖量化逻辑

`for s in cfg.input: x = quantize(x, s)` 在 99% 情况下迭代 0 或 1 次。循环结构让量化逻辑看起来比实际更复杂，且 `output[0]`/`output[1]` 的硬编码索引使代码脆弱。

---

## 2. 目标态设计

### 2.1 核心结构

```python
@dataclass(frozen=True)
class OpQuantConfig:
    """算子量化配置 — 两阶段模型。

    量化格式只分两种：
    - storage: 存储精度（per_tensor 浮点 elemwise cast），全模型统一
    - compute: 计算量化（per_block MX 等），按 tensor role 分别指定

    每个字段是 QuantScheme | None。没有 tuple，没有 pipeline，没有遍历。
    """

    # ---- 存储精度（全模型统一，所有 tensor 每个量化点都经过）----
    storage: QuantScheme | None = None

    # ---- 计算量化（每个 role 最多一个 scheme）----
    input:  QuantScheme | None = None
    weight: QuantScheme | None = None
    bias:   QuantScheme | None = None
    output: QuantScheme | None = None

    # ---- backward（QAT）----
    grad_output: QuantScheme | None = None
    grad_input:  QuantScheme | None = None
    grad_weight: QuantScheme | None = None
    grad_bias:   QuantScheme | None = None

    # ---- backward 内部 gemm 复量化 ----
    input_gw:       QuantScheme | None = None
    grad_output_gw: QuantScheme | None = None
    weight_gi:       QuantScheme | None = None
    grad_output_gi:  QuantScheme | None = None

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
        return any(
            getattr(self, f) is not None
            for f in _BACKWARD_FIELD_NAMES
        )
```

### 2.2 从 tuple pipeline 到两阶段的语义映射

| 旧（tuple） | 新（QuantScheme\|None） | 说明 |
|---|---|---|
| `input=(bf16, fp4)` | `storage=bf16, input=fp4` | storage 提取到统一字段 |
| `weight=(bf16, fp4)` | `storage=bf16, weight=fp4` | 同上 |
| `bias=(bf16,)` | `storage=bf16` | bias 只有 storage，不需要显式声明 |
| `output=(bf16, bf16)` | `storage=bf16` | 两步都是 storage，Linear 内部自动处理 |

### 2.3 算子量化代码：统一模式

每个量化点从 5-8 行缩减为 2-3 行，所有算子遵循同一模式：

```python
# — 模式：storage 先行，compute 后行 —
def _quantize_role(x, storage, compute, role, emit_fn):
    """所有算子、所有 role 统一的量化调用模式。"""
    if storage is not None:
        x = quantize(x, storage)
    if compute is not None:
        x = quantize(x, compute)
    return x
```

**Linear forward 示例**：

```python
# input
x = _quantize_role(x, cfg.storage, cfg.input, "input", emit_fn)
# weight
w = _quantize_role(w, cfg.storage, cfg.weight, "weight", emit_fn)
# bias (storage only, no compute)
if b is not None and cfg.storage is not None:
    b = quantize(b, cfg.storage)
# matmul
y = F.linear(x, w)
# output step 1 (post-matmul)
if cfg.storage is not None:
    y = quantize(y, cfg.storage)
# output step 2 (post-bias) — Linear 特有
if b is not None:
    y = y + b
    if cfg.storage is not None:
        y = quantize(y, cfg.storage)
```

**Conv forward 示例**（bias 在 conv 内部，只有一步 output）：

```python
x = _quantize_role(x, cfg.storage, cfg.input, "input", emit_fn)
w = _quantize_role(w, cfg.storage, cfg.weight, "weight", emit_fn)
if b is not None and cfg.storage is not None:
    b = quantize(b, cfg.storage)
y = F.conv2d(x, w, b, ...)
if cfg.storage is not None:
    y = quantize(y, cfg.storage)
```

### 2.4 inner_scheme 语义：cfg.input 自动复用

非 matmul 算子（Activation / Softmax / Norm / Pool / Elemwise）的每个中间量化步骤使用同一个 scheme。目标态中，`cfg.input` 自动承担这个角色：

```python
# QuantizedSigmoid.forward()
inner_scheme = self.cfg.input  # 不再需要 cfg.input[0]
if inner_scheme is not None:
    input = quantize(input, cfg.storage)   # storage
    input = quantize(input, inner_scheme)  # compute
```

`inner_scheme` 向下兼容参数保留但标记 deprecated，自动转换为 `cfg.input`。

### 2.5 single cfg → 全模型语义

当用户传一个 `OpQuantConfig` 给 `quantize_model(model, cfg)` 时：

| Op 族 | `cfg.storage` | `cfg.input` | `cfg.weight` | `cfg.output` |
|-------|-------------|------------|-------------|-------------|
| Linear/Conv/BMM | 所有 tensor | 主输入 compute | 权重 compute | 结果 compute |
| Activation (Sigmoid/Tanh/ReLU/...) | 输入 + 输出 | inner_scheme | 忽略 | 忽略 |
| Norm (BN/LN/GN/RMS) | 所有 tensor | inner_scheme | weight compute | 忽略 |
| Softmax | 输入 + 输出 | inner_scheme | 忽略 | 忽略 |
| Pool | 输入 + 输出 | inner_scheme | 忽略 | 忽略 |
| Elemwise (QuantizeContext inline) | 所有 operand | inner_scheme | 忽略 | 忽略 |

**原则**：无 weight 的算子静默忽略 `cfg.weight`。用户不需要为不同 op 族写不同的 cfg。

### 2.6 backward（QAT）简化

```python
# 当前：每个 role 遍历
for s in cfg.grad_output: grad_y = quantize(grad_y, s)
for s in cfg.input_gw: x_gw = quantize(x_gw, s)
for s in cfg.grad_output_gw: g_gw = quantize(g_gw, s)
for s in cfg.grad_weight: grad_w = quantize(grad_w, s)

# 目标：每行一句，storage 自动先行
grad_y = _quantize_role(grad_y, cfg.storage, cfg.grad_output, ...)
x_gw = _quantize_role(x_gw, cfg.storage, cfg.input_gw, ...)
g_gw = _quantize_role(g_gw, cfg.storage, cfg.grad_output_gw, ...)
grad_w = _quantize_role(grad_w, cfg.storage, cfg.grad_weight, ...)
```

### 2.7 ONNX export

```python
# symbolic() 同样简化为每 role 一句
if cfg.storage is not None:
    x = _emit_quantize_node(g, x, cfg.storage)
if cfg.input is not None:
    x = _emit_quantize_node(g, x, cfg.input)
```

`_emit_quantize_node` 签名不变，委托给 `scheme.format.export_onnx(g, x, scheme)`。

---

## 3. 不做的设计

- **不保留 tuple 兼容**：分支是开发分支，直接 breaking change
- **不修改 `quantize(x, scheme)` 签名**：算子层负责调用两次，quantize() 保持单一职责
- **不加 `compute` 默认字段**：当前保持显式。若后续发现 input/weight 重复设置频繁，可加 `compute` 作为 fallback
- **不拆分 MatmulQuantConfig / ElemwiseQuantConfig**：12 字段扁平结构已验证可工作，分子类增加复杂度但收益有限

---

## 4. 波及范围

| 层 | 文件 | 变更类型 |
|---|---|---|
| scheme | `src/scheme/op_config.py` | 重写：tuple → QuantScheme\|None，新增 storage |
| quantize | `src/quantize/` | 不变 |
| ops | `src/ops/linear.py` | 重写量化调用点（约 25 处 for 循环 → if 判断） |
| ops | `src/ops/conv.py` | 同上 |
| ops | `src/ops/norm.py` | 同上，cfg.input[0] → cfg.input |
| ops | `src/ops/activations.py` | 同上 |
| ops | `src/ops/softmax.py` | 同上 |
| ops | `src/ops/pooling.py` | 同上 |
| ops | `src/ops/elemwise.py` | vec_quantize 调用更新 |
| ops | `src/ops/vec_ops.py` | inner_scheme 参数处理更新 |
| mapping | `src/mapping/quantize_model.py` | _EMPTY_CFG 更新；cfg resolve 逻辑不变 |
| context | `src/context/quantize_context.py` | inline op cfg 消费更新 |
| onnx | `src/onnx/helpers.py` | symbolic 中 for 循环 → if |
| session | `src/session.py` | storage= 参数传递 |
| tests | `src/tests/_compat.py` | op_config_from_mx_specs 适配器更新 |
| tests | `src/tests/test_*.py` | 所有 OpQuantConfig 构造点更新 |
| docs | `docs/architecture/005-op-quant-config.md` | ADR 更新 |
| docs | `CLAUDE.md` | Section 3.2 更新 |

---

## 5. 验收标准

- [ ] `OpQuantConfig` 所有角色字段为 `QuantScheme | None`（非 tuple）
- [ ] `storage` 字段存在、验证、语义正确
- [ ] 所有算子 forward/backward 中无 `for s in cfg.xxx:` 循环
- [ ] 所有算子 forward/backward 中无 `GranularityMode` 引用
- [ ] `cfg.input[0]` / `cfg.output[0]` 模式全部替换为 `cfg.input` / `cfg.output`
- [ ] ONNX export symbolic() 中无 tuple 遍历
- [ ] 全部 1247+ 测试通过（0 xfail, 0 regression）
- [ ] 等价性测试 bit-exact（与 mx reference 一致）
- [ ] `inner_scheme` 向下兼容参数保留但 deprecated（自动转 cfg.input）
