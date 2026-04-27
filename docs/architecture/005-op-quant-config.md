# ADR-005: OpQuantConfig — 量化算子配置结构（两阶段模型）

**状态**: 已决策（2026-04-27 更新为两阶段模型）
**日期**: 2026-04-24（初版），2026-04-27（两阶段重构）

---

## 背景与问题

Phase 2 完成后 `QuantScheme` 是**张量级**量化配置（format + granularity + transform + round_mode）。进入 Phase 3（算子层），一个量化算子面对的是**多个 tensor × 两种量化类型 × 可选 QAT backward**。

### 两种量化类型

量化在算子层面可归为两个逻辑阶段：

| 阶段 | 类型 | 粒度 | 作用域 |
|---|---|---|---|
| **storage** | 存储精度（elemwise cast） | per-tensor | 所有 tensor 统一 |
| **compute** | 计算量化（MX per-block 等） | per-role | 每个 tensor 角色独立配置 |

同一 op 内最多两次量化（先 storage 后 compute），不存在三次或更多次的 pipeline。之前 `Tuple[QuantScheme, ...]` 的 pipeline 模型过度设计。

### 实际案例（Linear）

| Pass | Tensor 角色 | storage | compute |
|---|---|---|---|
| forward | input | bf16 elemwise | fp4 MX block（axis=-1） |
| forward | weight | bf16 elemwise | fp4 MX block（axis=-1） |
| forward | bias | bf16 elemwise | — |
| forward | output（matmul 后） | bf16 elemwise | — |
| forward | output（加 bias 后） | bf16 elemwise | fp4 MX block |
| backward | grad_output（入口） | bf16 elemwise | — |
| backward | input_gw（grad_weight gemm）| bf16 elemwise | fp4 MX block（axis=-2） |
| backward | grad_output_gw | bf16 elemwise | fp4 MX block（axis=-2） |
| backward | grad_weight（出口） | bf16 elemwise | — |
| backward | weight_gi（grad_input gemm）| bf16 elemwise | fp4 MX block（axis=0） |
| backward | grad_output_gi | bf16 elemwise | fp4 MX block（axis=-1） |
| backward | grad_input（出口） | bf16 elemwise | — |
| backward | grad_bias | bf16 elemwise | — |

每个 tensor 角色的量化刚好两类：storage（统一 elemwise cast）+ compute（可选 role-specific MX）。不含三次 pipeline 场景。

---

## 决策：两阶段模型

`OpQuantConfig` 每个字段为 `QuantScheme | None`（不再是 tuple）。

### 核心设计

```python
from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class OpQuantConfig:
    """算子级量化配置 — two-level 模型。

    - storage: 存储精度（per-tensor elemwise），所有 tensor 统一
    - compute: 计算量化（每 role 独立），如 per-block MX
    """

    # ---- storage（统一存储精度）----
    storage: Optional[QuantScheme] = None

    # ---- compute（每 role 计算量化）----
    input:  Optional[QuantScheme] = None
    weight: Optional[QuantScheme] = None
    bias:   Optional[QuantScheme] = None
    output: Optional[QuantScheme] = None

    # ---- backward（QAT）----
    grad_output: Optional[QuantScheme] = None
    grad_input:  Optional[QuantScheme] = None
    grad_weight: Optional[QuantScheme] = None
    grad_bias:   Optional[QuantScheme] = None

    # ---- backward 内部 gemm 复量化 ----
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
        return any(getattr(self, name) is not None
                   for name in _BACKWARD_FIELD_NAMES)
```

### 算子消费样板（Linear）

```python
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, cfg: OpQuantConfig):
        # input: storage → compute
        if cfg.storage: x = quantize(x, cfg.storage)
        if cfg.input:   x = quantize(x, cfg.input)
        # weight: storage → compute
        if cfg.storage: w = quantize(w, cfg.storage)
        if cfg.weight:  w = quantize(w, cfg.weight)
        # bias: storage
        if cfg.storage: b = quantize(b, cfg.storage)

        y = F.linear(x, w, b)

        # output: storage
        if cfg.storage: y = quantize(y, cfg.storage)
        # output compute
        if cfg.output:  y = quantize(y, cfg.output)

        return y

    @staticmethod
    def backward(ctx, grad_y):
        cfg = ctx.cfg

        # grad_output: storage → compute
        if cfg.storage:      grad_y = quantize(grad_y, cfg.storage)
        if cfg.grad_output:  grad_y = quantize(grad_y, cfg.grad_output)

        # grad_weight gemm: 每个输入 storage → compute
        if cfg.storage:         x_gw = quantize(x, cfg.storage)
        if cfg.input_gw:        x_gw = quantize(x_gw, cfg.input_gw)
        if cfg.storage:         g_gw = quantize(grad_y, cfg.storage)
        if cfg.grad_output_gw:  g_gw = quantize(g_gw, cfg.grad_output_gw)
        grad_w = g_gw.T @ x_gw
        if cfg.storage:        grad_w = quantize(grad_w, cfg.storage)
        if cfg.grad_weight:    grad_w = quantize(grad_w, cfg.grad_weight)

        # ... grad_input, grad_bias 同理
```

整个 forward/backward 是"按 cfg 字段机械消费"，没有循环，没有分支逻辑。

---

## inner_scheme 模式（Activation / Softmax / Pool / Norm）

这些算子每个量化步骤使用同一个 scheme，通过 `cfg.input` 作为 `inner_scheme`：

```python
class QuantizedSigmoid(ObservableMixin, nn.Sigmoid):
    def __init__(self, cfg=None, inner_scheme=None, ...):
        if inner_scheme is not None and cfg is None:
            bw = inner_scheme if quantize_backprop else None
            cfg = OpQuantConfig(input=inner_scheme, grad_input=bw)
        self.cfg = cfg or OpQuantConfig()

    def forward(self, input):
        inner_scheme = self.cfg.input
        if inner_scheme is None:
            return super().forward(input)
        if self.cfg.storage is not None:
            input = quantize(input, self.cfg.storage)
        result = SigmoidFunction.apply(input, inner_scheme, quantize_backprop)
        if self.cfg.storage is not None:
            result = quantize(result, self.cfg.storage)
        return result
```

Norm 算子同理，`cfg.input`/`cfg.weight`/`cfg.bias` 同时作为 entry 量化方案和内部 vec_op 的 `inner_scheme`。

---

## 为什么选两阶段扁平模型

| 对比项 | 旧 pipeline 模型 `Tuple[QuantScheme, ...]` | 新两阶段模型 `QuantScheme \| None` |
|---|---|---|
| 消耗方式 | `for s in cfg.input: quantize(x, s)` | `if cfg.input: quantize(x, cfg.input)` |
| 最大量化次数 | 无上限 | 固定 2（storage + compute） |
| 字段数 | 12（无 storage） | 13（含 storage） |
| 循环复杂度 | 每 tensor 需 for 循环 | 每 tensor 至多 2 句判断 |
| ONNX 导出 | 需遍历 pipeline | 固定 storage → compute |
| NAS 枚举 | 需展开 tuple 组合 | 每个字段独立 None/Some |

---

## `from_mx_specs` 适配器（仅测试内部使用）

放在 `src/tests/_compat.py`，负责把 mx_specs dict 转成两阶段 OpQuantConfig：

```python
def op_config_from_mx_specs(mx_specs: dict, op_type: str = "linear") -> OpQuantConfig:
    storage = _elem_scheme(mx_specs)           # bfloat/fp elemwise → None or QuantScheme
    input_mx = _mx_scheme(mx_specs, "a_elem_format", ...)  # MX per-block → None or QuantScheme
    weight_mx = _mx_scheme(mx_specs, "w_elem_format", ...)
    return OpQuantConfig(storage=storage, input=input_mx, weight=weight_mx, ...)
```

**此函数不进生产代码**，保持 `src/ops/` 零 MxSpecs 污染。

---

## 判断标准（已达成）

- [x] `src/scheme/op_config.py::OpQuantConfig` 存在、frozen、可哈希
- [x] `__post_init__` 验证每个字段是 `QuantScheme` 或 `None`
- [x] `is_training` 属性正确反映 backward 字段是否非 None
- [x] `src/tests/_compat.py::op_config_from_mx_specs` 适配器覆盖 Linear/Conv/Matmul/Norm/Activation/Softmax/Pool 全部 op_type
- [x] 所有算子通过 OpQuantConfig 驱动 + 与 mx 对应算子 bit-exact 等价（1201 tests passed）
- [x] 两阶段模型：无 `for s in cfg.xxx:` 循环；无 `GranularityMode` 导入于算子文件；无 `cfg.xxx[0]` 索引
