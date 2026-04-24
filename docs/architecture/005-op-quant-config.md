# ADR-005: OpQuantConfig — 量化算子配置结构

**状态**: 已决策
**日期**: 2026-04-24

---

## 背景与问题

Phase 2 完成后 `QuantScheme` 是**张量级**量化配置（format + granularity + transform + round_mode）。进入 Phase 3（算子层），一个量化算子面对的是**多个 tensor × 多步量化 × 可选 QAT backward**，光一个 `QuantScheme` 表达不完。

### `mx/linear.py` 的真实复杂度（以 Linear 为例）

| Pass | Tensor 角色 | 应用的 scheme |
|---|---|---|
| forward | input | elemwise bf16 cast → MX block quant（axis=-1，fp4/fp8/int8） |
| forward | weight | elemwise bf16 cast → MX block quant（axis=-1） |
| forward | bias | elemwise bf16 cast |
| forward | output（matmul 后） | elemwise bf16 cast |
| forward | output（加 bias 后） | elemwise bf16 cast |
| backward | grad_output（入口） | elemwise cast |
| backward | input（用于计算 grad_weight）| MX block quant（axis=-2） |
| backward | grad_output（用于计算 grad_weight）| MX block quant（axis=-2） |
| backward | grad_weight（出口） | elemwise cast |
| backward | weight（用于计算 grad_input）| MX block quant（axis=0） |
| backward | grad_output（用于计算 grad_input）| MX block quant（axis=-1） |
| backward | grad_input（出口） | elemwise cast |
| backward | grad_bias | elemwise cast |

每个角色上都是"0 到多个 QuantScheme 按顺序执行"的 pipeline。Phase 3 需要一个**明确、扁平、易枚举**的配置结构来容纳这些。

### 对 QuantScheme 的关系

`OpQuantConfig` 不替代 `QuantScheme`，也不包装它。它的每个字段都是**`QuantScheme` 的元组（pipeline）**。`QuantScheme` 保持单一职责——"描述一次张量量化的三轴配置"。

---

## 决策

引入 `src/scheme/op_config.py` 的 `OpQuantConfig` 数据类：**扁平字段、每字段一个 scheme pipeline**。

### 核心设计

```python
from dataclasses import dataclass, field
from typing import Tuple
from src.scheme.quant_scheme import QuantScheme


@dataclass(frozen=True)
class OpQuantConfig:
    """量化算子的完整 scheme 配置。

    每个字段是 tuple[QuantScheme, ...]，表示按顺序应用的量化 pipeline。
    - 空元组 = 不量化（identity pass-through）
    - 单元素 = 一次量化
    - 多元素 = 链式 pipeline，按顺序依次 quantize(x, s)

    forward 路径使用 input / weight / bias / output。
    backward 路径（QAT）额外使用 grad_* 和 _gw / _gi 后缀字段。
    - _gw = 用于 grad_weight 计算的 madtile gemm 输入重新量化
    - _gi = 用于 grad_input 计算的 madtile gemm 输入重新量化

    非 matmul 算子（activation / softmax / norm / elemwise）
    只使用 input / output / grad_output / grad_input，其余留空。
    """

    # ---------- forward ----------
    input:  Tuple[QuantScheme, ...] = ()
    weight: Tuple[QuantScheme, ...] = ()
    bias:   Tuple[QuantScheme, ...] = ()
    output: Tuple[QuantScheme, ...] = ()

    # ---------- backward（QAT）----------
    grad_output: Tuple[QuantScheme, ...] = ()   # backward 入口的 grad_output cast
    grad_input:  Tuple[QuantScheme, ...] = ()   # 出口的 grad_input cast
    grad_weight: Tuple[QuantScheme, ...] = ()   # 出口的 grad_weight cast
    grad_bias:   Tuple[QuantScheme, ...] = ()   # 出口的 grad_bias cast

    # ---------- backward 内部 gemm 的复量化 ----------
    # grad_weight = (grad_output_gw).T @ input_gw
    input_gw:       Tuple[QuantScheme, ...] = ()
    grad_output_gw: Tuple[QuantScheme, ...] = ()
    # grad_input = grad_output_gi @ weight_gi
    weight_gi:       Tuple[QuantScheme, ...] = ()
    grad_output_gi:  Tuple[QuantScheme, ...] = ()

    def __post_init__(self):
        for name, value in self._iter_fields():
            if not isinstance(value, tuple):
                raise TypeError(
                    f"OpQuantConfig.{name} must be tuple[QuantScheme, ...], "
                    f"got {type(value).__name__}"
                )
            for i, s in enumerate(value):
                if not isinstance(s, QuantScheme):
                    raise TypeError(
                        f"OpQuantConfig.{name}[{i}] must be QuantScheme, "
                        f"got {type(s).__name__}"
                    )

    def _iter_fields(self):
        for f in self.__dataclass_fields__:
            yield f, getattr(self, f)

    @property
    def is_training(self) -> bool:
        """True if any backward field is non-empty (QAT active)."""
        return any(
            getattr(self, f) for f in (
                "grad_output", "grad_input", "grad_weight", "grad_bias",
                "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi",
            )
        )
```

### 算子消费样板（Linear 例）

```python
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, cfg: OpQuantConfig):
        for s in cfg.input:  x = quantize(x, s)
        for s in cfg.weight: w = quantize(w, s)
        if b is not None:
            for s in cfg.bias: b = quantize(b, s)

        y = F.linear(x, w, b)
        for s in cfg.output: y = quantize(y, s)

        ctx.save_for_backward(x, w)
        ctx.cfg = cfg
        ctx.has_bias = b is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg

        for s in cfg.grad_output: grad_y = quantize(grad_y, s)

        # grad_weight gemm
        x_gw = x
        g_gw = grad_y
        for s in cfg.input_gw:       x_gw = quantize(x_gw, s)
        for s in cfg.grad_output_gw: g_gw = quantize(g_gw, s)
        grad_w = g_gw.reshape(-1, w.shape[0]).T @ x_gw.reshape(-1, w.shape[1])
        for s in cfg.grad_weight: grad_w = quantize(grad_w, s)

        # grad_input gemm
        w_gi = w
        g_gi = grad_y
        for s in cfg.weight_gi:      w_gi = quantize(w_gi, s)
        for s in cfg.grad_output_gi: g_gi = quantize(g_gi, s)
        grad_x = g_gi @ w_gi
        for s in cfg.grad_input: grad_x = quantize(grad_x, s)

        grad_b = None
        if ctx.has_bias:
            grad_b = grad_y.reshape(-1, w.shape[0]).sum(0)
            for s in cfg.grad_bias: grad_b = quantize(grad_b, s)

        return grad_x, grad_w, grad_b, None
```

整个 backward 是"按 cfg 字段机械消费"，没有分支逻辑，易于阅读 / 扩展 / 查错。

---

## 为什么选扁平字段而不是嵌套结构

| 方案 | 优点 | 劣势 |
|---|---|---|
| **扁平（当前选）** | NAS/搜索易枚举；序列化简单；IDE 自动补全；算子消费只读字段无分支 | 字段数 12，看起来多 |
| 嵌套 `forward: dict[str, ...], backward: dict[str, ...]` | 直观分组 | 字符串键打字错误静默；hash/eq 要手工 |
| 分子类 `LinearQuantConfig` / `ElemwiseQuantConfig` | 每种算子字段精简 | 算子一多就膨胀；跨算子的公共 role（input/output）在每个子类重复定义 |
| `List[Tuple[str, QuantScheme]]` 描述符序列 | 灵活 | 字符串绑定脆弱；难静态分析 |

**扁平 + 默认空元组**的好处：
- 非 matmul 算子只填 input/output/grad_output/grad_input，其他 8 个字段保持 `()` ——不是"浪费"，是"语义明确的不参与"。
- 新增算子家族不需要扩展 OpQuantConfig，已有 12 字段覆盖 `mx/` 全部算子模式（含 madtile 复量化）。
- 未来扩展（如 LoRA adapter、quantized attention）若真的需要新角色再加字段，破坏性变更成本可控。

---

## `from_mx_specs` 适配器（仅测试内部使用）

等价性测试需要把 `mx_specs` dict 转成 `OpQuantConfig` 以驱动新算子与 `mx.Linear` 对比。放在 `src/tests/_compat.py`：

```python
def op_config_from_mx_specs(mx_specs: dict, op_type: str = "linear") -> OpQuantConfig:
    """把 mx_specs 平铺到 OpQuantConfig 字段。仅用于等价性测试。

    op_type 影响某些字段的 axes 约定（Linear 用 [-1]，Conv 用 [1]，等等）。
    """
    ...
```

**此函数不进生产代码**，保持 `src/ops/` 零 MxSpecs 污染。

---

## 与 CLAUDE.md §1 "QAT 不在当前范围内" 的关系

该条款在本 ADR 落地后 **作废**。CLAUDE.md 同步更新为："Phase 3 起，QAT（训练感知量化）进入范围"。理由：
1. `mx/` 的 backward 本身就是 QAT 的参考实现，等价性测试必然要覆盖 backward。
2. 不做 QAT 意味着放弃 bit-exact 对标，与 P3 "所有算子重构前后必须完全一致" 冲突。
3. `OpQuantConfig` 的 backward 字段**默认为空**，inference-only 用户完全不受影响——QAT 只是"能开的开关"。

---

## 扩展性分析

| 新需求 | 要做什么 | 不要改什么 |
|---|---|---|
| 新算子类别（如 attention） | 新 `src/ops/attention.py`，消费现有 OpQuantConfig 字段 | OpQuantConfig 本身 |
| 新 tensor role（如 LoRA B-matrix） | OpQuantConfig 加 1 个字段 | 已有算子代码 |
| 新 granularity（动态分组） | GranularitySpec 加新 mode；iter_slices 加分支 | OpQuantConfig |
| NAS/搜索场景 | 遍历 OpQuantConfig 的 dataclass fields 枚举 | OpQuantConfig |

---

## 判断标准（Phase 3 infrastructure 完成条件）

- [ ] `src/scheme/op_config.py::OpQuantConfig` 存在、frozen、可哈希
- [ ] `__post_init__` 验证每个字段是 `tuple` 且元素都是 `QuantScheme`
- [ ] `is_training` 属性正确反映 backward 字段是否填充
- [ ] `src/tests/_compat.py::op_config_from_mx_specs` 适配器覆盖 Linear/Conv/Matmul 三种 op_type
- [ ] 至少一个算子（P3.1 Linear）通过 OpQuantConfig 驱动 + 与 `mx.Linear` bit-exact 等价
