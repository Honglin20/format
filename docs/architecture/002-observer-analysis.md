# ADR-002: 层级误差分析 — Observer 模式

**状态**: 已决策（2026-04-24 更新，增加 SliceAwareObserver / iter_slices / group_map）
**日期**: 2026-04-24（初稿）、2026-04-24（更新）

---

## 背景与问题

`mx/` 库通过继承 `torch.nn.Linear` 等模块并覆写 `forward` 的方式替换 torch 算子，天然能看到每个算子的输入。目标是利用这一特性，对每个量化算子做 FP32 vs 量化路径的误差对比（QSNR / MSE / 分布）形成层级分析报告。

**设计约束**：
- 量化算子代码（`src/ops/`）不能包含 analysis 逻辑，否则职责爆炸
- 关闭 analysis 时必须零开销
- 未来需要支持多种指标（QSNR、MSE、直方图、分布统计）且易于扩展
- **必须同时支持多种量化粒度**：per-tensor、per-channel、per-block，**以及未来的动态分组量化**
- 算子的量化 pipeline 可能含多步（如 elemwise bf16 cast + MX block quant），Observer 要能区分每步的误差

## 决策

采用 **Observer 模式 + 上下文管理器 + 粒度感知切片抽象**：
- 量化算子在关键点**发出事件**（emit），不做任何计算
- 事件携带 `QuantScheme`（含 granularity）+ `pipeline_index` + 可选 `group_map`
- 外部 `SliceAwareObserver` 基类提供按 granularity 切分张量的统一入口 `iter_slices`
- 具体 Observer（QSNR / MSE / Histogram）只实现 `_measure(key, fp32_slice, quant_slice)`
- `AnalysisContext` 上下文管理器控制挂载/卸载，离开时自动清理

---

## 接口规范

### 事件系统

```python
# src/analysis/events.py

@dataclass(frozen=True)
class QuantEvent:
    layer_name: str                 # 模型中的模块路径，如 "model.layer1.linear"
    role: str                       # tensor 角色："input" / "weight" / "output" / "grad_*"
    pipeline_index: int             # scheme pipeline 中第几步（0 = elemwise cast, 1 = MX block, ...）
    stage: str                      # 事件点名称（见下表）
    fp32_tensor: Tensor             # 这一步的输入（未量化），detach 后的只读视图
    quant_tensor: Tensor            # 这一步的输出（已量化），detach 后的只读视图
    scheme: QuantScheme             # 本步的量化配置（含 granularity）
    group_map: Optional[Tensor]     # 动态分组量化的 group_id 张量；None 表示静态粒度
```

**标准事件点**（stage 名称）：

| stage | 含义 | 触发位置 |
|---|---|---|
| `input_pre_quant` | 输入激活量化前后（每个 pipeline step）| 算子 forward 开头 |
| `weight_pre_quant` | 权重量化前后 | 算子 forward 中权重量化处 |
| `output_post_quant` | 输出量化前后（如有输出量化）| 算子 forward 末尾 |
| `grad_output_pre_quant` | 反向入口 grad_output 量化前后 | 算子 backward 开头 |
| `grad_weight_post_quant` | 反向出口 grad_weight 量化前后 | 算子 backward grad_weight gemm 后 |
| `grad_input_post_quant` | 反向出口 grad_input 量化前后 | 算子 backward grad_input gemm 后 |

**pipeline_index 的含义**：在一个 role 上可能有多个 QuantScheme 串联（见 ADR-005 OpQuantConfig）。`pipeline_index=0` 是第一个 scheme（通常是 elemwise bf16 cast），`pipeline_index=1` 是第二个（通常是 MX block quant）。让 observer 能分别统计每步误差，而不是只看"整条 pipeline 的累计误差"。

### 量化算子侧（事件发出）

量化算子继承 `ObservableMixin`：

```python
# src/analysis/mixin.py

class ObservableMixin:
    """量化算子继承此 Mixin 获得事件发出能力。零开销（无 observer 时直接返回）。"""

    _observers: list = []   # 实例级别，由 AnalysisContext 填充

    def _emit(self, role: str, pipeline_index: int, stage: str,
              fp32: Tensor, quant: Tensor, scheme: "QuantScheme",
              group_map: Optional[Tensor] = None) -> None:
        if not self._observers:          # 无 observer → 直接返回，零开销
            return
        event = QuantEvent(
            layer_name=getattr(self, "_analysis_name", type(self).__name__),
            role=role,
            pipeline_index=pipeline_index,
            stage=stage,
            fp32_tensor=fp32.detach(),
            quant_tensor=quant.detach(),
            scheme=scheme,
            group_map=group_map.detach() if group_map is not None else None,
        )
        for obs in self._observers:
            obs.on_event(event)
```

**算子内部的典型使用**（Linear 示例）：

```python
class QuantizedLinear(ObservableMixin, nn.Linear):
    def forward(self, x):
        for i, s in enumerate(self.cfg.input):
            x_pre = x
            x = quantize(x, s)
            self._emit("input", i, "input_pre_quant", fp32=x_pre, quant=x, scheme=s)

        w = self.weight
        for i, s in enumerate(self.cfg.weight):
            w_pre = w
            w = quantize(w, s)
            self._emit("weight", i, "weight_pre_quant", fp32=w_pre, quant=w, scheme=s)

        return F.linear(x, w, self.bias)
```

Phase 3 落地时 `ObservableMixin._emit` 的实现是**空 no-op 早返回**，`_observers = []` 固定。Phase 4 再加 `AnalysisContext` 挂载机制。

---

### 粒度感知的切片抽象（**本 ADR 更新重点**）

不同粒度下"一个误差值"的含义不同：
- PER_TENSOR：整张张量一个 QSNR
- PER_CHANNEL：每个 channel 一个 QSNR（沿 channel_axis 切）
- PER_BLOCK：每个 block 一个 QSNR（沿 block 维度切）
- 未来 DYNAMIC_GROUP：每个 group（运行时决定）一个 QSNR

让每个 Observer 自己实现切分逻辑是重复工作且易错。抽一个单一入口 `iter_slices`：

```python
# src/analysis/slicing.py

from typing import Iterator, Tuple, Optional
from torch import Tensor
from src.scheme.granularity import GranularitySpec, GranularityMode


SliceKey = Tuple[str, ...]   # 如 ("tensor",), ("channel", 3), ("block", 17), ("group", "id42")


def iter_slices(
    fp32: Tensor,
    quant: Tensor,
    granularity: GranularitySpec,
    group_map: Optional[Tensor] = None,
) -> Iterator[Tuple[SliceKey, Tensor, Tensor]]:
    """按 granularity 生成 (key, fp32_slice, quant_slice) 迭代器。

    单入口 — 新增 granularity mode 只改这里，所有 Observer 自动支持。
    """
    mode = granularity.mode

    if mode == GranularityMode.PER_TENSOR:
        yield ("tensor",), fp32, quant

    elif mode == GranularityMode.PER_CHANNEL:
        axis = granularity.channel_axis
        if axis < 0:
            axis = fp32.ndim + axis
        for i in range(fp32.shape[axis]):
            yield ("channel", i), fp32.select(axis, i), quant.select(axis, i)

    elif mode == GranularityMode.PER_BLOCK:
        # 沿最后一维切 block（与 mx 块量化约定一致）
        bs = granularity.block_size
        last_dim = fp32.shape[-1]
        n_blocks = (last_dim + bs - 1) // bs   # padding 由上游处理
        for b in range(n_blocks):
            sl = slice(b * bs, min((b + 1) * bs, last_dim))
            yield ("block", b), fp32[..., sl], quant[..., sl]

    elif mode == GranularityMode.DYNAMIC_GROUP:   # 未来
        assert group_map is not None, (
            "DYNAMIC_GROUP granularity requires event.group_map to be set. "
            "Make sure the format's quantize() returns the group_map tensor."
        )
        for gid in group_map.unique().tolist():
            mask = (group_map == gid)
            yield ("group", gid), fp32[mask], quant[mask]

    else:
        raise ValueError(f"Unknown granularity mode: {mode}")
```

**GranularityMode.DYNAMIC_GROUP** 是预留值，Phase 3 不启用。引入时：
1. `GranularitySpec.mode` 加新枚举
2. 对应 `Format.quantize()` 返回 `(x_quant, group_map)`（或 attach 到 event）
3. `iter_slices` 的分支已写好

---

### Observer 接口

```python
# src/analysis/observer.py

class ObserverBase(ABC):
    """抽象 observer 基类。用户可直接继承此类以完全自定义 on_event 逻辑。"""

    @abstractmethod
    def on_event(self, event: QuantEvent) -> None: ...

    def report(self) -> dict:
        """返回分层嵌套 dict：{layer_name: {role: {stage: {slice_key: metric}}}}"""
        return {}

    def reset(self):
        """清空累积状态。"""


class SliceAwareObserver(ObserverBase):
    """粒度感知 observer 基类。自动按 granularity 切片并调用 _measure。

    子类只需实现 _measure(key, fp32_slice, quant_slice) -> metric_dict。
    """

    def __init__(self):
        self._buffer = {}

    @abstractmethod
    def _measure(self, key: SliceKey, fp32: Tensor, quant: Tensor) -> dict:
        """每个 slice 计算一个指标 dict，如 {'qsnr_db': 38.2}。"""

    def on_event(self, event: QuantEvent):
        for key, f, q in iter_slices(
            event.fp32_tensor, event.quant_tensor,
            event.scheme.granularity, event.group_map,
        ):
            metric = self._measure(key, f, q)
            dst = (self._buffer
                   .setdefault(event.layer_name, {})
                   .setdefault(event.role, {})
                   .setdefault(f"{event.stage}[{event.pipeline_index}]", {}))
            dst[key] = metric

    def report(self) -> dict:
        return self._buffer

    def reset(self):
        self._buffer.clear()
```

**内置 Observer**（Phase 4 实现）：

```python
class QSNRObserver(SliceAwareObserver):
    """QSNR = 10 * log10(||fp32||² / ||fp32 - quant||²)"""
    def _measure(self, key, f, q):
        err = f - q
        num = f.pow(2).mean()
        den = err.pow(2).mean().clamp_min(1e-30)
        return {"qsnr_db": (10 * torch.log10(num / den)).item()}


class MSEObserver(SliceAwareObserver):
    def _measure(self, key, f, q):
        return {"mse": (f - q).pow(2).mean().item()}


class HistogramObserver(SliceAwareObserver):
    """收集 fp32 和量化张量的直方图，用于分布对比。"""
    def __init__(self, n_bins: int = 128):
        super().__init__()
        self.n_bins = n_bins

    def _measure(self, key, f, q):
        return {
            "fp32_hist":  torch.histc(f, bins=self.n_bins).cpu(),
            "quant_hist": torch.histc(q, bins=self.n_bins).cpu(),
            "err_hist":   torch.histc(f - q, bins=self.n_bins).cpu(),
        }
```

---

### AnalysisContext（上下文管理器）

```python
# src/analysis/context.py

class AnalysisContext:
    def __init__(self, model: nn.Module, observers: list[ObserverBase] = None):
        self.model = model
        self.observers = observers or [QSNRObserver()]

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ObservableMixin):
                module._observers = self.observers
                module._analysis_name = name
        return self

    def __exit__(self, *args):
        for module in self.model.modules():
            if isinstance(module, ObservableMixin):
                module._observers = []

    def report(self) -> dict:
        result = {}
        for obs in self.observers:
            for layer, role_map in obs.report().items():
                result.setdefault(layer, {}).update(role_map)
        return result
```

Phase 3 只定义 `ObservableMixin` + `QuantEvent` + `ObserverBase` + `SliceAwareObserver` + `iter_slices`（骨架）。**`AnalysisContext` 和具体 Observer 实现延后到 Phase 4**。

---

## 使用示例（Phase 4 落地后）

```python
from src.analysis import AnalysisContext, QSNRObserver, MSEObserver, HistogramObserver

model = MyQuantizedModel(...)
calibration_data = ...

with AnalysisContext(model, [QSNRObserver(), MSEObserver()]) as ctx:
    for batch in calibration_data:
        model(batch)

report = ctx.report()
# {
#   "layer1.linear": {
#       "input": {
#           "input_pre_quant[0]": {           # pipeline step 0: bf16 cast
#               ("tensor",): {"qsnr_db": 52.1, "mse": 3e-7},
#           },
#           "input_pre_quant[1]": {           # pipeline step 1: MX block
#               ("block", 0): {"qsnr_db": 38.2, "mse": 1.2e-5},
#               ("block", 1): {"qsnr_db": 41.0, "mse": 8.0e-6},
#               ...
#           },
#       },
#       "weight": { ... },
#   },
#   "layer2.conv": { ... },
# }
```

---

## 可扩展性分析

| 新需求 | 做什么 | 不做什么 |
|---|---|---|
| 新指标（如 SQNR、KL 散度） | 新 `SliceAwareObserver` 子类实现 `_measure` | `iter_slices` / `ObservableMixin` / 算子代码 |
| 新 granularity（动态分组、双层块） | `GranularitySpec` 加 mode；`iter_slices` 加分支；可能需要 `event.group_map` | Observer 子类 |
| 新事件点（如中间累加器值） | `ObservableMixin._emit` 调用点增加 + stage 枚举扩展 | Observer 骨架 |
| 全模型报告聚合 | Phase 4 `AnalysisContext.report()` | 底层 Observer |

---

## 被拒绝的方案

**方案 A：Shadow Forward（侵入式，在 forward 里同时跑 fp32 和量化两条路径）**
拒绝原因：需要对每个算子的 forward 做侵入式修改，且 shadow forward 的 fp32 权重需要单独保存（内存翻倍），耦合度高。

**方案 B：register_forward_hook（低侵入但能力受限）**
拒绝原因：`forward_hook` 只能拿到最终输入/输出，拿不到中间量化点（如 weight quant）的状态，也拿不到 QuantScheme 配置，信息不足以做深度 analysis。

**方案 C：继承同一基类，把 analysis 放在量化算子基类里**
拒绝原因：算子基类职责爆炸（量化 + analysis），且 analysis 指标的定义会随时间扩展，不适合放在算子层。

**方案 D：让每个 Observer 自己切分 granularity**
拒绝原因：每个 Observer 重复切分逻辑；新 granularity 要改所有 Observer。`iter_slices` 单入口是正解。
