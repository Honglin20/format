# ADR-002: 层级误差分析 — Observer 模式

**状态**: 已决策  
**日期**: 2026-04-24

---

## 背景与问题

`mx/` 库通过继承 `torch.nn.Linear` 等模块并覆写 `forward` 的方式替换 torch 算子，天然能看到每个算子的输入。目标是利用这一特性，对每个量化算子做 FP32 vs 量化路径的误差对比（QSNR / MSE），形成层级分析报告。

**设计约束**：
- 量化算子代码（`src/ops/`）不能包含 analysis 逻辑，否则职责爆炸
- 关闭 analysis 时必须零开销
- 未来需要支持多种指标（QSNR、MSE、直方图、分布统计）且易于扩展

## 决策

采用 **Observer 模式 + 上下文管理器**：
- 量化算子在关键点**发出事件**（emit），不做任何计算
- 外部 `Observer` 实现具体指标计算
- `AnalysisContext` 上下文管理器控制挂载/卸载，离开时自动清理

---

## 接口规范

### 事件系统

```python
# src/analysis/events.py

@dataclass
class QuantEvent:
    layer_name: str         # 模型中的模块路径，如 "model.layer1.linear"
    stage: str              # 事件点名称（见下表）
    fp32_tensor: Tensor     # 量化前的 FP32 张量（只读视图）
    quant_tensor: Tensor    # 量化后的张量（只读视图）
    scheme: QuantScheme     # 触发此事件的量化配置
```

**标准事件点**（stage 名称）：

| stage | 含义 | 触发位置 |
|---|---|---|
| `input_pre_quant` | 激活量化前后 | 算子 forward 开头 |
| `weight_pre_quant` | 权重量化前后 | 算子 forward 中权重量化处 |
| `output_post_quant` | 输出量化前后 | 算子 forward 末尾（如有输出量化） |

### 量化算子侧（事件发出）

量化算子继承一个轻量 `ObservableMixin`：

```python
# src/analysis/mixin.py

class ObservableMixin:
    """量化算子继承此 Mixin 获得事件发出能力。零开销（无 observer 时不执行）。"""
    _observers: list = []   # 类级别，由 AnalysisContext 管理

    def _emit(self, stage: str, fp32: Tensor, quant: Tensor, scheme: QuantScheme):
        if not self._observers:   # 无 observer → 直接返回，零开销
            return
        event = QuantEvent(
            layer_name=getattr(self, "_analysis_name", type(self).__name__),
            stage=stage,
            fp32_tensor=fp32.detach(),
            quant_tensor=quant.detach(),
            scheme=scheme,
        )
        for obs in self._observers:
            obs.on_event(event)
```

**算子 forward 中的用法**：
```python
class QuantizedLinear(ObservableMixin, nn.Linear):
    def forward(self, x):
        x_q = quantize(x, self.input_scheme)
        self._emit("input_pre_quant", fp32=x, quant=x_q, scheme=self.input_scheme)

        w_q = quantize(self.weight, self.weight_scheme)
        self._emit("weight_pre_quant", fp32=self.weight, quant=w_q, scheme=self.weight_scheme)

        return F.linear(x_q, w_q, self.bias)
```

### Observer 接口

```python
# src/analysis/observer.py

class ObserverBase(ABC):
    @abstractmethod
    def on_event(self, event: QuantEvent) -> None:
        """接收一个量化事件，做指标累积。"""

    def report(self) -> dict:
        """返回 {layer_name: {stage: metric_value}} 格式的报告。"""
        return {}

    def reset(self):
        """清空累积状态。"""
```

**内置 Observer**：

```python
class QSNRObserver(ObserverBase):
    """计算每层每事件点的量化信噪比（QSNR = 10 * log10(||fp32||² / ||fp32 - quant||²)）。"""

class MSEObserver(ObserverBase):
    """计算每层每事件点的均方误差。"""

class HistogramObserver(ObserverBase):
    """收集 fp32 和 quant 张量的直方图，用于分布对比。"""
```

### AnalysisContext（上下文管理器）

```python
# src/analysis/context.py

class AnalysisContext:
    def __init__(self, model: nn.Module, observers: list[ObserverBase] = None):
        self.model = model
        self.observers = observers or [QSNRObserver()]

    def __enter__(self):
        # 给所有 ObservableMixin 模块注册 observers
        # 给每个模块设置 _analysis_name（完整路径，如 "layer1.0.linear"）
        for name, module in self.model.named_modules():
            if isinstance(module, ObservableMixin):
                module._observers = self.observers
                module._analysis_name = name
        return self

    def __exit__(self, *args):
        # 卸载 observers，恢复零开销状态
        for module in self.model.modules():
            if isinstance(module, ObservableMixin):
                module._observers = []

    def report(self) -> dict:
        """合并所有 observer 的报告。"""
        result = {}
        for obs in self.observers:
            for layer, metrics in obs.report().items():
                result.setdefault(layer, {}).update(metrics)
        return result
```

---

## 使用示例

```python
from src.analysis import AnalysisContext, QSNRObserver, MSEObserver

model = MyQuantizedModel(...)
calibration_data = ...

observers = [QSNRObserver(), MSEObserver()]
with AnalysisContext(model, observers) as ctx:
    for batch in calibration_data:
        model(batch)

report = ctx.report()
# {
#   "layer1.linear": {
#       "input_pre_quant_qsnr": 38.2,
#       "weight_pre_quant_qsnr": 45.1,
#       "input_pre_quant_mse": 1.2e-5,
#   },
#   "layer2.conv": { ... },
# }
```

---

## 被拒绝的方案

**方案 A：Shadow Forward（侵入式，在 forward 里同时跑 fp32 和量化两条路径）**
拒绝原因：需要对每个算子的 forward 做侵入式修改，且 shadow forward 的 fp32 权重需要单独保存（内存翻倍），耦合度高。

**方案 C：register_forward_hook（低侵入但能力受限）**
拒绝原因：`forward_hook` 只能拿到最终输入/输出，拿不到中间量化点（如 weight quant）的状态，也拿不到 QuantScheme 配置，信息不足以做深度 analysis。

**方案：继承同一基类，把 analysis 放在量化算子基类里**
拒绝原因：算子基类职责爆炸（量化 + analysis），且 analysis 指标的定义会随时间扩展，不适合放在算子层。
