# 性能估算 (Cost Model)

使用 Roofline 模型估计量化模型的延迟和内存占用。

## Session 内置

```python
session = Session(model, QuantConfig(w_format="int8")).quantize().cost()

# 查看量化模型成本
print(session.result.cost.print_summary())

# fp32 参考模型成本
print(session.result.cost_fp32.print_summary())
```

## 独立调用

```python
from src.cost import DeviceSpec, analyze_model_cost

# 用 A100 参数估计
fp32_cost = analyze_model_cost(model, {"x": (1, 128)}, DeviceSpec.a100())
```

## DeviceSpec

`DeviceSpec` 包含硬件参数：峰值算力、内存带宽、SRAM 大小等。内置 `.a100()` 工厂方法。
