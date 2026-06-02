# 性能估算

> 第 9 章 · [Session 文档索引](INDEX.md)

使用 Roofline 模型估计量化模型的延迟和内存占用。

**前提**：已阅读 [第 1 章 · Session 概览](overview.md)。

## Session 内置

```python
session = Session(model, QuantConfig(w_format="int8"))
session.quantize(calib_data=calib_data)
session.cost()

# 查看量化模型成本
print(session.result.cost.print_summary())

# fp32 参考模型成本
print(session.result.cost_fp32.print_summary())
```

或在 `run()` 中一步完成：

```python
result = Session(model, cfg).run(calib_data, outputs=["cost"])
print(result.cost.print_summary())
```

## 独立调用

```python
from src.cost import DeviceSpec, analyze_model_cost

# 用 A100 参数估计
fp32_cost = analyze_model_cost(model, {"x": (1, 128)}, DeviceSpec.a100())
```

## DeviceSpec

`DeviceSpec` 包含硬件参数：峰值算力、内存带宽、SRAM 大小等。内置 `.a100()` 工厂方法。

## 相关图表

启用 cost 后，以下图表自动可用：

- `result.plot.cost_decomposition()` — FLOPs 分解堆叠柱状图
- `report.plot.pareto_frontier()` — 质量 vs 成本权衡散点图（需 Study）

---
← [上一章：ONNX 导出](export.md) | [Session 文档索引](INDEX.md) |
