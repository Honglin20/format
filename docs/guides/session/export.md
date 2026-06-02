# ONNX 导出

> 第 8 章 · [Session 文档索引](INDEX.md)

量化后的模型可通过 Session 一键导出为 ONNX 图。

**前提**：已阅读 [第 1 章 · Session 概览](overview.md)，完成 `quantize()`。

## Session 导出

```python
session = Session(model, cfg).quantize(calib_data=calib_data)

# 一行导出
session.qmodel.export_onnx(torch.randn(1, 128), "model.onnx")
```

导出的 ONNX 图中：
- **int/fp8 格式** → 标准 `QuantizeLinear` / `DequantizeLinear` 节点
- **MX per_block 格式** → `com.microxscaling::MxQuantize` 自定义算子

默认 opset 17。

## 底层 API 导出

```python
from src.session import quantize_model
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme

scheme = QuantScheme.per_channel("int8", axis=0)
cfg = OpQuantConfig(weight=scheme)
qmodel = quantize_model(model, cfg)

qmodel.export_onnx(torch.randn(1, 128), "model.onnx")
```

## 注意事项

导出的 scale 是占位符常量（1.0），图结构有效但不可直接推理——实际推理时需要从量化模型中加载真实 scale。

---
← [上一章：Study 多配置对比](study.md) | [Session 文档索引](INDEX.md) | [下一章：性能估算](cost.md) →
