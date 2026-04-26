# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建，在 `src/` 中提供高扩展性的张量级量化库。`mx/` 保留为只读参考，所有新代码在 `src/` 中。

## 核心设计

量化方案由三轴组合描述：

```
QuantScheme = format × granularity × transform
```

| 轴 | 描述 | 示例 |
|---|---|---|
| `format` | 数值格式（int8 / fp8 / fp4 / bf16 等） | `FormatBase.from_str("fp4_e2m1")` |
| `granularity` | 量化粒度（per_tensor / per_channel / per_block） | `GranularitySpec.per_block(32)` |
| `transform` | 量化前后变换（默认 Identity） | `IdentityTransform()` |

算子级配置由 `OpQuantConfig` 描述，每个 tensor 角色接一个 scheme pipeline：

```python
OpQuantConfig(input=(s,), weight=(s,), output=(s,))
```

---

## 安装

```bash
pip install -r requirements.txt
```

---

## 快速上手

### 1. 定义量化方案

```python
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig

# int8 per-tensor
fmt = FormatBase.from_str("int8")
scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())

# fp4 MX block（MX 格式，block_size=32）
mx_fmt = FormatBase.from_str("fp4_e2m1")
mx_scheme = QuantScheme(format=mx_fmt, granularity=GranularitySpec.per_block(32))

# 算子级配置
cfg = OpQuantConfig(input=(scheme,), weight=(scheme,), output=(scheme,))
```

### 2. 使用量化算子

所有量化算子都是对应 `nn.Module` 的直接替换：

```python
from src.ops.linear import QuantizedLinear
from src.ops.conv import QuantizedConv2d
from src.ops.norm import QuantizedLayerNorm
from src.ops.activations import QuantizedGELU
from src.ops.softmax import QuantizedSoftmax

# 替换 nn.Linear
linear = QuantizedLinear(in_features=768, out_features=768, cfg=cfg)

# 替换 nn.Conv2d
conv = QuantizedConv2d(in_channels=3, out_channels=64, kernel_size=3, cfg=cfg)

# 替换 nn.LayerNorm
norm = QuantizedLayerNorm(normalized_shape=[768], cfg=cfg)
```

### 3. 替换整个模型（quantize_model）

```python
from src.mapping.quantize_model import quantize_model

model = MyModel()

# 全模型统一 cfg
quantize_model(model, cfg=cfg)

# 或按名称 / 通配符分层配置
quantize_model(model, cfg={
    "encoder.*":  OpQuantConfig(input=(int8_scheme,), weight=(int8_scheme,), output=(int8_scheme,)),
    "decoder.*":  OpQuantConfig(input=(fp4_scheme,),  weight=(fp4_scheme,),  output=(fp4_scheme,)),
})
```

### 4. 直接量化张量

```python
from src.quantize import quantize

x_q = quantize(x, scheme)
```

---

## 支持的算子

| 类别 | 算子 |
|---|---|
| 线性 | `QuantizedLinear`, `quantized_matmul`, `quantized_bmm` |
| 卷积 | `QuantizedConv1d/2d/3d`, `QuantizedConvTranspose1d/2d/3d` |
| 归一化 | `QuantizedBatchNorm1d/2d/3d`, `QuantizedLayerNorm`, `QuantizedGroupNorm`, `QuantizedRMSNorm` |
| 激活 | `QuantizedReLU`, `QuantizedReLU6`, `QuantizedGELU`, `QuantizedSiLU`, `QuantizedSigmoid`, `QuantizedTanh`, `QuantizedLeakyReLU` |
| 注意力 | `QuantizedSoftmax` |
| 池化 | `QuantizedAdaptiveAvgPool2d` |
| Elementwise | `quantized_add`, `quantized_mul`, `quantized_div`, `quantized_sub`, `quantized_exp`, `quantized_sqrt` 等 |

---

## 支持的数值格式

| 格式名 | 类型 | 备注 |
|---|---|---|
| `int8` / `int4` / `int2` | 整数 | 标准量化 |
| `fp8_e4m3` / `fp8_e5m2` | 浮点 8 位 | OCP FP8 规范 |
| `fp4_e2m1` | 浮点 4 位 | MX FP4 |
| `fp6_e3m2` / `fp6_e2m3` | 浮点 6 位 | MX FP6 |
| `bf16` / `fp16` | 16 位浮点 | 硬件快捷路径 |

---

## 误差分析（Phase 4）

用 `AnalysisContext` 挂载 observer，无侵入地采集量化误差：

```python
from src.analysis.context import AnalysisContext
from src.analysis.observers import QSNRObserver, MSEObserver, HistogramObserver

with AnalysisContext(model, observers=[QSNRObserver(), MSEObserver()]) as ctx:
    for batch in calibration_data:
        model(batch)

report = ctx.report()
print(report)
```

支持的 observer：

| Observer | 指标 |
|---|---|
| `QSNRObserver` | 量化信噪比（dB） |
| `MSEObserver` | 均方误差 |
| `HistogramObserver` | fp32 / quant 值分布直方图 |
| `DistributionObserver` | 均值、方差、偏度、峰度、稀疏度等统计指纹 |

粒度感知：observer 自动按 scheme 的 granularity（per_tensor / per_channel / per_block）切片聚合。

---

## ONNX 导出（Phase 5）

```python
from src.onnx import export_quantized_model
import torch

model = QuantizedLinear(768, 768, cfg=cfg)
export_quantized_model(model, torch.randn(1, 768), "model.onnx")
```

导出规则：

| 量化方案 | ONNX 节点 |
|---|---|
| int8/int4/int2/fp8 + per_tensor/per_channel | `QuantizeLinear` / `DequantizeLinear`（标准 QDQ） |
| 任意格式 + per_block（MX block） | `com.microxscaling::MxQuantize`（自定义 domain） |

导出目标：图结构正确 + `onnx.checker` 通过。Scale 为占位常量（1.0），不保证 runtime 可推理。

---

## 量化训练（QAT）

`OpQuantConfig` 的 backward 字段非空时自动启用 QAT：

```python
cfg_qat = OpQuantConfig(
    input=(scheme,), weight=(scheme,), output=(scheme,),
    grad_output=(scheme,), grad_input=(scheme,), grad_weight=(scheme,),
)
linear = QuantizedLinear(768, 768, cfg=cfg_qat)
# forward + backward 全程量化，与 mx/ bit-exact 等价
```

---

## 与 mx/ 的等价性

`src/` 中所有算子与 `mx/` **bit-exact 等价**（`torch.equal`，不允许误差容忍）。可通过适配器从旧 `MxSpecs` dict 构造 `OpQuantConfig`：

```python
from src.scheme._compat import op_config_from_mx_specs

mx_specs = {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", ...}
cfg = op_config_from_mx_specs(mx_specs)
```

---

## 项目结构

```
src/
├── formats/        # FormatBase 及各格式实现（int / fp / bf16 等）
├── scheme/         # QuantScheme、GranularitySpec、OpQuantConfig
├── quantize/       # 核心量化函数（elemwise / mx_quantize）
├── ops/            # 量化算子（Linear / Conv / Norm / Activation 等）
├── analysis/       # 误差分析框架（AnalysisContext / Observer / iter_slices）
├── mapping/        # quantize_model 模型替换入口
└── onnx/           # ONNX 导出（export_quantized_model）

mx/                 # 原始 microsoft/microxcaling（只读，等价性参考）
```

---

## 测试

```bash
pytest src/tests/ -q
# 973 passed
```

所有等价性测试严格使用 `torch.equal`（bit-exact），不允许 atol/rtol 宽松匹配。
