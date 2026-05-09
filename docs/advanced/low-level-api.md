# 底层 API

当你需要将量化集成到自己的训练/推理脚本中，而不是走完整的 Session 管道时，使用底层 API。

## quantize_model：只做模块替换

```python
import torch, torch.nn as nn
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.session import quantize_model

model = nn.Linear(128, 256)

# 构建底层配置
scheme = QuantScheme.per_channel("int8", axis=0)
cfg = OpQuantConfig(weight=scheme)

# 替换模块
qmodel = quantize_model(model, cfg)
output = qmodel(torch.randn(4, 128))  # 自动走量化路径
qmodel.export_onnx(torch.randn(1, 128), "model.onnx")
```

## quantize() 数学层入口

```python
from src.quantize.elemwise import quantize
from src.scheme.quant_scheme import QuantScheme

scheme = QuantScheme.per_channel("int8", axis=0)
x_q = quantize(x, scheme)
# 内部三步：
#   1. x_t = scheme.transform.forward(x)
#   2. x_q = scheme.format.quantize(x_t, ...)
#   3. x_q = scheme.transform.inverse(x_q)
```

## OpQuantConfig：算子级逐角色配置

为每个 tensor 角色单独指定 QuantScheme：

```python
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme

w_scheme = QuantScheme.per_channel("int4", axis=0)
a_scheme = QuantScheme.per_tensor("int8")

# 只量化 input 和 weight
cfg = OpQuantConfig(input=a_scheme, weight=w_scheme)
# storage / output / grad_* 全部为 None（不量化）

# 带 storage 的两级量化
from src.formats.bf16_fp16 import BFloat16Format
storage = QuantScheme.per_tensor(BFloat16Format())
cfg = OpQuantConfig(input=a_scheme, weight=w_scheme, storage=storage)
```

## OpQuantConfig 完整角色

| 角色 | 说明 |
|------|------|
| `input` | 输入激活 |
| `weight` | 权重 |
| `output` | 输出激活 |
| `storage` | Element-wise storage（两级量化） |
| `grad_input` | 输入梯度 |
| `grad_weight` | 权重梯度 |
| `input_gw` | 中间变量（SQ 平滑后的输入） |
| `weight_gw` | 中间变量（SQ 平滑后的权重） |

## QuantScheme：方案工厂

```python
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.formats.int_formats import IntFormat

# per_tensor
scheme = QuantScheme.per_tensor("int8")
scheme = QuantScheme.per_tensor(IntFormat(bits=8))

# per_channel
scheme = QuantScheme.per_channel("int8", axis=0)

# per_block
scheme = QuantScheme.per_block("fp4_e2m1", block_size=32)

# 完整构造
scheme = QuantScheme(
    format=IntFormat(bits=8),
    granularity=GranularitySpec.per_channel(axis=0),
    transform=IdentityTransform(),
    round_mode="nearest",
    scale_storage="fp32",
)
```

## 三种入口对比

| 入口 | 输入 | 适用场景 |
|------|------|---------|
| `Session(model, QuantConfig)` | 字符串字段 | 日常量化实验 |
| `Study(configs, model)` | `List[QuantConfig]` | 多配置对比 |
| `quantize_model(model, OpQuantConfig)` | 对象 | 集成到自定义脚本 |
