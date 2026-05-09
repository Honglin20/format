# MX 位精确等价

本框架与 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的 MX 推理输出**位精确等价**（`torch.equal`），已通过全算子验证。

## 验证示例

```python
import torch, torch.nn as nn
import mx
from mx.specs import apply_mx_specs
from src.session import Session, QuantConfig

linear = nn.Linear(64, 128, bias=True).eval()
x = torch.randn(4, 64)

# 本框架
cfg = QuantConfig(name="w8a8", w_format="int8", w_granularity="per_block",
                  w_block_size=32, a_format="int8", a_granularity="per_block",
                  a_block_size=32, storage_bits=16, storage_kind="bfloat")
session_out = Session(linear, cfg).quantize()(x)

# 微软 MX
mx_specs = apply_mx_specs({"bfloat": 16, "w_elem_format": "int8",
                            "a_elem_format": "int8", "block_size": 32})
mx_out = mx.linear(x, linear.weight, linear.bias, mx_specs=mx_specs)

assert torch.equal(session_out, mx_out)  # bit-exact
```

## 覆盖范围

- **21 种模块** + **10 种 inline op**（matmul、add、softmax 等）
- **8 种格式**（int8/int4/int2/fp8_e4m3/fp8_e5m2/fp6_e3m2/fp6_e2m3/fp4_e2m1）
- **3 种 storage**（bfloat16 / float16 / disabled）
- **forward + backward**（STE 梯度对齐，4/5 规格 bit-exact）
