# 格式选择

## 内置格式

所有格式通过注册名在 `QuantConfig(w_format=...)` 中使用：

| 格式 | 注册名 | 位宽 | 适用场景 |
|------|--------|------|---------|
| INT8 | `"int8"` | 8 | 通用，兼容性最好 |
| INT4 | `"int4"` | 4 | 高压缩比，需校准 |
| INT2 | `"int2"` | 2 | 极限压缩，精度损失大 |
| FP8 E4M3 | `"fp8_e4m3"` | 8 | OCP 标准，训练/推理通用 |
| FP8 E5M2 | `"fp8_e5m2"` | 8 | 更大动态范围 |
| FP6 E3M2 | `"fp6_e3m2"` | 6 | MX 规格 |
| FP6 E2M3 | `"fp6_e2m3"` | 6 | MX 规格，更高精度 |
| FP4 E2M1 | `"fp4_e2m1"` | 4 | MX 规格，极限压缩 |
| NF4 | `"nf4"` | 4 | QLoRA 正态优化 LUT |
| BF16 | `"bfloat16"` | 16 | 硬件捷径，几乎无损失 |
| FP16 | `"float16"` | 16 | IEEE 半精度 |

## 在 QuantConfig 中使用

```python
from src.session import Session, QuantConfig

# 只量化权重
cfg = QuantConfig(w_format="int8", weight_only=True)

# 权重 INT4 + 激活 INT8
cfg = QuantConfig(w_format="int4", a_format="int8")

# FP8 全量化
cfg = QuantConfig(w_format="fp8_e4m3", a_format="fp8_e4m3")

# NF4 权重
cfg = QuantConfig(w_format="nf4", weight_only=True)
```

## 自动解析

任何 `fp<N>_e<E>m<M>` 或 `int<N>` 字符串在首次使用时自动注册，无需手动调用：

```python
cfg = QuantConfig(w_format="fp5_e3m1")   # 自动注册
cfg = QuantConfig(w_format="int6")       # 自动注册
```

## 自定义格式

```python
from src.formats import register_format, register_float_format, register_int_format

# 注册自定义浮点格式
register_float_format("fp5_e3m1", ebits=3, mbits=1)

# 注册自定义整数格式
register_int_format("int6", bits=6)

# 注册任意 FormatBase 子类实例
from src.formats.base import FormatBase
register_format("my_format", MyFormatInstance)
```
