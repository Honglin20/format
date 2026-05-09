# 自定义格式

## 快速注册

```python
from src.formats import register_float_format, register_int_format

# 注册浮点格式
register_float_format("fp5_e3m1", ebits=3, mbits=1)

# 注册整数格式
register_int_format("int6", bits=6)
```

注册后即可在 QuantConfig 中使用：

```python
cfg = QuantConfig(w_format="fp5_e3m1")
cfg = QuantConfig(w_format="int6")
```

## 自动解析

任何 `fp<N>_e<E>m<M>` 或 `int<N>` 字符串在首次使用时自动注册，无需手动调用。

## 注册 FormatBase 子类

```python
from src.formats import register_format
from src.formats.base import FormatBase

class MyCustomFormat(FormatBase):
    def quantize(self, x, granularity, round_mode="nearest", allow_denorm=True):
        ...

    def dequantize(self, x_q, scale, granularity):
        ...

    @property
    def name(self):
        return "my_format"

    @property
    def bit_width(self):
        return 8

register_format("my_format", MyCustomFormat())

# 使用
cfg = QuantConfig(w_format="my_format")
```

## FormatBase 必须实现的接口

| 方法/属性 | 说明 |
|-----------|------|
| `quantize(x, granularity, round_mode, allow_denorm)` | 量化：fp32 → quantized |
| `dequantize(x_q, scale, granularity)` | 反量化：quantized → fp32 |
| `name` (property) | 格式名 |
| `bit_width` (property) | 位宽 |

可选实现：

| 方法 | 说明 |
|------|------|
| `export_onnx(g, x, scheme)` | ONNX 导出支持 |
