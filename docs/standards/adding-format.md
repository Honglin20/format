# 新增 Format 规范

## 架构位置

所有格式实现位于 `src/formats/`，接口定义见 `docs/architecture/001-three-axis-quant-scheme.md`。

## 步骤

### 1. 阅读 ADR

先读 `docs/architecture/001-three-axis-quant-scheme.md`，理解 QuantScheme 三轴设计。

### 2. 确定基类

| 格式类型 | 基类 |
|----------|------|
| 整数格式（int4/int8） | `IntFormat` |
| 浮点格式（fp8_e4m3/fp8_e5m2/mxfp4） | `FPFormat` |
| 查找表格式（NF4） | `LookupFormat` |
| 全新数值类型 | `FormatBase` |

### 3. 实现必须的方法

- `quantize(x, granularity, round_mode)` — 核心量化逻辑
- `dequantize(x_q, granularity)` — 反量化
- `export_onnx(g, x, scheme)` — ONNX 导出行为（可选，默认走 `MxQuantize`）

### 4. 注册

在 `src/formats/` 对应文件中注册，使其可通过字符串解析：

```python
register_float_format("fp8_e4m3", FP8E4M3Format)
```

### 5. 测试

- 等价性测试：`torch.equal(mx_output, src_output)`
- per-tensor / per-channel / per-block 三种 granularity
- 正向路径 + 负面测试（每个 raise 点一条 `pytest.raises`）
- Block 格式专项：整除/不整除/退化（见 `quantization-testing.md`）

### 6. 文档

- 在 `docs/verification/` 写推导文档
- 在 `docs/plans/` 中写实现计划（如果是新功能点）

## 接口契约

- `format` 必须是 `FormatBase` 子类的实例，不是字符串
- `FormatBase` 子类必须实现 `__eq__`/`__hash__`（通常通过 frozen dataclass 自动获得）
- 量化函数签名固定为：`quantize(x: Tensor, scheme: QuantScheme) -> Tensor`
