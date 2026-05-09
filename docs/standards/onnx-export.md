# ONNX Export 接入规范

## 架构位置

ONNX 相关代码位于 `src/onnx/`，接口定义见 `docs/architecture/003-onnx-export.md`。

## 核心模式

### Format 自声明 ONNX 行为（Strategy 模式）

`_emit_quantize_node(g, x, scheme)` 委托给 `scheme.format.export_onnx(g, x, scheme)` — 每个 Format 自声明 ONNX 导出行为，与 `quantize()` 对称。

| Format | export_onnx 行为 |
|--------|-----------------|
| `IntFormat` | 标准 QDQ（`QuantizeLinear`/`DequantizeLinear`），非 PER_BLOCK |
| `FPFormat` | QDQ for `fp8_e4m3`/`fp8_e5m2` + non-PER_BLOCK；其余走 `MxQuantize` |
| `FormatBase`（默认） | `com.microxscaling::MxQuantize`（自定义 domain） |
| 自定义 | 覆写 `export_onnx(self, g, x, scheme)` |

### JIT Tracing Guard

`FormatBase._quantize_per_block()` 在 `torch.jit.is_tracing()` 时直接 `return x`，跳过 `_reshape_to_blocks`。symbolic() 负责生成真实的 ONNX 量化节点。

两阶段分离：forward() for shape inference，symbolic() for ONNX graph。

## 新增需导出 ONNX 的模块时

### 1. 为 autograd.Function 添加 symbolic()

```python
@staticmethod
def symbolic(g, x, weight, bias, ...):
    # 生成 ONNX 子图
    return g.op("com.microsoft::MxLinear", x, weight, bias, ...)
```

### 2. Format 需覆写 export_onnx()

如果需要不同于默认 `MxQuantize` 的行为，覆写 `export_onnx(self, g, x, scheme)`。

不要在 helpers.py 中硬编码 format name 分派。

### 3. 测试

- 图结构正确（节点类型、连接关系）
- `onnx.checker` 通过
- `netron` 可可视化

不要求 ORT 可执行推理。

## 约束

- 每个量化 `autograd.Function` 必须提供 `symbolic()` 方法
- 目标：`onnx.checker` 通过，`netron` 能可视化
- 不在当前范围：ORT runtime 推理
