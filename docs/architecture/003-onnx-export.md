# ADR-003: ONNX Export 策略

**状态**: 已决策  
**日期**: 2026-04-24

---

## 背景与问题

需要将量化模型（`src/ops/` 中的量化算子）导出为 ONNX 格式，用于模型交付、可视化和后续 runtime 对接。

**约束**：
- 目标：能导出 ONNX、图结构正确、`netron` 可视化清晰
- 不要求：ORT / TRT runtime 可执行推理（留给未来 Phase）
- 量化格式分两类：标准（int8/fp8）和 MX/NVFP4（非标准）

## 决策

**混合导出策略**：
- 标准格式 → ONNX 官方 `QuantizeLinear` / `DequantizeLinear`（QDQ）
- 非标准格式（MX系列、NVFP4）→ 自定义 domain `com.microxscaling.*`

---

## 格式分类

| 格式 | 类别 | 导出方式 |
|---|---|---|
| `int8`, `int4` | 标准 | ONNX QDQ（`QuantizeLinear` / `DequantizeLinear`） |
| `fp8_e4m3`, `fp8_e5m2` | 标准（ONNX >= 1.15） | ONNX QDQ（float8 variant） |
| `fp4_e2m1`（per-block/MX） | 非标准 | `com.microxscaling.MxQuantize` |
| `fp6_*`（per-block/MX） | 非标准 | `com.microxscaling.MxQuantize` |
| `int*`（per-block/MX） | 非标准 | `com.microxscaling.MxQuantize` |
| 未来 `nf4`, `apot` | 非标准 | `com.microxscaling.CustomQuantize` |

---

## 实现机制

### autograd.Function 的 symbolic 方法

每个量化的 `autograd.Function` 需实现 `symbolic()` 静态方法：

```python
# src/onnx/functions.py

class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scheme):
        return quantize(x, scheme)

    @staticmethod
    def symbolic(g, x, scheme):
        if _is_standard_format(scheme):
            # 标准格式：走 ONNX QDQ
            scale, zero_point = _compute_qdq_params(scheme)
            x_q = g.op("QuantizeLinear", x, scale, zero_point,
                       axis_i=_get_axis(scheme))
            return g.op("DequantizeLinear", x_q, scale, zero_point,
                        axis_i=_get_axis(scheme))
        else:
            # 非标准格式：走自定义 domain
            return g.op(
                "com.microxscaling::MxQuantize",
                x,
                elem_format_s=scheme.format.name,
                block_size_i=scheme.granularity.block_size,
                round_mode_s=scheme.round_mode,
            )
```

### 算子级导出

量化算子（`QuantizedLinear` 等）的导出通过 `torch.onnx.export` 自动触发各子算子的 `symbolic`：

```python
# src/onnx/export.py

def export_quantized_model(
    model: nn.Module,
    dummy_input: Tensor,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """导出量化模型到 ONNX。"""
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        custom_opsets={"com.microxscaling": 1},
        do_constant_folding=True,
    )
    _verify_onnx_graph(output_path)   # 验证图结构合法

def _verify_onnx_graph(path: str) -> None:
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)   # 会跳过未知 domain 的 custom op
```

### 自定义 Domain 算子定义

在 `src/onnx/custom_ops.py` 中声明自定义算子的 schema（供 `netron` 和工具链识别）：

```python
# com.microxscaling domain, opset version 1

CUSTOM_OPS = {
    "MxQuantize": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "elem_format": str,   # 格式名，如 "fp4_e2m1"
            "block_size": int,    # 块大小，如 32
            "round_mode": str,    # "nearest" | "floor" | "even"
        },
        "doc": "MX-style block quantization. Y = dequant(quant(X, elem_format, block_size)).",
    },
    "CustomQuantize": {
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": {
            "format_name": str,
            "granularity_mode": str,
            "block_size": int,
            "channel_axis": int,
            "round_mode": str,
            "transform_name": str,
        },
        "doc": "Generic quantize op for non-standard formats (nf4, apot, etc.).",
    },
}
```

---

## 验证方案

```python
# src/tests/test_onnx_export.py 的验证策略

def test_export_loads_without_error(model, dummy_input, tmp_path):
    """图能被 onnx.checker 验证（custom op 跳过语义检查）。"""

def test_export_node_types(onnx_model, expected_custom_ops):
    """图中出现预期的节点类型（MxQuantize / QuantizeLinear 等）。"""

def test_standard_format_uses_qdq(onnx_model):
    """INT8/FP8 量化算子导出为标准 QDQ 节点。"""

def test_mx_format_uses_custom_op(onnx_model):
    """MX 格式量化算子导出为 com.microxscaling.MxQuantize。"""
```

---

## 判断标准（Phase 5 完成条件）

- [ ] `export_quantized_model()` 对 QuantizedLinear + QuantizedConv2d 不报错
- [ ] `onnx.checker.check_model()` 通过
- [ ] `netron` 打开 .onnx 文件，图结构可读、节点有正确属性
- [ ] 标准格式走 QDQ，MX 格式走 `com.microxscaling.MxQuantize`
- [ ] `pytest src/tests/test_onnx_export.py` 全绿

## 不在当前范围内

- ORT custom op 实现（让 ORT 真正执行自定义 op）
- TensorRT plugin
- ONNX Runtime 推理精度验证
