# ONNX Export Refactor — Design

**Date**: 2026-05-09
**Branch**: `feature/refactor-src`

## Motivation

Current ONNX export has five gaps:

1. `dummy_input` only accepts single `Tensor` — no `list`/`dict`/`tuple` for multi-input models
2. `_last_input` only records `args[0]` — multi-input models lose inputs
3. NF4 has no custom `export_onnx()` — falls back to `MxQuantize` (wrong semantics)
4. QDQ scale values are placeholder 1.0 — calibration data not wired into export
5. No systematic format-coverage test — only int8/fp4 linear/conv tested

## Design

### 1. Input type generalization

```
dummy_input: Tensor → Union[Tensor, tuple, list, dict]
```

- `_QuantSession.__call__`: record full `args` (single arg stored as-is, 2+ stored as tuple)
- `export_onnx()` everywhere: wrap non-tuple into `(inp,)` before `torch.onnx.export`
- `torch.onnx.export` natively supports `tuple`/`list`/`dict` as `args`

Affected files:
- `src/onnx/export.py` — `export_quantized_model(dummy_input)`
- `src/session/_context.py` — `QuantizeContext.export_onnx(dummy_input)`
- `src/session/_model.py` — `_export_onnx(dummy_input)`
- `src/session/_quant.py` — `_QuantSession.export_onnx()` + `__call__`

### 2. NF4 ONNX export

`LookupFormat.export_onnx()` override emits `com.microxscaling::NF4Quantize`:

```
g.op("com.microxscaling::NF4Quantize", x, levels_f=[...])
```

Separates NF4 from MX block quantization. `levels_f` captures the full LUT.

Affected files:
- `src/formats/lookup_formats.py` — new `export_onnx` on `LookupFormat`

### 3. Calibration scale wiring

**Collection**: Before ONNX export, iterate `qmodel.named_modules()`, collect `_input_scale`/`_weight_scale` buffers into `QuantizeContext._export_scales: Dict[str, Tensor]`.

**Transmission**: Thread `name` parameter through:
```
symbolic(name) → _emit_quantize_node(name=...) → format.export_onnx(name=...)
```

**Emission**: `IntFormat.export_onnx` / `FPFormat.export_onnx` check `_export_scales` via active `QuantizeContext`. Found → real scale `Constant`; not found → fallback `Constant(1.0)`.

Affected files:
- `src/session/_context.py` — `_export_scales` dict + `_collect_export_scales()`
- `src/onnx/helpers.py` — `_emit_quantize_node(name=...)`
- `src/formats/base.py` — `export_onnx(name=None)`
- `src/formats/int_formats.py` — `export_onnx(name=None)`
- `src/formats/fp_formats.py` — `export_onnx(name=None)`
- `src/formats/lookup_formats.py` — `export_onnx(name=None)`
- `src/ops/linear.py` — `symbolic(..., name, ...)` pass name
- `src/ops/conv.py` — same
- `src/ops/matmul.py` — same
- `src/ops/bmm.py` — same

### 4. Format coverage test matrix

```
formats:  int8, int4, int2, fp8_e4m3, fp8_e5m2, fp4_e2m1, fp6_e3m2,
          fp6_e2m3, nf4, bf16, fp16
granularities: per_tensor, per_block(32)
ops:      linear, conv2d
inputs:   single tensor, tuple[Tensor,Tensor], list[Tensor], dict[str,Tensor]
special:  passthrough (no cfg), multi-stage (quantize→calibrate→export)
```

Each parametrized test validates:
- `onnx.checker.check_model()` passes
- Correct op emitted (QDQ / MxQuantize / NF4Quantize)
- Correct node count (no missing or duplicate quantize wrappers)

Affected files:
- `src/tests/test_onnx_export.py` — full rewrite with parametrized tests

### 5. API cleanup

- Update `export_quantized_model()` docstring — remove outdated placeholder note
- Align type annotations with `Union[Tensor, tuple, list, dict]`

## Non-goals

- Runtime inference correctness (scales are real but rounding/saturation may differ)
- Dynamic shapes / batch-dim annotation
- ONNX IR version upgrade

## Test plan

1. Unit: `_is_standard_format` × all format+grain combos
2. Unit: Each format's `export_onnx()` node type
3. Integration: `export_quantized_model()` with multi-input types
4. Integration: `Session.quantize().calibrate().export_onnx()` end-to-end
5. Integration: NF4 format ONNX checker passes
6. Regression: existing 2,093 tests unchanged
