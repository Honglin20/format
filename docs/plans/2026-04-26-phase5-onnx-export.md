# Phase 5: ONNX Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export quantized models (`QuantizedLinear`, `QuantizedConv{1,2,3}d`) to valid ONNX graphs where int8 quantization steps appear as `QuantizeLinear`/`DequantizeLinear` QDQ nodes and MX/block-format steps appear as `com.microxscaling::MxQuantize` custom nodes.

**Architecture:** Add `symbolic()` static methods to `LinearFunction` and `ConvFunction`/`ConvTransposeFunction` — PyTorch's ONNX exporter calls these instead of tracing through `forward()`. A shared `_emit_quantize_node()` helper dispatches to either standard QDQ (for int/fp8 + per_tensor/per_channel) or the custom MX domain op (for everything else). `export_quantized_model()` in `src/onnx/export.py` wraps `torch.onnx.export` + `onnx.checker.check_model`. Goal: graph structure correct + `onnx.checker` passes. Not required: ORT runtime execution.

**Tech Stack:** Python 3.10+, PyTorch 2.2.2 (old-style TorchScript ONNX exporter, `symbolic()` API), onnx 1.21.0, dataclasses (QuantScheme/OpQuantConfig/GranularitySpec)

**Pre-verified design facts:**
- `symbolic()` on `autograd.Function` works in PyTorch 2.2.2 ✓
- `OpQuantConfig` frozen dataclass passes through to `symbolic()` as Python object ✓
- `None` bias comes through as Python `None` (not JIT value) → `if b is None:` check works ✓
- `g.op("Constant", value_t=torch.tensor(...))` + `QuantizeLinear`/`DequantizeLinear` pass `onnx.checker` ✓
- Current export WITHOUT `symbolic()` traces through raw arithmetic → 78 unreadable nodes ✗

---

### Task 1: `src/onnx/helpers.py` — quantize node emitter

**Files:**
- Create: `src/onnx/__init__.py`
- Create: `src/onnx/helpers.py`
- Create: `src/tests/test_onnx_export.py` (partial — helper tests only)

**Step 1: Write failing tests for helpers**

Create `src/tests/test_onnx_export.py`:

```python
"""
Phase 5 ONNX export tests.
All tests verify graph structure (node types, attributes), not runtime correctness.
"""
import io
import pytest
import torch
import onnx
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.op_config import OpQuantConfig
from src.ops.linear import QuantizedLinear
from src.ops.conv import QuantizedConv2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export(model, x):
    """Export model to ONNX in-memory; return loaded onnx.ModelProto."""
    buf = io.BytesIO()
    torch.onnx.export(
        model, (x,), buf,
        opset_version=17,
        custom_opsets={"com.microxscaling": 1},
    )
    buf.seek(0)
    return onnx.load(buf)


def _node_ops(onnx_model):
    """Return list of (domain, op_type) for every node (excluding Constant)."""
    return [
        (n.domain or "onnx", n.op_type)
        for n in onnx_model.graph.node
        if n.op_type != "Constant"
    ]


def _has_op(onnx_model, op_type, domain="onnx"):
    return any(n.op_type == op_type and (n.domain or "onnx") == domain
               for n in onnx_model.graph.node)


def _standard_cfg(fmt_name, granularity=None):
    fmt = FormatBase.from_str(fmt_name)
    gran = granularity or GranularitySpec.per_tensor()
    s = QuantScheme(format=fmt, granularity=gran)
    return OpQuantConfig(input=(s,), weight=(s,), output=(s,))


def _mx_cfg(fmt_name="fp4_e2m1", block_size=32):
    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_block(block_size)
    s = QuantScheme(format=fmt, granularity=gran)
    return OpQuantConfig(input=(s,), weight=(s,), output=(s,))


# ---------------------------------------------------------------------------
# Task 1: _is_standard_format / _emit_quantize_node unit tests
# ---------------------------------------------------------------------------

def test_is_standard_format_int8_per_tensor():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("int8"),
                    granularity=GranularitySpec.per_tensor())
    assert _is_standard_format(s) is True


def test_is_standard_format_int4_per_channel():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("int4"),
                    granularity=GranularitySpec.per_channel(axis=0))
    assert _is_standard_format(s) is True


def test_is_standard_format_fp8_per_tensor():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("fp8_e4m3"),
                    granularity=GranularitySpec.per_tensor())
    assert _is_standard_format(s) is True


def test_is_standard_format_fp4_per_block_is_false():
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("fp4_e2m1"),
                    granularity=GranularitySpec.per_block(32))
    assert _is_standard_format(s) is False


def test_is_standard_format_int8_per_block_is_false():
    """int8 with PER_BLOCK (MX style) is NOT standard — block quantization is custom."""
    from src.onnx.helpers import _is_standard_format
    s = QuantScheme(format=FormatBase.from_str("int8"),
                    granularity=GranularitySpec.per_block(32))
    assert _is_standard_format(s) is False
```

**Step 2: Run to verify failure**

```bash
pytest src/tests/test_onnx_export.py::test_is_standard_format_int8_per_tensor -v
```

Expected: `ModuleNotFoundError: No module named 'src.onnx'`

**Step 3: Create `src/onnx/__init__.py`**

```python
from .export import export_quantized_model
```

(This will fail until `export.py` exists — that's fine, the import order is fixed in Task 4.)

For now, create a minimal placeholder:

```python
# src/onnx/__init__.py
# Populated in Task 4
```

**Step 4: Create `src/onnx/helpers.py`**

```python
"""
ONNX export helper utilities.

_is_standard_format: int8/int4/int2/fp8 + non-PER_BLOCK → True (standard QDQ).
_emit_quantize_node: emit QuantizeLinear/DequantizeLinear or MxQuantize node.
"""
import torch
from src.scheme.granularity import GranularityMode

# Formats that map to ONNX standard QDQ nodes (opset 13+).
# Per-block variants of these formats are excluded (MX block style → custom op).
_STANDARD_NAMES = {"int8", "int4", "int2", "fp8_e4m3", "fp8_e5m2"}


def _is_standard_format(scheme) -> bool:
    """Return True if scheme should export as ONNX QDQ (QuantizeLinear/DequantizeLinear).

    Rules:
    - PER_BLOCK granularity → always False (MX block quantization → custom op)
    - int8/int4/int2/fp8_e4m3/fp8_e5m2 + per_tensor or per_channel → True
    - All other formats → False (custom op)
    """
    if scheme.granularity.mode == GranularityMode.PER_BLOCK:
        return False
    return scheme.format.name in _STANDARD_NAMES


def _emit_quantize_node(g, x, scheme):
    """Emit a quantize+dequantize pair in the ONNX graph for the given scheme.

    Standard formats → QuantizeLinear(x, scale=1.0, zp=0) → DequantizeLinear.
    Scale is a placeholder constant (1.0); not intended for runtime inference.

    Non-standard / MX formats → com.microxscaling::MxQuantize custom node
    with elem_format, block_size, round_mode attributes.
    """
    if _is_standard_format(scheme):
        scale = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
        zp = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
        xq = g.op("QuantizeLinear", x, scale, zp)
        return g.op("DequantizeLinear", xq, scale, zp)
    else:
        block_size = (scheme.granularity.block_size
                      if scheme.granularity.mode == GranularityMode.PER_BLOCK
                      else 0)
        return g.op(
            "com.microxscaling::MxQuantize",
            x,
            elem_format_s=scheme.format.name,
            block_size_i=block_size,
            round_mode_s=scheme.round_mode,
        )
```

**Step 5: Run helper tests**

```bash
pytest src/tests/test_onnx_export.py -k "is_standard_format" -v
```

Expected: 5 PASS

**Step 6: Commit**

```bash
git add src/onnx/__init__.py src/onnx/helpers.py src/tests/test_onnx_export.py
git commit -m "feat(onnx): add _is_standard_format + _emit_quantize_node helpers (Phase 5 Task 1)"
```

---

### Task 2: `LinearFunction.symbolic()`

**Files:**
- Modify: `src/ops/linear.py` — add `symbolic()` to `LinearFunction`
- Modify: `src/tests/test_onnx_export.py` — add linear export tests

**Step 1: Add linear export tests to `src/tests/test_onnx_export.py`**

Append to the file:

```python
# ---------------------------------------------------------------------------
# Task 2: LinearFunction ONNX export
# ---------------------------------------------------------------------------

def test_linear_standard_format_uses_qdq():
    """int8 per_tensor → QuantizeLinear/DequantizeLinear nodes in graph."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "QuantizeLinear"), "Expected QDQ for int8"
    assert _has_op(onnx_model, "DequantizeLinear"), "Expected QDQ for int8"
    assert not _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
        "int8 should NOT use MxQuantize"


def test_linear_mx_format_uses_custom_op():
    """fp4_e2m1 per_block → com.microxscaling::MxQuantize nodes."""
    cfg = _mx_cfg("fp4_e2m1", block_size=32)
    model = QuantizedLinear(32, 64, cfg=cfg)
    x = torch.randn(2, 32)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
        "Expected MxQuantize for fp4 per_block"
    assert not _has_op(onnx_model, "QuantizeLinear"), \
        "fp4 per_block should NOT use QDQ"


def test_linear_export_checker_passes():
    """onnx.checker.check_model() passes for int8 linear export."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)  # raises if invalid


def test_linear_no_bias_exports_cleanly():
    """Linear without bias exports without error; graph has no Add after MatMul bias path."""
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, bias=False, cfg=cfg)
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)  # must not raise
    onnx.checker.check_model(onnx_model)


def test_linear_no_quantization_exports_cleanly():
    """Passthrough model (no cfg) exports as plain Gemm/MatMul."""
    model = QuantizedLinear(8, 16)  # no cfg → passthrough
    x = torch.randn(2, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)
    assert not _has_op(onnx_model, "QuantizeLinear"), \
        "Passthrough model should have no QDQ"
```

**Step 2: Run to verify failure**

```bash
pytest src/tests/test_onnx_export.py -k "linear" -v
```

Expected: FAIL — exports succeed but graph has 78 arithmetic nodes, not QDQ/MxQuantize.
(The `_has_op` assertions will fail: no QuantizeLinear present.)

**Step 3: Add `symbolic()` to `LinearFunction` in `src/ops/linear.py`**

Insert after the `backward()` method (after line ~190), before the `QuantizedLinear` class:

```python
    @staticmethod
    def symbolic(g, x, w, b, cfg, name, emit_fn):
        """ONNX symbolic: emit quantize nodes + MatMul + optional Add."""
        from src.onnx.helpers import _emit_quantize_node

        # Quantize input pipeline
        for scheme in cfg.input:
            x = _emit_quantize_node(g, x, scheme)

        # Quantize weight pipeline
        for scheme in cfg.weight:
            w = _emit_quantize_node(g, w, scheme)

        # MatMul(x, w^T)
        wt = g.op("Transpose", w, perm_i=[1, 0])
        y = g.op("MatMul", x, wt)

        # Quantize output[0] (post-matmul, pre-bias)
        if len(cfg.output) > 0:
            y = _emit_quantize_node(g, y, cfg.output[0])

        # Add bias
        if b is not None:
            # Quantize bias pipeline
            qb = b
            for scheme in cfg.bias:
                qb = _emit_quantize_node(g, qb, scheme)
            y = g.op("Add", y, qb)

            # Quantize output[1] (post-bias-add)
            if len(cfg.output) > 1:
                y = _emit_quantize_node(g, y, cfg.output[1])

        return y
```

**Step 4: Run linear tests**

```bash
pytest src/tests/test_onnx_export.py -k "linear" -v
```

Expected: 5 PASS

**Step 5: Run full suite to confirm no regression**

```bash
pytest src/tests/ -q --tb=short
```

Expected: 958 passed (same count — new tests are already in the 958 from Task 1)

**Step 6: Commit**

```bash
git add src/ops/linear.py src/tests/test_onnx_export.py
git commit -m "feat(onnx): add LinearFunction.symbolic() — QDQ for int8, MxQuantize for fp4 (Phase 5 Task 2)"
```

---

### Task 3: `ConvFunction.symbolic()` + `ConvTransposeFunction.symbolic()`

**Files:**
- Modify: `src/ops/conv.py` — add `symbolic()` to both Function classes
- Modify: `src/tests/test_onnx_export.py` — add conv export tests

**Step 1: Add conv export tests**

Append to `src/tests/test_onnx_export.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: ConvFunction ONNX export
# ---------------------------------------------------------------------------

def test_conv2d_standard_format_uses_qdq():
    """int8 per_tensor on Conv2d → QDQ nodes."""
    from src.ops.conv import QuantizedConv2d
    cfg = _standard_cfg("int8")
    model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 4, 8, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "QuantizeLinear"), "Expected QDQ for int8 conv"
    assert _has_op(onnx_model, "DequantizeLinear")


def test_conv2d_mx_format_uses_custom_op():
    """fp4_e2m1 per_block on Conv2d → MxQuantize nodes."""
    from src.ops.conv import QuantizedConv2d
    cfg = _mx_cfg("fp4_e2m1", block_size=32)
    model = QuantizedConv2d(32, 64, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 32, 8, 8)
    onnx_model = _export(model, x)
    assert _has_op(onnx_model, "MxQuantize", "com.microxscaling")
    assert not _has_op(onnx_model, "QuantizeLinear")


def test_conv2d_export_checker_passes():
    """onnx.checker passes for int8 Conv2d export."""
    from src.ops.conv import QuantizedConv2d
    cfg = _standard_cfg("int8")
    model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
    x = torch.randn(1, 4, 8, 8)
    onnx_model = _export(model, x)
    onnx.checker.check_model(onnx_model)
```

**Step 2: Run to verify failure**

```bash
pytest src/tests/test_onnx_export.py -k "conv" -v
```

Expected: FAIL — no `symbolic()` on ConvFunction yet.

**Step 3: Add `symbolic()` to `ConvFunction` in `src/ops/conv.py`**

Insert after `ConvFunction.backward()`, before class `QuantizedConv1d`:

```python
    @staticmethod
    def symbolic(g, input, weight, bias, stride, padding, dilation, groups,
                 cfg, name, emit_fn):
        """ONNX symbolic: emit quantize nodes + Conv."""
        from src.onnx.helpers import _emit_quantize_node

        # Quantize input and weight pipelines
        for scheme in cfg.input:
            input = _emit_quantize_node(g, input, scheme)
        for scheme in cfg.weight:
            weight = _emit_quantize_node(g, weight, scheme)

        # Quantize bias (if present)
        if bias is not None:
            for scheme in cfg.bias:
                bias = _emit_quantize_node(g, bias, scheme)

        # Infer kernel shape from weight type (available for static-shape export)
        weight_sizes = weight.type().sizes()
        kernel_shape = list(weight_sizes[2:]) if weight_sizes is not None else None

        # Build pads: PyTorch padding (ph, pw) → ONNX pads [ph, pw, ph, pw]
        pad_list = list(padding) if hasattr(padding, '__iter__') else [padding]
        onnx_pads = pad_list + pad_list  # symmetric

        conv_kwargs = dict(
            dilations_i=list(dilation) if hasattr(dilation, '__iter__') else [dilation],
            group_i=groups,
            pads_i=onnx_pads,
            strides_i=list(stride) if hasattr(stride, '__iter__') else [stride],
        )
        if kernel_shape is not None:
            conv_kwargs["kernel_shape_i"] = kernel_shape

        if bias is not None:
            output = g.op("Conv", input, weight, bias, **conv_kwargs)
        else:
            output = g.op("Conv", input, weight, **conv_kwargs)

        # Quantize output pipeline
        for scheme in cfg.output:
            output = _emit_quantize_node(g, output, scheme)

        return output
```

**Step 4: Add `symbolic()` to `ConvTransposeFunction` in `src/ops/conv.py`**

`ConvTransposeFunction.forward` signature:
`(ctx, input, weight, bias, stride, padding, output_padding, dilation, groups, cfg, name, emit_fn)`

Insert after `ConvTransposeFunction.backward()`:

```python
    @staticmethod
    def symbolic(g, input, weight, bias, stride, padding, output_padding,
                 dilation, groups, cfg, name, emit_fn):
        """ONNX symbolic: emit quantize nodes + ConvTranspose."""
        from src.onnx.helpers import _emit_quantize_node

        for scheme in cfg.input:
            input = _emit_quantize_node(g, input, scheme)
        for scheme in cfg.weight:
            weight = _emit_quantize_node(g, weight, scheme)
        if bias is not None:
            for scheme in cfg.bias:
                bias = _emit_quantize_node(g, bias, scheme)

        weight_sizes = weight.type().sizes()
        kernel_shape = list(weight_sizes[2:]) if weight_sizes is not None else None

        pad_list = list(padding) if hasattr(padding, '__iter__') else [padding]
        onnx_pads = pad_list + pad_list

        conv_kwargs = dict(
            dilations_i=list(dilation) if hasattr(dilation, '__iter__') else [dilation],
            group_i=groups,
            output_padding_i=list(output_padding) if hasattr(output_padding, '__iter__') else [output_padding],
            pads_i=onnx_pads,
            strides_i=list(stride) if hasattr(stride, '__iter__') else [stride],
        )
        if kernel_shape is not None:
            conv_kwargs["kernel_shape_i"] = kernel_shape

        if bias is not None:
            output = g.op("ConvTranspose", input, weight, bias, **conv_kwargs)
        else:
            output = g.op("ConvTranspose", input, weight, **conv_kwargs)

        for scheme in cfg.output:
            output = _emit_quantize_node(g, output, scheme)

        return output
```

**Step 5: Run conv tests + full suite**

```bash
pytest src/tests/test_onnx_export.py -k "conv" -v
pytest src/tests/ -q --tb=short
```

Expected: 3 conv PASS + 958 total PASS

**Step 6: Commit**

```bash
git add src/ops/conv.py src/tests/test_onnx_export.py
git commit -m "feat(onnx): add ConvFunction + ConvTransposeFunction symbolic() (Phase 5 Task 3)"
```

---

### Task 4: `src/onnx/export.py` + end-to-end test

**Files:**
- Create: `src/onnx/export.py`
- Modify: `src/onnx/__init__.py`
- Modify: `src/tests/test_onnx_export.py` — add end-to-end test

**Step 1: Add end-to-end test**

Append to `src/tests/test_onnx_export.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: export_quantized_model() end-to-end
# ---------------------------------------------------------------------------

def test_export_quantized_model_linear(tmp_path):
    """export_quantized_model() writes a valid .onnx file for a quantized linear model."""
    from src.onnx import export_quantized_model
    cfg = _standard_cfg("int8")
    model = QuantizedLinear(8, 16, cfg=cfg)
    x = torch.randn(2, 8)
    out_path = str(tmp_path / "model.onnx")
    export_quantized_model(model, x, out_path)  # must not raise
    # File written and valid
    loaded = onnx.load(out_path)
    onnx.checker.check_model(loaded)
    assert _has_op(loaded, "QuantizeLinear")


def test_export_quantized_model_mixed(tmp_path):
    """Export a small mixed model: one Linear + one Conv2d, both quantized."""
    from src.onnx import export_quantized_model
    from src.ops.conv import QuantizedConv2d

    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cfg_int8 = _standard_cfg("int8")
            self.conv = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg_int8)
            self.linear = QuantizedLinear(8 * 8 * 8, 16, cfg=cfg_int8)

        def forward(self, x):
            x = self.conv(x)
            x = x.flatten(1)
            return self.linear(x)

    model = SmallModel()
    x = torch.randn(1, 4, 8, 8)
    out_path = str(tmp_path / "mixed.onnx")
    export_quantized_model(model, x, out_path)
    loaded = onnx.load(out_path)
    onnx.checker.check_model(loaded)
    assert _has_op(loaded, "QuantizeLinear")
    assert _has_op(loaded, "Conv")
```

**Step 2: Run to verify failure**

```bash
pytest src/tests/test_onnx_export.py -k "export_quantized_model" -v
```

Expected: `ImportError: cannot import name 'export_quantized_model' from 'src.onnx'`

**Step 3: Create `src/onnx/export.py`**

```python
"""
export_quantized_model: export a quantized model to ONNX.

Wraps torch.onnx.export with com.microxscaling custom opset registration
and verifies the output graph with onnx.checker.
"""
import torch
import torch.nn as nn


def export_quantized_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export a quantized model to an ONNX file.

    Args:
        model: Module containing QuantizedLinear / QuantizedConv{1,2,3}d layers.
            Must have symbolic() methods on its autograd.Function subclasses
            (added in Phase 5).
        dummy_input: Representative input tensor (defines input shape in graph).
        output_path: Path to write the .onnx file.
        opset_version: ONNX opset version. Default: 17.

    The exported graph uses:
    - QuantizeLinear/DequantizeLinear for int8/int4/int2/fp8 formats.
    - com.microxscaling::MxQuantize for MX block-format quantization.

    Note: Scale values in QDQ nodes are placeholder constants (1.0);
    the graph is valid for visualization but not for runtime inference.
    """
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        opset_version=opset_version,
        custom_opsets={"com.microxscaling": 1},
        do_constant_folding=False,
    )
    _verify_onnx_graph(output_path)


def _verify_onnx_graph(path: str) -> None:
    """Load and validate the ONNX graph with onnx.checker.

    onnx.checker skips semantic validation for unknown custom op domains,
    so com.microxscaling nodes are accepted as long as the graph structure
    is valid.
    """
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)
```

**Step 4: Update `src/onnx/__init__.py`**

```python
from .export import export_quantized_model

__all__ = ["export_quantized_model"]
```

**Step 5: Run all onnx tests + full suite**

```bash
pytest src/tests/test_onnx_export.py -v
pytest src/tests/ -q --tb=short
```

Expected: all onnx tests PASS, full suite 958 PASS

**Step 6: Commit**

```bash
git add src/onnx/export.py src/onnx/__init__.py src/tests/test_onnx_export.py
git commit -m "feat(onnx): add export_quantized_model() + end-to-end tests (Phase 5 Task 4)"
```

---

### Task 5: Update `docs/status/CURRENT.md`

**Files:**
- Modify: `docs/status/CURRENT.md`

**Step 1: Rewrite CURRENT.md**

Update to reflect Phase 5 completion status. Use the standard format from CLAUDE.md Section 6.3.

Mark all tasks complete, set "下一步" to "Phase 5 完成" or next initiative, update "断点续传必读文件" to Phase 5 files.

**Step 2: Commit**

```bash
git add docs/status/CURRENT.md
git commit -m "docs(status): Phase 5 complete — ONNX export with QDQ + MxQuantize custom op"
```

---

## Phase 5 Completion Checklist (from ADR-003)

- [ ] `export_quantized_model()` works for `QuantizedLinear` + `QuantizedConv2d`
- [ ] `onnx.checker.check_model()` passes
- [ ] Standard formats (int8/int4/int2) → `QuantizeLinear`/`DequantizeLinear` QDQ
- [ ] MX formats (per_block) → `com.microxscaling::MxQuantize`
- [ ] `pytest src/tests/test_onnx_export.py` all green
- [ ] No regression in existing 958 tests

## Not in scope

- ORT custom op implementation (runtime execution)
- TensorRT plugin
- `netron` manual verification (done by user after export)
- fp8 QDQ (opset 21 not targeted; fp8 exports as MxQuantize)
- ConvTranspose export test (functional parity assumed; manual test if needed)
