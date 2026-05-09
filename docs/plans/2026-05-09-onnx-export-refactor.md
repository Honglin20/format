# ONNX Export Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor ONNX export to support multi-input models, NF4 format, real calibration scales, and full format-coverage tests.

**Architecture:** Five-phase TDD refactor. Phase 1 generalizes `dummy_input` types end-to-end. Phase 2 adds NF4 ONNX export. Phase 3 wires calibration scales into QDQ nodes. Phase 4 builds the parametrized format matrix. Phase 5 cleans up and runs full regression.

**Tech Stack:** PyTorch, ONNX, pytest parametrize.

**Key constraint:** Each _emit_quantize_node call already has access to a layer `name` via symbolic(); we thread it through to `format.export_onnx()` for scale lookup. Scale registry lives on `QuantizeContext` and is populated by walking `qmodel.named_modules()` for `_output_scale` buffers before export.

**Test entry point:** `pytest src/tests/test_onnx_export.py -q`
**Regression gate:** `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` (must be ≥ 2,093 passed)

---

### Task 1: Multi-input type — test first

**Files:**
- Modify: `src/tests/test_onnx_export.py`

**Step 1: Write multi-input tests**

Add these tests to the existing file (before Task 1 unit tests block):

```python
# ---------------------------------------------------------------------------
# Multi-input type tests
# ---------------------------------------------------------------------------

class TestMultiInput:
    """ONNX export with list / tuple / dict dummy_input."""

    @staticmethod
    def _make_two_input_model():
        """Model with two separate inputs."""
        cfg = _standard_cfg("int8")

        class TwoInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = QuantizedLinear(8, 16, cfg=cfg)
                self.linear2 = QuantizedLinear(8, 16, cfg=cfg)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)

        return TwoInputModel()

    def test_tuple_input(self):
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        buf = io.BytesIO()
        torch.onnx.export(model, (x, y), buf, opset_version=17,
                          custom_opsets={"com.microxscaling": 1})
        buf.seek(0)
        onnx_model = onnx.load(buf)
        onnx.checker.check_model(onnx_model)
        # Should have two graph inputs
        assert len(onnx_model.graph.input) >= 2

    def test_list_input(self):
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        buf = io.BytesIO()
        torch.onnx.export(model, [x, y], buf, opset_version=17,
                          custom_opsets={"com.microxscaling": 1})
        buf.seek(0)
        onnx_model = onnx.load(buf)
        onnx.checker.check_model(onnx_model)

    def test_dict_input(self):
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        buf = io.BytesIO()
        torch.onnx.export(model, {"x": x, "y": y}, buf, opset_version=17,
                          custom_opsets={"com.microxscaling": 1})
        buf.seek(0)
        onnx_model = onnx.load(buf)
        onnx.checker.check_model(onnx_model)

    def test_export_quantized_model_with_tuple(self, tmp_path):
        from src.onnx import export_quantized_model
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        out = str(tmp_path / "two_input.onnx")
        export_quantized_model(model, (x, y), out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_export_quantized_model_with_list(self, tmp_path):
        from src.onnx import export_quantized_model
        model = self._make_two_input_model()
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        out = str(tmp_path / "list_input.onnx")
        export_quantized_model(model, [x, y], out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
```

**Step 2: Run to verify failures**

```bash
pytest src/tests/test_onnx_export.py::TestMultiInput -q
```
Expected: some FAIL because `export_quantized_model` only handles single `Tensor`.

**Step 3: Implement multi-input support**

Fix in order of call chain:

a) `src/onnx/export.py:11-41` — Change signature:
```python
def export_quantized_model(
    model: nn.Module,
    dummy_input,  # Union[Tensor, tuple, list, dict]
    output_path: str,
    opset_version: int = 17,
) -> None:
    ...
    args = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
    torch.onnx.export(model, args, output_path, ...)
```

b) `src/session/_context.py:156-174` — `QuantizeContext.export_onnx`:
```python
def export_onnx(self, dummy_input, output_path, opset_version=17):
    args = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
    torch.onnx.export(self.model, args, output_path, ...)
```

c) `src/session/_model.py:489` — `_export_onnx`:
```python
def _export_onnx(self, dummy_input, output_path, opset_version=17):
    with QuantizeContext(...) as ctx:
        ctx.export_onnx(dummy_input, output_path, opset_version=opset_version)
```

d) `src/session/_quant.py:149-160` — `_QuantSession.__call__`:
```python
def __call__(self, *args, **kwargs):
    ...
    if args and self._last_input is None:
        self._last_input = args[0] if len(args) == 1 else args
    return self.qmodel(*args, **kwargs)
```

e) `src/session/_quant.py:242-259` — `_QuantSession.export_onnx`:
```python
def export_onnx(self, output_path, dummy_input=None, opset_version=17):
    inp = dummy_input if dummy_input is not None else self._last_input
    if inp is None:
        raise ValueError(...)
    self.qmodel.export_onnx(inp, output_path, opset_version=opset_version)
```

**Step 4: Run tests to verify**

```bash
pytest src/tests/test_onnx_export.py::TestMultiInput -q
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add src/onnx/export.py src/session/_context.py src/session/_model.py src/session/_quant.py src/tests/test_onnx_export.py
git commit -m "feat(onnx): support multi-input types (tuple, list, dict) for ONNX export"
```

---

### Task 2: Auto-record calibration/forward inputs

**Files:**
- Modify: `src/tests/test_onnx_export.py`
- Modify: `src/session/_quant.py`

**Step 1: Write tests for auto-input**

```python
class TestAutoInput:
    """_last_input auto-recording for ONNX export without dummy_input."""

    def test_export_after_forward_uses_recorded_input(self, tmp_path):
        from src.session._quant import _QuantSession
        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        session = _QuantSession(model, cfg)
        x = torch.randn(2, 8)
        session(x)  # records _last_input
        out = str(tmp_path / "auto.onnx")
        session.export_onnx(out)  # no dummy_input
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)

    def test_export_without_input_raises(self):
        from src.session._quant import _QuantSession
        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        session = _QuantSession(model, cfg)
        with pytest.raises(ValueError, match="No dummy_input"):
            session.export_onnx("nowhere.onnx")

    def test_export_after_multi_input_forward(self, tmp_path):
        from src.session._quant import _QuantSession
        cfg = _standard_cfg("int8")
        model = TestMultiInput._make_two_input_model()
        session = _QuantSession(model, cfg)
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        session(x, y)
        out = str(tmp_path / "multi_auto.onnx")
        session.export_onnx(out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
```

**Step 2: Run to verify failures**

```bash
pytest src/tests/test_onnx_export.py::TestAutoInput -q
```
Expected: `test_export_after_multi_input_forward` FAIL (records only `args[0]`).

**Step 3: Implement (already done in Task 1 if `__call__` was updated)**

If the `__call__` fix from Task 1 covers `_last_input = args[0] if len(args) == 1 else args`, all three tests should pass. Verify and adjust.

**Step 4: Commit**

```bash
git add src/tests/test_onnx_export.py
git commit -m "test(onnx): add auto-input recording tests for ONNX export"
```

---

### Task 3: NF4 ONNX export

**Files:**
- Modify: `src/formats/lookup_formats.py`
- Modify: `src/tests/test_onnx_export.py`

**Step 1: Write NF4 export tests**

```python
class TestNF4Export:
    """NF4 lookup-table format ONNX export."""

    def test_nf4_per_tensor_emits_nf4_quantize(self):
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "NF4Quantize", "com.microxscaling"), \
            "NF4 should emit NF4Quantize custom op"
        assert not _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
            "NF4 should NOT use MxQuantize"
        assert not _has_op(onnx_model, "QuantizeLinear"), \
            "NF4 should NOT use QDQ"

    def test_nf4_per_block_emits_nf4_quantize(self):
        """NF4 + per_block → still NF4Quantize (NF4 is block-agnostic in ONNX)."""
        from src.formats.base import FormatBase
        fmt = FormatBase.from_str("nf4")
        gran = GranularitySpec.per_block(32)
        s = QuantScheme(format=fmt, granularity=gran)
        cfg = OpQuantConfig(input=s, weight=s)
        model = QuantizedLinear(32, 64, cfg=cfg)
        x = torch.randn(2, 32)
        onnx_model = _export(model, x)
        assert _has_op(onnx_model, "NF4Quantize", "com.microxscaling")

    def test_nf4_export_checker_passes(self):
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)

    def test_nf4_levels_attribute(self):
        """NF4Quantize node must carry the levels_f attribute with 16 values."""
        from src.formats.lookup_formats import NF4Format
        cfg = _standard_cfg("nf4")
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        for n in onnx_model.graph.node:
            if n.op_type == "NF4Quantize":
                levels = [a for a in n.attribute if a.name == "levels_f"]
                assert len(levels) == 1
                assert len(levels[0].floats) == len(NF4Format.NF4_LEVELS)
                return
        pytest.fail("No NF4Quantize node found")
```

**Step 2: Run to verify failures**

```bash
pytest src/tests/test_onnx_export.py::TestNF4Export -q
```
Expected: all FAIL (NF4 has no `export_onnx`, falls back to MxQuantize).

**Step 3: Implement LookupFormat.export_onnx()**

```python
# In src/formats/lookup_formats.py, inside LookupFormat class:

def export_onnx(self, g, x, scheme, name=None):
    """Emit com.microxscaling::NF4Quantize custom node with LUT levels."""
    levels_list = self.levels.detach().cpu().tolist()
    return g.op(
        "com.microxscaling::NF4Quantize", x,
        levels_f=levels_list,
    )
```

**Step 4: Run tests to verify**

```bash
pytest src/tests/test_onnx_export.py::TestNF4Export -q
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add src/formats/lookup_formats.py src/tests/test_onnx_export.py
git commit -m "feat(onnx): add NF4 ONNX export via NF4Quantize custom op"
```

---

### Task 4: Calibration scale wiring

**Files:**
- Modify: `src/session/_context.py` — add `_export_scales`, `_collect_export_scales()`
- Modify: `src/onnx/helpers.py` — `_emit_quantize_node(name=...)`
- Modify: `src/formats/base.py` — `export_onnx(name=None)`
- Modify: `src/formats/int_formats.py` — `export_onnx(name=None)`
- Modify: `src/formats/fp_formats.py` — `export_onnx(name=None)`
- Modify: `src/formats/lookup_formats.py` — `export_onnx(name=None)` (NF4 ignores scale)
- Modify: `src/ops/linear.py` — pass name in symbolic
- Modify: `src/ops/conv.py` — pass name in symbolic
- Modify: `src/ops/matmul.py` — pass name in symbolic
- Modify: `src/ops/bmm.py` — pass name in symbolic
- Modify: `src/tests/test_onnx_export.py`

**Step 1: Write scale wiring test**

```python
class TestScaleWiring:
    """Real calibration scales embedded in ONNX QDQ nodes."""

    def test_calibrated_int8_exports_real_scale(self, tmp_path):
        from src.session._quant import _QuantSession
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy

        cfg = _standard_cfg("int8")
        model = QuantizedLinear(8, 16, cfg=cfg)
        session = _QuantSession(model, cfg)

        # Calibrate with known-scale data
        x = torch.ones(2, 8) * 5.0  # all activations = 5.0
        with session.calibrate():
            session(x)

        out = str(tmp_path / "scaled.onnx")
        session.export_onnx(out)

        # Verify scale in ONNX is not 1.0
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
        for init in loaded.graph.initializer:
            # Find the scale constant — should be roughly 1/5.0 = 0.2
            # (MaxScaleStrategy picks max(|x|) = 5.0, scale = 1/5.0)
            val = torch.from_numpy(init.to_array())
            if val.numel() == 1 and abs(val.item() - 1.0) > 0.01:
                # Found a non-1.0 scale
                assert 0.15 < val.item() < 0.25, \
                    f"Expected scale ~0.2, got {val.item()}"
                return
        # If we reach here, all scales were 1.0 — calibration didn't propagate
        pytest.fail("All QDQ scales are placeholder 1.0")
```

**Step 2: Run to verify failure**

```bash
pytest src/tests/test_onnx_export.py::TestScaleWiring -q
```
Expected: FAIL (scale still 1.0).

**Step 3: Implement scale wiring**

a) `src/session/_context.py` — Add to `QuantizeContext`:

```python
# At class level or __init__:
self._export_scales: Dict[str, torch.Tensor] = {}

def _collect_export_scales(self):
    """Walk model and collect _output_scale buffers keyed by module name."""
    self._export_scales.clear()
    for name, module in self.model.named_modules():
        if hasattr(module, "_output_scale"):
            self._export_scales[name] = module._output_scale
```

b) `src/onnx/helpers.py` — Thread name:

```python
def _emit_quantize_node(g, x, scheme, name=None):
    return scheme.format.export_onnx(g, x, scheme, name=name)
```

c) `src/formats/base.py` — Add name param:

```python
def export_onnx(self, g, x, scheme, name=None):
    ...
    # (unchanged, just add name to signature)
```

d) `src/formats/int_formats.py` — Use scale:

```python
def export_onnx(self, g, x, scheme, name=None):
    from src.scheme.granularity import GranularityMode
    if scheme.granularity.mode == GranularityMode.PER_BLOCK:
        return super().export_onnx(g, x, scheme, name=name)
    
    # Try to get real scale from active QuantizeContext
    scale_val = 1.0
    if name is not None:
        from src.session._context import QuantizeContext
        ctx = QuantizeContext.get_active()
        if ctx is not None and name in ctx._export_scales:
            raw = ctx._export_scales[name]
            if raw.numel() == 1:
                scale_val = raw.item()
            # per-channel scales would need zp dimension alignment;
            # keep 1.0 for now (per-tensor is the common case)

    scale = g.op("Constant", value_t=torch.tensor(scale_val, dtype=torch.float32))
    zp = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
    xq = g.op("QuantizeLinear", x, scale, zp)
    return g.op("DequantizeLinear", xq, scale, zp)
```

e) `src/formats/fp_formats.py` — Same pattern, only for fp8 QDQ case:

```python
def export_onnx(self, g, x, scheme, name=None):
    from src.scheme.granularity import GranularityMode
    if self.name in ("fp8_e4m3", "fp8_e5m2") and scheme.granularity.mode != GranularityMode.PER_BLOCK:
        scale_val = 1.0
        if name is not None:
            from src.session._context import QuantizeContext
            ctx = QuantizeContext.get_active()
            if ctx is not None and name in ctx._export_scales:
                raw = ctx._export_scales[name]
                if raw.numel() == 1:
                    scale_val = raw.item()
        scale = g.op("Constant", value_t=torch.tensor(scale_val, dtype=torch.float32))
        zp = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
        xq = g.op("QuantizeLinear", x, scale, zp)
        return g.op("DequantizeLinear", xq, scale, zp)
    return super().export_onnx(g, x, scheme, name=name)
```

f) Thread name in all symbolic methods — example for `src/ops/linear.py:186`:

```python
@staticmethod
def symbolic(g, x, w, b, cfg, name, emit_fn, output_scale=None):
    from src.onnx.helpers import _emit_quantize_node

    if cfg.storage is not None:
        x = _emit_quantize_node(g, x, cfg.storage, name=name)
    if cfg.input is not None:
        x = _emit_quantize_node(g, x, cfg.input, name=name)
    # ... same pattern for weight, bias, output
```

Same pattern in `src/ops/conv.py:216`, `src/ops/matmul.py:177`, `src/ops/bmm.py:138`.

g) In `src/session/_context.py:export_onnx` — call `_collect_export_scales()` before export:

```python
def export_onnx(self, dummy_input, output_path, opset_version=17):
    self._collect_export_scales()
    args = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
    torch.onnx.export(self.model, args, output_path, ...)
    _verify_onnx_graph(output_path)
```

**Step 4: Run scale tests**

```bash
pytest src/tests/test_onnx_export.py::TestScaleWiring -q
```

**Step 5: Commit**

```bash
git add src/session/_context.py src/onnx/helpers.py src/formats/base.py src/formats/int_formats.py src/formats/fp_formats.py src/formats/lookup_formats.py src/ops/linear.py src/ops/conv.py src/ops/matmul.py src/ops/bmm.py src/tests/test_onnx_export.py
git commit -m "feat(onnx): wire calibration scales into QDQ nodes during export"
```

---

### Task 5: Full format coverage matrix

**Files:**
- Modify: `src/tests/test_onnx_export.py`

**Step 1: Write parametrized format tests**

```python
# ---------------------------------------------------------------------------
# Format coverage matrix
# ---------------------------------------------------------------------------

ALL_STANDARD_FORMATS = ["int8", "int4", "int2", "fp8_e4m3", "fp8_e5m2"]
ALL_MX_FORMATS = ["fp4_e2m1", "fp6_e3m2", "fp6_e2m3"]
ALL_LOOKUP_FORMATS = ["nf4"]
ALL_TRUNC_FORMATS = ["bf16", "fp16"]
ALL_FORMATS = ALL_STANDARD_FORMATS + ALL_MX_FORMATS + ALL_LOOKUP_FORMATS + ALL_TRUNC_FORMATS


class TestFormatMatrix:
    """Every registered format exports valid ONNX for linear and conv2d."""

    @pytest.mark.parametrize("fmt_name", ALL_STANDARD_FORMATS)
    def test_standard_format_per_tensor_linear(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "QuantizeLinear"), \
            f"{fmt_name} per_tensor should use QDQ"

    @pytest.mark.parametrize("fmt_name", ALL_STANDARD_FORMATS)
    def test_standard_format_per_tensor_conv2d(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedConv2d(4, 8, kernel_size=3, padding=1, cfg=cfg)
        x = torch.randn(1, 4, 8, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "QuantizeLinear")

    @pytest.mark.parametrize("fmt_name", ALL_MX_FORMATS)
    def test_mx_format_per_block_linear(self, fmt_name):
        cfg = _mx_cfg(fmt_name, block_size=32)
        model = QuantizedLinear(32, 64, cfg=cfg)
        x = torch.randn(2, 32)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "MxQuantize", "com.microxscaling"), \
            f"{fmt_name} per_block should use MxQuantize"

    @pytest.mark.parametrize("fmt_name", ALL_MX_FORMATS)
    def test_mx_format_per_block_conv2d(self, fmt_name):
        cfg = _mx_cfg(fmt_name, block_size=32)
        model = QuantizedConv2d(32, 64, kernel_size=3, padding=1, cfg=cfg)
        x = torch.randn(1, 32, 8, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "MxQuantize", "com.microxscaling")

    @pytest.mark.parametrize("fmt_name", ALL_LOOKUP_FORMATS)
    def test_lookup_format_per_tensor_linear(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        assert _has_op(onnx_model, "NF4Quantize", "com.microxscaling")

    @pytest.mark.parametrize("fmt_name", ALL_TRUNC_FORMATS)
    def test_trunc_format_per_tensor_linear(self, fmt_name):
        cfg = _standard_cfg(fmt_name)
        model = QuantizedLinear(8, 16, cfg=cfg)
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
        # Truncation formats use MxQuantize custom op
        assert _has_op(onnx_model, "MxQuantize", "com.microxscaling")

    @pytest.mark.parametrize("fmt_name", ALL_FORMATS)
    def test_all_formats_export_passthrough(self, fmt_name):
        """Passthrough (no cfg) exports cleanly regardless of format."""
        model = QuantizedLinear(8, 16)  # no cfg → passthrough
        x = torch.randn(2, 8)
        onnx_model = _export(model, x)
        onnx.checker.check_model(onnx_model)
```

**Step 2: Run format matrix**

```bash
pytest src/tests/test_onnx_export.py::TestFormatMatrix -q -v
```

**Step 3: Fix any failures**

Formats that may need adjustments:
- `bf16`/`fp16` with per_tensor: `BFloat16Format` → `FormatBase.export_onnx()` → `MxQuantize`. Check if this is correct intent.
- `fp8_e4m3`/`fp8_e5m2` with per_block: should go to `MxQuantize` via `FPFormat.export_onnx()` → `super().export_onnx()`.

**Step 4: Commit**

```bash
git add src/tests/test_onnx_export.py
git commit -m "test(onnx): add format coverage matrix for all registered formats"
```

---

### Task 6: End-to-end session export test

**Files:**
- Modify: `src/tests/test_onnx_export.py`

**Step 1: Write E2E test**

```python
class TestSessionE2EExport:
    """Session → quantize → calibrate → export_onnx pipeline."""

    def test_session_quantize_calibrate_export(self, tmp_path):
        from src.session._session import Session
        from src.session._config import QuantConfig

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
                self.linear = torch.nn.Linear(8 * 8 * 8, 16)

            def forward(self, x):
                x = self.conv(x)
                x = x.flatten(1)
                return self.linear(x)

        model = SimpleModel()
        cfg = QuantConfig(
            format="int8",
            granularity="per_tensor",
            calibrator="max",
        )
        qcfg = cfg.to_op_config()

        session = Session(model.eval(), qcfg)
        session.quantize()
        x = torch.randn(1, 4, 8, 8)
        session.calibrate([x])
        out = str(tmp_path / "session.onnx")
        session.export_onnx(out)
        loaded = onnx.load(out)
        onnx.checker.check_model(loaded)
```

**Step 2: Run**

```bash
pytest src/tests/test_onnx_export.py::TestSessionE2EExport -q
```

**Step 3: Commit**

```bash
git add src/tests/test_onnx_export.py
git commit -m "test(onnx): add session e2e export test"
```

---

### Task 7: Regression verification

**Step 1: Run existing ONNX tests**

```bash
pytest src/tests/test_onnx_export.py -q -v
```
Expected: ALL passing (including original tests: `test_is_standard_format_*`, `test_linear_*`, `test_conv2d_*`, `test_export_quantized_model_*`).

**Step 2: Run full test suite**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"
```
Expected: ≥ 2,093 passed.

**Step 3: Fix regressions if any**

Check that no existing tests break due to the `name` parameter being added to `_emit_quantize_node` and `format.export_onnx()`. Since `name` has a default of `None`, existing callers without `name` should still work.

**Step 4: Final commit**

```bash
git commit -m "chore: confirm full regression passes after ONNX refactor"
```
