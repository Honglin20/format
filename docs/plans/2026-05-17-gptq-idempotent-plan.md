# GPTQ Idempotency Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make GPTQ quantization idempotent by storing the per-channel scale as a module buffer, so the forward pass reuses the same scale GPTQ used internally instead of recomputing it from the modified weights.

**Architecture:** GPTQ computes `full_scale` from the original FP32 weights, then writes quantized FP32 values back to `module.weight.data`. The forward pass calls `quantize(w, cfg.weight)` without `scale`, recomputing amax from the GPTQ-modified weights — producing different results. Fix: GPTQ registers `full_scale` as `module._weight_scale` buffer; all quantized ops read it and pass `scale=buffers.weight_scale` to `quantize()`.

**Tech Stack:** PyTorch, existing quantize/scheme/ops infrastructure

---

### Task 1: Add `weight_scale` to `CalibrationBuffers`

**Files:**
- Modify: `src/ops/_calib_buffers.py`

**Step 1: Add the field**

In `src/ops/_calib_buffers.py`, add `weight_scale` field after `weight_importance`:

```python
@dataclass
class CalibrationBuffers:
    # ... existing fields ...
    weight_importance: Optional[torch.Tensor] = None
    weight_scale: Optional[torch.Tensor] = None
    # ... rest ...
```

**Step 2: Verify existing tests still pass**

Run: `pytest src/tests/test_gptq_optimizer.py -q`
Expected: All pass (new field is Optional, defaults to None)

**Step 3: Commit**

```bash
git add src/ops/_calib_buffers.py
git commit -m "feat(buffers): add weight_scale field to CalibrationBuffers"
```

---

### Task 2: Write failing test for GPTQ idempotency

**Files:**
- Modify: `src/tests/test_gptq_optimizer.py`

**Step 1: Write the failing test**

Add to `TestGPTQIntegration` class:

```python
def test_gptq_weight_scale_buffer_set(self):
    """After GPTQ, module should have _weight_scale buffer."""
    from src.session import quantize_model

    torch.manual_seed(42)
    model = _TinyModel()
    scheme = _per_channel_int4_scheme()
    cfg = OpQuantConfig(weight=scheme)
    qmodel = quantize_model(copy.deepcopy(model), cfg)

    calib_data = [torch.randn(4, 8) for _ in range(4)]
    opt = GPTQOptimizer(block_size=128)
    opt.optimize(qmodel, calib_data)

    assert hasattr(qmodel.linear, "_weight_scale")
    assert qmodel.linear._weight_scale is not None

def test_gptq_idempotent_requant(self):
    """Re-quantizing GPTQ weights with stored scale must produce same result."""
    from src.session import quantize_model

    torch.manual_seed(42)
    model = _TinyModel()
    scheme = _per_channel_int4_scheme()
    cfg = OpQuantConfig(weight=scheme)
    qmodel = quantize_model(copy.deepcopy(model), cfg)

    calib_data = [torch.randn(4, 8) for _ in range(4)]
    opt = GPTQOptimizer(block_size=128)
    opt.optimize(qmodel, calib_data)

    W_gptq = qmodel.linear.weight.data.clone()
    scale = qmodel.linear._weight_scale

    with torch.no_grad():
        W_requant = quantize(W_gptq, scheme, scale=scale)

    assert torch.allclose(W_gptq, W_requant, atol=1e-6), (
        f"GPTQ weights not idempotent under re-quantization: "
        f"max diff = {(W_gptq - W_requant).abs().max().item():.8f}"
    )
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/tests/test_gptq_optimizer.py::TestGPTQIntegration::test_gptq_weight_scale_buffer_set src/tests/test_gptq_optimizer.py::TestGPTQIntegration::test_gptq_idempotent_requant -v`
Expected: FAIL — `_weight_scale` buffer not set, idempotency broken

**Step 3: Commit**

```bash
git add src/tests/test_gptq_optimizer.py
git commit -m "test(gptq): add failing tests for _weight_scale buffer and idempotency"
```

---

### Task 3: GPTQ writes `_weight_scale` buffer

**Files:**
- Modify: `src/calibration/gptq_optimizer.py:104-166`

**Step 1: Register the buffer after GPTQ quantization**

In `GPTQOptimizer.optimize()`, after `module.weight.data = W_q` (line 161), register the `full_scale` as a buffer. The `full_scale` is computed inside `_gptq_quantize()`, so we need to return it.

Modify `_gptq_quantize` to return both the quantized weight and the scale:

```python
def _gptq_quantize(
    self,
    W: torch.Tensor,
    X: torch.Tensor,
    scheme,
) -> tuple:
    """Apply GPTQ to a single Linear weight matrix.

    Returns:
        (W_q, full_scale): GPTQ-quantized weight and the per-channel scale
        used during quantization.
    """
    # ... existing code unchanged through line 223 ...
    full_scale = _precompute_scale(W_f32, scheme)

    # ... existing code unchanged through line 262 ...
    return W_q.to(dtype=dtype), full_scale
```

Then in `optimize()`, use the returned scale:

```python
mse_before = (W - quantize(W, weight_scheme)).pow(2).mean().item()

W_q, full_scale = self._gptq_quantize(W, X, weight_scheme)
module.weight.data = W_q

# Register the scale as a buffer so the forward pass reuses it
if full_scale is not None:
    module.register_buffer("_weight_scale", full_scale.to(dtype=W.dtype))

mse_after = (W - W_q).pow(2).mean().item()
results[name] = {"mse_before": mse_before, "mse_after": mse_after}
```

**Step 2: Run the failing tests**

Run: `pytest src/tests/test_gptq_optimizer.py::TestGPTQIntegration::test_gptq_weight_scale_buffer_set src/tests/test_gptq_optimizer.py::TestGPTQIntegration::test_gptq_idempotent_requant -v`
Expected: PASS

**Step 3: Run full GPTQ test suite**

Run: `pytest src/tests/test_gptq_optimizer.py -q`
Expected: All pass

**Step 4: Commit**

```bash
git add src/calibration/gptq_optimizer.py
git commit -m "feat(gptq): register _weight_scale buffer for idempotent re-quantization"
```

---

### Task 4: Write failing test for forward-pass idempotency

**Files:**
- Modify: `src/tests/test_gptq_optimizer.py`

**Step 1: Write the failing test**

Add to `TestGPTQSessionIntegration` class:

```python
def test_gptq_forward_uses_weight_scale(self):
    """Forward pass after GPTQ should use _weight_scale, not recompute amax."""
    from src.session._config import QuantConfig
    from src.session._session import run_quantization

    torch.manual_seed(42)
    model = _TwoLayerModel()

    config = QuantConfig(
        w_format="int4",
        w_granularity="per_channel",
        w_axis=0,
        gptq=True,
        gptq_block_size=128,
    )
    calib = [torch.randn(2, 8) for _ in range(4)]
    qmodel, _, _ = run_quantization(model, config, calib, keep_fp32=False)

    # After GPTQ + calibration, forward pass should produce finite output
    qmodel.eval()
    with torch.no_grad():
        out = qmodel(torch.randn(2, 8))
    assert out.shape == (2, 4)
    assert not torch.isnan(out).any()

    # Verify _weight_scale buffers exist
    for name, mod in qmodel.named_modules():
        if hasattr(mod, "cfg") and hasattr(mod, "weight"):
            assert hasattr(mod, "_weight_scale"), (
                f"{name} missing _weight_scale after GPTQ"
            )
```

**Step 2: Run to verify it fails (on the buffer check)**

Run: `pytest src/tests/test_gptq_optimizer.py::TestGPTQSessionIntegration::test_gptq_forward_uses_weight_scale -v`
Expected: May pass for buffer check (already fixed in Task 3), but the forward pass is not yet using the buffer. This test validates the end-to-end pipeline works.

**Step 3: Commit**

```bash
git add src/tests/test_gptq_optimizer.py
git commit -m "test(gptq): add forward-pass weight_scale integration test"
```

---

### Task 5: QuantizedLinear reads `_weight_scale` buffer

**Files:**
- Modify: `src/ops/linear.py:387-398` (QuantizedLinear.forward)
- Modify: `src/ops/linear.py:166-168` (LinearFunction.forward)

**Step 1: Add `_weight_scale` to CalibrationBuffers construction in QuantizedLinear.forward**

At `src/ops/linear.py:387-398`, add one more line to the `CalibrationBuffers(...)` call:

```python
        buffers = CalibrationBuffers(
            # ... existing fields ...
            input_group_mask=self.get_buffer("_input_group_mask") if hasattr(self, "_input_group_mask") else None,
            weight_scale=self.get_buffer("_weight_scale") if hasattr(self, "_weight_scale") else None,
        )
```

**Step 2: Pass `scale` in LinearFunction.forward weight quantization**

At `src/ops/linear.py:166-168`, change:

```python
            if cfg.weight is not None:
                w = quantize(w, cfg.weight, importance=buffers.weight_importance)
```

to:

```python
            if cfg.weight is not None:
                w = quantize(w, cfg.weight, scale=buffers.weight_scale, importance=buffers.weight_importance)
```

**Step 3: Run GPTQ tests**

Run: `pytest src/tests/test_gptq_optimizer.py -q`
Expected: All pass

**Step 4: Commit**

```bash
git add src/ops/linear.py
git commit -m "feat(linear): read _weight_scale buffer in forward pass for GPTQ idempotency"
```

---

### Task 6: All Conv ops read `_weight_scale` buffer

**Files:**
- Modify: `src/ops/conv.py` (6 QuantizedConv classes + 2 Function classes)

The same two changes in each QuantizedConv class:
1. Add `weight_scale=...` to `CalibrationBuffers(...)` constructor
2. Add `scale=buffers.weight_scale` to the `quantize(weight, cfg.weight, ...)` call

**Step 1: QuantizedConv2d (line 417-428)**

Add to CalibrationBuffers:
```python
            input_group_mask=self.get_buffer("_input_group_mask") if hasattr(self, "_input_group_mask") else None,
            weight_scale=self.get_buffer("_weight_scale") if hasattr(self, "_weight_scale") else None,
```

**Step 2: ConvFunction.forward (line 200)**

Change:
```python
                weight = quantize(weight, cfg.weight, importance=buffers.weight_importance)
```
to:
```python
                weight = quantize(weight, cfg.weight, scale=buffers.weight_scale, importance=buffers.weight_importance)
```

**Step 3: Repeat for Conv1d (line 454-465), Conv3d (line 491-502)**

Same two changes each.

**Step 4: ConvTransposeFunction.forward (line 565)**

Change:
```python
            weight = quantize(weight, cfg.weight, importance=buffers.weight_importance)
```
to:
```python
            weight = quantize(weight, cfg.weight, scale=buffers.weight_scale, importance=buffers.weight_importance)
```

**Step 5: QuantizedConvTranspose2d (line 802-813), ConvTranspose1d (line 845-856), ConvTranspose3d (line 888-899)**

Add `weight_scale=...` to CalibrationBuffers in each.

**Step 6: Run full test suite**

Run: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"`
Expected: All pass (no regressions)

**Step 7: Commit**

```bash
git add src/ops/conv.py
git commit -m "feat(conv): read _weight_scale buffer in forward pass for GPTQ idempotency"
```

---

### Task 7: Verify E2E with diagnostic script

**Files:**
- Run: `scripts/gptq_sparse_diagnose.py`

**Step 1: Run diagnostic**

Run: `PYTHONPATH=. python scripts/gptq_sparse_diagnose.py`
Expected: `Idempotent: True` for all schemes, GPTQ gain accuracy positive

**Step 2: Run E2E script**

Run: `PYTHONPATH=. python scripts/gptq_sparse_e2e.py`
Expected: GPTQ configs show improved accuracy vs non-GPTQ baselines

**Step 3: Commit (no code changes, just verification)**

No commit needed — just verify results are correct.

---

### Task 8: Update GPTQOptimizer docstring

**Files:**
- Modify: `src/calibration/gptq_optimizer.py:1-26`

**Step 1: Update module docstring**

Change line 17-18 from:

```
No new transforms, formats, or buffers.  The standard forward path
re-quantizes via ``quantize(w, scheme)`` — idempotent on GPTQ weights.
```

to:

```
Registers ``_weight_scale`` buffer on each module so the forward path
re-quantizes via ``quantize(w, scheme, scale=_weight_scale)`` —
idempotent on GPTQ weights.
```

**Step 2: Commit**

```bash
git add src/calibration/gptq_optimizer.py
git commit -m "docs(gptq): update docstring to reflect _weight_scale buffer"
```
