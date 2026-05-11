# True Error Accumulation — per-layer QSNR vs FP32 reference

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `Session.analyze()` and `result.qsnr_per_layer` measure true accumulated error (fp32 reference vs quantized output per layer), not local quantization error.

**Architecture:** Add `true_error` flag to `Session.analyze()` / `Session.run()`. When enabled, before the observer pass, run fp32_model to capture per-module reference outputs. After observer pass, compare against quantized per-module outputs and override `_qsnr_per_layer` / `_mse_per_layer`.

**Tech Stack:** PyTorch forward hooks, existing Session/observer infra

---

### Task 1: Add `_compute_true_error` to Session

**Files:**
- Modify: `src/session/_session.py`

**Changes:**

Add `true_error: bool = False` parameter to `Session.analyze()` and `Session.run()`, and a new method `_compute_true_error()`.

```python
# In Session.analyze():
def analyze(self, calib_data, *, outputs="default", eval_fn=None,
            true_error: bool = False) -> "Session":
    # ... existing observer analysis ...
    
    if true_error and self._keep_fp32 and self._quant_session.fp32_model is not None:
        self._compute_true_error(calib_data, eval_fn)
    
    return self
```

```python
# New method:
def _compute_true_error(self, calib_data, eval_fn=None) -> None:
    """Run fp32 model to capture reference outputs, then compare
    quantized outputs against them.  Overrides _qsnr_per_layer and
    _mse_per_layer with true accumulated error.
    """
    import math
    
    fp32_model = self._quant_session.fp32_model
    qmodel = self._quant_session.qmodel
    
    # --- 1. Capture fp32 reference outputs ---
    fp32_outs: Dict[str, torch.Tensor] = {}
    
    # Find quantized module names
    quant_names = set()
    for name, mod in qmodel.named_modules():
        if hasattr(mod, "cfg") and not getattr(mod, "_is_passthrough", False):
            quant_names.add(name)
    
    handles_fp32 = []
    for name, mod in fp32_model.named_modules():
        if name in quant_names:
            def _hook(m, inp, out, n=name):
                fp32_outs[n] = out.detach().clone()
            handles_fp32.append(mod.register_forward_hook(_hook))
    
    try:
        with torch.no_grad():
            _run_model(fp32_model, calib_data, eval_fn)
    finally:
        for h in handles_fp32:
            h.remove()
    
    # --- 2. Capture quantized outputs ---
    quant_outs: Dict[str, torch.Tensor] = {}
    
    handles_quant = []
    for name, mod in qmodel.named_modules():
        if name in quant_names:
            def _hook(m, inp, out, n=name):
                quant_outs[n] = out.detach().clone()
            handles_quant.append(mod.register_forward_hook(_hook))
    
    try:
        with torch.no_grad():
            _run_model(self._quant_session, calib_data, eval_fn)
    finally:
        for h in handles_quant:
            h.remove()
    
    # --- 3. Compute true accumulated QSNR / MSE ---
    self._qsnr_per_layer = {}
    self._mse_per_layer = {}
    
    for name in quant_names:
        fp = fp32_outs.get(name)
        q = quant_outs.get(name)
        if fp is None or q is None:
            continue
        num = fp.pow(2).mean()
        den = (fp - q).pow(2).mean()
        if den.item() > 1e-30:
            self._qsnr_per_layer[name] = 10.0 * math.log10(
                max(num.item(), 1e-12) / den.item()
            )
        self._mse_per_layer[name] = den.item()
```

### Task 2: Update `Session.run()` to pass through `true_error`

**Files:**
- Modify: `src/session/_session.py` (`Session.run()`)

Add `true_error` parameter to `run()` and pass it to `analyze()`.

### Task 3: Update test script to use `true_error=True`

**Files:**
- Modify: `scripts/test_true_error_accumulation.py`

Remove manual `capture_true_accumulated_error()` computation. Instead, pass `true_error=True` to `session.run()` and use `result.qsnr_per_layer` directly.

### Task 4: Run test and verify

Compare Session's `qsnr_per_layer` (with `true_error=True`) against the manual hook-based computation. Both should show degrading QSNR with depth.
